import argparse
import json
import os
import time
from contextlib import nullcontext
from datetime import datetime

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from utils.training_utils import (
    MODEL_PRESETS,
    PaperInspiredSPOTRNA,
    RNAPairDataset,
    collate_rna_batch,
    compute_feature_stats,
    evaluate_model,
    masked_bce_loss,
    search_best_threshold,
    set_seed,
)


PHASE_DEFAULTS = {
    "pretrain": {
        "dataset_type": "bpRNA",
        "archive_name": "bpRNA_dataset.zip",
        "train_split": "TR0",
        "val_split": "VL0",
        "test_split": "TS0",
    },
    "finetune": {
        "dataset_type": "pdb",
        "archive_name": "PDB_dataset.zip",
        "train_split": "TR1",
        "val_split": "VL1",
        "test_split": "TS1",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a paper-inspired SPOT-RNA reimplementation."
    )
    parser.add_argument("--phase", choices=sorted(PHASE_DEFAULTS), default="pretrain")
    parser.add_argument(
        "--datasets-dir",
        default="datasets",
        help="Directory containing bpRNA_dataset.zip and PDB_dataset.zip",
    )
    parser.add_argument(
        "--preset", choices=sorted(MODEL_PRESETS), default="paper-small"
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--positive-weight",
        default="auto",
        help='Positive class weight for BCE. Use "auto" or a numeric value.',
    )
    parser.add_argument("--threshold", type=float, default=0.335)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--amp", action="store_true", help="Enable autocast mixed precision on CUDA"
    )
    parser.add_argument(
        "--amp-dtype",
        choices=["bf16", "fp16"],
        default="bf16",
        help="Autocast dtype when --amp is enabled",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--init-checkpoint", default="", help="Optional checkpoint to initialize from"
    )
    parser.add_argument("--output-dir", default="training_runs")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument(
        "--resume-checkpoint",
        default="",
        help="Resume training state from a checkpoint",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--drop-multiplets",
        action="store_true",
        help="Drop truth pairs involved in multiplets during training/eval",
    )
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument("--threshold-min", type=float, default=0.1)
    parser.add_argument("--threshold-max", type=float, default=0.9)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument(
        "--standardize-input",
        action="store_true",
        help="Standardize input features using train-split mean/std",
    )
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_dataloader(
    dataset,
    batch_size,
    shuffle,
    num_workers,
    pin_memory,
):
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "collate_fn": collate_rna_batch,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(**loader_kwargs)


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    log_interval,
    positive_weight,
    amp_enabled,
    amp_dtype,
    scaler,
):
    model.train()
    total_loss = 0.0
    total_samples = 0
    running_loss = 0.0
    window_start = time.time()

    for step_idx, batch in enumerate(dataloader, start=1):
        optimizer.zero_grad(set_to_none=True)
        features = batch["features"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        if amp_enabled and device.type == "cuda":
            with autocast(
                device_type="cuda",
                dtype=torch.bfloat16 if amp_dtype == "bf16" else torch.float16,
            ):
                logits = model(features)
                loss = masked_bce_loss(
                    logits, targets, mask, positive_weight=positive_weight
                )
        else:
            logits = model(features)
            loss = masked_bce_loss(
                logits, targets, mask, positive_weight=positive_weight
            )

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * features.size(0)
        total_samples += features.size(0)
        running_loss += loss.item()

        if log_interval and step_idx % log_interval == 0:
            elapsed = time.time() - window_start
            print(
                "step={0} avg_step_loss={1:.4f} elapsed_sec={2:.1f}".format(
                    step_idx, running_loss / log_interval, elapsed
                ),
                flush=True,
            )
            running_loss = 0.0
            window_start = time.time()

    return total_loss / max(total_samples, 1)


def update_best_checkpoint_state(
    best_val_f1,
    best_threshold,
    best_epoch,
    val_metrics,
    epoch_idx,
):
    if val_metrics["f1"] > best_val_f1:
        return val_metrics["f1"], val_metrics["threshold"], epoch_idx, True
    return best_val_f1, best_threshold, best_epoch, False


def save_checkpoint(
    path,
    model,
    optimizer,
    args,
    model_config,
    history,
    epoch,
    best_metric,
    best_threshold,
    best_epoch,
    feature_mean,
    feature_std,
):
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "model_config": model_config,
            "history": history,
            "epoch": epoch,
            "best_val_f1": best_metric,
            "best_threshold": best_threshold,
            "best_epoch": best_epoch,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
        },
        path,
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True

    phase_config = PHASE_DEFAULTS[args.phase]
    archive_path = os.path.join(args.datasets_dir, phase_config["archive_name"])
    if not os.path.exists(archive_path):
        raise FileNotFoundError("Dataset archive not found: {0}".format(archive_path))

    device = resolve_device(args.device)
    pin_memory = device.type == "cuda"
    effective_num_workers = args.num_workers
    if device.type == "cuda":
        cpu_count = os.cpu_count() or args.num_workers or 1
        suggested_workers = 8 if phase_config["dataset_type"] == "bpRNA" else 4
        effective_num_workers = max(args.num_workers, min(suggested_workers, cpu_count))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    model_config = dict(MODEL_PRESETS[args.preset])
    model = PaperInspiredSPOTRNA(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler = None
    if args.amp and device.type == "cuda" and args.amp_dtype == "fp16":
        scaler = GradScaler("cuda")
    start_epoch = 1
    best_val_f1 = -1.0
    best_threshold = args.threshold
    best_epoch = 0
    history = []

    if args.resume_checkpoint:
        checkpoint = torch.load(
            args.resume_checkpoint, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        history = checkpoint.get("history", [])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_f1 = checkpoint.get("best_val_f1", -1.0)
        best_threshold = checkpoint.get("best_threshold", args.threshold)
        best_epoch = checkpoint.get("best_epoch", 0)
    elif args.init_checkpoint:
        checkpoint = torch.load(
            args.init_checkpoint, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["state_dict"], strict=False)

    run_name = args.run_name
    if not run_name:
        run_name = "{0}_{1}_{2}".format(
            args.phase, args.preset, datetime.now().strftime("%Y%m%d_%H%M%S")
        )
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    train_dataset = RNAPairDataset(
        archive_path=archive_path,
        dataset_type=phase_config["dataset_type"],
        split=phase_config["train_split"],
        limit=args.max_train_samples,
        drop_multiplets=args.drop_multiplets,
    )
    feature_mean = None
    feature_std = None
    if args.standardize_input:
        feature_mean, feature_std = compute_feature_stats(train_dataset)
        print(
            "num_workers={0} feature_mean={1} feature_std={2}".format(
                effective_num_workers,
                feature_mean.tolist(),
                feature_std.tolist(),
            ),
            flush=True,
        )
        train_dataset = RNAPairDataset(
            archive_path=archive_path,
            dataset_type=phase_config["dataset_type"],
            split=phase_config["train_split"],
            limit=args.max_train_samples,
            drop_multiplets=args.drop_multiplets,
            feature_mean=feature_mean,
            feature_std=feature_std,
        )

    val_dataset = RNAPairDataset(
        archive_path=archive_path,
        dataset_type=phase_config["dataset_type"],
        split=phase_config["val_split"],
        limit=args.max_val_samples,
        drop_multiplets=args.drop_multiplets,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    test_dataset = RNAPairDataset(
        archive_path=archive_path,
        dataset_type=phase_config["dataset_type"],
        split=phase_config["test_split"],
        limit=args.max_test_samples,
        drop_multiplets=args.drop_multiplets,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )

    train_loader = build_dataloader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=effective_num_workers,
        pin_memory=pin_memory,
    )
    val_loader = build_dataloader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=effective_num_workers,
        pin_memory=pin_memory,
    )
    test_loader = build_dataloader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=effective_num_workers,
        pin_memory=pin_memory,
    )

    best_checkpoint_path = os.path.join(run_dir, "best.pt")
    last_checkpoint_path = os.path.join(run_dir, "last.pt")

    for epoch_idx in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.log_interval,
            args.positive_weight,
            args.amp,
            args.amp_dtype,
            scaler,
        )
        threshold_grid = (
            torch.arange(
                args.threshold_min,
                args.threshold_max + 1e-8,
                args.threshold_step,
            )
            .cpu()
            .numpy()
        )
        val_metrics = search_best_threshold(
            model, val_loader, device, thresholds=threshold_grid
        )
        epoch_record = {
            "epoch": epoch_idx,
            "train_loss": train_loss,
            "val": val_metrics,
        }
        history.append(epoch_record)

        print(
            "epoch={0} train_loss={1:.4f} val_loss={2:.4f} val_f1={3:.4f} val_mcc={4:.4f} threshold={5:.2f}".format(
                epoch_idx,
                train_loss,
                val_metrics["loss"],
                val_metrics["f1"],
                val_metrics["mcc"],
                val_metrics["threshold"],
            )
        )

        best_val_f1, best_threshold, best_epoch, is_new_best = (
            update_best_checkpoint_state(
                best_val_f1,
                best_threshold,
                best_epoch,
                val_metrics,
                epoch_idx,
            )
        )

        save_checkpoint(
            last_checkpoint_path,
            model,
            optimizer,
            args,
            model_config,
            history,
            epoch_idx,
            best_val_f1,
            best_threshold,
            best_epoch,
            feature_mean,
            feature_std,
        )
        if args.save_every_epoch:
            epoch_path = os.path.join(run_dir, "epoch_{0}.pt".format(epoch_idx))
            save_checkpoint(
                epoch_path,
                model,
                optimizer,
                args,
                model_config,
                history,
                epoch_idx,
                best_val_f1,
                best_threshold,
                best_epoch,
                feature_mean,
                feature_std,
            )

        if is_new_best:
            save_checkpoint(
                best_checkpoint_path,
                model,
                optimizer,
                args,
                model_config,
                history,
                epoch_idx,
                best_val_f1,
                best_threshold,
                best_epoch,
                feature_mean,
                feature_std,
            )

    best_checkpoint = torch.load(
        best_checkpoint_path, map_location=device, weights_only=False
    )
    model.load_state_dict(best_checkpoint["state_dict"])
    best_threshold = best_checkpoint.get("best_threshold", best_threshold)
    test_metrics = evaluate_model(model, test_loader, device, best_threshold)

    summary = {
        "phase": args.phase,
        "preset": args.preset,
        "device": str(device),
        "dataset_archive": archive_path,
        "train_split": phase_config["train_split"],
        "val_split": phase_config["val_split"],
        "test_split": phase_config["test_split"],
        "best_val_f1": best_val_f1,
        "best_threshold": best_threshold,
        "best_epoch": best_epoch,
        "feature_mean": feature_mean.tolist() if feature_mean is not None else None,
        "feature_std": feature_std.tolist() if feature_std is not None else None,
        "test": test_metrics,
        "history": history,
    }

    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w") as summary_file:
        json.dump(summary, summary_file, indent=2)

    print(
        "test_loss={0:.4f} test_f1={1:.4f} test_precision={2:.4f} test_sensitivity={3:.4f} test_mcc={4:.4f}".format(
            test_metrics["loss"],
            test_metrics["f1"],
            test_metrics["precision"],
            test_metrics["sensitivity"],
            test_metrics["mcc"],
        )
    )
    print("saved_run={0}".format(run_dir))


if __name__ == "__main__":
    main()
