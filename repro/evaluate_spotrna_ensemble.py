import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from repro.training_utils import (
    allowed_pair_mask,
    RNAPairDataset,
    collate_rna_batch,
    compute_metrics_for_threshold,
    PaperInspiredSPOTRNA,
    valid_pair_mask,
)


PHASE_DEFAULTS = {
    "pretrain": {
        "dataset_type": "bpRNA",
        "archive_name": "bpRNA_dataset.zip",
        "val_split": "VL0",
        "test_split": "TS0",
    },
    "finetune": {
        "dataset_type": "pdb",
        "archive_name": "PDB_dataset.zip",
        "val_split": "VL1",
        "test_split": "TS1",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate an ensemble of PyTorch SPOT-RNA checkpoints."
    )
    parser.add_argument("--phase", choices=sorted(PHASE_DEFAULTS), default="finetune")
    parser.add_argument("--datasets-dir", default="datasets")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--drop-multiplets", action="store_true")
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_loader(
    archive_path,
    dataset_type,
    split,
    batch_size,
    num_workers,
    drop_multiplets,
    feature_mean,
    feature_std,
):
    dataset = RNAPairDataset(
        archive_path=archive_path,
        dataset_type=dataset_type,
        split=split,
        drop_multiplets=drop_multiplets,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_rna_batch,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
    )


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = PaperInspiredSPOTRNA(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def collect_ensemble_predictions(models, dataloader, device):
    models = [model.eval() for model in models]
    total_loss = 0.0
    total_count = 0
    samples = []

    for batch in dataloader:
        features = batch["features"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        logits_stack = torch.stack([model(features) for model in models], dim=0)
        logits = logits_stack.mean(dim=0)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        loss = (loss * mask).sum() / mask.sum().clamp_min(1.0)
        total_loss += loss.item() * features.size(0)
        total_count += features.size(0)

        for idx, length in enumerate(batch["lengths"]):
            sequence = batch["sequences"][idx]
            probabilities = (
                torch.sigmoid(logits[idx, 0, :length, :length]).detach().cpu().numpy()
            )
            probabilities = (
                probabilities * allowed_pair_mask(sequence) * valid_pair_mask(sequence)
            )
            samples.append(
                {
                    "sequence": sequence,
                    "true_pairs": batch["pairs"][idx],
                    "valid_positions": int(
                        batch["mask"][idx, 0, :length, :length].sum().item()
                    ),
                    "probabilities": probabilities,
                }
            )
    return samples, total_loss / max(total_count, 1)


def main():
    args = parse_args()
    device = resolve_device(args.device)
    phase = PHASE_DEFAULTS[args.phase]
    archive_path = os.path.join(args.datasets_dir, phase["archive_name"])
    checkpoint0 = torch.load(
        args.checkpoints[0], map_location="cpu", weights_only=False
    )
    feature_mean = checkpoint0.get("feature_mean")
    feature_std = checkpoint0.get("feature_std")
    effective_workers = args.num_workers

    val_loader = build_loader(
        archive_path,
        phase["dataset_type"],
        phase["val_split"],
        args.batch_size,
        effective_workers,
        args.drop_multiplets,
        feature_mean,
        feature_std,
    )
    test_loader = build_loader(
        archive_path,
        phase["dataset_type"],
        phase["test_split"],
        args.batch_size,
        effective_workers,
        args.drop_multiplets,
        feature_mean,
        feature_std,
    )
    models = [load_model(path, device) for path in args.checkpoints]

    val_samples, val_loss = collect_ensemble_predictions(models, val_loader, device)
    best_metrics = None
    for threshold in np.arange(0.1, 0.91, 0.01):
        metrics = compute_metrics_for_threshold(val_samples, float(threshold))
        metrics["loss"] = val_loss
        if (
            best_metrics is None
            or metrics["f1"] > best_metrics["f1"]
            or (
                metrics["f1"] == best_metrics["f1"]
                and metrics["mcc"] > best_metrics["mcc"]
            )
        ):
            best_metrics = metrics

    test_samples, test_loss = collect_ensemble_predictions(models, test_loader, device)
    test_metrics = compute_metrics_for_threshold(
        test_samples, best_metrics["threshold"]
    )
    test_metrics["loss"] = test_loss

    summary = {
        "phase": args.phase,
        "device": str(device),
        "checkpoints": args.checkpoints,
        "validation": best_metrics,
        "test": test_metrics,
    }
    print(json.dumps(summary, indent=2))
    if args.output_json:
        with open(args.output_json, "w") as output_file:
            json.dump(summary, output_file, indent=2)


if __name__ == "__main__":
    main()
