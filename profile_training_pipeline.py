import os
import time

import torch

from train_spotrna import build_dataloader
from utils.training_utils import (
    RNAPairDataset,
    PaperInspiredSPOTRNA,
    collate_rna_batch,
    compute_feature_stats,
    masked_bce_loss,
)


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("profile_training_pipeline.py requires CUDA")
    return torch.device("cuda")


def main():
    os.environ.setdefault("TMPDIR", os.path.abspath(".tmp"))
    archive_path = "datasets/bpRNA_dataset.zip"
    train_dataset = RNAPairDataset(
        archive_path=archive_path,
        dataset_type="bpRNA",
        split="TR0",
        limit=256,
    )
    feature_mean, feature_std = compute_feature_stats(train_dataset)
    train_dataset = RNAPairDataset(
        archive_path=archive_path,
        dataset_type="bpRNA",
        split="TR0",
        limit=256,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )

    for workers in [0, 2, 4, 8]:
        loader = build_dataloader(
            dataset=train_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
        )
        start = time.time()
        for idx, batch in enumerate(loader):
            if idx == 32:
                break
        print(f"loader workers={workers} seconds={time.time() - start:.3f}")

    device = require_cuda()
    model = PaperInspiredSPOTRNA(
        **{
            "channels": 64,
            "num_blocks": 30,
            "fc_layers": 1,
            "fc_channels": 512,
            "bilstm_hidden": 0,
            "dilation_cycle": 5,
        }
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = build_dataloader(
        dataset=train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )
    batch = next(iter(loader))
    features = batch["features"].to(device, non_blocking=True)
    targets = batch["targets"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.cuda.synchronize()

    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(features)
            loss = masked_bce_loss(logits, targets, mask, positive_weight=1.0)
        loss.backward()
        optimizer.step()

    step_times = []
    for _ in range(10):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(features)
            loss = masked_bce_loss(logits, targets, mask, positive_weight=1.0)
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_times.append(time.time() - start)
    print(f"train_step_mean={sum(step_times) / len(step_times):.3f}")

    longest_idx = max(
        range(len(train_dataset)),
        key=lambda idx: len(train_dataset.entries[idx]["sequence"]),
    )
    longest_sample = train_dataset[longest_idx]
    for batch_size in [1, 2, 4, 8]:
        try:
            batch = collate_rna_batch([longest_sample] * batch_size)
            features = batch["features"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            start = time.time()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(features)
                loss = masked_bce_loss(logits, targets, mask, positive_weight=1.0)
            loss.backward()
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            print(
                f"probe_batch={batch_size} max_len={batch['lengths'][0]} step_sec={time.time() - start:.3f}"
            )
        except RuntimeError as exc:
            print(f"probe_batch={batch_size} failed={exc}")
            if device.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
