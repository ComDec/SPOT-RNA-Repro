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

from repro.train_spotrna import PHASE_DEFAULTS
from repro.training_utils import (
    RNAPairDataset,
    PaperInspiredSPOTRNA,
    collate_rna_batch,
    evaluate_model,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Repair saved best-threshold metadata for a completed run."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path like training_runs/.../finetune_model1/main",
    )
    parser.add_argument("--datasets-dir", default="datasets")
    return parser.parse_args()


def infer_phase_from_summary(summary):
    phase = summary.get("phase")
    if phase not in PHASE_DEFAULTS:
        raise ValueError("Unable to infer training phase from summary.json")
    return phase


def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: to_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value


def main():
    args = parse_args()
    summary_path = os.path.join(args.run_dir, "summary.json")
    best_path = os.path.join(args.run_dir, "best.pt")
    last_path = os.path.join(args.run_dir, "last.pt")

    summary = None
    if os.path.exists(summary_path):
        try:
            with open(summary_path) as summary_file:
                summary = json.load(summary_file)
        except json.JSONDecodeError:
            summary = None

    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    if summary is None:
        checkpoint_args = checkpoint.get("args", {})
        phase = checkpoint_args.get("phase")
        preset = checkpoint_args.get("preset")
        if phase not in PHASE_DEFAULTS:
            raise ValueError("Unable to infer phase from checkpoint args")
        history = checkpoint.get("history", [])
        summary = {
            "phase": phase,
            "preset": preset,
            "device": checkpoint_args.get("device", "cpu"),
            "dataset_archive": os.path.join(
                args.datasets_dir, PHASE_DEFAULTS[phase]["archive_name"]
            ),
            "train_split": PHASE_DEFAULTS[phase]["train_split"],
            "val_split": PHASE_DEFAULTS[phase]["val_split"],
            "test_split": PHASE_DEFAULTS[phase]["test_split"],
            "best_val_f1": checkpoint.get("best_val_f1", -1.0),
            "feature_mean": checkpoint.get("feature_mean"),
            "feature_std": checkpoint.get("feature_std"),
            "history": history,
        }
    else:
        phase = infer_phase_from_summary(summary)
        history = summary.get("history", [])

    if not history:
        raise ValueError("No history entries available to repair threshold metadata")

    best_record = max(history, key=lambda item: (item["val"]["f1"], item["val"]["mcc"]))
    best_epoch = int(best_record["epoch"])
    best_threshold = float(best_record["val"]["threshold"])
    model = PaperInspiredSPOTRNA(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])

    phase_config = PHASE_DEFAULTS[phase]
    archive_path = os.path.join(args.datasets_dir, phase_config["archive_name"])
    dataset = RNAPairDataset(
        archive_path=archive_path,
        dataset_type=phase_config["dataset_type"],
        split=phase_config["test_split"],
        feature_mean=checkpoint.get("feature_mean"),
        feature_std=checkpoint.get("feature_std"),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_rna_batch,
        num_workers=0,
    )
    test_metrics = evaluate_model(
        model, dataloader, torch.device("cpu"), best_threshold
    )

    checkpoint["best_threshold"] = best_threshold
    checkpoint["best_epoch"] = best_epoch
    torch.save(checkpoint, best_path)

    if os.path.exists(last_path):
        last_checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        last_checkpoint["best_threshold"] = best_threshold
        last_checkpoint["best_epoch"] = best_epoch
        torch.save(last_checkpoint, last_path)

    summary["best_threshold"] = best_threshold
    summary["best_epoch"] = best_epoch
    summary["test"] = test_metrics
    with open(summary_path, "w") as summary_file:
        json.dump(to_jsonable(summary), summary_file, indent=2)

    print("repaired", args.run_dir)
    print(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "best_threshold": best_threshold,
                "test": test_metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
