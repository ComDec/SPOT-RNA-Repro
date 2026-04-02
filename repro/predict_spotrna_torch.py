import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from repro.training_utils import (
    allowed_pair_mask,
    PaperInspiredSPOTRNA,
    build_pair_feature,
    logits_to_pairs,
    read_fasta_records,
    standardize_feature_tensor,
    valid_pair_mask,
    write_bpseq_file,
    write_ct_file,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference with a PyTorch-trained SPOT-RNA checkpoint."
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to a train_spotrna.py checkpoint"
    )
    parser.add_argument("--input", required=True, help="Input FASTA file")
    parser.add_argument(
        "--output",
        default="outputs_torch",
        help="Directory for .ct/.bpseq/.prob outputs",
    )
    parser.add_argument("--threshold", type=float, default=-1.0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def main():
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = PaperInspiredSPOTRNA(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    feature_mean = checkpoint.get("feature_mean")
    feature_std = checkpoint.get("feature_std")
    threshold = (
        args.threshold
        if args.threshold >= 0
        else checkpoint.get("best_threshold", 0.335)
    )

    os.makedirs(args.output, exist_ok=True)
    for sample_id, sequence in read_fasta_records(args.input):
        feature_array = standardize_feature_tensor(
            build_pair_feature(sequence), feature_mean, feature_std
        )
        features = torch.from_numpy(feature_array).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(features)[0, 0, : len(sequence), : len(sequence)]
            probabilities = torch.sigmoid(logits).cpu().numpy()
            probabilities = (
                probabilities * allowed_pair_mask(sequence) * valid_pair_mask(sequence)
            )
        pred_pairs = sorted(logits_to_pairs(logits, sequence, threshold))
        write_ct_file(pred_pairs, sequence, sample_id, args.output)
        write_bpseq_file(pred_pairs, sequence, sample_id, args.output)
        np.savetxt(
            os.path.join(args.output, sample_id + ".prob"),
            probabilities,
            delimiter="\t",
        )
        print("saved {0}".format(sample_id))


if __name__ == "__main__":
    main()
