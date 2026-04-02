# Reproducibility

## Scope

This repository now exposes two distinct paths:

- `official/`: historical TensorFlow 1.x inference workflow for the original released predictor
- `repro/`: PyTorch reimplementation for training, evaluation, and checkpoint inference

The original public SPOT-RNA repository did not ship the original training graph. The `repro/` path is therefore a best-effort reproduction based on the paper, supplementary material, and observed public inference behavior.

## Paper Targets

Target metrics from the published transfer-learning result on `TS1`:

| Metric | Paper target |
| --- | --- |
| MCC | about `0.690` |
| F1 | about `0.687` |
| Precision | about `0.888` |
| Sensitivity | about `0.562` |

Protocol described in the paper and mirrored by the reproduction workflow:

- pretrain on bpRNA `TR0`
- validate on bpRNA `VL0`
- test on bpRNA `TS0`
- fine-tune on PDB `TR1`
- validate on PDB `VL1`
- test on PDB `TS1`
- evaluate a 5-model ensemble

## What Is Implemented

The `repro/` path includes real public-facing entrypoints for:

- model training: `python3 repro/train.py`
- checkpoint inference: `python3 repro/infer.py`
- ensemble evaluation: `python3 repro/eval.py`

The implementation also includes:

- direct parsing of the public `bpRNA` and `PDB` zip archives
- validation-threshold search
- saved checkpoint metadata including threshold and feature normalization stats
- ensemble evaluation over multiple fine-tuned checkpoints
- helper scripts for threshold repair and pipeline profiling

## Current Results Snapshot

Results documented so far come from the experiment log maintained during reproduction development.

### Verified Smoke Runs

| Experiment | Purpose | Result |
| --- | --- | --- |
| `EXP-000` | pretrain smoke test | end-to-end path worked, metrics not meaningful |
| `EXP-001` | finetune smoke test | end-to-end path worked, metrics not meaningful |
| `EXP-003` | medium probe with automatic positive weighting | non-zero validation learning signal, `val_f1=0.0066`, `val_mcc=0.0096`, `best_threshold=0.74` |

### Important Interpretation

- Smoke-test metrics should not be compared to the paper.
- The medium probe established that the pipeline learns something and that automatic positive weighting prevents all-negative collapse.
- Full paper-shaped multi-model runs were launched and documented, but the publication docs in this pass do not claim a final reproduced headline metric yet.

## Caveats

- The exact original training graph and some training details were not published.
- Loss implementation, epoch budgets, and some operational choices remain approximations guided by the paper and supplementary material.
- Official TensorFlow weights and PyTorch reproduction checkpoints are different artifacts.
- Reproduced results depend on external dataset archives, hardware availability, and runtime settings such as worker count and GPU contention.

## Source Of Truth

The experiment summary above is derived from the maintained reproduction experiment log used during development. If that log is later published into this worktree, it should be treated as the detailed run history.
