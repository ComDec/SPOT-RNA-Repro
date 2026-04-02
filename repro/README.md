# PyTorch Reproduction

`repro/` is the public workflow surface for the modern PyTorch SPOT-RNA reproduction path.

## What Lives Here

- `train.py`: public training entrypoint
- `eval.py`: public ensemble evaluation entrypoint
- `infer.py`: public checkpoint inference entrypoint
- `train_spotrna.py`: copied training implementation used by the public wrapper
- `predict_spotrna_torch.py`: copied checkpoint inference implementation
- `evaluate_spotrna_ensemble.py`: copied ensemble evaluation implementation
- `training_utils.py`: shared PyTorch model, dataset, metric, and IO helpers
- `run_repro_ensemble.sh`: single-GPU sequential launcher
- `run_paper_reproduction_2gpu.sh`: two-GPU paper-shaped launcher
- `repair_run_thresholds.py`: metadata repair helper for finished runs
- `profile_training_pipeline.py`: throughput probe script used during optimization work

## Environment Setup

Install PyTorch separately using the official selector for your CPU or CUDA stack:

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install torch
pip install -r repro/requirements.txt
```

Expected external inputs:

- `datasets/bpRNA_dataset.zip`
- `datasets/PDB_dataset.zip`

These archives are not tracked in Git and should be downloaded separately.

## Core Workflows

### Train A Single Model

```sh
python3 repro/train.py --phase pretrain --datasets-dir datasets --preset paper-small --epochs 3 --output-dir training_runs
python3 repro/train.py --phase finetune --datasets-dir datasets --preset paper-small --init-checkpoint training_runs/<pretrain-run>/best.pt --epochs 3 --output-dir training_runs
```

### Run Checkpoint Inference

```sh
python3 repro/infer.py --checkpoint training_runs/<run>/best.pt --input sample_inputs/single_seq.fasta --output outputs_torch
```

### Evaluate An Ensemble

```sh
python3 repro/eval.py --phase finetune --datasets-dir datasets --checkpoints ckpt0.pt ckpt1.pt ckpt2.pt ckpt3.pt ckpt4.pt --output-json training_runs/ensemble_summary.json
```

### Launch End-To-End Reproduction Runs

```sh
./repro/run_repro_ensemble.sh 1
./repro/run_paper_reproduction_2gpu.sh 0 1
```

## Notes

- The PyTorch path is separate from the historical TensorFlow inference release under `official/`.
- Validation threshold search is part of the saved checkpoint/evaluation workflow.
- Current reproduction status and caveats are summarized in [`REPRODUCIBILITY.md`](../REPRODUCIBILITY.md).
