# SPOT-RNA

SPOT-RNA currently supports two distinct workflows:

| If you want to... | Use this path | Entry point | Environment |
| --- | --- | --- | --- |
| Run the original published predictor on FASTA inputs | Official inference path | `SPOT-RNA.py` through Docker | Docker image built from `Dockerfile` and `environment.cpu.yml` |
| Reproduce training, evaluation, or checkpoint-based inference in this repo | PyTorch reproduction path | `train_spotrna.py`, `predict_spotrna_torch.py`, `evaluate_spotrna_ensemble.py` | Local Conda env or `venv` with modern PyTorch |

Choose the path first:

- Original predictor or bundled official checkpoints: use Docker.
- Training, evaluation, or your own reproduced checkpoints: use the PyTorch path.

The Docker path is inference-only. It packages the original TensorFlow 1.14 predictor for running published checkpoints, but it is not the original training graph. The PyTorch path is a separate reproduction of the paper pipeline for modern experimentation.

|![](./docs/SPOT-RNA-architecture.png)
|----|
| <p align="center"><b>Figure 1:</b> The published SPOT-RNA architecture. This repository keeps the original inference stack for official checkpoint prediction and a separate PyTorch reproduction for training and evaluation work.</p> |

## Official inference with Docker

Use this path when you want the original SPOT-RNA predictor behavior with the bundled published checkpoints.

Build the CPU image:

```bash
docker build -t spotrna-cpu .
```

Quick smoke test with the image defaults:

```bash
docker run --rm spotrna-cpu
```

That command runs the bundled sample FASTA inside the container, but because the container is removed afterwards, its output files are not kept on the host.

To run the bundled sample and keep the generated files on the host:

```bash
mkdir -p outputs
docker run --rm -v "$PWD/outputs:/workspace/SPOT-RNA/outputs" spotrna-cpu
```

Run your own FASTA by mounting a host directory that contains the input file and a place to write outputs:

```bash
docker run --rm \
  -v "$PWD:/data" \
  spotrna-cpu \
  --input /data/your_sequences.fasta \
  --output /data/outputs
```

Notes:

- The image is built from `environment.cpu.yml`.
- The Docker build downloads the official `SPOT-RNA-models/` checkpoints into the image.
- `SPOT-RNA.py` also performs the same checkpoint download at runtime if the model directory is absent.
- The official path is Docker-only in this repository. It does not require or document a host TensorFlow 1.14 install.
- This path is for published-model inference only, not for reproducing the original TensorFlow training graph.

Useful optional flags for the official path:

```bash
docker run --rm -v "$PWD:/data" spotrna-cpu --input /data/sample_inputs/batch_seq.fasta --output /data/outputs
docker run --rm -v "$PWD:/data" spotrna-cpu --input /data/sample_inputs/single_seq.fasta --output /data/outputs --plots
docker run --rm -v "$PWD:/data" spotrna-cpu --input /data/sample_inputs/single_seq.fasta --output /data/outputs --plots --motifs
```

`--plots` uses VARNA through Java. `--motifs` calls the bundled bpRNA script, but motif generation still depends on a working Perl setup with `Graph.pm`; if that dependency is missing, the script prints a bpRNA requirements message instead of producing motif files.

## PyTorch reproduction setup

Use this path for training, evaluation, and inference with checkpoints produced by `train_spotrna.py`.

Conda option:

```bash
conda create -n spotrna-repro python=3.11
conda activate spotrna-repro
pip install torch pandas tqdm numpy
```

`venv` option:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch pandas tqdm numpy
```

Notes:

- Install a CUDA-enabled PyTorch build if you plan to use `--device cuda`.
- The reproduction scripts are separate from the Docker image and do not use the official TensorFlow container.
- Main entry points for this path are `train_spotrna.py`, `predict_spotrna_torch.py`, and `evaluate_spotrna_ensemble.py`.

## Dataset preparation

The training and evaluation scripts expect external dataset archives under `datasets/`:

- `datasets/bpRNA_dataset.zip`
  Download: [Dropbox](https://www.dropbox.com/s/w3kc4iro8ztbf3m/bpRNA_dataset.zip) or [Nihao Cloud](https://app.nihaocloud.com/d/26df85ca06bc4eae8f6f/)
- `datasets/PDB_dataset.zip`
  Download: [Dropbox](https://www.dropbox.com/s/vnq0k9dg7vynu3q/PDB_dataset.zip?dl=0) or [Nihao Cloud](https://app.nihaocloud.com/f/d61638e03bc5484f8f83/)

These archives are local downloads and are ignored by git, along with the parsed cache directory:

- `datasets/.spotrna_cache/`

`RNAPairDataset` extracts and caches parsed split data under `.spotrna_cache` on first use. Keep the zip files in place after download so later runs can reuse the cache.

## Training

Minimal pretraining example:

```bash
python3 train_spotrna.py \
  --phase pretrain \
  --datasets-dir datasets \
  --preset paper-small \
  --epochs 3 \
  --output-dir training_runs/pretrain_demo \
  --run-name main
```

That writes checkpoints and metrics under `training_runs/pretrain_demo/main/`.

Minimal fine-tuning example from a pretrained checkpoint:

```bash
python3 train_spotrna.py \
  --phase finetune \
  --datasets-dir datasets \
  --preset paper-small \
  --epochs 3 \
  --init-checkpoint training_runs/pretrain_demo/main/best.pt \
  --output-dir training_runs/finetune_demo \
  --run-name main
```

Each run directory contains `best.pt`, `last.pt`, and `summary.json`.

## Inference with reproduced checkpoints

Use a checkpoint produced by `train_spotrna.py`:

```bash
python3 predict_spotrna_torch.py \
  --checkpoint training_runs/finetune_demo/main/best.pt \
  --input sample_inputs/single_seq.fasta \
  --output outputs_torch
```

This writes `.ct`, `.bpseq`, and `.prob` files for each FASTA record.

## Experimental and auxiliary scripts

These scripts are kept in the repository for extended experiments and maintenance, but they are not the main onboarding path:

- `evaluate_spotrna_ensemble.py`: evaluate an ensemble of reproduced checkpoints against the cached datasets.
- `run_repro_ensemble.sh`: single-GPU helper that pretrains, fine-tunes, and evaluates a five-model reproduced ensemble.
- `run_paper_reproduction_2gpu.sh`: two-GPU helper for a longer paper-style reproduction run across the five presets.
- `repair_run_thresholds.py`: repair missing or stale `best_threshold` metadata in an existing training run.
- `profile_training_pipeline.py`: lightweight loader and step-time profiling for the PyTorch training stack.

## Repository layout

- `SPOT-RNA.py`: official TensorFlow inference entry point used by the Docker image.
- `Dockerfile`: builds the official inference container.
- `environment.cpu.yml`: Conda environment used by the official Docker path.
- `train_spotrna.py`: PyTorch reproduction training entry point.
- `predict_spotrna_torch.py`: inference with reproduced PyTorch checkpoints.
- `evaluate_spotrna_ensemble.py`: reproduced-ensemble evaluation entry point.
- `utils/training_utils.py`: PyTorch reproduction datasets, feature building, model definitions, metrics, and output writers.
- `sample_inputs/`: example FASTA files for quick runs.
- `datasets/`: local dataset archives and generated cache.
- `docs/`: paper figures and PDFs.

## Results and paper context

The published SPOT-RNA method reports 94% precision at 50% sensitivity and improves performance on noncanonical and non-nested base pairs relative to prior predictors. The original publication is available through Nature Communications: https://doi.org/10.1038/s41467-019-13395-9

|![](./docs/benchmark_results.png)
|----|
| <p align="center"><b>Figure 2:</b> Benchmark results from the original SPOT-RNA paper comparing the published model family against existing RNA secondary-structure predictors.</p> |

SPOT-RNA is also available as a web server at https://sparks-lab.org/server/spot-rna/.

## Citation guide

If you use SPOT-RNA for your research, please cite:

Singh, J., Hanson, J., Paliwal, K., Zhou, Y. RNA secondary structure prediction using an ensemble of two-dimensional deep neural networks and transfer learning. Nat Commun 10, 5407 (2019). https://doi.org/10.1038/s41467-019-13395-9

If you use the SPOT-RNA datasets or post-processing pipeline, please also consider citing:

[1] Padideh Danaee, Mason Rouches, Michelle Wiley, Dezhong Deng, Liang Huang, David Hendrix, bpRNA: large-scale automated annotation and analysis of RNA secondary structure, Nucleic Acids Research, Volume 46, Issue 11, 20 June 2018, Pages 5381-5394, https://doi.org/10.1093/nar/gky285

[2] H.M. Berman, J. Westbrook, Z. Feng, G. Gilliland, T.N. Bhat, H. Weissig, I.N. Shindyalov, P.E. Bourne. (2000) The Protein Data Bank. Nucleic Acids Research, 28, 235-242.

[3] K. Darty, A. Denise, Y. Ponty. VARNA: Interactive drawing and editing of the RNA secondary structure. Bioinformatics, 25(15), 1974-1975 (2009).

## License

Mozilla Public License 2.0

## Contact

jaswinder.singh3@griffithuni.edu.au, yaoqi.zhou@griffith.edu.au
