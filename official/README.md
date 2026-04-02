# Official SPOT-RNA Inference

`official/` is the public entry point for the historical SPOT-RNA inference workflow.

## Why Docker-First

The original code depends on TensorFlow 1.x. To avoid polluting a host Python environment with legacy dependencies, this repository documents and wraps the official path as a containerized workflow.

## What Stays Historical

- inference entrypoint: `SPOT-RNA.py`
- preprocessing and postprocessing helpers: `utils/`
- expected model directory: `SPOT-RNA-models/`

This first publish-prep pass keeps those files at the repository root to minimize risky path rewrites while giving users a clear `official/` entry point.

## Quick Start

1. Build the image:

```sh
docker build -f official/docker/Dockerfile -t spot-rna-official .
```

2. Download the official model bundle into `SPOT-RNA-models/` at the repository root.

Legacy download locations from the original project:

- `https://www.dropbox.com/s/dsrcf460nbjqpxa/SPOT-RNA-models.tar.gz`
- `https://app.nihaocloud.com/f/fbf3315a91d542c0bdc2/?dl=1`

3. Run inference:

```sh
official/docker/run_inference.sh sample_inputs/single_seq.fasta outputs/ --cpu 32
```

The workflow writes `.ct`, `.bpseq`, and `.prob` files to the chosen output directory.

## Notes

- `sample_inputs/` contains example FASTA files.
- The Docker wrapper only mounts this repository, so input and output paths must live under the repo root.
- The published Docker image supports the core `.ct/.bpseq/.prob` inference path only.
- Optional `--plots` and `--motifs` features still depend on extra Java and Perl Graph tooling that is not installed in `official/docker/Dockerfile`.
- Official inference weights are separate from any future PyTorch reproduction checkpoints.
