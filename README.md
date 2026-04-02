# SPOT-RNA

SPOT-RNA predicts RNA secondary structure from sequence. This repository is being prepared for public release with two clear entry points:

| Path | Purpose |
| --- | --- |
| `official/` | Historical SPOT-RNA inference path, packaged for Docker-first use so TensorFlow 1.x stays isolated from the host environment. |
| `repro/` | Modern PyTorch reproduction surface for training, evaluation, and checkpoint inference work. |

## Choose A Path

### Official Inference

Use `official/` if you want the original SPOT-RNA inference workflow from the published project.

- Docker-first usage
- FASTA input to `.ct`, `.bpseq`, and `.prob` outputs
- historical TensorFlow 1.x model weights downloaded separately

Start here: [`official/README.md`](official/README.md)

### PyTorch Reproduction

Use `repro/` if you want the modern reproduction-oriented training and evaluation path.

- intended home for PyTorch training, evaluation, and checkpoint inference
- separate from the original TensorFlow inference release
- structured as the public-facing reproduction surface during this publish-prep pass

Start here: [`repro/README.md`](repro/README.md)

## Important Distinction

- The original public SPOT-RNA release exposed inference assets, not the original training graph.
- This repository keeps that historical inference path intact while preparing a separate reproduction surface.
- Official TensorFlow weights and future PyTorch checkpoints are different artifacts and should not be treated as interchangeable.

## Repository Layout

```text
official/
  README.md
  docker/
repro/
  README.md
  train.py
  eval.py
  infer.py
docs/
SPOT-RNA.py
utils/
sample_inputs/
```

The legacy `SPOT-RNA.py` and `utils/` code remain in place in this first pass to minimize churn while the public layout is established.

## Data And Weights

- Do not commit downloaded model weights, cached datasets, or experiment outputs.
- Official model files are expected under `SPOT-RNA-models/` at runtime, but should be fetched separately.
- Training outputs such as `training_runs*/`, `logs/`, and `outputs*/` are intentionally ignored.

## Quick References

- Official Docker inference: `official/docker/run_inference.sh --help`
- Reproduction CLI surfaces: `python3 repro/train.py --help`, `python3 repro/eval.py --help`, `python3 repro/infer.py --help`

## License

Mozilla Public License 2.0. See [`LICENSE`](LICENSE).
