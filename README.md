# COPRA Flow: Trace-Conditioned Flow Refinement for Robust Ultrafast Pulse Retrieval

COPRA Flow reconstructs ultrafast laser pulses from transient-grating
frequency-resolved optical gating (TG-FROG) measurements. It starts with a
complex electric field retrieved by COPRA, then uses a learned flow model to
refine the field, guided by the measured trace.

This repository includes everything needed to generate synthetic training
data, train the model, and reconstruct pulses from new measurements.

**Downloads:** [trained model](https://huggingface.co/atal-narayan/copra-flow)
and [training dataset](https://huggingface.co/datasets/atal-narayan/copra-flow-tg-frog)
are available on Hugging Face. See the [download guide](docs/downloads.md)
for versioned downloads, checksums, and dataset extraction.

## Getting started

Use Python 3.10–3.12; Python 3.12 is recommended. Install with
[uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
git clone https://github.com/atalnarayan/copra-flow.git
cd copra-flow
uv sync --locked
uv run copra-flow --help
```

The installation includes PyTorch and the `pypret` physics library. CPU and
Apple Silicon are supported out of the box. For an NVIDIA GPU, follow the
[CUDA setup instructions](docs/training.md#reference-training) first.

To check that everything works on your machine:

```bash
uv run python scripts/verify_release.py
```

This creates a tiny dataset, trains a small model, and runs evaluation and
inference. Results and logs are saved in `verification/smoke/`. This is a
workflow check; the small model is not intended for pulse reconstruction.

## Reconstruct a pulse

You need a trained checkpoint and a TG-FROG measurement. The training dataset
is not needed for inference. Install the Hugging Face CLI and download the
checkpoint:

```bash
uv tool install huggingface_hub
hf download atal-narayan/copra-flow copra-flow.pt --revision v0.1.0 --local-dir checkpoints
```

Start with a synthetic measurement:

```bash
uv run copra-flow example --out data/example.npz
uv run copra-flow infer --checkpoint checkpoints/copra-flow.pt \
  --input data/example.npz --out-dir runs/example
```

The command runs COPRA followed by 64 flow refinement steps. It saves:

- `prediction.npz`: the initial and refined complex fields, measured and
  reconstructed traces, and physical axes.
- `metrics.json`: trace agreement metrics and run settings.

For an experimental measurement in wavelength-delay text format, prepare it
first, then run inference:

```bash
uv run copra-flow prepare --input measurement.txt --out data/prepared.npz
uv run copra-flow infer --checkpoint checkpoints/copra-flow.pt \
  --input data/prepared.npz --out-dir runs/measurement
```

No ground-truth pulse is required. Inputs use a 256-point temporal grid with
2.5 fs spacing. See [measurement formats](docs/data_format.md) for the required
text and NPZ layouts, and the [inference guide](docs/inference.md) for options
such as refining an existing COPRA field.

## Train your own model

Download and extract the dataset (approximately 3.8 GB):

```bash
hf download atal-narayan/copra-flow-tg-frog --repo-type dataset \
  --revision v0.1.0 --local-dir data/flow_matching_10k
python data/flow_matching_10k/extract.py
```

Alternatively, generate it yourself:

```bash
uv run copra-flow generate --out-dir data/flow_matching_10k --workers 4
```

The default dataset contains 10,000 pulses from 17 pulse families, with 30
reconstruction candidates per pulse. Generation uses 1% trace noise and 400
COPRA iterations per pulse, so allow time for it to finish. You can rerun the
same command to resume interrupted generation.

Train using the supplied split of 9,000 training pulses and 1,000 validation
pulses:

```bash
uv run copra-flow train --data-dir data/flow_matching_10k \
  --split-file reproduction/selections/synthetic_split.json \
  --out-dir runs/copra-flow --amp
```

Training defaults to 50 epochs, a batch size of 64, and the AdamW optimizer
with a learning rate of 0.0003. Mixed precision is used on CUDA; CPU and Apple
MPS use full precision. Each run saves its settings, training history, and
`checkpoints/best.pt` and `checkpoints/last.pt` inside the output directory.
Use `best.pt` for inference.

The [training guide](docs/training.md) covers smaller runs, custom datasets,
and configuration options. Use a new output directory when changing dataset
generation settings or starting another training run.

## Evaluate the model

To reproduce the paper's synthetic benchmark, use the matching checkpoint
and dataset:

```bash
uv run python reproduction/synthetic_benchmark.py \
  --checkpoint checkpoints/copra-flow.pt --data-dir data/flow_matching_10k \
  --out-dir runs/benchmark
```

The benchmark evaluates the fixed 1,000-pulse validation set, saves predictions,
and compares the metrics with the reference results.

| Method | Mean pulse MAE ↓ | Mean trace SSIM ↑ |
| --- | ---: | ---: |
| COPRA | 0.4685 | 0.9709 |
| COPRA + direct Hann | 0.3345 | 0.8207 |
| COPRA Flow | 0.0674 | 0.9103 |

COPRA Flow improves pulse recovery on this dataset, although the refined
field can have lower agreement with the measured trace. These are validation
results; the same split was used to select the checkpoint. The experimental
workflow also runs, but its historical numerical results do not reproduce
exactly. See [reproducibility and limitations](docs/reproducibility.md) for
details and [verification results](docs/verification.md) for completed checks.

## Where to find things

- [Source code](src/pnp_copra): dataset generation, retrieval, training, and inference.
- [Configurations](configs): settings for generation, training, and inference.
- [Benchmarks](reproduction): evaluation scripts, fixed splits, and reference results.
- [Documentation](docs): input formats, usage guides, and scientific conventions.
- [Tests](tests): run with `uv run pytest -q`.

## License and citation

The code is available under the [MIT license](LICENSE). See
[third-party notices](THIRD_PARTY_NOTICES.md) for bundled dependencies.
The [model weights](https://huggingface.co/atal-narayan/copra-flow) use MIT;
the [synthetic dataset](https://huggingface.co/datasets/atal-narayan/copra-flow-tg-frog)
uses CC BY 4.0. Citation metadata is available in [CITATION.cff](CITATION.cff).
