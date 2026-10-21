# COPRA Flow: Trace-Conditioned Flow Refinement for Robust Ultrafast Pulse Retrieval

Trace-conditioned flow refinement for robust ultrafast pulse retrieval from
transient-grating frequency-resolved optical gating (TG-FROG) measurements.
COPRA retrieves a complex electric field; a conditional flow model then
refines that field using the measured trace as context.

## Install

Python 3.10–3.12 is supported; Python 3.12 is the tested reference interpreter.
One package includes generation, training, inference, and evaluation. The exact
report implementation of `pypret` is bundled; no separate checkout is needed.

```bash
uv sync --locked
uv run copra-flow --help
```

For a pip installation in a virtual environment:

```bash
python -m pip install .
```

PyTorch 2.5.1 is pinned to the report version. CPU and Apple MPS use the normal
installation. For CUDA training, see [training instructions](docs/training.md).

## Run the pretrained model

Model and dataset hosting is being prepared separately from this code release.
Until artifact links are published, place the report checkpoint at
`checkpoints/report.pt`. Inference does **not** require the training dataset.

First create a synthetic input measurement:

```bash
uv run copra-flow example --out data/example.npz
uv run copra-flow infer --config configs/inference_report.json \
  --checkpoint checkpoints/report.pt --input data/example.npz \
  --out-dir runs/example
```

This runs COPRA and 64 Euler refinement steps, then writes `prediction.npz` and
`metrics.json`. The NPZ contains the measured and reconstructed traces, COPRA
and refined complex fields, physical axes, carrier wavelength, and metadata.
There is no ground-truth field in the input and none is required by inference.

For a native experimental trace in the documented wavelength-delay text format:

```bash
uv run copra-flow prepare --input measurement.txt --out data/prepared.npz
uv run copra-flow infer --checkpoint checkpoints/report.pt \
  --input data/prepared.npz --out-dir runs/measurement
```

To refine a COPRA field you already computed, add `--source-field source.npy`
(or a `.npz` containing `field`). The field must share the input's temporal
grid. See [input formats](docs/data_format.md) and [inference](docs/inference.md).

## Verify the code without downloading model weights or a dataset

```bash
uv run pytest -q
uv run python scripts/verify_release.py
```

The verification script generates four synthetic pulse shards, checks resumable
generation, trains a small model for two epochs with real optimizer updates,
reloads its checkpoint, evaluates and saves predictions, plots comparisons,
and runs both Euler and RK4 inference plus the supplied-COPRA-field route.
Logs and a machine-readable summary are written to `verification/smoke/`.
The small model verifies execution; it is not the trained report model.

## Generate the training data

```bash
uv run copra-flow generate --config configs/dataset_report.json
uv run copra-flow splits --data-dir data/flow_matching_10k \
  --out data/flow_matching_10k/splits.json
uv run copra-flow inspect --data-dir data/flow_matching_10k
```

The report dataset contains 10,000 independent pulses from 17 families, with
30 reconstruction candidates per pulse. It uses a 256-point temporal grid at
2.5 fs spacing, TG-FROG, 1% Gaussian trace noise with nonnegative clipping,
and 400 COPRA iterations. Generation is resumable. Changing a generation
setting requires a new output directory; do not mix configurations.

## Train

```bash
uv run copra-flow train --config configs/train_report.json
```

The report configuration trains on 270,000 pairs from 9,000 pulses and
validates on 30,000 pairs from 1,000 disjoint pulses. It uses 50 epochs, batch
size 64, AdamW at learning rate 0.0003, and CUDA mixed precision where available.
The reported run took approximately 12 hours on one A100 40 GB GPU.

Commands accept explicit overrides after JSON defaults, for example
`--epochs 1 --workers 0 --device cpu --no-amp --out-dir runs/debug`.
See [training](docs/training.md) for small runs and saved artifacts.

## Reproduce the main result

```bash
uv run python reproduction/synthetic_benchmark.py \
  --checkpoint checkpoints/report.pt --data-dir data/flow_matching_10k \
  --out-dir runs/report_benchmark
```

This evaluates the fixed **1,000 validation pulses using final COPRA sources**,
saves reusable predictions, calculates the report metrics, and compares them
against the archived numerical reference. It uses 64 Euler steps and the
explicit pulse split supplied with this repository.

| Method | Mean Pulse MAE | Mean Trace SSIM |
| --- | ---: | ---: |
| COPRA | 0.4685 | 0.9709 |
| COPRA + direct Hann | 0.3345 | 0.8207 |
| COPRA Flow | 0.0674 | 0.9103 |

These are validation results: this partition also selected the checkpoint.
Flow improves pulse recovery but does not enforce trace consistency.
Experimental amplitude roughness measures smoothness, not recovery accuracy.
Read [reproducibility and limitations](docs/reproducibility.md) before comparing
results or applying the method to a new measurement domain.

## Repository layout

- `src/pnp_copra/`: pulse generation, retrieval, model, training and inference.
- `src/pypret/`: bundled physics and COPRA implementation.
- `configs/`: executable report settings.
- `reproduction/`: fixed selections, reference metrics and benchmark script.
- `scripts/verify_release.py`: independent end-to-end execution check.
- `tests/`: numerical, input-validation and workflow tests.
- `docs/`: formats, usage and scientific conventions.

The internal `pnp_copra` name preserves compatibility with the research
implementation. Use `copra-flow` as the public command. Generated data,
checkpoints, run outputs, historical experiments, and raw experimental
measurements are excluded from the code repository.

## Attribution and license

Code is provided under the MIT license; see [LICENSE](LICENSE) and
[third-party notices](THIRD_PARTY_NOTICES.md). Dataset and model licenses are
specified with their separate releases. Cite the report using `CITATION.cff`.

The full synthetic benchmark passes against the report reference. The ten-trace
experimental workflow runs, but its historical table does not match exactly
when the original scripts are rerun today. The controlled port comparisons and
new metrics are recorded in [reproducibility notes](docs/reproducibility.md).
