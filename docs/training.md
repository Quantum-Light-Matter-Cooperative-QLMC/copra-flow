# Training and dataset generation

Run commands from the repository root. JSON paths are relative to the current
working directory; explicit CLI arguments override JSON defaults.

## Dataset

`configs/dataset_report.json` records the final normalized generator settings.
The manifest fixes pulse families, continuous parameters, UIDs and seeds;
per-pulse seeds additionally incorporate a stable hash of the UID.

```bash
uv run copra-flow generate --config configs/dataset_report.json
uv run copra-flow splits --data-dir data/flow_matching_10k --out data/flow_matching_10k/splits.json
```

All sources derived from one pulse stay in one partition. For the published
benchmark, use `reproduction/selections/synthetic_split.json`. A different seed
or a different set of shards produces a different benchmark.

A lower-cost generation run can use `--count 4 --maxiter 4 --workers 2` and a
new output directory. It is suitable for trying out the workflow.
`--manifest FILE` reuses pulse specifications; `--workers` changes parallelism.
`status --out-dir DIR` shows completed, failed, and skipped tasks. Failures
produce per-pulse JSON diagnostics and a nonzero exit status.

## Reference training

`configs/train_report.json` specifies the 1.75M-parameter architecture, all
source candidates including hard negatives, source weights, peak normalization
and square-root trace preprocessing, AdamW, learning rate 0.0003, weight decay
0.0001, and gradient clipping at norm 1.0. The seed is 42. Validation is once
per epoch; best checkpoint selection uses validation flow loss.

```bash
uv run copra-flow train --config configs/train_report.json
```

To use an already downloaded dataset, override `--data-dir` and `--split-file`.
For a brief run with the full architecture, add:

```bash
--epochs 1 --max-train-batches 2 --max-val-batches 1 \
--workers 0 --device cpu --no-amp --out-dir runs/brief
```

A CUDA device is required to exercise CUDA mixed precision. On Linux, install
the PyTorch 2.5.1 CUDA build matching your system. The uv lock uses CPU wheels
on Linux/Windows by default. The paper used the cu118
build. After creating the environment, the corresponding pip command is:

```bash
uv pip install --python .venv/bin/python torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
```

Then run `.venv/bin/copra-flow train --config configs/train_report.json
--device cuda --amp` (or `uv run --no-sync copra-flow train ...`). Installing a
CUDA wheel is a platform-specific override of the standard lockfile; a normal
`uv run`/`uv sync` would restore the locked CPU build. Record `environment.json`
with results. These commands follow the [PyTorch version archive](https://pytorch.org/get-started/previous-versions/).
CPU and Apple MPS training use FP32. MPS may need normal process permissions
on macOS. `--device auto` chooses CUDA, then MPS, then CPU. Use `--workers 0`
in environments that prohibit shared-memory multiprocessing.

## Outputs

Each training run saves `config.json`, `environment.json`, `splits.json`,
`history.json`, `checkpoints/best.pt` and `checkpoints/last.pt`. The checkpoint
contains model and optimizer states, configurations, epoch, global step, and
best validation loss. The loader uses PyTorch's restricted tensor loader.
Do not assume exact training resumption: this release does not expose a resume
command or persist all RNG/scaler states. Use a fresh run directory per run.

The paper's best checkpoint is epoch 46, update 194074, validation flow loss
0.017534218672072208. Use this best checkpoint for the model described in the paper.
