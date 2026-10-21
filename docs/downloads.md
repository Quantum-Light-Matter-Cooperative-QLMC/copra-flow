# Download the model and dataset

The model and dataset are hosted separately on Hugging Face. Public downloads
do not require an account. Commands below run from the code repository root.

```bash
uv tool install huggingface_hub
```

## Model

```bash
hf download atal-narayan/copra-flow --revision v0.1.0 --local-dir checkpoints
```

Use `checkpoints/copra-flow.pt` for inference. The complete model download also
includes the model card, configuration, training history, split, and checksums.
The checkpoint contains the original model tensors and validation metadata;
optimizer state is omitted and machine-specific paths are replaced with
relative paths. Its checksum therefore differs from the original training
checkpoint. Both hashes are recorded in `release.json`.

The weights are licensed under MIT. This is a custom PyTorch checkpoint used
by `copra-flow`, rather than a Transformers model.

## Dataset

```bash
hf download atal-narayan/copra-flow-tg-frog --repo-type dataset \
  --revision v0.1.0 --local-dir data/flow_matching_10k
python data/flow_matching_10k/extract.py
uv run copra-flow inspect --data-dir data/flow_matching_10k
```

Allow approximately 8 GB of free space for the downloaded archives and
extracted files. Nine training archives contain 9,000 pulses, and one validation
archive contains 1,000 pulses. Every original NPZ shard is preserved byte for
byte. The extraction script checks archive and shard hashes before writing
the final files under `shards/`. After successful extraction, you may remove
the tar files under `archives/` to recover space.

Download and extract all ten archives for the supplied training and benchmark
commands, which validate the complete dataset against the fixed split.
Each pulse has 30 source–target pairs, for 300,000 pairs in total. Related
sources stay in the same partition. Use the supplied `splits.json` when training:

```bash
uv run copra-flow train --data-dir data/flow_matching_10k \
  --split-file data/flow_matching_10k/splits.json --out-dir runs/copra-flow --amp
```

The dataset uses CC BY 4.0. It contains simulated TG-FROG traces and pulse fields;
experimental measurements are not included. See the dataset card and
`DATA_FORMAT.md` for units, normalization, and limitations.

## Check downloaded files

Both repositories provide `SHA256SUMS`. On macOS, verify a complete download
from inside its directory with:

```bash
shasum -a 256 -c SHA256SUMS
```

On Linux, use `sha256sum -c SHA256SUMS`. Dataset extraction additionally verifies
every extracted shard against `inventory.json`.

The commands pin the `v0.1.0` artifact release. For an experiment that needs a
specific snapshot, record the resolved Hugging Face commit alongside your
settings and results. Updating code on GitHub does not change these artifact
files.

The [publication verification record](../reproduction/reference_results/artifact_publication.json)
lists the exact release commits and completed checks: anonymous downloads,
all 10,000 shard hashes, inference, two training optimizer updates with the
full architecture, and evaluation on eight validation pulses. These are
publication checks; the full 50-epoch training run was not repeated.
