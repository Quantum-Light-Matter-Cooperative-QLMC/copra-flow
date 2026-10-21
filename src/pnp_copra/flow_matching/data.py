"""PyTorch datasets for generated flow-matching artifact shards."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

import numpy as np

from .complex import complex_to_channels, preprocess_trace
from .config import DatasetConfig
from .inspection import discover_shards

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by users without the train extra.
    raise ModuleNotFoundError(
        "Flow-matching datasets require PyTorch. Install with `uv sync --extra train` "
        "from the package directory."
    ) from exc


REQUIRED_KEYS = (
    "target_field",
    "source_fields",
    "measured_trace",
    "source_weight",
    "hard_negative",
)
BASELINE_KIND_TO_FIELD = {
    "copra_hann_direct_final": "copra_denoised_final",
    "copra_hann_guarded_final": "copra_denoised_guarded_final",
}


@dataclass(frozen=True)
class IndexedExample:
    shard_index: int
    source_index: int


class ShardCache:
    """Small LRU cache for decoded ``.npz`` shards within one DataLoader worker."""

    def __init__(self, max_shards: int):
        self.max_shards = max(0, int(max_shards))
        self._cache: OrderedDict[Path, dict[str, np.ndarray]] = OrderedDict()

    def get(self, path: Path) -> dict[str, np.ndarray]:
        if self.max_shards > 0 and path in self._cache:
            arrays = self._cache.pop(path)
            self._cache[path] = arrays
            return arrays

        arrays = load_shard(path)
        if self.max_shards > 0:
            self._cache[path] = arrays
            while len(self._cache) > self.max_shards:
                self._cache.popitem(last=False)
        return arrays


def load_shard(path: Path) -> dict[str, np.ndarray]:
    """Load a generated artifact shard into ordinary NumPy arrays."""

    with np.load(path, allow_pickle=False) as handle:
        missing = [key for key in REQUIRED_KEYS if key not in handle.files]
        if missing:
            raise KeyError(f"{path} is missing required key(s): {missing}")
        return {key: np.asarray(handle[key]) for key in handle.files}


def split_shards(
    paths: list[Path],
    *,
    split: str,
    validation_fraction: float,
    seed: int,
    split_file: Path | None = None,
) -> list[Path]:
    """Deterministically split shard paths into train/validation partitions."""

    split = split.lower()
    if split_file is not None:
        from pnp_copra.splits import read_splits
        payload = read_splits(split_file, paths)
        if split in {"all", "full"}:
            return list(paths)
        if split not in {"train", "val", "validation"}:
            raise ValueError("split must be one of: train, val, validation, all.")
        selected = set(payload["val" if split == "validation" else split])
        return [path for path in paths if path.stem in selected]
    if split in {"all", "full"}:
        return list(paths)
    if split not in {"train", "val", "validation"}:
        raise ValueError("split must be one of: train, val, validation, all.")

    shuffled = list(paths)
    random.Random(int(seed)).shuffle(shuffled)
    val_count = int(round(len(shuffled) * float(validation_fraction)))
    if len(shuffled) > 1 and validation_fraction > 0.0:
        val_count = min(max(1, val_count), len(shuffled) - 1)
    val_paths = set(shuffled[:val_count])
    if split in {"val", "validation"}:
        return [path for path in paths if path in val_paths]
    return [path for path in paths if path not in val_paths]


def _valid_source_indices(path: Path, config: DatasetConfig) -> np.ndarray:
    with np.load(path, allow_pickle=False) as handle:
        if "source_fields" not in handle.files:
            raise KeyError(f"{path} is missing source_fields.")
        count = int(handle["source_fields"].shape[0])
        mask = np.ones(count, dtype=bool)
        if not config.include_hard_negatives and "hard_negative" in handle.files:
            mask &= ~np.asarray(handle["hard_negative"], dtype=bool)
        if "source_weight" in handle.files:
            weights = np.asarray(handle["source_weight"], dtype=np.float32)
            mask &= weights >= float(config.min_source_weight)
        if config.source_kind_filter is not None:
            if "source_kind" not in handle.files:
                mask &= False
            else:
                kinds = np.asarray(handle["source_kind"]).astype(str)
                mask &= kinds == str(config.source_kind_filter)
        return np.nonzero(mask)[0].astype(np.int64)


def _first_source_kind_index(arrays: dict[str, np.ndarray], source_kind: str) -> int | None:
    if "source_kind" not in arrays:
        return None
    matches = np.nonzero(np.asarray(arrays["source_kind"]).astype(str) == source_kind)[0]
    if matches.size == 0:
        return None
    return int(matches[0])


class FlowMatchingDataset(Dataset):
    """Dataset of source-to-target pulse refinements conditioned on a trace."""

    def __init__(self, config: DatasetConfig):
        super().__init__()
        self.config = config
        if config.source_mode not in {"all", "random"}:
            raise ValueError("source_mode must be 'all' or 'random'.")

        paths = split_shards(
            discover_shards(config.data_dir),
            split=config.split,
            validation_fraction=config.validation_fraction,
            seed=config.seed,
            split_file=config.split_file,
        )
        if config.max_shards is not None:
            paths = paths[: max(0, int(config.max_shards))]

        self.shard_paths: list[Path] = []
        self.source_indices_by_shard: list[np.ndarray] = []
        for path in paths:
            source_indices = _valid_source_indices(path, config)
            if source_indices.size == 0:
                continue
            self.shard_paths.append(path)
            self.source_indices_by_shard.append(source_indices)

        if not self.shard_paths:
            raise ValueError("No shards remain after applying dataset filters.")

        self.examples: list[IndexedExample] = []
        if config.source_mode == "all":
            for shard_index, source_indices in enumerate(self.source_indices_by_shard):
                self.examples.extend(
                    IndexedExample(shard_index=shard_index, source_index=int(source_index))
                    for source_index in source_indices
                )
        else:
            self.examples = [
                IndexedExample(shard_index=shard_index, source_index=-1)
                for shard_index in range(len(self.shard_paths))
            ]

        self.cache = ShardCache(config.cache_shards)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        path = self.shard_paths[example.shard_index]
        arrays = self.cache.get(path)
        source_index = example.source_index
        if source_index < 0:
            candidates = self.source_indices_by_shard[example.shard_index]
            source_index = int(candidates[np.random.randint(0, len(candidates))])

        target = complex_to_channels(arrays["target_field"])
        source = complex_to_channels(arrays["source_fields"][source_index])
        trace = preprocess_trace(
            arrays["measured_trace"],
            transform=self.config.trace_transform,
            normalize=self.config.normalize_trace,
        )
        weight = 1.0
        if self.config.use_source_weights:
            weights = np.asarray(arrays["source_weight"], dtype=np.float32)
            weight = float(weights[source_index])
            weight = min(max(weight, self.config.min_source_weight), self.config.max_source_weight)

        hard_negative = bool(np.asarray(arrays["hard_negative"])[source_index])
        uid = path.stem
        sample = {
            "source": torch.from_numpy(source),
            "target": torch.from_numpy(target),
            "trace": torch.from_numpy(trace),
            "weight": torch.tensor(weight, dtype=torch.float32),
            "hard_negative": torch.tensor(hard_negative, dtype=torch.bool),
            "source_index": torch.tensor(source_index, dtype=torch.long),
            "uid": uid,
        }
        if not self.config.include_evaluation_fields:
            return sample

        sample.update(
            {
                "source_kind": (
                    str(np.asarray(arrays["source_kind"])[source_index])
                    if "source_kind" in arrays
                    else "unknown"
                ),
                "measured_trace_raw": torch.from_numpy(
                    np.asarray(arrays["measured_trace"], dtype=np.float32)
                ),
                "delay_axis": torch.from_numpy(np.asarray(arrays["delay_axis"], dtype=np.float64)),
                "freq_axis": torch.from_numpy(np.asarray(arrays["freq_axis"], dtype=np.float64)),
                "time_axis": torch.from_numpy(np.asarray(arrays["time_axis"], dtype=np.float64)),
                "source_trace_ssim": torch.tensor(
                    float(np.asarray(arrays.get("trace_ssim", [np.nan]))[source_index])
                    if "trace_ssim" in arrays
                    else float("nan"),
                    dtype=torch.float32,
                ),
                "source_trace_error": torch.tensor(
                    float(np.asarray(arrays.get("trace_error", [np.nan]))[source_index])
                    if "trace_error" in arrays
                    else float("nan"),
                    dtype=torch.float32,
                ),
            }
        )
        time_window = arrays.get("time_window")
        if time_window is None:
            time_window = np.ones(arrays["target_field"].shape, dtype=np.float32)
        sample["time_window"] = torch.from_numpy(np.asarray(time_window, dtype=np.float32))

        zero_field = np.zeros_like(arrays["target_field"])
        for sample_key, source_kind in BASELINE_KIND_TO_FIELD.items():
            baseline_index = _first_source_kind_index(arrays, source_kind)
            available = baseline_index is not None
            baseline_field = arrays["source_fields"][baseline_index] if available else zero_field
            sample[sample_key] = torch.from_numpy(complex_to_channels(baseline_field))
            sample[f"{sample_key}_available"] = torch.tensor(available, dtype=torch.bool)
            trace_ssim_values = arrays.get("trace_ssim")
            sample[f"{sample_key}_trace_ssim"] = torch.tensor(
                float(trace_ssim_values[baseline_index])
                if available and trace_ssim_values is not None
                else float("nan"),
                dtype=torch.float32,
            )
            trace_error_values = arrays.get("trace_error")
            sample[f"{sample_key}_trace_error"] = torch.tensor(
                float(trace_error_values[baseline_index])
                if available and trace_error_values is not None
                else float("nan"),
                dtype=torch.float32,
            )
        return sample
