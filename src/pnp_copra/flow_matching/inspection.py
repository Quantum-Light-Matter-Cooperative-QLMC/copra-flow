"""Lightweight dataset inspection utilities that do not require PyTorch."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def discover_shards(data_dir: Path) -> list[Path]:
    root = Path(data_dir)
    shard_dir = root / "shards" if (root / "shards").is_dir() else root
    paths = sorted(shard_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz shards found under {shard_dir}.")
    return paths


def dataset_inventory(data_dir: Path, max_examples: int = 3) -> dict[str, Any]:
    paths = discover_shards(data_dir)
    examples: list[dict[str, Any]] = []
    total_sources = 0
    hard_negatives = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as handle:
            source_count = int(handle["source_fields"].shape[0])
            total_sources += source_count
            if "hard_negative" in handle.files:
                hard_negatives += int(np.asarray(handle["hard_negative"], dtype=bool).sum())
            if len(examples) < max_examples:
                examples.append(
                    {
                        "path": str(path),
                        "target_field_shape": tuple(handle["target_field"].shape),
                        "source_fields_shape": tuple(handle["source_fields"].shape),
                        "measured_trace_shape": tuple(handle["measured_trace"].shape),
                        "source_count": source_count,
                    }
                )
    return {
        "shards": len(paths),
        "source_examples": total_sources,
        "hard_negatives": hard_negatives,
        "examples": examples,
    }
