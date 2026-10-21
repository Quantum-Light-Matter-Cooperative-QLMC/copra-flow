"""Explicit pulse-level splits, independent of directory relocation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

from .flow_matching.inspection import discover_shards


def split_payload(paths: list[Path], fraction: float, seed: int) -> dict:
    if not 0.0 <= fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1).")
    paths = sorted(paths)
    shuffled = list(paths)
    random.Random(seed).shuffle(shuffled)
    count = int(round(len(paths) * fraction))
    if len(paths) > 1 and fraction > 0:
        count = min(max(1, count), len(paths) - 1)
    validation = {p.stem for p in shuffled[:count]}
    return {
        "schema_version": 1,
        "unit": "pulse_uid",
        "seed": seed,
        "validation_fraction": fraction,
        "train": [p.stem for p in paths if p.stem not in validation],
        "val": [p.stem for p in paths if p.stem in validation],
    }


def read_splits(path: Path, paths: list[Path]) -> dict:
    payload = json.loads(Path(path).read_text())
    if payload.get("unit") != "pulse_uid" or payload.get("schema_version") != 1:
        raise ValueError("Unsupported split manifest; expected schema_version=1, unit=pulse_uid.")
    train, val = payload["train"], payload["val"]
    if not all(isinstance(uid, str) for uid in train + val):
        raise ValueError("Split entries must be pulse UID strings.")
    if len(set(train)) != len(train) or len(set(val)) != len(val):
        raise ValueError("Duplicate UID within a split.")
    if set(train) & set(val):
        raise ValueError("Pulse leakage: train and validation splits overlap.")
    known = {p.stem for p in paths}
    excluded = payload.get("excluded", [])
    if set(excluded) & (set(train) | set(val)) or len(excluded) != len(set(excluded)):
        raise ValueError("Excluded pulse UIDs overlap an active split or contain duplicates.")
    listed = set(train) | set(val) | set(excluded)
    if known != listed:
        raise ValueError(f"Split manifest does not match dataset: {len(listed-known)} missing shards, "
                         f"{len(known-listed)} unassigned shards.")
    return payload


def write_splits(data_dir: Path, out: Path, *, fraction: float = 0.1, seed: int = 42) -> dict:
    payload = split_payload(discover_shards(data_dir), fraction, seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return {"path": str(out), "train_pulses": len(payload["train"]),
            "validation_pulses": len(payload["val"]),
            "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    print(json.dumps(write_splits(args.data_dir, args.out, fraction=args.val_fraction, seed=args.seed), indent=2))
    return 0
