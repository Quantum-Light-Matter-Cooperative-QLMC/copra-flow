"""Portable, restricted checkpoint loading for tensor-and-JSON checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not {"model", "configs"} <= payload.keys():
        raise ValueError("Expected a COPRA Flow checkpoint containing model and configs.")
    if not isinstance(payload["configs"], dict) or "model" not in payload["configs"]:
        raise ValueError("Checkpoint is missing the model configuration.")
    return payload


def dataset_settings(checkpoint: dict, checkpoint_path: Path, data_dir: Path | None,
                     split_file: Path | None = None) -> dict:
    """Resolve relocated datasets and a split saved alongside a training run."""
    payload = dict(checkpoint["configs"]["dataset"])
    payload["data_dir"] = Path(data_dir or payload["data_dir"]).resolve()
    selected = split_file or payload.get("split_file")
    if selected and not Path(selected).is_file():
        sibling = Path(checkpoint_path).resolve().parent.parent / "splits.json"
        if sibling.is_file():
            selected = sibling
        else:
            raise FileNotFoundError("Checkpoint split manifest is unavailable; pass --split-file.")
    payload["split_file"] = None if selected is None else Path(selected)
    return payload
