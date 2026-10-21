"""Configuration objects for flow-matching training and sampling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetConfig:
    """Controls how generated COPRA artifact shards are exposed to PyTorch."""

    data_dir: Path
    split: str = "train"
    validation_fraction: float = 0.1
    seed: int = 42
    source_mode: str = "all"
    include_hard_negatives: bool = True
    use_source_weights: bool = True
    min_source_weight: float = 0.0
    max_source_weight: float = 5.0
    trace_transform: str = "sqrt"
    normalize_trace: bool = True
    cache_shards: int = 16
    max_shards: int | None = None
    source_kind_filter: str | None = None
    include_evaluation_fields: bool = False
    split_file: Path | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1).")
        if self.cache_shards < 0 or (self.max_shards is not None and self.max_shards < 1):
            raise ValueError("cache_shards must be nonnegative and max_shards positive.")


@dataclass(frozen=True)
class ModelConfig:
    """Architecture settings for the conditional vector-field network."""

    field_length: int = 256
    hidden_channels: int = 128
    residual_blocks: int = 8
    condition_dim: int = 256
    time_embedding_dim: int = 128
    trace_base_channels: int = 24
    trace_layers: int = 4
    dropout: float = 0.0
    condition_on_source: bool = False

    def __post_init__(self) -> None:
        for name in ("field_length", "hidden_channels", "residual_blocks", "condition_dim",
                     "time_embedding_dim", "trace_base_channels", "trace_layers"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive.")
        if self.time_embedding_dim % 2 or not 0 <= self.dropout < 1:
            raise ValueError("time_embedding_dim must be even; dropout must be in [0, 1).")


@dataclass(frozen=True)
class FlowLossConfig:
    """Loss settings for straight-line conditional flow matching."""

    time_eps: float = 1e-4
    path_noise_std: float = 0.0

    def __post_init__(self) -> None:
        if not 0 <= self.time_eps < 0.5 or self.path_noise_std < 0:
            raise ValueError("time_eps must be in [0, 0.5); path_noise_std must be nonnegative.")


@dataclass(frozen=True)
class TrainingConfig:
    """Trainer settings."""

    out_dir: Path
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    device: str = "auto"
    seed: int = 42
    grad_clip_norm: float = 1.0
    amp: bool = False
    log_every: int = 25
    validate_every: int = 1
    checkpoint_every: int = 1
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    compile_model: bool = False

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.num_workers < 0:
            raise ValueError("epochs/batch_size must be positive; num_workers must be nonnegative.")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay nonnegative.")
        for name in ("validate_every", "checkpoint_every", "log_every"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive.")
        for name in ("max_train_batches", "max_val_batches"):
            if getattr(self, name) is not None and getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive when supplied.")


@dataclass(frozen=True)
class SamplingConfig:
    """ODE integration settings for turning a source field into a prediction."""

    steps: int = 64
    method: str = "euler"
    return_trajectory: bool = False


def dataclass_to_json_dict(value: object) -> dict[str, Any]:
    """Convert a dataclass config to a JSON-serializable dictionary."""

    data = asdict(value)
    return _json_ready(data)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value
