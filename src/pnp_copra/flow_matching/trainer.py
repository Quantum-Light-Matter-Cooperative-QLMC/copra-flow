"""Training loop for trace-conditioned flow matching."""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from pnp_copra.checkpoints import load_checkpoint

from .config import (
    DatasetConfig,
    FlowLossConfig,
    ModelConfig,
    TrainingConfig,
    dataclass_to_json_dict,
)
from .data import FlowMatchingDataset
from .losses import flow_matching_loss
from .model import FlowMatchingModel


def set_training_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(device: str) -> torch.device:
    requested = device.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {requested!r} was requested, but CUDA is not available."
        )
    if requested == "mps":
        backend = getattr(torch.backends, "mps", None)
        if backend is None or not backend.is_available():
            raise RuntimeError("MPS was requested, but the MPS backend is not available.")
    if requested not in {"cpu", "cuda", "mps"} and not requested.startswith("cuda:"):
        raise ValueError(f"Unsupported device {device!r}; use auto, cpu, cuda[:N], or mps.")
    return torch.device(requested)


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed + worker_id)


def build_dataloaders(
    dataset_config: DatasetConfig,
    training_config: TrainingConfig,
) -> tuple[DataLoader, DataLoader | None]:
    train_config = DatasetConfig(
        **{
            **dataclass_to_json_dict(dataset_config),
            "data_dir": dataset_config.data_dir,
            "split": "train",
        }
    )
    train_dataset = FlowMatchingDataset(train_config)
    generator = torch.Generator().manual_seed(int(training_config.seed))
    loader_kwargs = {
        "batch_size": int(training_config.batch_size),
        "num_workers": int(training_config.num_workers),
        "pin_memory": resolve_device(training_config.device).type == "cuda",
        "persistent_workers": int(training_config.num_workers) > 0,
        "worker_init_fn": seed_worker if int(training_config.num_workers) > 0 else None,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_kwargs)
    if dataset_config.validation_fraction <= 0.0:
        return train_loader, None

    val_config = DatasetConfig(
        **{
            **dataclass_to_json_dict(dataset_config),
            "data_dir": dataset_config.data_dir,
            "split": "val",
        }
    )
    val_dataset = FlowMatchingDataset(val_config)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp_path, path)


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    configs: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "best_val_loss": float(best_val_loss),
            "configs": configs,
        },
        tmp_path,
    )
    os.replace(tmp_path, path)


def load_model_from_checkpoint(path: Path, device: str = "auto") -> FlowMatchingModel:
    target_device = resolve_device(device)
    checkpoint = load_checkpoint(path)
    model_config = ModelConfig(**checkpoint["configs"]["model"])
    model = FlowMatchingModel(model_config).to(target_device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


@torch.no_grad()
def validate(
    model: FlowMatchingModel,
    loader: DataLoader,
    loss_config: FlowLossConfig,
    *,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    velocity_mses: list[float] = []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        metrics = flow_matching_loss(model, batch, loss_config)
        losses.append(float(metrics["loss"].detach().cpu()))
        velocity_mses.append(float(metrics["velocity_mse"].detach().cpu()))
    return {
        "loss": _mean(losses),
        "velocity_mse": _mean(velocity_mses),
    }


def train_flow_matching(
    dataset_config: DatasetConfig,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    loss_config: FlowLossConfig | None = None,
) -> dict[str, Any]:
    """Train a flow-matching model and write checkpoints/metrics."""

    loss_config = loss_config or FlowLossConfig()
    out_dir = Path(training_config.out_dir)
    if any((out_dir / "checkpoints").glob("*.pt")):
        raise FileExistsError("Training output already contains checkpoints; choose a fresh --out-dir.")
    set_training_seed(training_config.seed)
    device = resolve_device(training_config.device)
    if device.type in {"cuda", "mps"}:
        torch.set_float32_matmul_precision("high")

    train_loader, val_loader = build_dataloaders(dataset_config, training_config)
    example = train_loader.dataset[0]
    if example["source"].shape[-1] != model_config.field_length:
        raise ValueError("Model field_length does not match the dataset grid.")
    raw_model = FlowMatchingModel(model_config).to(device)
    model: nn.Module = raw_model
    if training_config.compile_model:
        model = torch.compile(raw_model)  # type: ignore[assignment]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config.learning_rate),
        weight_decay=float(training_config.weight_decay),
    )
    use_amp = bool(training_config.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Record the actual pulse membership, including any user-supplied split.
    split_path = out_dir / "splits.json"
    from pnp_copra.splits import split_payload
    from .inspection import discover_shards
    membership = split_payload(discover_shards(dataset_config.data_dir),
                               dataset_config.validation_fraction, dataset_config.seed)
    membership["train"] = [p.stem for p in train_loader.dataset.shard_paths]
    membership["val"] = [] if val_loader is None else [p.stem for p in val_loader.dataset.shard_paths]
    active = set(membership["train"]) | set(membership["val"])
    membership["excluded"] = [p.stem for p in discover_shards(dataset_config.data_dir) if p.stem not in active]
    save_json(split_path, membership)
    configs = {
        "dataset": dataclass_to_json_dict(dataset_config),
        "model": dataclass_to_json_dict(model_config),
        "training": dataclass_to_json_dict(training_config),
        "loss": dataclass_to_json_dict(loss_config),
    }
    with np.load(train_loader.dataset.shard_paths[0], allow_pickle=False) as shard:
        if "time_axis" in shard:
            axis = np.asarray(shard["time_axis"], dtype=np.float64)
            configs["grid"] = {"field_length": int(axis.size), "dt_s": float(np.diff(axis).mean())}
    configs["dataset"]["split_file"] = str(split_path.resolve())
    save_json(out_dir / "config.json", configs)
    import platform
    save_json(out_dir / "environment.json", {
        "python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "device": str(device), "cuda_build": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    })

    best_val_loss = float("inf")
    global_step = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(training_config.epochs) + 1):
        model.train()
        train_losses: list[float] = []
        progress = tqdm(
            train_loader,
            desc=f"epoch {epoch}/{training_config.epochs}",
            leave=True,
        )
        for batch_index, batch in enumerate(progress, start=1):
            if (
                training_config.max_train_batches is not None
                and batch_index > int(training_config.max_train_batches)
            ):
                break
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                metrics = flow_matching_loss(model, batch, loss_config)
                loss = metrics["loss"]
            scaler.scale(loss).backward()
            if training_config.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            loss_value = float(loss.detach().cpu())
            train_losses.append(loss_value)
            if batch_index % max(1, int(training_config.log_every)) == 0:
                progress.set_postfix(loss=f"{loss_value:.4e}")

        val_metrics: dict[str, float] | None = None
        if val_loader is not None and epoch % max(1, int(training_config.validate_every)) == 0:
            val_metrics = validate(
                model,
                val_loader,
                loss_config,
                device=device,
                max_batches=training_config.max_val_batches,
            )
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                save_checkpoint(
                    out_dir / "checkpoints" / "best.pt",
                    model=raw_model,
                    optimizer=optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    best_val_loss=best_val_loss,
                    configs=configs,
                )

        if epoch % training_config.checkpoint_every == 0 or epoch == training_config.epochs:
            save_checkpoint(
                out_dir / "checkpoints" / "last.pt",
                model=raw_model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                best_val_loss=best_val_loss,
                configs=configs,
            )

        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": _mean(train_losses),
            "val_loss": None if val_metrics is None else val_metrics["loss"],
            "val_velocity_mse": None if val_metrics is None else val_metrics["velocity_mse"],
            "best_val_loss": best_val_loss,
        }
        history.append(row)
        save_json(out_dir / "history.json", {"history": history})
        print(json.dumps(row, sort_keys=True), flush=True)

    return {
        "out_dir": str(out_dir),
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "epochs": int(training_config.epochs),
    }
