"""Pulse-domain evaluation for trained flow-matching models."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm
from pnp_copra.checkpoints import load_checkpoint, dataset_settings

from .complex import channels_to_complex
from .config import DatasetConfig, SamplingConfig, dataclass_to_json_dict
from .metrics import (
    HANN_AMP_FRAC,
    HANN_PHASE_FRAC,
    aligned_pulse_mae,
    append_group_metrics,
    hann_denoise_time,
    higher_is_better_comparison,
    infer_pulse_type,
    lower_is_better_comparison,
    stats,
    summarize_groups,
    trace_objectives_for_field,
)


@dataclass(frozen=True)
class EvaluationConfig:
    """Controls pulse-domain evaluation of a trained checkpoint."""

    checkpoint: Path
    data_dir: Path | None = None
    split_file: Path | None = None
    split: str = "val"
    source_kind_filter: str | None = None
    batch_size: int = 64
    num_workers: int = 4
    device: str = "auto"
    max_examples: int | None = None
    sampling_steps: int = 64
    sampling_method: str = "euler"
    include_hann_baselines: bool = True
    hann_amp_frac: float = HANN_AMP_FRAC
    hann_phase_frac: float = HANN_PHASE_FRAC
    hann_use_time_window: bool = True
    include_trace_metrics: bool = False
    trace_frog_method: str = "tg"
    out_json: Path | None = None


def _dataset_config_from_checkpoint(
    checkpoint: dict[str, Any],
    config: EvaluationConfig,
) -> DatasetConfig:
    dataset_payload = dataset_settings(checkpoint, config.checkpoint, config.data_dir, config.split_file)
    dataset_payload["split"] = config.split
    dataset_payload["source_kind_filter"] = config.source_kind_filter
    dataset_payload["include_evaluation_fields"] = True
    return DatasetConfig(**dataset_payload)


def _make_loader(dataset: Any, config: EvaluationConfig, device: Any) -> Any:
    from torch.utils.data import DataLoader

    workers = int(config.num_workers)
    return DataLoader(
        dataset,
        batch_size=int(config.batch_size),
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


def _trim_batch(batch: dict[str, Any], count: int) -> dict[str, Any]:
    import torch

    trimmed: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            trimmed[key] = value[:count]
        elif isinstance(value, list):
            trimmed[key] = value[:count]
        else:
            trimmed[key] = value
    return trimmed


def _channel_mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def _trace_ssim_for_prediction(
    *,
    enabled: bool,
    field: np.ndarray,
    batch: dict[str, Any],
    index: int,
    frog_method: str,
) -> float:
    if not enabled:
        return float("nan")
    metrics = trace_objectives_for_field(
        field,
        measured_trace=batch["measured_trace_raw"][index].detach().cpu().numpy(),
        delay_axis=batch["delay_axis"][index].detach().cpu().numpy(),
        freq_axis=batch["freq_axis"][index].detach().cpu().numpy(),
        time_axis=batch["time_axis"][index].detach().cpu().numpy(),
        frog_method=frog_method,
    )
    return float(metrics["trace_ssim"])


def _append_if_finite(values: list[float], value: float) -> None:
    if np.isfinite(value):
        values.append(float(value))


def evaluate_pulse_mae(config: EvaluationConfig) -> dict[str, Any]:
    """Evaluate whether model sampling reduces aligned pulse-domain MAE."""

    import torch

    from .data import FlowMatchingDataset
    from .sampling import sample
    from .trainer import load_model_from_checkpoint, move_batch_to_device, resolve_device

    checkpoint_path = config.checkpoint.resolve()
    checkpoint = load_checkpoint(checkpoint_path)
    dataset_config = _dataset_config_from_checkpoint(checkpoint, config)
    device = resolve_device(config.device)
    model = load_model_from_checkpoint(checkpoint_path, device=str(device))
    dataset = FlowMatchingDataset(dataset_config)
    loader = _make_loader(dataset, config, device)
    sampling_config = SamplingConfig(
        steps=int(config.sampling_steps),
        method=str(config.sampling_method),
        return_trajectory=False,
    )

    metric_lists: dict[str, list[float]] = {
        "source_pulse_mae": [],
        "predicted_pulse_mae": [],
        "hann_source_pulse_mae": [],
        "copra_hann_direct_final_pulse_mae": [],
        "predicted_for_copra_hann_direct_final_pulse_mae": [],
        "copra_hann_guarded_final_pulse_mae": [],
        "predicted_for_copra_hann_guarded_final_pulse_mae": [],
        "copra_hann_best_final_pulse_mae": [],
        "predicted_for_copra_hann_best_final_pulse_mae": [],
        "source_channel_mae": [],
        "predicted_channel_mae": [],
        "hann_source_channel_mae": [],
        "source_trace_ssim": [],
        "predicted_trace_ssim": [],
        "hann_source_trace_ssim": [],
        "copra_hann_direct_final_trace_ssim": [],
        "predicted_for_copra_hann_direct_final_trace_ssim": [],
        "copra_hann_guarded_final_trace_ssim": [],
        "predicted_for_copra_hann_guarded_final_trace_ssim": [],
        "copra_hann_best_final_trace_ssim": [],
        "predicted_for_copra_hann_best_final_trace_ssim": [],
    }
    by_type: dict[str, dict[str, list[float]]] = {}
    by_source_kind: dict[str, dict[str, list[float]]] = {}
    examples_seen = 0

    progress = tqdm(loader, desc="pulse-mae eval", leave=True)
    for batch in progress:
        if config.max_examples is not None:
            remaining = int(config.max_examples) - examples_seen
            if remaining <= 0:
                break
            if int(batch["source"].shape[0]) > remaining:
                batch = _trim_batch(batch, remaining)

        device_batch = move_batch_to_device(
            {"source": batch["source"], "trace": batch["trace"]},
            device,
        )
        prediction = sample(model, device_batch["source"], device_batch["trace"], sampling_config)
        if isinstance(prediction, tuple):
            prediction = prediction[0]

        source = batch["source"].detach().cpu().numpy()
        target = batch["target"].detach().cpu().numpy()
        predicted = prediction.detach().cpu().numpy()
        uids = list(batch["uid"])
        source_kinds = list(batch["source_kind"])

        for index, uid in enumerate(uids):
            target_field = channels_to_complex(target[index])
            source_field = channels_to_complex(source[index])
            predicted_field = channels_to_complex(predicted[index])
            source_mae = aligned_pulse_mae(target_field, source_field)
            predicted_mae = aligned_pulse_mae(target_field, predicted_field)
            source_ssim = float(batch["source_trace_ssim"][index])
            predicted_ssim = _trace_ssim_for_prediction(
                enabled=config.include_trace_metrics,
                field=predicted_field,
                batch=batch,
                index=index,
                frog_method=config.trace_frog_method,
            )

            metric_lists["source_pulse_mae"].append(source_mae)
            metric_lists["predicted_pulse_mae"].append(predicted_mae)
            metric_lists["source_channel_mae"].append(_channel_mae(source[index], target[index]))
            metric_lists["predicted_channel_mae"].append(_channel_mae(predicted[index], target[index]))
            _append_if_finite(metric_lists["source_trace_ssim"], source_ssim)
            _append_if_finite(metric_lists["predicted_trace_ssim"], predicted_ssim)

            group_metrics = {
                "source_pulse_mae": source_mae,
                "predicted_pulse_mae": predicted_mae,
                "source_trace_ssim": source_ssim,
                "predicted_trace_ssim": predicted_ssim,
            }
            _add_hann_source_metrics(config, batch, index, target_field, target[index], source_field, metric_lists, group_metrics)
            _add_stored_hann_metrics(batch, index, target_field, predicted_mae, predicted_ssim, metric_lists, group_metrics)
            append_group_metrics(by_type, infer_pulse_type(str(uid)), **group_metrics)
            append_group_metrics(by_source_kind, str(source_kinds[index]), **group_metrics)

        examples_seen += len(uids)
        if metric_lists["source_pulse_mae"]:
            progress.set_postfix(
                improved=(
                    f"{np.mean(np.asarray(metric_lists['predicted_pulse_mae']) < np.asarray(metric_lists['source_pulse_mae'])):.3f}"
                )
            )

    result = _build_result(
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        dataset_config=dataset_config,
        dataset=dataset,
        examples_seen=examples_seen,
        sampling_config=sampling_config,
        eval_config=config,
        metric_lists=metric_lists,
        by_type=by_type,
        by_source_kind=by_source_kind,
    )
    if config.out_json is not None:
        out_path = config.out_json.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def _add_hann_source_metrics(
    config: EvaluationConfig,
    batch: dict[str, Any],
    index: int,
    target_field: np.ndarray,
    target_channels: np.ndarray,
    source_field: np.ndarray,
    metric_lists: dict[str, list[float]],
    group_metrics: dict[str, float],
) -> None:
    if not config.include_hann_baselines:
        return
    hann_field = hann_denoise_time(
        source_field,
        amp_frac=float(config.hann_amp_frac),
        phase_frac=float(config.hann_phase_frac),
    )
    if config.hann_use_time_window:
        hann_field = hann_field * batch["time_window"][index].detach().cpu().numpy()
    hann_mae = aligned_pulse_mae(target_field, hann_field)
    hann_channels = np.stack((hann_field.real, hann_field.imag), axis=0)
    hann_ssim = _trace_ssim_for_prediction(
        enabled=config.include_trace_metrics,
        field=hann_field,
        batch=batch,
        index=index,
        frog_method=config.trace_frog_method,
    )

    metric_lists["hann_source_pulse_mae"].append(hann_mae)
    metric_lists["hann_source_channel_mae"].append(_channel_mae(hann_channels, target_channels))
    _append_if_finite(metric_lists["hann_source_trace_ssim"], hann_ssim)
    group_metrics["hann_source_pulse_mae"] = hann_mae
    group_metrics["hann_source_trace_ssim"] = hann_ssim


def _add_stored_hann_metrics(
    batch: dict[str, Any],
    index: int,
    target_field: np.ndarray,
    predicted_mae: float,
    predicted_ssim: float,
    metric_lists: dict[str, list[float]],
    group_metrics: dict[str, float],
) -> None:
    final_candidates: list[tuple[float, float]] = []
    for prefix, label in (
        ("copra_hann_direct_final", "direct"),
        ("copra_hann_guarded_final", "guarded"),
    ):
        if not bool(batch[f"{prefix}_available"][index]):
            continue
        field = channels_to_complex(batch[prefix][index].detach().cpu().numpy())
        mae = aligned_pulse_mae(target_field, field)
        ssim = float(batch[f"{prefix}_trace_ssim"][index])
        metric_lists[f"copra_hann_{label}_final_pulse_mae"].append(mae)
        metric_lists[f"predicted_for_copra_hann_{label}_final_pulse_mae"].append(predicted_mae)
        _append_if_finite(metric_lists[f"copra_hann_{label}_final_trace_ssim"], ssim)
        _append_if_finite(metric_lists[f"predicted_for_copra_hann_{label}_final_trace_ssim"], predicted_ssim)
        group_metrics[f"copra_hann_{label}_final_pulse_mae"] = mae
        group_metrics[f"predicted_for_copra_hann_{label}_final_pulse_mae"] = predicted_mae
        group_metrics[f"copra_hann_{label}_final_trace_ssim"] = ssim
        group_metrics[f"predicted_for_copra_hann_{label}_final_trace_ssim"] = predicted_ssim
        final_candidates.append((mae, ssim))

    if final_candidates:
        best_mae, best_ssim = min(final_candidates, key=lambda pair: pair[0])
        metric_lists["copra_hann_best_final_pulse_mae"].append(best_mae)
        metric_lists["predicted_for_copra_hann_best_final_pulse_mae"].append(predicted_mae)
        _append_if_finite(metric_lists["copra_hann_best_final_trace_ssim"], best_ssim)
        _append_if_finite(metric_lists["predicted_for_copra_hann_best_final_trace_ssim"], predicted_ssim)
        group_metrics["copra_hann_best_final_pulse_mae"] = best_mae
        group_metrics["predicted_for_copra_hann_best_final_pulse_mae"] = predicted_mae
        group_metrics["copra_hann_best_final_trace_ssim"] = best_ssim
        group_metrics["predicted_for_copra_hann_best_final_trace_ssim"] = predicted_ssim


def _build_result(
    *,
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    dataset_config: DatasetConfig,
    dataset: Any,
    examples_seen: int,
    sampling_config: SamplingConfig,
    eval_config: EvaluationConfig,
    metric_lists: dict[str, list[float]],
    by_type: dict[str, dict[str, list[float]]],
    by_source_kind: dict[str, dict[str, list[float]]],
) -> dict[str, Any]:
    flow_vs_source = lower_is_better_comparison(
        metric_lists["source_pulse_mae"],
        metric_lists["predicted_pulse_mae"],
    )
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_global_step": int(checkpoint.get("global_step", -1)),
        "checkpoint_best_val_loss": float(checkpoint.get("best_val_loss", float("nan"))),
        "data_dir": str(dataset_config.data_dir),
        "split": dataset_config.split,
        "source_kind_filter": dataset_config.source_kind_filter,
        "dataset_shards": len(dataset.shard_paths),
        "dataset_examples": len(dataset),
        "evaluated_examples": int(examples_seen),
        "sampling": dataclass_to_json_dict(sampling_config),
        "trace_metrics": {
            "included": bool(eval_config.include_trace_metrics),
            "frog_method": str(eval_config.trace_frog_method),
            "higher_is_better": True,
        },
        "hann_baselines": {
            "included": bool(eval_config.include_hann_baselines),
            "amp_frac": float(eval_config.hann_amp_frac),
            "phase_frac": float(eval_config.hann_phase_frac),
            "use_time_window": bool(eval_config.hann_use_time_window),
        },
        **{name: stats(values) for name, values in metric_lists.items()},
        "flow_vs_source": flow_vs_source,
        "hann_source_vs_source": lower_is_better_comparison(
            metric_lists["source_pulse_mae"],
            metric_lists["hann_source_pulse_mae"],
        ),
        "flow_vs_hann_source": lower_is_better_comparison(
            metric_lists["hann_source_pulse_mae"],
            metric_lists["predicted_pulse_mae"],
        ),
        "flow_vs_copra_hann_direct_final": lower_is_better_comparison(
            metric_lists["copra_hann_direct_final_pulse_mae"],
            metric_lists["predicted_for_copra_hann_direct_final_pulse_mae"],
        ),
        "flow_vs_copra_hann_guarded_final": lower_is_better_comparison(
            metric_lists["copra_hann_guarded_final_pulse_mae"],
            metric_lists["predicted_for_copra_hann_guarded_final_pulse_mae"],
        ),
        "flow_vs_copra_hann_best_final": lower_is_better_comparison(
            metric_lists["copra_hann_best_final_pulse_mae"],
            metric_lists["predicted_for_copra_hann_best_final_pulse_mae"],
        ),
        "trace_ssim_flow_vs_source": higher_is_better_comparison(
            metric_lists["source_trace_ssim"],
            metric_lists["predicted_trace_ssim"],
        ),
        "trace_ssim_flow_vs_copra_hann_best_final": higher_is_better_comparison(
            metric_lists["copra_hann_best_final_trace_ssim"],
            metric_lists["predicted_for_copra_hann_best_final_trace_ssim"],
        ),
        "absolute_reduction": flow_vs_source["absolute_reduction"],
        "relative_reduction": flow_vs_source["relative_reduction"],
        "improved_fraction": flow_vs_source["improved_fraction"],
        "by_pulse_type": summarize_groups(by_type),
        "by_source_kind": summarize_groups(by_source_kind),
    }
