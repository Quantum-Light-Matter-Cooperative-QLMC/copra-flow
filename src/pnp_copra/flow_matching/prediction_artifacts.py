"""Save and evaluate flow-matching prediction artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm
from pnp_copra.checkpoints import load_checkpoint, dataset_settings

from .complex import channels_to_complex
from .config import DatasetConfig, SamplingConfig, dataclass_to_json_dict
from .metrics import (
    aligned_pulse_mae,
    higher_is_better_comparison,
    lower_is_better_comparison,
    stats,
    trace_objectives_for_field,
)


@dataclass(frozen=True)
class PredictionArtifactConfig:
    """Controls one-time saving of flow outputs."""

    checkpoint: Path
    out_dir: Path
    data_dir: Path | None = None
    split_file: Path | None = None
    split: str = "val"
    source_kind: str = "copra_final"
    batch_size: int = 64
    num_workers: int = 4
    device: str = "auto"
    max_examples: int | None = None
    sampling_steps: int = 64
    sampling_method: str = "euler"
    include_measured_trace: bool = True
    overwrite: bool = False


@dataclass(frozen=True)
class PredictionArtifactEvaluationConfig:
    """Controls metrics computed from saved prediction artifacts."""

    predictions_dir: Path
    include_trace_metrics: bool = False
    trace_frog_method: str = "tg"
    max_examples: int | None = None
    out_json: Path | None = None


def save_prediction_artifacts(config: PredictionArtifactConfig) -> dict[str, Any]:
    """Run the model once and persist target/source/flow fields for later metrics."""

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
    loader = _make_loader(dataset, config.batch_size, config.num_workers, device)
    sampling_config = SamplingConfig(
        steps=int(config.sampling_steps),
        method=str(config.sampling_method),
        return_trajectory=False,
    )

    out_dir = config.out_dir.resolve()
    shard_dir = out_dir / "shards"
    checkpoint_digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    signature = {
        "checkpoint_sha256": checkpoint_digest,
        "source_kind": config.source_kind, "split": config.split,
        "sampling": dataclass_to_json_dict(sampling_config),
        "max_examples": config.max_examples, "include_measured_trace": config.include_measured_trace,
        "pulse_uids_sha256": hashlib.sha256("\n".join(p.stem for p in dataset.shard_paths).encode()).hexdigest(),
    }
    existing_paths = list(shard_dir.glob("*.npz"))
    if existing_paths:
        manifest_path = out_dir / "manifest.json"
        previous = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        if previous.get("run_signature") != signature:
            raise ValueError("Prediction directory belongs to a different or unverified run; choose a new --out-dir.")
    shard_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    seen = 0
    manifest_records: list[dict[str, Any]] = []

    with torch.no_grad():
        progress = tqdm(loader, desc="save predictions", leave=True)
        for batch in progress:
            if config.max_examples is not None:
                remaining = int(config.max_examples) - seen
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

            for index, uid in enumerate(batch["uid"]):
                source_index = int(batch["source_index"][index])
                source_kind = str(batch["source_kind"][index])
                shard_path = shard_dir / f"{uid}_src{source_index:02d}_{source_kind}.npz"
                if shard_path.exists() and not config.overwrite:
                    skipped += 1
                    continue

                arrays = _artifact_arrays(batch, prediction, index, config.include_measured_trace)
                _save_npz_atomic(shard_path, **arrays)
                written += 1
                manifest_records.append(
                    {
                        "path": str(shard_path.relative_to(out_dir)),
                        "uid": str(uid),
                        "source_index": source_index,
                        "source_kind": source_kind,
                    }
                )

            seen += int(prediction.shape[0])
            progress.set_postfix(written=written, skipped=skipped)

    manifest = {
        "run_signature": signature,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_global_step": int(checkpoint.get("global_step", -1)),
        "data_dir": str(dataset_config.data_dir),
        "split": dataset_config.split,
        "source_kind": config.source_kind,
        "dataset_examples": len(dataset),
        "seen_examples": int(seen),
        "written": int(written),
        "skipped_existing": int(skipped),
        "artifact_count": len(list(shard_dir.glob("*.npz"))),
        "sampling": dataclass_to_json_dict(sampling_config),
        "include_measured_trace": bool(config.include_measured_trace),
        "records_written_this_run": manifest_records,
        "artifact_paths": [
            str(path.relative_to(out_dir))
            for path in sorted(shard_dir.glob("*.npz"))
        ],
    }
    _write_json_atomic(out_dir / "manifest.json", manifest)
    return manifest


def evaluate_prediction_artifacts(config: PredictionArtifactEvaluationConfig) -> dict[str, Any]:
    """Compute metrics from saved prediction artifacts without loading the model."""

    shard_paths = sorted((config.predictions_dir.resolve() / "shards").glob("*.npz"))
    if not shard_paths:
        raise FileNotFoundError("No saved prediction shards found in the requested directory.")
    if config.max_examples is not None:
        shard_paths = shard_paths[: max(0, int(config.max_examples))]

    metric_lists: dict[str, list[float]] = {
        "source_pulse_mae": [],
        "predicted_pulse_mae": [],
        "copra_hann_direct_final_pulse_mae": [],
        "copra_hann_guarded_final_pulse_mae": [],
        "copra_hann_best_final_pulse_mae": [],
        "source_trace_ssim": [],
        "predicted_trace_ssim": [],
        "target_trace_ssim": [],
        "copra_hann_direct_final_trace_ssim": [],
        "copra_hann_guarded_final_trace_ssim": [],
        "copra_hann_best_final_trace_ssim": [],
    }

    for path in tqdm(shard_paths, desc="artifact metrics", leave=True):
        with np.load(path, allow_pickle=False) as arrays:
            target = np.asarray(arrays["target_field"])
            source = np.asarray(arrays["source_field"])
            predicted = np.asarray(arrays["predicted_field"])
            source_mae = aligned_pulse_mae(target, source)
            predicted_mae = aligned_pulse_mae(target, predicted)
            metric_lists["source_pulse_mae"].append(source_mae)
            metric_lists["predicted_pulse_mae"].append(predicted_mae)
            _append_if_finite(metric_lists["source_trace_ssim"], float(arrays["source_trace_ssim"]))

            best_hann: tuple[float, float] | None = None
            for key, label in (
                ("copra_hann_direct_final_field", "direct"),
                ("copra_hann_guarded_final_field", "guarded"),
            ):
                if key not in arrays.files:
                    continue
                mae = aligned_pulse_mae(target, np.asarray(arrays[key]))
                ssim = float(arrays[f"copra_hann_{label}_final_trace_ssim"])
                metric_lists[f"copra_hann_{label}_final_pulse_mae"].append(mae)
                _append_if_finite(metric_lists[f"copra_hann_{label}_final_trace_ssim"], ssim)
                if best_hann is None or mae < best_hann[0]:
                    best_hann = (mae, ssim)

            if best_hann is not None:
                metric_lists["copra_hann_best_final_pulse_mae"].append(best_hann[0])
                _append_if_finite(metric_lists["copra_hann_best_final_trace_ssim"], best_hann[1])

            if config.include_trace_metrics:
                predicted_metrics = trace_objectives_for_field(
                    predicted,
                    measured_trace=np.asarray(arrays["measured_trace"]),
                    delay_axis=np.asarray(arrays["delay_axis"]),
                    freq_axis=np.asarray(arrays["freq_axis"]),
                    time_axis=np.asarray(arrays["time_axis"]),
                    frog_method=config.trace_frog_method,
                )
                metric_lists["predicted_trace_ssim"].append(float(predicted_metrics["trace_ssim"]))
                target_metrics = trace_objectives_for_field(
                    target,
                    measured_trace=np.asarray(arrays["measured_trace"]),
                    delay_axis=np.asarray(arrays["delay_axis"]),
                    freq_axis=np.asarray(arrays["freq_axis"]),
                    time_axis=np.asarray(arrays["time_axis"]),
                    frog_method=config.trace_frog_method,
                )
                metric_lists["target_trace_ssim"].append(float(target_metrics["trace_ssim"]))

    result = _artifact_result(config, shard_paths, metric_lists)
    if config.out_json is not None:
        _write_json_atomic(config.out_json.resolve(), result)
    return result


def _dataset_config_from_checkpoint(
    checkpoint: dict[str, Any],
    config: PredictionArtifactConfig,
) -> DatasetConfig:
    dataset_payload = dataset_settings(checkpoint, config.checkpoint, config.data_dir, config.split_file)
    dataset_payload["split"] = config.split
    dataset_payload["source_mode"] = "all"
    dataset_payload["source_kind_filter"] = config.source_kind
    dataset_payload["include_evaluation_fields"] = True
    return DatasetConfig(**dataset_payload)


def _make_loader(dataset: Any, batch_size: int, workers: int, device: Any) -> Any:
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(workers),
        pin_memory=device.type == "cuda",
        persistent_workers=int(workers) > 0,
    )


def _trim_batch(batch: dict[str, Any], count: int) -> dict[str, Any]:
    import torch

    result: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            result[key] = value[:count]
        elif isinstance(value, list):
            result[key] = value[:count]
        else:
            result[key] = value
    return result


def _artifact_arrays(
    batch: dict[str, Any],
    prediction: Any,
    index: int,
    include_measured_trace: bool,
) -> dict[str, Any]:
    arrays: dict[str, Any] = {
        "uid": np.asarray(str(batch["uid"][index])),
        "source_kind": np.asarray(str(batch["source_kind"][index])),
        "source_index": np.asarray(int(batch["source_index"][index]), dtype=np.int32),
        "target_field": channels_to_complex(batch["target"][index].detach().cpu().numpy()),
        "source_field": channels_to_complex(batch["source"][index].detach().cpu().numpy()),
        "predicted_field": channels_to_complex(prediction[index].detach().cpu().numpy()),
        "source_trace_ssim": np.asarray(float(batch["source_trace_ssim"][index]), dtype=np.float32),
        "source_trace_error": np.asarray(float(batch["source_trace_error"][index]), dtype=np.float32),
        "time_window": batch["time_window"][index].detach().cpu().numpy().astype(np.float32),
        "delay_axis": batch["delay_axis"][index].detach().cpu().numpy().astype(np.float64),
        "freq_axis": batch["freq_axis"][index].detach().cpu().numpy().astype(np.float64),
        "time_axis": batch["time_axis"][index].detach().cpu().numpy().astype(np.float64),
    }
    if include_measured_trace:
        arrays["measured_trace"] = (
            batch["measured_trace_raw"][index].detach().cpu().numpy().astype(np.float32)
        )
    for prefix, output_key in (
        ("copra_hann_direct_final", "copra_hann_direct_final_field"),
        ("copra_hann_guarded_final", "copra_hann_guarded_final_field"),
    ):
        if bool(batch[f"{prefix}_available"][index]):
            arrays[output_key] = channels_to_complex(batch[prefix][index].detach().cpu().numpy())
            arrays[f"{prefix}_trace_ssim"] = np.asarray(
                float(batch[f"{prefix}_trace_ssim"][index]),
                dtype=np.float32,
            )
            arrays[f"{prefix}_trace_error"] = np.asarray(
                float(batch[f"{prefix}_trace_error"][index]),
                dtype=np.float32,
            )
    return arrays


def _artifact_result(
    config: PredictionArtifactEvaluationConfig,
    shard_paths: list[Path],
    metric_lists: dict[str, list[float]],
) -> dict[str, Any]:
    flow_vs_source = lower_is_better_comparison(
        metric_lists["source_pulse_mae"],
        metric_lists["predicted_pulse_mae"],
    )
    return {
        "predictions_dir": str(config.predictions_dir.resolve()),
        "evaluated_examples": len(shard_paths),
        "trace_metrics": {
            "included": bool(config.include_trace_metrics),
            "frog_method": str(config.trace_frog_method),
        },
        **{name: stats(values) for name, values in metric_lists.items()},
        "flow_vs_source": flow_vs_source,
        "flow_vs_copra_hann_direct_final": lower_is_better_comparison(
            metric_lists["copra_hann_direct_final_pulse_mae"],
            metric_lists["predicted_pulse_mae"],
        ),
        "flow_vs_copra_hann_guarded_final": lower_is_better_comparison(
            metric_lists["copra_hann_guarded_final_pulse_mae"],
            metric_lists["predicted_pulse_mae"],
        ),
        "flow_vs_copra_hann_best_final": lower_is_better_comparison(
            metric_lists["copra_hann_best_final_pulse_mae"],
            metric_lists["predicted_pulse_mae"],
        ),
        "trace_ssim_flow_vs_source": higher_is_better_comparison(
            metric_lists["source_trace_ssim"],
            metric_lists["predicted_trace_ssim"],
        ),
        "trace_ssim_flow_vs_copra_hann_best_final": higher_is_better_comparison(
            metric_lists["copra_hann_best_final_trace_ssim"],
            metric_lists["predicted_trace_ssim"],
        ),
    }


def _append_if_finite(values: list[float], value: float) -> None:
    if np.isfinite(value):
        values.append(float(value))


def _save_npz_atomic(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
    np.savez_compressed(tmp_path, **arrays)
    os.replace(tmp_path, path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp_path, path)
