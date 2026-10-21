"""Publication-style comparisons of saved pulse prediction artifacts."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from tqdm.auto import tqdm

from .metrics import align_field_to_target


FIELD_KEYS = {
    "target": "target_field",
    "copra": "source_field",
    "hann": "copra_hann_direct_final_field",
    "flow": "predicted_field",
}
LABELS = {
    "target": "Original target",
    "copra": "COPRA",
    "hann": "COPRA + Hann",
    "flow": "COPRA + flow",
}
STYLES = {
    "target": {"color": "#222222", "linestyle": "-", "linewidth": 1.8},
    "copra": {"color": "#3f6690", "linestyle": "--", "linewidth": 1.4},
    "hann": {"color": "#bd7b2b", "linestyle": "-.", "linewidth": 1.5},
    "flow": {"color": "#26734d", "linestyle": "-", "linewidth": 1.7},
}


@dataclass(frozen=True)
class PulseVisualizationConfig:
    """Controls visualization of saved Flow Matching prediction artifacts."""

    predictions_dir: Path
    out_dir: Path
    selection_json: Path | None = None
    pulse_uids: tuple[str, ...] = ()
    max_examples: int | None = None
    include_trace_diagnostics: bool = True
    trace_frog_method: str = "tg"
    phase_support_fraction: float = 0.05
    dpi: int = 180
    num_workers: int = 1
    overwrite: bool = False


def visualize_prediction_artifacts(config: PulseVisualizationConfig) -> dict[str, Any]:
    """Create one comparison figure per selected prediction artifact."""

    predictions_dir = config.predictions_dir.resolve()
    shard_paths = sorted((predictions_dir / "shards").glob("*.npz"))
    if not shard_paths:
        raise FileNotFoundError(f"No prediction shards found in {predictions_dir / 'shards'}")

    paths_by_uid = _index_shards_by_uid(shard_paths)
    requested_uids = _requested_uids(config)
    if requested_uids:
        missing = [uid for uid in requested_uids if uid not in paths_by_uid]
        if missing:
            preview = ", ".join(missing[:5])
            raise FileNotFoundError(
                f"Missing {len(missing)} requested prediction artifact(s): {preview}"
            )
        selected_paths = [paths_by_uid[uid] for uid in requested_uids]
    else:
        selected_paths = [paths_by_uid[uid] for uid in sorted(paths_by_uid)]
    if config.max_examples is not None:
        selected_paths = selected_paths[: max(0, int(config.max_examples))]

    out_dir = config.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = min(max(1, int(config.num_workers)), max(1, len(selected_paths)))
    if workers == 1:
        records = _visualize_serial(selected_paths, out_dir=out_dir, config=config)
    else:
        records = _visualize_parallel(
            selected_paths,
            out_dir=out_dir,
            config=config,
            workers=workers,
        )

    manifest = {
        "predictions_dir": str(predictions_dir),
        "out_dir": str(out_dir),
        "selection_json": (
            None if config.selection_json is None else str(config.selection_json.resolve())
        ),
        "include_trace_diagnostics": bool(config.include_trace_diagnostics),
        "trace_frog_method": str(config.trace_frog_method),
        "num_workers": workers,
        "artifact_count": len(records),
        "records": records,
    }
    _write_json_atomic(out_dir / "manifest.json", manifest)
    return manifest


def _visualize_serial(
    paths: list[Path],
    *,
    out_dir: Path,
    config: PulseVisualizationConfig,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    progress = tqdm(paths, desc="pulse visualizations", leave=True)
    for path in progress:
        record = visualize_prediction_artifact(path, out_dir=out_dir, config=config)
        records.append(record)
        progress.set_postfix(uid=record["pulse_uid"])
    return records


def _visualize_parallel(
    paths: list[Path],
    *,
    out_dir: Path,
    config: PulseVisualizationConfig,
    workers: int,
) -> list[dict[str, Any]]:
    order = {path.resolve(): index for index, path in enumerate(paths)}
    records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                visualize_prediction_artifact,
                path,
                out_dir=out_dir,
                config=config,
            ): path
            for path in paths
        }
        progress = tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"pulse visualizations ({workers} workers)",
            leave=True,
        )
        for future in progress:
            record = future.result()
            records.append(record)
            progress.set_postfix(uid=record["pulse_uid"])
    records.sort(key=lambda record: order[Path(record["artifact"]).resolve()])
    return records


def visualize_prediction_artifact(
    artifact_path: Path,
    *,
    out_dir: Path,
    config: PulseVisualizationConfig,
) -> dict[str, Any]:
    """Render one saved target/COPRA/Hann/flow comparison."""

    arrays = _load_artifact(artifact_path)
    uid = str(np.asarray(arrays["uid"]).item())
    png_path = out_dir / f"{uid}.png"
    pdf_path = out_dir / f"{uid}.pdf"
    if png_path.exists() and pdf_path.exists() and not config.overwrite:
        return {
            "pulse_uid": uid,
            "artifact": str(artifact_path.resolve()),
            "png": str(png_path),
            "pdf": str(pdf_path),
            "status": "skipped_existing",
        }

    target = np.asarray(arrays[FIELD_KEYS["target"]], dtype=np.complex128)
    fields: dict[str, np.ndarray] = {"target": target}
    pulse_mae: dict[str, float] = {"target": 0.0}
    for name in ("copra", "hann", "flow"):
        pulse_mae[name], fields[name] = align_field_to_target(
            target,
            np.asarray(arrays[FIELD_KEYS[name]], dtype=np.complex128),
        )

    trace_diagnostics: dict[str, dict[str, Any]] = {}
    if config.include_trace_diagnostics:
        trace_diagnostics = _compute_trace_diagnostics(
            fields,
            measured_trace=np.asarray(arrays["measured_trace"], dtype=np.float64),
            delay_axis=np.asarray(arrays["delay_axis"], dtype=np.float64),
            freq_axis=np.asarray(arrays["freq_axis"], dtype=np.float64),
            time_axis=np.asarray(arrays["time_axis"], dtype=np.float64),
            frog_method=config.trace_frog_method,
        )
    else:
        trace_diagnostics = _stored_trace_diagnostics(arrays)

    figure = _build_figure(
        uid=uid,
        arrays=arrays,
        fields=fields,
        pulse_mae=pulse_mae,
        trace_diagnostics=trace_diagnostics,
        phase_support_fraction=float(config.phase_support_fraction),
    )
    _save_figure_atomic(figure, png_path, dpi=int(config.dpi))
    _save_figure_atomic(figure, pdf_path, dpi=int(config.dpi))
    plt.close(figure)

    return {
        "pulse_uid": uid,
        "artifact": str(artifact_path.resolve()),
        "png": str(png_path),
        "pdf": str(pdf_path),
        "status": "written",
        "pulse_mae": {name: float(value) for name, value in pulse_mae.items()},
        "trace_ssim": {
            name: _finite_or_none(values.get("trace_ssim"))
            for name, values in trace_diagnostics.items()
        },
    }


def _load_artifact(path: Path) -> dict[str, np.ndarray]:
    required = {
        "uid",
        "target_field",
        "source_field",
        "copra_hann_direct_final_field",
        "predicted_field",
        "time_axis",
        "delay_axis",
        "freq_axis",
        "measured_trace",
    }
    with np.load(path, allow_pickle=False) as handle:
        missing = sorted(required.difference(handle.files))
        if missing:
            raise KeyError(f"{path} is missing required visualization arrays: {missing}")
        return {key: np.asarray(handle[key]) for key in handle.files}


def _index_shards_by_uid(paths: list[Path]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as handle:
            if "uid" not in handle.files:
                raise KeyError(f"{path} is missing uid.")
            uid = str(np.asarray(handle["uid"]).item())
        if uid in result:
            raise ValueError(f"Multiple prediction artifacts found for pulse UID {uid!r}.")
        result[uid] = path
    return result


def _requested_uids(config: PulseVisualizationConfig) -> list[str]:
    result = [str(uid) for uid in config.pulse_uids]
    if config.selection_json is None:
        return _deduplicate(result)
    payload = json.loads(config.selection_json.resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("selection_json must contain a JSON list.")
    for item in payload:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, Mapping) and "pulse_uid" in item:
            result.append(str(item["pulse_uid"]))
        else:
            raise ValueError(
                "Each selection_json item must be a pulse UID string or contain pulse_uid."
            )
    return _deduplicate(result)


def _deduplicate(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _compute_trace_diagnostics(
    fields: Mapping[str, np.ndarray],
    *,
    measured_trace: np.ndarray,
    delay_axis: np.ndarray,
    freq_axis: np.ndarray,
    time_axis: np.ndarray,
    frog_method: str,
) -> dict[str, dict[str, Any]]:
    from pnp_copra.pypret_bootstrap import setup_local_pypret
    from pnp_copra.trace_metrics import MetricConfig, score_trace_objectives

    setup_local_pypret()
    from pypret import FourierTransform, PNPS, Pulse  # noqa: WPS433

    if time_axis.size < 2:
        raise ValueError("time_axis must contain at least two samples.")
    ft = FourierTransform(int(time_axis.size), dt=float(np.mean(np.diff(time_axis))))
    pulse = Pulse(ft, 800e-9)
    pnps = PNPS(pulse, "frog", str(frog_method))
    measured_norm = _unit_max(measured_trace)
    result: dict[str, dict[str, Any]] = {}
    for name, field in fields.items():
        reconstructed = np.asarray(
            pnps.calculate(ft.forward(np.asarray(field, dtype=np.complex128)), delay_axis),
            dtype=np.float64,
        )
        metrics = score_trace_objectives(
            measured_trace,
            reconstructed,
            delay_axis=delay_axis,
            freq_axis=freq_axis,
            config=MetricConfig(),
        )
        residual = np.mean(np.abs(measured_norm - _unit_max(reconstructed)), axis=1)
        result[name] = {
            "trace_ssim": float(1.0 - metrics["ssim_distance"]),
            "trace_error": float(metrics["trace_r"]),
            "residual_profile": residual,
        }
    return result


def _stored_trace_diagnostics(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    stored = {
        "copra": "source_trace_ssim",
        "hann": "copra_hann_direct_final_trace_ssim",
    }
    for name in FIELD_KEYS:
        key = stored.get(name)
        value = float(np.asarray(arrays[key]).item()) if key in arrays else float("nan")
        result[name] = {"trace_ssim": value}
    return result


def _build_figure(
    *,
    uid: str,
    arrays: Mapping[str, np.ndarray],
    fields: Mapping[str, np.ndarray],
    pulse_mae: Mapping[str, float],
    trace_diagnostics: Mapping[str, Mapping[str, Any]],
    phase_support_fraction: float,
) -> plt.Figure:
    time_fs = np.asarray(arrays["time_axis"], dtype=np.float64) * 1e15
    delay_fs = np.asarray(arrays["delay_axis"], dtype=np.float64) * 1e15
    freq_phz = np.asarray(arrays["freq_axis"], dtype=np.float64) / (2.0 * np.pi) * 1e-15
    measured_trace = np.asarray(arrays["measured_trace"], dtype=np.float64)

    target_scale = max(float(np.max(np.abs(fields["target"]))), 1e-30)
    amplitudes = {name: np.abs(field) / target_scale for name, field in fields.items()}
    support = {
        name: amplitude >= phase_support_fraction * max(float(np.max(amplitude)), 1e-30)
        for name, amplitude in amplitudes.items()
    }
    phases = {
        name: np.where(support[name], np.angle(field), np.nan)
        for name, field in fields.items()
    }
    amplitude_residual = {
        name: np.abs(amplitudes[name] - amplitudes["target"])
        for name in ("copra", "hann", "flow")
    }
    phase_residual: dict[str, np.ndarray] = {}
    for name in ("copra", "hann", "flow"):
        valid = support["target"] & support[name]
        residual = np.full(fields["target"].shape, np.nan, dtype=np.float64)
        residual[valid] = np.abs(
            np.angle(np.exp(1j * (np.angle(fields[name][valid]) - np.angle(fields["target"][valid]))))
        )
        phase_residual[name] = residual

    figure = plt.figure(figsize=(22, 12))
    grid = gridspec.GridSpec(
        3,
        4,
        figure=figure,
        hspace=0.48,
        wspace=0.42,
        height_ratios=[1.0, 0.82, 0.82],
    )

    ax_trace = figure.add_subplot(grid[0, 0])
    image = ax_trace.pcolormesh(
        delay_fs,
        freq_phz,
        _unit_max(measured_trace).T,
        shading="auto",
        cmap="inferno",
    )
    ax_trace.set(xlabel="Delay (fs)", ylabel="Frequency (PHz)", title="Input TG-FROG")
    figure.colorbar(image, ax=ax_trace, shrink=0.72, label="Normalized intensity")

    ax_target = figure.add_subplot(grid[0, 1])
    ax_target.plot(time_fs, amplitudes["target"], **STYLES["target"])
    ax_target.set(xlabel="Time (fs)", ylabel="Normalized amplitude", title="Original Pulse")
    ax_target_phase = ax_target.twinx()
    ax_target_phase.plot(time_fs, phases["target"], color="#9c3d8f", linewidth=1.1)
    ax_target_phase.set_ylabel("Phase (rad)", color="#9c3d8f")
    ax_target_phase.set_ylim(-np.pi, np.pi)

    ax_amp = figure.add_subplot(grid[0, 2])
    _plot_methods(ax_amp, time_fs, amplitudes)
    ax_amp.set(
        xlabel="Time (fs)",
        ylabel="Normalized amplitude",
        title="Retrieved Amplitudes",
    )
    ax_amp.legend(fontsize=8)

    ax_summary = figure.add_subplot(grid[0, 3])
    ax_summary.axis("off")
    ax_summary.set_title("Summary")
    ax_summary.text(
        0.02,
        0.98,
        _summary_text(uid, pulse_mae, trace_diagnostics),
        transform=ax_summary.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
    )

    ax_true_phase = figure.add_subplot(grid[1, 0])
    ax_true_phase.plot(time_fs, phases["target"], **STYLES["target"])
    ax_true_phase.set(
        xlabel="Time (fs)", ylabel="Phase (rad)", title="Original Phase", ylim=(-np.pi, np.pi)
    )

    ax_phase = figure.add_subplot(grid[1, 1])
    _plot_methods(ax_phase, time_fs, phases)
    ax_phase.set(
        xlabel="Time (fs)",
        ylabel="Phase (rad)",
        title="Retrieved Phases",
        ylim=(-np.pi, np.pi),
    )
    ax_phase.legend(fontsize=8)

    ax_amp_residual = figure.add_subplot(grid[1, 2:4])
    _plot_candidates(ax_amp_residual, time_fs, amplitude_residual)
    ax_amp_residual.set(
        xlabel="Time (fs)", ylabel="Amplitude absolute error", title="Amplitude Residuals"
    )
    ax_amp_residual.legend(fontsize=8)

    ax_phase_residual = figure.add_subplot(grid[2, 0:2])
    _plot_candidates(ax_phase_residual, time_fs, phase_residual)
    ax_phase_residual.set(
        xlabel="Time (fs)", ylabel="Angular error (rad)", title="Phase Residuals"
    )
    ax_phase_residual.legend(fontsize=8)

    ax_trace_residual = figure.add_subplot(grid[2, 2:4])
    profiles = {
        name: values.get("residual_profile")
        for name, values in trace_diagnostics.items()
        if values.get("residual_profile") is not None
    }
    if profiles:
        for name in ("target", "copra", "hann", "flow"):
            if name in profiles:
                ax_trace_residual.semilogy(
                    delay_fs,
                    np.maximum(np.asarray(profiles[name]), 1e-12),
                    label=LABELS[name],
                    **STYLES[name],
                )
        ax_trace_residual.legend(fontsize=8)
    else:
        ax_trace_residual.text(
            0.5,
            0.5,
            "Trace residuals skipped",
            transform=ax_trace_residual.transAxes,
            ha="center",
            va="center",
        )
    ax_trace_residual.set(
        xlabel="Delay (fs)",
        ylabel="Mean absolute trace error",
        title="TG-FROG Trace Residuals",
    )

    for axis in figure.axes:
        if axis is not ax_summary:
            axis.grid(alpha=0.18, linewidth=0.6)
    figure.suptitle(
        f"TG-FROG Pulse Retrieval: Original vs COPRA vs Hann vs Flow - {uid}",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    return figure


def _plot_methods(axis: Any, x: np.ndarray, values: Mapping[str, np.ndarray]) -> None:
    for name in ("target", "copra", "hann", "flow"):
        axis.plot(x, values[name], label=LABELS[name], **STYLES[name])


def _plot_candidates(axis: Any, x: np.ndarray, values: Mapping[str, np.ndarray]) -> None:
    for name in ("copra", "hann", "flow"):
        axis.plot(x, values[name], label=LABELS[name], **STYLES[name])


def _summary_text(
    uid: str,
    pulse_mae: Mapping[str, float],
    trace_diagnostics: Mapping[str, Mapping[str, Any]],
) -> str:
    lines = [uid, ""]
    for name in ("target", "copra", "hann", "flow"):
        ssim = _finite_or_none(trace_diagnostics.get(name, {}).get("trace_ssim"))
        ssim_text = "n/a" if ssim is None else f"{ssim:.5f}"
        lines.extend(
            [
                LABELS[name],
                f"  Pulse MAE: {pulse_mae[name]:.5f}",
                f"  Trace SSIM: {ssim_text}",
                "",
            ]
        )
    return "\n".join(lines)


def _unit_max(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array / max(float(np.max(np.abs(array))), 1e-30)


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _save_figure_atomic(figure: plt.Figure, path: Path, *, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.stem}.tmp-{os.getpid()}{path.suffix}")
    figure.savefig(tmp_path, dpi=dpi, bbox_inches="tight")
    os.replace(tmp_path, path)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)
