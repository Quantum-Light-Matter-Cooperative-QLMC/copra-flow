"""Dataset artifact generation for trace-conditioned flow matching."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .pulse_catalog import (
    PULSE_LABELS,
    PULSE_LIST,
    PulseSpec,
    make_pypret_pulse,
    pulse_like_to_spec,
    sample_pulse_specs,
    sanitize_uid,
)
from .pypret_bootstrap import package_root, repo_root, setup_local_pypret
from .retrieval import (
    ALPHA,
    GRID_DT,
    GRID_N,
    MAXITER,
    NOISE_LEVEL,
    PNP_ACCEPT_TOL,
    PNP_BACKTRACK_GAM,
    PNP_LAMBDA_INIT,
    PNP_LAMBDA_MIN,
    WL0,
    denoise_time_classic,
    field_from_spectrum,
    make_context,
    make_initial_guess,
    run_abhi_pnp,
    smooth_hann,
    spectrum_from_field,
    align_and_pulse_mae,
)
from .trace_metrics import MetricConfig, score_trace_objectives

setup_local_pypret()

from pypret.retrieval.step_retriever import COPRARetriever  # noqa: E402


DEFAULT_OUT_DIR = package_root() / "data" / "flow_matching_10k"
EPS = 1e-30
OBJECTIVE_METRIC_NAMES = (
    "trace_r",
    "ssim_distance",
    "p1_wls",
    "p4_amplitude",
    "p4_log",
    "p5_multiscale",
    "q1_marginal",
    "q2_ridge_geometry",
)


class ArtifactCOPRARetriever(COPRARetriever):
    """COPRA retriever that records source spectra along the trajectory."""

    method = None

    def __init__(self, pnps, snapshot_stride: int = 25, snapshot_iters=(), **kwargs):
        super().__init__(pnps, **kwargs)
        self.snapshot_stride = max(1, int(snapshot_stride))
        self.snapshot_iters = {max(0, int(value)) for value in snapshot_iters}
        self.trajectory_snapshots: list[dict[str, Any]] = []

    def _should_snapshot(self, iteration: int) -> bool:
        return iteration == 0 or iteration in self.snapshot_iters or iteration % self.snapshot_stride == 0

    def _snapshot(self, iteration: int, trace_error: float, spectrum, stage: str) -> None:
        state = self._retrieval_state
        self.trajectory_snapshots.append(
            {
                "iteration": int(iteration),
                "trace_error": float(trace_error),
                "stage": str(stage),
                "mode": str(getattr(state, "mode", "")),
                "spectrum": np.asarray(spectrum).copy(),
            }
        )

    def _retrieve(self):
        options = self.options
        log = self.log
        result = self._result
        state = self._retrieval_state
        spectrum = self.initial_guess.copy()
        trace_error = result.trace_error
        last_iteration = -1

        for iteration in range(options.maxiter):
            last_iteration = iteration
            if self.logging:
                if state.approximate_error:
                    trace_error = self.trace_error(spectrum, store=False)
                log.trace_error.append(trace_error)
            if self._should_snapshot(iteration):
                self._snapshot(iteration, trace_error, spectrum, stage="pre_step")

            trace_error, new_spectrum = self._retrieve_step(iteration, spectrum.copy())
            if trace_error < result.trace_error:
                result.trace_error = trace_error
                result.approximate_error = state.approximate_error
                result.spectrum[:] = spectrum
                state.steps_since_improvement = 0
            else:
                state.steps_since_improvement += 1
            spectrum[:] = new_spectrum
            if not state.running:
                break

        self._snapshot(last_iteration + 1, result.trace_error, result.spectrum, stage="best")
        return result.spectrum


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp_path.write_text(text)
    os.replace(tmp_path, path)


def write_json_atomic(path: Path, payload: object) -> None:
    write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True))


def save_npz_atomic(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
    np.savez_compressed(tmp_path, **arrays)
    os.replace(tmp_path, path)


def load_json(path: Path, default: object | None = None) -> object:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_int_hash(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def git_revision() -> str | None:
    try:
        top = subprocess.check_output(["git", "rev-parse", "--show-toplevel"],
                                      cwd=repo_root(), text=True, stderr=subprocess.DEVNULL).strip()
        if Path(top).resolve() != repo_root().resolve():
            return None  # Do not report an unrelated parent research repository.
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root(),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        return result.stdout.strip()
    except Exception:
        return None


def spec_stem(spec: PulseSpec) -> str:
    return sanitize_uid(spec.uid)


def shard_path(out_dir: Path, spec: PulseSpec) -> Path:
    return out_dir / "shards" / f"{spec_stem(spec)}.npz"


def metadata_path(out_dir: Path, spec: PulseSpec) -> Path:
    return out_dir / "metadata" / f"{spec_stem(spec)}.json"


def failure_path(out_dir: Path, spec: PulseSpec) -> Path:
    return out_dir / "failures" / f"{spec_stem(spec)}.json"


def pulse_family(spec: PulseSpec) -> str:
    if spec.family != "named":
        return spec.family
    name = str(spec.params.get("name", ""))
    if name.startswith("mog_phase"):
        return "random_mixture"
    if name.startswith("phase_jump") or "phase_step" in name:
        return "phase_jump"
    if name.startswith("gaussian_phase_sine"):
        return "sinusoidal_phase"
    if name.startswith("biexponential"):
        return "biexponential"
    if name.startswith("double_pulse"):
        return "double_pulse"
    if name.startswith("gaussian"):
        return "gaussian"
    return "other"


def parse_snapshot_iters(text: str | None) -> tuple[int, ...]:
    if not text:
        return ()
    values = []
    for token in text.split(","):
        token = token.strip()
        if token:
            values.append(max(0, int(token)))
    return tuple(sorted(set(values)))


def target_field(context) -> np.ndarray:
    return field_from_spectrum(context.ft, context.original_spectrum)


def rms_scale(field: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.abs(field) ** 2))) + EPS


def scale_energy_like(candidate: np.ndarray, target: np.ndarray) -> np.ndarray:
    candidate = np.asarray(candidate, dtype=np.complex128)
    num = float(np.sqrt(np.sum(np.abs(target) ** 2)))
    den = float(np.sqrt(np.sum(np.abs(candidate) ** 2))) + EPS
    return candidate * (num / den)


def smooth_field(field: np.ndarray, amp_frac: float, phase_frac: float) -> np.ndarray:
    amplitude = np.abs(field).astype(np.float64)
    phase = np.unwrap(np.angle(field).astype(np.float64))
    amp_smoothed = np.maximum(smooth_hann(amplitude, amp_frac), 0.0)
    phase_smoothed = np.angle(np.exp(1j * smooth_hann(phase, phase_frac)))
    return (amp_smoothed * np.exp(1j * phase_smoothed)).astype(np.complex128)


def smooth_noise(rng: np.random.RandomState, size: int, frac: float) -> np.ndarray:
    values = rng.normal(size=size)
    values = smooth_hann(values, frac)
    std = float(np.std(values)) + EPS
    return values / std


def run_copra_with_snapshots(
    pnps,
    measurement,
    initial_spectrum: np.ndarray,
    maxiter: int,
    alpha: float,
    snapshot_stride: int,
    snapshot_iters: tuple[int, ...],
) -> tuple[np.ndarray, float, ArtifactCOPRARetriever]:
    retriever = ArtifactCOPRARetriever(
        pnps,
        verbose=False,
        logging=True,
        maxiter=maxiter,
        alpha=alpha,
        snapshot_stride=snapshot_stride,
        snapshot_iters=snapshot_iters,
    )
    retriever.retrieve(measurement, initial_spectrum.copy())
    return retriever._result.spectrum.copy(), float(retriever._result.trace_error), retriever


def compute_trace_objectives(context, spectrum: np.ndarray) -> dict[str, float]:
    reconstructed = context.pnps.calculate(spectrum, context.parameter).astype(np.float64)
    metrics = score_trace_objectives(
        context.measurement.data.astype(np.float64),
        reconstructed,
        delay_axis=context.parameter,
        freq_axis=context.pnps.process_w,
        config=MetricConfig(),
    )
    return {key: float(value) for key, value in metrics.items()}


def trace_ssim_from_objectives(context, spectrum: np.ndarray) -> float:
    metrics = compute_trace_objectives(context, spectrum)
    return float(1.0 - metrics["ssim_distance"])


def denoise_spectrum_direct(context, spectrum: np.ndarray) -> np.ndarray:
    denoised_time = denoise_time_classic(context.ft.backward(spectrum))
    if context.time_window is not None and len(context.time_window) == len(denoised_time):
        denoised_time = denoised_time * context.time_window
    return context.ft.forward(denoised_time).copy()


def denoise_spectrum_posthoc(context, spectrum: np.ndarray) -> tuple[np.ndarray, float, float]:
    spectrum_denoised = denoise_spectrum_direct(context, spectrum)
    lam = float(PNP_LAMBDA_INIT)
    ssim_base = trace_ssim_from_objectives(context, spectrum)
    while lam >= float(PNP_LAMBDA_MIN):
        spectrum_try = (1.0 - lam) * spectrum + lam * spectrum_denoised
        ssim_try = trace_ssim_from_objectives(context, spectrum_try)
        if ssim_try >= ssim_base * (1.0 - float(PNP_ACCEPT_TOL)):
            metrics = compute_trace_objectives(context, spectrum_try)
            return spectrum_try.copy(), float(metrics["trace_r"]), float(1.0 - metrics["ssim_distance"])
        lam *= float(PNP_BACKTRACK_GAM)

    metrics = compute_trace_objectives(context, spectrum)
    return spectrum.copy(), float(metrics["trace_r"]), float(1.0 - metrics["ssim_distance"])


def build_source_example(
    *,
    context,
    true_field: np.ndarray,
    norm_scale: float,
    source_kind: str,
    source_detail: str,
    iteration: int,
    hard_negative: bool,
    source_weight: float,
    spectrum_raw: np.ndarray | None = None,
    field_raw: np.ndarray | None = None,
) -> dict[str, Any]:
    if field_raw is None:
        if spectrum_raw is None:
            raise ValueError("Either field_raw or spectrum_raw is required.")
        field_raw = field_from_spectrum(context.ft, spectrum_raw)
    else:
        field_raw = np.asarray(field_raw, dtype=np.complex128)
    if spectrum_raw is None:
        spectrum_raw = spectrum_from_field(context.ft, field_raw)

    mae_aligned, field_aligned = align_and_pulse_mae(true_field, field_raw)
    spectrum_aligned = spectrum_from_field(context.ft, field_aligned)
    objective_metrics = compute_trace_objectives(context, spectrum_raw)
    trace_error = float(objective_metrics["trace_r"])
    trace_ssim = float(1.0 - objective_metrics["ssim_distance"])

    record: dict[str, Any] = {
        "source_kind": str(source_kind),
        "source_detail": str(source_detail),
        "iteration": int(iteration),
        "hard_negative": bool(hard_negative),
        "trace_good_pulse_bad": False,
        "source_weight": float(source_weight),
        "mae_aligned": float(mae_aligned),
        "trace_error": float(trace_error),
        "trace_ssim": float(trace_ssim),
        "scale_mu": float(objective_metrics.get("scale_mu", 0.0)),
        "field_l2_norm_raw": float(np.sqrt(np.sum(np.abs(field_raw) ** 2))),
        "field_l2_norm_aligned": float(np.sqrt(np.sum(np.abs(field_aligned) ** 2))),
        "_source_field": (field_aligned / norm_scale).astype(np.complex64),
        "_source_field_raw": (field_raw / norm_scale).astype(np.complex64),
        "_source_spectrum_raw": np.asarray(spectrum_raw).astype(np.complex64),
        "_source_spectrum_aligned": np.asarray(spectrum_aligned).astype(np.complex64),
    }
    for name in OBJECTIVE_METRIC_NAMES:
        record[name] = float(objective_metrics.get(name, np.nan))
    return record


def classify_difficulty(mae: float, copra_maes: list[float]) -> str:
    if not copra_maes:
        return "unscaled"
    q10, q50, q90 = np.percentile(np.asarray(copra_maes, dtype=np.float64), [10, 50, 90])
    if mae <= q10:
        return "easy"
    if mae <= q50:
        return "medium"
    if mae <= q90:
        return "hard"
    return "extreme"


def mark_trace_good_pulse_bad(records: list[dict[str, Any]]) -> None:
    copra_indices = [
        idx
        for idx, record in enumerate(records)
        if record["source_kind"].startswith("copra")
        and np.isfinite(record["trace_error"])
        and np.isfinite(record["mae_aligned"])
    ]
    if not copra_indices:
        return
    trace_values = np.asarray([records[idx]["trace_error"] for idx in copra_indices], dtype=np.float64)
    mae_values = np.asarray([records[idx]["mae_aligned"] for idx in copra_indices], dtype=np.float64)
    trace_threshold = float(np.percentile(trace_values, 40))
    mae_threshold = float(np.percentile(mae_values, 65))
    selected = [
        idx
        for idx in copra_indices
        if records[idx]["trace_error"] <= trace_threshold and records[idx]["mae_aligned"] >= mae_threshold
    ]
    if not selected and len(copra_indices) >= 2:
        best_trace_indices = [
            idx for idx in copra_indices if records[idx]["trace_error"] <= float(np.median(trace_values))
        ]
        selected = sorted(best_trace_indices, key=lambda idx: records[idx]["mae_aligned"], reverse=True)[:1]
    for idx in selected:
        records[idx]["trace_good_pulse_bad"] = True
        records[idx]["hard_negative"] = True
        records[idx]["source_weight"] = max(float(records[idx]["source_weight"]), 2.0)
        if "trace_good_pulse_bad" not in records[idx]["source_detail"]:
            records[idx]["source_detail"] += ";trace_good_pulse_bad"


def perturb_over_smooth(field: np.ndarray, context, rng: np.random.RandomState):
    amp_frac = rng.uniform(0.12, 0.28)
    phase_frac = rng.uniform(0.16, 0.34)
    candidate = smooth_field(field, amp_frac=amp_frac, phase_frac=phase_frac)
    return "hard_over_smooth", f"amp_frac={amp_frac:.3f};phase_frac={phase_frac:.3f}", candidate


def perturb_amp_phase_jitter(field: np.ndarray, context, rng: np.random.RandomState):
    n = field.size
    amp_strength = rng.uniform(0.10, 0.35)
    phase_strength = rng.uniform(0.35, 1.40)
    amp_mod = np.exp(amp_strength * smooth_noise(rng, n, rng.uniform(0.04, 0.12)))
    phase_mod = phase_strength * smooth_noise(rng, n, rng.uniform(0.04, 0.16))
    candidate = np.abs(field) * amp_mod * np.exp(1j * (np.angle(field) + phase_mod))
    candidate = scale_energy_like(candidate, field)
    detail = f"amp_strength={amp_strength:.3f};phase_strength={phase_strength:.3f}"
    return "hard_amp_phase_jitter", detail, candidate


def perturb_phase_defect(field: np.ndarray, context, rng: np.random.RandomState):
    t = np.linspace(-1.0, 1.0, field.size)
    center = rng.uniform(-0.35, 0.35)
    width = rng.uniform(0.015, 0.08)
    jump = rng.choice([-1.0, 1.0]) * rng.uniform(0.45 * np.pi, 1.4 * np.pi)
    ripple = rng.uniform(0.15, 0.85) * np.sin(
        2.0 * np.pi * rng.uniform(1.0, 6.0) * t + rng.uniform(-np.pi, np.pi)
    )
    phase_defect = 0.5 * jump * (1.0 + np.tanh((t - center) / width)) + ripple
    candidate = np.abs(field) * np.exp(1j * (np.angle(field) + phase_defect))
    detail = f"center={center:.3f};width={width:.3f};jump={jump:.3f}"
    return "hard_phase_defect", detail, candidate


def perturb_satellite(field: np.ndarray, context, rng: np.random.RandomState):
    n = field.size
    shift = int(rng.choice([-1, 1]) * rng.randint(max(8, n // 18), max(9, n // 4)))
    amplitude = rng.uniform(0.06, 0.35)
    phase = rng.uniform(-np.pi, np.pi)
    satellite = np.roll(field, shift) * amplitude * np.exp(1j * phase)
    candidate = scale_energy_like(field + satellite, field)
    detail = f"shift={shift};amplitude={amplitude:.3f};phase={phase:.3f}"
    return "hard_satellite", detail, candidate


def perturb_spectral_hole(field: np.ndarray, context, rng: np.random.RandomState):
    spectrum = spectrum_from_field(context.ft, field)
    w_norm = context.ft.w / (np.max(np.abs(context.ft.w)) + EPS)
    center = rng.uniform(-0.45, 0.45)
    width = rng.uniform(0.025, 0.12)
    depth = rng.uniform(0.30, 0.90)
    notch = 1.0 - depth * np.exp(-0.5 * ((w_norm - center) / width) ** 2)
    edge_roll = 1.0 / (1.0 + np.exp((np.abs(w_norm) - rng.uniform(0.55, 0.92)) / 0.025))
    candidate = context.ft.backward(spectrum * np.maximum(notch * edge_roll, 0.0))
    candidate = scale_energy_like(candidate, field)
    detail = f"center={center:.3f};width={width:.3f};depth={depth:.3f}"
    return "hard_spectral_hole", detail, candidate


def perturb_spectral_phase(field: np.ndarray, context, rng: np.random.RandomState):
    spectrum = spectrum_from_field(context.ft, field)
    w_norm = context.ft.w / (np.max(np.abs(context.ft.w)) + EPS)
    gdd = rng.uniform(-3.0, 3.0)
    tod = rng.uniform(-2.0, 2.0)
    ripple = rng.uniform(0.2, 1.4) * np.sin(
        2.0 * np.pi * rng.uniform(1.0, 4.0) * w_norm + rng.uniform(-np.pi, np.pi)
    )
    phase = gdd * w_norm**2 + tod * w_norm**3 + ripple
    candidate = context.ft.backward(spectrum * np.exp(1j * phase))
    candidate = scale_energy_like(candidate, field)
    detail = f"gdd_norm={gdd:.3f};tod_norm={tod:.3f}"
    return "hard_spectral_phase", detail, candidate


def perturb_foreign_family(field: np.ndarray, context, rng: np.random.RandomState, spec: PulseSpec):
    choices = [name for name in PULSE_LIST if pulse_family(pulse_like_to_spec(name)) != pulse_family(spec)]
    foreign_name = str(rng.choice(choices if choices else PULSE_LIST))
    foreign = make_pypret_pulse(foreign_name, context.ft, wl0=WL0)
    candidate = np.asarray(foreign.field).astype(np.complex128)
    candidate = scale_energy_like(candidate, field)
    detail = f"foreign_pulse={foreign_name}"
    return "hard_foreign_family", detail, candidate


def perturb_copra_blend(
    field: np.ndarray,
    context,
    rng: np.random.RandomState,
    copra_fields: list[np.ndarray],
):
    if not copra_fields:
        return perturb_satellite(field, context, rng)
    reference = np.asarray(copra_fields[int(rng.randint(0, len(copra_fields)))], dtype=np.complex128)
    _, reference_aligned = align_and_pulse_mae(field, reference)
    blend = rng.uniform(0.35, 0.85)
    phase = np.exp(1j * rng.uniform(-0.35, 0.35))
    candidate = scale_energy_like((1.0 - blend) * field + blend * phase * reference_aligned, field)
    detail = f"blend={blend:.3f};reference_pool={len(copra_fields)}"
    return "hard_copra_gt_blend", detail, candidate


def generate_hard_negative_fields(
    *,
    context,
    spec: PulseSpec,
    true_field: np.ndarray,
    rng: np.random.RandomState,
    count: int,
    copra_fields: list[np.ndarray],
) -> list[tuple[str, str, np.ndarray]]:
    factories = (
        lambda: perturb_over_smooth(true_field, context, rng),
        lambda: perturb_amp_phase_jitter(true_field, context, rng),
        lambda: perturb_phase_defect(true_field, context, rng),
        lambda: perturb_satellite(true_field, context, rng),
        lambda: perturb_spectral_hole(true_field, context, rng),
        lambda: perturb_spectral_phase(true_field, context, rng),
        lambda: perturb_foreign_family(true_field, context, rng, spec),
        lambda: perturb_copra_blend(true_field, context, rng, copra_fields),
    )
    hard_negatives: list[tuple[str, str, np.ndarray]] = []
    factory_index = 0
    while len(hard_negatives) < max(0, int(count)):
        factory = factories[factory_index % len(factories)]
        factory_index += 1
        try:
            kind, detail, field = factory()
            field = scale_energy_like(np.asarray(field, dtype=np.complex128), true_field)
            hard_negatives.append((kind, detail, field))
        except Exception as exc:
            hard_negatives.append(
                (
                    "hard_generation_failure",
                    f"factory={factory_index - 1};error={type(exc).__name__}",
                    true_field.copy(),
                )
            )
    return hard_negatives


def unique_snapshots(snapshots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    unique = []
    for snapshot in snapshots:
        key = (int(snapshot["iteration"]), str(snapshot["stage"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(snapshot)
    return unique


def arrays_from_records(
    *,
    context,
    records: list[dict[str, Any]],
    clean_field: np.ndarray,
    norm_scale: float,
    runtime_s: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    spec = context.spec
    source_fields = np.stack([record["_source_field"] for record in records], axis=0)
    source_fields_raw = np.stack([record["_source_field_raw"] for record in records], axis=0)
    source_spectra_raw = np.stack([record["_source_spectrum_raw"] for record in records], axis=0)
    source_spectra_aligned = np.stack([record["_source_spectrum_aligned"] for record in records], axis=0)
    target = (clean_field / norm_scale).astype(np.complex64)
    meta_records = [
        {
            key: value
            for key, value in record.items()
            if not key.startswith("_") and isinstance(value, (str, int, float, bool))
        }
        for record in records
    ]

    spec_payload = spec.to_dict()
    metadata = {
        "spec": spec_payload,
        "pulse_uid": spec.uid,
        "family": pulse_family(spec),
        "source_count": int(len(records)),
        "runtime_s": float(runtime_s),
        "created_at": now_iso(),
        "package_version": __version__,
        "norm_scale": float(norm_scale),
        "grid_n": int(context.ft.N),
        "grid_dt": float(context.ft.dt),
        "noise": float(config["noise"]),
        "pulse_peak_scale": float(context.pulse_peak_scale),
        "trace_peak_scale": float(context.trace_peak_scale),
        "maxiter": int(config["maxiter"]),
        "snapshot_stride": int(config["snapshot_stride"]),
        "records": meta_records,
    }

    arrays = {
        "target_field": target,
        "target_spectrum": np.asarray(context.original_spectrum).astype(np.complex64),
        "source_fields": source_fields,
        "source_fields_raw": source_fields_raw,
        "source_spectra_raw": source_spectra_raw,
        "source_spectra_aligned": source_spectra_aligned,
        "measured_trace": context.measurement.data.astype(np.float32),
        "delay_axis": context.parameter.astype(np.float64),
        "freq_axis": context.pnps.process_w.astype(np.float64),
        "time_axis": context.ft.t.astype(np.float64),
        "time_window": context.time_window.astype(np.float32),
        "source_kind": np.asarray([record["source_kind"] for record in records], dtype="U64"),
        "source_detail": np.asarray([record["source_detail"] for record in records], dtype="U256"),
        "iteration": np.asarray([record["iteration"] for record in records], dtype=np.int32),
        "mae_aligned": np.asarray([record["mae_aligned"] for record in records], dtype=np.float32),
        "trace_error": np.asarray([record["trace_error"] for record in records], dtype=np.float32),
        "trace_ssim": np.asarray([record["trace_ssim"] for record in records], dtype=np.float32),
        "hard_negative": np.asarray([record["hard_negative"] for record in records], dtype=np.bool_),
        "trace_good_pulse_bad": np.asarray([record["trace_good_pulse_bad"] for record in records], dtype=np.bool_),
        "source_weight": np.asarray([record["source_weight"] for record in records], dtype=np.float32),
        "scale_mu": np.asarray([record["scale_mu"] for record in records], dtype=np.float64),
        "pulse_peak_scale": np.asarray(float(context.pulse_peak_scale), dtype=np.float64),
        "trace_peak_scale": np.asarray(float(context.trace_peak_scale), dtype=np.float64),
        "spec_json": np.asarray(json.dumps(spec_payload, sort_keys=True)),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    for name in OBJECTIVE_METRIC_NAMES:
        arrays[name] = np.asarray([record[name] for record in records], dtype=np.float32)
    return {"arrays": arrays, "metadata": metadata}


def generate_one_task(task: dict[str, Any]) -> dict[str, Any]:
    out_dir = Path(task["out_dir"])
    spec = PulseSpec.from_dict(task["spec"])
    force = bool(task["force"])
    shard = shard_path(out_dir, spec)
    meta_path = metadata_path(out_dir, spec)

    if shard.exists() and meta_path.exists() and not force:
        metadata = load_json(meta_path, default={}) or {}
        return {
            "task_id": spec.uid,
            "pulse_uid": spec.uid,
            "status": "skipped_existing",
            "shard": str(shard),
            "metadata": str(meta_path),
            "source_count": int(metadata.get("source_count", 0)),
            "runtime_s": 0.0,
        }

    start = time.time()
    seed = int(task["seed"]) + stable_int_hash(spec.uid) % 1_000_000
    np.random.seed(seed)
    random.seed(seed)
    rng = np.random.RandomState(seed + 17)

    try:
        context = make_context(
            spec,
            seed=seed,
            noise=float(task["noise"]),
            grid_n=int(task["grid_n"]),
            grid_dt=float(task["grid_dt"]),
        )
        clean_field = target_field(context)
        norm_scale = rms_scale(clean_field)
        initial_spectrum = make_initial_guess(context.ft)
        records: list[dict[str, Any]] = []

        if bool(task["include_initial_guess"]):
            records.append(
                build_source_example(
                    context=context,
                    true_field=clean_field,
                    norm_scale=norm_scale,
                    source_kind="copra_initial_guess",
                    source_detail="initial_guess_before_copra",
                    iteration=0,
                    hard_negative=True,
                    source_weight=0.75,
                    spectrum_raw=initial_spectrum,
                )
            )

        copra_spectrum, _, retriever = run_copra_with_snapshots(
            context.pnps,
            context.measurement,
            initial_spectrum,
            maxiter=int(task["maxiter"]),
            alpha=float(task["alpha"]),
            snapshot_stride=int(task["snapshot_stride"]),
            snapshot_iters=tuple(task["snapshot_iters"]),
        )

        copra_reference_fields: list[np.ndarray] = []
        for snapshot in unique_snapshots(retriever.trajectory_snapshots):
            spectrum = np.asarray(snapshot["spectrum"]).copy()
            copra_reference_fields.append(field_from_spectrum(context.ft, spectrum))
            records.append(
                build_source_example(
                    context=context,
                    true_field=clean_field,
                    norm_scale=norm_scale,
                    source_kind="copra_snapshot",
                    source_detail=f"stage={snapshot['stage']};mode={snapshot['mode']}",
                    iteration=int(snapshot["iteration"]),
                    hard_negative=False,
                    source_weight=1.0,
                    spectrum_raw=spectrum,
                )
            )

        if not any(record["source_kind"] == "copra_final" for record in records):
            records.append(
                build_source_example(
                    context=context,
                    true_field=clean_field,
                    norm_scale=norm_scale,
                    source_kind="copra_final",
                    source_detail="best_copra_result",
                    iteration=int(task["maxiter"]),
                    hard_negative=False,
                    source_weight=1.0,
                    spectrum_raw=copra_spectrum,
                )
            )
        copra_reference_fields.append(field_from_spectrum(context.ft, copra_spectrum))

        denoised_direct = denoise_spectrum_direct(context, copra_spectrum)
        direct_metrics = compute_trace_objectives(context, denoised_direct)
        records.append(
            build_source_example(
                context=context,
                true_field=clean_field,
                norm_scale=norm_scale,
                source_kind="copra_denoised_final",
                source_detail=(
                    "direct_hann_no_trace_guard;"
                    f"trace_ssim={1.0 - direct_metrics['ssim_distance']:.6f}"
                ),
                iteration=int(task["maxiter"]),
                hard_negative=False,
                source_weight=1.0,
                spectrum_raw=denoised_direct,
            )
        )

        denoised_spectrum, _, denoised_trace_ssim = denoise_spectrum_posthoc(context, copra_spectrum)
        records.append(
            build_source_example(
                context=context,
                true_field=clean_field,
                norm_scale=norm_scale,
                source_kind="copra_denoised_guarded_final",
                source_detail=f"posthoc_hann_trace_guard;trace_ssim={denoised_trace_ssim:.6f}",
                iteration=int(task["maxiter"]),
                hard_negative=False,
                source_weight=1.0,
                spectrum_raw=denoised_spectrum,
            )
        )

        denoise_limit = int(task["denoise_snapshot_limit"])
        if denoise_limit > 0:
            late_snapshots = sorted(
                unique_snapshots(retriever.trajectory_snapshots),
                key=lambda item: int(item["iteration"]),
            )[-denoise_limit:]
            for snapshot in late_snapshots:
                spectrum = np.asarray(snapshot["spectrum"]).copy()
                den_direct = denoise_spectrum_direct(context, spectrum)
                direct_metrics = compute_trace_objectives(context, den_direct)
                records.append(
                    build_source_example(
                        context=context,
                        true_field=clean_field,
                        norm_scale=norm_scale,
                        source_kind="copra_denoised_snapshot",
                        source_detail=(
                            f"source_iteration={int(snapshot['iteration'])};"
                            f"stage={snapshot['stage']};direct_hann_no_trace_guard;"
                            f"trace_ssim={1.0 - direct_metrics['ssim_distance']:.6f}"
                        ),
                        iteration=int(snapshot["iteration"]),
                        hard_negative=False,
                        source_weight=1.0,
                        spectrum_raw=den_direct,
                    )
                )

        if bool(task["include_pnp"]):
            pnp_spectrum, pnp_trace_error, _ = run_abhi_pnp(
                context.pnps,
                context.measurement,
                initial_spectrum,
                context.time_window,
                maxiter=int(task["maxiter"]),
                alpha=float(task["alpha"]),
                verbose=False,
            )
            records.append(
                build_source_example(
                    context=context,
                    true_field=clean_field,
                    norm_scale=norm_scale,
                    source_kind="pnp_copra_final",
                    source_detail=f"abhi_pnp;trace_error={pnp_trace_error:.6e}",
                    iteration=int(task["maxiter"]),
                    hard_negative=False,
                    source_weight=1.0,
                    spectrum_raw=pnp_spectrum,
                )
            )

        mark_trace_good_pulse_bad(records)
        copra_maes = [
            float(record["mae_aligned"])
            for record in records
            if record["source_kind"].startswith("copra")
        ]
        for index, (kind, detail, field) in enumerate(
            generate_hard_negative_fields(
                context=context,
                spec=spec,
                true_field=clean_field,
                rng=rng,
                count=int(task["hard_negatives_per_pulse"]),
                copra_fields=copra_reference_fields,
            )
        ):
            record = build_source_example(
                context=context,
                true_field=clean_field,
                norm_scale=norm_scale,
                source_kind=kind,
                source_detail=f"{detail};hard_index={index}",
                iteration=-1,
                hard_negative=True,
                source_weight=1.5,
                field_raw=field,
            )
            difficulty = classify_difficulty(float(record["mae_aligned"]), copra_maes)
            record["source_detail"] += f";difficulty={difficulty}"
            records.append(record)

        runtime_s = time.time() - start
        packed = arrays_from_records(
            context=context,
            records=records,
            clean_field=clean_field,
            norm_scale=norm_scale,
            runtime_s=runtime_s,
            config=task,
        )
        save_npz_atomic(shard, **packed["arrays"])
        write_json_atomic(meta_path, packed["metadata"])

        fail_path = failure_path(out_dir, spec)
        if fail_path.exists():
            fail_path.unlink()

        return {
            "task_id": spec.uid,
            "pulse_uid": spec.uid,
            "status": "completed",
            "shard": str(shard),
            "metadata": str(meta_path),
            "source_count": len(records),
            "runtime_s": runtime_s,
            "mae_min": float(np.min([record["mae_aligned"] for record in records])),
            "mae_max": float(np.max([record["mae_aligned"] for record in records])),
            "trace_error_min": float(np.min([record["trace_error"] for record in records])),
            "trace_error_max": float(np.max([record["trace_error"] for record in records])),
        }
    except Exception as exc:
        payload = {
            "task_id": spec.uid,
            "pulse_uid": spec.uid,
            "status": "failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "created_at": now_iso(),
        }
        write_json_atomic(failure_path(out_dir, spec), payload)
        payload["runtime_s"] = time.time() - start
        return payload


def manifest_path(out_dir: Path) -> Path:
    return out_dir / "manifest.json"


def progress_path(out_dir: Path) -> Path:
    return out_dir / "progress.json"


def write_manifest(path: Path, specs: list[PulseSpec], config: dict[str, Any]) -> None:
    payload = {
        "created_at": now_iso(),
        "package_version": __version__,
        "git_revision": git_revision(),
        "count": len(specs),
        "config": config,
        "specs": [spec.to_dict() for spec in specs],
    }
    write_json_atomic(path, payload)


def load_manifest(path: Path) -> list[PulseSpec]:
    payload = load_json(path, default={})
    specs_payload = payload.get("specs", []) if isinstance(payload, dict) else []
    return [PulseSpec.from_dict(spec) for spec in specs_payload]


def update_progress(out_dir: Path, result: dict[str, Any]) -> None:
    path = progress_path(out_dir)
    progress = load_json(path, default={}) or {}
    progress.setdefault("started_at", now_iso())
    progress.setdefault("completed", [])
    progress.setdefault("skipped", [])
    progress.setdefault("failed", [])
    progress.setdefault("results", {})
    task_id = str(result.get("task_id", "unknown"))
    status = str(result.get("status", "unknown"))
    if status == "completed" and task_id not in progress["completed"]:
        progress["completed"].append(task_id)
    elif status == "skipped_existing" and task_id not in progress["skipped"]:
        progress["skipped"].append(task_id)
    elif status == "failed" and task_id not in progress["failed"]:
        progress["failed"].append(task_id)
    progress["results"][task_id] = {
        "status": status,
        "source_count": int(result.get("source_count", 0) or 0),
        "runtime_s": float(result.get("runtime_s", 0.0) or 0.0),
        "shard": result.get("shard"),
        "metadata": result.get("metadata"),
        "error": result.get("error"),
        "updated_at": now_iso(),
    }
    write_json_atomic(path, progress)


def resolve_specs(args: argparse.Namespace) -> list[PulseSpec]:
    if args.manifest:
        specs = load_manifest(args.manifest)
        if not specs:
            raise SystemExit(f"No specs found in manifest: {args.manifest}")
        return specs

    if args.pulses:
        unknown = [pulse for pulse in args.pulses if pulse not in PULSE_LIST]
        if unknown:
            raise SystemExit(f"Unknown pulse(s): {unknown}. Available: {PULSE_LIST}")
        specs = []
        for pulse_name in args.pulses:
            base = pulse_like_to_spec(pulse_name)
            for repeat in range(int(args.repeats)):
                specs.append(
                    PulseSpec(
                        uid=sanitize_uid(f"{pulse_name}_rep{repeat:05d}"),
                        family="named",
                        params={"name": str(base.params["name"])},
                        seed=int(args.seed) + repeat,
                        label=PULSE_LABELS.get(pulse_name, pulse_name),
                    )
                )
        return specs

    return sample_pulse_specs(int(args.count), seed=int(args.seed), prefix=str(args.prefix))


def make_run_config(args: argparse.Namespace, specs: list[PulseSpec]) -> dict[str, Any]:
    return {
        "out_dir": str(args.out_dir),
        "spec_count": len(specs),
        "count": int(getattr(args, "count", len(specs))),
        "pulses": list(args.pulses or []),
        "repeats": int(args.repeats),
        "grid_n": int(args.grid_n),
        "grid_dt": float(args.grid_dt),
        "maxiter": int(args.maxiter),
        "snapshot_stride": int(args.snapshot_stride),
        "snapshot_iters": list(parse_snapshot_iters(args.snapshot_iters)),
        "noise": float(args.noise),
        "alpha": float(args.alpha),
        "seed": int(args.seed),
        "hard_negatives_per_pulse": int(args.hard_negatives_per_pulse),
        "denoise_snapshot_limit": int(args.denoise_snapshot_limit),
        "include_initial_guess": bool(args.include_initial_guess),
        "include_pnp": bool(args.include_pnp),
        "workers": int(args.workers),
        "force": bool(args.force),
        "git_revision": git_revision(),
        "created_at": now_iso(),
        "package_version": __version__,
        "generator_sha256": generator_fingerprint(),
    }


def generator_fingerprint() -> str:
    """Identify generator source even for an uncommitted or wheel installation."""
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for name in ("artifacts.py", "pulse_catalog.py", "retrieval.py", "trace_metrics.py"):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    pypret_root = setup_local_pypret()
    for path in sorted(pypret_root.rglob("*.py")):
        digest.update(str(path.relative_to(pypret_root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def make_tasks(config: dict[str, Any], specs: list[PulseSpec]) -> list[dict[str, Any]]:
    tasks = []
    for spec in specs:
        task = dict(config)
        task["spec"] = spec.to_dict()
        tasks.append(task)
    return tasks


def run_tasks(tasks: list[dict[str, Any]], out_dir: Path, workers: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if workers <= 1:
        for index, task in enumerate(tasks, start=1):
            result = generate_one_task(task)
            update_progress(out_dir, result)
            results.append(result)
            print(f"[{index}/{len(tasks)}] {result['task_id']} {result['status']}", flush=True)
        return results

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_task = {executor.submit(generate_one_task, task): task for task in tasks}
        for index, future in enumerate(concurrent.futures.as_completed(future_to_task), start=1):
            result = future.result()
            update_progress(out_dir, result)
            results.append(result)
            print(f"[{index}/{len(tasks)}] {result['task_id']} {result['status']}", flush=True)
    return results


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [result for result in results if result.get("status") == "completed"]
    skipped = [result for result in results if result.get("status") == "skipped_existing"]
    failed = [result for result in results if result.get("status") == "failed"]
    return {
        "completed": len(completed),
        "skipped": len(skipped),
        "failed": len(failed),
        "source_count": int(sum(int(result.get("source_count", 0) or 0) for result in completed + skipped)),
        "runtime_s": float(sum(float(result.get("runtime_s", 0.0) or 0.0) for result in results)),
    }


def add_common_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--manifest", type=Path, help="Use an existing manifest instead of sampling specs.")
    parser.add_argument("--count", type=int, default=10_000, help="Number of stochastic pulses to sample.")
    parser.add_argument("--prefix", default="pulse", help="UID prefix for sampled pulse specs.")
    parser.add_argument("--pulses", nargs="+", help="Use named benchmark pulses instead of stochastic specs.")
    parser.add_argument("--repeats", type=int, default=1, help="Repeats for --pulses mode.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-n", type=int, default=GRID_N)
    parser.add_argument("--grid-dt", type=float, default=GRID_DT)
    parser.add_argument("--noise", type=float, default=NOISE_LEVEL)
    parser.add_argument("--maxiter", type=int, default=MAXITER)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--snapshot-stride", type=int, default=25)
    parser.add_argument("--snapshot-iters", default="")
    parser.add_argument("--hard-negatives-per-pulse", type=int, default=8)
    parser.add_argument("--denoise-snapshot-limit", type=int, default=2)
    parser.add_argument("--include-initial-guess", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-pnp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=False)


def run_command(args: argparse.Namespace) -> int:
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = resolve_specs(args)
    config = make_run_config(args, specs)
    if not specs or len({spec.uid for spec in specs}) != len(specs):
        raise ValueError("A generation run needs a nonempty set of unique pulse UIDs.")
    unrelated = {path.stem for path in (out_dir / "shards").glob("*.npz")} - {spec_stem(spec) for spec in specs}
    if unrelated:
        raise ValueError("Output contains shards outside this manifest; choose a new output directory.")
    existing = load_json(out_dir / "run_config.json", default={}) or {}
    scientific_keys = ("grid_n", "grid_dt", "maxiter", "noise", "alpha", "seed",
                       "hard_negatives_per_pulse", "denoise_snapshot_limit", "snapshot_stride",
                       "snapshot_iters", "include_initial_guess", "include_pnp", "generator_sha256")
    changed = [key for key in scientific_keys if key in existing and existing[key] != config[key]]
    if changed and not args.force:
        raise ValueError(f"Existing dataset uses different settings ({', '.join(changed)}). "
                         "Choose a new output directory or explicitly regenerate with --force.")
    previous_specs = load_manifest(manifest_path(out_dir)) if manifest_path(out_dir).exists() else []
    by_uid = {spec.uid: spec.to_dict() for spec in previous_specs}
    conflicting = [spec.uid for spec in specs if spec.uid in by_uid and by_uid[spec.uid] != spec.to_dict()]
    if conflicting:
        raise ValueError("An existing pulse UID has different parameters; use a new dataset directory.")
    write_manifest(manifest_path(out_dir), specs, config)
    write_json_atomic(out_dir / "run_config.json", config)
    tasks = make_tasks(config, specs)
    print(
        f"[run] specs={len(specs)} workers={args.workers} maxiter={args.maxiter} out={out_dir}",
        flush=True,
    )
    results = run_tasks(tasks, out_dir, workers=max(1, int(args.workers)))
    summary = summarize_results(results)
    write_json_atomic(out_dir / "last_run_summary.json", {"summary": summary, "results": results})
    print(f"[done] {summary}", flush=True)
    return 0 if summary["failed"] == 0 else 1


def manifest_command(args: argparse.Namespace) -> int:
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = resolve_specs(args)
    if not specs or len({spec.uid for spec in specs}) != len(specs):
        raise ValueError("A manifest needs unique, nonempty pulse specifications.")
    existing_shards = {path.stem for path in (out_dir / "shards").glob("*.npz")}
    if existing_shards:
        previous = {spec.uid: spec.to_dict() for spec in load_manifest(manifest_path(out_dir))}
        candidate = {spec.uid: spec.to_dict() for spec in specs}
        if any(uid not in candidate or candidate[uid] != previous.get(uid) for uid in existing_shards):
            raise ValueError("Manifest would change existing pulse specifications; choose a new output directory.")
    config = make_run_config(args, specs)
    write_manifest(manifest_path(out_dir), specs, config)
    print(f"[manifest] wrote {len(specs)} specs to {manifest_path(out_dir)}", flush=True)
    return 0


def status_command(args: argparse.Namespace) -> int:
    out_dir = args.out_dir.resolve()
    progress = load_json(progress_path(out_dir), default={}) or {}
    completed = len(progress.get("completed", []))
    skipped = len(progress.get("skipped", []))
    failed = len(progress.get("failed", []))
    shards = len(list((out_dir / "shards").glob("*.npz"))) if (out_dir / "shards").exists() else 0
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "completed": completed,
                "skipped": skipped,
                "failed": failed,
                "shards": shards,
            },
            indent=2,
        )
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate COPRA artifacts for flow-matching training.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Generate dataset shards.")
    add_common_generation_args(run_parser)

    manifest_parser = subparsers.add_parser("manifest", help="Write a pulse manifest without running COPRA.")
    add_common_generation_args(manifest_parser)

    status_parser = subparsers.add_parser("status", help="Summarize output/progress for a dataset directory.")
    status_parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "run":
        return run_command(args)
    if args.command == "manifest":
        return manifest_command(args)
    if args.command == "status":
        return status_command(args)
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
