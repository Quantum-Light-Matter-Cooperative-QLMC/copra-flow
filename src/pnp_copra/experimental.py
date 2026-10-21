"""Final-report TG-FROG preprocessing and target-free retrieval helpers.

The background and carrier conventions are preserved from the final comparison
scripts. Frequency coordinates are relative angular frequency, in rad/s.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

from .pypret_bootstrap import setup_local_pypret
from .trace_metrics import scaled_reconstruction, ssim_global, trace_nrmse_loss
from .checkpoints import load_checkpoint

GRID_N = 256
GRID_DT_S = 2.5e-15
SPEED_OF_LIGHT_M_S = 299_792_458.0
FLOAT_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?$")


@dataclass(frozen=True)
class GridSpec:
    delay_axis_s: np.ndarray
    time_axis_s: np.ndarray
    relative_frequency_axis_rad_s: np.ndarray
    carrier_frequency_rad_s: float

    @classmethod
    def from_checkpoint_or_default(cls, checkpoint: Path | None = None) -> "GridSpec":
        from pypret import FourierTransform, PNPS, Pulse
        n = GRID_N
        if checkpoint is not None:
            configs = load_checkpoint(checkpoint)["configs"]
            n = int(configs["model"]["field_length"])
            dt = float(configs.get("grid", {}).get("dt_s", GRID_DT_S))
            if not np.isclose(dt, GRID_DT_S, rtol=1e-7, atol=1e-22):
                raise ValueError("Physical inference supports the report's 2.5 fs grid; this checkpoint uses a different spacing.")
        ft = FourierTransform(n, dt=GRID_DT_S)
        pnps = PNPS(Pulse(ft, 800e-9), "frog", "tg")
        carrier = 2 * np.pi * SPEED_OF_LIGHT_M_S / 800e-9
        return cls(ft.t.copy(), ft.t.copy(), np.asarray(pnps.process_w - carrier), carrier)


def score_experimental_trace(measured: np.ndarray, reconstructed: np.ndarray,
                             *, delay_axis: np.ndarray, freq_axis: np.ndarray) -> dict[str, float]:
    del delay_axis, freq_axis
    scaled, scale = scaled_reconstruction(measured, reconstructed)
    return {"scale_mu": float(scale), "trace_ssim": float(ssim_global(measured, scaled)),
            "trace_nrmse": float(trace_nrmse_loss(measured, reconstructed))}


def _experimental_context(item: Mapping[str, Any]) -> tuple[Any, Any, Any, np.ndarray]:
    from pypret import FourierTransform, PNPS, Pulse
    time_axis = np.asarray(item["time_axis"], dtype=np.float64)
    ft = FourierTransform(time_axis.size, dt=GRID_DT_S)
    pnps = PNPS(Pulse(ft, float(np.asarray(item["wl0"]).item())), "frog", "tg")
    parameter = np.asarray(item["delay_axis"], dtype=np.float64)
    pnps.calculate(np.zeros(time_axis.size, dtype=np.complex128), parameter)
    measurement = pnps.trace
    measurement.data = np.asarray(item["trace_linear"], dtype=np.float64).copy()
    return ft, pnps, measurement, parameter

@dataclass
class ParsedTrace:
    path: Path
    wavelength_nm: np.ndarray
    delay_fs: np.ndarray
    intensity: np.ndarray
    source_hash: str
    numerical_hash: str

    @property
    def native_wavelength_bins(self) -> int:
        return int(self.wavelength_nm.size)

    @property
    def native_delay_bins(self) -> int:
        return int(self.delay_fs.size)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_float(text: str) -> bool:
    return bool(FLOAT_RE.fullmatch(text.strip()))


def _canonical_numeric_hash(wavelength_nm: np.ndarray, delay_fs: np.ndarray, intensity: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in (wavelength_nm, delay_fs, intensity):
        values = np.ascontiguousarray(np.asarray(array, dtype="<f8"))
        digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def parse_linshan_trace(path: Path) -> ParsedTrace:
    """Read one axis-embedded numerical measurement."""

    with path.open("r", errors="replace", newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    if not rows or len(rows[0]) < 3:
        raise ValueError("file does not contain a tabular axis header")
    if not all(_is_float(value) for value in rows[0]):
        raise ValueError("first row is not numerical")
    width = len(rows[0])
    if any(len(row) != width for row in rows[1:]):
        raise ValueError("ragged tabular data")
    try:
        numeric = np.asarray([[float(value) for value in row] for row in rows], dtype=np.float64)
    except ValueError as exc:
        raise ValueError("non-numerical table contents") from exc
    if numeric.shape[0] < 3 or numeric.shape[1] < 3:
        raise ValueError("measurement table is too small")
    wavelength = numeric[1:, 0]
    delay = numeric[0, 1:]
    intensity = numeric[1:, 1:]
    if not (np.all(np.isfinite(wavelength)) and np.all(np.isfinite(delay))):
        raise ValueError("axis contains non-finite values")
    if not np.all(np.isfinite(intensity)):
        raise ValueError("intensity contains non-finite values")
    if wavelength.size < 2 or delay.size < 2:
        raise ValueError("axis contains fewer than two samples")
    if np.any(np.diff(wavelength) == 0.0) or np.any(np.diff(delay) == 0.0):
        raise ValueError("axis contains duplicate coordinates")
    # The source files are normally increasing, but orienting here makes the
    # convention explicit and prevents an accidental transpose/reversal.
    if wavelength[0] > wavelength[-1]:
        wavelength = wavelength[::-1]
        intensity = intensity[::-1, :]
    if delay[0] > delay[-1]:
        delay = delay[::-1]
        intensity = intensity[:, ::-1]
    if np.any(wavelength <= 0) or np.any(np.diff(wavelength) <= 0) or np.any(np.diff(delay) <= 0):
        raise ValueError("Wavelengths must be positive; axes must be strictly monotone.")
    source_hash = _sha256(path)
    numerical_hash = _canonical_numeric_hash(wavelength, delay, intensity)
    return ParsedTrace(path, wavelength, delay, intensity, source_hash, numerical_hash)


def _carrier_frequency(wavelength_m: np.ndarray, cleaned: np.ndarray) -> float:
    angular = 2.0 * np.pi * SPEED_OF_LIGHT_M_S / wavelength_m
    marginal = np.sum(np.maximum(cleaned, 0.0), axis=1)
    if not np.any(marginal > 0.0):
        return float(np.median(angular))
    # The spectral centroid is less brittle than selecting a single noisy bin.
    return float(np.sum(angular * marginal) / np.sum(marginal))


def _interp_downsample_axis(
    source_axis: np.ndarray,
    source_trace: np.ndarray,
    target_axis: np.ndarray,
    *,
    axis: int,
    allow_zero_pad: bool = True,
    upsample_tolerance: float = 1.05,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Resample one axis without inventing resolution.

    Linear interpolation is used only where the target grid is no finer than
    the source grid.  Values outside the detector window are zero-padded; this
    retains the fixed model coordinates without turning a narrow measurement
    into a stretched 256-bin crop.
    """

    source_axis = np.asarray(source_axis, dtype=np.float64)
    target_axis = np.asarray(target_axis, dtype=np.float64)
    source_trace = np.asarray(source_trace, dtype=np.float64)
    source_step = float(np.median(np.diff(source_axis)))
    target_step = float(np.median(np.diff(target_axis)))
    if source_step <= 0.0 or target_step <= 0.0:
        raise ValueError("resampling axes must be strictly increasing")
    if target_step < source_step / float(upsample_tolerance):
        raise ValueError(
            f"target axis would upsample ({target_step:g} < {source_step:g})"
        )
    if axis == 0:
        result = np.empty((target_axis.size, source_trace.shape[1]), dtype=np.float64)
        for col in range(source_trace.shape[1]):
            result[:, col] = _resample_flux_conserving(
                source_axis,
                source_trace[:, col],
                target_axis,
            )
    elif axis == 1:
        result = np.empty((source_trace.shape[0], target_axis.size), dtype=np.float64)
        for row in range(source_trace.shape[0]):
            result[row, :] = _resample_flux_conserving(
                source_axis,
                source_trace[row, :],
                target_axis,
            )
    else:
        raise ValueError("axis must be 0 or 1")
    inside = (target_axis >= source_axis[0]) & (target_axis <= source_axis[-1])
    if not allow_zero_pad and not np.all(inside):
        raise ValueError("target axis is not covered by the source axis")
    return result, {
        "source_step": source_step,
        "target_step": target_step,
        "downsample_factor": source_step / target_step,
        "target_bins_inside_source": int(np.count_nonzero(inside)),
        "target_bins_zero_padded": int(np.count_nonzero(~inside)),
    }


def _resample_flux_conserving(
    source_axis: np.ndarray,
    source_values: np.ndarray,
    target_axis: np.ndarray,
) -> np.ndarray:
    """Flux-conserving 1D resampling for a monotone coordinate."""

    source_axis = np.asarray(source_axis, dtype=np.float64)
    values = np.asarray(source_values, dtype=np.float64)
    target_axis = np.asarray(target_axis, dtype=np.float64)
    if source_axis.size < 2:
        return np.full(target_axis.shape, values[0] if values.size else 0.0)
    source_edges = np.empty(source_axis.size + 1, dtype=np.float64)
    source_edges[1:-1] = 0.5 * (source_axis[:-1] + source_axis[1:])
    source_edges[0] = source_axis[0] - 0.5 * (source_axis[1] - source_axis[0])
    source_edges[-1] = source_axis[-1] + 0.5 * (source_axis[-1] - source_axis[-2])
    target_edges = np.empty(target_axis.size + 1, dtype=np.float64)
    if target_axis.size > 1:
        target_edges[1:-1] = 0.5 * (target_axis[:-1] + target_axis[1:])
        target_edges[0] = target_axis[0] - 0.5 * (target_axis[1] - target_axis[0])
        target_edges[-1] = target_axis[-1] + 0.5 * (target_axis[-1] - target_axis[-2])
    else:
        target_edges[:] = [target_axis[0] - 0.5, target_axis[0] + 0.5]
    cumulative = np.concatenate(([0.0], np.cumsum(np.maximum(values, 0.0) * np.diff(source_edges))))
    cumulative_edges = np.interp(target_edges, source_edges, cumulative, left=0.0, right=float(cumulative[-1]))
    widths = np.diff(target_edges)
    return np.maximum(np.diff(cumulative_edges) / np.maximum(widths, 1e-30), 0.0)


def _align_field_to_reference(reference: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    reference = np.asarray(reference, dtype=np.complex128)
    candidate = np.asarray(candidate, dtype=np.complex128)
    def align_once(value: np.ndarray) -> tuple[float, np.ndarray, int, complex]:
        corr = np.fft.ifft(np.fft.fft(np.abs(reference)) * np.conj(np.fft.fft(np.abs(value))))
        shift = int(np.argmax(np.real(corr)))
        shifted = np.roll(value, shift)
        scale = np.vdot(shifted, reference) / (np.vdot(shifted, shifted) + 1e-30)
        aligned = scale * shifted
        return float(np.mean(np.abs(reference - aligned))), aligned, shift, complex(scale)
    direct = align_once(candidate)
    reverse = align_once(np.conj(candidate[::-1]))
    chosen = reverse if reverse[0] < direct[0] else direct
    return chosen[1], {"branch": "conjugate_reversed" if chosen is reverse else "direct", "circular_shift": chosen[2], "complex_scale_real": chosen[3].real, "complex_scale_imag": chosen[3].imag, "alignment_mae": chosen[0]}


def _make_copra_initial_guess(ft: Any, wl0: float, seed: int) -> np.ndarray:
    from pypret import Pulse

    rng_state = np.random.get_state()
    np.random.seed(int(seed))
    pulse = Pulse(ft, wl0)
    t_fs = ft.t * 1e15
    center = float(np.random.uniform(-80.0, 80.0))
    width = float(np.random.uniform(25.0, 70.0))
    second = float(np.random.uniform(0.4, 1.0))
    field = np.exp(-0.5 * ((t_fs - center) / width) ** 2) + second * np.exp(-0.5 * ((t_fs + center) / width) ** 2)
    pulse.field = field.astype(np.complex128)
    guess = pulse.spectrum.copy()
    np.random.set_state(rng_state)
    return guess


def run_experimental_copra(
    item: Mapping[str, Any],
    *,
    seeds: Sequence[int] = (11, 29, 47),
    maxiter: int = 400,
    alpha: float = 0.25,
) -> dict[str, Any]:
    """Run deterministic COPRA initializations and retain the best trace fit."""

    from .retrieval import field_from_spectrum, run_copra

    ft, pnps, measurement, parameter = _experimental_context(item)
    best: dict[str, Any] | None = None
    attempts: list[dict[str, Any]] = []
    wl0 = float(np.asarray(item["wl0"]).item())
    for seed in seeds:
        np.random.seed(int(seed))
        random.seed(int(seed))
        guess = _make_copra_initial_guess(ft, wl0, int(seed))
        spectrum, _, retriever = run_copra(pnps, measurement, guess, maxiter=maxiter, alpha=alpha, verbose=False)
        reconstructed = np.asarray(pnps.calculate(spectrum, parameter), dtype=np.float64)
        metrics = score_experimental_trace(item["trace_linear"], reconstructed, delay_axis=parameter, freq_axis=item["freq_axis"])
        result = retriever.result(full=False)
        attempt = {"seed": int(seed), **metrics, "iterations": int(len(getattr(getattr(retriever, "log", None), "trace_error", []))), "spectrum": spectrum, "field": field_from_spectrum(ft, spectrum, wl0=wl0), "reconstructed_trace": reconstructed}
        attempts.append({key: value for key, value in attempt.items() if key not in {"spectrum", "field", "reconstructed_trace"}})
        if best is None or float(metrics["trace_nrmse"]) < float(best["metrics"]["trace_nrmse"]):
            best = {"seed": int(seed), "metrics": metrics, "spectrum": spectrum, "field": attempt["field"], "reconstructed_trace": reconstructed, "iterations": attempt["iterations"], "attempts": attempts}
    if best is None:
        raise RuntimeError("COPRA produced no solution")
    best["attempts"] = attempts
    return best


def _center_and_phase_normalize(field: np.ndarray) -> tuple[np.ndarray, float, int]:
    field = np.asarray(field, dtype=np.complex128)
    intensity = np.abs(field) ** 2
    if float(np.sum(intensity)) <= 0.0:
        return field.copy(), 1.0, 0
    positions = np.arange(field.size, dtype=np.float64)
    centroid = float(np.sum(positions * intensity) / np.sum(intensity))
    shift = int(round(field.size // 2 - centroid))
    centered = np.roll(field, shift)
    peak_index = int(np.argmax(np.abs(centered)))
    centered = centered * np.exp(-1j * np.angle(centered[peak_index]))
    rms = float(np.sqrt(np.mean(np.abs(centered) ** 2)))
    return centered / (rms + 1e-30), rms, shift


def _flow_branches(field: np.ndarray) -> list[tuple[str, np.ndarray, float, int]]:
    direct, rms, shift = _center_and_phase_normalize(field)
    reverse = np.conj(direct[::-1])
    return [("direct", direct, rms, shift), ("conjugate_reversed", reverse, rms, shift)]


def run_experimental_flow(
    item: Mapping[str, Any],
    copra_field: np.ndarray,
    checkpoint: Path,
    *,
    sampling_steps: int = 64,
    sampling_method: str = "euler",
    device: str = "cpu",
) -> dict[str, Any]:
    """Run Flow from direct and conjugate-reversed COPRA branches."""

    import torch

    from .flow_matching.complex import channels_to_complex, complex_to_channels, preprocess_trace
    from .flow_matching.config import SamplingConfig
    from .flow_matching.sampling import sample
    from .flow_matching.trainer import load_model_from_checkpoint, resolve_device

    checkpoint = Path(checkpoint).resolve()
    resolved_device = resolve_device(device)
    model = load_model_from_checkpoint(checkpoint, device=str(resolved_device))
    trace_config = load_checkpoint(checkpoint).get("configs", {}).get("dataset", {})
    trace = preprocess_trace(item["trace_linear"], transform=str(trace_config.get("trace_transform", "sqrt")), normalize=bool(trace_config.get("normalize_trace", True)))
    trace_tensor = torch.from_numpy(trace[None]).to(resolved_device)
    ft, pnps, measurement, parameter = _experimental_context(item)
    candidates: list[dict[str, Any]] = []
    with torch.no_grad():
        for branch, source_field, rms, shift in _flow_branches(copra_field):
            source_tensor = torch.from_numpy(complex_to_channels(source_field)[None]).to(resolved_device)
            output = sample(model, source_tensor, trace_tensor, SamplingConfig(steps=sampling_steps, method=sampling_method))
            if isinstance(output, tuple):
                output = output[0]
            prediction = channels_to_complex(output[0].detach().cpu().numpy())
            if branch == "conjugate_reversed":
                prediction = np.conj(prediction[::-1])
            prediction = prediction * rms
            reconstructed = np.asarray(pnps.calculate(ft.forward(prediction), parameter), dtype=np.float64)
            metrics = score_experimental_trace(item["trace_linear"], reconstructed, delay_axis=parameter, freq_axis=item["freq_axis"])
            candidates.append({"branch": branch, "field": prediction, "reconstructed_trace": reconstructed, "metrics": metrics, "center_shift": shift, "rms": rms})
    return min(candidates, key=lambda value: float(value["metrics"]["trace_nrmse"]))


EDGE_FRACTION = 0.18

def _smooth_rows(values: np.ndarray) -> np.ndarray:
    """Suppress row outliers and retain smooth wavelength-dependent structure."""

    values = np.asarray(values, dtype=np.float64)
    filtered = median_filter(values, size=9, mode="nearest")
    window = min(21, values.size if values.size % 2 else values.size - 1)
    if window < 5:
        return filtered
    return savgol_filter(filtered, window_length=window, polyorder=2, mode="interp")


def _far_delay_background(parsed: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a wavelength-dependent linear baseline from both far-delay edges."""

    intensity = np.asarray(parsed.intensity, dtype=np.float64)
    delay_fs = np.asarray(parsed.delay_fs, dtype=np.float64)
    n_edge = max(2, int(round(EDGE_FRACTION * delay_fs.size)))
    left_raw = np.median(intensity[:, :n_edge], axis=1)
    right_raw = np.median(intensity[:, -n_edge:], axis=1)
    left = _smooth_rows(left_raw)
    right = _smooth_rows(right_raw)
    tau_left = float(np.median(delay_fs[:n_edge]))
    tau_right = float(np.median(delay_fs[-n_edge:]))
    slope = (right - left) / (tau_right - tau_left)
    background = left[:, None] + slope[:, None] * (delay_fs[None, :] - tau_left)
    corrected = np.maximum(intensity - background, 0.0)

    edge_mask = np.zeros_like(intensity, dtype=bool)
    edge_mask[:, :n_edge] = True
    edge_mask[:, -n_edge:] = True
    before = intensity[edge_mask]
    after_unclipped = (intensity - background)[edge_mask]
    diagnostics = {
        "edge_fraction": EDGE_FRACTION,
        "edge_columns_per_side": n_edge,
        "left_edge_delay_range_fs": [float(delay_fs[0]), float(delay_fs[n_edge - 1])],
        "right_edge_delay_range_fs": [float(delay_fs[-n_edge]), float(delay_fs[-1])],
        "left_raw": left_raw,
        "right_raw": right_raw,
        "left_fit": left,
        "right_fit": right,
        "slope": slope,
        "background": background,
        "edge_rms_before": float(np.sqrt(np.mean(np.square(before)))),
        "edge_rms_after_unclipped": float(np.sqrt(np.mean(np.square(after_unclipped)))),
        "edge_median_before": float(np.median(before)),
        "edge_median_after_unclipped": float(np.median(after_unclipped)),
    }
    return corrected, diagnostics


def _prior_comparison_carrier(parsed: Any) -> tuple[float, float]:
    """Reproduce the carrier convention used by the prior denoising PDF."""

    intensity = np.asarray(parsed.intensity, dtype=np.float64)
    height, width = intensity.shape
    band_y = max(1, int(round(0.05 * height)))
    band_x = max(1, int(round(0.05 * width)))
    border = np.concatenate(
        (
            intensity[:band_y, :].ravel(),
            intensity[-band_y:, :].ravel(),
            intensity[:, :band_x].ravel(),
            intensity[:, -band_x:].ravel(),
        )
    )
    border_median = float(np.median(border))
    carrier_trace = np.maximum(intensity - border_median, 0.0)
    carrier = _carrier_frequency(np.asarray(parsed.wavelength_nm) * 1e-9, carrier_trace)
    return carrier, border_median


def _main_lobe_fwhm(x: np.ndarray, y: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    peak_index = int(np.argmax(y))
    half = 0.5 * float(y[peak_index])
    left = peak_index
    right = peak_index
    while left > 0 and y[left - 1] >= half:
        left -= 1
    while right < y.size - 1 and y[right + 1] >= half:
        right += 1

    def crossing(i0: int, i1: int) -> float:
        if i0 == i1 or y[i1] == y[i0]:
            return float(x[i0])
        return float(x[i0] + (half - y[i0]) * (x[i1] - x[i0]) / (y[i1] - y[i0]))

    xl = float(x[0]) if left == 0 else crossing(left - 1, left)
    xr = float(x[-1]) if right == y.size - 1 else crossing(right, right + 1)
    return max(0.0, xr - xl)



def prepare_native(path: Path, grid: GridSpec | None = None) -> dict[str, Any]:
    """Apply the exact final-report background and carrier conventions."""
    parsed = parse_linshan_trace(Path(path))
    grid = grid or GridSpec.from_checkpoint_or_default()
    n = grid.time_axis_s.size
    if min(parsed.intensity.shape) < n:
        raise ValueError(f"Native trace needs at least {n} samples on each axis; upsampling is unsupported.")
    corrected, diagnostics = _far_delay_background(parsed)
    carrier, border_median = _prior_comparison_carrier(parsed)
    angular = 2 * np.pi * SPEED_OF_LIGHT_M_S / (parsed.wavelength_nm * 1e-9)
    order = np.argsort(angular)
    frequency_resampled, frequency_meta = _interp_downsample_axis(
        angular[order] - carrier, corrected[order], grid.relative_frequency_axis_rad_s, axis=0)
    trace, delay_meta = _interp_downsample_axis(
        parsed.delay_fs * 1e-15, frequency_resampled.T, grid.delay_axis_s, axis=0)
    trace = np.maximum(trace, 0)
    peak = float(trace.max())
    if not np.isfinite(peak) or peak <= 0:
        raise ValueError("Background-corrected trace contains no positive signal.")
    trace /= peak
    return {
        "trace_linear": trace, "measured_trace": trace,
        "delay_axis": grid.delay_axis_s, "time_axis": grid.time_axis_s,
        "freq_axis": grid.relative_frequency_axis_rad_s,
        "wl0": np.asarray(2 * np.pi * SPEED_OF_LIGHT_M_S / carrier),
        "native_trace": parsed.intensity, "background": diagnostics["background"],
        "corrected_native_trace": corrected,
        "native_wavelength_nm": parsed.wavelength_nm, "native_delay_fs": parsed.delay_fs,
        "metadata_json": np.asarray(json.dumps({
            "preprocessing": "report_far_delay_v1", "edge_fraction": EDGE_FRACTION,
            "median_filter_rows": 9, "savgol_rows": 21, "savgol_order": 2,
            "carrier_estimation": "5% border-median corrected spectral centroid",
            "carrier_border_median": border_median,
            "source_filename": Path(path).name, "source_sha256": parsed.source_hash,
            "native_shape": list(parsed.intensity.shape), "normalization_peak": peak,
            "frequency_resampling": frequency_meta, "delay_resampling": delay_meta,
            "frequency_reference": "relative to experimental carrier", "frequency_units": "rad/s",
        }, sort_keys=True)),
    }


def compare_report_summary(summary: dict, expected: dict, *, traces: int, seeds: list[int]) -> dict:
    """Compare rounded paper metrics without changing their definitions."""
    if expected["traces"] != traces or expected["seeds"] != seeds:
        raise ValueError("Reference metrics use a different trace count or seed list.")
    checks = {f"{method}.{key}": {"expected": value, "observed": summary[method][key],
              "absolute_error": abs(summary[method][key]-value),
              "passed": abs(summary[method][key]-value) <= 1e-5}
              for method, values in expected["summary"].items() for key, value in values.items()}
    return {"tolerance": 1e-5, "checks": checks,
            "status": "passed" if all(v["passed"] for v in checks.values()) else "failed"}


def benchmark_main(argv: list[str] | None = None) -> int:
    """Run all paired seeds and persist fields and metrics, including plot selection."""
    from .inference import load_measurement, write_json, save_npz
    from .retrieval import run_copra, run_abhi_pnp, field_from_spectrum, estimate_time_window_from_measured_trace

    parser = argparse.ArgumentParser(description=benchmark_main.__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="JSON records with uid and path (relative to manifest).")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 29, 47])
    parser.add_argument("--maxiter", type=int, default=400)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--reference-results", type=Path,
                        help="Optional rounded report metrics to check after the benchmark.")
    parser.add_argument("--strict-reference", action="store_true",
                        help="Fail if the historical reference comparison exceeds tolerance.")
    args = parser.parse_args(argv)
    payload = json.loads(args.manifest.read_text())
    records = payload["records"]
    if not records or len({r["uid"] for r in records}) != len(records):
        raise ValueError("Benchmark manifest must contain unique nonempty trace records.")
    rows = []
    primary_seeds = {}
    for record in records:
        uid = record["uid"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", uid):
            raise ValueError(f"Invalid trace UID: {uid!r}")
        path = Path(record["path"])
        if not path.is_absolute():
            path = args.manifest.parent / path
        if path.suffix.lower() == ".txt" and record.get("source_sha256"):
            if _sha256(path) != record["source_sha256"]:
                raise ValueError(f"Source checksum does not match the frozen measurement for {uid}.")
        print(f"Preparing {uid}", flush=True)
        item = load_measurement(path, args.checkpoint)
        ft, _, _, _ = _experimental_context(item)
        window = estimate_time_window_from_measured_trace(item["trace_linear"], item["delay_axis"], ft.t)
        per_seed = []
        for seed in args.seeds:
            print(f"{uid}: paired seed {seed}", flush=True)
            methods = {}
            for name in ("copra", "pnp"):
                ft, pnps, measurement, parameter = _experimental_context(item)
                wl0 = float(item["wl0"])
                initial = _make_copra_initial_guess(ft, wl0, seed)
                np.random.seed(seed)
                random.seed(seed)
                if name == "copra":
                    spectrum, _, _ = run_copra(pnps, measurement, initial,
                                               maxiter=args.maxiter, alpha=args.alpha, verbose=False)
                else:
                    spectrum, _, _ = run_abhi_pnp(pnps, measurement, initial, window,
                                                  maxiter=args.maxiter, alpha=args.alpha, verbose=False)
                methods[name] = {"field": field_from_spectrum(ft, spectrum, wl0=wl0),
                                 "reconstructed_trace": np.asarray(pnps.calculate(spectrum, parameter))}
            methods["flow"] = run_experimental_flow(item, methods["copra"]["field"], args.checkpoint,
                                                     sampling_steps=args.steps, device=args.device)
            seed_rows = []
            for name, result in methods.items():
                field, _, _ = _center_and_phase_normalize(result["field"])
                if name != "copra":
                    # Match the report's display/roughness alignment to paired COPRA.
                    field, _ = _align_field_to_reference(methods["copra"]["field"], result["field"])
                    field, _, _ = _center_and_phase_normalize(field)
                amp = np.abs(field)
                amp /= max(float(amp.max()), 1e-30)
                metrics = score_experimental_trace(item["trace_linear"], result["reconstructed_trace"],
                                                   delay_axis=item["delay_axis"], freq_axis=item["freq_axis"])
                row = {"uid": uid, "seed": seed, "method": name, **metrics,
                       "roughness": float(np.sqrt(np.mean(np.diff(amp, n=2)**2))),
                       "fwhm_fs": _main_lobe_fwhm(item["time_axis"]*1e15, amp**2)}
                rows.append(row)
                seed_rows.append(row)
                save_npz(args.out_dir / "fields" / f"{uid}_{seed}_{name}.npz",
                         {"field": result["field"], "reconstructed_trace": result["reconstructed_trace"],
                          "measured_trace": item["trace_linear"], "time_axis": item["time_axis"],
                          "delay_axis": item["delay_axis"], "freq_axis": item["freq_axis"], "wl0": item["wl0"]})
            per_seed.append((sum(r["trace_nrmse"] for r in seed_rows), seed))
        primary_seeds[uid] = min(per_seed)[1]
    summary = {name: {key: float(np.mean([r[key] for r in rows if r["method"] == name]))
                      for key in ("trace_nrmse", "trace_ssim", "roughness", "fwhm_fs")}
               for name in ("copra", "pnp", "flow")}
    write_json(args.out_dir / "metrics.json", {"records": rows, "summary": summary,
               "primary_figure_seeds": primary_seeds, "seeds": args.seeds,
               "checkpoint_sha256": _sha256(args.checkpoint), "steps": args.steps,
               "maxiter": args.maxiter, "alpha": args.alpha,
               "warning": "Roughness is smoothness, not ground-truth accuracy."})
    if args.reference_results:
        expected = json.loads(args.reference_results.read_text())
        comparison = compare_report_summary(summary, expected, traces=len(records), seeds=args.seeds)
        write_json(args.out_dir / "comparison.json", comparison)
        if args.strict_reference and comparison["status"] != "passed":
            raise AssertionError("Experimental metrics differ from the report; inspect comparison.json.")
        print(f"Historical reference comparison: {comparison['status']}; see {args.out_dir / 'comparison.json'}", flush=True)
    print(json.dumps(summary, indent=2))
    return 0
