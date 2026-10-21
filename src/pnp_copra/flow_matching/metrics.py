"""Metrics used by flow-matching evaluation and prediction artifacts."""

from __future__ import annotations

import re
from typing import Any

import numpy as np


HANN_AMP_FRAC = 0.08
HANN_PHASE_FRAC = 0.10
RELATIVE_DENOMINATOR_MIN = 1e-8
PULSE_UID_RE = re.compile(r"^pulse_\d+_(?P<pulse_type>.+)_[0-9a-f]{8}$")


def best_circular_shift_by_magnitude(field_true: np.ndarray, field_est: np.ndarray) -> int:
    correlation = np.fft.ifft(
        np.fft.fft(np.abs(field_true)) * np.conj(np.fft.fft(np.abs(field_est)))
    )
    return int(np.argmax(np.real(correlation)))


def aligned_pulse_mae(field_true: np.ndarray, field_est: np.ndarray) -> float:
    mae, _ = align_field_to_target(field_true, field_est)
    return mae


def align_field_to_target(
    field_true: np.ndarray,
    field_est: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Align a field to a target over the standard FROG ambiguities."""

    target = np.asarray(field_true)
    estimate = np.asarray(field_est)
    if target.ndim != 1 or estimate.ndim != 1 or target.shape != estimate.shape:
        raise ValueError(
            "field_true and field_est must be one-dimensional arrays with equal shape."
        )

    def align_once(candidate: np.ndarray) -> tuple[float, np.ndarray]:
        shift = best_circular_shift_by_magnitude(field_true, candidate)
        shifted = np.roll(candidate, shift)
        scale = np.vdot(shifted, target) / (np.vdot(shifted, shifted) + 1e-20)
        aligned = scale * shifted
        return float(np.mean(np.abs(target - aligned))), aligned

    direct = align_once(estimate)
    reversed_field = align_once(np.conj(estimate[::-1]))
    return reversed_field if reversed_field[0] < direct[0] else direct


def hann_denoise_time(
    field_time: np.ndarray,
    *,
    amp_frac: float = HANN_AMP_FRAC,
    phase_frac: float = HANN_PHASE_FRAC,
) -> np.ndarray:
    def windowed_smooth(values: np.ndarray, frac: float) -> np.ndarray:
        length = max(3, int(round(float(frac) * values.size)))
        if length % 2 == 0:
            length += 1
        window = np.hanning(length)
        window = window / (window.sum() if window.sum() > 0 else 1.0)
        return np.convolve(values, window, mode="same")

    field = np.asarray(field_time)
    amplitude = np.abs(field).astype(np.float64)
    phase = np.unwrap(np.angle(field).astype(np.float64))
    amplitude_smoothed = np.maximum(windowed_smooth(amplitude, amp_frac), 0.0)
    phase_smoothed = np.angle(np.exp(1j * windowed_smooth(phase, phase_frac)))
    return (amplitude_smoothed * np.exp(1j * phase_smoothed)).astype(np.complex64)


def trace_objectives_for_field(
    field_time: np.ndarray,
    *,
    measured_trace: np.ndarray,
    delay_axis: np.ndarray,
    freq_axis: np.ndarray | None,
    time_axis: np.ndarray,
    wl0: float = 800e-9,
    frog_method: str = "tg",
) -> dict[str, float]:
    from pnp_copra.pypret_bootstrap import setup_local_pypret
    from pnp_copra.trace_metrics import MetricConfig, score_trace_objectives

    setup_local_pypret()
    from pypret import FourierTransform, PNPS, Pulse  # noqa: WPS433

    field = np.asarray(field_time, dtype=np.complex128)
    time = np.asarray(time_axis, dtype=np.float64)
    if time.size < 2:
        raise ValueError("time_axis must contain at least two samples.")

    ft = FourierTransform(int(field.size), dt=float(np.mean(np.diff(time))))
    # The experimental pipeline supplies the measured carrier wavelength.
    # Keeping this explicit prevents UV traces from silently using the
    # synthetic 800 nm training default.
    pulse = Pulse(ft, float(wl0))
    pnps = PNPS(pulse, "frog", str(frog_method))
    reconstructed = pnps.calculate(ft.forward(field), np.asarray(delay_axis, dtype=np.float64))
    metrics = score_trace_objectives(
        np.asarray(measured_trace, dtype=np.float64),
        np.asarray(reconstructed, dtype=np.float64),
        delay_axis=np.asarray(delay_axis, dtype=np.float64),
        freq_axis=None if freq_axis is None else np.asarray(freq_axis, dtype=np.float64),
        config=MetricConfig(),
    )
    return {
        **{key: float(value) for key, value in metrics.items()},
        "trace_ssim": float(1.0 - metrics["ssim_distance"]),
    }


def infer_pulse_type(uid: str) -> str:
    match = PULSE_UID_RE.match(uid)
    if match is None:
        return "unknown"
    return match.group("pulse_type")


def stats(values: list[float] | np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p10": float("nan"),
            "p90": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def lower_is_better_comparison(
    baseline: list[float] | np.ndarray,
    candidate: list[float] | np.ndarray,
) -> dict[str, Any]:
    baseline_array, candidate_array = _paired_finite_arrays(baseline, candidate)
    if baseline_array.size == 0:
        return _empty_lower_comparison()

    reduction = baseline_array - candidate_array
    valid_relative = baseline_array > RELATIVE_DENOMINATOR_MIN
    relative = reduction[valid_relative] / baseline_array[valid_relative]
    return {
        "count": int(baseline_array.size),
        "absolute_reduction": stats(reduction),
        "relative_reduction": stats(relative),
        "relative_denominator_min": RELATIVE_DENOMINATOR_MIN,
        "relative_count": int(np.count_nonzero(valid_relative)),
        "improved_fraction": float(np.mean(candidate_array < baseline_array)),
        "worse_fraction": float(np.mean(candidate_array > baseline_array)),
    }


def higher_is_better_comparison(
    baseline: list[float] | np.ndarray,
    candidate: list[float] | np.ndarray,
) -> dict[str, Any]:
    baseline_array, candidate_array = _paired_finite_arrays(baseline, candidate)
    if baseline_array.size == 0:
        return {
            "count": 0,
            "delta": stats([]),
            "degradation": stats([]),
            "improved_fraction": float("nan"),
            "degraded_fraction": float("nan"),
        }

    delta = candidate_array - baseline_array
    return {
        "count": int(baseline_array.size),
        "delta": stats(delta),
        "degradation": stats(-delta),
        "improved_fraction": float(np.mean(candidate_array > baseline_array)),
        "degraded_fraction": float(np.mean(candidate_array < baseline_array)),
    }


def append_group_metrics(
    groups: dict[str, dict[str, list[float]]],
    group_name: str,
    **metrics: float,
) -> None:
    group = groups.setdefault(group_name, {})
    for name, value in metrics.items():
        if np.isfinite(value):
            group.setdefault(name, []).append(float(value))


def summarize_groups(groups: dict[str, dict[str, list[float]]]) -> dict[str, Any]:
    return {
        group_name: summarize_metric_group(metrics)
        for group_name, metrics in sorted(groups.items())
    }


def summarize_metric_group(metrics: dict[str, list[float]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "examples": len(metrics.get("predicted_pulse_mae", [])),
    }
    for name, values in sorted(metrics.items()):
        summary[name] = stats(values)
    for comparison_name, baseline_name, candidate_name in _lower_comparison_specs():
        if baseline_name in metrics and candidate_name in metrics:
            summary[comparison_name] = lower_is_better_comparison(
                metrics[baseline_name],
                metrics[candidate_name],
            )
    for comparison_name, baseline_name, candidate_name in _higher_comparison_specs():
        if baseline_name in metrics and candidate_name in metrics:
            summary[comparison_name] = higher_is_better_comparison(
                metrics[baseline_name],
                metrics[candidate_name],
            )
    return summary


def _paired_finite_arrays(
    baseline: list[float] | np.ndarray,
    candidate: list[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    baseline_array = np.asarray(baseline, dtype=np.float64)
    candidate_array = np.asarray(candidate, dtype=np.float64)
    count = min(baseline_array.size, candidate_array.size)
    baseline_array = baseline_array[:count]
    candidate_array = candidate_array[:count]
    finite = np.isfinite(baseline_array) & np.isfinite(candidate_array)
    return baseline_array[finite], candidate_array[finite]


def _empty_lower_comparison() -> dict[str, Any]:
    return {
        "count": 0,
        "absolute_reduction": stats([]),
        "relative_reduction": stats([]),
        "relative_denominator_min": RELATIVE_DENOMINATOR_MIN,
        "relative_count": 0,
        "improved_fraction": float("nan"),
        "worse_fraction": float("nan"),
    }


def _lower_comparison_specs() -> tuple[tuple[str, str, str], ...]:
    return (
        ("flow_vs_source", "source_pulse_mae", "predicted_pulse_mae"),
        ("hann_source_vs_source", "source_pulse_mae", "hann_source_pulse_mae"),
        ("flow_vs_hann_source", "hann_source_pulse_mae", "predicted_pulse_mae"),
        (
            "flow_vs_copra_hann_direct_final",
            "copra_hann_direct_final_pulse_mae",
            "predicted_for_copra_hann_direct_final_pulse_mae",
        ),
        (
            "flow_vs_copra_hann_guarded_final",
            "copra_hann_guarded_final_pulse_mae",
            "predicted_for_copra_hann_guarded_final_pulse_mae",
        ),
        (
            "flow_vs_copra_hann_best_final",
            "copra_hann_best_final_pulse_mae",
            "predicted_for_copra_hann_best_final_pulse_mae",
        ),
    )


def _higher_comparison_specs() -> tuple[tuple[str, str, str], ...]:
    return (
        ("trace_ssim_flow_vs_source", "source_trace_ssim", "predicted_trace_ssim"),
        ("trace_ssim_hann_source_vs_source", "source_trace_ssim", "hann_source_trace_ssim"),
        ("trace_ssim_flow_vs_hann_source", "hann_source_trace_ssim", "predicted_trace_ssim"),
        (
            "trace_ssim_flow_vs_copra_hann_direct_final",
            "copra_hann_direct_final_trace_ssim",
            "predicted_for_copra_hann_direct_final_trace_ssim",
        ),
        (
            "trace_ssim_flow_vs_copra_hann_guarded_final",
            "copra_hann_guarded_final_trace_ssim",
            "predicted_for_copra_hann_guarded_final_trace_ssim",
        ),
        (
            "trace_ssim_flow_vs_copra_hann_best_final",
            "copra_hann_best_final_trace_ssim",
            "predicted_for_copra_hann_best_final_trace_ssim",
        ),
    )
