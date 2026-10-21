"""Trace-domain objective metrics used to score COPRA candidates.

All public scores are losses: smaller is better.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize_scalar


EPS = 1e-30
TINY = np.finfo(np.float64).tiny


@dataclass(frozen=True)
class MetricConfig:
    eps_fraction: float = 1e-6
    wls_read_noise_fraction: float = 0.01
    wls_shot_noise_fraction: float = 0.02
    wls_floor_fraction: float = 1e-3
    multiscale_sigmas: tuple[float, ...] = (0.0, 1.0, 2.0, 4.0, 8.0)
    multiscale_gradient_weight: float = 0.35
    q1_derivative_weight: float = 0.25
    q1_cdf_weight: float = 0.50
    q1_moment_weight: float = 0.25
    q2_entropy_weight: float = 0.20
    q2_centroid_weight: float = 0.40


def _as_trace(trace: np.ndarray) -> np.ndarray:
    values = np.asarray(trace, dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(values, 0.0)


def _normal_axis(length: int, axis: np.ndarray | None) -> np.ndarray:
    if axis is None:
        values = np.linspace(-1.0, 1.0, int(length), dtype=np.float64)
    else:
        values = np.asarray(axis, dtype=np.float64)
        if values.size != int(length):
            values = np.linspace(-1.0, 1.0, int(length), dtype=np.float64)
    values = values - float(np.mean(values))
    scale = float(np.max(np.abs(values)))
    if scale <= TINY:
        return np.linspace(-1.0, 1.0, int(length), dtype=np.float64)
    return values / scale


def _normalize_sum(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    total = float(np.sum(values))
    if total <= TINY:
        return np.zeros_like(values, dtype=np.float64)
    return values / total


def _normalize_energy(trace: np.ndarray) -> np.ndarray:
    trace = _as_trace(trace)
    total = float(np.sum(trace))
    if total <= TINY:
        return np.zeros_like(trace, dtype=np.float64)
    return trace / total


def _normalize_max(trace: np.ndarray) -> np.ndarray:
    trace = _as_trace(trace)
    peak = float(np.max(trace))
    if peak <= TINY:
        return np.zeros_like(trace, dtype=np.float64)
    return trace / peak


def optimal_intensity_scale(
    measured: np.ndarray,
    reconstructed: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    measured = _as_trace(measured)
    reconstructed = _as_trace(reconstructed)
    w2 = 1.0 if weights is None else np.asarray(weights, dtype=np.float64) ** 2
    numerator = float(np.sum(measured * reconstructed * w2))
    denominator = float(np.sum(reconstructed * reconstructed * w2))
    if denominator <= TINY:
        return 0.0
    return max(0.0, numerator / denominator)


def scaled_reconstruction(
    measured: np.ndarray,
    reconstructed: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    mu = optimal_intensity_scale(measured, reconstructed, weights=weights)
    return mu * _as_trace(reconstructed), mu


def trace_nrmse_loss(measured: np.ndarray, reconstructed: np.ndarray) -> float:
    measured = _as_trace(measured)
    scaled, _ = scaled_reconstruction(measured, reconstructed)
    residual = measured - scaled
    peak = float(np.max(measured))
    return float(np.sqrt(np.sum(residual * residual) / (measured.size * (peak + EPS) ** 2)))


def ssim_global(a: np.ndarray, b: np.ndarray) -> float:
    x = _normalize_max(a)
    y = _normalize_max(b)
    mean_x = float(np.mean(x))
    mean_y = float(np.mean(y))
    var_x = float(np.var(x))
    var_y = float(np.var(y))
    cov_xy = float(np.mean((x - mean_x) * (y - mean_y)))
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2.0 * mean_x * mean_y + c1) * (2.0 * cov_xy + c2)
    denominator = (mean_x * mean_x + mean_y * mean_y + c1) * (var_x + var_y + c2)
    return float(numerator / (denominator + EPS))


def ssim_distance_loss(measured: np.ndarray, reconstructed: np.ndarray) -> float:
    scaled, _ = scaled_reconstruction(measured, reconstructed)
    return float(max(0.0, 1.0 - min(1.0, ssim_global(measured, scaled))))


def build_wls_weights(
    measured: np.ndarray,
    read_noise_fraction: float = 0.01,
    shot_noise_fraction: float = 0.02,
    floor_fraction: float = 1e-3,
) -> np.ndarray:
    measured = _as_trace(measured)
    peak = float(np.max(measured))
    if peak <= TINY:
        return np.ones_like(measured, dtype=np.float64)
    read_noise = float(read_noise_fraction) * peak
    floor = float(floor_fraction) * peak
    variance = read_noise * read_noise
    variance = variance + max(0.0, float(shot_noise_fraction)) * peak * measured
    variance = variance + floor * floor
    weights = 1.0 / np.sqrt(np.maximum(variance, EPS))
    median = float(np.median(weights[weights > 0])) if np.any(weights > 0) else 1.0
    return weights / (median + TINY)


def wls_trace_loss(measured: np.ndarray, reconstructed: np.ndarray, config: MetricConfig) -> float:
    measured = _as_trace(measured)
    weights = build_wls_weights(
        measured,
        read_noise_fraction=config.wls_read_noise_fraction,
        shot_noise_fraction=config.wls_shot_noise_fraction,
        floor_fraction=config.wls_floor_fraction,
    )
    scaled, _ = scaled_reconstruction(measured, reconstructed, weights=weights)
    residual = (measured - scaled) * weights
    peak = float(np.max(measured * weights))
    return float(np.sqrt(np.sum(residual * residual) / (measured.size * (peak + EPS) ** 2)))


def amplitude_trace_loss(
    measured: np.ndarray,
    reconstructed: np.ndarray,
    config: MetricConfig,
) -> float:
    measured = _as_trace(measured)
    reconstructed = _as_trace(reconstructed)
    eps = float(config.eps_fraction) * (float(np.max(measured)) + EPS)
    measured_amp = np.sqrt(measured + eps)
    reconstructed_amp = np.sqrt(reconstructed + eps)
    denominator = float(np.sum(reconstructed_amp * reconstructed_amp))
    scale = 0.0 if denominator <= TINY else max(
        0.0,
        float(np.sum(measured_amp * reconstructed_amp)) / denominator,
    )
    residual = measured_amp - scale * reconstructed_amp
    norm = float(np.sqrt(np.max(measured) + eps))
    return float(np.sqrt(np.mean(residual * residual)) / (norm + EPS))


def log_trace_loss(measured: np.ndarray, reconstructed: np.ndarray, config: MetricConfig) -> float:
    measured = _as_trace(measured)
    reconstructed = _as_trace(reconstructed)
    peak = float(np.max(measured))
    eps = float(config.eps_fraction) * (peak + EPS)
    mu0 = max(optimal_intensity_scale(measured, reconstructed), TINY)
    target = np.log(measured + eps)

    def objective(log_mu: float) -> float:
        candidate = np.log(np.exp(log_mu) * reconstructed + eps)
        residual = target - candidate
        return float(np.mean(residual * residual))

    center = float(np.log(mu0))
    result = minimize_scalar(
        objective,
        bounds=(center - 8.0, center + 8.0),
        method="bounded",
        options={"xatol": 1e-3},
    )
    return float(np.sqrt(result.fun if result.success else objective(center)))


def multiscale_trace_loss(
    measured: np.ndarray,
    reconstructed: np.ndarray,
    config: MetricConfig,
) -> float:
    measured = _normalize_energy(measured)
    reconstructed = _normalize_energy(scaled_reconstruction(measured, reconstructed)[0])
    total = 0.0
    weight_sum = 0.0
    for sigma in config.multiscale_sigmas:
        if sigma > 0:
            y = gaussian_filter(measured, sigma=sigma, mode="nearest")
            x = gaussian_filter(reconstructed, sigma=sigma, mode="nearest")
        else:
            y = measured
            x = reconstructed
        scale_weight = 1.0 / (1.0 + float(sigma))
        residual = y - x
        denom = float(np.mean(y * y)) + EPS
        intensity = float(np.mean(residual * residual) / denom)
        grad_num = 0.0
        grad_den = 0.0
        for gy, gx in zip(np.gradient(y), np.gradient(x)):
            diff = gy - gx
            grad_num += float(np.mean(diff * diff))
            grad_den += float(np.mean(gy * gy))
        gradient = grad_num / (grad_den + EPS)
        total += scale_weight * (intensity + config.multiscale_gradient_weight * gradient)
        weight_sum += scale_weight
    return float(total / (weight_sum + EPS))


def _profile_loss(
    measured_profile: np.ndarray,
    reconstructed_profile: np.ndarray,
    axis: np.ndarray,
    config: MetricConfig,
) -> float:
    y = _normalize_sum(np.maximum(measured_profile, 0.0))
    x = _normalize_sum(np.maximum(reconstructed_profile, 0.0))
    l2 = float(np.mean((y - x) ** 2) / (np.mean(y * y) + EPS))
    cdf = float(np.mean((np.cumsum(y) - np.cumsum(x)) ** 2))
    dy = np.gradient(y)
    dx = np.gradient(x)
    derivative = float(np.mean((dy - dx) ** 2) / (np.mean(dy * dy) + EPS))
    mean_y = float(np.sum(axis * y))
    mean_x = float(np.sum(axis * x))
    std_y = float(np.sqrt(np.sum((axis - mean_y) ** 2 * y)))
    std_x = float(np.sqrt(np.sum((axis - mean_x) ** 2 * x)))
    moment = (mean_y - mean_x) ** 2 + (std_y - std_x) ** 2
    return float(
        l2
        + config.q1_cdf_weight * cdf
        + config.q1_derivative_weight * derivative
        + config.q1_moment_weight * moment
    )


def marginal_trace_loss(
    measured: np.ndarray,
    reconstructed: np.ndarray,
    delay_axis: np.ndarray | None,
    freq_axis: np.ndarray | None,
    config: MetricConfig,
) -> float:
    measured = _as_trace(measured)
    reconstructed = scaled_reconstruction(measured, reconstructed)[0]
    delay = _normal_axis(measured.shape[0], delay_axis)
    freq = _normal_axis(measured.shape[1], freq_axis)
    delay_loss = _profile_loss(np.sum(measured, axis=1), np.sum(reconstructed, axis=1), delay, config)
    freq_loss = _profile_loss(np.sum(measured, axis=0), np.sum(reconstructed, axis=0), freq, config)
    return float(0.5 * (delay_loss + freq_loss))


def _soft_centroid_rows(trace: np.ndarray, axis: np.ndarray, across_rows: bool) -> np.ndarray:
    values = _as_trace(trace)
    if across_rows:
        weights = values / (np.sum(values, axis=1, keepdims=True) + EPS)
        return np.sum(weights * axis[None, :], axis=1)
    weights = values / (np.sum(values, axis=0, keepdims=True) + EPS)
    return np.sum(weights * axis[:, None], axis=0)


def ridge_geometry_loss(
    measured: np.ndarray,
    reconstructed: np.ndarray,
    delay_axis: np.ndarray | None,
    freq_axis: np.ndarray | None,
    config: MetricConfig,
) -> float:
    measured = _normalize_energy(measured)
    reconstructed = _normalize_energy(scaled_reconstruction(measured, reconstructed)[0])
    delay = _normal_axis(measured.shape[0], delay_axis)
    freq = _normal_axis(measured.shape[1], freq_axis)
    freq_centroid_y = _soft_centroid_rows(measured, freq, across_rows=True)
    freq_centroid_x = _soft_centroid_rows(reconstructed, freq, across_rows=True)
    delay_centroid_y = _soft_centroid_rows(measured, delay, across_rows=False)
    delay_centroid_x = _soft_centroid_rows(reconstructed, delay, across_rows=False)
    centroid_loss = float(
        np.mean((freq_centroid_y - freq_centroid_x) ** 2)
        + np.mean((delay_centroid_y - delay_centroid_x) ** 2)
    )
    entropy_y = -float(np.sum(measured * np.log(measured + EPS)))
    entropy_x = -float(np.sum(reconstructed * np.log(reconstructed + EPS)))
    entropy_loss = (entropy_y - entropy_x) ** 2 / (abs(entropy_y) + EPS)
    overlap_loss = float(1.0 - np.sum(np.sqrt(measured * reconstructed)))
    return float(
        overlap_loss
        + config.q2_centroid_weight * centroid_loss
        + config.q2_entropy_weight * entropy_loss
    )


def score_trace_objectives(
    measured: np.ndarray,
    reconstructed: np.ndarray,
    *,
    delay_axis: np.ndarray | None = None,
    freq_axis: np.ndarray | None = None,
    config: MetricConfig | None = None,
) -> dict[str, float]:
    config = config or MetricConfig()
    scaled, scale_mu = scaled_reconstruction(measured, reconstructed)
    return {
        "trace_r": trace_nrmse_loss(measured, reconstructed),
        "ssim_distance": ssim_distance_loss(measured, reconstructed),
        "p1_wls": wls_trace_loss(measured, reconstructed, config),
        "p4_amplitude": amplitude_trace_loss(measured, reconstructed, config),
        "p4_log": log_trace_loss(measured, reconstructed, config),
        "p5_multiscale": multiscale_trace_loss(measured, scaled, config),
        "q1_marginal": marginal_trace_loss(measured, scaled, delay_axis, freq_axis, config),
        "q2_ridge_geometry": ridge_geometry_loss(measured, scaled, delay_axis, freq_axis, config),
        "scale_mu": float(scale_mu),
    }
