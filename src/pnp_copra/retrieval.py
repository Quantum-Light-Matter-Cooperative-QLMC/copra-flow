"""COPRA, post-hoc denoising, and Abhi-style PnP-COPRA utilities."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from .pulse_catalog import PulseLike, PulseSpec, make_pypret_pulse, pulse_like_to_spec
from .pypret_bootstrap import setup_local_pypret
from .trace_metrics import ssim_global

setup_local_pypret()

from pypret import FourierTransform, PNPS, Pulse, Retriever, lib  # noqa: E402
from pypret.retrieval.step_retriever import COPRARetriever  # noqa: E402


GRID_N = 256
GRID_DT = 2.5e-15
WL0 = 800e-9
ALPHA = 0.25
NOISE_LEVEL = 0.01
RNG_SEED = 42

MAXITER = 400
PNP_LAMBDA_INIT = 0.7
PNP_LAMBDA_MIN = 0.05
PNP_BACKTRACK_GAM = 0.5
PNP_ACCEPT_TOL = 1e-6
INIT_KIND = "double-hump"
PHASE_ALIGN_FRAC = 0.05

SUPPORT_KEEP_FRACTION = 0.995
SUPPORT_MIN_FULL_FS = 20.0
SUPPORT_SAFETY_PAD_FS = 5.0
SUPPORT_TUKEY_FRAC = 0.12

HANN_AMP_FRAC = 0.08
HANN_PHASE_FRAC = 0.10


@dataclass
class RetrievalContext:
    spec: PulseSpec
    ft: Any
    pulse: Any
    original_spectrum: np.ndarray
    pnps: Any
    parameter: np.ndarray
    measurement: Any
    time_window: np.ndarray
    noise: float
    pulse_peak_scale: float
    trace_peak_scale: float


def mixture_of_gaussians_amp(t_fs: np.ndarray) -> np.ndarray:
    t_absmax = float(np.max(np.abs(t_fs)))
    mu1 = np.random.uniform(-0.40 * t_absmax, -0.08 * t_absmax)
    mu2 = np.random.uniform(0.08 * t_absmax, 0.40 * t_absmax)
    sigma1 = np.random.uniform(0.04 * t_absmax, 0.12 * t_absmax)
    sigma2 = np.random.uniform(0.04 * t_absmax, 0.12 * t_absmax)
    w1 = np.random.uniform(0.5, 1.2)
    w2 = np.random.uniform(0.5, 1.2)
    amp = (
        w1 * np.exp(-((t_fs - mu1) ** 2) / (2.0 * sigma1**2))
        + w2 * np.exp(-((t_fs - mu2) ** 2) / (2.0 * sigma2**2))
    ).astype(np.float64)
    amp = np.maximum(amp, 0.0)
    amp /= amp.max() + 1e-12
    return amp


def _hann_window(length: int) -> np.ndarray:
    length = max(3, int(length))
    if length % 2 == 0:
        length += 1
    window = np.hanning(length)
    return window / (window.sum() if window.sum() > 0 else 1.0)


def smooth_hann(values: np.ndarray, frac: float) -> np.ndarray:
    length = max(3, int(round(float(frac) * values.size)))
    window = _hann_window(length)
    return np.convolve(values, window, mode="same")


def _unwrap_phase(phi: np.ndarray) -> np.ndarray:
    return np.unwrap(phi.astype(np.float64))


def _rewrap_phase(phi: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * phi))


def denoise_time_classic(field_time: np.ndarray) -> np.ndarray:
    amplitude = np.abs(field_time).astype(np.float64)
    phase = np.angle(field_time).astype(np.float64)
    amplitude_smoothed = smooth_hann(amplitude, HANN_AMP_FRAC)
    phase_smoothed = smooth_hann(_unwrap_phase(phase), HANN_PHASE_FRAC)
    phase_smoothed = _rewrap_phase(phase_smoothed)
    amplitude_smoothed = np.maximum(amplitude_smoothed, 0.0)
    return (amplitude_smoothed * np.exp(1j * phase_smoothed)).astype(np.complex128)


def build_soft_time_window(t_s: np.ndarray, half_extent_s: float, ramp_frac: float) -> np.ndarray:
    if half_extent_s <= 0:
        return np.ones_like(t_s, dtype=np.float64)
    full_width = 2.0 * half_extent_s
    ramp_s = float(ramp_frac) * full_width
    core = max(0.0, half_extent_s - 0.5 * ramp_s)
    window = np.zeros_like(t_s, dtype=np.float64)
    abs_t = np.abs(t_s)
    core_mask = abs_t <= core + 1e-30
    window[core_mask] = 1.0
    if ramp_s > 0:
        ramp_mask = (abs_t > core) & (abs_t <= half_extent_s + 1e-30)
        x = (abs_t[ramp_mask] - core) / max(ramp_s, 1e-30)
        window[ramp_mask] = 0.5 * (1.0 + np.cos(np.pi * x))
    return window


def _smooth_1d(values: np.ndarray, win: int = 11) -> np.ndarray:
    win = max(3, int(win) | 1)
    window = np.hanning(win)
    window /= window.sum()
    return np.convolve(values, window, mode="same")


def estimate_time_window_from_measured_trace(
    meas_df: np.ndarray,
    delays_s: np.ndarray,
    t_s: np.ndarray,
    keep_fraction: float = SUPPORT_KEEP_FRACTION,
    min_full_width_fs: float = SUPPORT_MIN_FULL_FS,
    safety_pad_fs: float = SUPPORT_SAFETY_PAD_FS,
    ramp_frac: float = SUPPORT_TUKEY_FRAC,
) -> np.ndarray:
    if meas_df is None or meas_df.size == 0:
        return np.ones_like(t_s, dtype=np.float64)

    marginal = np.sum(np.asarray(meas_df, dtype=np.float64), axis=1)
    if not np.any(marginal > 0):
        return np.ones_like(t_s, dtype=np.float64)

    marginal = _smooth_1d(marginal, win=max(9, len(marginal) // 40))
    marginal = np.maximum(marginal, 0.0)
    marginal_sum = float(marginal.sum())
    if marginal_sum <= 0:
        return np.ones_like(t_s, dtype=np.float64)

    weights = marginal / marginal_sum
    cdf = np.cumsum(weights)
    tail = max(0.0, min(0.5 * (1.0 - float(keep_fraction)), 0.5))
    i_left = int(np.searchsorted(cdf, tail))
    i_right = int(np.searchsorted(cdf, 1.0 - tail) - 1)
    i_left = max(0, min(i_left, len(cdf) - 1))
    i_right = max(i_left, min(i_right, len(cdf) - 1))

    delay_span_s = float(delays_s[i_right] - delays_s[i_left])
    half_extent_s = 0.25 * abs(delay_span_s)
    half_extent_s += float(safety_pad_fs) * 1e-15
    half_extent_s = max(half_extent_s, 0.5 * float(min_full_width_fs) * 1e-15)
    half_extent_s = min(half_extent_s, float(np.max(np.abs(t_s))))
    return build_soft_time_window(t_s, half_extent_s, ramp_frac)


def _unit_max(values: np.ndarray) -> np.ndarray:
    return values / (np.max(values) + 1e-12)


def compute_trace_metrics(pnps, measurement, spectrum: np.ndarray, parameter: np.ndarray) -> tuple[float, float]:
    measured = measurement.data.astype(np.float64)
    reconstructed = pnps.calculate(spectrum, parameter).astype(np.float64)
    mu = np.sum(measured * reconstructed) / (np.sum(reconstructed * reconstructed) + 1e-30)
    scaled = mu * reconstructed
    residual = measured - scaled
    trace_error = np.sqrt(np.sum(residual * residual) / (measured.size * (measured.max() + 1e-30) ** 2))
    trace_ssim = ssim_global(measured, scaled)
    return float(trace_error), float(trace_ssim)


def compute_trace_ssim(pnps, measurement, spectrum: np.ndarray, parameter: np.ndarray) -> float:
    return compute_trace_metrics(pnps, measurement, spectrum, parameter)[1]


def mix_with_ssim_backtracking(
    spectrum_base: np.ndarray,
    spectrum_denoised: np.ndarray,
    trace_ssim_fn,
    lam0: float = PNP_LAMBDA_INIT,
    lam_min: float = PNP_LAMBDA_MIN,
    backtrack_gamma: float = PNP_BACKTRACK_GAM,
    accept_tol: float = PNP_ACCEPT_TOL,
) -> tuple[np.ndarray, float, float, bool]:
    lam = float(lam0)
    ssim_base = float(trace_ssim_fn(spectrum_base))
    while lam >= lam_min:
        spectrum_try = (1.0 - lam) * spectrum_base + lam * spectrum_denoised
        ssim_try = float(trace_ssim_fn(spectrum_try))
        if ssim_try >= ssim_base * (1.0 - accept_tol):
            return spectrum_try, ssim_try, float(lam), True
        lam *= backtrack_gamma
    return spectrum_base.copy(), ssim_base, 0.0, False


def _best_circ_shift_by_mag(field_true: np.ndarray, field_est: np.ndarray) -> int:
    correlation = np.fft.ifft(
        np.fft.fft(np.abs(field_true)) * np.conj(np.fft.fft(np.abs(field_est)))
    )
    return int(np.argmax(np.real(correlation)))


def align_and_pulse_mae(field_true: np.ndarray, field_est: np.ndarray) -> tuple[float, np.ndarray]:
    def align_once(candidate: np.ndarray) -> tuple[float, np.ndarray]:
        shift = _best_circ_shift_by_mag(field_true, candidate)
        shifted = np.roll(candidate, shift)
        scale = np.vdot(shifted, field_true) / (np.vdot(shifted, shifted) + 1e-20)
        aligned = scale * shifted
        mae = float(np.mean(np.abs(field_true - aligned)))
        return mae, aligned

    mae_direct, aligned_direct = align_once(field_est)
    mae_reverse, aligned_reverse = align_once(np.conj(field_est[::-1]))
    if mae_reverse < mae_direct:
        return mae_reverse, aligned_reverse
    return mae_direct, aligned_direct


def field_from_spectrum(ft, spectrum: np.ndarray, wl0: float = WL0) -> np.ndarray:
    pulse = Pulse(ft, wl0)
    pulse.spectrum = np.asarray(spectrum).copy()
    return np.asarray(pulse.field).astype(np.complex128)


def spectrum_from_field(ft, field: np.ndarray) -> np.ndarray:
    return ft.forward(np.asarray(field, dtype=np.complex128)).astype(np.complex128)


def compute_pulse_mae(
    spectrum_retrieved: np.ndarray,
    spectrum_original: np.ndarray,
    ft,
    wl0: float = WL0,
) -> tuple[float, np.ndarray]:
    field_true = field_from_spectrum(ft, spectrum_original, wl0=wl0)
    field_retrieved = field_from_spectrum(ft, spectrum_retrieved, wl0=wl0)
    return align_and_pulse_mae(field_true, field_retrieved)


def make_initial_guess(ft) -> np.ndarray:
    guess_pulse = Pulse(ft, WL0)
    if INIT_KIND.lower().startswith("double"):
        t_s = (np.arange(ft.N) - ft.N // 2) * ft.dt
        amplitude = mixture_of_gaussians_amp(t_s * 1e15)
        return ft.forward(amplitude.astype(np.complex128))
    import pypret

    pypret.random_gaussian(guess_pulse, 50e-15, phase_max=0.0)
    return guess_pulse.spectrum.copy()


class AbhiPnPCOPRAClassic(COPRARetriever):
    """COPRA global step followed by Hann denoising with SSIM backtracking."""

    method = None

    def __init__(
        self,
        pnps,
        lam0: float = PNP_LAMBDA_INIT,
        lam_min: float = PNP_LAMBDA_MIN,
        backtrack_gamma: float = PNP_BACKTRACK_GAM,
        accept_tol: float = PNP_ACCEPT_TOL,
        time_window: np.ndarray | None = None,
        **kwargs,
    ):
        super().__init__(pnps, **kwargs)
        self._lam0 = float(lam0)
        self._lam_min = float(lam_min)
        self._bt = float(backtrack_gamma)
        self._tol = float(accept_tol)
        self._time_window = np.asarray(time_window, dtype=np.float64) if time_window is not None else None

    def _trace_ssim_given_en(self, en_candidate: np.ndarray) -> float:
        rs = self._retrieval_state
        trace_candidate = self.pnps.calculate(en_candidate, self.parameter)
        return ssim_global(self.Tmn_meas, rs.mu * trace_candidate)

    def _retrieve_step(self, iteration, en):
        ft = self.ft
        pnps = self.pnps
        rs = self._retrieval_state
        options = self.options
        measured = self.Tmn_meas

        rs.previous_max_gradient = rs.current_max_gradient
        rs.current_max_gradient = 0.0
        if rs.steps_since_improvement == 5:
            rs.mode = "global"

        if rs.mode == "local":
            trace = np.zeros((self.M, self.N))
            for m in np.random.permutation(np.arange(self.M)):
                parameter_value = self.parameter[m]
                trace[m, :] = pnps.calculate(en, parameter_value)
                projected = self._project(measured[m, :] / rs.mu, pnps.Smk)
                gradient = pnps.gradient(projected, parameter_value)
                z_value = lib.norm2(projected - pnps.Smk)
                gradient_norm = lib.norm2(gradient)
                if gradient_norm > rs.current_max_gradient:
                    rs.current_max_gradient = gradient_norm
                gamma = z_value / max(rs.current_max_gradient, rs.previous_max_gradient)
                en -= gamma * gradient
            rs.approximate_error = True
            return self._R(trace), en

        trace = pnps.calculate(en, self.parameter)
        r_value = self._r(trace)
        r_metric = self._Rr(r_value)

        w2 = self._weights * self._weights
        grad_rmk = (
            -4
            * ft.dt
            / (ft.dw * lib.twopi)
            * ft.backward(rs.mu * ft.forward(pnps.Smk) * (measured - rs.mu * trace) * w2)
        )
        eta_r = options.alpha * r_value / max(lib.norm2(grad_rmk), 1e-20)
        smk2 = pnps.Smk - eta_r * grad_rmk
        grad_zn = pnps.gradient(smk2, self.parameter).sum(axis=0)
        z_value = lib.norm2(smk2 - pnps.Smk)
        eta_z = options.alpha * z_value / max(lib.norm2(grad_zn), 1e-20)
        en_gd = en - eta_z * grad_zn

        denoised_time = denoise_time_classic(ft.backward(en_gd))
        if self._time_window is not None and len(self._time_window) == self.N:
            denoised_time = denoised_time * self._time_window
        en_denoised = ft.forward(denoised_time)

        en_next, _, _, _ = mix_with_ssim_backtracking(
            en_gd,
            en_denoised,
            self._trace_ssim_given_en,
            lam0=self._lam0,
            lam_min=self._lam_min,
            backtrack_gamma=self._bt,
            accept_tol=self._tol,
        )
        return r_metric, en_next


def run_copra(
    pnps,
    measurement,
    initial_spectrum: np.ndarray,
    maxiter: int = MAXITER,
    alpha: float = ALPHA,
    verbose: bool = False,
):
    retriever = Retriever(
        pnps,
        "copra",
        verbose=verbose,
        logging=True,
        maxiter=maxiter,
        alpha=alpha,
    )
    retriever.retrieve(measurement, initial_spectrum.copy())
    return retriever._result.spectrum.copy(), float(retriever._result.trace_error), retriever


def run_copra_denoised_posthoc(
    pnps,
    measurement,
    copra_spectrum: np.ndarray,
    ft,
    parameter: np.ndarray,
    time_window: np.ndarray | None,
) -> tuple[np.ndarray, float, float]:
    denoised_time = denoise_time_classic(ft.backward(copra_spectrum))
    if time_window is not None and len(time_window) == len(denoised_time):
        denoised_time = denoised_time * time_window
    spectrum_denoised = ft.forward(denoised_time)
    spectrum_backtracked, trace_ssim, _, _ = mix_with_ssim_backtracking(
        copra_spectrum,
        spectrum_denoised,
        lambda spectrum: compute_trace_ssim(pnps, measurement, spectrum, parameter),
        lam0=PNP_LAMBDA_INIT,
        lam_min=PNP_LAMBDA_MIN,
        backtrack_gamma=PNP_BACKTRACK_GAM,
        accept_tol=PNP_ACCEPT_TOL,
    )
    trace_error, trace_ssim = compute_trace_metrics(pnps, measurement, spectrum_backtracked, parameter)
    return spectrum_backtracked.copy(), trace_error, trace_ssim


def run_abhi_pnp(
    pnps,
    measurement,
    initial_spectrum: np.ndarray,
    time_window: np.ndarray | None,
    maxiter: int = MAXITER,
    alpha: float = ALPHA,
    verbose: bool = False,
):
    retriever = AbhiPnPCOPRAClassic(
        pnps,
        verbose=verbose,
        logging=True,
        maxiter=maxiter,
        alpha=alpha,
        time_window=time_window,
    )
    retriever.retrieve(measurement, initial_spectrum.copy())
    return retriever._result.spectrum.copy(), float(retriever._result.trace_error), retriever


def make_context(
    pulse_like: PulseLike,
    *,
    seed: int = RNG_SEED,
    noise: float = NOISE_LEVEL,
    grid_n: int = GRID_N,
    grid_dt: float = GRID_DT,
    wl0: float = WL0,
    frog_method: str = "tg",
) -> RetrievalContext:
    np.random.seed(int(seed))
    random.seed(int(seed))
    spec = pulse_like_to_spec(pulse_like)

    ft = FourierTransform(int(grid_n), dt=float(grid_dt))
    pulse = make_pypret_pulse(spec, ft, wl0=wl0)
    field = np.asarray(pulse.field, dtype=np.complex128)
    field_peak = float(np.max(np.abs(field))) if field.size else 0.0
    if np.isfinite(field_peak) and field_peak > 0.0:
        pulse.field = field / field_peak
    else:
        field_peak = 1.0
    original_spectrum = pulse.spectrum.copy()
    pnps = PNPS(pulse, "frog", frog_method)
    parameter = ft.t
    pnps.calculate(pulse.spectrum, parameter)
    measurement = pnps.trace
    trace_peak = float(np.max(measurement.data)) if measurement.data.size else 0.0
    if np.isfinite(trace_peak) and trace_peak > 0.0:
        measurement.data = measurement.data / trace_peak
    else:
        trace_peak = 1.0

    if noise > 0.0:
        std = float(measurement.data.max()) * float(noise)
        measurement.data += std * np.random.normal(size=measurement.data.shape)
        measurement.data = np.maximum(measurement.data, 0.0)

    time_window = estimate_time_window_from_measured_trace(
        measurement.data,
        parameter,
        ft.t,
        keep_fraction=SUPPORT_KEEP_FRACTION,
        min_full_width_fs=SUPPORT_MIN_FULL_FS,
        safety_pad_fs=SUPPORT_SAFETY_PAD_FS,
        ramp_frac=SUPPORT_TUKEY_FRAC,
    )
    return RetrievalContext(
        spec=spec,
        ft=ft,
        pulse=pulse,
        original_spectrum=original_spectrum,
        pnps=pnps,
        parameter=parameter,
        measurement=measurement,
        time_window=time_window,
        noise=float(noise),
        pulse_peak_scale=float(field_peak),
        trace_peak_scale=float(trace_peak),
    )
