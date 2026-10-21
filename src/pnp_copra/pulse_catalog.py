"""Named and stochastic pulse factories for synthetic COPRA datasets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from scipy.ndimage import uniform_filter1d

from .pypret_bootstrap import setup_local_pypret

setup_local_pypret()

from pypret import Pulse  # noqa: E402


TWOPI = 2.0 * np.pi
WL0_DEFAULT = 800e-9


BASE_PULSES = [
    "double_pulse_equal",
    "gaussian_tl",
    "gaussian_chirped",
    "sech",
    "super_gaussian_4",
    "double_pulse_unequal",
    "gaussian_gdd_tod",
    "asymmetric",
    "spectral_hole",
    "sinc",
    "random_phase",
    "spm_fiber",
    "double_pulse_pi",
]

STEP_PHASE_PULSES = [
    "gaussian_phase_step_pi",
    "gaussian_phase_step_halfpi",
    "gaussian_phase_step_double_pi",
    "gaussian_phase_step_offset",
]

SINUSOIDAL_PHASE_PULSES = [
    "gaussian_phase_sine_low",
    "gaussian_phase_sine_mid",
    "gaussian_phase_sine_high",
    "gaussian_phase_sine_chirped",
]

BIEXPONENTIAL_PULSES = [
    "biexponential_coupled_soft",
    "biexponential_coupled_medium",
    "biexponential_coupled_strong",
]

PHASE_JUMP_PULSES = [
    "phase_jump_center_pi",
    "phase_jump_dual",
    "phase_jump_triplet",
]

RANDOM_MIXTURE_PULSES = [f"mog_phase_{index:02d}" for index in range(1, 11)]

PULSE_LIST = (
    BASE_PULSES
    + STEP_PHASE_PULSES
    + SINUSOIDAL_PHASE_PULSES
    + BIEXPONENTIAL_PULSES
    + PHASE_JUMP_PULSES
    + RANDOM_MIXTURE_PULSES
)

PULSE_LABELS = {
    "double_pulse_equal": "Gaussian Double Hump (equal, Abhi benchmark)",
    "gaussian_tl": "Transform-Limited Gaussian",
    "gaussian_chirped": "Chirped Gaussian",
    "sech": "Sech",
    "super_gaussian_4": "Super-Gaussian (order=4)",
    "double_pulse_unequal": "Double Pulse (unequal, pi/3 phase)",
    "gaussian_gdd_tod": "Higher-Order Chirp (GDD+TOD)",
    "asymmetric": "Asymmetric Higher-Order Chirp",
    "spectral_hole": "Spectral Hole Pulse",
    "sinc": "Sinc / Rectangular Spectrum",
    "random_phase": "Random Spectral Phase",
    "spm_fiber": "Self-Phase Modulation",
    "double_pulse_pi": "Double Pulse (pi phase)",
    "gaussian_phase_step_pi": "Gaussian Amplitude + pi Phase Step",
    "gaussian_phase_step_halfpi": "Gaussian Amplitude + pi/2 Phase Step",
    "gaussian_phase_step_double_pi": "Gaussian Amplitude + Dual pi Phase Steps",
    "gaussian_phase_step_offset": "Gaussian Amplitude + Offset Phase Step",
    "gaussian_phase_sine_low": "Gaussian Amplitude + Low-Frequency Sinusoidal Phase",
    "gaussian_phase_sine_mid": "Gaussian Amplitude + Mid-Frequency Sinusoidal Phase",
    "gaussian_phase_sine_high": "Gaussian Amplitude + High-Frequency Sinusoidal Phase",
    "gaussian_phase_sine_chirped": "Gaussian Amplitude + Sinusoidal/Quadratic Phase",
    "biexponential_coupled_soft": "Biexponential Amplitude-Phase Coupling (soft)",
    "biexponential_coupled_medium": "Biexponential Amplitude-Phase Coupling (medium)",
    "biexponential_coupled_strong": "Biexponential Amplitude-Phase Coupling (strong)",
    "phase_jump_center_pi": "Gaussian Amplitude + Central pi Jump",
    "phase_jump_dual": "Gaussian Amplitude + Dual Phase Jumps",
    "phase_jump_triplet": "Gaussian Amplitude + Triplet Phase Jumps",
}
for index, pulse_name in enumerate(RANDOM_MIXTURE_PULSES, start=1):
    PULSE_LABELS[pulse_name] = f"Random Gaussian Mixture {index:02d}"


FAMILY_WEIGHTS = {
    "gaussian": 0.10,
    "sech": 0.06,
    "super_gaussian": 0.06,
    "double_pulse": 0.09,
    "pulse_train": 0.08,
    "phase_step": 0.09,
    "sinusoidal_phase": 0.08,
    "biexponential": 0.06,
    "random_mixture": 0.10,
    "random_spectral": 0.08,
    "spectral_hole": 0.06,
    "spm": 0.05,
    "sinc": 0.03,
    "asymmetric_tail": 0.06,
    "lorentzian": 0.04,
    "airy": 0.04,
    "spectral_comb": 0.06,
}


@dataclass(frozen=True)
class PulseSpec:
    """A serializable synthetic pulse specification."""

    uid: str
    family: str
    params: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None
    label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "family": self.family,
            "params": self.params,
            "seed": self.seed,
            "label": self.label or self.family,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PulseSpec":
        return cls(
            uid=str(payload["uid"]),
            family=str(payload["family"]),
            params=dict(payload.get("params") or {}),
            seed=None if payload.get("seed") is None else int(payload["seed"]),
            label=None if payload.get("label") is None else str(payload["label"]),
        )


PulseLike = str | PulseSpec | Mapping[str, Any]


def sanitize_uid(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def spec_digest(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def pulse_like_to_spec(pulse_like: PulseLike) -> PulseSpec:
    if isinstance(pulse_like, PulseSpec):
        return pulse_like
    if isinstance(pulse_like, str):
        if pulse_like not in PULSE_LIST:
            raise ValueError(f"Unknown named pulse: {pulse_like}")
        return PulseSpec(
            uid=sanitize_uid(pulse_like),
            family="named",
            params={"name": pulse_like},
            label=PULSE_LABELS.get(pulse_like, pulse_like),
        )
    return PulseSpec.from_dict(pulse_like)


def _time_fs(ft) -> np.ndarray:
    return ft.t * 1e15


def _normalize_real(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / (np.max(np.abs(values)) + 1e-12)


def _gaussian_envelope(t_s: np.ndarray, fwhm_s: float) -> np.ndarray:
    sigma_t = fwhm_s / (2.0 * np.sqrt(np.log(2.0)))
    return np.exp(-0.5 * (t_s / sigma_t) ** 2)


def _step_phase(t_fs: np.ndarray, jump_locations_fs: list[float], jump_values: list[float]) -> np.ndarray:
    phase = np.zeros_like(t_fs, dtype=np.float64)
    for location_fs, jump_value in zip(jump_locations_fs, jump_values):
        phase = phase + float(jump_value) * (t_fs >= float(location_fs))
    return phase


def _sinusoidal_phase(
    t_fs: np.ndarray,
    cycles: float,
    depth_rad: float,
    phase_offset: float = 0.0,
) -> np.ndarray:
    extent = np.max(np.abs(t_fs)) + 1e-12
    return float(depth_rad) * np.sin(TWOPI * float(cycles) * t_fs / (2.0 * extent) + phase_offset)


def _build_phase_step_field(
    ft,
    fwhm_fs: float,
    jump_locations_fs: list[float],
    jump_values: list[float],
) -> np.ndarray:
    amplitude = _gaussian_envelope(ft.t, float(fwhm_fs) * 1e-15)
    phase = _step_phase(_time_fs(ft), jump_locations_fs, jump_values)
    return amplitude * np.exp(1j * phase)


def _build_sinusoidal_phase_field(
    ft,
    fwhm_fs: float,
    cycles: float,
    depth_rad: float,
    quadratic_scale: float = 0.0,
) -> np.ndarray:
    t_fs = _time_fs(ft)
    extent = np.max(np.abs(t_fs)) + 1e-12
    amplitude = _gaussian_envelope(ft.t, float(fwhm_fs) * 1e-15)
    phase = _sinusoidal_phase(t_fs, cycles=float(cycles), depth_rad=float(depth_rad))
    phase = phase + float(quadratic_scale) * (t_fs / extent) ** 2
    return amplitude * np.exp(1j * phase)


def _build_biexponential_coupled_field(
    ft,
    tau_fast_fs: float,
    tau_slow_fs: float,
    phase_scale: float,
    skew: float,
) -> np.ndarray:
    t_fs = _time_fs(ft)
    abs_t = np.abs(t_fs)
    amplitude = np.exp(-abs_t / float(tau_fast_fs)) + float(skew) * np.exp(
        -abs_t / float(tau_slow_fs)
    )
    amplitude = _normalize_real(amplitude)
    phase = float(phase_scale) * (
        np.exp(-abs_t / max(0.5 * float(tau_fast_fs), 1e-9))
        - 0.65 * np.exp(-abs_t / float(tau_slow_fs))
    )
    phase = phase * np.sign(t_fs + 1e-12)
    return amplitude * np.exp(1j * phase)


def _build_multi_jump_field(
    ft,
    fwhm_fs: float,
    jump_locations_fs: list[float],
    jump_values: list[float],
    ramp_cycles: float = 0.0,
) -> np.ndarray:
    t_fs = _time_fs(ft)
    amplitude = _gaussian_envelope(ft.t, float(fwhm_fs) * 1e-15)
    phase = _step_phase(t_fs, jump_locations_fs, jump_values)
    if ramp_cycles > 0.0:
        phase = phase + _sinusoidal_phase(t_fs, cycles=float(ramp_cycles), depth_rad=0.45 * np.pi)
    return amplitude * np.exp(1j * phase)


def _random_gaussian_mixture_field_from_seed(ft, index: int) -> np.ndarray:
    rng = np.random.RandomState(1200 + int(index))
    params = _sample_random_mixture_params(np.random.default_rng(rng.randint(0, 2**31 - 1)), ft)
    return _random_gaussian_mixture_field(ft, params)


def _random_gaussian_mixture_field(ft, params: Mapping[str, Any]) -> np.ndarray:
    t_fs = _time_fs(ft)
    amplitude = np.zeros_like(t_fs, dtype=np.float64)
    for component in params["components"]:
        mu = float(component["mu_fs"])
        sigma = float(component["sigma_fs"])
        weight = float(component["weight"])
        amplitude = amplitude + weight * np.exp(-0.5 * ((t_fs - mu) / sigma) ** 2)
    amplitude = _normalize_real(amplitude)

    phase_kind = str(params.get("phase_kind", "sine"))
    if phase_kind == "sine":
        phase = _sinusoidal_phase(
            t_fs,
            cycles=float(params["cycles"]),
            depth_rad=float(params["depth_rad"]),
            phase_offset=float(params.get("phase_offset", 0.0)),
        )
    elif phase_kind == "step":
        phase = _step_phase(t_fs, list(params["jump_locations_fs"]), list(params["jump_values"]))
    elif phase_kind == "poly_sine":
        extent = np.max(np.abs(t_fs)) + 1e-12
        phase = float(params["quadratic"]) * (t_fs / extent) ** 2
        phase = phase + float(params.get("cubic", 0.0)) * (t_fs / extent) ** 3
        phase = phase + _sinusoidal_phase(
            t_fs,
            cycles=float(params["cycles"]),
            depth_rad=float(params["depth_rad"]),
        )
    else:
        center = float(params.get("tanh_center_fs", 0.0))
        width = float(params.get("tanh_width_fs", 20.0))
        phase = float(params.get("tanh_scale", 1.0)) * np.tanh((t_fs - center) / width)
        phase = phase + _sinusoidal_phase(
            t_fs,
            cycles=float(params.get("cycles", 2.0)),
            depth_rad=float(params.get("depth_rad", 0.6)),
        )
    return amplitude * np.exp(1j * phase)


def _make_named_pulse(pulse_name: str, ft, wl0: float) -> Pulse:
    pulse = Pulse(ft, wl0)
    t_s = ft.t
    omega_grid = ft.w
    grid_size = ft.N

    if pulse_name == "double_pulse_equal":
        fwhm_t = 30e-15
        separation = 150e-15
        sigma_t = fwhm_t / (2.0 * np.sqrt(np.log(2.0)))
        field_left = np.exp(-0.5 * ((t_s + separation / 2.0) / sigma_t) ** 2)
        field_right = np.exp(-0.5 * ((t_s - separation / 2.0) / sigma_t) ** 2)
        pulse.field = field_left + field_right
    elif pulse_name == "double_pulse_unequal":
        fwhm_t = 30e-15
        separation = 150e-15
        ratio = 0.5
        rel_phase = np.pi / 3.0
        sigma_t = fwhm_t / (2.0 * np.sqrt(np.log(2.0)))
        field_left = np.exp(-0.5 * ((t_s + separation / 2.0) / sigma_t) ** 2)
        field_right = ratio * np.exp(1j * rel_phase) * np.exp(
            -0.5 * ((t_s - separation / 2.0) / sigma_t) ** 2
        )
        pulse.field = field_left + field_right
    elif pulse_name == "gaussian_tl":
        pulse.field = _gaussian_envelope(t_s, 50e-15)
    elif pulse_name == "gaussian_chirped":
        fwhm_t0 = 50e-15
        gdd = 2000e-30
        sigma_t0 = fwhm_t0 / (2.0 * np.sqrt(np.log(2.0)))
        dw_spec = 1.0 / sigma_t0
        pulse.spectrum = np.exp(-0.5 * (omega_grid / dw_spec) ** 2) * np.exp(
            0.5j * gdd * omega_grid**2
        )
    elif pulse_name == "sech":
        fwhm_t = 50e-15
        t0 = fwhm_t / (2.0 * np.log(1.0 + np.sqrt(2.0)))
        pulse.field = 1.0 / np.cosh(t_s / t0)
    elif pulse_name == "super_gaussian_4":
        fwhm_t = 80e-15
        order = 4
        sigma_t = fwhm_t / (2.0 * np.log(2.0) ** (1.0 / (2 * order)))
        pulse.field = np.exp(-0.5 * (t_s / sigma_t) ** (2 * order))
    elif pulse_name == "gaussian_gdd_tod":
        fwhm_t0 = 50e-15
        gdd = 1000e-30
        tod = 5000e-45
        sigma_t0 = fwhm_t0 / (2.0 * np.sqrt(np.log(2.0)))
        dw_spec = 1.0 / sigma_t0
        phase = 0.5 * gdd * omega_grid**2 + tod / 6.0 * omega_grid**3
        pulse.spectrum = np.exp(-0.5 * (omega_grid / dw_spec) ** 2) * np.exp(1j * phase)
    elif pulse_name == "asymmetric":
        fwhm_t0 = 50e-15
        gdd = 500e-30
        tod = 20000e-45
        sigma_t0 = fwhm_t0 / (2.0 * np.sqrt(np.log(2.0)))
        dw_spec = 1.0 / sigma_t0
        phase = 0.5 * gdd * omega_grid**2 + tod / 6.0 * omega_grid**3
        pulse.spectrum = np.exp(-0.5 * (omega_grid / dw_spec) ** 2) * np.exp(1j * phase)
    elif pulse_name == "spectral_hole":
        fwhm_t0 = 50e-15
        gdd = 1000e-30
        hole_center = TWOPI * 5e12
        hole_width = TWOPI * 2e12
        hole_depth = 0.9
        sigma_t0 = fwhm_t0 / (2.0 * np.sqrt(np.log(2.0)))
        dw_spec = 1.0 / sigma_t0
        sigma_h = hole_width / (2.0 * np.sqrt(np.log(2.0)))
        envelope = np.exp(-0.5 * (omega_grid / dw_spec) ** 2)
        notch = 1.0 - hole_depth * np.exp(-0.5 * ((omega_grid - hole_center) / sigma_h) ** 2)
        pulse.spectrum = envelope * notch * np.exp(0.5j * gdd * omega_grid**2)
    elif pulse_name == "sinc":
        bandwidth = TWOPI * 20e12
        spectrum = np.zeros(grid_size, dtype=complex)
        spectrum[np.abs(omega_grid) <= bandwidth / 2.0] = 1.0
        pulse.spectrum = spectrum
    elif pulse_name == "random_phase":
        fwhm_t0 = 50e-15
        sigma_t0 = fwhm_t0 / (2.0 * np.sqrt(np.log(2.0)))
        dw_spec = 1.0 / sigma_t0
        rng = np.random.RandomState(12345)
        random_phase = uniform_filter1d(rng.uniform(-np.pi, np.pi, size=grid_size), size=5)
        pulse.spectrum = np.exp(-0.5 * (omega_grid / dw_spec) ** 2) * np.exp(1j * random_phase)
    elif pulse_name == "spm_fiber":
        fwhm_t = 50e-15
        b_integral = 5.0
        sigma_t = fwhm_t / (2.0 * np.sqrt(np.log(2.0)))
        envelope = np.exp(-0.5 * (t_s / sigma_t) ** 2)
        pulse.field = envelope * np.exp(1j * (b_integral * envelope**2))
    elif pulse_name == "double_pulse_pi":
        fwhm_t = 30e-15
        separation = 80e-15
        sigma_t = fwhm_t / (2.0 * np.sqrt(np.log(2.0)))
        field_left = np.exp(-0.5 * ((t_s + separation / 2.0) / sigma_t) ** 2)
        field_right = np.exp(1j * np.pi) * np.exp(-0.5 * ((t_s - separation / 2.0) / sigma_t) ** 2)
        pulse.field = field_left + field_right
    elif pulse_name == "gaussian_phase_step_pi":
        pulse.field = _build_phase_step_field(ft, 55.0, [0.0], [np.pi])
    elif pulse_name == "gaussian_phase_step_halfpi":
        pulse.field = _build_phase_step_field(ft, 60.0, [15.0], [0.5 * np.pi])
    elif pulse_name == "gaussian_phase_step_double_pi":
        pulse.field = _build_phase_step_field(ft, 65.0, [-30.0, 30.0], [np.pi, -np.pi])
    elif pulse_name == "gaussian_phase_step_offset":
        pulse.field = _build_phase_step_field(ft, 58.0, [40.0], [0.75 * np.pi])
    elif pulse_name == "gaussian_phase_sine_low":
        pulse.field = _build_sinusoidal_phase_field(ft, 55.0, 1.5, 0.9)
    elif pulse_name == "gaussian_phase_sine_mid":
        pulse.field = _build_sinusoidal_phase_field(ft, 55.0, 3.0, 1.6)
    elif pulse_name == "gaussian_phase_sine_high":
        pulse.field = _build_sinusoidal_phase_field(ft, 55.0, 5.5, 2.8)
    elif pulse_name == "gaussian_phase_sine_chirped":
        pulse.field = _build_sinusoidal_phase_field(ft, 55.0, 4.0, 2.2, quadratic_scale=1.8)
    elif pulse_name == "biexponential_coupled_soft":
        pulse.field = _build_biexponential_coupled_field(ft, 18.0, 75.0, 1.2, 0.45)
    elif pulse_name == "biexponential_coupled_medium":
        pulse.field = _build_biexponential_coupled_field(ft, 14.0, 90.0, 1.9, 0.65)
    elif pulse_name == "biexponential_coupled_strong":
        pulse.field = _build_biexponential_coupled_field(ft, 11.0, 105.0, 2.6, 0.85)
    elif pulse_name == "phase_jump_center_pi":
        pulse.field = _build_multi_jump_field(ft, 60.0, [0.0], [np.pi])
    elif pulse_name == "phase_jump_dual":
        pulse.field = _build_multi_jump_field(ft, 62.0, [-40.0, 40.0], [0.75 * np.pi, -0.75 * np.pi], 1.0)
    elif pulse_name == "phase_jump_triplet":
        pulse.field = _build_multi_jump_field(
            ft,
            65.0,
            [-55.0, 0.0, 55.0],
            [0.5 * np.pi, -np.pi, 0.5 * np.pi],
            1.6,
        )
    elif pulse_name in RANDOM_MIXTURE_PULSES:
        index = int(pulse_name.rsplit("_", 1)[-1])
        pulse.field = _random_gaussian_mixture_field_from_seed(ft, index=index)
    else:
        raise ValueError(f"Unknown pulse: {pulse_name}")
    return pulse


def _sample_random_mixture_params(rng: np.random.Generator, ft=None) -> dict[str, Any]:
    extent = 320.0 if ft is None else float(np.max(np.abs(_time_fs(ft))))
    components = []
    for _ in range(int(rng.integers(2, 6))):
        components.append(
            {
                "mu_fs": float(rng.uniform(-0.42 * extent, 0.42 * extent)),
                "sigma_fs": float(rng.uniform(0.035 * extent, 0.12 * extent)),
                "weight": float(rng.uniform(0.35, 1.15)),
            }
        )
    phase_kind = str(rng.choice(["sine", "step", "poly_sine", "tanh"]))
    params: dict[str, Any] = {"components": components, "phase_kind": phase_kind}
    if phase_kind == "sine":
        params.update(
            cycles=float(rng.uniform(1.0, 7.0)),
            depth_rad=float(rng.uniform(0.4, 3.0)),
            phase_offset=float(rng.uniform(-np.pi, np.pi)),
        )
    elif phase_kind == "step":
        n_jumps = int(rng.integers(1, 4))
        params.update(
            jump_locations_fs=sorted(float(v) for v in rng.uniform(-0.35 * extent, 0.35 * extent, size=n_jumps)),
            jump_values=[float(v) for v in rng.uniform(-1.25 * np.pi, 1.25 * np.pi, size=n_jumps)],
        )
    elif phase_kind == "poly_sine":
        params.update(
            quadratic=float(rng.uniform(-2.2, 2.2)),
            cubic=float(rng.uniform(-1.2, 1.2)),
            cycles=float(rng.uniform(1.0, 6.0)),
            depth_rad=float(rng.uniform(0.3, 2.2)),
        )
    else:
        params.update(
            tanh_center_fs=float(rng.uniform(-0.2 * extent, 0.2 * extent)),
            tanh_width_fs=float(rng.uniform(0.03 * extent, 0.12 * extent)),
            tanh_scale=float(rng.uniform(0.5, 2.4)),
            cycles=float(rng.uniform(0.8, 4.0)),
            depth_rad=float(rng.uniform(0.2, 1.4)),
        )
    return params


def _sample_smooth_terms(
    rng: np.random.Generator,
    *,
    count_min: int,
    count_max: int,
    amp_range: tuple[float, float],
    cycles_range: tuple[float, float],
) -> list[dict[str, float]]:
    terms = []
    for _ in range(int(rng.integers(count_min, count_max + 1))):
        terms.append(
            {
                "amp": float(rng.uniform(*amp_range)),
                "cycles": float(rng.uniform(*cycles_range)),
                "phase": float(rng.uniform(-np.pi, np.pi)),
            }
        )
    return terms


def sample_pulse_specs(
    count: int,
    *,
    seed: int = 42,
    prefix: str = "pulse",
    family_weights: Mapping[str, float] | None = None,
) -> list[PulseSpec]:
    """Sample a deterministic manifest of varied synthetic pulse specs."""

    rng = np.random.default_rng(int(seed))
    weights = dict(family_weights or FAMILY_WEIGHTS)
    families = list(weights)
    probs = np.asarray([weights[name] for name in families], dtype=np.float64)
    probs = probs / float(np.sum(probs))
    specs: list[PulseSpec] = []

    for index in range(int(count)):
        family = str(rng.choice(families, p=probs))
        pulse_seed = int(rng.integers(0, 2**31 - 1))
        local_rng = np.random.default_rng(pulse_seed)
        params = sample_family_params(family, local_rng)
        base_uid = f"{prefix}_{index:05d}_{family}"
        uid = f"{sanitize_uid(base_uid)}_{spec_digest({'family': family, 'params': params, 'seed': pulse_seed})[:8]}"
        specs.append(
            PulseSpec(
                uid=uid,
                family=family,
                params=params,
                seed=pulse_seed,
                label=f"{family} synthetic pulse",
            )
        )
    return specs


def sample_family_params(family: str, rng: np.random.Generator) -> dict[str, Any]:
    if family == "gaussian":
        return {
            "fwhm_fs": float(rng.uniform(25.0, 110.0)),
            "gdd_fs2": float(rng.uniform(-4500.0, 4500.0)),
            "tod_fs3": float(rng.uniform(-25000.0, 25000.0)),
            "phase_scale": float(rng.uniform(0.0, 1.2)),
        }
    if family == "sech":
        return {
            "fwhm_fs": float(rng.uniform(25.0, 95.0)),
            "chirp": float(rng.uniform(-2.0, 2.0)),
        }
    if family == "super_gaussian":
        return {
            "fwhm_fs": float(rng.uniform(45.0, 130.0)),
            "order": int(rng.integers(3, 9)),
            "chirp": float(rng.uniform(-1.6, 1.6)),
        }
    if family == "double_pulse":
        return {
            "fwhm_fs": float(rng.uniform(18.0, 70.0)),
            "separation_fs": float(rng.uniform(30.0, 190.0)),
            "ratio": float(rng.uniform(0.25, 1.05)),
            "rel_phase": float(rng.uniform(-np.pi, np.pi)),
            "chirp": float(rng.uniform(-1.0, 1.0)),
        }
    if family == "pulse_train":
        pulses = []
        for _ in range(int(rng.integers(3, 7))):
            pulses.append(
                {
                    "center_fs": float(rng.uniform(-145.0, 145.0)),
                    "fwhm_fs": float(rng.uniform(14.0, 58.0)),
                    "weight": float(rng.uniform(0.18, 1.0)),
                    "phase": float(rng.uniform(-np.pi, np.pi)),
                }
            )
        return {
            "pulses": sorted(pulses, key=lambda item: item["center_fs"]),
            "chirp": float(rng.uniform(-1.4, 1.4)),
        }
    if family == "phase_step":
        n_jumps = int(rng.integers(1, 4))
        return {
            "fwhm_fs": float(rng.uniform(35.0, 110.0)),
            "jump_locations_fs": sorted(float(v) for v in rng.uniform(-70.0, 70.0, size=n_jumps)),
            "jump_values": [float(v) for v in rng.uniform(-1.5 * np.pi, 1.5 * np.pi, size=n_jumps)],
            "ramp_cycles": float(rng.uniform(0.0, 2.0)),
        }
    if family == "sinusoidal_phase":
        return {
            "fwhm_fs": float(rng.uniform(35.0, 110.0)),
            "cycles": float(rng.uniform(0.8, 7.5)),
            "depth_rad": float(rng.uniform(0.3, 3.5)),
            "quadratic_scale": float(rng.uniform(-2.2, 2.2)),
        }
    if family == "biexponential":
        return {
            "tau_fast_fs": float(rng.uniform(8.0, 28.0)),
            "tau_slow_fs": float(rng.uniform(55.0, 130.0)),
            "phase_scale": float(rng.uniform(0.6, 3.2)),
            "skew": float(rng.uniform(0.25, 0.95)),
        }
    if family == "random_mixture":
        return _sample_random_mixture_params(rng)
    if family == "random_spectral":
        components = []
        for _ in range(int(rng.integers(2, 6))):
            components.append(
                {
                    "center": float(rng.uniform(-0.45, 0.45)),
                    "width": float(rng.uniform(0.05, 0.28)),
                    "weight": float(rng.uniform(0.25, 1.2)),
                }
            )
        return {
            "components": components,
            "amp_terms": _sample_smooth_terms(
                rng,
                count_min=1,
                count_max=4,
                amp_range=(-0.25, 0.25),
                cycles_range=(1.0, 7.0),
            ),
            "phase_terms": _sample_smooth_terms(
                rng,
                count_min=1,
                count_max=5,
                amp_range=(-1.2, 1.2),
                cycles_range=(0.7, 8.0),
            ),
            "gdd_norm": float(rng.uniform(-2.5, 2.5)),
            "tod_norm": float(rng.uniform(-1.4, 1.4)),
            "fod_norm": float(rng.uniform(-0.8, 0.8)),
        }
    if family == "spectral_hole":
        return {
            "fwhm_fs": float(rng.uniform(30.0, 95.0)),
            "gdd_fs2": float(rng.uniform(-3000.0, 3000.0)),
            "hole_center_thz": float(rng.uniform(-10.0, 10.0)),
            "hole_width_thz": float(rng.uniform(1.0, 6.0)),
            "hole_depth": float(rng.uniform(0.25, 0.95)),
        }
    if family == "spm":
        return {
            "fwhm_fs": float(rng.uniform(30.0, 95.0)),
            "b_integral": float(rng.uniform(0.5, 8.5)),
            "pre_chirp": float(rng.uniform(-1.5, 1.5)),
        }
    if family == "sinc":
        return {
            "bandwidth_thz": float(rng.uniform(7.0, 38.0)),
            "gdd_fs2": float(rng.uniform(-1800.0, 1800.0)),
        }
    if family == "asymmetric_tail":
        return {
            "center_fs": float(rng.uniform(-30.0, 30.0)),
            "left_fwhm_fs": float(rng.uniform(18.0, 75.0)),
            "right_fwhm_fs": float(rng.uniform(35.0, 150.0)),
            "tail_side": int(rng.choice([-1, 1])),
            "tail_tau_fs": float(rng.uniform(30.0, 160.0)),
            "tail_weight": float(rng.uniform(0.05, 0.45)),
            "chirp": float(rng.uniform(-2.0, 2.0)),
            "cubic_phase": float(rng.uniform(-1.0, 1.0)),
        }
    if family == "lorentzian":
        return {
            "width_thz": float(rng.uniform(5.0, 28.0)),
            "order": int(rng.integers(1, 4)),
            "gdd_fs2": float(rng.uniform(-2800.0, 2800.0)),
            "tod_fs3": float(rng.uniform(-16000.0, 16000.0)),
        }
    if family == "airy":
        return {
            "fwhm_fs": float(rng.uniform(25.0, 90.0)),
            "cubic_phase": float(rng.uniform(-7.0, 7.0)),
            "quadratic_phase": float(rng.uniform(-1.8, 1.8)),
        }
    if family == "spectral_comb":
        return {
            "n_lines": int(rng.integers(3, 9)),
            "spacing_thz": float(rng.uniform(2.0, 9.0)),
            "line_width_thz": float(rng.uniform(0.5, 3.0)),
            "envelope_width_thz": float(rng.uniform(9.0, 32.0)),
            "phase_jitter": float(rng.uniform(0.0, np.pi)),
            "gdd_fs2": float(rng.uniform(-2200.0, 2200.0)),
        }
    raise ValueError(f"Unknown pulse family: {family}")


def _make_sampled_pulse(spec: PulseSpec, ft, wl0: float) -> Pulse:
    pulse = Pulse(ft, wl0)
    p = spec.params
    t_s = ft.t
    t_fs = _time_fs(ft)
    extent = np.max(np.abs(t_fs)) + 1e-12
    omega_grid = ft.w
    family = spec.family

    if family == "gaussian":
        fwhm = float(p["fwhm_fs"]) * 1e-15
        sigma_t0 = fwhm / (2.0 * np.sqrt(np.log(2.0)))
        dw_spec = 1.0 / sigma_t0
        gdd = float(p["gdd_fs2"]) * 1e-30
        tod = float(p["tod_fs3"]) * 1e-45
        phase = 0.5 * gdd * omega_grid**2 + tod / 6.0 * omega_grid**3
        if float(p.get("phase_scale", 0.0)) > 0.0:
            phase = phase + float(p["phase_scale"]) * np.sin(2.0 * np.pi * omega_grid / (np.max(np.abs(omega_grid)) + 1e-12))
        pulse.spectrum = np.exp(-0.5 * (omega_grid / dw_spec) ** 2) * np.exp(1j * phase)
    elif family == "sech":
        fwhm_t = float(p["fwhm_fs"]) * 1e-15
        t0 = fwhm_t / (2.0 * np.log(1.0 + np.sqrt(2.0)))
        phase = float(p.get("chirp", 0.0)) * (t_fs / extent) ** 2
        pulse.field = (1.0 / np.cosh(t_s / t0)) * np.exp(1j * phase)
    elif family == "super_gaussian":
        order = int(p["order"])
        fwhm_t = float(p["fwhm_fs"]) * 1e-15
        sigma_t = fwhm_t / (2.0 * np.log(2.0) ** (1.0 / (2 * order)))
        amplitude = np.exp(-0.5 * np.abs(t_s / sigma_t) ** (2 * order))
        phase = float(p.get("chirp", 0.0)) * (t_fs / extent) ** 2
        pulse.field = amplitude * np.exp(1j * phase)
    elif family == "double_pulse":
        fwhm_t = float(p["fwhm_fs"]) * 1e-15
        separation = float(p["separation_fs"]) * 1e-15
        sigma_t = fwhm_t / (2.0 * np.sqrt(np.log(2.0)))
        left = np.exp(-0.5 * ((t_s + separation / 2.0) / sigma_t) ** 2)
        right = float(p["ratio"]) * np.exp(1j * float(p["rel_phase"])) * np.exp(
            -0.5 * ((t_s - separation / 2.0) / sigma_t) ** 2
        )
        phase = float(p.get("chirp", 0.0)) * (t_fs / extent) ** 2
        pulse.field = (left + right) * np.exp(1j * phase)
    elif family == "pulse_train":
        field = np.zeros(ft.N, dtype=np.complex128)
        for component in p["pulses"]:
            sigma_t = float(component["fwhm_fs"]) * 1e-15 / (2.0 * np.sqrt(np.log(2.0)))
            center_s = float(component["center_fs"]) * 1e-15
            field = field + float(component["weight"]) * np.exp(1j * float(component["phase"])) * np.exp(
                -0.5 * ((t_s - center_s) / sigma_t) ** 2
            )
        phase = float(p.get("chirp", 0.0)) * (t_fs / extent) ** 2
        pulse.field = field * np.exp(1j * phase)
    elif family == "phase_step":
        pulse.field = _build_multi_jump_field(
            ft,
            fwhm_fs=float(p["fwhm_fs"]),
            jump_locations_fs=list(p["jump_locations_fs"]),
            jump_values=list(p["jump_values"]),
            ramp_cycles=float(p.get("ramp_cycles", 0.0)),
        )
    elif family == "sinusoidal_phase":
        pulse.field = _build_sinusoidal_phase_field(
            ft,
            fwhm_fs=float(p["fwhm_fs"]),
            cycles=float(p["cycles"]),
            depth_rad=float(p["depth_rad"]),
            quadratic_scale=float(p.get("quadratic_scale", 0.0)),
        )
    elif family == "biexponential":
        pulse.field = _build_biexponential_coupled_field(
            ft,
            tau_fast_fs=float(p["tau_fast_fs"]),
            tau_slow_fs=float(p["tau_slow_fs"]),
            phase_scale=float(p["phase_scale"]),
            skew=float(p["skew"]),
        )
    elif family == "random_mixture":
        pulse.field = _random_gaussian_mixture_field(ft, p)
    elif family == "random_spectral":
        w_norm = omega_grid / (np.max(np.abs(omega_grid)) + 1e-12)
        amplitude = np.zeros_like(w_norm, dtype=np.float64)
        for component in p["components"]:
            amplitude = amplitude + float(component["weight"]) * np.exp(
                -0.5 * ((w_norm - float(component["center"])) / float(component["width"])) ** 2
            )
        for term in p.get("amp_terms", []):
            amplitude = amplitude * (
                1.0
                + float(term["amp"])
                * np.sin(2.0 * np.pi * float(term["cycles"]) * w_norm + float(term["phase"]))
            )
        amplitude = np.maximum(amplitude, 0.0)
        amplitude = _normalize_real(amplitude)
        phase = (
            float(p.get("gdd_norm", 0.0)) * w_norm**2
            + float(p.get("tod_norm", 0.0)) * w_norm**3
            + float(p.get("fod_norm", 0.0)) * w_norm**4
        )
        for term in p.get("phase_terms", []):
            phase = phase + float(term["amp"]) * np.sin(
                2.0 * np.pi * float(term["cycles"]) * w_norm + float(term["phase"])
            )
        pulse.spectrum = amplitude * np.exp(1j * phase)
    elif family == "spectral_hole":
        fwhm_t0 = float(p["fwhm_fs"]) * 1e-15
        sigma_t0 = fwhm_t0 / (2.0 * np.sqrt(np.log(2.0)))
        dw_spec = 1.0 / sigma_t0
        hole_center = TWOPI * float(p["hole_center_thz"]) * 1e12
        hole_width = TWOPI * float(p["hole_width_thz"]) * 1e12
        sigma_h = hole_width / (2.0 * np.sqrt(np.log(2.0)))
        envelope = np.exp(-0.5 * (omega_grid / dw_spec) ** 2)
        notch = 1.0 - float(p["hole_depth"]) * np.exp(-0.5 * ((omega_grid - hole_center) / sigma_h) ** 2)
        gdd = float(p["gdd_fs2"]) * 1e-30
        pulse.spectrum = envelope * np.maximum(notch, 0.0) * np.exp(0.5j * gdd * omega_grid**2)
    elif family == "spm":
        fwhm_t = float(p["fwhm_fs"]) * 1e-15
        sigma_t = fwhm_t / (2.0 * np.sqrt(np.log(2.0)))
        envelope = np.exp(-0.5 * (t_s / sigma_t) ** 2)
        phase = float(p["b_integral"]) * envelope**2 + float(p.get("pre_chirp", 0.0)) * (t_fs / extent) ** 2
        pulse.field = envelope * np.exp(1j * phase)
    elif family == "sinc":
        bandwidth = TWOPI * float(p["bandwidth_thz"]) * 1e12
        spectrum = np.zeros(ft.N, dtype=complex)
        spectrum[np.abs(omega_grid) <= bandwidth / 2.0] = 1.0
        gdd = float(p.get("gdd_fs2", 0.0)) * 1e-30
        pulse.spectrum = spectrum * np.exp(0.5j * gdd * omega_grid**2)
    elif family == "asymmetric_tail":
        center = float(p["center_fs"])
        left_sigma = float(p["left_fwhm_fs"]) / (2.0 * np.sqrt(np.log(2.0)))
        right_sigma = float(p["right_fwhm_fs"]) / (2.0 * np.sqrt(np.log(2.0)))
        sigma = np.where(t_fs < center, left_sigma, right_sigma)
        amplitude = np.exp(-0.5 * ((t_fs - center) / sigma) ** 2)
        side = int(p["tail_side"])
        tail_distance = side * (t_fs - center)
        tail = np.where(
            tail_distance > 0.0,
            np.exp(-tail_distance / float(p["tail_tau_fs"])),
            0.0,
        )
        amplitude = _normalize_real(amplitude + float(p["tail_weight"]) * tail)
        t_norm = (t_fs - center) / extent
        phase = float(p["chirp"]) * t_norm**2 + float(p["cubic_phase"]) * t_norm**3
        pulse.field = amplitude * np.exp(1j * phase)
    elif family == "lorentzian":
        width = TWOPI * float(p["width_thz"]) * 1e12
        order = int(p["order"])
        amplitude = 1.0 / (1.0 + (omega_grid / (width + 1e-30)) ** 2) ** (0.5 * order)
        gdd = float(p["gdd_fs2"]) * 1e-30
        tod = float(p["tod_fs3"]) * 1e-45
        phase = 0.5 * gdd * omega_grid**2 + tod / 6.0 * omega_grid**3
        pulse.spectrum = amplitude * np.exp(1j * phase)
    elif family == "airy":
        fwhm = float(p["fwhm_fs"]) * 1e-15
        sigma_t0 = fwhm / (2.0 * np.sqrt(np.log(2.0)))
        dw_spec = 1.0 / sigma_t0
        w_norm = omega_grid / (np.max(np.abs(omega_grid)) + 1e-12)
        amplitude = np.exp(-0.5 * (omega_grid / dw_spec) ** 2)
        phase = float(p["quadratic_phase"]) * w_norm**2 + float(p["cubic_phase"]) * w_norm**3
        pulse.spectrum = amplitude * np.exp(1j * phase)
    elif family == "spectral_comb":
        n_lines = int(p["n_lines"])
        spacing = TWOPI * float(p["spacing_thz"]) * 1e12
        line_width = TWOPI * float(p["line_width_thz"]) * 1e12
        envelope_width = TWOPI * float(p["envelope_width_thz"]) * 1e12
        centers = (np.arange(n_lines) - 0.5 * (n_lines - 1)) * spacing
        amplitude = np.zeros(ft.N, dtype=np.float64)
        phase = np.zeros(ft.N, dtype=np.float64)
        rng_seed = spec.seed if spec.seed is not None else int(spec_digest(spec.to_dict()), 16)
        rng = np.random.default_rng(rng_seed)
        for center in centers:
            line = np.exp(-0.5 * ((omega_grid - center) / line_width) ** 2)
            weight = np.exp(-0.5 * (center / envelope_width) ** 2)
            line_phase = rng.uniform(-float(p["phase_jitter"]), float(p["phase_jitter"]))
            amplitude = amplitude + weight * line
            phase = phase + line_phase * line
        gdd = float(p["gdd_fs2"]) * 1e-30
        pulse.spectrum = amplitude * np.exp(1j * (phase + 0.5 * gdd * omega_grid**2))
    else:
        raise ValueError(f"Unknown sampled pulse family: {family}")
    return pulse


def make_pypret_pulse(pulse_like: PulseLike, ft, wl0: float = WL0_DEFAULT) -> Pulse:
    """Build a :class:`pypret.Pulse` from a named pulse or stochastic spec."""

    spec = pulse_like_to_spec(pulse_like)
    if spec.family == "named":
        return _make_named_pulse(str(spec.params["name"]), ft, wl0)
    return _make_sampled_pulse(spec, ft, wl0)
