"""Complex-valued pulse helpers for real-valued neural networks."""

from __future__ import annotations

import numpy as np


EPS = 1e-12


def complex_to_channels(values: np.ndarray) -> np.ndarray:
    """Represent a complex field as real-valued channels.

    A field shaped ``(N,)`` becomes ``(2, N)``. A batch shaped ``(..., N)``
    becomes ``(..., 2, N)``.
    """

    array = np.asarray(values)
    if not np.iscomplexobj(array):
        raise TypeError("complex_to_channels expects a complex-valued array.")
    return np.stack((array.real, array.imag), axis=-2).astype(np.float32, copy=False)


def channels_to_complex(values: np.ndarray) -> np.ndarray:
    """Convert ``(..., 2, N)`` real-valued channels back to complex fields."""

    array = np.asarray(values)
    if array.shape[-2] != 2:
        raise ValueError(f"Expected channel dimension of size 2, got shape {array.shape}.")
    return (array[..., 0, :] + 1j * array[..., 1, :]).astype(np.complex64, copy=False)


def rms_normalize_channels(values: np.ndarray, eps: float = EPS) -> tuple[np.ndarray, float]:
    """RMS-normalize a real-channel complex field and return the scale."""

    array = np.asarray(values, dtype=np.float32)
    scale = float(np.sqrt(np.mean(array * array)) + eps)
    return array / scale, scale


def preprocess_trace(
    trace: np.ndarray,
    *,
    transform: str = "sqrt",
    normalize: bool = True,
    eps: float = EPS,
) -> np.ndarray:
    """Prepare a measured FROG trace for the trace encoder.

    The generated traces are already normalized during artifact creation, but
    this function makes inference-time preprocessing explicit and repeatable.
    """

    result = np.asarray(trace, dtype=np.float32)
    if normalize:
        scale = float(np.max(np.abs(result)) + eps)
        result = result / scale

    clipped = np.maximum(result, 0.0)
    transform = transform.lower()
    if transform == "linear":
        result = result
    elif transform == "sqrt":
        result = np.sqrt(clipped)
    elif transform == "log1p":
        result = np.log1p(100.0 * clipped) / np.log1p(100.0)
    else:
        raise ValueError(f"Unknown trace transform: {transform!r}.")

    return result[np.newaxis, ...].astype(np.float32, copy=False)
