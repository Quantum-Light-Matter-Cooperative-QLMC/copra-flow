"""Reuse identical TG-FROG calculations during PnP-COPRA retrieval."""

import numpy as np

from pypret.pnps import FROG


class CachedTGFROG(FROG):
    """Retain the scalar-delay FFTs and cache repeated fields and traces."""

    _supported_processes = None  # Keep this class out of pyPret's factory registry.
    _previous_spectrum = None

    def calculate(self, spectrum, parameter):
        delays = np.atleast_1d(parameter)
        full_trace = delays.size > 1
        # The next COPRA iteration can reuse the last accepted PnP trace.
        if (
            full_trace
            and np.array_equal(spectrum, self._previous_spectrum)
            and np.array_equal(delays, self.parameter)
        ):
            return self.Tmn
        self._undelayed = self.ft.backward(spectrum)
        self._undelayed_squared = self._undelayed * self._undelayed
        trace = super().calculate(spectrum, parameter)
        self._previous_spectrum = spectrum.copy() if full_trace else None
        return trace

    def _calculate(self, spectrum, parameter):
        delay, shifted, field, signal, trace = self._get_tmp(parameter)
        self.ft.backward(delay * spectrum, out=shifted)
        field[:] = self._undelayed
        signal[:] = shifted.conj() * self._undelayed_squared
        trace[:] = self.measure(signal)
        return trace, signal


from pypret import lib


def transform_rows(data, phase_in, phase_out, forward):
    """Use the same scalar DUCC FFT as scipy.fftpack for each delay."""
    try:
        from scipy.fft._duccfft.pyduccfft import c2c
    except ImportError as exc:
        raise ImportError("The scalar backend needs SciPy DUCC internals; use --backend cached or stock.") from exc
    transformed = phase_in * data
    normalization = 0 if forward else 2
    for row in transformed:
        c2c(row, (-1,), forward, normalization, row, 1)
    return phase_out * transformed


class ScalarFftTGFROG(CachedTGFROG):
    """Keep pyPret's equations and FFT rounding while reducing Python overhead."""

    _delays = None

    def calculate(self, spectrum, parameter):
        delays = np.atleast_1d(parameter)
        if delays.size == 1:
            return super().calculate(spectrum, parameter)
        if np.array_equal(spectrum, self._previous_spectrum) and np.array_equal(delays, self.parameter):
            return self.Tmn
        if not np.array_equal(delays, self._delays):
            self._delays = delays.copy()
            self._phase = np.stack([self._get_tmp(delay)[0] for delay in delays])
        ft = self.ft
        self._shifted = transform_rows(self._phase * spectrum, ft._br, ft._bs, True)
        self._field = ft.backward(spectrum)
        self.Smk = self._shifted.conj() * (self._field * self._field)
        self.Tmn = lib.abs2(transform_rows(self.Smk, ft._fs, ft._fr, False))
        self.parameter = delays
        self.spectrum = spectrum
        self._previous_spectrum = spectrum.copy()
        return self.Tmn

    def gradient(self, signal, parameter):
        if np.size(parameter) == 1:
            return super().gradient(signal, parameter)
        if not np.array_equal(parameter, self.parameter):
            raise ValueError("The TG-FROG gradient requires the current simulated trace.")
        ft = self.ft
        difference = signal - self.Smk
        field = self._field
        first = transform_rows(difference.conj() * field * field, ft._fs, ft._fr, False)
        second = transform_rows(difference * field.conj() * self._shifted, ft._fs, ft._fr, False)
        gradient = self._phase.conj() * first + 2 * second
        gradient *= -2.0 * lib.twopi * ft.dw / ft.dt
        return gradient
