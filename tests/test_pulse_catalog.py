from __future__ import annotations

import numpy as np

from pnp_copra.pulse_catalog import FAMILY_WEIGHTS, PulseSpec, sample_family_params, sample_pulse_specs
from pnp_copra.retrieval import make_context


def test_sample_pulse_specs_are_deterministic() -> None:
    a = [spec.to_dict() for spec in sample_pulse_specs(5, seed=123)]
    b = [spec.to_dict() for spec in sample_pulse_specs(5, seed=123)]
    assert a == b
    assert len({spec["uid"] for spec in a}) == 5


def test_make_context_for_sampled_pulse_smoke() -> None:
    spec = sample_pulse_specs(1, seed=7)[0]
    context = make_context(spec, seed=11, noise=0.0, grid_n=64, grid_dt=5e-15)
    assert context.measurement.data.shape == (64, 64)
    assert np.isfinite(context.measurement.data).all()


def test_all_stochastic_families_build_traces() -> None:
    for index, family in enumerate(FAMILY_WEIGHTS):
        params = sample_family_params(family, np.random.default_rng(1000 + index))
        spec = PulseSpec(uid=f"test_{family}", family=family, params=params, seed=2000 + index)
        context = make_context(spec, seed=3000 + index, noise=0.0, grid_n=32, grid_dt=8e-15)
        assert context.measurement.data.shape == (32, 32), family
        assert np.isfinite(context.measurement.data).all(), family
        assert float(np.max(context.measurement.data)) <= 1.0 + 1e-12, family
