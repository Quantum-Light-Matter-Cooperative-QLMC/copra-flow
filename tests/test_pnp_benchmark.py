"""The benchmark accelerators must preserve the original retrieval result."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from pnp_copra.retrieval import AbhiPnPCOPRAClassic, make_context, make_initial_guess
from pnp_copra.pulse_catalog import PulseSpec


REPRODUCTION = Path(__file__).resolve().parents[1] / "reproduction"
sys.path.insert(0, str(REPRODUCTION))
spec = importlib.util.spec_from_file_location("pnp_benchmark", REPRODUCTION / "pnp_copra_benchmark.py")
benchmark = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = benchmark
spec.loader.exec_module(benchmark)


@pytest.mark.parametrize("backend", ["cached", "scalar"])
@pytest.mark.parametrize("tolerance", [0.0, 0.01, 0.7])
def test_acceleration_and_logging_preserve_stock_output(backend, tolerance):
    if backend == "scalar":
        pytest.importorskip("scipy.fft._duccfft.pyduccfft")
    context = make_context(PulseSpec("test_gaussian", "gaussian",
                                    {"fwhm_fs": 40.0, "gdd_fs2": 50.0, "tod_fs3": 0.0}), seed=19,
                           grid_n=32, grid_dt=5e-15, noise=0.01)
    initial = make_initial_guess(context.ft)
    rng_state = np.random.get_state()
    config = {"maxiter": 80, "alpha": 0.25}

    # Compare to the uninstrumented production retriever, including its
    # original best-trace-error output rule.
    stock = AbhiPnPCOPRAClassic(
        context.pnps, maxiter=config["maxiter"], alpha=config["alpha"],
        time_window=context.time_window, accept_tol=tolerance, logging=False,
    )
    stock.retrieve(context.measurement, initial.copy())
    benchmark.set_backend(context, backend)
    candidate = benchmark.retrieve(context, initial, rng_state, config, tolerance)
    assert candidate.global_steps > 0
    assert candidate.max_relative_step_drop <= tolerance + 1e-12
    np.testing.assert_array_equal(candidate._result.spectrum, stock._result.spectrum)


def test_reusing_tolerance_trajectories_preserves_independent_results():
    context = make_context(PulseSpec("test_gaussian", "gaussian",
                                    {"fwhm_fs": 40.0, "gdd_fs2": 50.0, "tod_fs3": 0.0}), seed=19,
                           grid_n=32, grid_dt=5e-15, noise=0.01)
    initial = make_initial_guess(context.ft)
    rng_state = np.random.get_state()
    config = {"maxiter": 80, "alpha": 0.25}
    primary = benchmark.retrieve(context, initial, rng_state, config, 0.7, [0.0, 0.01, 0.1, 0.7])
    assert 0.7 in primary.equivalent_tolerances
    for tolerance in primary.equivalent_tolerances:
        independent = benchmark.retrieve(context, initial, rng_state, config, tolerance)
        np.testing.assert_array_equal(independent._result.spectrum, primary._result.spectrum)
        np.testing.assert_array_equal(independent.last_spectrum, primary.last_spectrum)


@pytest.mark.parametrize("config_name", ["run_config.json", "generation_config.json"])
def test_benchmark_loads_local_and_published_dataset_layouts(tmp_path, monkeypatch, config_name):
    data, predictions, reproduction = [tmp_path / name for name in ("data", "predictions", "reproduction")]
    data.mkdir()
    predictions.mkdir()
    (reproduction / "selections").mkdir(parents=True)
    uids = ["pulse_a", "pulse_b"]
    config = {"seed": 42, "noise": 0.01, "maxiter": 400, "alpha": 0.25}
    (data / config_name).write_text(json.dumps(config))
    (data / "manifest.json").write_text(json.dumps({"specs": [
        {"uid": uid, "family": "gaussian", "params": {}} for uid in uids
    ]}))
    (predictions / "manifest.json").write_text(json.dumps({"records_written_this_run": [
        {"uid": uid, "path": f"shards/{uid}.npz"} for uid in uids
    ]}))
    (reproduction / "selections/synthetic_split.json").write_text(json.dumps({"val": uids}))
    monkeypatch.setattr(benchmark, "HERE", reproduction)
    records, loaded = benchmark.load_records(data, predictions)
    assert [record.uid for record in records] == uids
    assert loaded == config
