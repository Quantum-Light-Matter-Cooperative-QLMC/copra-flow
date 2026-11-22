#!/usr/bin/env python3
"""Reproduce the in-loop PnP-COPRA tolerance pilot and validation benchmark.

The solver, denoiser, support estimate, and pyPret output rule are unchanged.
Only accept_tol is varied. Paths and outputs are independent of the original
research workspace. See docs/pnp_copra.md for the archived experiment.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import scipy

from pnp_copra.artifacts import stable_int_hash
from pnp_copra.flow_matching.metrics import align_field_to_target
from pnp_copra.pulse_catalog import PulseSpec
from pnp_copra import retrieval
from pnp_copra.retrieval import (
    AbhiPnPCOPRAClassic, compute_trace_ssim,
    estimate_time_window_from_measured_trace, make_context, make_initial_guess,
)
from pnp_copra.trace_metrics import trace_nrmse_loss
from pnp_copra_backend import CachedTGFROG, ScalarFftTGFROG

HERE = Path(__file__).resolve().parent
FINAL_SSIM_FLOOR = 0.30  # Reporting check, never a post-retrieval rejection rule.

def read_json(path: Path):
    return json.loads(path.read_text())

def write_json(path: Path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)

@dataclass(frozen=True)
class PulseRecord:
    uid: str
    spec: dict
    prediction: str

class LoggedPnPCOPRA(AbhiPnPCOPRAClassic):
    """Record backtracking decisions and the last iterate without changing retrieval."""

    def __init__(self, *args, equivalent_tolerances=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.global_steps = 0
        self.accepted_steps = 0
        self.full_mix_steps = 0
        self.max_relative_step_drop = 0.0
        self.equivalent_tolerances = list(equivalent_tolerances)

    def _trace_ssim_given_en(self, en_candidate):
        score = super()._trace_ssim_given_en(en_candidate)
        self._scores.append(score)
        return score

    def _retrieve_step(self, iteration, spectrum):
        self._scores = []
        error, self.last_spectrum = super()._retrieve_step(iteration, spectrum)
        if self._scores:
            self.global_steps += 1
            before, after = self._scores[0], self._scores[-1]
            if len(self._scores) > 1 and after >= before * (1 - self._tol):
                self.accepted_steps += 1
                self.full_mix_steps += int(len(self._scores) == 2)
                if before > 0:
                    self.max_relative_step_drop = max(self.max_relative_step_drop, 1 - after / before)
            # Two tolerances produce the same next iterate when every tested
            # mix gets the same verdict. Intersect this set across iterations.
            trials = self._scores[1:]
            verdicts = [score >= before * (1 - self._tol) for score in trials]
            self.equivalent_tolerances = [
                tol for tol in self.equivalent_tolerances
                if [score >= before * (1 - tol) for score in trials] == verdicts
            ]
        return error, self.last_spectrum

def prepare_pulse(record: PulseRecord, config: dict, data_dir: Path):
    seed = config["seed"] + stable_int_hash(record.uid) % 1_000_000
    context = make_context(
        PulseSpec.from_dict(record.spec), seed=seed, noise=config["noise"],
        grid_n=config["grid_n"], grid_dt=config["grid_dt"],
    )
    with np.load(data_dir / "shards" / f"{record.uid}.npz", allow_pickle=False) as shard:
        measured = shard["measured_trace"]
    restored_precision = np.array_equal(context.measurement.data.astype(measured.dtype), measured)
    if not restored_precision:
        context.measurement.data = measured.astype(np.float64)
        context.time_window = estimate_time_window_from_measured_trace(
            context.measurement.data, context.parameter, context.ft.t,
        )
    # Start retrieval at the same RNG state used to generate the COPRA dataset.
    initial = make_initial_guess(context.ft)
    rng_state = np.random.get_state()
    return context, initial, rng_state, restored_precision

def score_field(context, spectrum, target):
    field = context.ft.backward(spectrum).astype(np.complex64)
    mae, aligned = align_field_to_target(target, field)
    # Use the same trace orientation and global SSIM as the in-loop gate.
    ssim = compute_trace_ssim(context.pnps, context.measurement, spectrum, context.parameter)
    trace = context.pnps.Tmn
    nrmse = trace_nrmse_loss(context.measurement.data, trace)
    return {"mae": mae, "ssim": ssim, "nrmse": nrmse}, field, aligned

def saved_baselines(prediction):
    target = prediction["target_field"]
    methods = (
        ("copra", "source_field", "source"),
        ("direct_hann", "copra_hann_direct_final_field", "copra_hann_direct_final"),
        ("flow", "predicted_field", None),
    )
    results = {}
    for method, field, prefix in methods:
        mae, _ = align_field_to_target(target, prediction[field])
        results[method] = {"mae": mae}
        if prefix:
            results[method]["ssim"] = float(prediction[f"{prefix}_trace_ssim"])
            results[method]["nrmse"] = float(prediction[f"{prefix}_trace_error"])
    return results

def retrieve(
    context, initial, rng_state, config, tol, equivalent_tolerances=(),
    *, solver_class=LoggedPnPCOPRA, **pnp_options,
):
    np.random.set_state(rng_state)
    solver = solver_class(
        context.pnps, maxiter=config["maxiter"], alpha=config["alpha"],
        time_window=context.time_window, accept_tol=tol, logging=False,
        equivalent_tolerances=equivalent_tolerances, **pnp_options,
    )
    solver.retrieve(context.measurement, initial.copy())
    return solver

def describe(values):
    values = np.asarray(values)
    return {
        "mean": float(values.mean()), "median": float(np.median(values)),
        "p10": float(np.quantile(values, .1)), "p90": float(np.quantile(values, .9)),
        "min": float(values.min()), "max": float(values.max()),
    }

def summarize(rows):
    methods = ("copra", "direct_hann", "flow", "pnp", "last_iterate")
    summary = {
        method: {name: describe([row[method][name] for row in rows]) for name in rows[0][method]}
        for method in methods
    }
    for method in ("pnp", "last_iterate"):
        mae_delta = np.asarray([row[method]["mae"] - row["direct_hann"]["mae"] for row in rows])
        ssim_deficit = np.asarray([1 - row[method]["ssim"] for row in rows])
        summary[method]["mae_minus_direct_hann"] = describe(mae_delta)
        summary[method]["pulses_at_or_below_hann_mae"] = int(np.count_nonzero(mae_delta <= 0))
        summary[method]["ssim_deficit_from_measured"] = describe(ssim_deficit)
        summary[method]["below_final_ssim_floor"] = int(np.count_nonzero(ssim_deficit > 1 - FINAL_SSIM_FLOOR))
    global_steps = sum(row["global_steps"] for row in rows)
    summary["backtracking"] = {
        "global_steps": global_steps,
        "accept_fraction": sum(row["accepted_steps"] for row in rows) / global_steps if global_steps else None,
        "full_mix_fraction": sum(row["full_mix_steps"] for row in rows) / global_steps if global_steps else None,
        "largest_accepted_relative_drop": max(row["max_relative_step_drop"] for row in rows),
    }
    summary["count"] = len(rows)
    summary["exactly_reproduced_trace_count"] = sum(row["restored_trace_precision"] for row in rows)
    return summary

def write_metrics_csv(path: Path, rows):
    flat_rows = [
        {
            **{key: value for key, value in row.items() if not isinstance(value, dict)},
            **{f"{key}_{metric}": x for key, value in row.items() if isinstance(value, dict) for metric, x in value.items()},
        }
        for row in rows
    ]
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generation_config_path(data_dir: Path) -> Path:
    """Support locally generated datasets and the published Hugging Face layout."""
    local = data_dir / "run_config.json"
    return local if local.is_file() else data_dir / "generation_config.json"


def load_records(data_dir: Path, predictions_dir: Path, pilot: bool = False):
    predictions = read_json(predictions_dir / "manifest.json")["records_written_this_run"]
    specs = {item["uid"]: item for item in read_json(data_dir / "manifest.json")["specs"]}
    expected = set(read_json(HERE / "selections/synthetic_split.json")["val"])
    actual = [item["uid"] for item in predictions]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Predictions must cover exactly the published 1,000-pulse validation split.")
    records = {
        item["uid"]: PulseRecord(item["uid"], specs[item["uid"]], item["path"])
        for item in predictions
    }
    selected = read_json(HERE / "selections/pnp_copra_pilot.json")["uids"] if pilot else sorted(expected)
    if len(selected) != len(set(selected)) or not set(selected) <= expected:
        raise ValueError("The pilot must contain unique validation pulse UIDs.")
    return [records[uid] for uid in selected], read_json(generation_config_path(data_dir))


def set_backend(context, backend):
    if backend == "cached":
        context.pnps = CachedTGFROG(context.pulse, "tg")
    elif backend == "scalar":
        # This is the accelerated backend used for the archived full run.
        context.pnps = ScalarFftTGFROG(context.pulse, "tg")


def result_path(out_dir: Path, uid: str, tolerance: float) -> Path:
    return out_dir / "runs" / f"tol_{tolerance:g}" / f"{uid}.npz"


def run_pulse(record, config, data_dir, predictions_dir, out_dir, tolerances, backend):
    context, initial, rng_state, restored = prepare_pulse(record, config, data_dir)
    set_backend(context, backend)
    with np.load(predictions_dir / record.prediction, allow_pickle=False) as saved:
        if not np.array_equal(context.measurement.data.astype(np.float32), saved["measured_trace"]):
            raise ValueError(f"Prediction and dataset traces disagree for {record.uid}")
        target = saved["target_field"]
        baseline = saved_baselines(saved)
    pending = sorted(tolerances, reverse=True)
    while pending:
        tolerance = pending[0]
        start = time.monotonic()
        solver = retrieve(context, initial, rng_state, config, tolerance, pending)
        returned, raw, aligned = score_field(context, solver._result.spectrum, target)
        last, _, _ = score_field(context, solver.last_spectrum, target)
        covered = solver.equivalent_tolerances
        if tolerance not in covered or solver.max_relative_step_drop > min(covered) + 1e-12:
            raise ValueError(f"Backtracking exceeded the requested tolerance for {record.uid}")
        for equivalent in covered:
            row = {
                "uid": record.uid, "family": record.spec["family"],
                "evaluated_tolerance": tolerance, "tolerance": equivalent,
                "initial_mixing": 0.7, "pnp": returned, "last_iterate": last,
                **baseline, "global_steps": solver.global_steps,
                "accepted_steps": solver.accepted_steps, "full_mix_steps": solver.full_mix_steps,
                "max_relative_step_drop": solver.max_relative_step_drop,
                "restored_trace_precision": restored, "runtime_s": time.monotonic() - start,
            }
            path = result_path(out_dir, record.uid, equivalent)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, metrics=json.dumps(row, allow_nan=False),
                                field=raw, field_aligned=aligned)
            temporary.replace(path)
        pending = [tol for tol in pending if tol not in covered]
    return record.uid


def summarize_results(records, tolerances, out_dir, pilot):
    pilot_uids = set(read_json(HERE / "selections/pnp_copra_pilot.json")["uids"])
    summaries, all_rows = [], []
    for tolerance in tolerances:
        rows = []
        for record in records:
            with np.load(result_path(out_dir, record.uid, tolerance), allow_pickle=False) as saved:
                row = json.loads(saved["metrics"].item())
            if row["uid"] != record.uid or row["tolerance"] != tolerance:
                raise ValueError("Saved result has an unexpected pulse UID or tolerance.")
            row["in_pilot"] = record.uid in pilot_uids
            rows.append(row)
        summary = summarize(rows)
        summary.update(tolerance=tolerance, mixing=0.7, initialization="double-hump",
                       alpha=0.25, min_mix=0.05, support_sigma=None)
        summary["count_improved_vs_copra"] = sum(r["pnp"]["mae"] < r["copra"]["mae"] for r in rows)
        outside = [row for row in rows if not row["in_pilot"]]
        if outside:
            summary["outside_pilot"] = summarize(outside)
        summaries.append(summary)
        all_rows.extend(rows)
        print(f"delta={tolerance:g}: mean pulse MAE {summary['pnp']['mae']['mean']:.6f}, "
              f"mean trace SSIM {summary['pnp']['ssim']['mean']:.6f}", flush=True)
    prefix = "pilot" if pilot else "full"
    write_json(out_dir / f"{prefix}_summary.json", {
        "uids": [r.uid for r in records], "settings": summaries,
        "final_ssim_floor": FINAL_SSIM_FLOOR,
    })
    write_metrics_csv(out_dir / f"{prefix}_per_pulse.csv", all_rows)


def verify_stock(records, config, data_dir, backend, tolerance):
    """Require bitwise agreement of returned/last spectra and solver decisions."""
    observations = []
    cases = [records[0], records[-1]] if len(records) > 1 else records
    for record in cases:
        context, initial, rng_state, _ = prepare_pulse(record, config, data_dir)
        stock = retrieve(context, initial, rng_state, config, tolerance)
        set_backend(context, backend)
        candidate = retrieve(context, initial, rng_state, config, tolerance)
        np.testing.assert_array_equal(candidate._result.spectrum, stock._result.spectrum)
        np.testing.assert_array_equal(candidate.last_spectrum, stock.last_spectrum)
        for name in ("global_steps", "accepted_steps", "full_mix_steps", "max_relative_step_drop"):
            if getattr(candidate, name) != getattr(stock, name):
                raise AssertionError(f"Backend changed {name} for {record.uid}")
        observations.append({"uid": record.uid, "tolerance": tolerance,
                             "backend": backend, "exact_stock_spectra_and_decisions": True})
        print(f"Verified stock agreement: {record.uid}", flush=True)
    return observations


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True,
                        help="Predictions from reproduction/synthetic_benchmark.py.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true", help="Use the fixed 50-pulse pilot.")
    parser.add_argument("--tolerances", type=float, nargs="+", default=[0.01])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--backend", choices=("stock", "cached", "scalar"), default="cached")
    parser.add_argument("--verify-only", action="store_true", help="Check two pulses against stock, then exit.")
    parser.add_argument("--summarize-only", action="store_true", help="Summarize completed results without retrieval.")
    parser.add_argument("--limit", type=int, help="Limit to a small smoke run; not a full benchmark.")
    args = parser.parse_args(argv)
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers and limit must be positive")
    if any(not np.isfinite(tol) or not 0 <= tol <= 0.7 for tol in args.tolerances):
        parser.error("tolerances must be finite and between 0 and 0.7")
    args.data_dir, args.predictions_dir = args.data_dir.resolve(), args.predictions_dir.resolve()
    records, config = load_records(args.data_dir, args.predictions_dir, args.pilot)
    if config["maxiter"] != 400 or config["alpha"] != 0.25:
        raise ValueError("The paper comparison uses 400 iterations and COPRA alpha=0.25.")
    if args.limit is not None:
        records = records[:args.limit]
    tolerances = sorted(set(args.tolerances))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.verify_only:
        verification = verify_stock(records, config, args.data_dir, args.backend, tolerances[0])
        write_json(args.out_dir / "solver_verification.json", verification)
        return
    signature = {
        "dataset_manifest_sha256": sha256(args.data_dir / "manifest.json"),
        "prediction_manifest_sha256": sha256(args.predictions_dir / "manifest.json"),
        "dataset_config_sha256": sha256(generation_config_path(args.data_dir)),
        "retrieval_sha256": sha256(Path(retrieval.__file__)),
        "benchmark_sha256": sha256(Path(__file__)),
        "backend_sha256": sha256(HERE / "pnp_copra_backend.py"),
        "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "uids": [r.uid for r in records], "tolerances": tolerances, "backend": args.backend,
    }
    settings = args.out_dir / "run_signature.json"
    if settings.exists() and read_json(settings) != signature:
        raise ValueError("Output directory belongs to different inputs/settings; use a new directory.")
    write_json(settings, signature)
    if not args.summarize_only:
        jobs = [(r, [tol for tol in tolerances if not result_path(args.out_dir, r.uid, tol).exists()])
                for r in records]
        jobs = [(record, values) for record, values in jobs if values]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_pulse, record, config, args.data_dir, args.predictions_dir,
                                   args.out_dir, values, args.backend) for record, values in jobs]
            for count, task in enumerate(as_completed(futures), 1):
                uid = task.result()
                if count == 1 or count % 10 == 0 or count == len(futures):
                    print(f"Completed {count}/{len(futures)} pending pulses: {uid}", flush=True)
    summarize_results(records, tolerances, args.out_dir, args.pilot)


if __name__ == "__main__":
    main()
