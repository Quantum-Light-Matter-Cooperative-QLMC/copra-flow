#!/usr/bin/env python3
"""Exercise the installed CLI end to end using generated data and real training.

Run from the repository root after installation. No report artifacts, network,
experimental measurements, or parent research checkout are required.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("verification/smoke"))
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    data, run, predictions = out / "data", out / "run", out / "predictions"
    steps = []
    environment = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", MPLBACKEND="Agg")
    def command(name, *arguments):
        start = time.monotonic()
        result = subprocess.run([sys.executable, "-m", "pnp_copra.cli", *map(str, arguments)],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=environment)
        (out / f"{name}.log").write_text(result.stdout)
        if result.returncode:
            raise RuntimeError(f"{name} failed; see {out / (name + '.log')}\n{result.stdout[-3000:]}")
        steps.append({"task": name, "status": "passed", "seconds": round(time.monotonic()-start, 2)})
        print(f"PASS {name}", flush=True)
    command("manifest", "manifest", "--count", 4, "--out-dir", data)
    command("generate", "generate", "--manifest", data / "manifest.json", "--out-dir", data,
            "--maxiter", 4, "--workers", 2, "--hard-negatives-per-pulse", 2)
    command("resume-generation", "generate", "--manifest", data / "manifest.json", "--out-dir", data,
            "--maxiter", 4, "--workers", 2, "--hard-negatives-per-pulse", 2)
    command("status", "status", "--out-dir", data)
    command("inspect", "inspect", "--data-dir", data)
    command("splits", "splits", "--data-dir", data, "--out", data / "splits.json", "--val-fraction", 0.25)
    command("train", "train", "--data-dir", data, "--split-file", data / "splits.json",
            "--out-dir", run, "--epochs", 2, "--batch-size", 4, "--workers", 2, "--device", "cpu",
            "--hidden-channels", 16, "--residual-blocks", 2, "--condition-dim", 32,
            "--time-embedding-dim", 16, "--trace-base-channels", 4, "--trace-layers", 2)
    checkpoint = run / "checkpoints/best.pt"
    shared = ["--checkpoint", checkpoint, "--data-dir", data, "--workers", 2, "--device", "cpu", "--steps", 4]
    command("evaluate", "evaluate", *shared, "--source-kind", "copra_final", "--trace-metrics",
            "--out-json", out / "evaluation.json")
    command("save-predictions", "save-predictions", *shared, "--out-dir", predictions)
    command("score-predictions", "evaluate-predictions", "--predictions-dir", predictions,
            "--trace-metrics", "--out-json", out / "saved_evaluation.json")
    command("visualize", "visualize", "--predictions-dir", predictions, "--out-dir", out / "figures", "--workers", 2)
    command("example", "example", "--out", out / "measurement.npz")
    for method in ("euler", "rk4"):
        command(f"infer-{method}", "infer", "--input", out / "measurement.npz", "--checkpoint", checkpoint,
                "--out-dir", out / f"inference-{method}", "--device", "cpu", "--steps", 4,
                "--maxiter", 4, "--method", method)
    # Exercise the supplied-COPRA route using the field just retrieved above.
    import numpy as np
    with np.load(out / "inference-euler/prediction.npz", allow_pickle=False) as arrays:
        np.save(out / "source.npy", arrays["source_field"])
    command("infer-supplied-source", "infer", "--input", out / "measurement.npz", "--checkpoint", checkpoint,
            "--source-field", out / "source.npy", "--out-dir", out / "inference-supplied", "--device", "cpu", "--steps", 4)
    manifest = out / "experimental_example.json"
    manifest.write_text(json.dumps({"records": [{"uid": "synthetic_example", "path": str(out / "measurement.npz")}]}))
    command("experimental-interface", "benchmark-experimental", "--manifest", manifest,
            "--checkpoint", checkpoint, "--out-dir", out / "experimental_interface",
            "--maxiter", 4, "--steps", 4, "--seeds", 11, 29, "--device", "cpu")
    a = json.loads((out / "evaluation.json").read_text())
    b = json.loads((out / "saved_evaluation.json").read_text())
    if abs(a["predicted_pulse_mae"]["mean"]-b["predicted_pulse_mae"]["mean"]) > 1e-6:
        raise AssertionError("Direct and persisted prediction metrics disagree.")
    (out / "summary.json").write_text(json.dumps({"steps": steps, "status": "passed",
        "note": "Workflow verification with a small model; not scientific model validation."}, indent=2)+"\n")
    print(f"Verification passed: {out / 'summary.json'}")


if __name__ == "__main__":
    main()
