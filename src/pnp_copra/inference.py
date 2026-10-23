"""Target-free inference from measured TG-FROG traces and optional COPRA fields."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .experimental import (
    GRID_DT_S, SPEED_OF_LIGHT_M_S, GridSpec, _experimental_context,
    prepare_native, run_experimental_copra, run_experimental_flow,
)
from .checkpoints import load_checkpoint


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def save_npz(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def load_measurement(path: Path, checkpoint: Path | None = None) -> dict[str, np.ndarray]:
    """Read only measurement/source keys; targets are never used for inference."""
    grid = GridSpec.from_checkpoint_or_default(checkpoint)
    if path.suffix.lower() == ".txt":
        return prepare_native(path, grid)
    if path.suffix.lower() != ".npz":
        raise ValueError("Input must be a prepared .npz or an axis-embedded .txt measurement.")
    required = ("time_axis", "delay_axis", "freq_axis", "wl0")
    with np.load(path, allow_pickle=False) as handle:
        missing = [key for key in required if key not in handle]
        if missing:
            raise ValueError(f"Measurement is missing {missing}; see docs/data_format.md.")
        trace_key = "measured_trace" if "measured_trace" in handle else "trace_linear"
        if trace_key not in handle:
            raise ValueError("Measurement needs measured_trace (delay x frequency).")
        item = {key: np.asarray(handle[key]).copy() for key in required}
        item["trace_linear"] = np.asarray(handle[trace_key], dtype=np.float64).copy()
        if "source_field" in handle:
            item["source_field"] = np.asarray(handle["source_field"]).copy()
    n = grid.time_axis_s.size
    trace = item["trace_linear"]
    if trace.shape != (n, n) or not np.isfinite(trace).all() or np.any(trace < 0) or trace.max() <= 0:
        raise ValueError(f"Trace must be a finite, nonnegative {n} x {n} delay-frequency array with positive signal.")
    for key, expected in (("time_axis", grid.time_axis_s), ("delay_axis", grid.delay_axis_s)):
        axis = item[key]
        if axis.shape != (n,) or not np.isfinite(axis).all() or not np.allclose(axis, expected, rtol=1e-7, atol=1e-22):
            raise ValueError(f"{key} must match the centered {n}-point, 2.5 fs checkpoint grid.")
    if item["wl0"].size != 1:
        raise ValueError("wl0 must be one carrier wavelength in metres.")
    wl0 = float(item["wl0"].item())
    if not np.isfinite(wl0) or wl0 <= 0:
        raise ValueError("wl0 must be a positive finite wavelength in metres.")
    freq = item["freq_axis"]
    expected = grid.relative_frequency_axis_rad_s
    carrier = 2 * np.pi * SPEED_OF_LIGHT_M_S / wl0
    if freq.shape != (n,) or not np.isfinite(freq).all():
        raise ValueError("freq_axis must contain one finite coordinate per frequency bin.")
    if np.allclose(freq, expected, rtol=1e-7, atol=1):
        pass
    elif np.allclose(freq - carrier, expected, rtol=1e-7, atol=1):
        item["freq_axis"] = freq - carrier
    else:
        raise ValueError("freq_axis must match the checkpoint's relative angular-frequency grid (rad/s).")
    # A prepared measurement is already unit-peak. Normalize external inputs identically.
    item["trace_linear"] /= trace.max()
    return item


def infer(input_path: Path, checkpoint: Path, out_dir: Path, *, source_field: Path | None = None,
          steps: int = 64, method: str = "euler", device: str = "auto", seed: int = 11,
          maxiter: int = 400, alpha: float = 0.25) -> dict:
    """Run inference with no target field or pulse-domain selection criterion."""
    if steps < 1 or maxiter < 1:
        raise ValueError("steps and maxiter must be positive.")
    item = load_measurement(input_path, checkpoint)
    source = item.pop("source_field", None)
    source_origin = "input_source_field" if source is not None else "copra"
    if source_field is not None:
        if source_field.suffix == ".npy":
            source = np.load(source_field, allow_pickle=False)
        else:
            with np.load(source_field, allow_pickle=False) as arrays:
                source = np.asarray(arrays["field"])
        source_origin = "supplied_source_field"
    copra = None
    if source is None:
        copra = run_experimental_copra(item, seeds=(seed,), maxiter=maxiter, alpha=alpha)
        source = copra["field"]
    source = np.asarray(source, dtype=np.complex128)
    n = item["time_axis"].size
    if source.shape != (n,) or not np.isfinite(source).all() or not np.any(np.abs(source) > 0):
        raise ValueError(f"Source field must be a finite, nonzero complex array of shape ({n},).")
    result = run_experimental_flow(item, source, checkpoint, sampling_steps=steps,
                                   sampling_method=method, device=device)
    ft, pnps, _, parameter = _experimental_context(item)
    source_trace = np.asarray(pnps.calculate(ft.forward(source), parameter))
    output = out_dir / "prediction.npz"
    arrays = {key: item[key] for key in ("time_axis", "delay_axis", "freq_axis", "wl0")}
    arrays.update(measured_trace=item["trace_linear"], source_field=source,
                  predicted_field=result["field"], source_reconstructed_trace=source_trace,
                  predicted_reconstructed_trace=result["reconstructed_trace"])
    metadata = {
        "schema_version": 1, "input_filename": input_path.name,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "steps": steps, "method": method, "seed": seed, "maxiter": maxiter, "alpha": alpha,
        "source_origin": source_origin, "selected_branch": result["branch"],
        "center_shift_samples": result["center_shift"], "source_rms": result["rms"],
        "metrics": result["metrics"], "field_coordinates": "centered; FROG ambiguities remain",
        "target_used": False,
    }
    if copra is not None:
        metadata["copra_metrics"] = copra["metrics"]
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False))
    save_npz(output, arrays)
    write_json(out_dir / "metrics.json", metadata)
    return {"prediction": str(output), **metadata}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Apply native TG-FROG trace preprocessing.")
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--checkpoint", type=Path)
    run = sub.add_parser("infer", help="Run target-free COPRA Flow inference.")
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--out-dir", type=Path, required=True)
    run.add_argument("--source-field", type=Path)
    run.add_argument("--steps", type=int, default=64)
    run.add_argument("--method", choices=["euler", "rk4"], default="euler")
    run.add_argument("--device", default="auto")
    run.add_argument("--seed", type=int, default=11)
    run.add_argument("--maxiter", type=int, default=400)
    run.add_argument("--alpha", type=float, default=0.25)
    example = sub.add_parser("example", help="Simulate a Gaussian TG-FROG input without a target file.")
    example.add_argument("--out", type=Path, required=True)
    example.add_argument("--noise", type=float, default=0.01)
    example.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare_native(args.input, GridSpec.from_checkpoint_or_default(args.checkpoint))
        save_npz(args.out, result)
        print(json.dumps({"prepared": str(args.out), "shape": list(result["trace_linear"].shape)}))
    elif args.command == "example":
        from pypret import FourierTransform, PNPS, Pulse
        if args.noise < 0:
            raise ValueError("noise must be nonnegative.")
        ft = FourierTransform(256, dt=GRID_DT_S)
        pulse = Pulse(ft, 800e-9)
        pulse.field = np.exp(-0.5*(ft.t/30e-15)**2) * np.exp(0.2j*np.sin(ft.t/20e-15))
        pnps = PNPS(pulse, "frog", "tg")
        trace = np.asarray(pnps.calculate(pulse.spectrum, ft.t)).copy()
        trace /= trace.max()
        trace = np.maximum(trace + args.noise*np.random.default_rng(args.seed).normal(size=trace.shape), 0)
        trace /= trace.max()
        grid = GridSpec.from_checkpoint_or_default()
        save_npz(args.out, {"measured_trace": trace, "time_axis": ft.t, "delay_axis": ft.t,
                           "freq_axis": grid.relative_frequency_axis_rad_s, "wl0": np.asarray(800e-9)})
        print(json.dumps({"example": str(args.out)}))
    else:
        kwargs = vars(args).copy()
        kwargs.pop("command")
        kwargs["input_path"] = kwargs.pop("input")
        print(json.dumps(infer(**kwargs), indent=2))
    return 0
