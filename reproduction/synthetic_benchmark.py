#!/usr/bin/env python3
"""Evaluate the paper's 1,000-pulse COPRA-final validation benchmark."""
from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path

from pnp_copra.flow_matching.prediction_artifacts import (
    PredictionArtifactConfig, PredictionArtifactEvaluationConfig,
    save_prediction_artifacts, evaluate_prediction_artifacts,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--reuse-predictions", action="store_true",
                        help="Recompute metrics only, after validating saved-run provenance.")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    predictions = args.out_dir / "predictions"
    if args.reuse_predictions:
        manifest = json.loads((predictions / "manifest.json").read_text())
        signature = manifest.get("run_signature", {})
        digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        if signature.get("checkpoint_sha256") != digest or signature.get("source_kind") != "copra_final":
            raise ValueError("Saved predictions do not identify the requested checkpoint/source kind.")
        if signature.get("sampling", {}).get("steps") != 64 or signature.get("sampling", {}).get("method") != "euler":
            raise ValueError("Saved predictions do not use 64-step Euler.")
        expected = set(json.loads((here / "selections/synthetic_split.json").read_text())["val"])
        actual = {p.stem.rsplit("_src", 1)[0] for p in (predictions / "shards").glob("*.npz")}
        if actual != expected:
            raise ValueError("Saved predictions do not cover exactly the validation pulse UIDs.")
    else:
        save_prediction_artifacts(PredictionArtifactConfig(
        checkpoint=args.checkpoint, data_dir=args.data_dir, out_dir=predictions,
        split_file=here / "selections/synthetic_split.json", source_kind="copra_final",
        split="val", sampling_steps=64, sampling_method="euler", device=args.device,
            batch_size=args.batch_size, num_workers=args.workers))
    metrics = evaluate_prediction_artifacts(PredictionArtifactEvaluationConfig(
        predictions_dir=predictions, include_trace_metrics=True, out_json=args.out_dir / "metrics.json"))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
