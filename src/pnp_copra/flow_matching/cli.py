"""Command-line interface for flow-matching training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pnp_copra.pypret_bootstrap import package_root

from .config import DatasetConfig, FlowLossConfig, ModelConfig, TrainingConfig
from .inspection import dataset_inventory


DEFAULT_DATA_DIR = package_root() / "data" / "flow_matching_10k"
DEFAULT_RUN_DIR = package_root() / "runs" / "flow_matching"


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-mode", choices=["all", "random"], default="all")
    parser.add_argument("--trace-transform", choices=["linear", "sqrt", "log1p"], default="sqrt")
    parser.add_argument("--cache-shards", type=int, default=16)
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--exclude-hard-negatives", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--no-source-weights", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-source-weight", type=float, default=5.0)


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--field-length", type=int, default=256)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--residual-blocks", type=int, default=8)
    parser.add_argument("--condition-dim", type=int, default=256)
    parser.add_argument("--time-embedding-dim", type=int, default=128)
    parser.add_argument("--trace-base-channels", type=int, default=24)
    parser.add_argument("--trace-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--condition-on-source", action=argparse.BooleanOptionalAction, default=False)


def add_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RUN_DIR / "baseline")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--validate-every", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False, dest="compile_model")
    parser.add_argument("--time-eps", type=float, default=1e-4)
    parser.add_argument("--path-noise-std", type=float, default=0.0)


def dataset_config_from_args(args: argparse.Namespace, *, split: str = "train") -> DatasetConfig:
    return DatasetConfig(
        data_dir=args.data_dir.resolve(),
        split=split,
        validation_fraction=float(args.val_fraction),
        seed=int(args.seed),
        source_mode=str(args.source_mode),
        include_hard_negatives=not bool(args.exclude_hard_negatives),
        use_source_weights=not bool(args.no_source_weights),
        max_source_weight=float(args.max_source_weight),
        trace_transform=str(args.trace_transform),
        cache_shards=int(args.cache_shards),
        max_shards=args.max_shards,
        split_file=args.split_file,
    )


def model_config_from_args(args: argparse.Namespace) -> ModelConfig:
    return ModelConfig(
        field_length=int(args.field_length),
        hidden_channels=int(args.hidden_channels),
        residual_blocks=int(args.residual_blocks),
        condition_dim=int(args.condition_dim),
        time_embedding_dim=int(args.time_embedding_dim),
        trace_base_channels=int(args.trace_base_channels),
        trace_layers=int(args.trace_layers),
        dropout=float(args.dropout),
        condition_on_source=bool(args.condition_on_source),
    )


def training_config_from_args(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        out_dir=args.out_dir.resolve(),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.lr),
        weight_decay=float(args.weight_decay),
        num_workers=int(args.workers),
        device=str(args.device),
        seed=int(args.seed),
        grad_clip_norm=float(args.grad_clip_norm),
        amp=bool(args.amp),
        log_every=int(args.log_every),
        validate_every=int(args.validate_every),
        checkpoint_every=int(args.checkpoint_every),
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        compile_model=bool(args.compile_model),
    )


def loss_config_from_args(args: argparse.Namespace) -> FlowLossConfig:
    return FlowLossConfig(
        time_eps=float(args.time_eps),
        path_noise_std=float(args.path_noise_std),
    )


def inspect_command(args: argparse.Namespace) -> int:
    print(json.dumps(dataset_inventory(args.data_dir.resolve()), indent=2, sort_keys=True))
    return 0


def train_command(args: argparse.Namespace) -> int:
    from .trainer import train_flow_matching

    result = train_flow_matching(
        dataset_config=dataset_config_from_args(args),
        model_config=model_config_from_args(args),
        training_config=training_config_from_args(args),
        loss_config=loss_config_from_args(args),
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


def evaluate_command(args: argparse.Namespace) -> int:
    from .evaluate import EvaluationConfig, evaluate_pulse_mae

    result = evaluate_pulse_mae(
        EvaluationConfig(
            checkpoint=args.checkpoint,
            data_dir=args.data_dir,
            split_file=args.split_file,
            split=str(args.split),
            source_kind_filter=args.source_kind,
            batch_size=int(args.batch_size),
            num_workers=int(args.workers),
            device=str(args.device),
            max_examples=args.max_examples,
            sampling_steps=int(args.steps),
            sampling_method=str(args.method),
            include_hann_baselines=not bool(args.no_hann_baselines),
            hann_amp_frac=float(args.hann_amp_frac),
            hann_phase_frac=float(args.hann_phase_frac),
            hann_use_time_window=not bool(args.no_hann_time_window),
            include_trace_metrics=bool(args.trace_metrics),
            trace_frog_method=str(args.trace_frog_method),
            out_json=args.out_json,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


def save_predictions_command(args: argparse.Namespace) -> int:
    from .prediction_artifacts import PredictionArtifactConfig, save_prediction_artifacts

    result = save_prediction_artifacts(
        PredictionArtifactConfig(
            checkpoint=args.checkpoint,
            out_dir=args.out_dir,
            data_dir=args.data_dir,
            split_file=args.split_file,
            split=str(args.split),
            source_kind=str(args.source_kind),
            batch_size=int(args.batch_size),
            num_workers=int(args.workers),
            device=str(args.device),
            max_examples=args.max_examples,
            sampling_steps=int(args.steps),
            sampling_method=str(args.method),
            include_measured_trace=not bool(args.no_measured_trace),
            overwrite=bool(args.overwrite),
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


def evaluate_predictions_command(args: argparse.Namespace) -> int:
    from .prediction_artifacts import (
        PredictionArtifactEvaluationConfig,
        evaluate_prediction_artifacts,
    )

    result = evaluate_prediction_artifacts(
        PredictionArtifactEvaluationConfig(
            predictions_dir=args.predictions_dir,
            include_trace_metrics=bool(args.trace_metrics),
            trace_frog_method=str(args.trace_frog_method),
            max_examples=args.max_examples,
            out_json=args.out_json,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


def visualize_predictions_command(args: argparse.Namespace) -> int:
    from .visualization import PulseVisualizationConfig, visualize_prediction_artifacts

    result = visualize_prediction_artifacts(
        PulseVisualizationConfig(
            predictions_dir=args.predictions_dir,
            out_dir=args.out_dir,
            selection_json=args.selection_json,
            pulse_uids=tuple(args.pulse_uid or ()),
            max_examples=args.max_examples,
            include_trace_diagnostics=not bool(args.skip_trace_diagnostics),
            trace_frog_method=str(args.trace_frog_method),
            phase_support_fraction=float(args.phase_support_fraction),
            dpi=int(args.dpi),
            num_workers=int(args.workers),
            overwrite=bool(args.overwrite),
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0






def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and inspect flow-matching pulse models.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect generated .npz shards.")
    inspect_parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)

    train_parser = subparsers.add_parser("train", help="Train a trace-conditioned flow model.")
    add_dataset_args(train_parser)
    add_model_args(train_parser)
    add_training_args(train_parser)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate whether a checkpoint reduces aligned pulse-domain MAE.",
    )
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.add_argument("--data-dir", type=Path)
    evaluate_parser.add_argument("--split-file", type=Path)
    evaluate_parser.add_argument(
        "--split",
        choices=["train", "val", "validation", "all"],
        default="val",
    )
    evaluate_parser.add_argument("--batch-size", type=int, default=64)
    evaluate_parser.add_argument("--workers", type=int, default=4)
    evaluate_parser.add_argument("--device", default="auto")
    evaluate_parser.add_argument("--max-examples", type=int)
    evaluate_parser.add_argument("--source-kind")
    evaluate_parser.add_argument("--steps", type=int, default=64)
    evaluate_parser.add_argument("--method", choices=["euler", "rk4"], default="euler")
    evaluate_parser.add_argument("--no-hann-baselines", action=argparse.BooleanOptionalAction, default=False)
    evaluate_parser.add_argument("--hann-amp-frac", type=float, default=0.08)
    evaluate_parser.add_argument("--hann-phase-frac", type=float, default=0.10)
    evaluate_parser.add_argument("--no-hann-time-window", action=argparse.BooleanOptionalAction, default=False)
    evaluate_parser.add_argument("--trace-metrics", action=argparse.BooleanOptionalAction, default=False)
    evaluate_parser.add_argument("--trace-frog-method", default="tg")
    evaluate_parser.add_argument("--out-json", type=Path)

    save_parser = subparsers.add_parser(
        "save-predictions",
        help="Run flow inference once and save prediction fields as .npz artifacts.",
    )
    save_parser.add_argument("--checkpoint", type=Path, required=True)
    save_parser.add_argument("--out-dir", type=Path, required=True)
    save_parser.add_argument("--data-dir", type=Path)
    save_parser.add_argument("--split-file", type=Path)
    save_parser.add_argument("--split", choices=["train", "val", "validation", "all"], default="val")
    save_parser.add_argument("--source-kind", default="copra_final")
    save_parser.add_argument("--batch-size", type=int, default=64)
    save_parser.add_argument("--workers", type=int, default=4)
    save_parser.add_argument("--device", default="auto")
    save_parser.add_argument("--max-examples", type=int)
    save_parser.add_argument("--steps", type=int, default=64)
    save_parser.add_argument("--method", choices=["euler", "rk4"], default="euler")
    save_parser.add_argument("--no-measured-trace", action=argparse.BooleanOptionalAction, default=False)
    save_parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)

    artifact_eval_parser = subparsers.add_parser(
        "evaluate-predictions",
        help="Evaluate saved prediction artifacts without running the model.",
    )
    artifact_eval_parser.add_argument("--predictions-dir", type=Path, required=True)
    artifact_eval_parser.add_argument("--trace-metrics", action=argparse.BooleanOptionalAction, default=False)
    artifact_eval_parser.add_argument("--trace-frog-method", default="tg")
    artifact_eval_parser.add_argument("--max-examples", type=int)
    artifact_eval_parser.add_argument("--out-json", type=Path)

    visualize_parser = subparsers.add_parser(
        "visualize-predictions",
        help="Plot original, COPRA, direct-Hann, and flow fields from saved predictions.",
    )
    visualize_parser.add_argument("--predictions-dir", type=Path, required=True)
    visualize_parser.add_argument("--out-dir", type=Path, required=True)
    visualize_parser.add_argument("--selection-json", type=Path)
    visualize_parser.add_argument("--pulse-uid", action="append")
    visualize_parser.add_argument("--max-examples", type=int)
    visualize_parser.add_argument("--skip-trace-diagnostics", action=argparse.BooleanOptionalAction, default=False)
    visualize_parser.add_argument("--trace-frog-method", default="tg")
    visualize_parser.add_argument("--phase-support-fraction", type=float, default=0.05)
    visualize_parser.add_argument("--dpi", type=int, default=180)
    visualize_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of pulse visualizations to compute concurrently.",
    )
    visualize_parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "inspect":
        return inspect_command(args)
    if args.command == "train":
        return train_command(args)
    if args.command == "evaluate":
        return evaluate_command(args)
    if args.command == "save-predictions":
        return save_predictions_command(args)
    if args.command == "evaluate-predictions":
        return evaluate_predictions_command(args)
    if args.command == "visualize-predictions":
        return visualize_predictions_command(args)
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
