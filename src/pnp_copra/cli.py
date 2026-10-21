"""One public entry point for generation, training, inference and evaluation."""

from __future__ import annotations

import argparse
import sys

from .configuration import with_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="COPRA Flow: TG-FROG pulse refinement.")
    sub = parser.add_subparsers(dest="command", required=True)
    commands = {
        "generate": "Generate synthetic COPRA training shards.",
        "manifest": "Create deterministic pulse specifications without retrieval.",
        "status": "Inspect generation progress.",
        "splits": "Freeze explicit pulse-level train/validation membership.",
        "inspect": "Inspect shard contents and counts.",
        "train": "Train the conditional flow model.",
        "infer": "Refine a measured trace or a supplied COPRA reconstruction.",
        "prepare": "Prepare a native wavelength-delay text trace.",
        "evaluate": "Evaluate a checkpoint against synthetic ground truth.",
        "save-predictions": "Persist synthetic benchmark predictions.",
        "evaluate-predictions": "Score saved benchmark predictions.",
        "visualize": "Plot saved synthetic predictions.",
        "benchmark-experimental": "Run paired COPRA/PnP/Flow experimental comparisons.",
        "example": "Generate a synthetic example measurement.",
    }
    for name, help_text in commands.items():
        sub.add_parser(name, help=help_text, add_help=False)
    args, remaining = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if args.command in {"generate", "manifest", "status"}:
        from . import artifacts
        return artifacts.main(["run" if args.command == "generate" else args.command,
                               *with_config(remaining)])
    if args.command == "splits":
        from .splits import main as split_main
        return split_main(remaining)
    if args.command in {"infer", "prepare", "example"}:
        from .inference import main as inference_main
        return inference_main([args.command, *with_config(remaining)])
    if args.command == "benchmark-experimental":
        from .experimental import benchmark_main
        return benchmark_main(with_config(remaining))
    from .flow_matching.cli import main as flow_main
    sections = ("dataset", "model", "training", "loss") if args.command == "train" else ()
    command = "visualize-predictions" if args.command == "visualize" else args.command
    return flow_main([command, *with_config(remaining, sections=sections)])


if __name__ == "__main__":
    raise SystemExit(main())
