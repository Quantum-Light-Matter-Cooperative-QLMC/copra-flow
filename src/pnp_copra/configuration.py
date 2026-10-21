"""Translate checked JSON configurations into the same arguments as the CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ALIASES = {
    "validation_fraction": "val_fraction", "learning_rate": "lr", "num_workers": "workers",
    "compile_model": "compile",
}
INVERTED = {
    "include_hard_negatives": "exclude-hard-negatives",
    "use_source_weights": "no-source-weights",
}


def with_config(argv: list[str], *, sections: tuple[str, ...] = ()) -> list[str]:
    """Apply JSON defaults before explicit CLI overrides; reject unknown keys later.

    Paths in a configuration are relative to the working directory, like CLI
    paths. Unknown keys become unknown arguments instead of being ignored.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path)
    args, remaining = parser.parse_known_args(argv)
    if args.config is None:
        return remaining
    payload = json.loads(args.config.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Configuration must be a JSON object.")
    if sections:
        unknown = set(payload) - set(sections)
        if unknown:
            raise ValueError(f"Unknown configuration section(s): {sorted(unknown)}")
        flattened = {}
        for name in sections:
            values = payload.get(name, {})
            if not isinstance(values, dict):
                raise ValueError(f"Configuration section {name!r} must be an object.")
            for key, value in values.items():
                if key in flattened and flattened[key] != value:
                    raise ValueError(f"Conflicting configuration values for {key!r}.")
                flattened[key] = value
        payload = flattened
    arguments = []
    for key, value in payload.items():
        if value is None:
            continue
        if key in INVERTED:
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be boolean.")
            arguments.append(("--no-" if value else "--") + INVERTED[key])
            continue
        flag = "--" + ALIASES.get(key, key).replace("_", "-")
        if isinstance(value, bool):
            arguments.append(flag if value else "--no-" + flag[2:])
        elif isinstance(value, list):
            arguments.extend([flag, *(str(item) for item in value)])
        elif isinstance(value, (str, int, float)):
            # = avoids argparse treating negative scientific notation as a flag.
            arguments.append(f"{flag}={value}")
        else:
            raise ValueError(f"Unsupported configuration value for {key!r}.")
    return arguments + remaining
