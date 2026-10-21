"""Locate the exact pypret implementation bundled in the installed wheel."""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path


def package_root() -> Path:
    """Use the working directory for user outputs, never site-packages."""
    return Path.cwd()


def repo_root() -> Path:
    """Find the public source checkout, if running from one."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd()


def setup_local_pypret(local_repo_root: str | Path | None = None) -> Path:
    """Compatibility helper; imports no longer depend on sys.path mutation."""
    if local_repo_root is not None:
        raise ValueError("COPRA Flow uses its bundled pypret; alternate checkouts are unsupported.")
    spec = find_spec("pypret")
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError("Install copra-flow to obtain the bundled pypret implementation.")
    return Path(spec.origin).parent
