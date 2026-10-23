#!/usr/bin/env python3
"""Create an allowlisted source ZIP without datasets, weights, caches or history."""
from __future__ import annotations

import hashlib
from pathlib import Path
import zipfile


ROOT_FILES = {
    "README.md", "LICENSE", "CITATION.cff", "THIRD_PARTY_NOTICES.md",
    "pyproject.toml", "uv.lock", "MANIFEST.in", ".gitignore", ".gitattributes",
}
DIRECTORIES = {"src", "configs", "docs", "reproduction", "scripts", "examples", "tests", "third_party", ".github"}


def source_paths(root: Path) -> list[Path]:
    paths = []
    candidates = [root / name for name in ROOT_FILES]
    for directory in sorted(DIRECTORIES):
        candidates.extend((root / directory).rglob("*"))
    for path in candidates:
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if rel.parts[0] not in DIRECTORIES and str(rel) not in ROOT_FILES:
            continue
        if any(part in {"__pycache__", ".pytest_cache"} or part.endswith(".egg-info") for part in rel.parts):
            continue
        if path.suffix in {".pyc", ".pt", ".pth", ".npz", ".npy", ".safetensors"} or path.name == ".DS_Store":
            continue
        if path.is_symlink():
            raise ValueError(f"Refusing symlink in release: {rel}")
        paths.append(path)
    return sorted(paths)


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "dist"
    out.mkdir(exist_ok=True)
    paths = source_paths(root)
    archive = out / "copra-flow-v0.1.1.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in paths:
            handle.write(path, "copra-flow/" + str(path.relative_to(root)))
    checksums = [(hashlib.sha256(p.read_bytes()).hexdigest(), p.name) for p in sorted(out.iterdir())
                 if p.is_file() and p.name != "SHA256SUMS"]
    (out / "SHA256SUMS").write_text("".join(f"{digest}  {name}\n" for digest, name in checksums))
    print(f"Created {archive} ({len(paths)} source files, {archive.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
