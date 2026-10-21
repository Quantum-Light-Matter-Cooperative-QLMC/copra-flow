"""Keep plotting caches writable in containers and restricted workspaces."""
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "copra-flow-test-matplotlib"))
