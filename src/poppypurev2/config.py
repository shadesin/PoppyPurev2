"""Load and validate project configuration."""

from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path
from typing import Any

CLASSES_VOLUME_PERCENT = (0, 10, 20, 30, 40, 50, 100)
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "base.toml"


def load_config(path: Path = DEFAULT_CONFIG) -> tuple[dict[str, Any], str]:
    """Return parsed config and its SHA-256 so runs can identify the exact file."""
    raw = path.read_bytes()
    config = tomllib.loads(raw.decode("utf-8"))
    project = config["project"]
    classes = tuple(project["classes_volume_percent"])
    if classes != CLASSES_VOLUME_PERCENT:
        raise ValueError(f"Expected volume-percentage classes {CLASSES_VOLUME_PERCENT}")
    if not isinstance(project["seed"], int):
        raise ValueError("project.seed must be an integer")
    if config["evaluation"]["primary_metric"] != "macro_f1":
        raise ValueError("Primary metric must be macro_f1")
    return config, hashlib.sha256(raw).hexdigest()
