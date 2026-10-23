"""Create traceable experiment records."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from poppypurev2 import __version__


def git_revision(project_root: Path) -> str | None:
    """Return the current commit when the project belongs to a Git checkout."""
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    top = Path(result.stdout.strip()).resolve()
    if top != project_root.resolve():
        return None
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return revision.stdout.strip() if revision.returncode == 0 else None


def package_versions() -> dict[str, str]:
    names = (
        "numpy",
        "Pillow",
        "scikit-learn",
        "scikit-image",
        "matplotlib",
        "torch",
        "torchvision",
    )
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def source_fingerprint(project_root: Path) -> str:
    """Identify local source even before the project is committed to Git."""
    digest = hashlib.sha256()
    for path in sorted((project_root / "src").rglob("*.py")):
        digest.update(str(path.relative_to(project_root)).encode("utf-8") + b"\x00")
        digest.update(path.read_bytes())
        digest.update(b"\x00")
    return digest.hexdigest()


def make_record(
    *,
    kind: str,
    project_root: Path,
    config_sha256: str,
    seed: int,
    status: str,
    dataset_manifest_sha256: str | None = None,
    split_sha256: str | None = None,
    preprocessing: dict[str, Any] | None = None,
    model: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the common run schema; training code will fill the optional fields."""
    return {
        "schema_version": 1,
        "kind": kind,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "project_version": __version__,
        "git_revision": git_revision(project_root),
        "source_tree_sha256": source_fingerprint(project_root),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "package_versions": package_versions(),
        "config_sha256": config_sha256,
        "seed": seed,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "split_sha256": split_sha256,
        "preprocessing": preprocessing,
        "model": model,
        "metrics": metrics,
    }


def write_record(record: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Run record already exists: {output}")
    output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
