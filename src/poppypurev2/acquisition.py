"""Download, verify, and safely extract the canonical archive on Backup Plus."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_member(name: str) -> PurePosixPath:
    member = PurePosixPath(name)
    if member.is_absolute() or ".." in member.parts or "\\" in name:
        raise ValueError(f"Unsafe archive member: {name}")
    return member


def acquire(project_root: Path, external_root: Path, workers: int = 8) -> dict:
    """Fetch bounded byte ranges, verify SHA-256, then extract without alterations."""
    if not external_root.parent.is_mount():
        raise ValueError(f"External drive is not mounted: {external_root.parent}")
    source = json.loads((project_root / "data/sources.json").read_text())["canonical"]
    archives, raw = external_root / "source_archives", external_root / "raw"
    archives.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    archive = archives / source["local_filename"]
    if not archive.exists():
        chunks = archives / ".download_chunks"
        chunks.mkdir(exist_ok=True)
        chunk_size = 8 * 1024 * 1024
        ranges = [
            (i, start, min(start + chunk_size, source["size_bytes"]) - 1)
            for i, start in enumerate(range(0, source["size_bytes"], chunk_size))
        ]

        def fetch(part: tuple[int, int, int]) -> Path:
            index, start, end = part
            path = chunks / f"{index:04d}.part"
            expected = end - start + 1
            if path.exists() and path.stat().st_size == expected:
                return path
            subprocess.run(
                [
                    "curl",
                    "--location",
                    "--fail",
                    "--silent",
                    "--show-error",
                    "--retry",
                    "4",
                    "--max-time",
                    "300",
                    "--range",
                    f"{start}-{end}",
                    "--output",
                    str(path),
                    source["download_url"],
                ],
                check=True,
            )
            if path.stat().st_size != expected:
                raise ValueError(f"Unexpected byte-range size: {path}")
            print(f"Downloaded chunk {index + 1}/{len(ranges)}", flush=True)
            return path

        with ThreadPoolExecutor(max_workers=workers) as pool:
            paths = list(pool.map(fetch, ranges))
        assembling = archive.with_suffix(".assembling")
        with assembling.open("wb") as output:
            for path in paths:
                with path.open("rb") as stream:
                    shutil.copyfileobj(stream, output)
        if sha256_file(assembling) != source["sha256"]:
            raise ValueError("Downloaded archive does not match published SHA-256")
        assembling.rename(archive)
    if archive.stat().st_size != source["size_bytes"]:
        raise ValueError("Archive size does not match published metadata")
    digest = sha256_file(archive)
    if digest != source["sha256"]:
        raise ValueError("Archive does not match published SHA-256")
    print("Published archive SHA-256 verified", flush=True)
    with zipfile.ZipFile(archive) as zipped:
        members = zipped.infolist()
        for info in members:
            member = validate_member(info.filename)
            target = raw.joinpath(*member.parts)
            if not target.resolve().is_relative_to(raw.resolve()):
                raise ValueError(
                    f"Archive member escapes raw directory: {info.filename}"
                )
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            # Existing files are compared with the ZIP data rather than overwritten.
            with zipped.open(info) as stream:
                if target.exists():
                    if hashlib.sha256(stream.read()).hexdigest() != sha256_file(target):
                        raise ValueError(f"Existing extracted file differs: {target}")
                else:
                    with target.open("xb") as output:
                        shutil.copyfileobj(stream, output)
    for name, target in (("source_archives", archives), ("raw", raw)):
        link = project_root / "data" / name
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise ValueError(f"Existing link points somewhere else: {link}")
        elif link.exists():
            raise ValueError(f"Refusing to replace existing local data: {link}")
        else:
            link.symlink_to(target, target_is_directory=True)
    result = {
        "archive_sha256": digest,
        "archive_size_bytes": archive.stat().st_size,
        "archive_path": str(archive),
        "raw_path": str(raw),
        "archive_entries": len(members),
        "verified": True,
    }
    (project_root / "data/acquisition.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result
