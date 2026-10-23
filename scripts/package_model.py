"""Build or verify the downloadable model asset without changing the model."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/poppypure-v1"
ASSET_MANIFEST = ROOT / "releases/poppypure-v1/ASSET_MANIFEST.json"
PREFIX = "models/poppypure-v1/"


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def checked_members(release: dict) -> dict[str, str]:
    members = release["files"]
    for name in members:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ValueError(f"Unsafe bundle path: {name}")
    return dict(members)


def verify_archive(path: Path, expected: dict) -> dict:
    if digest(path) != expected["archive_sha256"]:
        raise ValueError("ZIP checksum differs from the committed asset manifest")
    with ZipFile(path) as archive:
        release_bytes = archive.read(PREFIX + "RELEASE_MANIFEST.json")
        if (
            hashlib.sha256(release_bytes).hexdigest()
            != expected["release_manifest_sha256"]
        ):
            raise ValueError("Packaged release manifest checksum differs")
        release = json.loads(release_bytes)
        members = checked_members(release)
        members["RELEASE_MANIFEST.json"] = expected["release_manifest_sha256"]
        names = archive.namelist()
        if len(names) != len(members) or set(names) != {PREFIX + n for n in members}:
            raise ValueError("ZIP contains missing, duplicate or unexpected files")
        for name, sha256 in members.items():
            with archive.open(PREFIX + name) as handle:
                if hashlib.file_digest(handle, "sha256").hexdigest() != sha256:
                    raise ValueError(f"Packaged file checksum differs: {name}")
        if release["checkpoint_sha256"] != expected["checkpoint_sha256"]:
            raise ValueError("Checkpoint identity differs")
    return {"status": "passed", "files_verified": len(members), **expected}


def build(output: Path, overwrite: bool = False) -> dict:
    checksum = output.with_name(output.name + ".sha256")
    if not overwrite and (output.exists() or checksum.exists()):
        raise ValueError("Asset already exists; use --overwrite to rebuild it")
    release_path = MODEL / "RELEASE_MANIFEST.json"
    release = json.loads(release_path.read_text())
    members = checked_members(release)
    members["RELEASE_MANIFEST.json"] = digest(release_path)
    for name, sha256 in members.items():
        if digest(MODEL / name) != sha256:
            raise ValueError(f"Source model bundle changed: {name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="model-zip-", dir=output.parent) as temp:
        temporary = Path(temp) / output.name
        with ZipFile(
            temporary, "w", compression=ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for name in sorted(members):
                # Fixed ZIP metadata; only the release's explicitly listed files.
                info = ZipInfo(PREFIX + name)
                info.compress_type = ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, (MODEL / name).read_bytes(), compresslevel=6)
        expected = {
            "schema_version": 1,
            "model_version": release["model_version"],
            "percentage_basis": "volume",
            "archive_name": output.name,
            "archive_bytes": temporary.stat().st_size,
            "archive_sha256": digest(temporary),
            "archive_files": len(members),
            "release_manifest_sha256": members["RELEASE_MANIFEST.json"],
            "checkpoint_sha256": release["checkpoint_sha256"],
            "extract_into": "repository root",
            "archive_prefix": PREFIX,
        }
        verify_archive(temporary, expected)
        temporary.replace(output)
    checksum.write_text(f"{expected['archive_sha256']}  {output.name}\n")
    ASSET_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    ASSET_MANIFEST.write_text(json.dumps(expected, indent=2) + "\n")
    return expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("build", help="Package the existing verified model")
    pack.add_argument("--output", type=Path, default=ROOT / "poppypure-v1.zip")
    pack.add_argument("--overwrite", action="store_true")
    verify = commands.add_parser(
        "verify", help="Check a downloaded asset before extraction"
    )
    verify.add_argument(
        "archive", type=Path, nargs="?", default=ROOT / "poppypure-v1.zip"
    )
    args = parser.parse_args()
    try:
        result = (
            build(args.output, args.overwrite)
            if args.command == "build"
            else verify_archive(args.archive, json.loads(ASSET_MANIFEST.read_text()))
        )
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Model asset check failed: {error}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
