"""Save metadata and ZIP-directory signatures without a second image archive."""

from __future__ import annotations

import json
import struct
import subprocess
from pathlib import Path

from poppypurev2.audit import write_csv


def snapshot_sources(project_root: Path) -> None:
    data = project_root / "data"
    sources = json.loads((data / "sources.json").read_text())
    snapshots = data / "source_metadata"
    snapshots.mkdir(parents=True, exist_ok=True)
    for source in sources.values():
        result = subprocess.run(
            [
                "curl",
                "--fail",
                "--silent",
                "--show-error",
                "--max-time",
                "30",
                source["api_url"],
            ],
            capture_output=True,
            check=True,
        )
        metadata = json.loads(result.stdout)
        if metadata["version"] != source["version"]:
            raise ValueError("Source version changed; pinned source needs review")
        (snapshots / f"{source['dataset_id']}-v1.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
    subset = sources["related_subset"]
    size = subset["size_bytes"]
    start = max(0, size - 200000)
    result = subprocess.run(
        [
            "curl",
            "--location",
            "--fail",
            "--silent",
            "--show-error",
            "--max-time",
            "60",
            "--range",
            f"{start}-{size - 1}",
            subset["download_url"],
        ],
        capture_output=True,
        check=True,
    )
    tail = result.stdout
    if len(tail) != size - start:
        raise ValueError("Unexpected ZIP-directory response length")
    end = tail.rfind(b"PK\x05\x06")
    if end < 0:
        raise ValueError("ZIP end-of-directory record not found")
    _, _, _, _, count, directory_size, offset, _ = struct.unpack_from(
        "<4s4H2IH", tail, end
    )
    cursor = offset - start
    if cursor < 0 or cursor + directory_size > len(tail):
        raise ValueError("ZIP central directory does not fit requested tail")
    rows = []
    for _ in range(count):
        if tail[cursor : cursor + 4] != b"PK\x01\x02":
            raise ValueError("Invalid ZIP central directory")
        crc, compressed, uncompressed = struct.unpack_from("<III", tail, cursor + 16)
        name_size, extra_size, comment_size = struct.unpack_from(
            "<HHH", tail, cursor + 28
        )
        name = tail[cursor + 46 : cursor + 46 + name_size].decode("utf-8")
        rows.append(
            {
                "source_path": name,
                "size_bytes": uncompressed,
                "compressed_bytes": compressed,
                "crc32": f"{crc:08x}",
            }
        )
        cursor += 46 + name_size + extra_size + comment_size
    write_csv(data / "related_subset_inventory.csv", rows)
    print(
        f"Saved two source metadata snapshots and {len(rows)} subset directory entries"
    )
