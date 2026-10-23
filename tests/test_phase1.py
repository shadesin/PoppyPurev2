"""Verify label interpretation, archive safety, and audit inclusion decisions."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from poppypurev2.acquisition import sha256_file, validate_member
from poppypurev2.audit import freeze_dataset, labels_from_path, run_audit


class Phase1Tests(unittest.TestCase):
    def test_volume_labels_and_conditions(self) -> None:
        cases = {
            "root/10_/White background/Micro/Sample 1.jpg": (10, "white", "micro"),
            "root/40_/Black background/4x/Sample 1.jpg": (40, "black", "4x"),
            "root/Poppy seed background(white)/4X/image.jpg": (0, "white", "4x"),
            "root/Poppy seed background/Micro/image.jpg": (0, "black", "micro"),
            "root/Suji background (white)/4X/image.jpg": (100, "white", "4x"),
            "root/Suji background (black)/Micro/image.jpg": (100, "black", "micro"),
        }
        for path, expected in cases.items():
            result = labels_from_path(path)
            self.assertEqual(
                (
                    result["volume_percent"],
                    result["background"],
                    result["magnification"],
                ),
                expected,
            )
        with self.assertRaises(ValueError):
            labels_from_path("root/mystery/Micro/image.jpg")

    def test_archive_paths_cannot_escape_destination(self) -> None:
        for unsafe in (
            "../outside.jpg",
            "/outside.jpg",
            "folder/../../outside.jpg",
            "folder\\outside.jpg",
        ):
            with self.assertRaises(ValueError):
                validate_member(unsafe)

    def test_audit_tracks_duplicates_loose_files_and_raw_integrity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            raw = data / "raw"
            archives = data / "source_archives"
            archives.mkdir(parents=True)
            image_paths = [
                "root/10_/Black background/Micro/Sample 1.jpg",
                "root/10_/Black background/Micro/Sample 2.jpg",
                "root/10_/White background.jpg",
            ]
            for index, name in enumerate(image_paths):
                path = raw / name
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new(
                    "RGB", (32, 32), (20, 30, 40) if index < 2 else (90, 80, 70)
                ).save(path)
            archive = archives / "fixture.zip"
            with zipfile.ZipFile(archive, "w") as zipped:
                for name in image_paths:
                    zipped.write(raw / name, name)
            (data / "sources.json").write_text(
                json.dumps(
                    {
                        "canonical": {
                            "local_filename": archive.name,
                            "sha256": sha256_file(archive),
                        },
                    }
                )
            )
            summary = run_audit(root)
            self.assertEqual(summary["image_entries"], 3)
            self.assertEqual(summary["unique_byte_images"], 2)
            with self.assertRaisesRegex(ValueError, "still need review"):
                freeze_dataset(root)
            self.assertEqual(
                summary["decision_counts"],
                {
                    "included": 1,
                    "excluded": 1,
                    "pending_review": 1,
                },
            )
            with (data / "manifest.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len({row["group_id"] for row in rows}), 1)
            # Editing a raw file must be detected even if it still decodes correctly.
            Image.new("RGB", (32, 32), "red").save(raw / image_paths[0])
            with self.assertRaisesRegex(ValueError, "differs from archive"):
                run_audit(root)


if __name__ == "__main__":
    unittest.main()
