"""Phase 0 checks for stable labels, config, and run-record behaviour."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from poppypurev2.cli import main
from poppypurev2.config import CLASSES_VOLUME_PERCENT, DEFAULT_CONFIG, load_config


class Phase0Tests(unittest.TestCase):
    def test_config_has_seven_volume_classes(self) -> None:
        config, digest = load_config(DEFAULT_CONFIG)
        self.assertEqual(
            tuple(config["project"]["classes_volume_percent"]), CLASSES_VOLUME_PERCENT
        )
        self.assertEqual(len(digest), 64)

    def test_smoke_record_and_no_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "smoke.json"
            self.assertEqual(main(["smoke", "--output", str(output)]), 0)
            record = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(record["schema_version"], 1)
            self.assertEqual(record["kind"], "phase0_smoke")
            self.assertEqual(record["status"], "passed")
            self.assertIsNone(record["dataset_manifest_sha256"])
            self.assertIsNone(record["metrics"])
            with self.assertRaises(FileExistsError):
                main(["smoke", "--output", str(output)])


if __name__ == "__main__":
    unittest.main()
