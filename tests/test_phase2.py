"""Checks for honest metrics and split integrity, without fitting a model."""

import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from poppypurev2.config import DEFAULT_CONFIG
from poppypurev2.evaluation import (
    assign_conditions,
    classification_metrics,
    freeze_protocol,
    validate_assignments,
)


class Phase2Tests(unittest.TestCase):
    def test_metrics_known_errors_and_missing_classes(self):
        true = [0, 10, 20, 30, 40, 50, 100]
        predicted = [10, 0, 20, 30, 40, 50, 50]
        result = classification_metrics(true, predicted)
        self.assertAlmostEqual(result["macro_f1"], (3 + 2 / 3) / 7)
        self.assertAlmostEqual(result["balanced_accuracy"], 4 / 7)
        self.assertAlmostEqual(result["volume_mae_pp"], 10)
        self.assertEqual(result["binary"]["specificity"], 0)
        self.assertAlmostEqual(result["binary"]["sensitivity"], 5 / 6)
        self.assertEqual(result["confusion_matrix"][0][1], 1)
        missing = classification_metrics([10], [10])
        self.assertIsNone(missing["balanced_accuracy"])
        self.assertIsNone(missing["binary"]["specificity"])
        with self.assertRaises(ValueError):
            classification_metrics([10], [15])
        with self.assertRaises(ValueError):
            classification_metrics([], [])

    def test_real_manifest_protocol_and_tamper_detection(self):
        project = DEFAULT_CONFIG.resolve().parent.parent
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            for name in (
                "manifest.csv",
                "review_decisions.json",
                "dataset_freeze.json",
                "preparation_provenance.json",
            ):
                shutil.copyfile(project / "data" / name, root / "data" / name)
            frozen = freeze_protocol(root)
            self.assertEqual(frozen, freeze_protocol(root))
            with (root / "data/splits/v1/assignments.csv").open(newline="") as handle:
                assigned = list(csv.DictReader(handle))
            self.assertEqual(len(assigned), 548)
            self.assertEqual(
                sum(r["partition"] == "development" for r in assigned), 277
            )
            v1_bytes = (root / "data/splits/v1/protocol.json").read_bytes()
            v2 = freeze_protocol(root, "v2")
            self.assertEqual(v2, freeze_protocol(root, "v2"))
            self.assertEqual(
                v1_bytes, (root / "data/splits/v1/protocol.json").read_bytes()
            )
            with (root / "data/splits/v2/assignments.csv").open(newline="") as handle:
                revised = list(csv.DictReader(handle))
            development = [r for r in revised if r["partition"] == "development"]
            holdout = [r for r in revised if r["partition"] == "holdout"]
            self.assertEqual(len(development), 414)
            self.assertEqual(len(holdout), 134)
            self.assertTrue(
                all(
                    r["background"] == "white" and r["magnification"] == "micro"
                    for r in holdout
                )
            )
            self.assertEqual({r["background"] for r in development}, {"black", "white"})
            self.assertEqual(
                {r["validation_fold"] for r in development},
                {"black_4x", "black_micro", "white_4x"},
            )
            for fold in {r["validation_fold"] for r in development}:
                for subset in (
                    [r for r in development if r["validation_fold"] == fold],
                    [r for r in development if r["validation_fold"] != fold],
                ):
                    self.assertEqual(
                        {int(r["volume_percent"]) for r in subset},
                        {0, 10, 20, 30, 40, 50, 100},
                    )
            pair = [r for r in revised if r["image_id"] in {"img_0181", "img_0198"}]
            self.assertEqual({r["validation_fold"] for r in pair}, {"black_micro"})
            v3 = freeze_protocol(root, "v3")
            self.assertEqual(v3, freeze_protocol(root, "v3"))
            with (root / "data/splits/v3/assignments.csv").open(newline="") as handle:
                current = list(csv.DictReader(handle))
            self.assertEqual(len({r["preparation_group_id"] for r in current}), 547)
            units = {
                partition: {
                    r["preparation_group_id"]
                    for r in current
                    if r["partition"] == partition
                }
                for partition in ("development", "holdout")
            }
            self.assertFalse(units["development"] & units["holdout"])
            self.assertEqual(len(units["development"]), 413)
            self.assertEqual(len(units["holdout"]), 134)
            v4 = freeze_protocol(root, "v4")
            self.assertEqual(v4, freeze_protocol(root, "v4"))
            with (root / "data/splits/v4/assignments.csv").open(newline="") as handle:
                balanced = list(csv.DictReader(handle))
            seen = set()
            for partition, count in (("train", 384), ("validation", 82), ("test", 82)):
                subset = [r for r in balanced if r["partition"] == partition]
                self.assertEqual(len(subset), count)
                self.assertEqual(
                    len(
                        {
                            (r["volume_percent"], r["background"], r["magnification"])
                            for r in subset
                        }
                    ),
                    28,
                )
                groups = {r["preparation_group_id"] for r in subset}
                self.assertFalse(groups & seen)
                seen.update(groups)
            altered = root / "data/splits/v4/assignments.csv"
            original_v4 = altered.read_bytes()
            altered.write_bytes(original_v4 + b" ")
            with self.assertRaisesRegex(ValueError, "Refusing changed"):
                freeze_protocol(root, "v4")
            altered.write_bytes(original_v4)
            with (root / "data/manifest.csv").open(newline="") as handle:
                rows = [
                    r for r in csv.DictReader(handle) if r["decision"] == "included"
                ]
            pairs = json.loads((root / "data/review_decisions.json").read_text())[
                "pairs"
            ]
            # Simulate a related view incorrectly tagged to the other fold.
            next(r for r in rows if r["image_id"] == "img_0181")["magnification"] = "4x"
            with self.assertRaisesRegex(ValueError, "Known related views"):
                validate_assignments(rows, assign_conditions(rows), pairs)
            protocol = root / "data/splits/v1/protocol.json"
            original = protocol.read_bytes()
            protocol.write_bytes(original + b" ")
            with self.assertRaisesRegex(ValueError, "Refusing changed"):
                freeze_protocol(root)
            protocol.write_bytes(original)
            with (root / "data/manifest.csv").open("a") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "Frozen Phase 1 input changed"):
                freeze_protocol(root)
