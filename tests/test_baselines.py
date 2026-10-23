"""Phase 4 checks for feature isolation, checkpoint selection and controls."""

import json
import unittest

import numpy as np
from PIL import Image

from poppypurev2.baselines import family_columns, pick_best, train_mlp
from poppypurev2.config import DEFAULT_CONFIG
from poppypurev2.controls import low_gradient_features, metadata_inputs
from poppypurev2.evaluation import classification_metrics


class BaselineTests(unittest.TestCase):
    def test_pooled_feature_family_selection(self):
        root = DEFAULT_CONFIG.resolve().parent.parent
        schema = json.loads((root / "data/features/v1/schema.json").read_text())
        for representation in ("whole", "crop", "crop_detail"):
            names = schema[representation]
            colour = family_columns(names, "colour")
            texture = family_columns(names, "texture")
            multiplier = 2 if representation == "crop_detail" else 1
            self.assertEqual(len(colour), multiplier * 96)
            self.assertEqual(len(texture), multiplier * 100)
            self.assertFalse(set(colour) & set(texture))
            self.assertEqual(set(colour) | set(texture), set(range(len(names))))

    def test_near_tie_prefers_size_and_excludes_nonconvergence(self):
        small = {
            "status": "passed",
            "run_id": "small",
            "validation_macro_f1": 0.8,
            "fitted_parameters": 20,
            "training_seconds": 1,
        }
        big = {
            **small,
            "run_id": "big",
            "validation_macro_f1": 0.804,
            "fitted_parameters": 100,
        }
        failed = {
            **small,
            "run_id": "failed",
            "validation_macro_f1": 1,
            "status": "not_converged",
        }
        self.assertEqual(pick_best([small, big, failed])["run_id"], "small")
        stronger = {**big, "run_id": "stronger", "validation_macro_f1": 0.82}
        self.assertEqual(pick_best([small, big, stronger])["run_id"], "stronger")

    def test_mlp_scaler_fits_train_and_checkpoint_matches_curve(self):
        rng = np.random.default_rng(7)
        y = np.tile([0, 10, 20, 30, 40, 50, 100], 6)
        x = rng.normal(size=(len(y), 4))
        yv = np.array([0, 10, 20, 30, 40, 50, 100])
        xv = rng.normal(size=(7, 4)) + 100
        settings = {
            "hidden_layer_sizes": [8],
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.01,
            "learning_rate_init": 0.001,
            "batch_size": 16,
            "max_epochs": 6,
            "patience": 2,
        }
        model, curve = train_mlp(x, y, xv, yv, 42, settings)
        scaler = model.steps[0][1]
        np.testing.assert_allclose(scaler.mean_, x.mean(axis=0))
        self.assertEqual(scaler.n_samples_seen_, len(x))
        classifier = model.steps[-1][1]
        best = max(row["validation_macro_f1"] for row in curve)
        earliest = next(
            row["epoch"] for row in curve if row["validation_macro_f1"] == best
        )
        self.assertEqual(classifier.selected_epoch_, earliest)
        score = classification_metrics(yv.tolist(), model.predict(xv).tolist())[
            "macro_f1"
        ]
        self.assertAlmostEqual(score, best)

    def test_control_inputs_exclude_class_and_identity_fields(self):
        row = {
            "width": "100",
            "height": "200",
            "capture_date": "2021-01-01",
            "background": "white",
            "magnification": "micro",
            "volume_percent": "0",
            "source_path": "0/example.jpg",
            "image_id": "img_1",
        }
        changed = {
            **row,
            "volume_percent": "100",
            "source_path": "100/other.jpg",
            "image_id": "img_2",
        }
        np.testing.assert_array_equal(
            metadata_inputs([row]), metadata_inputs([changed])
        )
        descriptor, selection = low_gradient_features(
            Image.new("RGB", (32, 32), "white"),
            {"border_exclusion_pixels": 2, "gradient_quantile": 0.2, "colour_bins": 16},
        )
        self.assertEqual(descriptor.shape, (96,))
        self.assertFalse(selection[:2].any())
        for index in range(0, 96, 16):
            self.assertAlmostEqual(float(descriptor[index : index + 16].sum()), 1)
