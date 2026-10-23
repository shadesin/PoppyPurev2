"""Final fit boundaries, one-time test access and portable inference."""

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.preprocessing import StandardScaler
from torch import nn
from torchvision.models import resnet18

from poppypurev2.acquisition import sha256_file
from poppypurev2.cli import main
from poppypurev2.config import DEFAULT_CONFIG
from poppypurev2.deep import DeepClassifier, EmbeddingHead, cpu_state
from poppypurev2.deep_inputs import CLASSES, class_indices, load_settings
from poppypurev2.evaluation import encoded
from poppypurev2.final import (
    bootstrap_indices,
    claim_test_access,
    fit_fixed_head,
    verify_reference_scaler,
)
from poppypurev2.inference import load_bundle, predict_image


class FinalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = DEFAULT_CONFIG.resolve().parent.parent
        cls.config, cls.source = load_settings(cls.root)
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.bundle = Path(cls.temp.name)
        encoder = resnet18(weights=None)
        encoder.fc = nn.Identity()
        head = EmbeddingHead(
            512, "mlp", np.zeros(512), np.ones(512), cls.config["head"]
        )
        model = DeepClassifier(encoder, head).eval()
        torch.save(
            {
                "model_state": cpu_state(model),
                "model_version": "test-v1",
                "classes_volume_percent": list(CLASSES),
                "config": cls.config,
                "source_preprocessing": cls.source,
                "specification": {
                    "backbone": "resnet18",
                    "head": "mlp",
                    "representation": "crop_detail",
                    "stage": "frozen",
                },
            },
            cls.bundle / "model.pt",
        )
        cls.metadata = {
            "schema_version": 1,
            "model_version": "test-v1",
            "percentage_basis": "volume",
            "classes_volume_percent": list(CLASSES),
            "checkpoint_sha256": sha256_file(cls.bundle / "model.pt"),
        }
        (cls.bundle / "bundle.json").write_bytes(encoded(cls.metadata))
        cls.image = cls.bundle / "image.jpg"
        Image.new("RGB", (1200, 1000), "tan").save(cls.image)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_fixed_refit_runs_exact_count_and_retains_last_epoch(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(21, 6)).astype(np.float32)
        config = copy.deepcopy(self.config)
        config["head"]["patience"] = 0  # Ignored by the fixed refit.
        head, curve, score = fit_fixed_head(
            x, class_indices(list(CLASSES) * 3), config, 3, 42
        )
        self.assertEqual([r["epoch"] for r in curve], [1, 2, 3])
        self.assertFalse(any("validation" in k for r in curve for k in r))
        self.assertEqual(curve[-1]["training_macro_f1"], score["metrics"]["macro_f1"])
        np.testing.assert_allclose(head.mean.numpy(), x.mean(0), atol=1e-7)
        with torch.inference_mode():
            np.testing.assert_allclose(
                head(torch.from_numpy(x)).softmax(1).numpy(),
                score["probabilities"],
                rtol=1e-6,
            )

    def test_test_access_claim_cannot_be_repeated(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "PRETEST_LOCK.json").write_text("{}")
            claim_test_access(run, "a" * 64, "cpu", 82)
            with self.assertRaises(FileExistsError):
                claim_test_access(run, "b" * 64, "cpu", 82)
            saved = json.loads((run / "TEST_ACCESS_STARTED.json").read_text())
            self.assertEqual(saved["primary_checkpoint_sha256"], "a" * 64)

    def test_scaler_verification_matches_double_precision_accumulation(self):
        x = np.tile(np.array([0.5, 38.93346, 12345.67], dtype=np.float32), (466, 1))
        scaler = StandardScaler().fit(x)
        self.assertGreater(float(np.max(np.abs(x.mean(0) - scaler.mean_))), 1e-6)
        verify_reference_scaler(scaler, x)
        with self.assertRaises(ValueError):
            verify_reference_scaler(StandardScaler().fit(x[:400]), x)

    def test_bootstrap_keeps_same_preparation_views_together(self):
        rows = [
            {"volume_percent": v, "preparation_group_id": f"prep{v}"} for v in CLASSES
        ]
        rows += [
            {"volume_percent": 0, "preparation_group_id": "prep0"},
            {"volume_percent": 0, "preparation_group_id": "another0"},
        ]
        rng = np.random.default_rng(7)
        for _ in range(30):
            indices = bootstrap_indices(rows, rng)
            self.assertEqual(int((indices == 0).sum()), int((indices == 7).sum()))
            self.assertEqual(
                {int(rows[i]["volume_percent"]) for i in indices}, set(CLASSES)
            )

    def test_bundle_predicts_without_dataset_or_pretrained_files(self):
        self.assertFalse((self.bundle / "models/pretrained").exists())
        prediction = predict_image(self.bundle, self.image, "cpu")
        self.assertEqual(prediction["percentage_basis"], "volume")
        self.assertIn(prediction["predicted_volume_percent"], CLASSES)
        self.assertAlmostEqual(
            sum(prediction["class_probabilities"].values()), 1, places=6
        )
        with contextlib.redirect_stdout(io.StringIO()) as output:
            result = main(
                [
                    "--config",
                    "/missing/project.toml",
                    "predict",
                    str(self.image),
                    "--model-dir",
                    str(self.bundle),
                    "--device",
                    "cpu",
                ]
            )
        self.assertEqual(result, 0)
        self.assertEqual(
            json.loads(output.getvalue())["predicted_volume_percent"],
            prediction["predicted_volume_percent"],
        )

    def test_bundle_rejects_wrong_labels_corruption_and_small_images(self):
        for changed in (
            {**self.metadata, "percentage_basis": "mass"},
            {**self.metadata, "checkpoint_sha256": "b" * 64},
        ):
            try:
                (self.bundle / "bundle.json").write_bytes(encoded(changed))
                with self.assertRaises(ValueError):
                    load_bundle(self.bundle, "cpu")
            finally:
                (self.bundle / "bundle.json").write_bytes(encoded(self.metadata))
        tiny = self.bundle / "small.png"
        Image.new("RGB", (200, 200), "white").save(tiny)
        with self.assertRaisesRegex(ValueError, "too small"):
            predict_image(self.bundle, tiny, "cpu")


if __name__ == "__main__":
    unittest.main()
