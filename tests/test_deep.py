"""Meaningful checks of CNN geometry, pooling, targets and fitting boundaries."""

import copy
import unittest

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.models import resnet18

from poppypurev2.config import DEFAULT_CONFIG
from poppypurev2.deep import DeepClassifier, EmbeddingHead, fine_tuning_mode, fit_head
from poppypurev2.deep_inputs import (
    CLASSES,
    class_indices,
    image_regions,
    letterbox_tensor,
    load_settings,
    pooled_embedding,
)


class DeepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = DEFAULT_CONFIG.resolve().parent.parent
        cls.config, cls.source = load_settings(cls.root)
        torch.set_num_threads(1)

    def test_letterbox_has_neutral_padding_and_correct_rgb(self):
        tensor = letterbox_tensor(
            Image.new("RGB", (200, 100), (255, 0, 0)), self.config["preprocessing"]
        )
        self.assertEqual(tuple(tensor.shape), (3, 224, 224))
        self.assertTrue(torch.equal(tensor[:, :56], torch.zeros(3, 56, 224)))
        self.assertTrue(torch.equal(tensor[:, 168:], torch.zeros(3, 56, 224)))
        expected = (
            torch.tensor([1.0, 0.0, 0.0]) - torch.tensor([0.485, 0.456, 0.406])
        ) / torch.tensor([0.229, 0.224, 0.225])
        torch.testing.assert_close(
            tensor[:, 56:168, 0], expected[:, None].expand(3, 112)
        )

    def test_detail_regions_pool_once_per_source_image(self):
        class Average(nn.Module):
            def forward(self, inputs):
                return inputs.mean(dim=(-2, -1))

        images = torch.arange(2 * 5 * 3 * 4 * 4).float().reshape(2, 5, 3, 4, 4)
        pooled = pooled_embedding(Average(), images)
        self.assertEqual(tuple(pooled.shape), (2, 3))
        torch.testing.assert_close(pooled, images.mean(dim=(1, 3, 4)))
        result = image_regions(
            Image.new("RGB", (1200, 1000), "red"),
            "crop_detail",
            self.config,
            self.source,
        )
        self.assertEqual(tuple(result.shape), (5, 3, 224, 224))

    def test_volume_labels_are_indices_only_inside_loss(self):
        self.assertEqual(class_indices(CLASSES).tolist(), list(range(7)))
        with self.assertRaises(KeyError):
            class_indices([60])

    def test_fine_tuning_freezes_early_blocks_and_batchnorm(self):
        encoder = resnet18(weights=None)
        encoder.fc = nn.Identity()
        model = DeepClassifier(
            encoder,
            EmbeddingHead(512, "mlp", np.zeros(512), np.ones(512), self.config["head"]),
        )
        fine_tuning_mode(model)
        self.assertTrue(model.head.training)
        self.assertFalse(model.encoder.training)
        for name, parameter in model.encoder.named_parameters():
            self.assertEqual(parameter.requires_grad, name.startswith("layer4."))
        for layer in model.encoder.modules():
            if isinstance(layer, nn.BatchNorm2d):
                self.assertFalse(layer.training)

    def test_scaling_and_earliest_best_checkpoint(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(28, 8)).astype(np.float32)
        xv = (rng.normal(size=(7, 8)) + 40).astype(np.float32)
        config = copy.deepcopy(self.config)
        config["head"].update(max_epochs=5, patience=2, batch_size=14)
        y, yv = class_indices(list(CLASSES) * 4), class_indices(CLASSES)
        head, curve, epoch, _, val_score = fit_head(x, y, xv, yv, "mlp", config)
        np.testing.assert_allclose(head.mean.numpy(), x.mean(axis=0), atol=1e-7)
        best = max(row["validation_macro_f1"] for row in curve)
        self.assertEqual(
            epoch,
            next(row["epoch"] for row in curve if row["validation_macro_f1"] == best),
        )
        self.assertAlmostEqual(best, val_score["metrics"]["macro_f1"])
        self.assertTrue(np.isfinite(val_score["metrics"]["cross_entropy"]))


if __name__ == "__main__":
    unittest.main()
