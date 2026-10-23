"""Known-pattern checks for normalization, texture meaning and image pooling."""

import json
import unittest

import numpy as np
from PIL import Image

from poppypurev2.config import DEFAULT_CONFIG
from poppypurev2.features import descriptor_names, feature_views, image_descriptor
from poppypurev2.masks import prototype_masks, removal_overlay


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = DEFAULT_CONFIG.resolve().parent.parent
        cls.config = json.loads((root / "configs/features-v1.json").read_text())
        cls.preprocessing = json.loads(
            (root / "configs/preprocessing-v1.json").read_text()
        )

    def test_constant_image_has_expected_histograms_and_texture(self):
        image = Image.new("RGB", (64, 64), (128, 128, 128))
        descriptor = image_descriptor(image, self.config)
        names = descriptor_names(self.config)
        self.assertEqual(len(names), 196)
        self.assertEqual(len(set(names)), len(names))
        values = dict(zip(names, descriptor, strict=True))
        for start in range(0, 96, 16):
            self.assertAlmostEqual(float(descriptor[start : start + 16].sum()), 1)
        self.assertAlmostEqual(float(descriptor[96:106].sum()), 1)
        self.assertAlmostEqual(float(descriptor[106:124].sum()), 1)
        self.assertEqual(values["lbp_p8_r1_bin_08"], 1)
        self.assertEqual(values["lbp_p16_r2_bin_16"], 1)
        for distance in (1, 2, 4):
            for angle in (0, 45, 90, 135):
                self.assertEqual(values[f"glcm_contrast_d{distance}_a{angle}"], 0)
                self.assertEqual(values[f"glcm_energy_d{distance}_a{angle}"], 1)
                self.assertEqual(values[f"glcm_correlation_d{distance}_a{angle}"], 1)
        self.assertTrue(np.isfinite(descriptor).all())

    def test_checkerboard_distance_changes_contrast(self):
        pixels = np.uint8(np.indices((64, 64)).sum(axis=0) % 2 * 255)
        image = Image.fromarray(np.repeat(pixels[..., None], 3, axis=2))
        descriptor = image_descriptor(image, self.config)
        values = dict(zip(descriptor_names(self.config), descriptor, strict=True))
        self.assertEqual(values["glcm_contrast_d1_a0"], 31**2)
        self.assertEqual(values["glcm_contrast_d2_a0"], 0)
        np.testing.assert_array_equal(descriptor, image_descriptor(image, self.config))

    def test_detail_pooling_keeps_one_image_representation(self):
        image = Image.new("RGB", (700, 600), (128, 128, 128))
        views = feature_views(image, self.preprocessing, self.config)
        self.assertEqual(views["whole"].shape, (196,))
        self.assertEqual(views["crop_detail"].shape, (392,))
        np.testing.assert_array_equal(views["crop_detail"][:196], views["crop"])
        np.testing.assert_array_equal(views["crop_detail"][196:], np.zeros(196))
        with self.assertRaises(ValueError):
            image_descriptor(image, {**self.config, "mask": "unreviewed"})

    def test_simple_dark_background_mask_and_overlay(self):
        pixels = np.zeros((64, 64, 3), dtype=np.uint8)
        pixels[16:48, 16:48] = 255
        image = Image.fromarray(pixels)
        masks, details = prototype_masks(image)
        self.assertEqual(details["polarity_from_pixels"], "dark")
        mask = masks["lightness_otsu"]
        self.assertEqual(mask.dtype, np.bool_)
        self.assertTrue(mask[20:40, 20:40].all())
        self.assertFalse(mask[:8].any())
        overlay = np.asarray(removal_overlay(image, mask))
        self.assertGreater(overlay[0, 0, 0], overlay[0, 0, 1])
        np.testing.assert_array_equal(overlay[32, 32], [255, 255, 255])
        # Dense material reaches every border; border median must not flip polarity.
        dense = np.full((64, 64, 3), 255, dtype=np.uint8)
        dense[20:44, 20:44] = 0
        dense_masks, dense_details = prototype_masks(Image.fromarray(dense))
        self.assertEqual(dense_details["polarity_from_pixels"], "dark")
        self.assertTrue(dense_masks["lightness_otsu"][4, 4])
        self.assertFalse(dense_masks["lightness_otsu"][32, 32])
