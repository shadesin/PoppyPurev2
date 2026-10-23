"""Image orientation, metadata isolation and geometric processing invariants."""

import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from poppypurev2.preprocessing import (
    crop_box,
    decode_rgb,
    detail_tiles,
    process_image,
    square_input,
)


class PreprocessingTests(unittest.TestCase):
    def test_orientation_applied_metadata_removed_source_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oriented.jpg"
            image = Image.new("RGB", (40, 20), "white")
            exif = Image.Exif()
            exif[274] = 6
            exif[272] = "TEST CAMERA"
            image.save(path, exif=exif)
            before = path.read_bytes()
            result = decode_rgb(path)
            self.assertEqual(result.size, (20, 40))
            self.assertEqual(result.mode, "RGB")
            self.assertFalse(result.info)
            self.assertFalse(result.getexif())
            self.assertEqual(before, path.read_bytes())

    def test_crop_excludes_footer_and_preserves_aspect(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "footer.png"
            image = Image.new("RGB", (200, 100), "white")
            ImageDraw.Draw(image).rectangle((0, 90, 199, 99), fill="red")
            image.save(path)
            result, record = process_image(
                path,
                {
                    "crop_fraction_ltrb": [0, 0, 1, 0.8],
                    "long_side": 100,
                },
            )
            self.assertEqual(result.size, (100, 40))
            self.assertEqual(result.getextrema(), ((255, 255),) * 3)
            self.assertEqual(record["crop_box"], [0, 0, 200, 80])
            square, bounds = square_input(image, 64)
            self.assertEqual(square.size, (64, 64))
            self.assertEqual(bounds, (0, 16, 64, 48))
        with self.assertRaises(ValueError):
            crop_box((100, 100), [0.9, 0, 0.1, 1])

    def test_detail_regions_are_unique_in_bounds_and_repeatable(self):
        image = Image.new("RGB", (900, 700), "white")
        tiles = detail_tiles(image, 512)
        self.assertEqual(len(tiles), 5)
        self.assertEqual(
            [b for _, b in tiles], [b for _, b in detail_tiles(image, 512)]
        )
        for tile, (left, top, right, bottom) in tiles:
            self.assertEqual(tile.size, (512, 512))
            self.assertTrue(0 <= left < right <= 900 and 0 <= top < bottom <= 700)
        self.assertEqual(len(detail_tiles(Image.new("RGB", (100, 100)), 512)), 1)
