"""Offline tests for advisory image quality checks."""
import unittest

import numpy as np
from PIL import Image

from image_quality import assess_image_quality


class ImageQualityTests(unittest.TestCase):
    def test_dark_uniform_image_is_flagged(self):
        image = Image.fromarray(np.zeros((96, 96, 3), dtype=np.uint8), mode="RGB")
        result = assess_image_quality(image)
        self.assertLess(result["brightness"], 45)
        self.assertEqual(result["status"], "review")
        self.assertTrue(result["warnings"])

    def test_checkerboard_has_high_edge_variation(self):
        grid = (np.indices((128, 128)).sum(axis=0) % 2 * 255).astype(np.uint8)
        image = Image.fromarray(np.stack([grid, grid, grid], axis=-1), mode="RGB")
        result = assess_image_quality(image)
        self.assertGreater(result["sharpness"], 20)
        self.assertGreater(result["contrast"], 18)

    def test_result_has_expected_metrics_and_advisory_status(self):
        pixels = np.full((80, 100, 3), 120, dtype=np.uint8)
        pixels[20:60, 25:75] = 180
        result = assess_image_quality(Image.fromarray(pixels, mode="RGB"))
        self.assertTrue({"brightness", "contrast", "sharpness", "warnings", "status"}.issubset(result))
        self.assertIn(result["status"], {"ok", "review"})

    def test_none_is_rejected(self):
        with self.assertRaises(ValueError):
            assess_image_quality(None)


if __name__ == "__main__":
    unittest.main()
