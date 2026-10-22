"""CPU checks for held-out beauty filters: HF-reduction measure, calibration search, masked blend."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import beauty_smoothing_filters as beauty
import calibrate_beauty_filters as calibration


def textured(seed, size=(120, 100)):
    rng = np.random.default_rng(seed)
    base = 150+rng.normal(0, 15, size)
    array = np.clip(np.stack([base, base*.85, base*.7], -1), 0, 255).astype(np.uint8)
    image = Image.fromarray(array)
    return image, array, beauty.mask(image)


def fake_full(name, array, value):
    # Monotone stand-in: Gaussian blur whose sigma grows with the parameter.
    return cv2.GaussianBlur(array, (0, 0), max(.3, float(value)*100))


class BeautyFilterTests(unittest.TestCase):
    def test_skin_change_measure(self):
        _, array, mask = textured(3)
        self.assertEqual(beauty.skin_change(array, array, mask), 0.)
        shifted = np.clip(array.astype(int)+4, 0, 255).astype(np.uint8)
        self.assertAlmostEqual(beauty.skin_change(array, shifted, mask), 4., places=4)

    def test_hf_reduction_bounds(self):
        _, array, mask = textured(0)
        self.assertAlmostEqual(beauty.hf_reduction(array, array, mask), 0.)
        flat = np.full_like(array, 150)
        self.assertAlmostEqual(beauty.hf_reduction(array, flat, mask), 1., places=6)
        mild = cv2.GaussianBlur(array, (0, 0), 1)
        self.assertTrue(0 < beauty.hf_reduction(array, mild, mask) < 1)

    def test_calibration_hits_target_and_rejects_out_of_range(self):
        images = [textured(s) for s in range(4)]
        grid = np.geomspace(.003, .05, 9)
        with patch.object(beauty, 'full', side_effect=fake_full):
            low, high = [float(np.median(calibration.changes(images, 'fast_global', g))) for g in [grid[2], grid[6]]]
            target = (low+high)/2
            result = calibration.calibrate(images, 'fast_global', target, grid)
            self.assertLess(abs(result['median']-target), .1*target)
            self.assertTrue(grid[2] < result['value'] < grid[6])
            self.assertEqual(result['parameter'], 'spatial_fraction')
            with self.assertRaisesRegex(ValueError, 'outside grid'):
                calibration.calibrate(images, 'fast_global', 500., grid)

    def test_blend_uses_training_mask_and_strength(self):
        image, array, mask = textured(1)
        with patch.object(beauty, 'full', side_effect=lambda n, a, v: np.zeros_like(a)):
            out = np.asarray(beauty.smooth(image, 'fast_global', .01, .75)).astype(float)
        expected = np.rint(array*(1-.75*mask[..., None]))
        self.assertTrue(np.array_equal(out, expected))
        self.assertTrue(np.array_equal(out[0, 0], array[0, 0]))   # outside the face mask stays untouched

    @unittest.skipUnless(hasattr(cv2, 'ximgproc'), 'opencv-contrib not installed locally')
    def test_real_filters_run_and_smooth(self):
        _, array, mask = textured(2)
        for name in beauty.FILTERS:
            value = .02 if beauty.PARAMETER[name] == 'spatial_fraction' else 200.
            out = beauty.full(name, array, value)
            self.assertEqual(out.shape, array.shape)
            self.assertGreater(beauty.hf_reduction(array, out, mask), 0)


if __name__ == '__main__':
    unittest.main()
