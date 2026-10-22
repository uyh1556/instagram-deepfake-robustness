"""Synthetic checks for the smoothing score-shift diagnostic; no models or data."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import analyze_smoothing_threshold_shift as shift


def write(folder, real_push, fake_push=.2):
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(400, int), np.ones(400, int)]
    base = rng.normal(0, 1, 800)+2*y
    for k, c in enumerate(shift.CONDITIONS):
        # Smoothing pushes scores toward "fake" by a fixed amount per level, separately per class.
        margin = base+k*np.where(y == 0, real_push, fake_push)
        pd.DataFrame(dict(crop_path=[f'f{i:03d}.png' for i in range(800)][::-1], class_index=y[::-1],
                          raw_margin=margin[::-1])).to_csv(folder/f'm_{c}.csv', index=False)
    return folder/'m'


class ThresholdShiftTests(unittest.TestCase):
    def test_auc_matches_pairwise_count(self):
        neg, pos = np.array([0., 1, 2]), np.array([1., 3])
        pairs = np.mean([(p > n)+.5*(p == n) for n in neg for p in pos])
        self.assertAlmostEqual(shift.auc(neg, pos), pairs)

    def test_no_push_is_neutral_and_push_raises_false_positives(self):
        with tempfile.TemporaryDirectory() as tmp:
            still = shift.measure(*shift.load(write(Path(tmp), 0, 0)))
            pushed = shift.measure(*shift.load(write(Path(tmp), .6)))
        self.assertAlmostEqual(still['skew'], 0)
        self.assertAlmostEqual(still['fpr_strong@normal5'], .05, places=2)
        self.assertEqual(still['flip_real@normal5'], 0)
        self.assertGreater(pushed['skew'], .1)
        self.assertGreater(pushed['fpr_strong@normal5'], .3)
        self.assertGreater(pushed['flip_real@normal5'], .25)
        self.assertAlmostEqual(pushed['real_shift']/pushed['fake_shift'], 3, places=6)
        self.assertLess(pushed['strong_real_vs_normal_fake'], pushed['normal_auc'])

    def test_mismatched_images_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix = write(Path(tmp), .2)
            frame = pd.read_csv(f'{prefix}_smooth_weak.csv')
            frame.loc[0, 'crop_path'] = 'other.png'
            frame.to_csv(f'{prefix}_smooth_weak.csv', index=False)
            with self.assertRaises(ValueError):
                shift.load(prefix)


if __name__ == '__main__':
    unittest.main()
