"""CPU checks for held-out beauty-filter evaluation: cache build/restore, reuse, metrics, bootstrap."""
import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import evaluate_beauty_filter_generalization as general
import test_smoothing_rank_sda_pool_v2 as fixture

FAKE_SIGMA = {'weighted_median': .8, 'fast_global': 1.5, 'domain_transform': 3.}


def fake_full(name, array, value):
    # Stand-in for OpenCV contrib filters (not installed locally); distinct per filter.
    return cv2.GaussianBlur(array, (0, 0), FAKE_SIGMA[name]*value)


class GeneralizationTests(unittest.TestCase):
    setUp = fixture.PoolV2Tests.setUp
    tearDown = fixture.PoolV2Tests.tearDown

    def test_prepare_evaluate_and_reuse(self):
        a = copy.copy(self.args)
        a.strategies, a.epochs, a.bootstrap = list(general.ablation.STRATEGIES), 1, 10
        a.runs_root, a.manifest_root, a.output_root = self.root/'abl', self.root/'abl_m', self.root/'abl_r'
        general.ablation.train(a)
        general.ablation.evaluate(a)
        calibration = self.root/'calibration.json'
        calibration.write_text(json.dumps(dict(protocol='beauty_filter_calibration_v1',
            filters={n: dict(value=1.) for n in general.beauty.FILTERS})))
        g = copy.copy(a)
        g.ablation_output_root, g.calibration, g.output_root = a.output_root, calibration, self.root/'gen'
        g.filter_cache, g.cache_archive, g.cache_workers = self.root/'cache/beauty_filter_cache_v1', self.root/'drive/cache.tar', 1
        with patch.object(general.beauty, 'full', side_effect=fake_full):
            general.prepare(g)
            self.assertTrue(g.cache_archive.is_file())
            detected = pd.read_csv(self.test, dtype=str).roi_status.eq('detected').sum()
            self.assertEqual(len(list((g.filter_cache/'fast_global').rglob('*.png'))), detected)
            general.evaluate(g)
            table = pd.read_csv(g.output_root/'summary.csv')
            self.assertEqual(len(table), 5*4)                     # 5 checkpoints x (bilateral + 3 held-out)
            self.assertEqual(set(table['filter']), {'bilateral', *general.beauty.FILTERS})
            ref = pd.read_csv(a.output_root/'summary.csv').set_index('strategy').smoothing_gap
            bil = table[table['filter'].eq('bilateral')].set_index('strategy').smoothing_gap
            self.assertTrue(np.allclose(bil[ref.index], ref))     # reused predictions give identical bilateral gaps
            boot = pd.read_csv(g.output_root/'bootstrap_smoothing_gap_vs_aug_pair_rank.csv')
            self.assertEqual(len(boot), 4*4*2)                    # filters x other conditions x (real, fake) side
            self.assertEqual(set(boot.metric), {'smoothing_gap', 'fake_side_gap'})
            self.assertTrue(np.allclose(table.fake_side_gap, table.normal_auc-table.normal_real_vs_strong_fake))
            times = {p.name: p.stat().st_mtime_ns for p in g.output_root.glob('*_smooth_strong.csv')}
            general.evaluate(g)
            self.assertEqual(times, {p.name: p.stat().st_mtime_ns for p in g.output_root.glob('*_smooth_strong.csv')})
            fresh = copy.copy(g); fresh.filter_cache = self.root/'cache2/beauty_filter_cache_v1'
            general.prepare(fresh)                                  # restored from the archive, nothing recomputed
            self.assertEqual(len(list((fresh.filter_cache/'fast_global').rglob('*.png'))), detected)
        bad = copy.copy(g); bad.ablation_output_root = self.root/'missing'
        with self.assertRaises(FileNotFoundError):
            general.reused(bad, 'dinov2_b14', 'baseline', 'normal', None, None)


if __name__ == '__main__':
    unittest.main()
