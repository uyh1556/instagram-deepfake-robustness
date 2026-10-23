"""CPU checks for candidate 3 (smoothing-direction removal): projection math and an end-to-end run on tiny checkpoints."""
import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import diagnose_smoothing_direction_removal as removal
import test_beauty_filter_generalization as gen_fixture
import test_smoothing_rank_sda_pool_v2 as fixture


class ProjectionTests(unittest.TestCase):
    def test_removes_the_shared_shift_and_keeps_the_rest(self):
        rng = np.random.default_rng(0)
        f0 = rng.normal(size=(200, 8))
        trace = np.r_[0, 0, 3., np.zeros(5)]
        u = removal.directions(f0, f0+trace+rng.normal(0, .01, f0.shape), 1)
        self.assertAlmostEqual(abs(u[2, 0]), 1., places=2)
        moved = removal.project(f0+trace, u)-removal.project(f0, u)
        self.assertLess(np.abs(moved).max(), .05)                       # the trace is gone after projection
        self.assertTrue(np.allclose(removal.project(f0, u)[:, :2], f0[:, :2], atol=.05))
        self.assertTrue(np.array_equal(removal.project(f0, removal.directions(f0, f0, 0)), f0))   # k=0 = identity

    def test_gaps_follow_definitions(self):
        y = np.r_[np.zeros(50), np.ones(50)].astype(int)
        clean = np.r_[np.zeros(50), np.ones(50)]+np.random.default_rng(1).normal(0, .5, 100)
        g = removal.gaps(y, clean, clean+np.where(y == 1, -10., 0.))     # fakes pushed to real
        self.assertGreater(g['fn_gap'], .3)
        self.assertAlmostEqual(g['fp_gap'], 0., places=6)
        self.assertLess(g['fake_shift'], 0)


class RemovalRunTests(unittest.TestCase):
    setUp = fixture.PoolV2Tests.setUp
    tearDown = fixture.PoolV2Tests.tearDown

    def test_end_to_end(self):
        a = copy.copy(self.args)
        a.strategies, a.epochs = list(removal.general.ablation.STRATEGIES), 1
        a.runs_root = self.root/'runs'/'ffpp_c23_smoothing_source_ablation_dinov2_b14'
        a.manifest_root, a.output_root = self.root/'abl_m', self.root/'abl_r'
        removal.general.ablation.train(a)
        calibration = self.root/'calibration.json'
        calibration.write_text(json.dumps(dict(protocol='beauty_filter_calibration_v1',
            filters={n: dict(value=1.) for n in removal.beauty.FILTERS})))
        r = copy.copy(a)
        r.runs_root, r.calibration, r.output_root = self.root/'runs', calibration, self.root/'removal'
        r.images, r.folds, r.batch_size, r.ks, r.c = 16, 2, 4, [0, 1], 1.
        r.strategies, r.models = ['baseline', 'aug_pair_rank'], ['dinov2_b14']
        r.protocol = removal.general.ROOT/'configs/ffpp_c23_celebdf_smoothing_v1.json'
        with patch.object(removal.beauty, 'full', side_effect=gen_fixture.fake_full):
            table = removal.run(r)
        self.assertEqual(len(table), 2*(1+2)*len(removal.FILTERS))      # checkpoints x (model head + k in {0,1}) x filters
        self.assertEqual(set(table.variant), {'model_head', 'refit', 'remove_k1'})
        self.assertTrue(table.normal_auc.between(0, 1).all())
        self.assertFalse(json.loads((r.output_root/'protocol.json').read_text())['held_out_filters_used_for_fitting'])


if __name__ == '__main__':
    unittest.main()
