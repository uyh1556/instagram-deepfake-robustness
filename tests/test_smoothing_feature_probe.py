"""CPU checks for the D1 feature probe: grouped probe/transfer AUC, feature capture, end-to-end on tiny checkpoints."""
import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import diagnose_smoothing_feature_probe as probe
import test_beauty_filter_generalization as gen_fixture
import test_smoothing_rank_sda_pool_v2 as fixture


class ProbeMathTests(unittest.TestCase):
    def test_probe_separates_a_trace_and_ignores_noise(self):
        rng = np.random.default_rng(0)
        groups = np.repeat(np.arange(20), 5)
        f0 = rng.normal(size=(100, 6))
        traced = f0+np.r_[3., np.zeros(5)]
        self.assertGreater(probe.probe(f0, traced, groups, 5), .95)
        noise = rng.normal(size=(100, 6))
        self.assertLess(abs(probe.probe(f0, noise, groups, 5)-.5), .15)
        # A detector of the bilateral trace fires on a filter leaving the same trace, not on an orthogonal one.
        self.assertGreater(probe.transfer(f0, traced, groups, 5, (f0, traced)), .95)
        other = f0+np.r_[0., 3., np.zeros(4)]
        self.assertLess(probe.transfer(f0, other, groups, 5, (f0, traced)), .75)


class ProbeRunTests(unittest.TestCase):
    setUp = fixture.PoolV2Tests.setUp
    tearDown = fixture.PoolV2Tests.tearDown

    def test_end_to_end_on_ablation_checkpoints(self):
        a = copy.copy(self.args)
        a.strategies, a.epochs = list(probe.general.ablation.STRATEGIES), 1
        a.runs_root, a.manifest_root, a.output_root = self.root/'abl', self.root/'abl_m', self.root/'abl_r'
        probe.general.ablation.train(a)
        calibration = self.root/'calibration.json'
        calibration.write_text(json.dumps(dict(protocol='beauty_filter_calibration_v1',
            filters={n: dict(value=1.) for n in probe.beauty.FILTERS})))
        d = copy.copy(a)
        d.calibration, d.output_root, d.images, d.folds, d.batch_size = calibration, self.root/'probe', 16, 2, 4
        d.protocol = probe.general.ROOT/'configs/ffpp_c23_celebdf_smoothing_v1.json'
        with patch.object(probe.beauty, 'full', side_effect=gen_fixture.fake_full):
            table = probe.run(d)
        self.assertEqual(len(table), 5*len(probe.FILTERS))
        self.assertTrue(table.probe_auc.between(0, 1).all() and table.transfer_auc.between(0, 1).all())
        self.assertTrue((table.feature_shift >= 0).all())
        frames = pd.read_csv(d.output_root/'frames.csv', dtype=str)
        val = pd.read_csv(a.source_root/'manifests/ffpp_c23_4methods_roi16_trainval_v1.csv', dtype=str)
        allowed = set(val[val.split.eq('val') & val.roi_status.eq('detected')].crop_path)
        self.assertTrue(set(frames.crop_path) <= allowed)                 # FF++ val only, never test or train
        self.assertFalse(json.loads((d.output_root/'protocol.json').read_text())['test_used'])


if __name__ == '__main__':
    unittest.main()
