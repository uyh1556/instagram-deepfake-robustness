"""CPU check of aug_pair_rank_nsf: shared non-skin scorer, nsf run build, fused predictions inside the v2 evaluation."""
import argparse
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import beauty_smoothing_v2 as v2
import evaluate_beauty_smoothing_v2 as evaluation
import ffpp_c23_smoothing_v2_diagonal as diagonal
import nonskin_fusion_v2 as nsf
import test_beauty_smoothing_v2 as v2_fixture
import test_nonskin_fusion as fusion_fixture
import test_smoothing_v2_diagonal as diag_fixture


class NsfTests(unittest.TestCase):
    setUp = diag_fixture.DiagonalTests.setUp
    tearDown = diag_fixture.DiagonalTests.tearDown
    train_args = diag_fixture.DiagonalTests.train_args
    build_caches = diag_fixture.DiagonalTests.build_caches

    def test_build_and_evaluate(self):
        with patch.object(v2, 'SkinParser', v2_fixture.FakeParser), patch.object(v2, 'full', side_effect=v2_fixture.fake_full):
            self.build_caches()
            t = self.train_args(strategies=diagonal.MAIN_STRATEGIES[2:3]+['aug_pair_rank_nsf'], epochs=1)
            diagonal.train(t)   # nsf is skipped by the trainer
        self.assertFalse((t.runs_root/diagonal.run_name('dinov2_b14', 'bilateral', nsf.LABEL, 42)).exists())
        b = argparse.Namespace(source_root=self.args.source_root, baseline_runs_root=self.args.baseline_runs_root, runs_root=t.runs_root,
                               train_filter='bilateral', models=['dinov2_b14'], work_dir=self.root/'nsf_work', seed=42, baseline_seed=42,
                               batch_size=4, workers=0, parse_batch=5, device='cpu')
        with patch.object(nsf, 'SETTINGS', {**nsf.SETTINGS, 'train_images': 16, 'val_images': 16}), \
             patch.object(nsf.fusion, 'InnerParser', fusion_fixture.FakeInner):
            nsf.build(b)
            run = t.runs_root/diagonal.run_name('dinov2_b14', 'bilateral', nsf.LABEL, 42)
            done = json.loads((run/'completed.json').read_text())
            self.assertEqual(done['weight'], .2)
            with patch.object(nsf, 'load', side_effect=AssertionError('rebuilt')):
                nsf.build(b)   # finished nsf run and scorer are reused
            e = argparse.Namespace(dataset='cdf_test', data_root=self.args.target_root, baseline_runs_root=self.args.baseline_runs_root,
                archive_dir=self.root/'drive', output_root=self.root/'eval', cache_root=self.root/'cache_eval', protocol=self.args.protocol,
                models=['dinov2_b14'], seed=42, baseline_seed=42, eval_batch_size=4, workers=0, device='cpu', runs_root=t.runs_root,
                train_filters=['bilateral'], strategies=['aug_pair_rank', nsf.LABEL], fc_weights=[100.], fc_ramp_epochs=2., adv_weights=[1.],
                adv_ramp_epochs=2., proj_ks=[4], membership=None, checkpoint='best', summary_name='summary.csv',
                nsf_work_dir=self.root/'nsf_eval', parse_batch=5)
            table = evaluation.evaluate(e)
        self.assertEqual(list(table.strategy.unique()), ['baseline', 'aug_pair_rank@bilateral', f'{nsf.LABEL}@bilateral'])
        self.assertEqual(len(table), 3*len(v2.FILTERS))
        rank = pd.read_csv(e.output_root/'dinov2_b14_aug_pair_rank@bilateral_fast_global_smooth_strong.csv')
        fused = pd.read_csv(e.output_root/f'dinov2_b14_{nsf.LABEL}@bilateral_fast_global_smooth_strong.csv')
        scores = pd.read_csv(t.runs_root/'nonskin_probe/dinov2_b14_cdf_test_face_inner_scores.csv')
        self.assertTrue(np.allclose(fused.raw_margin, nsf.fused(rank.raw_margin.to_numpy(), scores.score.to_numpy(), done)))
        # The non-skin score is shared by every condition: fused - .8 z(rank) is the same column everywhere.
        normal = pd.read_csv(e.output_root/f'dinov2_b14_{nsf.LABEL}@bilateral_normal.csv')
        rn = pd.read_csv(e.output_root/'dinov2_b14_aug_pair_rank@bilateral_normal.csv')
        z = lambda m: .8*(m-done['rank_val_mean'])/done['rank_val_std']
        self.assertTrue(np.allclose(normal.raw_margin-z(rn.raw_margin), fused.raw_margin-z(rank.raw_margin)))


if __name__ == '__main__':
    unittest.main()
