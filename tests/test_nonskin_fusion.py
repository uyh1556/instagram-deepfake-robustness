"""CPU check of the non-skin fusion diagnostic: face-inner masks, fusion grid, oracle verdict."""
import argparse
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import beauty_smoothing_v2 as v2
import build_beauty_smoothing_v2_cache as builder
import diagnose_nonskin_fusion as fusion
import ffpp_c23_smoothing_v2_diagonal as diagonal
import test_beauty_smoothing_v2 as v2_fixture
import test_smoothing_rank_sda_pool_v2 as fixture


class FakeInner:
    def __init__(self, device=None):
        pass

    def masks(self, images):
        out = []
        for image in images:
            m = np.zeros(image.size[::-1], np.float32); m[:image.size[1]//3] = 1.
            out.append(m)
        return out


class NonskinFusionTests(unittest.TestCase):
    setUp = fixture.PoolV2Tests.setUp
    tearDown = fixture.PoolV2Tests.tearDown
    cache_args = v2_fixture.CacheAndEvaluationTests.cache_args

    def test_decide_requires_clean_and_three_filters(self):
        rows = []
        for w, clean, gain in [(0., .80, 0.), (.1, .795, .03), (.2, .78, .05)]:
            for k, f in enumerate(v2.FILTERS):
                rows.append(dict(model='m', variant='v', w=w, filter=f, normal_auc=clean, worse_cross_auc=.6+(gain if k < 3 else 0)))
        verdict = fusion.decide(pd.DataFrame(rows)).iloc[0]
        self.assertTrue(verdict.passed)
        self.assertEqual((verdict.w, verdict.filters_improved), (.1, 3))   # w=.2 improves more but loses .02 clean AUC

    def test_run(self):
        a = self.args
        v2_fixture.write_calibration(self.root/'calibration.json')
        rng = np.random.default_rng(5)
        rows = [dict(crop_path=f'dev_{y}_{i}.png', video_id=f'dev_{y}_{i//2}', class_index=str(y), partition='adapt_val',
                     roi_status='detected') for y in [0, 1] for i in range(6)]
        for r in rows:
            Image.fromarray(rng.integers(0, 256, (24, 24, 3), dtype=np.uint8)).save(a.target_root/r['crop_path'])
        membership = self.root/'membership_labelled.csv'
        pd.DataFrame(rows).to_csv(membership, index=False)
        with patch.object(v2, 'SkinParser', v2_fixture.FakeParser), patch.object(v2, 'full', side_effect=v2_fixture.fake_full):
            builder.build(self.cache_args('ffpp_trainval', a.source_root, membership=None))
            builder.build(self.cache_args('cdf_dev', a.target_root, membership=membership))
        args = argparse.Namespace(source_root=a.source_root, cdf_root=a.target_root, membership=membership,
                                  baseline_runs_root=a.baseline_runs_root, archive_dir=self.root/'drive', output_root=self.root/'fus',
                                  cache_root=self.root/'cache_fus', work_dir=self.root/'work', models=['dinov2_b14'], seed=42,
                                  baseline_seed=42, train_images=16, val_images=16, cdf_images=12, batch_size=4, workers=0,
                                  parse_batch=5, device='cpu', full_model='baseline', runs_parent=None, train_filter='bilateral')
        with patch.object(fusion, 'InnerParser', FakeInner):
            table = fusion.run(args)
        self.assertEqual(len(table), len(fusion.VARIANTS)*len(fusion.WEIGHTS)*len(v2.FILTERS))
        self.assertTrue((self.root/'work/inner/dev_0_1.png').is_file())
        # w = 0 is the full-face detector alone: identical across variants.
        ref = table[table.w.eq(0)].groupby('filter').worse_cross_auc.nunique()
        self.assertTrue(ref.eq(1).all())
        self.assertTrue((self.root/'fus/nonskin_fusion_verdict_seed42.csv').is_file())
        with patch.object(fusion, 'InnerParser', side_effect=AssertionError('reparsed')):
            fusion.parse_inner(args, fusion.occlusion.frames(args))   # masks are reused
        # On top of a finished aug_pair_rank run: w = 0 is rank alone.
        runs = self.root/'runs_parent'/'ffpp_c23_smoothing_v2_diagonal_dinov2_b14'
        with patch.object(v2, 'full', side_effect=v2_fixture.fake_full):
            diagonal.train(argparse.Namespace(source_root=a.source_root, baseline_runs_root=a.baseline_runs_root, runs_root=runs,
                manifest_root=self.root/'diag_manifests', archive_dir=self.root/'drive', source_validation=None, cache_root=self.root/'cache_train',
                train_filter='bilateral', models=['dinov2_b14'], strategies=['aug_pair_rank', 'aug_pair_rank_fc'], seed=42, baseline_seed=42, epochs=1, patience=3,
                source_images=16, val_images=16, batch_size=4, workers=0, encoder_lr=1e-3, head_lr=1e-3, weight_decay=1e-4, ranking_weight=.25,
                fc_weights=[diagonal.FC_WEIGHT], fc_ramp_epochs=diagonal.FC_RAMP, adv_weights=[1.], adv_ramp_epochs=2., adv_lr=1e-3, proj_ks=[4], replace_runs=[], device='cpu'))
        on_rank = argparse.Namespace(**{**vars(args), 'full_model': 'rank', 'runs_parent': self.root/'runs_parent'})
        with patch.object(fusion, 'InnerParser', side_effect=AssertionError('reparsed')):
            ranked = fusion.run(on_rank)
        self.assertEqual(set(ranked.full_model), {'aug_pair_rank@bilateral'})
        self.assertTrue((self.root/'fus/nonskin_fusion_on_rank_bilateral_verdict_seed42.csv').is_file())
        # The non-skin probe is the same (baseline features) in both runs.
        self.assertTrue(np.allclose(ranked.nonskin_alone_auc, table.nonskin_alone_auc))
        on_fc = argparse.Namespace(**{**vars(on_rank), 'full_model': 'fc'})
        with patch.object(fusion, 'InnerParser', side_effect=AssertionError('reparsed')):
            fc = fusion.run(on_fc)
        self.assertEqual(set(fc.full_model), {'aug_pair_rank_fc100_ramp2@bilateral'})
        self.assertTrue((self.root/'fus/nonskin_fusion_on_fc_bilateral_verdict_seed42.csv').is_file())


if __name__ == '__main__':
    unittest.main()
