"""CPU check of the skin-occlusion pre-check: grey fill inside the v2 mask, head/probe AUCs per view and FF++ method."""
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
import diagnose_skin_occlusion as occlusion
import test_beauty_smoothing_v2 as v2_fixture
import test_smoothing_rank_sda_pool_v2 as fixture


class SkinOcclusionTests(unittest.TestCase):
    setUp = fixture.PoolV2Tests.setUp          # same fixture as the v2 diagonal trace-probe test
    tearDown = fixture.PoolV2Tests.tearDown
    cache_args = v2_fixture.CacheAndEvaluationTests.cache_args

    def test_occlude_fills_mask_only(self):
        image = Image.fromarray(np.full((4, 4, 3), 10, np.uint8))
        mask = np.zeros((4, 4), np.float32); mask[:2] = 1.; mask[2, 0] = .5
        out = np.asarray(occlusion.occlude(image, mask))
        self.assertTrue((out[:2] == 128).all() and (out[3] == 10).all())
        self.assertTrue((out[2, 0] == 69).all())   # soft edge: 10*.5 + 128*.5

    def test_run_reports_both_views_and_methods(self):
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
        ff = pd.read_csv(a.source_root/'manifests/ffpp_c23_4methods_roi16_trainval_v1.csv', dtype=str)
        args = argparse.Namespace(source_root=a.source_root, cdf_root=a.target_root, membership=membership,
                                  baseline_runs_root=a.baseline_runs_root, archive_dir=self.root/'drive', output_root=self.root/'occ',
                                  cache_root=self.root/'cache_occ', models=['dinov2_b14'], seed=42, baseline_seed=42,
                                  train_images=16, val_images=16, cdf_images=12, batch_size=4, workers=0, device='cpu')
        table = occlusion.run(args)
        self.assertEqual(set(table.view), set(occlusion.VIEWS))
        self.assertEqual(set(table.scorer), {'head', 'probe'})
        self.assertEqual(set(table.dataset), {'ffpp_val', 'cdf_dev'})
        methods = set(ff[ff.class_index.eq('1')].method) if 'method' in ff else set()
        self.assertEqual(set(table[table.dataset.eq('ffpp_val')].subset), {'all', *methods} if methods else {'all'})
        self.assertEqual(set(table[table.dataset.eq('cdf_dev')].subset), {'all'})
        self.assertTrue(table.auc.between(0, 1).all())
        self.assertTrue((self.root/'occ/skin_occlusion_seed42.csv').is_file())


if __name__ == '__main__':
    unittest.main()
