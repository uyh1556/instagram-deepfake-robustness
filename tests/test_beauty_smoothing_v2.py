"""CPU checks for the v2 beauty-smoothing protocol: mask/blend, calibration, cache build/restore, evaluation.

The face parser and the OpenCV-contrib filters are replaced by deterministic stand-ins (no model download)."""
import argparse
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import beauty_smoothing_v2 as v2
import build_beauty_smoothing_v2_cache as builder
import calibrate_beauty_smoothing_v2 as calibration
import evaluate_beauty_smoothing_v2 as evaluation
import test_smoothing_preservation as fixture

# Each fake filter reaches the anchor's blur at a different multiple of its approved reach, so calibration must move.
FACTOR = {'bilateral': 1., 'weighted_median': 1.3, 'fast_global': .8, 'domain_transform': 1.1}


def fake_full(name, array, spatial):
    return cv2.GaussianBlur(array, (0, 0), max(.3, 2*spatial/v2.APPROVED[name]*FACTOR[name]))


class FakeParser:
    def __init__(self, device=None):
        self.device = 'cpu'

    def masks(self, images):
        out = []
        for image in images:
            array = np.asarray(image)
            h, w = array.shape[:2]
            y, x = np.mgrid[:h, :w]
            skin = (((x-w/2)/(.35*w))**2+((y-h/2)/(.4*h))**2) < 1
            skin &= array.max() > 0           # an all-black frame has no parsed skin
            out.append(v2.finish_mask(skin, min(h, w)))
        return out


def textured(path, seed, size=64):
    rng = np.random.default_rng(seed)
    base = 150+rng.normal(0, 15, (size, size))
    array = np.clip(np.stack([base, base*.85, base*.7], -1), 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def write_calibration(path, scale=1.):
    path.write_text(json.dumps(dict(protocol=v2.PROTOCOL, values={n: v*scale for n, v in v2.APPROVED.items()})))


class ProtocolTests(unittest.TestCase):
    def test_mask_blend_and_view(self):
        rng = np.random.default_rng(0)
        image = Image.fromarray(rng.integers(0, 256, (40, 40, 3), dtype=np.uint8))
        mask = FakeParser().masks([image])[0]
        self.assertEqual(mask.shape, (40, 40))
        self.assertTrue(0 < mask.max() <= 1 and mask[0, 0] == 0)
        restored = v2.image_to_mask(v2.mask_to_image(mask))
        self.assertLessEqual(np.abs(restored-mask).max(), .5/255+1e-7)
        black = np.zeros((40, 40, 3), np.uint8)
        out = np.asarray(v2.blend(image, black, mask, .75)).astype(float)
        expected = np.rint(np.asarray(image)*(1-.75*mask[..., None]))
        self.assertTrue(np.array_equal(out, expected))
        self.assertTrue(np.array_equal(out[0, 0], np.asarray(image)[0, 0]))     # outside the skin untouched
        self.assertEqual(v2.skin_change(np.asarray(image), np.asarray(image), mask), 0.)
        cache = v2.SmoothingCache('/nonexistent', 'cdf_test')
        self.assertIs(cache.view(image, 'x.png', 'bilateral', .75, detected=False), image)   # fallback unchanged
        self.assertIs(cache.view(image, 'x.png', 'bilateral', 0), image)
        with self.assertRaises(ValueError):
            cache.path('mask', '../escape.png')


class CalibrationTests(unittest.TestCase):
    def test_anchor_target_and_search(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = []
            for i in range(16):
                path = f'img/{i}.png'
                textured(root/'ff'/path, i)
                rows.append(dict(crop_path=path, split='train' if i < 14 else 'val', class_index=str(i % 2),
                                 roi_status='detected' if i else 'fallback_letterbox'))
            (root/'ff/manifests').mkdir(parents=True)
            pd.DataFrame(rows).to_csv(root/'ff/manifests/ffpp_c23_4methods_roi16_trainval_v1.csv', index=False)
            with patch.object(v2, 'SkinParser', FakeParser), patch.object(v2, 'full', side_effect=fake_full):
                result = calibration.main(['--source-root', str(root/'ff'), '--output-root', str(root/'out'), '--images', '10'])
            saved = json.loads((root/'out/calibration.json').read_text())
            self.assertEqual(saved['values'][v2.ANCHOR], v2.APPROVED[v2.ANCHOR])
            self.assertEqual(v2.calibrated_values(root/'out/calibration.json'), saved['values'])
            for name in v2.FILTERS:
                self.assertLess(abs(result['filters'][name]['median']-saved['target_value']), .05*saved['target_value'])
                if name != v2.ANCHOR:   # moved toward approved/FACTOR
                    self.assertLess(abs(saved['values'][name]*FACTOR[name]/v2.APPROVED[name]-1), .1)
            self.assertTrue((root/'out/preview_strong.png').is_file())
            frames = pd.read_csv(root/'out/calibration_frames.csv')
            self.assertTrue(set(frames.crop_path) <= {r['crop_path'] for r in rows if r['split'] == 'train' and r['roi_status'] == 'detected'})


class CacheAndEvaluationTests(unittest.TestCase):
    setUp = fixture.PreservationTests.setUp
    tearDown = fixture.PreservationTests.tearDown

    def cache_args(self, dataset, data_root, **extra):
        values = dict(dataset=dataset, data_root=data_root, calibration=self.root/'calibration.json',
                      archive_dir=self.root/'drive', cache_root=self.root/'cache_v2', parts=v2.SmoothingCache.PARTS,
                      workers=1, parse_batch=3, device='cpu')
        return argparse.Namespace(**{**values, **extra})

    def eval_args(self, dataset, data_root, **extra):
        a = self.args
        values = dict(dataset=dataset, data_root=data_root, baseline_runs_root=a.baseline_runs_root,
                      archive_dir=self.root/'drive', output_root=self.root/f'eval_{dataset}', cache_root=self.root/'cache_eval',
                      protocol=a.protocol, models=['dinov2_b14', 'xception'], seed=42, baseline_seed=42, eval_batch_size=4, workers=0,
                      device='cpu', runs_root=None, train_filters=[], strategies=[], fc_weights=[1.], fc_ramp_epochs=0., adv_weights=[1.], adv_ramp_epochs=2.,
                      proj_ks=[4], checkpoint='best', summary_name='summary.csv')
        return argparse.Namespace(**{**values, **extra})

    def test_build_restore_and_evaluate(self):
        a = self.args
        write_calibration(self.root/'calibration.json')
        black = pd.read_csv(self.test).query('roi_status == "detected"').crop_path.iloc[0]
        Image.fromarray(np.zeros((24, 24, 3), np.uint8)).save(a.target_root/black)   # no parsed skin -> unchanged
        with patch.object(v2, 'SkinParser', FakeParser), patch.object(v2, 'full', side_effect=fake_full):
            builder.build(self.cache_args('cdf_test', a.target_root))
        detected = pd.read_csv(self.test).query('roi_status == "detected"').crop_path.tolist()
        for part in v2.SmoothingCache.PARTS:
            self.assertTrue((self.root/f'drive/cdf_test_{part}.tar').is_file())
            self.assertEqual(len(list((self.root/'cache_v2/cdf_test'/part).rglob('*.png'))), len(detected))
        cache = v2.SmoothingCache(self.root/'cache_v2', 'cdf_test')
        ident = cache.check()
        self.assertEqual(ident['values'], v2.APPROVED)
        with Image.open(a.target_root/black) as opened:
            image = opened.convert('RGB')
        self.assertTrue(np.array_equal(np.asarray(cache.view(image, black, 'bilateral', .75)), np.asarray(image)))
        never = patch.object(v2, 'full', side_effect=AssertionError('recomputed'))
        with never, patch.object(v2, 'SkinParser', side_effect=AssertionError('reparsed')):
            builder.build(self.cache_args('cdf_test', a.target_root, cache_root=self.root/'cache_restored'))
        write_calibration(self.root/'other.json', scale=1.5)
        with self.assertRaisesRegex(ValueError, 'provenance differs'):
            builder.build(self.cache_args('cdf_test', a.target_root, cache_root=self.root/'cache_other',
                                          calibration=self.root/'other.json'))
        with self.assertRaisesRegex(ValueError, 'must contain test frames only'):
            (a.source_root/'manifests/ffpp_c23_4methods_roi16_test_v1.csv').write_bytes(
                (a.source_root/'manifests/ffpp_c23_4methods_roi16_trainval_v1.csv').read_bytes())
            builder.build(self.cache_args('ffpp_test', a.source_root))

        table = evaluation.evaluate(self.eval_args('cdf_test', a.target_root))
        self.assertEqual(len(table), 2*len(v2.FILTERS))
        self.assertTrue(np.allclose(table.max_gap, np.maximum(table.fp_gap.abs(), table.fn_gap.abs())))
        self.assertTrue(np.allclose(table.fn_gap, table.normal_auc-table.fn_cross_auc))
        out = self.root/'eval_cdf_test'
        self.assertEqual(len(list(out.glob('*_normal.csv'))), 2)
        self.assertEqual(len(list(out.glob('*_smooth_*.csv'))), 2*len(v2.FILTERS)*3)
        times = {p.name: p.stat().st_mtime_ns for p in out.glob('*.csv') if p.name != 'summary.csv'}
        with patch.object(evaluation.engine, 'infer', side_effect=AssertionError('re-inferred')):
            evaluation.evaluate(self.eval_args('cdf_test', a.target_root))
        self.assertEqual(times, {p.name: p.stat().st_mtime_ns for p in out.glob('*.csv') if p.name != 'summary.csv'})
        protocol = json.loads((out/'protocol.json').read_text())
        self.assertFalse(protocol['test_used_for_selection'])
        self.assertEqual(protocol['cache_identity']['dataset'], 'cdf_test')

    def test_cdf_dev_refuses_official_test_and_evaluates(self):
        a = self.args
        write_calibration(self.root/'calibration.json')
        rng = np.random.default_rng(5)
        rows = []
        for y in [0, 1]:
            for i in range(6):
                path = f'dev_{y}_{i}.png'
                Image.fromarray(rng.integers(0, 256, (24, 24, 3), dtype=np.uint8)).save(a.target_root/path)
                rows.append(dict(crop_path=path, video_id=f'dev_{y}_{i//2}', class_index=str(y), partition='adapt_val',
                                 roi_status='detected' if i else 'fallback_letterbox'))
        membership = self.root/'membership_labelled.csv'
        pd.DataFrame(rows).to_csv(membership, index=False)
        with patch.object(v2, 'SkinParser', FakeParser), patch.object(v2, 'full', side_effect=fake_full):
            with self.assertRaisesRegex(ValueError, 'needs --membership'):
                builder.build(self.cache_args('cdf_dev', a.target_root))
            leaked = self.root/'leaked.csv'
            test_row = pd.read_csv(self.test, dtype=str).iloc[[0]].assign(partition='adapt_val')
            pd.concat([pd.DataFrame(rows), test_row]).to_csv(leaked, index=False)
            with self.assertRaisesRegex(ValueError, 'overlaps the official test'):
                builder.build(self.cache_args('cdf_dev', a.target_root, membership=leaked))
            builder.build(self.cache_args('cdf_dev', a.target_root, membership=membership))
        self.assertEqual(len(list((self.root/'cache_v2/cdf_dev/mask').rglob('*.png'))), 10)
        table = evaluation.evaluate(self.eval_args('cdf_dev', a.target_root, membership=membership, models=['xception']))
        self.assertEqual(len(table), len(v2.FILTERS))
        self.assertEqual(json.loads((self.root/'eval_cdf_dev/protocol.json').read_text())['dataset'], 'cdf_dev')

    def test_ffpp_test_in_domain(self):
        a = self.args
        write_calibration(self.root/'calibration.json')
        trainval = pd.read_csv(a.source_root/'manifests/ffpp_c23_4methods_roi16_trainval_v1.csv', dtype=str)
        test = trainval.groupby('class_index').head(4).copy()
        test['split'] = 'test'
        test.to_csv(a.source_root/'manifests/ffpp_c23_4methods_roi16_test_v1.csv', index=False)
        with patch.object(v2, 'SkinParser', FakeParser), patch.object(v2, 'full', side_effect=fake_full):
            builder.build(self.cache_args('ffpp_test', a.source_root))
            with self.assertRaisesRegex(ValueError, 'must not contain test frames'):
                bad = pd.concat([trainval, test])
                bad.to_csv(a.source_root/'manifests/ffpp_c23_4methods_roi16_trainval_v1.csv', index=False)
                builder.build(self.cache_args('ffpp_trainval', a.source_root))
        table = evaluation.evaluate(self.eval_args('ffpp_test', a.source_root, models=['clip_b16']))
        self.assertEqual(set(table['filter']), set(v2.FILTERS))
        self.assertTrue(table.normal_auc.between(0, 1).all())
        with self.assertRaisesRegex(FileNotFoundError, 'Build the cdf_test v2 cache first'):
            evaluation.evaluate(self.eval_args('cdf_test', a.target_root, cache_root=self.root/'empty', archive_dir=self.root/'none'))


if __name__ == '__main__':
    unittest.main()
