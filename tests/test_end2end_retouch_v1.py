"""Retouched-cache build/restore and the metric/fusion rows of the end-to-end retoucher check, without GFPGAN."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from PIL import Image

import end2end_retouch_v1 as er


class FakeRetoucher:
    """Brightens the crop by 30; reports no face for crops whose top-left pixel is 0."""
    identity = dict(settings=dict(name='fake'))

    def __call__(self, array, relative):
        if array[0, 0, 0] == 0:
            return array.copy(), 0
        return np.clip(array.astype(np.int16)+30, 0, 255).astype(np.uint8), 1


class BuildTests(unittest.TestCase):
    def test_build_retouches_detected_frames_packs_and_restores(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root/'cdf'
            rows = []
            for i, (status, value) in enumerate([('detected', 10), ('detected', 0), ('fallback', 10)]):
                p = f'v{i}/f.png'
                (data/p).parent.mkdir(parents=True)
                Image.fromarray(np.full((12, 10, 3), value, np.uint8)).save(data/p)
                rows.append(dict(crop_path=p, roi_status=status, split='test', video_id=f'v{i}', class_index=str(i % 2)))
            (data/'manifests').mkdir()
            pd.DataFrame(rows).to_csv(data/er.builder.MANIFESTS['cdf_test'], index=False)
            args = argparse.Namespace(cdf_root=data, archive_dir=root/'drive', cache_root=root/'cache', view='abpn', device='cpu')
            with mock.patch.object(er, 'load_retoucher', return_value=FakeRetoucher()):
                er.build(args)
            self.assertTrue((root/'drive/cdf_test_abpn.tar').is_file())
            self.assertTrue((root/'drive/preview_abpn.jpg').is_file())
            frames = pd.read_csv(root/'cache/cdf_test/abpn/frames.csv')
            self.assertEqual(frames.crop_path.tolist(), ['v0/f.png', 'v1/f.png'])   # fallback frame not cached
            self.assertEqual(frames.faces.tolist(), [1, 0])
            cache, idents = er.restore(argparse.Namespace(cache_root=root/'fresh', archive_dir=root/'drive'), ['abpn'])
            self.assertEqual(idents['abpn']['settings'], dict(name='fake'))
            clean = Image.open(data/'v0/f.png').convert('RGB')
            self.assertTrue((np.asarray(cache.view(clean, 'v0/f.png', 'abpn')) == 40).all())
            self.assertTrue((np.asarray(cache.view(clean, 'v2/f.png', 'abpn', detected=False)) == 10).all())
            with self.assertRaises(FileNotFoundError):
                er.restore(argparse.Namespace(cache_root=root/'fresh', archive_dir=root/'drive'), ['gfpgan'])

    def test_sample_writes_sheet_without_caching(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root/'cdf'
            rows = []
            for i in range(4):
                p = f'v{i}/f.png'
                (data/p).parent.mkdir(parents=True)
                Image.fromarray(np.full((12, 10, 3), 10, np.uint8)).save(data/p)
                rows.append(dict(crop_path=p, roi_status='detected', split='test', video_id=f'v{i}', class_index=str(i % 2)))
            (data/'manifests').mkdir()
            pd.DataFrame(rows).to_csv(data/er.builder.MANIFESTS['cdf_test'], index=False)
            args = er.parse_args(['sample', '--view', 'gfpgan', '--cdf-root', str(data), '--count', '2', '--sample-out', str(root/'s.jpg'),
                                  '--weights-dir', str(root/'w'), '--device', 'cpu'])
            with mock.patch.object(er, 'load_retoucher', return_value=FakeRetoucher()):
                er.sample(args)
            with Image.open(root/'s.jpg') as sheet:
                self.assertEqual(sheet.size, (768, 512))
            self.assertEqual(sorted(p.name for p in root.iterdir()), ['cdf', 's.jpg'])

    def test_check_accepts_the_first_version_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            cache = er.Retouched(temp)
            for view in ['gfpgan', 'abpn']:   # both folders hold the old gfpgan identity
                folder = cache.root/view
                folder.mkdir(parents=True)
                (folder/'complete.json').write_text('{}')
                (folder/'identity.json').write_text(json.dumps(dict(protocol=er.PROTOCOL, part='gfpgan', weights_sha256='x')))
            self.assertEqual(cache.check('gfpgan')['part'], 'gfpgan')
            with self.assertRaises(ValueError):
                cache.check('abpn')

    def test_frame_seed_is_stable(self):
        self.assertEqual(er.frame_seed('a/b.png'), er.frame_seed('a/b.png'))
        self.assertNotEqual(er.frame_seed('a/b.png'), er.frame_seed('a/c.png'))


class RowTests(unittest.TestCase):
    def test_metrics_and_rows(self):
        rng = np.random.default_rng(0)
        y = np.r_[np.zeros(40, int), np.ones(60, int)]
        clean = rng.normal(size=100)+2*y
        r = er.metrics(y, clean, clean+1.)                 # retoucher shifts every score up by one
        self.assertAlmostEqual(r['clean_auc'], r['retouched_auc'])
        self.assertGreater(r['clean_auc'], r['fp_cross_auc'])  # retouched real look more fake
        self.assertGreater(r['fn_cross_auc'], r['clean_auc'])
        self.assertEqual(r['worse_cross_auc'], r['fp_cross_auc'])
        self.assertEqual(r['fn_flip'], 0.)
        preds = {('baseline', 42): (clean, {'abpn': clean-.5, 'gfpgan': clean-1.})}
        cals = {}
        for f in er.FILTERS:
            preds[(f'aug_pair@{f}', 42)] = (clean, {'abpn': clean-.2, 'gfpgan': clean-.4})
            cals[(42, f)] = dict(rank_val_mean=0., rank_val_std=1.)
        b0 = rng.normal(size=100)+y
        b = {'normal': {v: b0 for v in er.sb.VARIANTS}, 'abpn': dict(saved=b0, smooth_pix=b0, smooth_all=b0+.1),
             'gfpgan': dict(saved=b0, smooth_pix=b0, smooth_all=b0+.5)}
        rows = er.rows_for(y, preds, b, [42], 'xception', ['abpn', 'gfpgan'], calibrations=cals)
        self.assertEqual(rows[rows.view.eq('abpn')].groupby('method').size().to_dict(),
                         {'baseline': 1, 'pair': 4, 'pair_nsf_saved': 4, 'pair_nsf_smooth_pix': 4, 'pair_nsf_smooth_all': 4})
        self.assertEqual(len(rows), 34)
        nsf = rows[rows.view.eq('abpn') & rows.method.eq('pair_nsf_saved')].iloc[0]
        a = clean
        expected = er.metrics(y, .8*a+.2*b0, .8*(a-.2)+.2*b0)
        self.assertAlmostEqual(nsf.worse_cross_auc, expected['worse_cross_auc'])


if __name__ == '__main__':
    unittest.main()
