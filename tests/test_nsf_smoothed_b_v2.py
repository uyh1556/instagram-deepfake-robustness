"""Inputs, mask agreement and the fusion/reproduction path of the smoothed-B check, without neural inference."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import nsf_smoothed_b_v2 as sb


class FakeCache:
    """Smoothing stand-in: brightens detected frames by 50 (strength ignored)."""
    def view(self, image, relative, name, strength, detected=True):
        from PIL import Image
        if not detected:
            return image
        return Image.fromarray(np.clip(np.asarray(image, np.int16)+50, 0, 255).astype(np.uint8))


class InputTests(unittest.TestCase):
    def test_inner_inputs_use_clean_mask_then_own_mask_on_smoothed_pixels(self):
        import torch
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            Image.fromarray(np.full((20, 20, 3), 10, np.uint8)).save(root/'a.png')
            clean, own = np.zeros((20, 20), np.float32), np.zeros((20, 20), np.float32)
            clean[:10], own[10:] = 1, 1
            for view, mask in [('normal', clean), ('bilateral', own)]:
                out = sb.mask_path(root/'work', view, 'a.png'); out.parent.mkdir(parents=True)
                sb.v2.mask_to_image(mask).save(out)
            frame = pd.DataFrame(dict(crop_path=['a.png'], roi_status=['detected']))
            to_tensor = lambda im: torch.from_numpy(np.asarray(im, dtype=np.float32))
            x, _ = sb.InnerInputs(frame, root, FakeCache(), 'normal', root/'work', to_tensor)[0]
            self.assertEqual(tuple(x.shape), (1, 20, 20, 3))
            self.assertTrue(torch.all(x[0, :10] == 10))
            self.assertAlmostEqual(float(x[0, 15, 5, 0]), 128., delta=1)
            x, _ = sb.InnerInputs(frame, root, FakeCache(), 'bilateral', root/'work', to_tensor)[0]
            self.assertEqual(tuple(x.shape), (2, 20, 20, 3))
            self.assertTrue(torch.all(x[0, :10] == 60))      # smoothed pixels, clean-image mask
            self.assertTrue(torch.all(x[1, 10:] == 60))      # smoothed pixels, own mask
            self.assertAlmostEqual(float(x[1, 2, 5, 0]), 128., delta=1)
            fallback = frame.assign(roi_status='fallback')
            x, _ = sb.InnerInputs(fallback, root, FakeCache(), 'bilateral', root/'work', to_tensor)[0]
            self.assertTrue(torch.all(x[0, :10] == 10))      # fallback frames are never smoothed

    def test_parse_view_parses_the_smoothed_image_and_logs_agreement(self):
        from PIL import Image

        class FakeParser:
            """Keeps pixels brighter than 30, so the mask follows the (fake) smoothing."""
            def __init__(self, device):
                pass

            def masks(self, images):
                return [(np.asarray(im)[..., 0] > 30).astype(np.float32) for im in images]

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            a = np.full((10, 10, 3), 10, np.uint8); a[:5] = 40
            Image.fromarray(a).save(root/'a.png')
            frame = pd.DataFrame(dict(crop_path=['a.png'], roi_status=['detected']))
            args = argparse.Namespace(work_dir=root/'work', output_root=root/'out', cdf_root=root, parse_batch=4, workers=0, device='cpu')
            (args.output_root/'scores').mkdir(parents=True)
            original, sb.fusion.InnerParser = sb.fusion.InnerParser, FakeParser
            try:
                sb.parse_view(args, frame, FakeCache(), 'normal')
                sb.parse_view(args, frame, FakeCache(), 'bilateral')
            finally:
                sb.fusion.InnerParser = original
            with Image.open(sb.mask_path(args.work_dir, 'bilateral', 'a.png')) as opened:
                self.assertTrue((sb.v2.image_to_mask(opened) == 1).all())
            stats = pd.read_csv(args.output_root/'scores/mask_agreement_bilateral.csv')
            self.assertAlmostEqual(float(stats.iou.iloc[0]), .5)
            self.assertAlmostEqual(float(stats.area_ratio.iloc[0]), 2.)

    def test_agreement(self):
        a = np.zeros((10, 10)); a[:5] = 1
        b = np.zeros((10, 10)); b[:4] = 1
        r = sb.agreement(a, b)
        self.assertAlmostEqual(r['iou'], .8)
        self.assertAlmostEqual(r['area_ratio'], .8)
        self.assertTrue(np.isnan(sb.agreement(np.zeros((3, 3)), np.zeros((3, 3)))['iou']))

    def test_check_clean(self):
        s = np.random.default_rng(0).normal(size=100)
        self.assertGreater(sb.check_clean(s+1e-6, s)['corr'], .9999)
        with self.assertRaises(ValueError):
            sb.check_clean(np.random.default_rng(1).normal(size=100), s)


class EvaluateTests(unittest.TestCase):
    def test_evaluate_reproduces_saved_and_writes_headline(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = argparse.Namespace(model='xception', seeds=[42], runs_root=root/'runs', evaluation_root=root/'eval',
                                      pair_nsf_root=root/'pair_nsf', output_root=root/'out')
            rng = np.random.default_rng(1)
            n = 80
            y = np.r_[np.zeros(30, int), np.ones(50, int)]
            paths = [f'v{i}/f.png' for i in range(n)]
            probe = args.runs_root/'nonskin_probe'; probe.mkdir(parents=True)
            info = dict(settings=sb.nsf.SETTINGS, pickle_sha256='p', val_mean=.1, val_std=.5, ffpp_val_auc=.7)
            (probe/'xception_face_inner_seed42.json').write_text(json.dumps(info))
            inner = rng.normal(size=n)+y
            pd.DataFrame(dict(crop_path=paths, score=inner, scorer_pickle_sha256='p')).to_csv(probe/'xception_cdf_test_face_inner_scores.csv', index=False)
            scores = args.output_root/'scores'; scores.mkdir(parents=True)
            pd.DataFrame(dict(crop_path=paths, clean=inner+1e-7)).to_csv(scores/'cdf_test_normal.csv', index=False)
            for view in sb.FILTERS:
                pd.DataFrame(dict(crop_path=paths, smooth_pix=inner+rng.normal(scale=.1, size=n),
                                  smooth_all=inner+rng.normal(scale=.3, size=n))).to_csv(scores/f'cdf_test_{view}.csv', index=False)
                pd.DataFrame(dict(crop_path=paths, roi_status='detected', iou=.9, area_ratio=1.)).to_csv(scores/f'mask_agreement_{view}.csv', index=False)
            src = args.evaluation_root/'seed42/cdf_test'; src.mkdir(parents=True)
            cal_dir = args.pair_nsf_root/'calibration'; cal_dir.mkdir(parents=True)
            ref_rows = []
            for filt in sb.FILTERS:
                cal = dict(identity=dict(source_checkpoint_sha256='h'+filt), rank_val_mean=.2, rank_val_std=1.5,
                           scorer_val_mean=info['val_mean'], scorer_val_std=info['val_std'], weight=.2)
                (cal_dir/f'seed42_{filt}.json').write_text(json.dumps(cal))
                base = rng.normal(size=n)+y
                margins = {}
                for view in sb.VIEWS:
                    m = base if view == 'normal' else base+rng.normal(scale=.3, size=n)+.5
                    stem = 'normal' if view == 'normal' else view+'_smooth_strong'
                    pd.DataFrame(dict(crop_path=paths, video_id=paths, class_index=y, raw_margin=m,
                                      checkpoint_sha256='h'+filt)).to_csv(src/f'xception_aug_pair@{filt}_{stem}.csv', index=False)
                    margins[view] = sb.nsf.fused(m, inner, cal)
                for ev in sb.FILTERS:
                    r = sb.rc.metrics(y, margins['normal'], margins[ev])
                    ref_rows.append(dict(train_filter=filt, filter=ev, fp_cross_auc=r['fp_cross_auc'], fn_cross_auc=r['fn_cross_auc']))
            (args.pair_nsf_root/'seed42/cdf_test').mkdir(parents=True)
            pd.DataFrame(ref_rows).to_csv(args.pair_nsf_root/'seed42/cdf_test/summary.csv', index=False)
            sb.evaluate(args)
            head = pd.read_csv(args.output_root/'smoothed_b_headline.csv')
            self.assertEqual(head.variant.tolist(), ['none', *sb.VARIANTS])
            self.assertAlmostEqual(float(head.set_index('variant').loc['saved', 'delta_all16_vs_saved']), 0.)
            compare = pd.read_csv(args.output_root/'b_compare.csv')
            self.assertTrue((compare[compare.variant.eq('saved')].mean_abs_delta_z == 0).all())
            self.assertIn('mask_iou_mean', compare)
            bad = pd.DataFrame(ref_rows).assign(fp_cross_auc=lambda t: t.fp_cross_auc+.01)
            bad.to_csv(args.pair_nsf_root/'seed42/cdf_test/summary.csv', index=False)
            with self.assertRaises(ValueError):
                sb.evaluate(args)


if __name__ == '__main__':
    unittest.main()
