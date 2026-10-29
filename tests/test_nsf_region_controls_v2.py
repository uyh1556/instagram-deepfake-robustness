"""Region masks, metric definitions and the fusion/reproduction path of the region controls, without neural inference."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import nsf_region_controls_v2 as rc

IDS = {n: i for i, n in enumerate(['background', *rc.FACE, 'hair'])}


class MaskTests(unittest.TestCase):
    def setUp(self):
        self.label = np.zeros((60, 50), dtype=np.uint8)
        self.label[10:50, 10:40] = IDS['skin']
        self.label[20:24, 15:22] = IDS['l_eye']
        self.label[38:42, 18:32] = IDS['u_lip']
        self.label[:5] = IDS['hair']

    def test_random_square_matches_inner_area_inside_face_and_is_fixed(self):
        face = np.isin(self.label, [IDS[n] for n in rc.FACE])
        area = int(np.isin(self.label, [IDS[n] for n in rc.fusion.INNER]).sum())
        a = rc.random_square(face, area, '42|x.png')
        self.assertTrue(np.array_equal(a, rc.random_square(face, area, '42|x.png')))
        self.assertLessEqual(abs(int(a.sum())-area), 2*int(np.sqrt(area))+1)
        self.assertTrue((a & face).any())
        draws = {rc.random_square(face, area, f'42|{i}.png').tobytes() for i in range(10)}
        self.assertGreater(len(draws), 1)
        self.assertFalse(rc.random_square(face, 0, 'k').any())

    def test_keep_masks_face_covers_inner_and_excludes_hair(self):
        keep = rc.keep_masks(self.label, IDS, '42|x.png')
        self.assertEqual(keep['face'].shape, self.label.shape)
        self.assertGreater(keep['face'][30, 25], .9)
        self.assertLess(keep['face'][1, 25], .05)
        for m in keep.values():
            self.assertTrue((m >= 0).all() and (m <= 1).all())


class ViewTests(unittest.TestCase):
    def test_region_views_returns_full_face_random_with_grey_outside(self):
        import torch
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            label = np.zeros((40, 40), dtype=np.uint8)
            label[8:32, 8:32] = IDS['skin']
            label[14:18, 12:20] = IDS['l_eye']
            Image.fromarray(np.full((40, 40, 3), 10, np.uint8)).save(root/'a.png')
            args = argparse.Namespace(work_dir=root/'work')
            out = rc.label_path(args.work_dir, 'cdf_test', 'a.png'); out.parent.mkdir(parents=True)
            Image.fromarray(label, mode='L').save(out)
            frame = pd.DataFrame(dict(crop_path=['a.png'], roi_status=['detected']))
            to_tensor = lambda im: torch.from_numpy(np.asarray(im, dtype=np.float32))
            x, i = rc.RegionViews(frame, root, 'cdf_test', args, IDS, to_tensor)[0]
            self.assertEqual(tuple(x.shape), (3, 40, 40, 3))
            self.assertTrue(torch.all(x[0] == 10))
            self.assertAlmostEqual(float(x[1, 0, 0, 0]), 128., delta=1)
            self.assertLess(float(x[1, 20, 20, 0]), 20)


class MetricTests(unittest.TestCase):
    def test_metrics_match_threshold_shift_measure(self):
        rng = np.random.default_rng(0)
        y = np.r_[np.zeros(40, int), np.ones(60, int)]
        normal = rng.normal(size=100)+y
        strong = normal+.4+rng.normal(scale=.2, size=100)
        margins = {c: normal+i*(strong-normal)/3 for i, c in enumerate(rc.shift.CONDITIONS)}
        ref, mine = rc.shift.measure(y, margins), rc.metrics(y, normal, strong)
        for a, b in [('normal_auc', 'clean_auc'), ('fp_cross_auc', 'fp_cross_auc'), ('fn_cross_auc', 'fn_cross_auc'),
                     ('real_shift', 'real_shift'), ('fake_shift', 'fake_shift')]:
            self.assertAlmostEqual(ref[a], mine[b])
        self.assertEqual(mine['worse_cross_auc'], min(mine['fp_cross_auc'], mine['fn_cross_auc']))


class EvaluateTests(unittest.TestCase):
    def test_evaluate_reproduces_face_inner_and_writes_headline(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = argparse.Namespace(model='xception', seed=42, runs_root=root/'runs', evaluation_root=root/'eval',
                                      pair_nsf_root=root/'pair_nsf', output_root=root/'out')
            rng = np.random.default_rng(1)
            n = 80
            y = np.r_[np.zeros(30, int), np.ones(50, int)]
            paths = [f'v{i}/f.png' for i in range(n)]
            probe = args.runs_root/'nonskin_probe'; probe.mkdir(parents=True)
            info = dict(settings=rc.nsf.SETTINGS, pickle_sha256='p', val_mean=.1, val_std=.5, ffpp_val_auc=.7)
            (probe/'xception_face_inner_seed42.json').write_text(json.dumps(info))
            inner = rng.normal(size=n)+y
            pd.DataFrame(dict(crop_path=paths, score=inner, scorer_pickle_sha256='p')).to_csv(probe/'xception_cdf_test_face_inner_scores.csv', index=False)
            scores = args.output_root/'scores'; scores.mkdir(parents=True)
            stats = {r: dict(val_mean=0., val_std=1., ffpp_val_auc=.6) for r in rc.REGIONS}
            (scores/'probes.json').write_text(json.dumps(dict(stats=stats)))
            for view in rc.VIEWS:
                pd.DataFrame(dict(crop_path=paths, **{r: rng.normal(size=n)+y for r in rc.REGIONS})).to_csv(scores/f'cdf_test_{view}.csv', index=False)
            src = args.evaluation_root/'seed42/cdf_test'; src.mkdir(parents=True)
            cal_dir = args.pair_nsf_root/'calibration'; cal_dir.mkdir(parents=True)
            ref_rows = []
            for filt in rc.FILTERS:
                cal = dict(identity=dict(source_checkpoint_sha256='h'+filt), rank_val_mean=.2, rank_val_std=1.5,
                           scorer_val_mean=info['val_mean'], scorer_val_std=info['val_std'], weight=.2)
                (cal_dir/f'seed42_{filt}.json').write_text(json.dumps(cal))
                base = rng.normal(size=n)+y
                margins = {}
                for view in rc.VIEWS:
                    m = base if view == 'normal' else base+rng.normal(scale=.3, size=n)+.5
                    stem = 'normal' if view == 'normal' else view+'_smooth_strong'
                    pd.DataFrame(dict(crop_path=paths, video_id=paths, class_index=y, raw_margin=m,
                                      checkpoint_sha256='h'+filt)).to_csv(src/f'xception_aug_pair@{filt}_{stem}.csv', index=False)
                    margins[view] = rc.nsf.fused(m, inner, cal)
                for ev in rc.FILTERS:
                    r = rc.metrics(y, margins['normal'], margins[ev])
                    ref_rows.append(dict(train_filter=filt, filter=ev, fp_cross_auc=r['fp_cross_auc'], fn_cross_auc=r['fn_cross_auc']))
            (args.pair_nsf_root/'seed42/cdf_test').mkdir(parents=True)
            pd.DataFrame(ref_rows).to_csv(args.pair_nsf_root/'seed42/cdf_test/summary.csv', index=False)
            rc.evaluate(args)
            head = pd.read_csv(args.output_root/'region_headline.csv')
            self.assertEqual(head.region.tolist(), ['none', 'face_inner', *rc.REGIONS])
            self.assertTrue((args.output_root/'b_alone.csv').is_file())
            # A broken reference must be caught.
            bad = pd.DataFrame(ref_rows).assign(fp_cross_auc=lambda t: t.fp_cross_auc+.01)
            bad.to_csv(args.pair_nsf_root/'seed42/cdf_test/summary.csv', index=False)
            with self.assertRaises(ValueError):
                rc.evaluate(args)


if __name__ == '__main__':
    unittest.main()
