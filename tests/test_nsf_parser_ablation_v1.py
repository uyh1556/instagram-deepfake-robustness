"""Mask building/restoring, mask variants and the headline verdict of the NSF parser ablation, without real parsers."""
import argparse
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch
from PIL import Image

import nsf_parser_ablation_v1 as pa


class FakeParser:
    """Eyes/mouth = a centred box whose size depends on the family."""
    def __init__(self, args, pad=3):
        self.pad = pad

    def hard(self, arrays):
        out = []
        for a in arrays:
            m = np.zeros(a.shape[:2], bool)
            m[self.pad:-self.pad, self.pad:-self.pad] = True
            out.append(m)
        return out


class MaskTests(unittest.TestCase):
    def test_keep_mask_matches_pair_nsf_and_morph_is_monotonic(self):
        hard = np.zeros((400, 480), bool)
        hard[120:280, 160:320] = True
        side = 400
        original = pa.v2.cv2.GaussianBlur(hard.astype(np.float32), (0, 0), max(1., pa.v2.PARSER['blur_fraction']*side)).clip(0, 1)
        np.testing.assert_array_equal(pa.keep_mask(hard), pa.v2.image_to_mask(pa.v2.mask_to_image(original)))
        areas = [(pa.keep_mask(hard, pa.MORPH.get(v, 0.)) > .5).sum() for v in ['erode1', 'segformer', 'dilate1']]
        self.assertEqual(areas, sorted(areas))
        self.assertEqual(len(set(areas)), 3)

    def test_landmark_mask_fills_polygons(self):
        self.assertFalse(pa.landmark_mask(None, (20, 30)).any())
        points = np.full((478, 2), .5)
        points[pa.MEDIAPIPE['polygons']['lips']] = [(.2, .2), (.8, .2), (.8, .8), (.2, .8)]*5
        m = pa.landmark_mask(points, (50, 50))
        self.assertTrue(m[25, 25])
        self.assertFalse(m[2, 2])

    def test_variant_order(self):
        self.assertEqual(pa.VARIANTS, ['segformer', 'erode1', 'dilate1', 'bisenet', 'mediapipe'])


class BuildTests(unittest.TestCase):
    def test_masks_pack_restore_and_variant_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frame = pd.DataFrame(dict(crop_path=['v0/a.png', 'v1/b.png'], roi_status=['detected', 'fallback'], class_index=['0', '1']))
            for p in frame.crop_path:
                (root/'cdf'/p).parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.full((24, 20, 3), 90, np.uint8)).save(root/'cdf'/p)
            sets = [('ffpp', frame, root/'cdf'), *((v, frame, root/'cdf') for v in pa.VIEWS)]

            class Cache:
                def view(self, image, path, name, strength, detected):
                    return image

            args = argparse.Namespace(mask_dir=root/'drive', work_dir=root/'work', cache_root=root/'c', archive_dir=root/'a',
                                      parse_batch=2, workers=0)
            pads = dict(segformer=3, bisenet=4, mediapipe=5)
            with mock.patch.object(pa, 'mask_sets', return_value=sets), \
                    mock.patch.object(pa.diagonal, 'restore_cache', return_value=Cache()):
                for family in pa.FAMILIES:
                    with mock.patch.dict(pa.PARSERS, {family: lambda a, p=pads[family]: FakeParser(a, p)}):
                        pa.masks(argparse.Namespace(**vars(args), family=family))
                    self.assertTrue((root/'drive'/f'parser_masks_{family}.tar').is_file())
            fresh = argparse.Namespace(**{**vars(args), 'work_dir': root/'fresh'})
            pa.restore_masks(fresh, sets)
            self.assertTrue(pa.read_hard(pa.mask_path(root/'fresh', 'bisenet', 'normal', 'v0/a.png'))[10, 10])
            with self.assertRaises(ValueError):
                pa.restore_masks(fresh, [('ffpp', pd.concat([frame, frame]), root/'cdf')])
            data = pa.VariantInputs(frame, root/'cdf', 'bilateral', root/'fresh', lambda im: torch.from_numpy(np.asarray(im).copy()), Cache())
            x, i, agree = data[0]
            self.assertEqual(tuple(x.shape), (len(pa.VARIANTS), 24, 20, 3))
            self.assertAlmostEqual(float(agree[0, 0]), 1.)          # segformer vs itself
            self.assertLess(float(agree[pa.VARIANTS.index('mediapipe'), 1]), 1.)   # smaller box
            self.assertTrue((x[pa.VARIANTS.index('mediapipe'), 0, 0] == pa.occlusion.FILL).all())


class HeadlineTests(unittest.TestCase):
    def test_runs_over_pair_and_verdict(self):
        rows = []
        for seed in [42, 43, 44]:
            for tf in pa.FILTERS:
                for ev in pa.FILTERS:
                    for v, worse in [('none', .60), *((v, .65) for v in pa.VARIANTS)]:
                        if v == 'bisenet':
                            worse = .62 if seed != 44 else .55      # 8/12 runs over pair
                        if v == 'mediapipe':
                            worse = .63                              # 12/12 but .02 below segformer
                        rows.append(dict(seed=seed, train_filter=tf, filter=ev, held_out=ev != tf, variant=v, clean_auc=.8,
                                         worse_cross_auc=worse, real_shift=0., fake_shift=0.))
        head, _ = pa.headline(pd.DataFrame(rows))
        self.assertEqual(head.variant.tolist(), [*pa.VARIANTS, 'none'])
        h = head.set_index('variant')
        self.assertEqual(h.loc['segformer', 'runs_over_pair'], 12)
        self.assertTrue(h.loc['dilate1', 'holds'])
        self.assertEqual(h.loc['bisenet', 'runs_over_pair'], 8)
        self.assertFalse(h.loc['bisenet', 'holds'])
        self.assertFalse(h.loc['mediapipe', 'holds'])
        self.assertAlmostEqual(h.loc['mediapipe', 'gap_to_segformer'], -.02)


if __name__ == '__main__':
    unittest.main()
