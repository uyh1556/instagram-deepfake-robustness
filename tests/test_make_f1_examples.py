"""CPU check of the F1 example picker: flip rules, one frame per video, previews and the final figure."""
import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import beauty_smoothing_v2 as v2
import make_f1_examples as f1

MODELS = [m for m, _ in f1.MODELS]


def logit(p):
    return np.log(p/(1-p))


class F1Tests(unittest.TestCase):
    def test_pick_and_render(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cdf, ev = root/'cdf', root/'eval'
            rows = []   # (crop, video, class, roi, {model: (normal, bilateral, dt)})
            flip_real = {m: (.2, .8, .3) for m in MODELS}
            weak_real = {m: (.45, .55, .3) for m in MODELS}
            partial_real = {**flip_real, 'xception': (.2, .4, .3)}      # Xception does not flip -> excluded
            flip_fake = {'clip_b16': (.7, .9, .8), 'dinov2_b14': (.9, .9, .2), 'xception': (.8, .8, .3)}
            spec = [('r0.png', 'v0', 0, 'detected', flip_real), ('r1.png', 'v0', 0, 'detected', weak_real),
                    ('r2.png', 'v1', 0, 'detected', weak_real), ('r3.png', 'v2', 0, 'detected', partial_real),
                    ('r4.png', 'v3', 0, 'fallback_letterbox', flip_real), ('f0.png', 'v4', 1, 'detected', flip_fake)]
            for path, _, _, _, _ in spec:
                (cdf/path).parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.full((16, 16, 3), 120, np.uint8)).save(cdf/path)
            (cdf/'manifests').mkdir()
            pd.DataFrame([dict(crop_path=p, video_id=v, class_index=str(y), roi_status=r) for p, v, y, r, _ in spec]).to_csv(
                cdf/'manifests/celebdf_v2_roi16_test_v1.csv', index=False)
            for m in MODELS:
                out = ev/f'ffpp_c23_smoothing_v2_diagonal_{m}'/'seed42'/'cdf_test'; out.mkdir(parents=True)
                for k, stem in enumerate(['normal', 'bilateral_smooth_strong', 'domain_transform_smooth_strong']):
                    pd.DataFrame([dict(crop_path=p, video_id=v, class_index=y, raw_margin=logit(s[m][k])) for p, v, y, _, s in spec]).to_csv(
                        out/f'{m}_baseline_{stem}.csv', index=False)
            cache = root/'cache'/'cdf_test'
            for part in ['mask', 'bilateral', 'domain_transform']:
                (cache/part).mkdir(parents=True)
                (cache/part/'identity.json').write_text(json.dumps({'dataset': 'cdf_test'}))
                (cache/part/'complete.json').write_text('{}')
                for p, *_ in spec:
                    img = v2.mask_to_image(np.ones((16, 16), np.float32)) if part == 'mask' else Image.fromarray(np.full((16, 16, 3), 200, np.uint8))
                    img.save(cache/part/p)
            args = argparse.Namespace(eval_root=ev, cdf_root=cdf, archive_dir=root/'drive', output_root=root/'out', cache_root=root/'cache',
                                      seed=42, top=8, real_pick=0, fake_pick=0)
            picks = f1.run(args)
            self.assertEqual(list(picks['real'].crop_path), ['r0.png', 'r2.png'])   # one per video, strongest flip first
            self.assertEqual(list(picks['fake'].crop_path), ['f0.png'])            # CLIP not required for the fake case
            self.assertAlmostEqual(picks['real'].flip_margin.iloc[0], .3)
            self.assertTrue((root/'out/F1_examples.png').is_file())
            self.assertTrue((root/'out/F1_candidate_real_1.png').is_file())


if __name__ == '__main__':
    unittest.main()
