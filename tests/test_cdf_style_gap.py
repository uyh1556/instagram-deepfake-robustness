"""CPU checks for the FDA style-gap diagnostic: the swap itself and a tiny end-to-end run."""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import diagnose_cdf_style_gap as style
import test_smoothing_preservation as fixture


class StyleGapTests(unittest.TestCase):
    setUp = fixture.PreservationTests.setUp
    tearDown = fixture.PreservationTests.tearDown

    def test_swap_replaces_only_low_frequency_amplitude(self):
        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, (32, 32, 3)).astype(np.float64)
        reference = style.amplitude(rng.integers(0, 256, (32, 32, 3)).astype(np.float64))
        self.assertTrue(np.array_equal(style.swap_low_frequency(image, style.amplitude(image), .1), image.astype(np.uint8)))
        out = style.swap_low_frequency(image, reference, .1).astype(np.float64)
        c, b = 16, 3
        # Low band now follows the reference; far band and phase come from the image (up to uint8 rounding).
        self.assertLess(np.abs(style.amplitude(out)[c-b:c+b+1, c-b:c+b+1]-reference[c-b:c+b+1, c-b:c+b+1]).mean(),
                        np.abs(style.amplitude(image)[c-b:c+b+1, c-b:c+b+1]-reference[c-b:c+b+1, c-b:c+b+1]).mean()/10)
        self.assertLess(np.abs(style.amplitude(out)[:4, :4]-style.amplitude(image)[:4, :4]).mean()/style.amplitude(image)[:4, :4].mean(), .05)
        self.assertEqual(out.shape, image.shape)

    def test_end_to_end_reuses_and_reports_every_variant(self):
        a = copy.copy(self.args)
        a.models, a.strategies, a.betas, a.size, a.reference_images = ['xception'], list(style.STRATEGIES), [.05], 16, 8
        a.ce_rank_runs_root, a.evaluation_pixel_cache = self.root/'ce_runs', self.root/'pixels'
        run = a.ce_rank_runs_root/style.rank.run_name('xception', 'ce_rank', 42)
        run.mkdir(parents=True)
        weights = {k: v+.1 for k, v in fixture.Tiny().state_dict().items()}
        torch.save(dict(config=dict(strategy='ce_rank', initial_checkpoint_sha256=self.initial_hashes['xception']),
                        model_state_dict=weights), run/'best.pt')
        style.evaluate(a)
        table = pd.read_csv(a.output_root/'summary.csv')
        self.assertEqual(set(zip(table.strategy, table.variant)),
                         {(s, v) for s in style.STRATEGIES for v in ['resize', 'ffpp_b0.05', 'cdf_b0.05']})
        self.assertTrue((table[table.variant.eq('resize')].normal_auc_vs_resize == 0).all())
        self.assertTrue(table[['normal_auc', 'strong_auc', 'strong_real_vs_normal_fake']].stack().between(0, 1).all())
        times = {p.name: p.stat().st_mtime_ns for p in a.output_root.glob('*_normal.csv')}
        style.evaluate(a)
        self.assertEqual(times, {p.name: p.stat().st_mtime_ns for p in a.output_root.glob('*_normal.csv')})
        bad = copy.copy(a); bad.ce_rank_runs_root = self.root/'other'
        torch.save(dict(config=dict(strategy='ce_rank', initial_checkpoint_sha256='x'), model_state_dict=weights),
                   (bad.ce_rank_runs_root/run.name).mkdir(parents=True) or bad.ce_rank_runs_root/run.name/'best.pt')
        with self.assertRaisesRegex(ValueError, 'baseline'):
            style.checkpoints(bad, 'xception')


if __name__ == '__main__':
    unittest.main()
