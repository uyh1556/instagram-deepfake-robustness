import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import friedman_test_training_methods_v2 as ft  # noqa: E402
import summarize_v2_diagonal as summary  # noqa: E402

FILTERS = ['bilateral', 'weighted_median', 'fast_global', 'domain_transform']


def fake_rows(rng, effects, model='m'):
    rows = []
    blocks = {(s, t): rng.normal(.6, .03) for s in [42, 43, 44] for t in FILTERS}
    for method, e in effects.items():
        for (seed, train), base in blocks.items():
            run = base + e + rng.normal(0, .01)
            for f in FILTERS:
                w = run + rng.normal(0, .003)
                rows.append(dict(model=model, method=method, seed=seed, train_filter=train, filter=f, normal_auc=.7+e,
                                 fp_cross_auc=w, fn_cross_auc=w+.05, worse_cross_auc=w, max_gap=.1,
                                 **{'fp_flip@normal5': .05, 'fn_flip@normal5': .05}, real_shift=.1, fake_shift=-.1,
                                 epoch0=False))
    return pd.DataFrame(rows)


class FriedmanTest(unittest.TestCase):
    def test_detects_a_clearly_worse_method_only(self):
        rows = fake_rows(np.random.default_rng(0), {'aug_single': 0, 'aug_pair': 0, 'aug_pair_rank': 0,
                                                    'aug_pair_rank_fc100_ramp2': -.05})
        head = summary.headline(rows).drop(columns=['epoch0_runs'])
        x = ft.blocks(rows, 'm', head)['all_worse']
        self.assertEqual(x.shape, (12, 4))
        r = ft.friedman(x)
        self.assertLess(r['p_friedman'], .001)
        self.assertGreater(r['rank_aug_pair_rank_fc100_ramp2'], 3.5)  # worst in (almost) every block
        p = ft.pairwise(x).set_index(['a', 'b'])
        self.assertLess(p.loc[('aug_pair', 'aug_pair_rank_fc100_ramp2'), 'p_wilcoxon_holm'], .05)
        self.assertGreater(p.loc[('aug_single', 'aug_pair'), 'p_wilcoxon'], .05)

    def test_missing_run_stops(self):
        rows = fake_rows(np.random.default_rng(1), dict.fromkeys(ft.METHODS, 0))
        head = summary.headline(rows).drop(columns=['epoch0_runs'])
        rows = rows[~(rows.method.eq('aug_pair') & rows.seed.eq(44) & rows.train_filter.eq('bilateral'))]
        with self.assertRaises(SystemExit):
            ft.blocks(rows, 'm', head)  # headline mismatch or wrong run count

    def test_main_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            ev = Path(d)
            for i, model in enumerate(ft.MODELS):
                rows = fake_rows(np.random.default_rng(5+i), dict.fromkeys(ft.METHODS, 0), model)
                folder = ev/f'ffpp_c23_pair_nsf_v2_{model}'
                folder.mkdir()
                rows.to_csv(folder/'comparison_rows.csv', index=False)
                summary.headline(rows).drop(columns=['epoch0_runs']).to_csv(folder/'pair_nsf_headline.csv', index=False)
            sys.argv = ['x', '--evaluations', str(ev), '--output', str(ev/'out')]
            ft.main()
            self.assertEqual(len(pd.read_csv(ev/'out'/'friedman.csv')), 3*len(ft.METRICS))
            self.assertEqual(len(pd.read_csv(ev/'out'/'pairwise.csv')), 3*len(ft.METRICS)*6)


if __name__ == '__main__':
    unittest.main()
