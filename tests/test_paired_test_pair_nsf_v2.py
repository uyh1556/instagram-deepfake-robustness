import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import paired_test_pair_nsf_v2 as pt  # noqa: E402
import summarize_v2_diagonal as summary  # noqa: E402

FILTERS = ['bilateral', 'weighted_median', 'fast_global', 'domain_transform']


def fake_rows(seed_rng, gain):
    rows = []
    bases = {(seed, train): seed_rng.normal(.6, .02) for seed in [42, 43, 44] for train in FILTERS}  # shared run
    for seed in [42, 43, 44]:
        for method, g in [('aug_pair', 0.0), ('aug_pair_nsf', gain)]:
            for train in FILTERS:
                base = bases[seed, train]
                for f in FILTERS:
                    w = base + g + seed_rng.normal(0, .005)
                    rows.append(dict(model='m', method=method, seed=seed, train_filter=train, filter=f,
                                     normal_auc=.7+g, fp_cross_auc=w, fn_cross_auc=w+.05, worse_cross_auc=w,
                                     max_gap=.1, **{'fp_flip@normal5': .05, 'fn_flip@normal5': .05},
                                     real_shift=.1, fake_shift=-.1, epoch0=False))
    return pd.DataFrame(rows)


class PairedTest(unittest.TestCase):
    def test_run_values_reproduce_headline_and_detect_gain(self):
        rows = fake_rows(np.random.default_rng(0), .02)
        head = summary.headline(rows).drop(columns=['epoch0_runs'])
        v = {k: pt.run_values(rows, m) for k, m in pt.METHODS.items()}
        for k, m in pt.METHODS.items():
            pt.check_headline(v[k], head, 'm', m)  # must not raise
        self.assertEqual(len(v['pair']), 12)
        j = v['pair'].merge(v['pair_nsf'], on=['train_filter', 'seed'], suffixes=('_pair', '_pair_nsf'))
        r = pt.paired(j.all_worse_pair, j.all_worse_pair_nsf)
        self.assertGreater(r['diff_mean'], 0)
        self.assertLess(r['p_t'], .01)

    def test_headline_mismatch_stops(self):
        rows = fake_rows(np.random.default_rng(1), .0)
        head = summary.headline(rows).drop(columns=['epoch0_runs'])
        head.loc[head.method.eq('aug_pair'), 'all_worse_mean'] += .01
        with self.assertRaises(SystemExit):
            pt.check_headline(pt.run_values(rows, 'aug_pair'), head, 'm', 'aug_pair')

    def test_holm(self):
        np.testing.assert_allclose(pt.holm(np.array([.01, .04, .03])), [.03, .06, .06])

    def test_main_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            ev = Path(d)
            for i, model in enumerate(pt.MODELS):
                rows = fake_rows(np.random.default_rng(10+i), .02).assign(model=model)
                folder = ev/f'ffpp_c23_pair_nsf_v2_{model}'
                folder.mkdir()
                rows.to_csv(folder/'comparison_rows.csv', index=False)
                summary.headline(rows).drop(columns=['epoch0_runs']).to_csv(folder/'pair_nsf_headline.csv', index=False)
            sys.argv = ['x', '--evaluations', str(ev), '--output', str(ev/'out')]
            pt.main()
            res = pd.read_csv(ev/'out'/'paired_tests.csv')
            self.assertEqual(len(res), 3*len(pt.METRICS))
            self.assertTrue(res[res.metric.eq('all_worse')].p_t_holm.notna().all())


if __name__ == '__main__':
    unittest.main()
