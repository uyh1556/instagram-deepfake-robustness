"""CPU check of the seed-mean summary of the v2 diagonal experiment."""
import argparse
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import summarize_v2_diagonal as summary

F = summary.FILTERS


def write(root, model, seed, rank_bump, nsf_bump):
    rows = []
    def add(strategy, method, train, filt, clean, fp, fn):
        rows.append({'model': model, 'strategy': strategy, 'train_filter': train, 'method': method, 'filter': filt,
                     'held_out': train is not None and filt != train, 'normal_auc': clean, 'fp_cross_auc': fp, 'fn_cross_auc': fn,
                     'fp_flip@normal5': .1, 'fn_flip@normal5': .05, 'real_shift': .2, 'fake_shift': .1, 'max_gap': clean-min(fp, fn)})
    for f in F:
        add('baseline', 'baseline', None, f, .70, .50, .80)
    for train in F:
        for f in F:
            if train == 'domain_transform':   # this rank run fell back to epoch 0
                add(f'aug_pair_rank@{train}', 'aug_pair_rank', train, f, .70, .50, .80)
            else:
                add(f'aug_pair_rank@{train}', 'aug_pair_rank', train, f, .72, .60+rank_bump+(.1 if f == train else 0), .80)
            add(f'aug_pair_rank_nsf@{train}', 'aug_pair_rank_nsf', train, f, .73, .62+nsf_bump+(.1 if f == train else 0), .80)
    out = root/f'ffpp_c23_smoothing_v2_diagonal_{model}'/f'seed{seed}'/'cdf_test'
    out.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(out/'summary.csv', index=False)


class SummaryTests(unittest.TestCase):
    def test_headline(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write(root, 'clip_b16', 42, 0., 0.)
            write(root, 'clip_b16', 43, .02, .02)
            head, means = summary.run(argparse.Namespace(eval_root=root, output_root=root/'out', models=['clip_b16'], seeds=[42, 43, 44],
                                                         dataset='cdf_test'))
            h = head.set_index('method')
            self.assertEqual(list(h.index), ['baseline', 'aug_pair_rank', 'aug_pair_rank_nsf'])
            self.assertEqual(h.loc['aug_pair_rank', 'seeds'], 2)
            self.assertEqual(h.loc['aug_pair_rank', 'epoch0_runs'], 2)        # DT run, both seeds
            self.assertEqual(h.loc['aug_pair_rank_nsf', 'epoch0_runs'], 0)
            self.assertAlmostEqual(h.loc['aug_pair_rank_nsf', 'diag_worse_mean'], .73)   # min(.62+.01+.1, .80) mean over seeds
            self.assertAlmostEqual(h.loc['aug_pair_rank_nsf', 'heldout_worse_mean'], .63)
            self.assertAlmostEqual(h.loc['aug_pair_rank_nsf', 'clean_auc_mean'], .73)
            self.assertAlmostEqual(h.loc['baseline', 'all_worse_mean'], .50)
            self.assertEqual(h.loc['aug_pair_rank_nsf', 'cells_better_than_rank'], 16)
            self.assertTrue(np.isnan(h.loc['aug_pair_rank', 'cells_better_than_rank']))
            self.assertEqual(len(means[means.method.eq('aug_pair_rank')]), 16)
            self.assertTrue((root/'out/v2_diagonal_headline.csv').is_file())


if __name__ == '__main__':
    unittest.main()
