"""Are the four additional-training methods different? Friedman test over filter x seed blocks (2026-11-23).

Committee request (Prof. 백성용): are the differences in the training-method comparison table noise?
Methods: aug_single (random aug), aug_pair (pair aug), aug_pair_rank (cross ranking),
aug_pair_rank_fc100_ramp2 (feature consistency). They were trained separately, but every method uses the same
4 train filters x seeds 42/43/44, so each (train filter, seed) is one block of 4 values (12 blocks per backbone).
Per-run metric as in paired_test_pair_nsf_v2.py (mean over its four test filters); the frozen
comparison_rows.csv must reproduce pair_nsf_headline.csv. Per backbone and metric: Friedman chi2 + Kendall's W,
mean rank per method (1 = best), then all 6 pairwise two-sided exact Wilcoxon signed-rank and paired t tests with
Holm within the backbone. Nothing is trained, inferred or tuned. Weaker pairing than pair vs pair+NSF (separate
training), and only 12 blocks: a non-significant result means "no evidence of a difference", not equivalence.
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from paired_test_pair_nsf_v2 import MODELS, METRICS, check_headline, holm, run_values

METHODS = ['aug_single', 'aug_pair', 'aug_pair_rank', 'aug_pair_rank_fc100_ramp2']
NAMES = {'aug_single': '무작위 증강', 'aug_pair': '동시 증강', 'aug_pair_rank': '교차 순위',
         'aug_pair_rank_fc100_ramp2': '특징 일관성'}


def blocks(rows, model, head):
    wide = {}
    for m in METHODS:
        v = run_values(rows, m)
        check_headline(v, head, model, m)
        wide[m] = v.set_index(['train_filter', 'seed'])
    keys = wide[METHODS[0]].index
    for m in METHODS:
        if len(wide[m]) != 12 or not wide[m].index.sort_values().equals(keys.sort_values()):
            raise SystemExit(f'{model} {m}: expected the same 12 (train filter, seed) runs as {METHODS[0]}')
    return {metric: pd.DataFrame({m: wide[m].loc[keys, metric] for m in METHODS}) for metric in METRICS}


def friedman(x):
    chi2, p = stats.friedmanchisquare(*[x[m] for m in METHODS])
    n, k = x.shape
    ranks = x.rank(axis=1, ascending=False)  # 1 = highest AUC in the block
    return dict(n_blocks=n, chi2=chi2, p_friedman=p, kendall_w=chi2/(n*(k-1)),
                **{f'mean_{m}': x[m].mean() for m in METHODS}, **{f'rank_{m}': ranks[m].mean() for m in METHODS})


def pairwise(x):
    out = []
    for a, b in combinations(METHODS, 2):
        d = x[b]-x[a]
        out.append(dict(a=a, b=b, diff_mean=d.mean(), diff_sd=d.std(ddof=1), b_higher=int((d > 0).sum()),
                        b_lower=int((d < 0).sum()), p_t=stats.ttest_rel(x[b], x[a]).pvalue,
                        p_wilcoxon=stats.wilcoxon(x[b], x[a], method='exact').pvalue if (d != 0).any() else 1.0))
    out = pd.DataFrame(out)
    out['p_t_holm'] = holm(out.p_t.to_numpy())
    out['p_wilcoxon_holm'] = holm(out.p_wilcoxon.to_numpy())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evaluations', type=Path, required=True, help='Drive evaluations/ folder')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    omnibus, pairs = [], []
    for model in MODELS:
        folder = args.evaluations/f'ffpp_c23_pair_nsf_v2_{model}'
        data = blocks(pd.read_csv(folder/'comparison_rows.csv'), model, pd.read_csv(folder/'pair_nsf_headline.csv'))
        for metric, x in data.items():
            omnibus.append(dict(model=model, metric=metric, **friedman(x)))
            pairs.append(pairwise(x).assign(model=model, metric=metric))
    omnibus, pairs = pd.DataFrame(omnibus), pd.concat(pairs, ignore_index=True)
    args.output.mkdir(parents=True, exist_ok=True)
    omnibus.to_csv(args.output/'friedman.csv', index=False)
    pairs.to_csv(args.output/'pairwise.csv', index=False)
    pd.set_option('display.width', 220)
    fmt = lambda v: f'{v:.4g}'  # noqa: E731
    cols = ['model', 'metric', 'chi2', 'p_friedman', 'kendall_w']+[f'rank_{m}' for m in METHODS]
    print(omnibus[cols].rename(columns={f'rank_{m}': NAMES[m] for m in METHODS}).to_string(index=False, float_format=fmt))
    main_pairs = pairs[pairs.metric.eq('all_worse')].replace(NAMES)
    print('\nall_worse pairwise (b - a):')
    print(main_pairs[['model', 'a', 'b', 'diff_mean', 'diff_sd', 'b_higher', 'b_lower', 'p_wilcoxon', 'p_wilcoxon_holm',
                      'p_t_holm']].to_string(index=False, float_format=fmt))
    print('saved:', args.output, flush=True)


if __name__ == '__main__':
    main()
