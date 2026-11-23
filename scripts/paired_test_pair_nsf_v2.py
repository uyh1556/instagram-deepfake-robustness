"""Paired significance test: pair+NSF vs pair on the CDF test (committee request, 2026-11-23).

pair+NSF fuses score B into the SAME aug_pair checkpoint, so each (train filter, seed) run gives one paired
difference. Unit = run (4 train filters x seeds 42/43/44 = 12 per backbone); per-run metric = mean over its
four test filters, so the mean of the 12 run values equals the headline seed mean. Reads the frozen
comparison_rows.csv written by evaluate_pair_nsf_v2.py and checks it reproduces pair_nsf_headline.csv
(1e-9); nothing is inferred, trained or tuned. Tests per backbone and metric (two-sided): paired t,
Wilcoxon signed-rank (exact), sign test; Holm over the three backbones for the main metric (all_worse).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

MODELS = ['xception', 'clip_b16', 'dinov2_b14']  # report order: Xception -> CLIP -> DINO
METHODS = {'pair': 'aug_pair', 'pair_nsf': 'aug_pair_nsf'}
METRICS = {'all_worse': None, 'heldout_worse': 'held', 'diag_worse': 'diag', 'clean_auc': 'clean'}
MAIN = 'all_worse'


def run_values(rows, method):
    """One value per (train_filter, seed) run for every metric."""
    t = rows[rows.method.eq(method)]
    out = []
    for (train, seed), g in t.groupby(['train_filter', 'seed'], sort=True):
        seen = g['filter'].eq(train)
        out.append(dict(train_filter=train, seed=int(seed), all_worse=g.worse_cross_auc.mean(),
                        heldout_worse=g[~seen].worse_cross_auc.mean(), diag_worse=g[seen].worse_cross_auc.mean(),
                        clean_auc=g.normal_auc.iloc[0]))
    return pd.DataFrame(out)


def check_headline(values, head, model, method):
    row = head[head.method.eq(method)].iloc[0]
    for m in METRICS:
        got = values.groupby('seed')[m].mean()
        if abs(got.mean()-row[m+'_mean']) > 1e-9 or abs(got.std()-row[m+'_sd']) > 1e-9:
            raise SystemExit(f'{model} {method} {m}: rows do not reproduce pair_nsf_headline.csv')


def paired(a, b):
    d = b-a
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    return dict(n=len(d), pair_mean=a.mean(), pair_nsf_mean=b.mean(), diff_mean=d.mean(), diff_sd=d.std(ddof=1),
                diff_min=d.min(), diff_max=d.max(), runs_up=pos, runs_down=neg,
                t=stats.ttest_rel(b, a).statistic, p_t=stats.ttest_rel(b, a).pvalue,
                p_wilcoxon=stats.wilcoxon(b, a, method='exact').pvalue if (d != 0).any() else 1.0,
                p_sign=stats.binomtest(pos, pos+neg, .5).pvalue if pos+neg else 1.0)


def holm(p):
    order = np.argsort(p)
    adj, running = np.empty(len(p)), 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (len(p)-k)*p[i]))
        adj[i] = running
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evaluations', type=Path, required=True, help='Drive evaluations/ folder')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    results, per_run = [], []
    for model in MODELS:
        folder = args.evaluations/f'ffpp_c23_pair_nsf_v2_{model}'
        rows = pd.read_csv(folder/'comparison_rows.csv')
        head = pd.read_csv(folder/'pair_nsf_headline.csv')
        v = {k: run_values(rows, m) for k, m in METHODS.items()}
        for k, m in METHODS.items():
            check_headline(v[k], head, model, m)
        j = v['pair'].merge(v['pair_nsf'], on=['train_filter', 'seed'], suffixes=('_pair', '_pair_nsf'), validate='1:1')
        if len(j) != 12:
            raise SystemExit(f'{model}: expected 12 paired runs, found {len(j)}')
        per_run.append(j.assign(model=model))
        for m in METRICS:
            results.append(dict(model=model, metric=m, **paired(j[m+'_pair'], j[m+'_pair_nsf'])))
    res = pd.DataFrame(results)
    main_rows = res.metric.eq(MAIN)
    for col in ['p_t', 'p_wilcoxon', 'p_sign']:
        res.loc[main_rows, col+'_holm'] = holm(res.loc[main_rows, col].to_numpy())
    args.output.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.output/'paired_tests.csv', index=False)
    pd.concat(per_run).to_csv(args.output/'paired_runs.csv', index=False)
    pd.set_option('display.width', 200)
    show = res[['model', 'metric', 'pair_mean', 'pair_nsf_mean', 'diff_mean', 'diff_sd', 'runs_up', 'runs_down',
                'p_t', 'p_wilcoxon', 'p_sign', 'p_t_holm', 'p_wilcoxon_holm']]
    print(show.to_string(index=False, float_format=lambda x: f'{x:.4g}'), flush=True)
    print('saved:', args.output, flush=True)


if __name__ == '__main__':
    main()
