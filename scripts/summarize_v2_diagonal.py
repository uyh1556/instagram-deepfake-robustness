"""One readable summary of the v2 diagonal experiment over seeds (reads the saved summary.csv files only; no GPU).

For every backbone x condition: mean +- std over seeds of
  clean_auc       original-image AUC (mean over the 4 training filters)
  diag_worse      worse cross AUC = min(strong-real vs clean-fake, clean-real vs strong-fake) on the TRAINING filter
  heldout_worse   the same on the 3 filters NOT trained on (mean over the 12 held-out cells)
  dt_heldout      held-out DT only (the laundering case; mean over the 3 non-DT training filters)
  all_worse       mean over all 16 cells (Baseline: its 4 filters) -- the one-number comparison
  fp_flip/fn_flip decision flips at the clean 5% FPR threshold (strong smoothing), mean over the 16 cells
  epoch0_runs     how many (training filter, seed) runs selected epoch 0 (= identical to the Baseline)
plus, per backbone, how many of the 16 seed-mean cells beat aug_pair_rank. Writes
  v2_diagonal_seed_means.csv  (backbone x condition x train filter x eval filter, seed mean/std)
  v2_diagonal_headline.csv    (the table above)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FILTERS = ['bilateral', 'weighted_median', 'fast_global', 'domain_transform']
SHORT = {'bilateral': 'bil', 'weighted_median': 'WMF', 'fast_global': 'FGS', 'domain_transform': 'DT'}
ORDER = ['baseline', 'aug_single', 'aug_pair', 'aug_pair_rank', 'aug_pair_rank_fc100_ramp2', 'aug_pair_rank_nsf']


def load(args):
    frames = []
    for model in args.models:
        for seed in args.seeds:
            path = args.eval_root/f'ffpp_c23_smoothing_v2_diagonal_{model}'/f'seed{seed}'/args.dataset/'summary.csv'
            if not path.is_file():
                print('missing:', path, flush=True)
                continue
            t = pd.read_csv(path).assign(seed=seed)
            t['worse_cross_auc'] = np.minimum(t.fp_cross_auc, t.fn_cross_auc)
            base = t[t.method.eq('baseline')].set_index('filter')
            # epoch 0 = the run's rows are exactly the baseline rows (best.pt fell back to the starting checkpoint)
            t['epoch0'] = [r.method != 'baseline' and np.isclose(r.normal_auc, base.loc[r['filter'], 'normal_auc'])
                           and np.isclose(r.fp_cross_auc, base.loc[r['filter'], 'fp_cross_auc']) for _, r in t.iterrows()]
            frames.append(t)
    if not frames:
        raise FileNotFoundError('No summary.csv found')
    return pd.concat(frames, ignore_index=True)


def headline(t):
    rows = []
    for (model, method, seed), g in t.groupby(['model', 'method', 'seed'], sort=False):
        if method == 'baseline':
            rows.append(dict(model=model, method=method, seed=seed, clean_auc=g.normal_auc.iloc[0], all_worse=g.worse_cross_auc.mean(),
                             fp_flip=g['fp_flip@normal5'].mean(), fn_flip=g['fn_flip@normal5'].mean(), epoch0_runs=0))
            continue
        diag = g[g['filter'].eq(g.train_filter)]
        held = g[g['filter'].ne(g.train_filter)]
        dt = held[held['filter'].eq('domain_transform')]
        rows.append(dict(model=model, method=method, seed=seed, clean_auc=g.groupby('train_filter').normal_auc.first().mean(),
                         diag_worse=diag.worse_cross_auc.mean(), heldout_worse=held.worse_cross_auc.mean(),
                         dt_heldout=dt.worse_cross_auc.mean(), all_worse=g.worse_cross_auc.mean(),
                         fp_flip=g['fp_flip@normal5'].mean(), fn_flip=g['fn_flip@normal5'].mean(),
                         epoch0_runs=int(g.groupby('train_filter').epoch0.first().sum())))
    per_seed = pd.DataFrame(rows)
    metrics = ['clean_auc', 'diag_worse', 'heldout_worse', 'dt_heldout', 'all_worse', 'fp_flip', 'fn_flip']
    agg = per_seed.groupby(['model', 'method'], sort=False)
    out = agg[metrics].mean().add_suffix('_mean').join(agg[metrics].std().add_suffix('_sd'))
    out['seeds'] = agg.seed.nunique()
    out['epoch0_runs'] = agg.epoch0_runs.sum()
    return out.reset_index()


def seed_means(t):
    keys = ['model', 'method', 'train_filter', 'filter']
    t = t.assign(train_filter=t.train_filter.fillna('-'))
    cols = ['normal_auc', 'worse_cross_auc', 'max_gap', 'fp_flip@normal5', 'fn_flip@normal5', 'real_shift', 'fake_shift']
    agg = t.groupby(keys, sort=False)
    return agg[cols].mean().add_suffix('_mean').join(agg[cols].std().add_suffix('_sd')).join(agg.seed.nunique().rename('seeds')).reset_index()


def wins_vs_rank(means):
    rows = []
    for model, g in means.groupby('model', sort=False):
        rank = g[g.method.eq('aug_pair_rank')].set_index(['train_filter', 'filter']).worse_cross_auc_mean
        for method, h in g[~g.method.isin(['baseline', 'aug_pair_rank'])].groupby('method', sort=False):
            diff = h.set_index(['train_filter', 'filter']).worse_cross_auc_mean-rank
            rows.append(dict(model=model, method=method, cells_better_than_rank=int((diff > 0).sum()), of=len(diff),
                             mean_gain_vs_rank=float(diff.mean())))
    return pd.DataFrame(rows)


def run(args):
    t = load(args)
    means = seed_means(t)
    head = headline(t).merge(wins_vs_rank(means), on=['model', 'method'], how='left')
    head['method'] = pd.Categorical(head.method, [m for m in ORDER if m in set(head.method)] +
                                    sorted(set(head.method)-set(ORDER)), ordered=True)
    head = head.sort_values(['model', 'method']).reset_index(drop=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    means.to_csv(args.output_root/'v2_diagonal_seed_means.csv', index=False)
    head.to_csv(args.output_root/'v2_diagonal_headline.csv', index=False)
    with pd.option_context('display.width', 250, 'display.max_columns', 40):
        show = head[['model', 'method', 'seeds', 'epoch0_runs', 'clean_auc_mean', 'diag_worse_mean', 'heldout_worse_mean',
                     'dt_heldout_mean', 'all_worse_mean', 'fp_flip_mean', 'cells_better_than_rank']]
        print(show.round(3).to_string(index=False))
    print('Saved:', args.output_root, flush=True)
    return head, means


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--eval-root', type=Path, required=True, help='Drive evaluations/ folder')
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--models', nargs='+', default=['xception', 'clip_b16', 'dinov2_b14'])
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    p.add_argument('--dataset', default='cdf_test')
    return p.parse_args(argv)


if __name__ == '__main__':
    run(parse_args())
