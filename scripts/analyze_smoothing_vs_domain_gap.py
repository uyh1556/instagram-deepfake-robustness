"""Split CDF AUC loss into a domain part and a smoothing part from saved cross-condition AUCs.

CPU-only; reads cross_condition_auc.csv files (model, strategy, real_condition, fake_condition, auc)
written by the pool v2 evaluations. No models, images or inference.

  matched      mean AUC when real and fake share a condition (discrimination left inside one condition)
  pooled16     mean of all 16 real x fake condition AUCs (= pooled AUC over the 4 conditions)
  shift_cost   matched - pooled16: AUC lost only because conditions are mixed (score shift by smoothing)
  info_loss    AUC(normal, normal) - AUC(strong, strong): discrimination destroyed by strong smoothing
  fp_side      mean AUC where the real image is MORE smoothed than the fake (smoothed real looks fake)
  fn_side      mean AUC where the fake image is MORE smoothed than the real
  domain_gap   reference matched - matched (reference = target-labelled SDA, i.e. what CDF labels buy)

Gap to the reference pooled16 = domain_gap + (shift_cost - reference shift_cost).
TEST-LABEL DIAGNOSTIC ONLY: describes where loss comes from; not used for selection.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CONDITIONS = ['normal', 'smooth_weak', 'smooth_medium', 'smooth_strong']
REFERENCE = 'sda_new_balanced(ref)'


def decompose(group):
    grid = group.pivot(index='real_condition', columns='fake_condition', values='auc').loc[CONDITIONS, CONDITIONS].to_numpy()
    rank = np.arange(len(CONDITIONS))
    real_more, fake_more = rank[:, None] > rank[None, :], rank[:, None] < rank[None, :]
    matched, pooled = float(np.mean(np.diag(grid))), float(grid.mean())
    return dict(normal_auc=grid[0, 0], matched=matched, pooled16=pooled, shift_cost=matched-pooled,
                info_loss=grid[0, 0]-grid[-1, -1], fp_side=float(grid[real_more].mean()),
                fn_side=float(grid[fake_more].mean()), strong_real_vs_normal_fake=grid[-1, 0])


def analyse(frames, reference):
    rows = [dict(model=model, strategy=strategy, **decompose(group))
            for (model, strategy), group in pd.concat(frames).groupby(['model', 'strategy'], sort=False)]
    table = pd.DataFrame(rows)
    for model, part in table.groupby('model'):
        ref = part[part.strategy == reference]
        if ref.empty:
            continue
        ref = ref.iloc[0]
        index = part.index
        table.loc[index, 'domain_gap'] = ref.matched-part.matched
        table.loc[index, 'smoothing_gap'] = part.shift_cost-ref.shift_cost
        table.loc[index, 'gap_to_reference'] = ref.pooled16-part.pooled16
    return table


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('inputs', nargs='+', type=Path, help='cross_condition_auc.csv files')
    p.add_argument('--reference', default=REFERENCE)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    table = analyse([pd.read_csv(path) for path in args.inputs], args.reference)
    table = table.drop_duplicates(['model', 'strategy'])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, index=False)
    with pd.option_context('display.width', 250, 'display.max_columns', 20):
        print(table.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
