"""How far smoothing pushes CDF test scores toward "fake", at AUC level and at fixed thresholds.

CPU-only; reads saved frame predictions <prefix>_<condition>.csv (crop_path, class_index, raw_margin)
for the 4 conditions of one model/strategy. No models, images or inference.

  skew                 mean AUC where the FAKE is more smoothed - mean AUC where the REAL is more smoothed
                       (0 = smoothing does not move scores; positive = smoothed images look more fake.
                       Pooled AUC cannot see this because the two directions cancel)
  strong_real_vs_normal_fake   AUC(strong-smoothed real vs unsmoothed fake)
  normal_real_vs_strong_fake   AUC(unsmoothed real vs strong-smoothed fake)
  real_side_gap        normal AUC - strong_real_vs_normal_fake  (false-positive direction: beautified real)
  fake_side_gap        normal AUC - normal_real_vs_strong_fake  (miss direction: beautified fake).
                       Invariance needs BOTH near 0; one small and one large only moved the bias.
  fpr_<cond>@0         share of real frames called fake at the model's own threshold (raw margin 0)
  fpr_strong@normal5   threshold set so 5% of UNSMOOTHED real frames are called fake; share of
                       STRONG-smoothed real frames called fake at that same threshold
  flip_real@normal5    real frames called real when unsmoothed but fake when strongly smoothed (paired)
  fnr_strong@normal5 / flip_fake@normal5   same threshold, fake side: strong-smoothed fakes missed /
                       fakes caught when unsmoothed but missed when strongly smoothed (paired)
  real_shift / fake_shift   mean raw-margin change normal->strong, in units of the normal real std

TEST-LABEL ORACLE DIAGNOSTIC ONLY: the normal5 threshold uses CDF test real labels. It describes the
score shift; it is not a deployable operating point or a selection rule.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CONDITIONS = ['normal', 'smooth_weak', 'smooth_medium', 'smooth_strong']
NORMAL_FPR = .05


def auc(negative, positive):
    ranks = pd.Series(np.r_[negative, positive]).rank().to_numpy()
    return float((ranks[len(negative):].sum()-len(positive)*(len(positive)+1)/2)/(len(negative)*len(positive)))


def load(prefix):
    frames = {c: pd.read_csv(f'{prefix}_{c}.csv', usecols=['crop_path', 'class_index', 'raw_margin']).sort_values('crop_path', ignore_index=True)
              for c in CONDITIONS}
    first = frames['normal']
    for c, frame in frames.items():
        if not (frame.crop_path.equals(first.crop_path) and frame.class_index.equals(first.class_index)) or not np.isfinite(frame.raw_margin).all():
            raise ValueError(f'{prefix}_{c}.csv does not match the normal-condition images')
    return first.class_index.to_numpy().astype(int), {c: f.raw_margin.to_numpy() for c, f in frames.items()}


def measure(y, margins):
    real, fake = y == 0, y == 1
    grid = np.array([[auc(margins[r][real], margins[f][fake]) for f in CONDITIONS] for r in CONDITIONS])
    rank = np.arange(len(CONDITIONS))
    row = dict(pooled16=float(grid.mean()), normal_auc=grid[0, 0],
               skew=float(grid[rank[:, None] < rank[None, :]].mean()-grid[rank[:, None] > rank[None, :]].mean()),
               strong_real_vs_normal_fake=grid[-1, 0], normal_real_vs_strong_fake=grid[0, -1],
               real_side_gap=grid[0, 0]-grid[-1, 0], fake_side_gap=grid[0, 0]-grid[0, -1])
    for c in CONDITIONS:
        row[f'fpr_{c}@0'] = float((margins[c][real] >= 0).mean())
        row[f'fnr_{c}@0'] = float((margins[c][fake] < 0).mean())
    threshold = np.quantile(margins['normal'][real], 1-NORMAL_FPR)
    row['fpr_strong@normal5'] = float((margins['smooth_strong'][real] >= threshold).mean())
    row['fnr_normal@normal5'] = float((margins['normal'][fake] < threshold).mean())
    row['flip_real@normal5'] = float(((margins['normal'][real] < threshold) & (margins['smooth_strong'][real] >= threshold)).mean())
    row['fnr_strong@normal5'] = float((margins['smooth_strong'][fake] < threshold).mean())
    row['flip_fake@normal5'] = float(((margins['normal'][fake] >= threshold) & (margins['smooth_strong'][fake] < threshold)).mean())
    scale = margins['normal'][real].std()
    row['real_shift'] = float((margins['smooth_strong'][real]-margins['normal'][real]).mean()/scale)
    row['fake_shift'] = float((margins['smooth_strong'][fake]-margins['normal'][fake]).mean()/scale)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--entry', action='append', required=True, metavar='MODEL,STRATEGY,PREFIX',
                   help='prediction prefix without _<condition>.csv; repeatable')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    rows = []
    for entry in args.entry:
        model, strategy, prefix = entry.split(',', 2)
        rows.append(dict(model=model, strategy=strategy, **measure(*load(prefix))))
    table = pd.DataFrame(rows)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, index=False)
    with pd.option_context('display.width', 250, 'display.max_columns', 30):
        print(table.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
