"""Is a v2-trained model compensating (detect the smoothing trace, subtract a fixed score) or invariant?

Two checks that need only the TRAINING filter, run after the v2 diagonal training/evaluation:
  curve  Strength-response curve from the saved evaluation predictions (no GPU, no inference): per checkpoint, filter
         and weak/medium/strong, the paired score shift (smoothed - original) of reals and of fakes in units of the
         clean-real std. Compensation subtracts a near-constant amount once the trace is seen, so it over-corrects the
         WEAK level (sign flips toward real) while the baseline grows with strength; invariance stays flat near 0.
  probe  Smoothing-trace probe on FF++ VAL (in-domain, no Celeb-DF): head-input features of the original and the
         train-filter strong view (v2 cache) -> 5-fold video-grouped logistic AUC "smoothed vs original" (0.5 = no
         trace) and the relative feature shift. Compensation keeps the trace (high AUC); method A should remove it.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

import beauty_smoothing_v2 as v2
import diagnose_smoothing_feature_probe as d1
import ffpp_c23_smoothing_v2_diagonal as diagonal

base, evaluation = diagonal.base, diagonal.evaluation


def curve(args):
    summary = pd.read_csv(args.eval_root/'summary.csv')
    rows = []
    for (name, label), _ in summary.groupby(['model', 'strategy'], sort=False):
        normal = pd.read_csv(args.eval_root/f'{name}_{label}_normal.csv', dtype={'crop_path': str})
        y, m0 = normal.class_index.to_numpy(), normal.raw_margin.to_numpy()
        std = m0[y == 0].std()
        for filt in v2.FILTERS:
            row = dict(model=name, strategy=label, filter=filt)
            for condition in v2.LEVELS:
                pred = pd.read_csv(args.eval_root/f'{name}_{label}_{filt}_{condition}.csv', dtype={'crop_path': str})
                if pred.crop_path.tolist() != normal.crop_path.tolist():
                    raise ValueError(f'Prediction rows differ: {label} {filt} {condition}')
                delta = pred.raw_margin.to_numpy()-m0
                level = condition.removeprefix('smooth_')
                row[f'real_{level}'], row[f'fake_{level}'] = delta[y == 0].mean()/std, delta[y == 1].mean()/std
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(args.eval_root/'strength_curve.csv', index=False)
    with pd.option_context('display.width', 250, 'display.max_columns', 20):
        print('Paired score shift (sigma of clean reals): + = toward fake, - = toward real')
        print(table.round(2).to_string(index=False))
    print('Saved:', args.eval_root/'strength_curve.csv', flush=True)
    return table


def probe(args):
    frame = d1.sample(args)                                   # FF++ val, detected, half real / half fake
    y, groups = frame.class_index.astype(int).to_numpy(), frame.video_id.to_numpy()
    cache = diagonal.restore_cache(args.cache_root, args.archive_dir, 'ffpp_trainval', ['mask', args.train_filter])
    original, smoothed = [], []
    for path in frame.crop_path:
        with Image.open(args.source_root/path) as opened:
            image = opened.convert('RGB')
        original.append(image)
        smoothed.append(cache.view(image, path, args.train_filter, v2.LEVELS['smooth_strong']))
    rows = []
    for name in args.models:
        initial, initial_hash = evaluation.checkpoint(argparse.Namespace(runs_root=args.baseline_runs_root, seed=args.baseline_seed), name)
        states = [('baseline', initial['model_state_dict'])]
        for label, _, _ in diagonal.variants(args.strategies, args.fc_weights):
            weights, _, _ = diagonal.completed_checkpoint(args.runs_root, name, args.train_filter, label, args.seed, initial_hash)
            states.append((f'{label}@{args.train_filter}', weights))
        for label, weights in states:
            model, transform = base.build_model(name, pretrained=False)
            model.load_state_dict(weights, strict=True)
            model.to(args.device).eval()
            f0, m0 = d1.features(args, model, name, transform, original)
            f1, m1 = d1.features(args, model, name, transform, smoothed)
            spread = float(np.linalg.norm(f0-f0.mean(0), axis=1).mean())
            std = m0[y == 0].std()
            rows.append(dict(model=name, strategy=label, filter=args.train_filter, probe_auc=d1.probe(f0, f1, groups, args.folds),
                             feature_shift=float(np.linalg.norm(f1-f0, axis=1).mean()/spread),
                             real_shift=float((m1-m0)[y == 0].mean()/std), fake_shift=float((m1-m0)[y == 1].mean()/std)))
            print(f"{name} {label:32s} probe AUC {rows[-1]['probe_auc']:.3f} | feature shift {rows[-1]['feature_shift']:.3f} | "
                  f"score shift real {rows[-1]['real_shift']:+.2f} fake {rows[-1]['fake_shift']:+.2f} (sigma)", flush=True)
            del model
            if args.device == 'cuda':
                torch.cuda.empty_cache()
    table = pd.DataFrame(rows)
    args.output_root.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_root/f'trace_probe_{args.train_filter}_seed{args.seed}.csv', index=False)
    print('Saved:', args.output_root, flush=True)
    return table


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=['curve', 'probe'])
    p.add_argument('--eval-root', type=Path, help='curve: one evaluation folder (…/seed42/cdf_dev)')
    for field in ['source-root', 'baseline-runs-root', 'runs-root', 'archive-dir', 'output-root']:
        p.add_argument('--'+field, type=Path)
    p.add_argument('--cache-root', type=Path, default=Path('/content/beauty_smoothing_v2_cache'))
    p.add_argument('--train-filter', choices=v2.FILTERS, default='bilateral')
    p.add_argument('--models', nargs='+', choices=list(base.PREPROCESS), default=['clip_b16'])
    p.add_argument('--strategies', nargs='+', choices=diagonal.STRATEGIES, default=['aug_pair_rank', 'aug_pair_rank_fc'])
    p.add_argument('--fc-weights', nargs='+', type=float, default=[1.])
    for field, default in [('seed', 42), ('baseline-seed', 42), ('images', 1200), ('folds', 5), ('batch-size', 64), ('workers', 2)]:
        p.add_argument('--'+field, type=int, default=default)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    a = p.parse_args(argv)
    needed = ['eval_root'] if a.action == 'curve' else ['source_root', 'baseline_runs_root', 'runs_root', 'archive_dir', 'output_root']
    if any(getattr(a, k) is None for k in needed):
        p.error(f'{a.action} needs --' + ' --'.join(k.replace('_', '-') for k in needed))
    if a.action == 'probe' and a.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime')
    return a


if __name__ == '__main__':
    arguments = parse_args()
    {'curve': curve, 'probe': probe}[arguments.action](arguments)
