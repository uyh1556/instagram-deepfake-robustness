"""Source-domain check of the NSF fusion weight: the same w rule as diagnose_nonskin_fusion.py --select-w, but scored on
FF++ VALIDATION frames instead of the Celeb-DF dev split (inference only, no Celeb-DF frame at all).

Why: w = .2 was chosen on cdf_dev (labelled non-test Celeb-DF videos). That uses target-domain labels, so this repeats the
choice with source-domain data only. Everything else is identical to the cdf_dev run on the aug_pair model:
  A  whole-face margin of the finished v2 diagonal aug_pair@<train filter> run (seed 42)
  B  face_inner probe score: baseline features of the eyes/brows/lips/mouth-only image, standardized logistic regression
     fitted on the same 2,000 balanced FF++ TRAIN frames (seed 42) as before
  z  both standardized with the same 1,200 balanced FF++ VAL frames (seed 42) as before
  fusion (1-w) z(A) + w z(B), w in {0, .1, ..., .5}
Selection set: every face-detected FF++ VAL frame (or a balanced subset with --select-images). Only clean frames are
needed because the rule looks at the clean AUC alone. Reported per backbone and w: clean AUC overall and per manipulation.
Pre-registered rule (2026-11-10, before running): walking up the grid from w = 0, the last w > 0 before the first w whose
clean AUC (rounded to 3 decimals) falls below w = 0 on any backbone (diagnose_nonskin_fusion.select_w).
Reading fixed before running: the thesis keeps w = .2 (cdf_dev, already fixed); this result is reported as a source-domain
check. If it selects a different w, both are reported and any extra test evaluation is a separate decision.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import diagnose_nonskin_fusion as fusion

occlusion, diagonal, base, evaluation = fusion.occlusion, fusion.diagonal, fusion.base, fusion.evaluation
SOURCE_LABEL = 'aug_pair'


def frames(args):
    ff = pd.read_csv(args.source_root/'manifests/ffpp_c23_4methods_roi16_trainval_v1.csv', dtype=str, keep_default_na=False)
    val = ff[ff.split.eq('val')]
    if args.select_images:
        select = occlusion.balanced(val, args.select_images, args.seed)
    else:
        select = val[val.roi_status.eq('detected')].sort_values('crop_path').reset_index(drop=True)
    return {'ffpp_train': (occlusion.balanced(ff[ff.split.eq('train')], args.train_images, args.seed), args.source_root, 'ffpp_trainval'),
            'ffpp_val': (occlusion.balanced(val, args.val_images, args.seed), args.source_root, 'ffpp_trainval'),
            'ffpp_select': (select, args.source_root, 'ffpp_trainval')}


def aucs(y, method, score):
    out = {'normal_auc': roc_auc_score(y, score)}
    for m in sorted(set(method[y == 1])):
        keep = (y == 0) | (method == m)
        out[f'auc_{m}'] = roc_auc_score(y[keep], score[keep])
    return out


def run(args):
    sets = frames(args)
    fusion.parse_inner(argparse.Namespace(work_dir=args.work_dir, parse_batch=args.parse_batch, device=args.device), sets)
    sel = sets['ffpp_select'][0]
    y, method = sel.class_index.astype(int).to_numpy(), sel.method.to_numpy()
    print(f'selection set: {len(sel)} detected FF++ val frames ({int((y == 0).sum())} real / {int((y == 1).sum())} fake)', flush=True)
    rows = []
    for name in args.models:
        initial, initial_hash = evaluation.checkpoint(argparse.Namespace(runs_root=args.baseline_runs_root, seed=args.baseline_seed), name)
        weights, _, epoch = diagonal.completed_checkpoint(args.runs_parent/f'ffpp_c23_smoothing_v2_diagonal_{name}', name,
                                                          args.train_filter, SOURCE_LABEL, args.seed, initial_hash)
        print(f'{name}: whole-face score from {SOURCE_LABEL}@{args.train_filter} seed{args.seed} (epoch {epoch})', flush=True)

        def score(model, transform, s, mode):
            f, root, _ = sets[s]
            print(f'{name} {s} {mode}', flush=True)
            return occlusion.features(args, model, name, fusion.Inputs(f, root, None, mode, transform, args.work_dir))

        model, transform = fusion.load(args, name, weights)
        val_a, sel_a = score(model, transform, 'ffpp_val', 'full')[1], score(model, transform, 'ffpp_select', 'full')[1]
        del model
        model, transform = fusion.load(args, name, initial['model_state_dict'])   # B always on the baseline features
        probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.))
        probe.fit(score(model, transform, 'ffpp_train', 'face_inner')[0], sets['ffpp_train'][0].class_index.astype(int))
        val_b = probe.decision_function(score(model, transform, 'ffpp_val', 'face_inner')[0])
        sel_b = probe.decision_function(score(model, transform, 'ffpp_select', 'face_inner')[0])
        del model
        if args.device == 'cuda':
            torch.cuda.empty_cache()
        za, zb = (sel_a-val_a.mean())/val_a.std(), (sel_b-val_b.mean())/val_b.std()
        for w in fusion.WEIGHTS:
            rows.append(dict(model=name, variant='face_inner', w=w, **aucs(y, method, (1-w)*za+w*zb),
                             nonskin_alone_auc=roc_auc_score(y, sel_b), frames=len(sel)))
    table = pd.DataFrame(rows).assign(full_model=f'{SOURCE_LABEL}@{args.train_filter}', selection_set='ffpp_val')
    args.output_root.mkdir(parents=True, exist_ok=True)
    stem = f'nonskin_fusion_on_pair_{args.train_filter}_ffppval'
    table.to_csv(args.output_root/f'{stem}_seed{args.seed}.csv', index=False)
    chosen, steps = fusion.select_w(table)
    steps.to_csv(args.output_root/f'{stem}_wselect_seed{args.seed}.csv', index=False)
    with pd.option_context('display.width', 220, 'display.max_rows', 200):
        cols = ['model', 'w', 'normal_auc', *[c for c in table if c.startswith('auc_')]]
        print(table[cols].round(4).to_string(index=False))
        print(steps.round(3).to_string(index=False))
    print(f'Selected w on FF++ val (largest w with no clean-AUC loss on {", ".join(args.models)}): {chosen}', flush=True)
    print('Saved:', args.output_root, flush=True)
    return table, chosen


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for field in ['source-root', 'baseline-runs-root', 'runs-parent', 'output-root']:
        p.add_argument('--'+field, type=Path, required=True)
    p.add_argument('--work-dir', type=Path, default=Path('/content/nonskin_fusion_work'))
    p.add_argument('--train-filter', choices=fusion.v2.FILTERS, default='bilateral')
    p.add_argument('--models', nargs='+', choices=list(base.PREPROCESS), default=['xception', 'clip_b16', 'dinov2_b14'])
    for field, default in [('seed', 42), ('baseline-seed', 42), ('train-images', 2000), ('val-images', 1200), ('select-images', 0),
                           ('batch-size', 64), ('workers', 2), ('parse-batch', 16)]:
        p.add_argument('--'+field, type=int, default=default)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    a = p.parse_args(argv)
    if a.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime')
    return a


if __name__ == '__main__':
    run(parse_args())
