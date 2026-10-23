"""Candidate 3: does REMOVING the smoothing direction from the features stop the compensation (no training)?

D1 showed the v1 augmented checkpoints keep a linearly detectable smoothing trace shared across filters and subtract
a near-constant score whenever it is present. Here, per v1 source-ablation checkpoint (seed42):
  1. Directions: top-k right singular vectors of the paired feature changes f(bilateral strong) - f(original) on FF++
     TRAIN frames. Only the training filter (bilateral) is used, so the three held-out filters stay unseen.
  2. Projection P = I - U U^T removes them; a logistic head is refit on projected FF++ train ORIGINAL features
     (backbone untouched). k=0 refits on unprojected features = the head-refit control.
  3. Scored on FF++ VAL (original + strong view of bilateral and the held-out v1 filters): clean AUC, fp/fn gaps,
     paired score shifts in units of the clean real std, and the smoothing-probe AUC after projection.
If removal works, the held-out-filter reversal (Xception domain transform fakes pushed to real) and the bilateral
residual shift shrink toward 0 while clean AUC holds. No Celeb-DF, no test frames; diagnosis only.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import diagnose_smoothing_feature_probe as d1

general, beauty, base = d1.general, d1.beauty, d1.base
FILTERS = d1.FILTERS


def sample(args, split, n, seed):
    frame = pd.read_csv(args.source_root/'manifests/ffpp_c23_4methods_roi16_trainval_v1.csv', dtype=str, keep_default_na=False)
    frame = frame[frame.split.eq(split) & frame.roi_status.eq('detected')]
    parts = [frame[frame.class_index.eq(c)].sample(n=min(n//2, frame.class_index.eq(c).sum()), random_state=seed) for c in ['0', '1']]
    return pd.concat(parts).sort_values('crop_path').reset_index(drop=True)


def train_views(args, frame):
    import json
    settings = json.loads(args.protocol.read_text())['bilateral']
    out = {'original': [], 'bilateral': []}
    for path in frame.crop_path:
        with Image.open(args.source_root/path) as opened:
            image = opened.convert('RGB')
        out['original'].append(image)
        out['bilateral'].append(beauty.pixel_cache.blend(image, beauty.bilateral_full(image, settings), d1.STRENGTH))
    return out


def directions(f0, f1, k):
    if k == 0:
        return np.zeros((f0.shape[1], 0))
    _, _, vt = np.linalg.svd(f1-f0, full_matrices=False)
    return vt[:k].T


def project(f, u):
    return f-(f@u)@u.T if u.shape[1] else f


def gaps(y, clean, smooth):
    real, fake = y == 0, y == 1
    normal = roc_auc_score(y, clean)
    fp_cross = roc_auc_score(np.r_[np.zeros(real.sum()), np.ones(fake.sum())], np.r_[smooth[real], clean[fake]])
    fn_cross = roc_auc_score(np.r_[np.zeros(real.sum()), np.ones(fake.sum())], np.r_[clean[real], smooth[fake]])
    std = clean[real].std()
    return dict(normal_auc=normal, fp_gap=normal-fp_cross, fn_gap=normal-fn_cross,
                real_shift=float((smooth-clean)[real].mean()/std), fake_shift=float((smooth-clean)[fake].mean()/std))


def run(args):
    tr, va = sample(args, 'train', args.images, args.seed), sample(args, 'val', args.images, args.seed)
    ytr, yva = tr.class_index.astype(int).to_numpy(), va.class_index.astype(int).to_numpy()
    print(f'FF++ train {len(tr)} (directions + head) | FF++ val {len(va)} (scoring)', flush=True)
    vtr = train_views(args, tr)
    vva = d1.views(args, va)                  # original + strong of all four v1 filters, shared by every checkpoint
    groups = va.video_id.to_numpy()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in args.models:
        sub = argparse.Namespace(**{**vars(args), 'runs_root': args.runs_root/f'ffpp_c23_smoothing_source_ablation_{name}'})
        for label, weights, checkpoint_hash in general.checkpoints(sub, name):
            if label not in args.strategies:
                continue
            model, transform = base.build_model(name, pretrained=False)
            model.load_state_dict(weights, strict=True)
            model.to(args.device).eval()
            ftr = {v: d1.features(args, model, name, transform, imgs) for v, imgs in vtr.items()}
            fva = {v: d1.features(args, model, name, transform, imgs) for v, imgs in vva.items()}
            del model
            if args.device == 'cuda':
                torch.cuda.empty_cache()
            original = {f: gaps(yva, fva['original'][1], fva[f][1]) for f in FILTERS}   # the checkpoint's own head
            for f in FILTERS:
                rows.append(dict(model=name, strategy=label, variant='model_head', k=-1, filter=f, **original[f]))
            for k in args.ks:
                u = directions(ftr['original'][0], ftr['bilateral'][0], k)
                head = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=args.c))
                head.fit(project(ftr['original'][0], u), ytr)
                clean = head.decision_function(project(fva['original'][0], u))
                for f in FILTERS:
                    smooth = head.decision_function(project(fva[f][0], u))
                    probe = d1.probe(project(fva['original'][0], u), project(fva[f][0], u), groups, args.folds)
                    rows.append(dict(model=name, strategy=label, variant='refit' if k == 0 else f'remove_k{k}', k=k, filter=f,
                                     probe_auc=probe, **gaps(yva, clean, smooth)))
                last = [r for r in rows[-len(FILTERS):]]
                print(f"{name} {label:14s} k={k:2d} " + ' '.join(f"{r['filter'][:6]} fn {r['fn_gap']:+.3f} fake {r['fake_shift']:+.2f}" for r in last), flush=True)
    table = pd.DataFrame(rows)
    table['max_gap'] = np.maximum(table.fp_gap.abs(), table.fn_gap.abs())
    table.to_csv(args.output_root/'direction_removal.csv', index=False)
    base.common.json_write(args.output_root/'protocol.json', dict(experiment='smoothing_direction_removal_v1',
        directions='top-k SVD of bilateral-strong minus original features, FF++ train', head='logistic, FF++ train original',
        scoring='FF++ val original vs strong v1 filters', ks=args.ks, images=args.images, test_used=False,
        held_out_filters_used_for_fitting=False))
    with pd.option_context('display.width', 250, 'display.max_rows', 500):
        for metric in ['normal_auc', 'max_gap', 'fn_gap', 'fake_shift', 'real_shift']:
            print(metric)
            print(table.pivot_table(index=['model', 'strategy', 'variant'], columns='filter', values=metric, sort=False)[FILTERS].round(3).to_string())
    print('Saved:', args.output_root, flush=True)
    return table


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for field in ['source-root', 'baseline-runs-root', 'runs-root', 'calibration', 'output-root']:
        p.add_argument('--'+field, type=Path, required=True)   # runs-root = parent of ffpp_c23_smoothing_source_ablation_<model>
    p.add_argument('--protocol', type=Path, default=general.ROOT/'configs/ffpp_c23_celebdf_smoothing_v1.json')
    p.add_argument('--models', nargs='+', choices=list(base.PREPROCESS), default=['xception', 'clip_b16', 'dinov2_b14'])
    p.add_argument('--strategies', nargs='+', default=['baseline', 'aug_pair', 'aug_pair_rank'])
    p.add_argument('--ks', nargs='+', type=int, default=[0, 1, 4, 16])
    p.add_argument('--c', type=float, default=1.)
    for field, default in [('seed', 42), ('images', 1200), ('folds', 5), ('batch-size', 64), ('workers', 2)]:
        p.add_argument('--'+field, type=int, default=default)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    a = p.parse_args(argv)
    if a.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime')
    return a


if __name__ == '__main__':
    run(parse_args())
