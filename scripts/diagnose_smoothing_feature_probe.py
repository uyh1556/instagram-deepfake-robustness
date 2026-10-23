"""D1: do smoothing-augmented detectors become blind to smoothing, or do they learn to DETECT and compensate it?

Inference only on FF++ VAL frames (in-domain; no Celeb-DF, no test). For each v1 source-ablation checkpoint of one
backbone (baseline, clean_only, aug_single, aug_pair, aug_pair_rank; trained with v1 bilateral pairs only) the
penultimate features (the classifier-head input) of the original and the strong (.75) smoothed view of the same frames
are extracted for the v1 training bilateral and the three held-out v1 filters (calibrated values, same geometric mask).
  probe_auc      5-fold video-grouped logistic probe "smoothed vs original" on standardized features (0.5 = no trace)
  transfer_auc   probe trained on bilateral pairs, scored on this filter's pairs: does the bilateral detector fire?
  feature_shift  mean ||f(smoothed) - f(original)|| / mean ||f(original) - mean f|| (how far smoothing moves features)
  margin_shift   mean margin(smoothed) - margin(original) for real and for fake frames (the score consequence)
Compensation predicts: augmented models keep a HIGH probe/transfer AUC with ~zero bilateral margin shift (they see the
smoothing and cancel it) and a negative margin shift on a held-out filter the bilateral detector fires on; true
invariance predicts probe AUC near 0.5.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

import beauty_smoothing_filters as beauty
import evaluate_beauty_filter_generalization as general

base, evaluation = general.base, general.evaluation
FILTERS = ['bilateral', *beauty.FILTERS]
STRENGTH = .75


def sample(args):
    frame = pd.read_csv(args.source_root/'manifests/ffpp_c23_4methods_roi16_trainval_v1.csv', dtype=str, keep_default_na=False)
    frame = frame[frame.split.eq('val') & frame.roi_status.eq('detected')]
    half = args.images//2
    parts = [frame[frame.class_index.eq(c)].sample(n=min(half, frame.class_index.eq(c).sum()), random_state=args.seed) for c in ['0', '1']]
    return pd.concat(parts).sort_values('crop_path').reset_index(drop=True)


def views(args, frame):
    """Original + strong view of every filter, computed once and shared by all checkpoints."""
    settings = json.loads(args.protocol.read_text())['bilateral']
    values = general.calibration_values(args.calibration)
    out = {name: [] for name in ['original', *FILTERS]}
    for k, path in enumerate(frame.crop_path, 1):
        with Image.open(args.source_root/path) as opened:
            image = opened.convert('RGB')
        out['original'].append(image)
        out['bilateral'].append(beauty.pixel_cache.blend(image, beauty.bilateral_full(image, settings), STRENGTH))
        for name in beauty.FILTERS:
            out[name].append(beauty.smooth(image, name, values[name], STRENGTH))
        if k % 200 == 0 or k == len(frame):
            print(f'  views {k}/{len(frame)}', flush=True)
    return out


class Images(Dataset):
    def __init__(self, images, transform):
        self.images, self.transform = images, transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.transform(self.images[index]), index


def features(args, model, name, transform, images):
    captured = {}
    hook = base.head(model, name).register_forward_pre_hook(lambda m, inp: captured.__setitem__('x', inp[0].detach()))
    feats, margins = [], []
    try:
        with torch.inference_mode():
            for x, _ in DataLoader(Images(images, transform), batch_size=args.batch_size, num_workers=args.workers):
                logits = model(x.to(args.device)).float()
                feats.append(captured['x'].float().flatten(1).cpu().numpy())
                margins.append((logits[:, 1]-logits[:, 0]).cpu().numpy())
    finally:
        hook.remove()
    return np.concatenate(feats), np.concatenate(margins)


def probe(original, smoothed, groups, folds):
    x = np.concatenate([original, smoothed])
    y = np.r_[np.zeros(len(original)), np.ones(len(smoothed))]
    g = np.r_[groups, groups]
    scores = np.empty(len(y))
    for train, test in GroupKFold(n_splits=folds).split(x, y, g):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.))
        scores[test] = model.fit(x[train], y[train]).decision_function(x[test])
    return float(roc_auc_score(y, scores))


def transfer(original, smoothed, groups, folds, train_pairs):
    """Probe fitted on bilateral pairs of the training folds, scored on this filter's pairs of the held-out folds."""
    bo, bs = train_pairs
    scores, labels = [], []
    for train, test in GroupKFold(n_splits=folds).split(original, groups=groups):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.))
        model.fit(np.concatenate([bo[train], bs[train]]), np.r_[np.zeros(len(train)), np.ones(len(train))])
        scores.append(model.decision_function(np.concatenate([original[test], smoothed[test]])))
        labels.append(np.r_[np.zeros(len(test)), np.ones(len(test))])
    return float(roc_auc_score(np.concatenate(labels), np.concatenate(scores)))


def run(args):
    frame = sample(args)
    y = frame.class_index.astype(int).to_numpy()
    groups = frame.video_id.to_numpy()
    print(f'FF++ val frames: {len(frame)} ({(y == 0).sum()} real / {(y == 1).sum()} fake, {len(set(groups))} videos)', flush=True)
    images = views(args, frame)
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in args.models:
        for label, weights, checkpoint_hash in general.checkpoints(args, name):
            model, transform = base.build_model(name, pretrained=False)
            model.load_state_dict(weights, strict=True)
            model.to(args.device).eval()
            out = {view: features(args, model, name, transform, imgs) for view, imgs in images.items()}
            f0, m0 = out['original']
            spread = float(np.linalg.norm(f0-f0.mean(0), axis=1).mean())
            for filt in FILTERS:
                f1, m1 = out[filt]
                delta = m1-m0
                rows.append(dict(model=name, strategy=label, filter=filt, checkpoint_sha256=checkpoint_hash,
                    probe_auc=probe(f0, f1, groups, args.folds),
                    transfer_auc=transfer(f0, f1, groups, args.folds, (f0, out['bilateral'][0])),
                    feature_shift=float(np.linalg.norm(f1-f0, axis=1).mean()/spread),
                    margin_shift_real=float(delta[y == 0].mean()), margin_shift_fake=float(delta[y == 1].mean()),
                    margin_std_real=float(m0[y == 0].std())))
                print(f"{name} {label:14s} {filt:17s} probe {rows[-1]['probe_auc']:.3f} transfer {rows[-1]['transfer_auc']:.3f} "
                      f"shift {rows[-1]['feature_shift']:.3f} margin real {rows[-1]['margin_shift_real']:+.3f} fake {rows[-1]['margin_shift_fake']:+.3f}", flush=True)
            del model
            if args.device == 'cuda':
                torch.cuda.empty_cache()
    table = pd.DataFrame(rows)
    table.to_csv(args.output_root/'feature_probe.csv', index=False)
    frame[['crop_path', 'video_id', 'class_index']].to_csv(args.output_root/'frames.csv', index=False)
    base.common.json_write(args.output_root/'protocol.json', dict(experiment='smoothing_feature_probe_v1', data='FF++ val, roi detected',
        images=len(frame), strength=STRENGTH, folds=args.folds, filters=FILTERS, checkpoints='v1 source ablation seed42',
        test_used=False))
    with pd.option_context('display.width', 250):
        for metric in ['probe_auc', 'transfer_auc', 'feature_shift', 'margin_shift_real', 'margin_shift_fake']:
            print(metric)
            print(table.pivot_table(index=['model', 'strategy'], columns='filter', values=metric, sort=False)[FILTERS].round(3).to_string())
    print('Saved:', args.output_root, flush=True)
    return table


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for field in ['source-root', 'baseline-runs-root', 'runs-root', 'calibration', 'output-root']:
        p.add_argument('--'+field, type=Path, required=True)
    p.add_argument('--protocol', type=Path, default=general.ROOT/'configs/ffpp_c23_celebdf_smoothing_v1.json')
    p.add_argument('--models', nargs='+', choices=list(base.PREPROCESS), default=['clip_b16'])
    for field, default in [('seed', 42), ('images', 1200), ('folds', 5), ('batch-size', 64), ('workers', 2)]:
        p.add_argument('--'+field, type=int, default=default)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    a = p.parse_args(argv)
    if a.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime')
    return a


if __name__ == '__main__':
    run(parse_args())
