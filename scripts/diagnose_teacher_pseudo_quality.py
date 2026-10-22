"""Would a CE+Rank teacher give cleaner pseudo-labels than the baseline teacher? Inference-only.

Scores every pool v2 adapt-train frame (unlabelled during UDA) under the four smoothing conditions
with the baseline and CE+Rank (delta_transfer_v1) checkpoints, then applies the TS rule
(pseudo-label = argmax when max softmax >= threshold). The labels in membership_labelled.csv are
read ONLY to score the pseudo-labels: accuracy, real/fake balance, and how often TRUE real frames
are confidently called fake per smoothing level (the bias that TS re-learned).
No training, no checkpoint selection; official test frames are not touched.
The bilateral filter is cached once per frame and shared by every checkpoint and condition.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

import diagnose_cdf_style_gap as style

rank, engine, base, evaluation, common = style.rank, style.engine, style.base, style.evaluation, style.common
ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'teacher_pseudo_quality_v1'
CONDITIONS = engine.CONDITIONS
REAL_KINDS = ['Celeb-real', 'YouTube-real']


def auc(labels, scores):
    ranks = rankdata(scores)
    positive = labels == 1
    n1, n0 = positive.sum(), (~positive).sum()
    return float((ranks[positive].sum()-n1*(n1+1)/2)/(n1*n0))


def pseudo_rows(frame, probability, threshold):
    """TS rule on one bank of fake-probabilities; frame supplies true labels and kinds (diagnosis only)."""
    y = frame.class_index.astype(int).to_numpy()
    confident = np.maximum(probability, 1-probability) >= threshold
    pseudo = (probability >= .5).astype(int)
    real, fake = y == 0, y == 1
    sel_real, sel_fake = confident & (pseudo == 0), confident & (pseudo == 1)
    row = dict(threshold=threshold, frames=len(y), auc=auc(y, probability), selected_fraction=float(confident.mean()),
               selected_real=int(sel_real.sum()), selected_fake=int(sel_fake.sum()),
               pseudo_accuracy=float((pseudo[confident] == y[confident]).mean()) if confident.any() else np.nan,
               fake_side_error=float(real[sel_fake].mean()) if sel_fake.any() else np.nan,
               real_side_error=float(fake[sel_real].mean()) if sel_real.any() else np.nan,
               true_real_called_fake=float(sel_fake[real].mean()), true_fake_called_real=float(sel_real[fake].mean()))
    for kind in REAL_KINDS:
        mask = frame.kind.eq(kind).to_numpy()
        row[f'{kind}_called_fake'] = float(sel_fake[mask].mean()) if mask.any() else np.nan
    return row


def evaluate(args):
    frame = pd.read_csv(args.membership, dtype=str, keep_default_na=False)
    frame = frame[frame.partition.eq('adapt_train')].reset_index(drop=True)
    if frame.empty or frame.crop_path.duplicated().any() or not set(frame.class_index) <= {'0', '1'}:
        raise ValueError(f'Unexpected pool v2 membership: {args.membership}')
    for path in frame.crop_path:
        if not (args.target_root/path).is_file():
            raise FileNotFoundError(args.target_root/path)
    cfg = style.json.loads(args.protocol.read_text())
    args.output_root.mkdir(parents=True, exist_ok=True)
    membership_sha = base.common.digest(args.membership)
    native_cache = common.pixel_cache.BilateralCache(args.pixel_cache, cfg['bilateral'],
        dict(membership_sha256=membership_sha, bilateral=cfg['bilateral'], opencv=evaluation.cv2.__version__,
             smoothing_code_sha256=base.common.digest(ROOT/'scripts/smoothing_roi_pilot.py'),
             cache_code_sha256=base.common.digest(Path(common.pixel_cache.__file__)),
             storage='native uint8 RGB bilateral only; lossless PNG'))
    print(f'adapt-train frames: {len(frame)} | videos: {frame.video_id.nunique()} | kinds: {frame.kind.value_counts().to_dict()}', flush=True)
    rows = []
    for name in args.models:
        for strategy, (weights, checkpoint_hash) in style.checkpoints(args, name).items():
            model, banks = None, {}
            for condition in CONDITIONS:
                pred = args.output_root/f'{name}_{strategy}_{condition}.csv'
                if pred.is_file():
                    saved = pd.read_csv(pred, dtype={'crop_path': str})
                    if saved.crop_path.tolist() != frame.crop_path.tolist():
                        raise ValueError(f'Saved predictions differ: {pred}')
                    print('Reuse:', pred.name, flush=True)
                else:
                    if model is None:
                        model, transform = base.build_model(name, pretrained=False)
                        model.load_state_dict(weights, strict=True)
                        model.to(args.device).eval()
                    probability, margin = engine.infer(args, model, frame,
                        engine.TestViews(frame, args, transform, cfg, condition, native_cache, None), f'{name} {strategy} {condition}')
                    saved = frame[['crop_path']].copy()
                    saved['probability_fake'], saved['raw_margin'] = probability, margin
                    temp = pred.with_suffix('.tmp.csv'); saved.to_csv(temp, index=False); temp.replace(pred)
                banks[condition] = saved.probability_fake.to_numpy()
            for threshold in args.thresholds:
                for condition in CONDITIONS:
                    rows.append(dict(model=name, strategy=strategy, condition=condition, checkpoint_sha256=checkpoint_hash,
                                     **pseudo_rows(frame, banks[condition], threshold)))
                # TS draws the observed view uniformly from the four conditions.
                pooled = pd.concat([frame]*len(CONDITIONS), ignore_index=True)
                rows.append(dict(model=name, strategy=strategy, condition='all', checkpoint_sha256=checkpoint_hash,
                                 **pseudo_rows(pooled, np.concatenate([banks[c] for c in CONDITIONS]), threshold)))
            del model
            if args.device == 'cuda':
                torch.cuda.empty_cache()
    table = pd.DataFrame(rows)
    table.to_csv(args.output_root/'summary.csv', index=False)
    base.common.json_write(args.output_root/'protocol.json', dict(protocol=PROTOCOL, membership_sha256=membership_sha,
        partition='adapt_train', thresholds=args.thresholds, labels_used='scoring pseudo-labels only', used_for_selection=False,
        ce_rank_checkpoint='ffpp_c23_smoothing_delta_transfer_v1 ce_rank best.pt (FF++ val selected)'))
    view = table[table.threshold.eq(args.thresholds[0])]
    columns = ['model', 'strategy', 'condition', 'selected_fraction', 'pseudo_accuracy', 'selected_real', 'selected_fake',
               'fake_side_error', 'true_real_called_fake', 'true_fake_called_real', 'auc']
    with pd.option_context('display.width', 250, 'display.max_columns', 20):
        print(view[columns].round(3).to_string(index=False))
    print('Saved:', args.output_root/'summary.csv', flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for field in ['membership', 'target-root', 'baseline-runs-root', 'ce-rank-runs-root', 'output-root']:
        p.add_argument('--'+field, type=Path, required=True)
    p.add_argument('--pixel-cache', type=Path, default=Path('/content/teacher_quality_bilateral_cache_v1'))
    p.add_argument('--protocol', type=Path, default=ROOT/'configs/ffpp_c23_celebdf_smoothing_v1.json')
    p.add_argument('--models', nargs='+', choices=list(base.PREPROCESS), default=['clip_b16', 'dinov2_b14', 'xception'])
    p.add_argument('--strategies', nargs='+', choices=style.STRATEGIES, default=style.STRATEGIES)
    p.add_argument('--thresholds', nargs='+', type=float, default=[.9, .95])
    for field, default in [('seed', 42), ('batch-size', 64), ('workers', 2)]:
        p.add_argument('--'+field, type=int, default=default)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    a = p.parse_args(argv)
    if not all(.5 < t < 1 for t in a.thresholds):
        p.error('thresholds must be in (0.5, 1)')
    if a.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime')
    return a


if __name__ == '__main__':
    evaluate(parse_args())
