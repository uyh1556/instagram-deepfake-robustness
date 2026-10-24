"""Is there real/fake evidence OUTSIDE the smoothed skin? Pre-check for a skin-occlusion auxiliary loss (inference only).

v2 smoothing edits only the face-parsing skin(+nose) mask; eyes, brows, lips and the mouth stay untouched and lie inside
the manipulated region of all four FF++ methods. Before training a detector to also decide without the skin, measure how
much evidence survives when that skin is hidden: for the FF++ baseline of each backbone, every frame is scored as the
original and with the v2 skin mask filled grey (same soft mask edge as the smoothing). Two readouts per view:
  head   the baseline's own classifier (zero-shot: it never saw occluded faces, so this is a lower bound)
  probe  standardized logistic regression on the classifier-input features, fitted on FF++ TRAIN frames of the same view
         and scored on FF++ val and cdf_dev (how much evidence the features still carry, independent of the old head)
AUC overall and per FF++ manipulation (real vs that method). No Celeb-DF test frame is used.
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
from torch.utils.data import DataLoader, Dataset

import build_beauty_smoothing_v2_cache as builder
import ffpp_c23_smoothing_v2_diagonal as diagonal

base, evaluation = diagonal.base, diagonal.evaluation
VIEWS = ['original', 'skin_occluded']
FILL = 128.


def balanced(frame, n, seed):
    """n detected frames, half real; fakes split evenly over the manipulation methods when the manifest has them."""
    frame = frame[frame.roi_status.eq('detected')]
    real = frame[frame.class_index.eq('0')]
    fake = frame[frame.class_index.eq('1')]
    parts = [real.sample(n=min(n//2, len(real)), random_state=seed)]
    if 'method' in fake:
        methods = sorted(fake.method.unique())
        parts += [g.sample(n=min(n//2//len(methods), len(g)), random_state=seed) for _, g in fake.groupby('method')]
    else:
        parts.append(fake.sample(n=min(n//2, len(fake)), random_state=seed))
    return pd.concat(parts).sort_values('crop_path').reset_index(drop=True)


def frames(args):
    ff = pd.read_csv(args.source_root/'manifests/ffpp_c23_4methods_roi16_trainval_v1.csv', dtype=str, keep_default_na=False)
    cdf = builder.dev_frame(args.membership, args.cdf_root)
    return {'ffpp_train': (balanced(ff[ff.split.eq('train')], args.train_images, args.seed), args.source_root, 'ffpp_trainval'),
            'ffpp_val': (balanced(ff[ff.split.eq('val')], args.val_images, args.seed), args.source_root, 'ffpp_trainval'),
            'cdf_dev': (balanced(cdf, args.cdf_images, args.seed), args.cdf_root, 'cdf_dev')}


def occlude(image, mask):
    array = np.asarray(image, dtype=np.float32)
    weight = mask[..., None]
    return Image.fromarray(np.clip(np.rint(array*(1-weight)+FILL*weight), 0, 255).astype(np.uint8))


class Views(Dataset):
    def __init__(self, frame, root, cache, view, transform):
        self.paths, self.root, self.cache, self.view, self.transform = frame.crop_path.tolist(), root, cache, view, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(self.root/path) as opened:
            image = opened.convert('RGB')
        if self.view == 'skin_occluded':
            image = occlude(image, self.cache.mask(path, image.size))
        return self.transform(image), index


def features(args, model, name, dataset):
    captured = {}
    hook = base.head(model, name).register_forward_pre_hook(lambda m, inp: captured.__setitem__('x', inp[0].detach()))
    feats, margins = [], []
    try:
        with torch.inference_mode():
            for x, _ in DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers):
                logits = model(x.to(args.device)).float()
                feats.append(captured['x'].float().flatten(1).cpu().numpy())
                margins.append((logits[:, 1]-logits[:, 0]).cpu().numpy())
    finally:
        hook.remove()
    return np.concatenate(feats), np.concatenate(margins)


def area(cache, root, path):
    with Image.open(root/path) as opened:
        return cache.mask(path, opened.size).mean()


def aucs(frame, scores):
    y = frame.class_index.astype(int).to_numpy()
    out = {'all': roc_auc_score(y, scores)}
    if 'method' in frame:
        for method in sorted(frame.method[y == 1].unique()):
            keep = (y == 0) | frame.method.eq(method).to_numpy()
            out[method] = roc_auc_score(y[keep], scores[keep])
    return out


def run(args):
    sets = frames(args)
    caches = {d: diagonal.restore_cache(args.cache_root, args.archive_dir, d, ['mask']) for d in ['ffpp_trainval', 'cdf_dev']}
    coverage = {s: float(np.mean([area(caches[d], root, p) for p in f.crop_path[:200]])) for s, (f, root, d) in sets.items()}
    print('Mean skin-mask area (first 200 frames):', {k: round(v, 3) for k, v in coverage.items()}, flush=True)
    rows = []
    for name in args.models:
        initial, _ = evaluation.checkpoint(argparse.Namespace(runs_root=args.baseline_runs_root, seed=args.baseline_seed), name)
        model, transform = base.build_model(name, pretrained=False)
        model.load_state_dict(initial['model_state_dict'], strict=True)
        model.to(args.device).eval()
        for view in VIEWS:
            out = {}
            for s, (f, root, d) in sets.items():
                print(f'{name} {view} {s}: {len(f)} frames', flush=True)
                out[s] = features(args, model, name, Views(f, root, caches[d], view, transform))
            probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.))
            probe.fit(out['ffpp_train'][0], sets['ffpp_train'][0].class_index.astype(int))
            for s in ['ffpp_val', 'cdf_dev']:
                f = sets[s][0]
                for scorer, scores in [('head', out[s][1]), ('probe', probe.decision_function(out[s][0]))]:
                    for subset, auc in aucs(f, scores).items():
                        rows.append(dict(model=name, view=view, scorer=scorer, dataset=s, subset=subset, auc=auc,
                                         n=len(f), skin_area=coverage[s]))
        del model
        if args.device == 'cuda':
            torch.cuda.empty_cache()
    table = pd.DataFrame(rows)
    args.output_root.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_root/f'skin_occlusion_seed{args.seed}.csv', index=False)
    wide = table.pivot_table(index=['model', 'dataset', 'subset', 'scorer'], columns='view', values='auc', sort=False)[VIEWS]
    wide['drop'] = wide.skin_occluded-wide.original
    with pd.option_context('display.width', 200, 'display.max_rows', 200):
        print(wide.round(3).to_string())
    print('Saved:', args.output_root, flush=True)
    return table


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for field in ['source-root', 'cdf-root', 'membership', 'baseline-runs-root', 'archive-dir', 'output-root']:
        p.add_argument('--'+field, type=Path, required=True)
    p.add_argument('--cache-root', type=Path, default=Path('/content/beauty_smoothing_v2_cache'))
    p.add_argument('--models', nargs='+', choices=list(base.PREPROCESS), default=['xception', 'clip_b16', 'dinov2_b14'])
    for field, default in [('seed', 42), ('baseline-seed', 42), ('train-images', 2000), ('val-images', 1200), ('cdf-images', 2000),
                           ('batch-size', 64), ('workers', 2)]:
        p.add_argument('--'+field, type=int, default=default)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    a = p.parse_args(argv)
    if a.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime')
    return a


if __name__ == '__main__':
    run(parse_args())
