"""Can non-skin evidence SUPPORT the full-face detector under skin smoothing? (inference only, no training)

The skin-occlusion pre-check showed non-skin regions cannot REPLACE the skin (probe AUC -.14 to -.16 on cdf_dev). This
asks a different question: v2 smoothing edits only the skin mask, so a score computed without the skin is unchanged by
ANY v2 filter (also held-out ones). Fusing it with the full-face score dilutes the smoothing shift; if the non-skin
evidence is complementary it may also keep (or raise) the clean AUC. Two non-skin views, each scored by a standardized
logistic probe on the FF++ baseline's classifier-input features (fitted on FF++ TRAIN frames of the same view):
  noskin      everything except the v2 skin(+nose) mask (mask filled grey) -- includes hair/background
  face_inner  only eyes, brows, lips, mouth (face parsing, feathered; rest grey) -- the manipulated non-skin parts
Fusion on cdf_dev: (1-w) z(full-face margin) + w z(non-skin probe score), z from FF++ val clean statistics, for a fixed
grid of w. The non-skin score is the same for every smoothing condition by construction (the edit stays in the skin).
Reported per filter: clean AUC and the worse cross AUC (min of strong-real vs clean-fake, clean-real vs strong-fake).
Because w is picked over a grid ON cdf_dev, this is an oracle upper bound: a fail means the direction is dead; a pass
only makes it worth a trained pilot. No Celeb-DF test frame is used.
--full-model rank: the full-face score comes from the finished v2 diagonal aug_pair_rank run (--train-filter, same seed)
instead of the baseline; the non-skin probe stays on the baseline features. w = 0 is then rank alone, so the verdict asks
whether non-skin evidence still adds on top of smoothing augmentation (rank already removed much of the shift).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

import analyze_smoothing_threshold_shift as shift
import beauty_smoothing_v2 as v2
import diagnose_skin_occlusion as occlusion

diagonal, base, evaluation = occlusion.diagonal, occlusion.base, occlusion.evaluation
INNER = ['l_eye', 'r_eye', 'l_brow', 'r_brow', 'eye_g', 'mouth', 'u_lip', 'l_lip']
WEIGHTS = [0., .1, .2, .3, .4, .5]
VARIANTS = ['noskin', 'face_inner']


class InnerParser(v2.SkinParser):
    """Same pinned face parser as the v2 skin mask; keeps the face-internal non-skin classes."""
    def masks(self, images):
        torch_ = self.torch
        names = {v: int(k) for k, v in self.model.config.id2label.items()}
        missing = [n for n in INNER if n not in names]
        if missing:
            raise ValueError(f'Face parser lacks labels {missing}')
        keep = [names[n] for n in INNER]
        with torch_.inference_mode():
            pixels = self.processor(images=images, return_tensors='pt')['pixel_values'].to(self.device)
            logits = self.model(pixel_values=pixels).logits.float()
        out = []
        for logit, image in zip(logits, images):
            label = torch_.nn.functional.interpolate(logit[None], size=image.size[::-1], mode='bilinear',
                                                     align_corners=False).argmax(1)[0].cpu().numpy()
            side = min(image.size)
            out.append(v2.cv2.GaussianBlur(np.isin(label, keep).astype(np.float32), (0, 0),
                                           max(1., v2.PARSER['blur_fraction']*side)).clip(0, 1))
        return out


def parse_inner(args, sets):
    """Face-inner keep masks of every sampled frame, written once as PNGs under --work-dir."""
    todo = [(root, p) for f, root, _ in sets.values() for p in f.crop_path if not (args.work_dir/'inner'/p).is_file()]
    if not todo:
        return
    parser = InnerParser(args.device)
    for start in range(0, len(todo), args.parse_batch):
        chunk = todo[start:start+args.parse_batch]
        images = []
        for root, p in chunk:
            with Image.open(root/p) as opened:
                images.append(opened.convert('RGB'))
        for (_, p), mask in zip(chunk, parser.masks(images)):
            out = args.work_dir/'inner'/p
            out.parent.mkdir(parents=True, exist_ok=True)
            v2.mask_to_image(mask).save(out)
        print(f'  face-inner masks {min(start+args.parse_batch, len(todo))}/{len(todo)}', flush=True)
    del parser


class Inputs(Dataset):
    """full: original or a v2 smoothed view; noskin: skin mask grey; face_inner: only eyes/brows/lips/mouth kept."""
    def __init__(self, frame, root, cache, mode, transform, work_dir, filt=None, strength=0.):
        self.paths, self.root, self.cache, self.mode, self.transform = frame.crop_path.tolist(), root, cache, mode, transform
        self.work_dir, self.filt, self.strength = work_dir, filt, strength

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(self.root/path) as opened:
            image = opened.convert('RGB')
        if self.mode == 'full' and self.filt is not None:
            image = self.cache.view(image, path, self.filt, self.strength)
        elif self.mode == 'noskin':
            image = occlusion.occlude(image, self.cache.mask(path, image.size))
        elif self.mode == 'face_inner':
            with Image.open(self.work_dir/'inner'/path) as opened:
                keep = v2.image_to_mask(opened)
            image = occlusion.occlude(image, 1.-keep)
        return self.transform(image), index


def load(args, name, weights):
    model, transform = base.build_model(name, pretrained=False)
    model.load_state_dict(weights, strict=True)
    return model.to(args.device).eval(), transform


def worse(row):
    return min(row['fp_cross_auc'], row['fn_cross_auc'])


def run(args):
    sets = occlusion.frames(args)
    caches = {'ffpp_trainval': diagonal.restore_cache(args.cache_root, args.archive_dir, 'ffpp_trainval', ['mask']),
              'cdf_dev': diagonal.restore_cache(args.cache_root, args.archive_dir, 'cdf_dev', v2.SmoothingCache.PARTS)}
    parse_inner(args, sets)
    cdf, cdf_root, _ = sets['cdf_dev']
    y = cdf.class_index.astype(int).to_numpy()
    rows = []
    for name in args.models:
        initial, initial_hash = evaluation.checkpoint(argparse.Namespace(runs_root=args.baseline_runs_root, seed=args.baseline_seed), name)
        if args.full_model == 'rank':
            full_weights, _, epoch = diagonal.completed_checkpoint(args.runs_parent/f'ffpp_c23_smoothing_v2_diagonal_{name}', name,
                                                                    args.train_filter, 'aug_pair_rank', args.seed, initial_hash)
            print(f'{name}: full-face score from aug_pair_rank@{args.train_filter} seed{args.seed} (epoch {epoch})', flush=True)
        else:
            full_weights = initial['model_state_dict']
        model, transform = load(args, name, full_weights)

        def score(s, mode, filt=None, strength=0.):
            f, root, d = sets[s]
            print(f'{name} {s} {mode} {filt or ""} {strength or ""}', flush=True)
            return occlusion.features(args, model, name, Inputs(f, root, caches[d], mode, transform, args.work_dir, filt, strength))

        val_full = score('ffpp_val', 'full')[1]
        full = {'normal': score('cdf_dev', 'full')[1]}
        for filt in v2.FILTERS:
            for condition, strength in v2.LEVELS.items():
                full[(filt, condition)] = score('cdf_dev', 'full', filt, strength)[1]
        zf = lambda m: (m-val_full.mean())/val_full.std()
        if args.full_model == 'rank':   # non-skin probe always on the baseline features
            del model
            model, transform = load(args, name, initial['model_state_dict'])
        for variant in VARIANTS:
            train_f = score('ffpp_train', variant)[0]
            probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.))
            probe.fit(train_f, sets['ffpp_train'][0].class_index.astype(int))
            val_p = probe.decision_function(score('ffpp_val', variant)[0])
            cdf_p = probe.decision_function(score('cdf_dev', variant)[0])
            zp = (cdf_p-val_p.mean())/val_p.std()
            for w in WEIGHTS:
                for filt in v2.FILTERS:
                    margins = {'normal': (1-w)*zf(full['normal'])+w*zp}
                    for condition in v2.LEVELS:
                        margins[condition] = (1-w)*zf(full[(filt, condition)])+w*zp
                    m = shift.measure(y, margins)
                    rows.append(dict(model=name, variant=variant, w=w, filter=filt, normal_auc=m['normal_auc'],
                                     fp_cross_auc=m['fp_cross_auc'], fn_cross_auc=m['fn_cross_auc'], worse_cross_auc=worse(m),
                                     real_shift=m['real_shift'], fake_shift=m['fake_shift'],
                                     nonskin_alone_auc=shift.auc(cdf_p[y == 0], cdf_p[y == 1])))
        del model
        if args.device == 'cuda':
            torch.cuda.empty_cache()
    table = pd.DataFrame(rows).assign(full_model=args.full_model if args.full_model == 'baseline' else f'aug_pair_rank@{args.train_filter}')
    args.output_root.mkdir(parents=True, exist_ok=True)
    stem = 'nonskin_fusion' if args.full_model == 'baseline' else f'nonskin_fusion_on_rank_{args.train_filter}'
    table.to_csv(args.output_root/f'{stem}_seed{args.seed}.csv', index=False)
    verdict = decide(table)
    verdict.to_csv(args.output_root/f'{stem}_verdict_seed{args.seed}.csv', index=False)
    with pd.option_context('display.width', 220, 'display.max_rows', 300):
        print(table.pivot_table(index=['model', 'variant', 'w'], columns='filter', values='worse_cross_auc', sort=False)[v2.FILTERS]
              .join(table.groupby(['model', 'variant', 'w'], sort=False).normal_auc.first()).round(3).to_string())
        print(verdict.round(3).to_string(index=False))
    print('Saved:', args.output_root, flush=True)
    return table


def decide(table):
    """Pre-registered (2026-10-25): some w > 0 keeps clean AUC within -0.01 of w = 0 and raises the worse cross AUC by
    >= +0.02 on at least 3 of the 4 filters."""
    out = []
    for (name, variant), g in table.groupby(['model', 'variant'], sort=False):
        ref = g[g.w.eq(0)].set_index('filter')
        best = dict(model=name, variant=variant, passed=False, w=None, clean_change=None, filters_improved=0)
        for w, h in g[g.w.gt(0)].groupby('w'):
            h = h.set_index('filter')
            clean = h.normal_auc.iloc[0]-ref.normal_auc.iloc[0]
            improved = int((h.worse_cross_auc-ref.worse_cross_auc).ge(.02).sum())
            ok = clean >= -.01 and improved >= 3
            if ok and not best['passed'] or (best['passed'] == ok and improved > best['filters_improved']):
                best.update(passed=ok, w=w, clean_change=clean, filters_improved=improved)
        out.append(best)
    return pd.DataFrame(out)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for field in ['source-root', 'cdf-root', 'membership', 'baseline-runs-root', 'archive-dir', 'output-root']:
        p.add_argument('--'+field, type=Path, required=True)
    p.add_argument('--cache-root', type=Path, default=Path('/content/beauty_smoothing_v2_cache'))
    p.add_argument('--work-dir', type=Path, default=Path('/content/nonskin_fusion_work'))
    p.add_argument('--full-model', choices=['baseline', 'rank'], default='baseline')
    p.add_argument('--runs-parent', type=Path, help='rank: folder holding ffpp_c23_smoothing_v2_diagonal_<model>')
    p.add_argument('--train-filter', choices=v2.FILTERS, default='bilateral', help='rank: training filter of the aug_pair_rank run')
    p.add_argument('--models', nargs='+', choices=list(base.PREPROCESS), default=['xception', 'clip_b16', 'dinov2_b14'])
    for field, default in [('seed', 42), ('baseline-seed', 42), ('train-images', 2000), ('val-images', 1200), ('cdf-images', 2000),
                           ('batch-size', 64), ('workers', 2), ('parse-batch', 16)]:
        p.add_argument('--'+field, type=int, default=default)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    a = p.parse_args(argv)
    if a.full_model == 'rank' and a.runs_parent is None:
        p.error('--full-model rank needs --runs-parent')
    if a.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime')
    return a


if __name__ == '__main__':
    run(parse_args())
