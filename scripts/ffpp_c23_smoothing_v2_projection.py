"""Method A' (aug_pair_rank_proj<k>): remove the smoothing-trace directions from a trained aug_pair_rank model, post hoc.

Candidate 3 showed (Xception, v1) that removing the bilateral trace direction and refitting the head on ORIGINAL frames
only erased the held-out-filter reversal: compensation lives in a head that learned "trace seen -> subtract", and on
the trace directions. Here, for the completed aug_pair_rank run of train filter k (same seed), on FF++ TRAIN only:
  1. Features (head input, eval mode, FP32) of N balanced train frames: original + filter-k view at the same per-frame
     random weak/medium/strong strength as training.
  2. Directions U = top-k right singular vectors of f(smoothed) - f(original). Projection P = I - U U^T.
  3. Head refit on P f(original) only (standardised logistic regression), folded back into the model's own linear head:
     margin = (P w')·f + b'. The backbone is untouched, so the result is an ordinary checkpoint for the v2 evaluation.
Saved as a completed run aug_pair_rank_proj<k> (best.pt = last.pt) beside the diagonal runs. No Celeb-DF, no val/test.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

import beauty_smoothing_v2 as v2
import ffpp_c23_smoothing_v2_diagonal as diagonal

base, evaluation, adaptation, common, engine = diagonal.base, diagonal.evaluation, diagonal.adaptation, diagonal.common, diagonal.engine
PROTOCOL = 'ffpp_c23_smoothing_v2_projection_v1'


class TrainPairs(Dataset):
    def __init__(self, frame, args, transform, cache):
        self.rows, self.args, self.transform, self.cache = frame.to_dict('records'), args, transform, cache

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        strength = random.Random(self.args.seed+index).randrange(1, 4)
        with Image.open(self.args.source_root/row['crop_path']) as opened:
            clean = opened.convert('RGB')
        smooth = self.cache.view(clean, row['crop_path'], self.args.train_filter, v2.LEVELS[diagonal.CONDITIONS[strength]],
                                 row['roi_status'] == 'detected')
        return self.transform(clean), self.transform(smooth), int(row['class_index']), index


def features(args, name, model, dataset):
    f0, f1, labels = [], [], []
    with torch.inference_mode():
        for clean, smooth, y, _ in DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers):
            f0.append(adaptation.outputs(name, model, clean.to(args.device))[1].float().cpu().numpy())
            f1.append(adaptation.outputs(name, model, smooth.to(args.device))[1].float().cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(f0), np.concatenate(f1), np.concatenate(labels)


def projected_head(f0, f1, y, k, c):
    """(weight[2,D], bias[2], directions[D,k]) of the refit head applied to RAW features: margin = (P w')·f + b'."""
    u = np.linalg.svd(f1-f0, full_matrices=False)[2][:k].T
    projected = f0-(f0@u)@u.T
    scaler = StandardScaler().fit(projected)
    clf = LogisticRegression(max_iter=5000, C=c).fit(scaler.transform(projected), y)
    w = clf.coef_[0]/scaler.scale_
    b = float(clf.intercept_[0]-(w*scaler.mean_).sum())
    w = w-u@(u.T@w)                                          # P w (P symmetric): the head never sees the trace directions
    weight = np.stack([-w/2, w/2]).astype(np.float32)        # logit1 - logit0 = w·f + b
    return weight, np.array([-b/2, b/2], np.float32), u


def build(args):
    source, _, data = common.shared.source_inputs(args)
    cache = diagonal.restore_cache(args.cache_root, args.archive_dir, 'ffpp_trainval', ['mask', args.train_filter])
    frame = adaptation.balanced_source(source, args.images, args.seed)
    for name in args.models:
        initial, initial_hash = evaluation.checkpoint(argparse.Namespace(runs_root=args.baseline_runs_root, seed=args.baseline_seed), name)
        weights, source_digest, epoch = diagonal.completed_checkpoint(args.runs_root, name, args.train_filter, 'aug_pair_rank',
                                                                       args.seed, initial_hash)
        todo = []
        for k in args.proj_ks:
            run = args.runs_root/diagonal.run_name(name, args.train_filter, f'aug_pair_rank_proj{k}', args.seed)
            config = dict(protocol=PROTOCOL, model=name, train_filter=args.train_filter, strategy='aug_pair_rank_proj', k=k,
                          seed=args.seed, baseline_seed=args.baseline_seed, initial_checkpoint_sha256=initial_hash,
                          source_run='aug_pair_rank', source_checkpoint_sha256=source_digest, source_epoch=epoch,
                          images=args.images, c=args.c, data=data, smoothing=dict(cache.identity, levels=v2.LEVELS),
                          method='top-k SVD trace directions of f(train-filter view) - f(original) on FF++ train; head refit on projected ORIGINAL features (standardised logistic), folded into the linear head',
                          code_sha256={Path(__file__).name: base.common.digest(Path(__file__)),
                                       Path(diagonal.__file__).name: base.common.digest(Path(diagonal.__file__))})
            if (run/'completed.json').is_file() and json.loads((run/'config.json').read_text()) == config:
                print('=== ALREADY COMPLETED', name, args.train_filter, f'aug_pair_rank_proj{k}', '===', flush=True)
                continue
            todo.append((k, run, config))
        if not todo:
            continue
        model, transform = base.build_model(name, pretrained=False)
        model.load_state_dict(weights, strict=True)
        model.to(args.device).eval()
        print(f'{name}: features of {len(frame)} FF++ train frames (original + {args.train_filter}) from aug_pair_rank epoch {epoch}', flush=True)
        f0, f1, y = features(args, name, model, TrainPairs(frame, args, transform, cache))
        head = base.head(model, name)
        for k, run, config in todo:
            weight, bias, u = projected_head(f0, f1, y, k, args.c)
            state = {key: value.clone() for key, value in model.state_dict().items()}
            prefix = next(key for key, value in model.state_dict().items() if value.data_ptr() == head.weight.data_ptr())[:-len('weight')]
            state[prefix+'weight'] = torch.from_numpy(weight)
            state[prefix+'bias'] = torch.from_numpy(bias)
            identity = evaluation.key(config)
            run.mkdir(parents=True, exist_ok=True)
            engine.freeze_json(run/'config.json', config)
            for file in ['best.pt', 'last.pt']:
                base.common.save(run/file, dict(identity=identity, config=config, model_state_dict=state, epoch=epoch))
            np.save(run/'trace_directions.npy', u.astype(np.float32))
            shift = (f1-f0)@weight[1]-(f1-f0)@weight[0]
            base.common.json_write(run/'completed.json', dict(identity=identity, selected_epoch=epoch, k=k,
                train_score_shift_real=float(shift[y == 0].mean()), train_score_shift_fake=float(shift[y == 1].mean())))
            print(f'=== BUILT {name} {args.train_filter} aug_pair_rank_proj{k}: train score shift real {shift[y == 0].mean():+.3f} '
                  f'fake {shift[y == 1].mean():+.3f} (raw margin) ===', flush=True)
        del model
        if args.device == 'cuda':
            torch.cuda.empty_cache()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for field in ['source-root', 'baseline-runs-root', 'runs-root', 'archive-dir']:
        p.add_argument('--'+field, type=Path, required=True)
    p.add_argument('--source-validation', type=Path)
    p.add_argument('--cache-root', type=Path, default=Path('/content/beauty_smoothing_v2_cache'))
    p.add_argument('--train-filter', choices=v2.FILTERS, required=True)
    p.add_argument('--models', nargs='+', choices=list(base.PREPROCESS), default=['clip_b16'])
    p.add_argument('--proj-ks', nargs='+', type=int, default=[4])
    for field, default in [('seed', 42), ('baseline-seed', 42), ('images', 6000), ('val-images', 1200), ('batch-size', 64), ('workers', 2)]:
        p.add_argument('--'+field, type=int, default=default)
    p.add_argument('--c', type=float, default=1.)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    a = p.parse_args(argv)
    if a.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime')
    return a


if __name__ == '__main__':
    build(parse_args())
