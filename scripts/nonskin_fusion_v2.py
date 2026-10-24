"""aug_pair_rank_nsf: the finished aug_pair_rank detector fused with a smoothing-invariant non-skin score (post-hoc).

v2 smoothing edits only the skin mask, so a score computed from the face-internal non-skin parts (eyes, brows, lips,
mouth; same pinned SegFormer parser, rest filled grey) is unchanged by every v2 filter. Settings FIXED on 2026-10-25
before any cdf_test look (chosen on cdf_dev bilateral seed42 as the largest weight with no clean-AUC loss on all three
backbones; RESEARCH_CONTEXT):
  non-skin scorer  standardized logistic regression (C=1) on the FF++ BASELINE (seed 42) classifier-input features of
                   the face_inner view, fitted once per backbone on 2,000 balanced FF++ TRAIN frames (seed 42)
  fusion           (1-w) z(rank margin) + w z(non-skin score), w = 0.2; z from 1,200 balanced FF++ VAL clean frames
                   (rank: per run; non-skin: per backbone)
build() writes the shared scorer (<runs_root>/nonskin_probe/) and, per completed aug_pair_rank run, an
aug_pair_rank_nsf run folder holding the z statistics; evaluate_beauty_smoothing_v2.py then writes ordinary prediction
CSVs for it from the saved rank predictions and the per-frame non-skin scores. No training, no Celeb-DF frame used.
"""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import diagnose_nonskin_fusion as fusion
import ffpp_c23_smoothing_v2_diagonal as diagonal

occlusion, base, evaluation = fusion.occlusion, diagonal.base, diagonal.evaluation
LABEL = 'aug_pair_rank_nsf'
SETTINGS = dict(variant='face_inner', weight=.2, probe_c=1., train_images=2000, val_images=1200, sample_seed=42)
PROTOCOL = 'nonskin_fusion_v2_v1'


def parse(paths, root, work_dir, device, batch):
    """face_inner keep masks (PNG under work_dir/inner/<crop_path>) for every path not parsed yet."""
    todo = [p for p in paths if not (work_dir/'inner'/p).is_file()]
    if not todo:
        return
    parser = fusion.InnerParser(device)
    for start in range(0, len(todo), batch):
        chunk = todo[start:start+batch]
        images = []
        for p in chunk:
            with Image.open(root/p) as opened:
                images.append(opened.convert('RGB'))
        for p, mask in zip(chunk, parser.masks(images)):
            out = work_dir/'inner'/p
            out.parent.mkdir(parents=True, exist_ok=True)
            fusion.v2.mask_to_image(mask).save(out)
        if (start//batch) % 20 == 0 or start+batch >= len(todo):
            print(f'  face-inner masks {min(start+batch, len(todo))}/{len(todo)}', flush=True)
    del parser


def outputs(args, model, name, transform, frame, root, mode):
    return occlusion.features(args, model, name, fusion.Inputs(frame, root, None, mode, transform, args.work_dir))


def load(args, name, weights):
    model, transform = base.build_model(name, pretrained=False)
    model.load_state_dict(weights, strict=True)
    return model.to(args.device).eval(), transform


def ffpp_samples(args):
    frame = pd.read_csv(args.source_root/'manifests/ffpp_c23_4methods_roi16_trainval_v1.csv', dtype=str, keep_default_na=False)
    seed = SETTINGS['sample_seed']
    return (occlusion.balanced(frame[frame.split.eq('train')], SETTINGS['train_images'], seed),
            occlusion.balanced(frame[frame.split.eq('val')], SETTINGS['val_images'], seed))


def scorer(args, name):
    """(pipeline, info) of the shared non-skin scorer of this backbone; fitted once, then loaded."""
    folder = args.runs_root/'nonskin_probe'
    pkl, info_path = folder/f'{name}_face_inner_seed{args.baseline_seed}.pkl', folder/f'{name}_face_inner_seed{args.baseline_seed}.json'
    _, initial_hash = evaluation.checkpoint(argparse.Namespace(runs_root=args.baseline_runs_root, seed=args.baseline_seed), name)
    if pkl.is_file() and info_path.is_file():
        info = json.loads(info_path.read_text())
        if info['baseline_sha256'] != initial_hash or info['settings'] != SETTINGS:
            raise ValueError(f'Non-skin scorer was built with other settings: {info_path}')
        if info['pickle_sha256'] != base.common.digest(pkl):
            raise ValueError(f'Non-skin scorer file changed: {pkl}')
        with open(pkl, 'rb') as f:
            return pickle.load(f), info
    if getattr(args, 'source_root', None) is None:
        raise FileNotFoundError(f'Non-skin scorer missing; run nonskin_fusion_v2.py (build) first: {pkl}')
    train, val = ffpp_samples(args)
    parse([*train.crop_path, *val.crop_path], args.source_root, args.work_dir, args.device, args.parse_batch)
    initial, _ = evaluation.checkpoint(argparse.Namespace(runs_root=args.baseline_runs_root, seed=args.baseline_seed), name)
    model, transform = load(args, name, initial['model_state_dict'])
    f_train = outputs(args, model, name, transform, train, args.source_root, SETTINGS['variant'])[0]
    f_val = outputs(args, model, name, transform, val, args.source_root, SETTINGS['variant'])[0]
    del model
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=SETTINGS['probe_c']))
    pipe.fit(f_train, train.class_index.astype(int))
    s_val = pipe.decision_function(f_val)
    folder.mkdir(parents=True, exist_ok=True)
    with open(pkl, 'wb') as f:
        pickle.dump(pipe, f)
    info = dict(protocol=PROTOCOL, model=name, settings=SETTINGS, baseline_sha256=initial_hash, pickle_sha256=base.common.digest(pkl),
                val_mean=float(s_val.mean()), val_std=float(s_val.std()),
                train_paths_sha256=hashlib.sha256('\n'.join(train.crop_path).encode()).hexdigest(),
                val_paths_sha256=hashlib.sha256('\n'.join(val.crop_path).encode()).hexdigest(),
                ffpp_val_auc=float(fusion.shift.auc(s_val[val.class_index.eq('0').to_numpy()], s_val[val.class_index.eq('1').to_numpy()])))
    base.common.json_write(info_path, info)
    print(f'{name}: non-skin scorer fitted (FF++ val AUC {info["ffpp_val_auc"]:.3f})', flush=True)
    return pipe, info


def build(args):
    _, val = ffpp_samples(args)
    for name in args.models:
        _, info = scorer(args, name)
        initial, initial_hash = evaluation.checkpoint(argparse.Namespace(runs_root=args.baseline_runs_root, seed=args.baseline_seed), name)
        run = args.runs_root/diagonal.run_name(name, args.train_filter, LABEL, args.seed)
        weights, rank_sha, epoch = diagonal.completed_checkpoint(args.runs_root, name, args.train_filter, 'aug_pair_rank', args.seed, initial_hash)
        config = dict(protocol=PROTOCOL, model=name, train_filter=args.train_filter, seed=args.seed, baseline_seed=args.baseline_seed,
                      source_run='aug_pair_rank', source_checkpoint_sha256=rank_sha, source_epoch=epoch, settings=SETTINGS,
                      scorer_pickle_sha256=info['pickle_sha256'])
        if (run/'completed.json').is_file() and json.loads((run/'config.json').read_text()) == config:
            print('=== ALREADY COMPLETED', name, args.train_filter, LABEL, f'seed{args.seed}', '===', flush=True)
            continue
        model, transform = load(args, name, weights)
        margins = outputs(args, model, name, transform, val, args.source_root, 'full')[1]
        del model
        if args.device == 'cuda':
            torch.cuda.empty_cache()
        run.mkdir(parents=True, exist_ok=True)
        base.common.json_write(run/'config.json', config)
        base.common.json_write(run/'completed.json', dict(identity=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
            selected_epoch=epoch, rank_val_mean=float(margins.mean()), rank_val_std=float(margins.std()),
            scorer_val_mean=info['val_mean'], scorer_val_std=info['val_std'], weight=SETTINGS['weight']))
        print(f'=== BUILT {name} {args.train_filter} {LABEL} seed{args.seed} (rank epoch {epoch}) ===', flush=True)


def dataset_scores(args, name, dataset, frame, data_root):
    """Per-frame non-skin scores of an evaluation set, cached next to the scorer (same for every run of this backbone)."""
    pipe, info = scorer(args, name)
    path = args.runs_root/'nonskin_probe'/f'{name}_{dataset}_face_inner_scores.csv'
    if path.is_file():
        saved = pd.read_csv(path, dtype={'crop_path': str})
        if saved.crop_path.tolist() == frame.crop_path.tolist() and saved.scorer_pickle_sha256.iloc[0] == info['pickle_sha256']:
            return saved.score.to_numpy()
    parse(frame.crop_path.tolist(), data_root, args.work_dir/dataset, args.device, args.parse_batch)
    initial, _ = evaluation.checkpoint(argparse.Namespace(runs_root=args.baseline_runs_root, seed=args.baseline_seed), name)
    model, transform = load(args, name, initial['model_state_dict'])
    sub = argparse.Namespace(**{**vars(args), 'work_dir': args.work_dir/dataset})
    features = outputs(sub, model, name, transform, frame, data_root, SETTINGS['variant'])[0]
    del model
    score = pipe.decision_function(features)
    out = frame[['crop_path']].copy()
    out['score'], out['scorer_pickle_sha256'] = score, info['pickle_sha256']
    temp = path.with_suffix('.tmp.csv'); out.to_csv(temp, index=False); temp.replace(path)
    return score


def fused(rank_margin, nonskin, done):
    w = done['weight']
    return ((1-w)*(rank_margin-done['rank_val_mean'])/done['rank_val_std']
            + w*(nonskin-done['scorer_val_mean'])/done['scorer_val_std'])


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for field in ['source-root', 'baseline-runs-root', 'runs-root']:
        p.add_argument('--'+field, type=Path, required=True)
    p.add_argument('--train-filter', choices=diagonal.v2.FILTERS, required=True)
    p.add_argument('--models', nargs='+', choices=list(base.PREPROCESS), default=['clip_b16'])
    p.add_argument('--work-dir', type=Path, default=Path('/content/nsf_work'))
    for field, default in [('seed', 42), ('baseline-seed', 42), ('batch-size', 64), ('workers', 2), ('parse-batch', 16)]:
        p.add_argument('--'+field, type=int, default=default)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    a = p.parse_args(argv)
    if a.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime')
    return a


if __name__ == '__main__':
    build(parse_args())
