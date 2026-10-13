"""Frozen official Celeb-DF test: paired image-only evaluation, no adaptation."""
import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault('USE_TF', '0')
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import train_ffpp_c23_baseline as baseline
from smoothing_roi_pilot import smooth

ROOT = Path(__file__).resolve().parents[1]


def key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def inputs(args):
    cfg = json.loads(args.protocol.read_text())
    frame = pd.read_csv(args.manifest, dtype=str, keep_default_na=False)
    required = {'crop_path', 'video_id', 'class_index', 'split', 'roi_status'}
    if not required.issubset(frame.columns):
        raise ValueError(f'Missing columns: {required-set(frame.columns)}')
    if not frame.split.eq('test').all() or frame.crop_path.duplicated().any():
        raise ValueError('Official test-only, unique image manifest required')
    counts = frame.class_index.value_counts().to_dict()
    if (len(frame) != cfg['test_images'] or frame.video_id.nunique() != cfg['test_videos']
            or counts != {'0': cfg['real_images'], '1': cfg['fake_images']}):
        raise ValueError(f'Official test counts differ: {len(frame)}, {counts}')
    for path in frame.crop_path:
        relative = Path(path)
        if relative.is_absolute() or '..' in relative.parts or not (args.data_root/relative).is_file():
            raise FileNotFoundError(args.data_root/relative)
    provenance = {'manifest_sha256': baseline.common.digest(args.manifest), 'config': cfg,
                  'opencv': cv2.__version__,
                  'smoothing_code_sha256': baseline.common.digest(ROOT/'scripts/smoothing_roi_pilot.py')}
    return frame, cfg, provenance


def view_path(args, condition, row):
    if condition == 'normal' or row['roi_status'] != 'detected':
        return args.data_root/row['crop_path']
    return args.cache_root/condition/row['crop_path']


def prepare(args):
    frame, cfg, provenance = inputs(args)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    config_path = args.cache_root/'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != provenance:
        raise ValueError('Cache protocol changed; use a new cache root')
    baseline.common.json_write(config_path, provenance)

    def one(row):
        if row['roi_status'] != 'detected':
            return
        missing = [c for c in cfg['conditions'] if not view_path(args, c, row).is_file()]
        if missing:
            with Image.open(args.data_root/row['crop_path']) as image:
                image = image.convert('RGB')
                for condition in missing:
                    dest = view_path(args, condition, row)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    temp = dest.with_suffix('.tmp.png')
                    smooth(image, cfg['conditions'][condition], cfg['bilateral']).save(temp, format='PNG')
                    temp.replace(dest)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for _ in tqdm(pool.map(one, frame.to_dict('records')), total=len(frame),
                      desc='Prepare smoothing', mininterval=5):
            pass
    baseline.common.json_write(args.cache_root/'ready.json', {'identity': key(provenance)})
    # Freeze exact evaluation membership for future adaptation exclusion.
    args.output_root.mkdir(parents=True, exist_ok=True)
    frozen = args.output_root/'evaluation_manifest.csv'
    if frozen.exists() and frozen.read_bytes() != args.manifest.read_bytes():
        raise ValueError('Evaluation membership changed; use a new output root')
    if not frozen.exists():
        frozen.write_bytes(args.manifest.read_bytes())
    videos = sorted(frame.video_id.unique().tolist())
    baseline.common.json_write(args.output_root/'evaluation_protocol.json', {
        **provenance, 'excluded_from_adaptation_video_ids': videos,
        'identity_overlap_allowed': True, 'source_parent_audit_complete': False})
    print(f'Ready: {len(frame)} images, fallback unchanged: {frame.roi_status.ne("detected").sum()}', flush=True)


class Views(Dataset):
    def __init__(self, frame, args, transform, condition):
        self.rows = frame.to_dict('records')
        self.args, self.transform, self.condition = args, transform, condition

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        with Image.open(view_path(self.args, self.condition, self.rows[index])) as image:
            return self.transform(image.convert('RGB')), index


def checkpoint(args, name):
    run = args.runs_root/f'{name}_ffpp_c23_roi16_baseline_v1_seed{args.seed}'
    for file in ['best.pt', 'config.json', 'completed.json']:
        if not (run/file).is_file():
            raise FileNotFoundError(f'Training must be complete: {run/file}')
    state = torch.load(run/'best.pt', map_location='cpu', weights_only=True)
    cfg = state['config']
    saved = json.loads((run/'config.json').read_text())
    completed = json.loads((run/'completed.json').read_text())
    if (cfg != saved or state['identity'] != key(cfg) or completed['identity'] != state['identity']
            or cfg['model'] != name or cfg['seed'] != args.seed
            or cfg['protocol'] != 'ffpp_c23_4methods_baseline_v1'
            or cfg['preprocessing'] != baseline.PREPROCESS[name]):
        raise ValueError(f'Checkpoint/config mismatch: {run}')
    return state, baseline.common.digest(run/'best.pt')


def evaluate(args):
    frame, cfg, provenance = inputs(args)
    if json.loads((args.cache_root/'ready.json').read_text())['identity'] != key(provenance):
        raise ValueError('Run prepare with the same protocol first')
    for condition in cfg['conditions']:
        for row in frame.to_dict('records'):
            if not view_path(args, condition, row).is_file():
                raise FileNotFoundError(view_path(args, condition, row))
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('GPU runtime required')
    # Fail early on an incomplete requested model, before expensive inference.
    for name in args.models:
        state, _ = checkpoint(args, name)
        del state
    summaries, deltas = [], []
    for name in args.models:
        state, checkpoint_hash = checkpoint(args, name)
        identity = key({'data': provenance, 'checkpoint': checkpoint_hash, 'precision': 'FP32',
                        'torch': str(torch.__version__),
                        'evaluation_code': baseline.common.digest(Path(__file__)),
                        'model_code': baseline.common.digest(Path(baseline.__file__)),
                        'shared_model_code': baseline.common.digest(Path(baseline.common.__file__))})
        model, transform = None, None
        clean = None
        for condition in cfg['conditions']:
            prefix = args.output_root/f'{name}_{condition}'
            metrics_path, prediction_path = prefix.with_suffix('.json'), prefix.with_suffix('.csv')
            values = None
            if metrics_path.is_file() and prediction_path.is_file():
                saved = json.loads(metrics_path.read_text())
                predicted = pd.read_csv(prediction_path, dtype={'crop_path': str, 'class_index': str})
                if saved.get('identity') != identity:
                    raise ValueError(f'Existing result differs; preserve it and use a new output root: {metrics_path}')
                if (predicted.crop_path.tolist() == frame.crop_path.tolist()
                        and predicted.class_index.tolist() == frame.class_index.tolist()
                        and len(predicted) == len(frame) and np.isfinite(predicted.probability_fake).all()
                        and predicted.probability_fake.between(0, 1).all()):
                    values = saved
                    print('Reuse completed:', name, condition, flush=True)
            if values is None:
                if model is None:
                    model, transform = baseline.build_model(name, pretrained=False)
                    model.load_state_dict(state['model_state_dict'], strict=True)
                    model.to(args.device).eval()
                    del state
                loader = DataLoader(Views(frame, args, transform, condition), batch_size=args.batch_size,
                                    num_workers=args.workers, shuffle=False, pin_memory=args.device == 'cuda')
                probabilities = np.empty(len(frame))
                with torch.inference_mode():
                    for images, indexes in tqdm(loader, desc=f'{name} / {condition}', mininterval=5):
                        logits = model(images.to(args.device, non_blocking=True))
                        p = logits.float().softmax(1)[:, 1].cpu().numpy()
                        if not np.isfinite(p).all():
                            raise RuntimeError('Nonfinite prediction; no completed result saved')
                        probabilities[indexes.numpy()] = p
                predicted = frame[['crop_path', 'video_id', 'class_index', 'roi_status']].copy()
                predicted['probability_fake'] = probabilities
                values = baseline.binary_metrics(frame.class_index.astype(int).to_numpy(), probabilities)
                values.update(identity=identity, model=name, condition=condition, images=len(frame),
                              checkpoint_sha256=checkpoint_hash, threshold=0.5, precision='FP32',
                              manifest_sha256=provenance['manifest_sha256'], unit='frame')
                temp = prediction_path.with_suffix('.tmp.csv')
                predicted.to_csv(temp, index=False)
                temp.replace(prediction_path)
                baseline.common.json_write(metrics_path, values)
            summaries.append({k: values[k] for k in ['model', 'condition', 'images', 'auc', 'accuracy', 'f1', 'real_fpr', 'fake_fnr']})
            if condition == 'normal':
                clean = values
            else:
                deltas.append({'model': name, 'condition': condition,
                               **{f'delta_{k}': values[k]-clean[k] for k in ['auc', 'real_fpr', 'fake_fnr']}})
            print(f'{name} {condition}: AUC={values["auc"]:.4f}, FPR={values["real_fpr"]:.2%}, FNR={values["fake_fnr"]:.2%}', flush=True)
        del model
        if 'state' in locals():
            del state
        if args.device == 'cuda':
            torch.cuda.empty_cache()
    pd.DataFrame(summaries).to_csv(args.output_root/'condition_summary.csv', index=False)
    pd.DataFrame(deltas).to_csv(args.output_root/'paired_deltas.csv', index=False)
    print('Saved:', args.output_root, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'evaluate'])
    for field in ['data-root', 'manifest', 'cache-root', 'runs-root', 'output-root']:
        p.add_argument('--'+field, type=Path, required=True)
    p.add_argument('--protocol', type=Path, default=ROOT/'configs/ffpp_c23_celebdf_smoothing_v1.json')
    p.add_argument('--models', nargs='+', choices=list(baseline.PREPROCESS), default=list(baseline.PREPROCESS))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = p.parse_args()
    (prepare if args.action == 'prepare' else evaluate)(args)


if __name__ == '__main__':
    main()
