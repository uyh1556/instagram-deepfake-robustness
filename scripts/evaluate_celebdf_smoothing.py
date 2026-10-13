"""Paired clean/three-strength evaluation of the two frozen Celeb-DF baselines."""
import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset
from tqdm import tqdm

import train_celebdf_baseline as baseline
from smoothing_roi_pilot import smooth

ROOT = Path(__file__).resolve().parents[1]


def key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def inputs(args):
    frame, split = baseline.load_manifest(args)
    frame = frame[frame.split.eq('test')].reset_index(drop=True)
    if len(frame) != 1824 or frame.video_id.nunique() != 114:
        raise ValueError('Expected frozen80/10/10 seed42 test1824 images /114 videos')
    protocol = json.loads(args.protocol.read_text())
    provenance = {'protocol': protocol, 'manifest_sha256': split['manifest_sha256'],
                  'test_paths_sha256': key(frame.crop_path.tolist()), 'opencv': cv2.__version__,
                  'smoothing_code_sha256': baseline.digest(ROOT/'scripts/smoothing_roi_pilot.py'),
                  'evaluator_code_sha256': baseline.digest(Path(__file__)), 'seed': args.seed}
    return frame, split, protocol, provenance


def cached(args, condition, row):
    return args.cache_root / condition / row['crop_path']


def prepare(args):
    frame, _, cfg, provenance = inputs(args)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    config = args.cache_root/'config.json'
    if config.exists() and json.loads(config.read_text()) != provenance:
        raise ValueError('Cache configuration differs; preserve it and use another cache root')
    baseline.json_write(config, provenance)
    def one(row):
        if row['roi_status'] != 'detected':
            return
        missing = [c for c in cfg['conditions'] if c != 'normal' and not cached(args, c, row).is_file()]
        if not missing:
            return
        with Image.open(args.data_root/row['crop_path']) as f:
            original = f.convert('RGB')
            for condition in missing:
                dest = cached(args, condition, row)
                dest.parent.mkdir(parents=True, exist_ok=True)
                temp = dest.with_suffix('.tmp.png')
                smooth(original, cfg['conditions'][condition], cfg['bilateral']).save(temp, format='PNG')
                temp.replace(dest)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for _ in tqdm(pool.map(one, frame.to_dict('records')), total=len(frame), desc='Native smoothing cache', mininterval=2):
            pass
    baseline.json_write(args.cache_root/'ready.json', {'identity': key(provenance)})
    fallback = frame[frame.roi_status.ne('detected')]
    print('Test:', len(frame), 'fallback unchanged:', len(fallback), fallback.class_index.value_counts().to_dict(), flush=True)
    # Train-side illustrations only: do not choose smoothing parameters from test previews.
    full, _ = baseline.load_manifest(args)
    examples = full[(full.split.eq('train')) & full.roi_status.eq('detected')].groupby('class_index').head(1)
    if len(examples) and all((args.data_root/r.crop_path).is_file() for r in examples.itertuples()):
        canvas = Image.new('RGB', (224*4, 250*len(examples)), 'white')
        draw = ImageDraw.Draw(canvas)
        for iy, row in enumerate(examples.to_dict('records')):
            with Image.open(args.data_root/row['crop_path']) as f:
                original = f.convert('RGB')
                for ix, (name, strength) in enumerate(cfg['conditions'].items()):
                    view = original if not strength else smooth(original, strength, cfg['bilateral'])
                    canvas.paste(view.resize((224,224)), (224*ix, 250*iy+26))
                    draw.text((224*ix+3,250*iy+4), row['label']+' '+name, fill='black')
        canvas.save(args.cache_root/'train_preview.png')


class Views(Dataset):
    def __init__(self, frame, args, transform, condition):
        self.rows, self.args = frame.to_dict('records'), args
        self.transform, self.condition = transform, condition

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.args.data_root/row['crop_path']
        if self.condition != 'normal' and row['roi_status'] == 'detected':
            path = cached(self.args, self.condition, row)
        with Image.open(path) as f:
            x = self.transform(f.convert('RGB'))
        return x, int(row['class_index']), index


def infer(model, loader, frame, args, title):
    model.eval()
    probabilities = np.empty(len(frame))
    with torch.inference_mode():
        for x, _, indexes in tqdm(loader, desc=title, mininterval=2):
            with torch.autocast(device_type=args.device, enabled=args.device == 'cuda'):
                logits = model(x.to(args.device))
            probabilities[indexes.numpy()] = logits.float().softmax(1)[:,1].cpu().numpy()
    return baseline.metrics(frame, probabilities)


def paired_bootstrap(clean, changed, repeats, seed):
    videos = clean.groupby('video_id').agg(y=('class_index','first'), p=('probability_fake','mean'))
    shifted = changed.groupby('video_id').probability_fake.mean().reindex(videos.index).to_numpy()
    y, p = videos.y.astype(int).to_numpy(), videos.p.to_numpy()
    rng = np.random.default_rng(seed)
    real, fake = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    records = []
    for _ in range(repeats):
        idx = np.r_[rng.choice(real, len(real)), rng.choice(fake, len(fake))]
        yr, a, b = y[idx], p[idx], shifted[idx]
        records.append([roc_auc_score(yr,b)-roc_auc_score(yr,a),
                        np.mean(b[yr==0]>=.5)-np.mean(a[yr==0]>=.5),
                        np.mean(b[yr==1]<.5)-np.mean(a[yr==1]<.5)])
    bounds = np.quantile(records, [.025,.975], axis=0)
    return {f'delta_video_{metric}_ci95_{end}': float(bounds[i,j])
            for j, metric in enumerate(['auc','real_fpr','fake_fnr']) for i,end in enumerate(['low','high'])}


def evaluate(args):
    frame, split, protocol, provenance = inputs(args)
    if json.loads((args.cache_root/'ready.json').read_text())['identity'] != key(provenance):
        raise ValueError('Prepare matching cache first')
    args.output_root.mkdir(parents=True, exist_ok=True)
    all_summaries, changes = [], []
    for name in args.models:
        run = args.runs_root/f'{name}_celebdf_identity_v1_seed{args.seed}'
        checkpoint = run/'best.pt'
        # Small copied manifest/config and checkpoint checks, no archive-wide hash scans.
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if state['config']['split'] != split or state['config']['model'] != name or state['config']['seed'] != args.seed:
            raise ValueError('Checkpoint split/model/seed mismatch')
        checkpoint_hash = baseline.digest(checkpoint)
        identity = key({'cache': provenance, 'checkpoint': checkpoint_hash, 'torch': torch.__version__,
                        'bootstrap_repeats': args.bootstrap_repeats})
        model, transform = baseline.model_and_transform(name, pretrained=False)
        model.load_state_dict(state['model_state_dict'], strict=True)
        model.to(args.device)
        del state
        clean, clean_values = None, None
        for condition in protocol['conditions']:
            prefix = args.output_root/f'{name}_{condition}'
            metrics_path = prefix.with_suffix('.json')
            prediction_path = prefix.with_suffix('.csv')
            if metrics_path.exists() and prediction_path.exists():
                saved = json.loads(metrics_path.read_text())
                if saved['identity'] != identity:
                    raise ValueError('Saved evaluation identity mismatch; use new output root')
                values = saved['metrics']
                predictions = pd.read_csv(prediction_path, dtype={'video_id':str,'crop_path':str,'roi_status':str})
                if predictions.crop_path.tolist() != frame.crop_path.tolist():
                    raise ValueError('Prediction rows do not match test')
                print('Reuse completed:', name, condition, flush=True)
            else:
                loader = baseline.make_loader(Views(frame, args, transform, condition), args)
                values, predictions = infer(model, loader, frame, args, f'{name} {condition}')
                predictions.to_csv(prediction_path, index=False)
                baseline.json_write(metrics_path, {'identity': identity, 'metrics': values})
            row = {'model': name, 'condition': condition, 'blend': protocol['conditions'][condition],
                   'images': len(frame), 'videos': frame.video_id.nunique(),
                   'unchanged_fallback_images': int(frame.roi_status.ne('detected').sum())}
            row.update({f'{level}_{metric}': values[level][metric] for level in ['frame','video']
                        for metric in ['auc','accuracy','f1','real_fpr','fake_fnr']})
            detected = predictions[predictions.roi_status.eq('detected')]
            if set(detected.class_index.astype(int)) == {0,1}:
                v, _ = baseline.metrics(detected, detected.probability_fake.to_numpy())
                row.update({f'detected_frame_{m}': v['frame'][m] for m in ['auc','real_fpr','fake_fnr']})
            all_summaries.append(row)
            if condition == 'normal':
                clean, clean_values = predictions, row
                reference_path = run/'test_metrics.json'
                if reference_path.exists():
                    reference = json.loads(reference_path.read_text())
                    if reference['manifest_sha256'] != split['manifest_sha256']:
                        raise ValueError('Original reference manifest mismatch')
                    if reference['checkpoint_sha256'] != checkpoint_hash:
                        raise ValueError('Original reference belongs to another checkpoint')
                    differences = {f'{level}_{m}': values[level][m]-reference[level][m]
                                   for level in ['frame','video'] for m in ['auc','real_fpr','fake_fnr']}
                    baseline.json_write(args.output_root/f'{name}_clean_reproduction.json', differences)
                    if max(abs(v) for v in differences.values()) > .002:
                        raise RuntimeError('Clean baseline does not reproduce within0.002; inspect before interpreting smoothing')
            else:
                delta = {'model': name, 'condition': condition}
                delta.update({f'delta_{level}_{m}': row[f'{level}_{m}']-clean_values[f'{level}_{m}']
                              for level in ['frame','video'] for m in ['auc','real_fpr','fake_fnr']})
                delta.update(paired_bootstrap(clean, predictions, args.bootstrap_repeats, args.seed))
                p0, p1 = clean.probability_fake.to_numpy(), predictions.probability_fake.to_numpy()
                for label, label_name in [(0,'real'),(1,'fake')]:
                    chosen = predictions.class_index.astype(int).to_numpy()==label
                    delta[f'{label_name}_mean_probability_change'] = float(np.mean(p1[chosen]-p0[chosen]))
                delta['correct_to_wrong_frames'] = int(np.sum(((p0>=.5)==predictions.class_index.astype(int).to_numpy()) &
                                                              ((p1>=.5)!=predictions.class_index.astype(int).to_numpy())))
                changes.append(delta)
        del model
        if args.device == 'cuda':
            torch.cuda.empty_cache()
    pd.DataFrame(all_summaries).to_csv(args.output_root/'condition_summary.csv', index=False)
    pd.DataFrame(changes).to_csv(args.output_root/'paired_deltas.csv', index=False)
    baseline.json_write(args.output_root/'protocol.json', provenance)
    print(pd.DataFrame(all_summaries)[['model','condition','video_auc','video_real_fpr','video_fake_fnr']].to_string(index=False), flush=True)
    print('Results:', args.output_root, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['prepare','evaluate'])
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--manifest-root', type=Path, required=True)
    p.add_argument('--runs-root', type=Path, required=True)
    p.add_argument('--cache-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--protocol', type=Path, default=ROOT/'configs/celebdf_smoothing_evaluation_v1.json')
    p.add_argument('--models', nargs='+', choices=['xception','clip_b16'], default=['xception','clip_b16'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--bootstrap-repeats', type=int, default=1000)
    p.add_argument('--device', choices=['cpu','cuda'], default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    if args.bootstrap_repeats < 1:
        p.error('bootstrap-repeats must be positive')
    baseline.seed_all(args.seed)
    (prepare if args.action == 'prepare' else evaluate)(args)


if __name__ == '__main__':
    main()
