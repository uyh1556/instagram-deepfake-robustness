"""Paired saved-probability analysis only; no torch, models, images or inference."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS = ['xception', 'clip_b16', 'dinov2_b14']
CONDITIONS = {'smooth_weak': 25, 'smooth_medium': 50, 'smooth_strong': 75}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prefix(args, model, condition):
    middle = '' if args.strategy == 'baseline' else args.strategy+'_'
    return args.prediction_root/f'{model}_{middle}{condition}'


def load(args, model, condition):
    path = prefix(args, model, condition).with_suffix('.csv')
    frame = pd.read_csv(path, dtype={'crop_path': str, 'video_id': str, 'roi_status': str})
    required = {'crop_path', 'video_id', 'class_index', 'roi_status', 'probability_fake'}
    if not required.issubset(frame.columns) or frame.crop_path.duplicated().any():
        raise ValueError(f'Incomplete/duplicate prediction rows: {path}')
    if args.expected_images and len(frame) != args.expected_images:
        raise ValueError(f'Expected {args.expected_images} images, found {len(frame)}: {path}')
    if (not frame.class_index.isin([0, 1]).all() or not np.isfinite(frame.probability_fake).all()
            or not frame.probability_fake.between(0, 1).all()):
        raise ValueError(f'Invalid label/probability: {path}')
    report = json.loads(prefix(args, model, condition).with_suffix('.json').read_text())
    if report.get('unit', 'frame') != 'frame':
        raise ValueError('Image/frame predictions required')
    return frame, report


def paired(clean, changed):
    if set(clean.crop_path) != set(changed.crop_path):
        raise ValueError('Original/smoothing image membership differs')
    changed = changed.set_index('crop_path').loc[clean.crop_path].reset_index()
    for field in ['class_index', 'video_id', 'roi_status']:
        if clean[field].tolist() != changed[field].tolist():
            raise ValueError(f'Paired metadata differ: {field}')
    result = clean[['crop_path', 'video_id', 'class_index', 'roi_status']].copy()
    p0, p1 = clean.probability_fake.to_numpy(), changed.probability_fake.to_numpy()
    valid = (p0 > 0) & (p0 < 1) & (p1 > 0) & (p1 < 1)
    m0, m1 = np.full(len(p0), np.nan), np.full(len(p0), np.nan)
    # Do NOT clip endpoint probabilities and pretend the result is an exact logit.
    m0[valid] = np.log(p0[valid])-np.log1p(-p0[valid])
    m1[valid] = np.log(p1[valid])-np.log1p(-p1[valid])
    result['p_original'], result['p_smoothing'] = p0, p1
    result['delta_probability'] = p1-p0
    result['margin_original'], result['margin_smoothing'] = m0, m1
    result['delta_margin'] = m1-m0
    result['finite_margin_pair'] = valid
    return result


def summarize(frame, model, condition, scope, label):
    group = frame[frame.class_index.eq(label)]
    if scope == 'detected':
        group = group[group.roi_status.eq('detected')]
    elif scope == 'fallback':
        group = group[group.roi_status.ne('detected')]
    n = len(group)
    delta = group.delta_margin.dropna().to_numpy()
    dp = group.delta_probability.to_numpy()
    row = {'model': model, 'condition': condition, 'strength_percent': CONDITIONS[condition],
           'scope': scope, 'class': 'real' if label == 0 else 'fake', 'images': n,
           'finite_margin_pairs': len(delta), 'saturated_pairs': n-len(delta)}
    row.update(saturation_rate=(n-len(delta))/n if n else None,
               probability_up_rate=float(np.mean(dp > 0)) if n else None,
               probability_down_rate=float(np.mean(dp < 0)) if n else None,
               probability_tie_rate=float(np.mean(dp == 0)) if n else None,
               delta_probability_median=float(np.median(dp)) if n else None,
               delta_margin_mean=float(np.mean(delta)) if len(delta) else None,
               delta_margin_median=float(np.median(delta)) if len(delta) else None,
               delta_margin_q25=float(np.quantile(delta, .25)) if len(delta) else None,
               delta_margin_q75=float(np.quantile(delta, .75)) if len(delta) else None,
               margin_up_rate_valid=float(np.mean(delta > 0)) if len(delta) else None,
               prediction_real_to_fake_crossing_rate=float(np.mean((group.p_original < .5) & (group.p_smoothing >= .5))) if n else None,
               prediction_fake_to_real_crossing_rate=float(np.mean((group.p_original >= .5) & (group.p_smoothing < .5))) if n else None)
    return row


def figures(pairs, args):
    for model in args.models:
        fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)
        for col, condition in enumerate(CONDITIONS):
            for label in [0, 1]:
                ax = axes[label, col]
                frame = pairs[(model, condition)]
                values = frame[frame.class_index.eq(label) & frame.roi_status.eq('detected')].delta_margin.dropna().to_numpy()
                if len(values):
                    ax.hist(values, bins=60, color='#557b9c', alpha=.8)
                    ax.axvline(np.median(values), color='#ba4a40', label=f'Median {np.median(values):.3f}')
                    ax.legend(fontsize=8)
                else:
                    ax.text(.5, .5, 'No finite margin pairs', ha='center', transform=ax.transAxes)
                ax.axvline(0, color='black', linestyle='--', linewidth=1)
                ax.set_title(f'{"Real" if label == 0 else "Fake"} / {CONDITIONS[condition]}% / n={len(values)}')
                ax.set_xlabel('Smoothing margin - original margin')
                ax.set_ylabel('Images')
        fig.suptitle(f'{model}: detected faces only; endpoint probabilities excluded')
        fig.savefig(args.output_root/f'{model}_margin_shift.png', dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(len(args.models), 2, figsize=(10, 3*len(args.models)), squeeze=False, constrained_layout=True)
    for i, model in enumerate(args.models):
        frame = pairs[(model, 'smooth_strong')]
        for label in [0, 1]:
            group = frame[frame.class_index.eq(label) & frame.roi_status.eq('detected')]
            ax = axes[i, label]
            if len(group):
                for column, title, color in [('p_original', 'Original', '#557b9c'), ('p_smoothing', 'Smoothing75%', '#ba4a40')]:
                    ax.hist(group[column], bins=np.linspace(0, 1, 41), histtype='step', density=True, label=title, color=color, linewidth=1.5)
                ax.legend(fontsize=8)
            ax.axvline(.5, color='black', linestyle='--', linewidth=1)
            ax.set_title(f'{model} / {"Real" if label == 0 else "Fake"} / detected only')
            ax.set_xlabel('Probability fake'); ax.set_ylabel('Density')
    fig.savefig(args.output_root/'probability_distributions.png', dpi=150)
    plt.close(fig)


def analyze(args):
    inputs = {}
    for model in args.models:
        for condition in ['normal', *CONDITIONS]:
            for suffix in ['.csv', '.json']:
                path = prefix(args, model, condition).with_suffix(suffix)
                if not path.is_file():
                    raise FileNotFoundError(path)
                inputs[str(path)] = digest(path)
    provenance = {'input_sha256': inputs, 'strategy': args.strategy, 'models': args.models,
                  'analysis_code_sha256': digest(Path(__file__)),
                  'numpy': np.__version__, 'pandas': pd.__version__, 'matplotlib': matplotlib.__version__,
                  'margin': 'log(p)-log1p(-p), recoverable binary softmax margin, not separately stored raw logits',
                  'endpoint_policy': 'exact p=0/1 excluded from margin statistics; no clipping; probability statistics retain all pairs',
                  'inference': 'none; paired saved CSV only', 'uncertainty': 'descriptive frame statistics; no independence/significance claim'}
    args.output_root.mkdir(parents=True, exist_ok=True)
    config_path = args.output_root/'analysis_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != provenance:
        raise ValueError('Analysis inputs/code differ; preserve old results and use a new output root')
    pairs, summary = {}, []
    for model in args.models:
        clean, clean_report = load(args, model, 'normal')
        for condition in CONDITIONS:
            changed, report = load(args, model, condition)
            for field in ['checkpoint_sha256', 'manifest_sha256', 'identity']:
                if clean_report.get(field) != report.get(field):
                    raise ValueError(f'Different original/smoothing provenance: {model}/{field}')
            frame = paired(clean, changed)
            pairs[(model, condition)] = frame
            frame.to_csv(args.output_root/f'{model}_{condition}_paired_shift.csv', index=False)
            for scope in ['all', 'detected', 'fallback']:
                for label in [0, 1]:
                    summary.append(summarize(frame, model, condition, scope, label))
            print(f'Analyzed: {model}/{condition}, pairs={len(frame)}, saturated={int((~frame.finite_margin_pair).sum())}', flush=True)
    pd.DataFrame(summary).to_csv(args.output_root/'logit_shift_summary.csv', index=False)
    figures(pairs, args)
    config_path.write_text(json.dumps(provenance, indent=2)+'\n')
    print('Saved:', args.output_root, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prediction-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--models', nargs='+', choices=MODELS, default=MODELS)
    p.add_argument('--strategy', choices=['baseline', 'source_ft', 'augmentation', 'uda_normal', 'uda_smoothing', 'sda_normal', 'sda_smoothing'], default='baseline')
    p.add_argument('--expected-images', type=int, default=8288)
    args = p.parse_args()
    analyze(args)


if __name__ == '__main__':
    main()
