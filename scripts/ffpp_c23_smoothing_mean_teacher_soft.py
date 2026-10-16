"""Soft-label CDF ablation using the unchanged, completed Mean Teacher engine."""
import argparse
import copy
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

import ffpp_c23_smoothing_mean_teacher as engine

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'ffpp_c23_smoothing_mean_teacher_soft_v1'
HARD_PROTOCOL = engine.PROTOCOL
WEIGHTS = {'soft_w05': .5, 'soft_w025': .25}
METRICS = ['real_fpr', 'fake_fnr', 'balanced_error_rate', 'auc', 'auc_probability']


def soft_pseudo_loss(student_logits, teacher_logits, threshold):
    """T=1 soft CE, confidence mask and ALL-target denominator match hard UDA."""
    probabilities = teacher_logits.detach().float().softmax(1)
    confidence, labels = probabilities.max(1)
    selected = confidence >= threshold
    per_image = -(probabilities * F.log_softmax(student_logits.float(), dim=1)).sum(1)
    loss = (per_image * selected).mean()
    stats = torch.stack([(labels == y).sum() for y in [0, 1]] +
                        [((labels == y) & selected).sum() for y in [0, 1]])
    return loss, stats, confidence.sum()


@contextmanager
def soft_protocol():
    """Temporarily adapt public hooks; preserve the original engine and its hashes.

    Original code files, checkpoints and native-pixel cache identities stay intact.
    The extension's own hash is recorded separately in every new checkpoint config.
    Dataset classes keep their importable original module for DataLoader workers.
    """
    old_protocol, old_configuration, old_loss = engine.PROTOCOL, engine.configuration, engine.pseudo_loss

    def configuration(args, name, strategy, data, cfg, initial, initial_hash):
        result = old_configuration(args, name, strategy, data, cfg, initial, initial_hash)
        result.update(
            loss='source clean+random smoothing CE; CDF masked soft-target CE divided by ALL target examples; linear ramp',
            target_views='teacher observed normal/weak/medium/strong uniformly; student complementary horizontal flip; no extra smoothing',
            pseudo_label_kind='soft', pseudo_temperature=1., target_weight=WEIGHTS[strategy],
            extension_code_sha256=engine.base.common.digest(Path(__file__)),
            hard_reference_manifest=args._hard_reference_identity)
        return result

    engine.PROTOCOL, engine.configuration, engine.pseudo_loss = PROTOCOL, configuration, soft_pseudo_loss
    try:
        yield engine
    finally:
        engine.PROTOCOL, engine.configuration, engine.pseudo_loss = old_protocol, old_configuration, old_loss


def hard_manifest(args):
    root = args.hard_manifest_root
    candidates = []
    for path in sorted(root.rglob('protocol.json')):
        folder = path.parent
        if not all((folder/name).is_file() for name in ['source_validation.csv', 'target_unlabelled.csv']):
            continue
        protocol = json.loads(path.read_text())
        if protocol.get('protocol') == HARD_PROTOCOL and protocol.get('seed') == args.seed:
            candidates.append(folder)
    if not candidates:
        raise FileNotFoundError(f'Previous hard UDA manifests missing: {root}')
    signatures = [{name: engine.base.common.digest(folder/name) for name in
                   ['protocol.json', 'source_validation.csv', 'target_unlabelled.csv']} for folder in candidates]
    if any(signature != signatures[0] for signature in signatures):
        raise ValueError('Previous hard UDA groups have different data; choose one --hard-manifest-root')
    return candidates[0], signatures[0]


def verify_membership(args):
    folder, identity = hard_manifest(args)
    previous = json.loads((folder/'protocol.json').read_text())
    _, val, source_data = engine.common.shared.source_inputs(args)
    _, other_val, target, _, target_data, _ = engine.adaptation.dataset_inputs(args)
    target = target[['crop_path', 'video_id', 'roi_status']]
    if (previous['data'] != {**source_data, **target_data}
            or previous['source_images'] != args.source_images
            or previous['target_images'] != args.target_images
            or val.crop_path.tolist() != other_val.crop_path.tolist()):
        raise ValueError('Soft/hard dataset or budget mismatch')
    for name, frame in [('source_validation.csv', val), ('target_unlabelled.csv', target)]:
        saved = pd.read_csv(folder/name, dtype=str, keep_default_na=False)
        if list(saved) != list(frame) or not saved.equals(frame.astype(str)):
            raise ValueError(f'Soft/hard frozen membership differs: {name}')
    args._hard_reference_identity = identity
    print(f'Matched previous hard UDA: validation={len(val)}, target={len(target)}; same seed and membership', flush=True)


def train(args):
    verify_membership(args)
    with soft_protocol():
        for strategy in args.strategies:
            run_args = copy.copy(args)
            run_args.strategies = [strategy]
            run_args.target_weight = WEIGHTS[strategy]
            engine.train(run_args)


def prior_predictions(args, model, strategy, condition, frame, provenance, baseline_hash):
    """Read old predictions only; never load or infer the old adapted models."""
    matches = []
    for report in sorted(args.hard_evaluations.rglob(f'{model}_{strategy}_{condition}.json')):
        prediction, protocol = report.with_suffix('.csv'), report.parent/'evaluation_protocol.json'
        if not prediction.is_file() or not protocol.is_file():
            continue
        saved_protocol = json.loads(protocol.read_text())
        if any(saved_protocol.get(key) != value for key, value in provenance.items()):
            continue
        values = json.loads(report.read_text())
        if (values.get('precision') != 'FP32' or values.get('threshold') != .5
                or values.get('unit') != 'frame'
                or values.get('manifest_sha256') != provenance['manifest_sha256']
                or not values.get('checkpoint_sha256')
                or strategy == 'baseline' and values['checkpoint_sha256'] != baseline_hash):
            continue
        saved = pd.read_csv(prediction, dtype={'crop_path':str, 'video_id':str, 'class_index':str}, keep_default_na=False)
        columns = ['crop_path', 'video_id', 'class_index', 'roi_status']
        if (not set(columns+['raw_margin', 'probability_fake']).issubset(saved)
                or not saved[columns].equals(frame[columns].reset_index(drop=True))
                or not np.isfinite(saved.raw_margin).all()
                or not np.isfinite(saved.probability_fake).all()
                or not saved.probability_fake.between(0, 1).all()):
            continue
        metrics = engine.raw_metrics(frame, saved.probability_fake.to_numpy(), saved.raw_margin.to_numpy())
        if any(not np.isclose(values.get(key, np.nan), metrics[key], atol=1e-10, rtol=0)
               for key in METRICS):
            raise ValueError(f'Previous metrics/predictions differ: {report}')
        matches.append((metrics, report, values['checkpoint_sha256']))
    if not matches:
        raise FileNotFoundError(f'Matching saved {model}/{strategy}/{condition} missing under {args.hard_evaluations}')
    if any(item[2] != matches[0][2] or item[0] != matches[0][0] for item in matches[1:]):
        raise ValueError(f'Ambiguous previous results for {model}/{strategy}/{condition}; narrow --hard-evaluations')
    return matches[0]


def reference_records(args):
    ea = argparse.Namespace(data_root=args.target_root,
        manifest=args.target_root/'manifests/celebdf_v2_roi16_test_v1.csv',
        protocol=args.protocol, cache_root=args.cache_root, output_root=args.output_root, workers=args.workers)
    frame, _, provenance = engine.evaluation.inputs(ea)
    records, sources = [], []
    for model in args.models:
        baseline, baseline_hash = engine.evaluation.checkpoint(
            argparse.Namespace(runs_root=args.baseline_runs_root, seed=args.seed), model)
        del baseline
        for strategy in ['baseline', 'augmentation', 'mean_teacher']:
            for condition in engine.CONDITIONS:
                metrics, report, checkpoint_hash = prior_predictions(
                    args, model, strategy, condition, frame, provenance, baseline_hash)
                records.append({'model':model, 'strategy':strategy, 'condition':condition,
                                **{key:metrics[key] for key in METRICS}})
                sources.append({'model':model, 'strategy':strategy, 'condition':condition,
                    'report':str(report), 'report_sha256':engine.base.common.digest(report),
                    'prediction_sha256':engine.base.common.digest(report.with_suffix('.csv')),
                    'checkpoint_sha256':checkpoint_hash})
    return records, sources


def evaluate(args):
    # Fail on missing old results before spending time on new model inference.
    records, sources = reference_records(args)
    extension_hash = engine.base.common.digest(Path(__file__))
    for model in args.models:
        for strategy in args.strategies:
            run = args.runs_root/f'{model}_{strategy}_{PROTOCOL}_seed{args.seed}'
            config = json.loads((run/'config.json').read_text())
            if (config.get('extension_code_sha256') != extension_hash
                    or config.get('pseudo_label_kind') != 'soft'
                    or config.get('pseudo_temperature') != 1.
                    or config.get('target_weight') != WEIGHTS[strategy]):
                raise ValueError(f'Soft checkpoint/config/code mismatch: {run}')
    run_args = copy.copy(args)
    run_args.reference_evaluations = args.hard_evaluations
    with soft_protocol():
        engine.evaluate(run_args)
    new = pd.read_csv(args.output_root/'comparison.csv')
    new = new[new.strategy.isin(args.strategies)]
    combined = pd.concat([pd.DataFrame(records), new], ignore_index=True)
    expected = pd.MultiIndex.from_product([args.models, ['baseline','augmentation','mean_teacher',*args.strategies],
                                           engine.CONDITIONS], names=['model','strategy','condition'])
    actual = pd.MultiIndex.from_frame(combined[['model','strategy','condition']])
    if actual.has_duplicates or set(actual) != set(expected):
        raise ValueError('Incomplete or duplicate comparison rows')
    combined.to_csv(args.output_root/'comparison_with_hard.csv', index=False)
    means = combined.groupby(['model','strategy'], sort=False)[METRICS].mean().reset_index()
    means.to_csv(args.output_root/'condition_means_with_hard.csv', index=False)
    hard = means[means.strategy.eq('mean_teacher')].set_index('model')
    deltas = means[means.strategy.isin(args.strategies)].copy()
    for metric in METRICS:
        deltas[f'delta_{metric}_vs_hard'] = deltas[metric]-deltas.model.map(hard[metric])
    deltas.to_csv(args.output_root/'deltas_vs_hard.csv', index=False)
    engine.base.common.json_write(args.output_root/'reused_hard_results.json',
        {'reference_protocol':HARD_PROTOCOL, 'sources':sources, 'old_models_retrained':False})
    print('Saved hard/soft comparison:', args.output_root/'condition_means_with_hard.csv', flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['train','evaluate'])
    for field in ['source-root','target-root','baseline-runs-root','runs-root','manifest-root','validation-cache','output-root',
                  'hard-manifest-root','hard-evaluations']:
        p.add_argument('--'+field, type=Path, required=True)
    p.add_argument('--source-validation', type=Path)
    p.add_argument('--train-cache', type=Path, default=Path('/content/smoothing_pivot_bilateral_cache_v1'))
    p.add_argument('--target-train-cache', type=Path, default=Path('/content/smoothing_mean_teacher_target_cache_v1'))
    p.add_argument('--cache-root', type=Path, default=Path('/content/celebdf_official_smoothing_cache_v1'))
    p.add_argument('--evaluation-pixel-cache', type=Path, default=Path('/content/smoothing_ranking_cdf_bilateral_cache_v1'))
    p.add_argument('--protocol', type=Path, default=ROOT/'configs/ffpp_c23_celebdf_smoothing_v1.json')
    p.add_argument('--models', nargs='+', choices=list(engine.base.PREPROCESS), default=['clip_b16','xception'])
    p.add_argument('--strategies', nargs='+', choices=list(WEIGHTS), default=list(WEIGHTS))
    for field, default in [('seed',42),('epochs',10),('patience',3),('source-images',6000),('val-images',1200),
                           ('target-images',5000),('batch-size',16),('workers',2)]:
        p.add_argument('--'+field, type=int, default=default)
    for field, default in [('encoder-lr',1e-5),('head-lr',1e-5),('weight-decay',1e-4),
                           ('ema-decay',.999),('pseudo-threshold',.9),('ramp-epochs',2.)]:
        p.add_argument('--'+field, type=float, default=default)
    p.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    args = p.parse_args()
    if (min(args.epochs,args.patience,args.source_images,args.val_images,args.target_images,args.batch_size) < 1
            or args.source_images%8 or args.val_images%8 or args.workers<0
            or not 0 <= args.ema_decay < 1 or not .5 <= args.pseudo_threshold <= 1
            or args.ramp_epochs<=0 or min(args.encoder_lr,args.head_lr)<=0 or args.weight_decay<0
            or not np.isfinite([args.ema_decay,args.pseudo_threshold,args.ramp_epochs,
                               args.encoder_lr,args.head_lr,args.weight_decay]).all()
            or len(set(args.models)) != len(args.models) or len(set(args.strategies)) != len(args.strategies)):
        p.error('Invalid counts, duplicate conditions, or optimization settings')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime')
    return args


if __name__ == '__main__':
    args = parse_args()
    (train if args.action == 'train' else evaluate)(args)
