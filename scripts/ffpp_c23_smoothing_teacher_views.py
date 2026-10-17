"""Separate target-domain adaptation from teacher/student smoothing exposure."""
import argparse
import copy
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset

import ffpp_c23_smoothing_mean_teacher as engine
import ffpp_c23_smoothing_mean_teacher_soft as references

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'ffpp_c23_smoothing_teacher_views_v1'
STRATEGIES = ['normal_only', 'clean_to_mixed']
REFERENCE_STRATEGIES = ['baseline', 'augmentation', 'mean_teacher', 'mean_teacher_smoothing']
METRICS = references.METRICS


class TargetViews(Dataset):
    """Teacher always sees the pre-smoothing crop; no target labels are returned."""
    def __init__(self, plan, args, transform, cfg, extra_smoothing):
        if extra_smoothing:
            raise ValueError('This protocol never applies smoothing twice')
        self.plan, self.args, self.transform, self.cfg = plan, args, transform, cfg

    def __len__(self):
        return len(self.plan)

    def __getitem__(self, index):
        row = self.plan[index]
        with Image.open(self.args.target_root/row['crop_path']) as opened:
            teacher = opened.convert('RGB')
        student = teacher
        if row['roi_status'] == 'detected' and row['condition'] != 'normal':
            student = self.args._target_cache.smooth(teacher, row['crop_path'],
                self.cfg['conditions'][row['condition']])
        if row['flip']:
            teacher = teacher.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if not row['flip']:
            student = student.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return self.transform(teacher), self.transform(student)


@contextmanager
def view_protocol(strategy=None):
    """Reuse the unchanged EMA engine, optimization, preprocessing and pixel caches."""
    if strategy is not None and strategy not in STRATEGIES:
        raise ValueError(f'Unknown target view strategy: {strategy}')
    old = (engine.PROTOCOL, engine.configuration, engine.TargetViews, engine.target_plan)

    def target_plan(target, n, seed):
        if strategy is None:
            raise RuntimeError('Training requires an explicit target view strategy')
        plan = old[3](target, n, seed)
        for row in plan:
            # Draw the complete legacy plan first: paths and flips match old UDA exactly.
            row['scheduled_condition'] = row['condition']
            row['teacher_condition'] = 'normal'
            if strategy == 'normal_only':
                row['condition'] = 'normal'
        return plan

    def configuration(args, name, selected, data, cfg, initial, initial_hash):
        if selected != strategy:
            raise ValueError('Train one named target view strategy at a time')
        result = old[1](args, name, selected, data, cfg, initial, initial_hash)
        result.update(
            target_views=('teacher pre-smoothing crop; student same crop; complementary horizontal flip'
                if selected == 'normal_only' else
                'teacher pre-smoothing crop; student normal/weak/medium/strong uniformly; complementary horizontal flip'),
            target_data_assumption='unlabelled pre-smoothing target crops are available during adaptation',
            teacher_condition='normal',
            student_condition='normal' if selected == 'normal_only' else 'uniform_four_conditions',
            target_plan_condition_field='actual student condition; scheduled_condition retains legacy random draw',
            pseudo_label_kind='hard', target_weight=.5,
            extension_code_sha256=engine.base.common.digest(Path(__file__)),
            reference_helper_sha256=engine.base.common.digest(Path(references.__file__)),
            hard_reference_manifest=args._hard_reference_identity)
        return result

    engine.PROTOCOL, engine.configuration = PROTOCOL, configuration
    engine.TargetViews, engine.target_plan = TargetViews, target_plan
    try:
        yield engine
    finally:
        engine.PROTOCOL, engine.configuration, engine.TargetViews, engine.target_plan = old


def train(args):
    references.verify_membership(args)
    for strategy in args.strategies:
        run_args = copy.copy(args)
        run_args.strategies = [strategy]
        # Only the view plan differs. Keep distinct frozen plan files for both conditions.
        run_args.manifest_root = args.manifest_root/strategy
        run_args.target_weight = .5
        print(f'New target views: {strategy}; teacher always normal; original FF++ baseline initialization', flush=True)
        with view_protocol(strategy):
            engine.train(run_args)


def reference_records(args):
    ea = argparse.Namespace(data_root=args.target_root,
        manifest=args.target_root/'manifests/celebdf_v2_roi16_test_v1.csv',
        protocol=args.protocol, cache_root=args.cache_root, output_root=args.output_root, workers=args.workers)
    frame, _, provenance = engine.evaluation.inputs(ea)
    records, sources, mixtures = [], [], []
    labels = frame.class_index.astype(int).to_numpy()
    for model in args.models:
        initial, initial_hash = engine.evaluation.checkpoint(
            argparse.Namespace(runs_root=args.baseline_runs_root, seed=args.seed), model)
        del initial
        for strategy in REFERENCE_STRATEGIES:
            banks = []
            for condition in engine.CONDITIONS:
                metrics, report, checkpoint_hash = references.prior_predictions(
                    args, model, strategy, condition, frame, provenance, initial_hash)
                records.append({'model':model, 'strategy':strategy, 'condition':condition,
                    **{key:metrics[key] for key in METRICS}})
                prediction = report.with_suffix('.csv')
                banks.append(pd.read_csv(prediction, usecols=['raw_margin']).raw_margin.to_numpy())
                sources.append({'model':model, 'strategy':strategy, 'condition':condition,
                    'report':str(report), 'report_sha256':engine.base.common.digest(report),
                    'prediction_sha256':engine.base.common.digest(prediction),
                    'checkpoint_sha256':checkpoint_hash})
            mixtures.append({'model':model, 'strategy':strategy,
                'pooled_auc':float(roc_auc_score(np.tile(labels, len(banks)), np.concatenate(banks)))})
    return records, sources, mixtures


def separation_summary(combined, mixtures):
    records = []
    pooled = mixtures.set_index(['model','strategy']).pooled_auc
    for (model, strategy), group in combined.groupby(['model','strategy'], sort=False):
        conditions = group.set_index('condition')
        normal, strong = conditions.loc['normal'], conditions.loc['smooth_strong']
        row = {'model':model, 'strategy':strategy,
               'pooled_auc':float(pooled.loc[(model,strategy)])}
        for metric in METRICS:
            row['normal_'+metric] = float(normal[metric])
            row['mean_'+metric] = float(group[metric].mean())
            row['strong_minus_normal_'+metric] = float(strong[metric]-normal[metric])
        records.append(row)
    table = pd.DataFrame(records)
    for strategy in ['augmentation','mean_teacher']:
        ref = table[table.strategy.eq(strategy)].set_index('model')
        for metric in ['normal_balanced_error_rate','normal_auc','mean_balanced_error_rate','mean_auc','pooled_auc']:
            table[f'delta_{metric}_vs_{strategy}'] = table[metric]-table.model.map(ref[metric])
    return table


def evaluate(args):
    # Validate existing reference files before spending time on new GPU inference.
    records, sources, old_mixtures = reference_records(args)
    _, reference_identity = references.hard_manifest(args)
    for model in args.models:
        for strategy in args.strategies:
            run = args.runs_root/f'{model}_{strategy}_{PROTOCOL}_seed{args.seed}'
            cfg = json.loads((run/'config.json').read_text())
            if (cfg.get('extension_code_sha256') != engine.base.common.digest(Path(__file__))
                    or cfg.get('reference_helper_sha256') != engine.base.common.digest(Path(references.__file__))
                    or cfg.get('teacher_condition') != 'normal' or cfg.get('pseudo_label_kind') != 'hard'
                    or cfg.get('student_condition') != ('normal' if strategy == 'normal_only' else 'uniform_four_conditions')
                    or cfg.get('hard_reference_manifest') != reference_identity or cfg.get('target_weight') != .5):
                raise ValueError(f'Teacher view checkpoint/config/code mismatch: {run}')
    run_args = copy.copy(args)
    run_args.reference_evaluations = args.hard_evaluations
    with view_protocol():
        engine.evaluate(run_args)
    new = pd.read_csv(args.output_root/'comparison.csv')
    new = new[new.strategy.isin(args.strategies)]
    combined = pd.concat([pd.DataFrame(records), new], ignore_index=True)
    expected = pd.MultiIndex.from_product([args.models, [*REFERENCE_STRATEGIES,*args.strategies],
        engine.CONDITIONS], names=['model','strategy','condition'])
    actual = pd.MultiIndex.from_frame(combined[['model','strategy','condition']])
    if actual.has_duplicates or set(actual) != set(expected):
        raise ValueError('Incomplete or duplicate teacher view comparison rows')
    combined.to_csv(args.output_root/'comparison_with_reference.csv', index=False)
    combined.groupby(['model','strategy'], sort=False)[METRICS].mean().reset_index().to_csv(
        args.output_root/'condition_means_with_reference.csv', index=False)
    new_mixtures = pd.read_csv(args.output_root/'mixture_summary.csv')
    mixtures = pd.concat([pd.DataFrame(old_mixtures),
        new_mixtures[new_mixtures.strategy.isin(args.strategies)][['model','strategy','pooled_auc']]], ignore_index=True)
    summary = separation_summary(combined, mixtures)
    summary.to_csv(args.output_root/'domain_and_smoothing.csv', index=False)
    engine.base.common.json_write(args.output_root/'reused_reference_results.json', {
        'reference_protocol':references.HARD_PROTOCOL, 'sources':sources,
        'old_models_retrained':False, 'target_labels_used':False,
        'target_assumption':'pre-smoothing unlabelled target crops available',
        'reporting':'Normal, four-condition mean, pooled AUC, signed Strong-minus-Normal gaps; no target threshold tuning'})
    print('Saved comparison:', args.output_root/'condition_means_with_reference.csv', flush=True)
    print('Saved domain/smoothing summary:', args.output_root/'domain_and_smoothing.csv', flush=True)


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
    p.add_argument('--models', nargs='+', choices=list(engine.base.PREPROCESS), default=['dinov2_b14'])
    p.add_argument('--strategies', nargs='+', choices=STRATEGIES, default=STRATEGIES)
    for field, default in [('seed',42),('epochs',10),('patience',3),('source-images',6000),('val-images',1200),
                           ('target-images',5000),('batch-size',16),('workers',2)]:
        p.add_argument('--'+field, type=int, default=default)
    for field, default in [('encoder-lr',1e-5),('head-lr',1e-5),('weight-decay',1e-4),
                           ('ema-decay',.999),('pseudo-threshold',.9),('ramp-epochs',2.)]:
        p.add_argument('--'+field, type=float, default=default)
    p.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    args = p.parse_args()
    args.target_weight = .5
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
