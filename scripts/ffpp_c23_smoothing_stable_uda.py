"""EMA UDA with smoothing-agreement selection and a within-batch random control."""
import argparse
import copy
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset

import ffpp_c23_smoothing_mean_teacher as engine
import ffpp_c23_smoothing_mean_teacher_soft as references

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'ffpp_c23_smoothing_stable_uda_v1'
STRATEGIES = ['stable_agreement', 'random_agreement_count']
MATCHING_LIMIT = ('Each run matches its OWN teacher\'s counterfactual agreement count per '
    'batch and predicted class. Independent EMA trajectories need not have identical '
    'cross-run retention totals or class counts.')


class TargetViews(Dataset):
    """Observed view first, then the other three strengths; all teacher flips match."""
    def __init__(self, plan, args, transform, cfg, extra_smoothing):
        if extra_smoothing:
            raise ValueError('No repeated smoothing in this protocol')
        self.plan, self.args, self.transform, self.cfg = plan, args, transform, cfg

    def __len__(self):
        return len(self.plan)

    def __getitem__(self, index):
        row = self.plan[index]
        with Image.open(self.args.target_root/row['crop_path']) as opened:
            original = opened.convert('RGB')
        images = {'normal': original}
        # The expensive native bilateral operation is performed/read only once per item.
        filtered = (self.args._target_cache.filtered(original, row['crop_path'])
                    if row['roi_status'] == 'detected' else None)
        for condition in engine.CONDITIONS[1:]:
            images[condition] = (engine.common.pixel_cache.blend(original, filtered,
                self.cfg['conditions'][condition]) if filtered is not None else original)
        order = [row['condition'], *[c for c in engine.CONDITIONS if c != row['condition']]]
        teacher = [images[c] for c in order]
        student = images[row['condition']]
        if row['flip']:
            teacher = [image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for image in teacher]
        else:
            student = student.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return torch.stack([self.transform(image) for image in teacher]), self.transform(student)


class MultiViewTeacher(nn.Module):
    """Bound teacher inference memory; parameter/buffer order matches the student."""
    def __init__(self, model, batch_size):
        super().__init__()
        self.model, self.batch_size = model, batch_size

    def forward(self, images):
        if images.ndim != 5 or images.shape[1] != 4:
            raise ValueError('Expected teacher input [batch, four views, channels, height, width]')
        flat = images.flatten(0, 1)
        outputs = torch.cat([self.model(part).float() for part in flat.split(self.batch_size)])
        if not torch.isfinite(outputs).all():
            raise RuntimeError('Nonfinite four-view teacher logits')
        return outputs.reshape(len(images), 4, 2)


def selection_masks(teacher_logits, threshold, strategy, seed):
    if strategy not in STRATEGIES or teacher_logits.ndim != 3 or teacher_logits.shape[1:] != (4, 2):
        raise ValueError('Unknown selection strategy or invalid four-view logits')
    probabilities = teacher_logits.detach().float().softmax(-1)
    confidence, labels = probabilities[:, 0].max(-1)
    candidates = confidence >= threshold
    agreement = probabilities.argmax(-1).eq(labels[:, None]).all(-1)
    stable = candidates & agreement
    selected = stable.clone()
    if strategy == 'random_agreement_count':
        selected.zero_()
        # Local CPU RNG never consumes the student's dropout/optimizer RNG stream.
        generator = torch.Generator().manual_seed((int(seed)+491009) % (2**63-1))
        for label in [0, 1]:
            indexes = torch.where(candidates & labels.eq(label))[0].cpu()
            count = int((stable & labels.eq(label)).sum())
            chosen = indexes[torch.randperm(len(indexes), generator=generator)[:count]]
            selected[chosen.to(selected.device)] = True
    return labels, confidence, candidates, stable, selected


class SelectionLog:
    def __init__(self, strategy):
        self.strategy, self.rows = strategy, []

    def loss(self, student_logits, teacher_logits, threshold):
        labels, confidence, candidates, stable, selected = selection_masks(
            teacher_logits, threshold, self.strategy, torch.initial_seed())
        per_image = nn.functional.cross_entropy(student_logits.float(), labels, reduction='none')
        # Match the old UDA denominator: rejected samples have zero target loss.
        loss = (per_image*selected).mean()
        row = {'batch': len(self.rows), 'seed': int(torch.initial_seed()), 'images': len(labels)}
        for name, mask in [('candidate', candidates), ('agreement', stable), ('selected', selected)]:
            for label, label_name in enumerate(['real', 'fake']):
                row[f'{name}_{label_name}'] = int((mask & labels.eq(label)).sum())
        row['selected_unstable'] = int((selected & ~stable).sum())
        row['selected_mask_sha256'] = engine.evaluation.key(selected.cpu().tolist())
        self.rows.append(row)
        stats = torch.stack([(labels == y).sum() for y in [0, 1]] +
                            [((labels == y) & selected).sum() for y in [0, 1]])
        return loss, stats, confidence.sum()

    def attach(self, path, state):
        if path.name != 'last.pt':
            return
        if not self.rows:
            raise RuntimeError('Missing selection diagnostics for completed epoch')
        summary = {key: sum(row[key] for row in self.rows) for key in self.rows[0]
                   if key not in ['batch', 'seed', 'selected_mask_sha256']}
        summary.update(batches=len(self.rows), matching_scope='within-run teacher, batch and predicted class')
        for label in ['real', 'fake']:
            if summary[f'selected_{label}'] != summary[f'agreement_{label}']:
                raise RuntimeError('Within-batch class quota mismatch')
        state['history'][-1]['selection_diagnostics'] = summary
        engine.freeze_csv(path.parent/f'selection_epoch{state["epoch"]:02d}.csv', pd.DataFrame(self.rows))
        print(f'Selection: candidates={summary["candidate_real"]}/{summary["candidate_fake"]}, '
              f'agreement={summary["agreement_real"]}/{summary["agreement_fake"]}, '
              f'used={summary["selected_real"]}/{summary["selected_fake"]} (real/fake predicted)', flush=True)
        self.rows.clear()


@contextmanager
def stable_protocol(args, strategy=None):
    if strategy is not None and strategy not in STRATEGIES:
        raise ValueError('Unknown strategy')
    old = (engine.PROTOCOL, engine.configuration, engine.TargetViews, engine.copy,
           engine.pseudo_loss, engine.base.common.save)
    log = SelectionLog(strategy)

    def configuration(a, name, selected, data, cfg, initial, initial_hash):
        if selected not in STRATEGIES:
            raise ValueError('Unknown configuration strategy')
        result = old[1](a, name, selected, data, cfg, initial, initial_hash)
        result.update(
            extension_code_sha256=engine.base.common.digest(Path(__file__)),
            reference_helper_sha256=engine.base.common.digest(Path(references.__file__)),
            hard_reference_manifest=a._hard_reference_identity,
            pseudo_label_kind='hard', pseudo_selection=selected,
            agreement_rule='all four teacher argmax labels equal; confidence >= threshold on observed view only',
            target_views='teacher all four strengths with matched flip; observed first; student observed view with complementary flip',
            target_data_assumption='pre-smoothing unlabelled target crops available; four training views are generated locally',
            random_matching_limit=MATCHING_LIMIT, teacher_batch_size=a.teacher_batch_size,
            loss='unchanged source CE; selected observed-view hard-pseudo CE divided by ALL target examples; weight0.5 with old ramp',
            target_weight=.5, inference_model='student; one input view; no agreement gate during inference')
        return result

    def save(path, state):
        log.attach(Path(path), state)
        return old[5](path, state)

    def teacher(model):
        return MultiViewTeacher(old[3].deepcopy(model), args.teacher_batch_size)

    engine.PROTOCOL, engine.configuration, engine.TargetViews = PROTOCOL, configuration, TargetViews
    engine.copy = SimpleNamespace(deepcopy=teacher)
    engine.pseudo_loss, engine.base.common.save = log.loss, save
    try:
        yield
    finally:
        (engine.PROTOCOL, engine.configuration, engine.TargetViews, engine.copy,
         engine.pseudo_loss, engine.base.common.save) = old


def train(args):
    references.verify_membership(args)
    for strategy in args.strategies:
        run_args = copy.copy(args)
        run_args.strategies, run_args.target_weight = [strategy], .5
        with stable_protocol(run_args, strategy):
            engine.train(run_args)


def evaluate(args):
    # Verify saved reference results before any new inference.
    records, sources = references.reference_records(args)
    _, args._hard_reference_identity = references.hard_manifest(args)
    for name in args.models:
        for strategy in args.strategies:
            path = args.runs_root/f'{name}_{strategy}_{PROTOCOL}_seed{args.seed}'/'config.json'
            config = json.loads(path.read_text())
            if (config.get('extension_code_sha256') != engine.base.common.digest(Path(__file__))
                    or config.get('reference_helper_sha256') != engine.base.common.digest(Path(references.__file__))
                    or config.get('hard_reference_manifest') != args._hard_reference_identity
                    or config.get('pseudo_selection') != strategy or config.get('target_weight') != .5
                    or config.get('teacher_batch_size') != args.teacher_batch_size):
                raise ValueError(f'Stable UDA checkpoint/config/code mismatch: {path}')
    run_args = copy.copy(args)
    run_args.reference_evaluations = args.hard_evaluations
    with stable_protocol(run_args):
        engine.evaluate(run_args)
    new = pd.read_csv(args.output_root/'comparison.csv')
    combined = pd.concat([pd.DataFrame(records), new[new.strategy.isin(args.strategies)]], ignore_index=True)
    expected = pd.MultiIndex.from_product([args.models,
        ['baseline', 'augmentation', 'mean_teacher', *args.strategies], engine.CONDITIONS])
    actual = pd.MultiIndex.from_frame(combined[['model', 'strategy', 'condition']])
    if actual.has_duplicates or set(actual) != set(expected):
        raise ValueError('Incomplete comparison')
    combined.to_csv(args.output_root/'comparison_with_reference.csv', index=False)
    means = combined.groupby(['model', 'strategy'], sort=False)[references.METRICS].mean().reset_index()
    means.to_csv(args.output_root/'condition_means_with_reference.csv', index=False)
    deltas = []
    for level, table in [('condition', combined), ('mean', means)]:
        keys = ['model', 'condition'] if level == 'condition' else ['model']
        for reference in ['baseline', 'augmentation', 'mean_teacher']:
            ref = table[table.strategy.eq(reference)][keys+references.METRICS]
            joined = table[table.strategy.isin(args.strategies)].merge(ref, on=keys, suffixes=('', '_ref'), validate='many_to_one')
            for row in joined.to_dict('records'):
                delta = {key: row[key] for key in [*keys, 'strategy']}
                delta.update(level=level, reference=reference)
                for metric in references.METRICS:
                    delta[f'delta_{metric}'] = row[metric]-row[metric+'_ref']
                delta['both_errors_lower'] = row['real_fpr'] < row['real_fpr_ref'] and row['fake_fnr'] < row['fake_fnr_ref']
                delta['auc_maintained_numeric'] = row['auc'] >= row['auc_ref']-1e-10
                deltas.append(delta)
    pd.DataFrame(deltas).to_csv(args.output_root/'deltas_vs_reference.csv', index=False)
    engine.base.common.json_write(args.output_root/'reused_reference_results.json',
        {'sources': sources, 'matching_limit': MATCHING_LIMIT,
         'cdf_test_for_model_selection': False, 'threshold': .5, 'old_models_retrained': False})
    print('Saved:', args.output_root/'condition_means_with_reference.csv', flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['train', 'evaluate'])
    for field in ['source-root', 'target-root', 'baseline-runs-root', 'runs-root', 'manifest-root',
                  'validation-cache', 'output-root', 'hard-manifest-root', 'hard-evaluations']:
        p.add_argument('--'+field, type=Path, required=True)
    p.add_argument('--source-validation', type=Path)
    for field, default in [('train-cache', '/content/smoothing_pivot_bilateral_cache_v1'),
            ('target-train-cache', '/content/smoothing_mean_teacher_target_cache_v1'),
            ('cache-root', '/content/celebdf_official_smoothing_cache_v1'),
            ('evaluation-pixel-cache', '/content/smoothing_ranking_cdf_bilateral_cache_v1')]:
        p.add_argument('--'+field, type=Path, default=Path(default))
    p.add_argument('--protocol', type=Path, default=ROOT/'configs/ffpp_c23_celebdf_smoothing_v1.json')
    p.add_argument('--models', nargs='+', choices=list(engine.base.PREPROCESS), default=['xception'])
    p.add_argument('--strategies', nargs='+', choices=STRATEGIES, default=STRATEGIES)
    for field, default in [('seed', 42), ('epochs', 10), ('patience', 3), ('source-images', 6000),
            ('val-images', 1200), ('target-images', 5000), ('batch-size', 16), ('teacher-batch-size', 32), ('workers', 2)]:
        p.add_argument('--'+field, type=int, default=default)
    for field, default in [('encoder-lr', 1e-5), ('head-lr', 1e-5), ('weight-decay', 1e-4),
            ('ema-decay', .999), ('pseudo-threshold', .9), ('ramp-epochs', 2.)]:
        p.add_argument('--'+field, type=float, default=default)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = p.parse_args()
    numbers = [args.encoder_lr, args.head_lr, args.weight_decay, args.ema_decay, args.pseudo_threshold, args.ramp_epochs]
    if (min(args.epochs, args.patience, args.source_images, args.val_images, args.target_images,
            args.batch_size, args.teacher_batch_size) < 1 or args.source_images % 8 or args.val_images % 8
            or args.workers < 0 or not np.isfinite(numbers).all()
            or min(args.encoder_lr, args.head_lr, args.ramp_epochs) <= 0 or args.weight_decay < 0
            or not 0 <= args.ema_decay < 1 or not .5 <= args.pseudo_threshold <= 1
            or len(set(args.models)) != len(args.models) or len(set(args.strategies)) != len(args.strategies)):
        p.error('Invalid counts, duplicate choices, or optimization settings')
    args.target_weight = .5
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime')
    return args


if __name__ == '__main__':
    args = parse_args()
    (train if args.action == 'train' else evaluate)(args)
