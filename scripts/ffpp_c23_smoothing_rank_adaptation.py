"""Matched CE+Rank with hard Mean Teacher or labelled non-test CDF supervision.

The completed source-only CE+Rank is reused. SDA is a diagnostic reference,
not UDA or a guaranteed upper bound. Test labels never supervise training.
"""
import argparse
import copy
import json
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import ffpp_c23_smoothing_delta_transfer as rank

engine, common, base = rank.engine, rank.common, rank.base
adaptation, evaluation, reference = rank.adaptation, rank.evaluation, rank.reference
ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'ffpp_c23_smoothing_rank_adaptation_v1'
STRATEGIES = ['rank_ts', 'rank_sda']
CONDITIONS = engine.CONDITIONS


def code_hashes():
    return {**rank.code_hashes(), Path(__file__).name: base.common.digest(Path(__file__))}


def run_name(model, strategy, seed):
    return f'{model}_{strategy}_{PROTOCOL}_seed{seed}'


@contextmanager
def protocol_context():
    previous = engine.PROTOCOL, engine.code_hashes
    engine.PROTOCOL, engine.code_hashes = PROTOCOL, code_hashes
    try:
        yield
    finally:
        engine.PROTOCOL, engine.code_hashes = previous


def labelled_target(target, root):
    """Restore labels only for SDA, after GT-free membership/sampling is frozen."""
    paths = [root/'manifests'/f'celebdf_v2_roi16_non_test_{kind}_v1.csv' for kind in ['real', 'fake']]
    catalog = pd.concat([pd.read_csv(p, dtype=str, keep_default_na=False) for p in paths], ignore_index=True)
    if catalog.crop_path.duplicated().any() or not catalog.split.eq('non_test_pool').all():
        raise ValueError('Invalid non-test label catalog')
    lookup = catalog.set_index('crop_path')
    if not target.crop_path.isin(lookup.index).all():
        raise ValueError('Target missing from non-test label catalog')
    matched = lookup.loc[target.crop_path]
    if (matched.video_id.tolist() != target.video_id.tolist()
            or matched.roi_status.tolist() != target.roi_status.tolist()
            or not matched.class_index.isin(['0', '1']).all()):
        raise ValueError('Non-test target metadata/label mismatch')
    result = target.copy()
    result['class_index'] = matched.class_index.to_numpy()
    return result


def find_ce_run(args, name):
    basename = rank.run_name(name, 'ce_rank', args.seed)
    folders = sorted({p.parent for p in args.ce_rank_runs_root.rglob('config.json') if p.parent.name == basename})
    if len(folders) != 1:
        raise FileNotFoundError(f'Expected one completed CE+Rank run for {name}, found {len(folders)} under {args.ce_rank_runs_root}')
    run = folders[0]
    for filename in ['config.json', 'completed.json', 'best.pt']:
        if not (run/filename).is_file():
            raise FileNotFoundError(run/filename)
    return run


def verify_ce_reference(args, name, data, cfg, initial, initial_hash):
    run = find_ce_run(args, name)
    saved = json.loads((run/'config.json').read_text())
    done = json.loads((run/'completed.json').read_text())
    expected = dict(protocol=rank.PROTOCOL, model=name, strategy='ce_rank', seed=args.seed,
        data=data, smoothing=cfg, initial_checkpoint_sha256=initial_hash,
        preprocessing=base.PREPROCESS[name], train_blocks=initial['config']['train_blocks'],
        epochs=args.epochs, patience=args.patience, min_delta=.0001,
        source_images=args.source_images, val_images=args.val_images, target_images=args.target_images,
        batch_size=args.batch_size, encoder_lr=args.encoder_lr, head_lr=args.head_lr,
        weight_decay=args.weight_decay, ranking_weight=args.ranking_weight,
        donor_domain='none', code_sha256=rank.code_hashes())
    mismatch = [key for key, value in expected.items() if saved.get(key) != value]
    if mismatch or done.get('identity') != evaluation.key(saved):
        raise ValueError(f'CE+Rank is not a matched control ({mismatch}): {run}')
    state = torch.load(run/'best.pt', map_location='cpu', weights_only=True)
    if (state.get('config') != saved or state.get('identity') != done['identity']
            or state.get('epoch') != done['selected_epoch']):
        raise ValueError(f'CE+Rank checkpoint/config mismatch: {run}')
    del state
    return dict(run=str(run), identity=done['identity'], selected_epoch=done['selected_epoch'],
        checkpoint_sha256=base.common.digest(run/'best.pt'))


def inputs(args):
    reference.verify_membership(args)
    pool, val, source_data = common.shared.source_inputs(args)
    _, other_val, target, cfg, target_data, blocked = adaptation.dataset_inputs(args)
    if not val.equals(other_val) or list(cfg['conditions']) != CONDITIONS:
        raise ValueError('Source validation/smoothing mismatch')
    target = target[['crop_path', 'video_id', 'roi_status']]
    data = {**source_data, **target_data}
    args.manifest_root.mkdir(parents=True, exist_ok=True)
    engine.freeze_json(args.manifest_root/'protocol.json', dict(protocol=PROTOCOL, data=data,
        seed=args.seed, excluded_video_ids=blocked, source_images=args.source_images,
        target_images=args.target_images, target_supervision={'rank_ts':'unlabelled', 'rank_sda':'actual non-test labels; diagnostic only'}))
    engine.freeze_csv(args.manifest_root/'source_validation.csv', val)
    engine.freeze_csv(args.manifest_root/'target_unlabelled.csv', target)
    labelled = labelled_target(target, args.target_root) if 'rank_sda' in args.strategies else None
    if labelled is not None:
        engine.freeze_csv(args.manifest_root/'target_sda_labelled.csv', labelled)
    return pool, val, target, labelled, cfg, data


def configuration(args, name, strategy, data, cfg, initial, initial_hash, control, labelled):
    return dict(protocol=PROTOCOL, model=name, strategy=strategy, seed=args.seed, data=data,
        smoothing=cfg, initial_checkpoint_sha256=initial_hash, preprocessing=base.PREPROCESS[name],
        train_blocks=initial['config']['train_blocks'], epochs=args.epochs, patience=args.patience,
        min_delta=.0001, source_images=args.source_images, val_images=args.val_images,
        target_images=args.target_images, batch_size=args.batch_size, encoder_lr=args.encoder_lr,
        head_lr=args.head_lr, weight_decay=args.weight_decay, ranking_weight=args.ranking_weight,
        ema_decay=args.ema_decay, pseudo_threshold=args.pseudo_threshold,
        target_weight=args.target_weight, ramp_epochs=args.ramp_epochs,
        source_loss='same completed CE+Rank: mean GT CE + ranking_weight*mean softplus(real_margin-fake_margin), all cross-view pairs',
        target_loss=('EMA hard-label CE, confidence gate; whole target batch denominator' if strategy == 'rank_ts'
                     else 'actual non-test label CE on ALL observed target examples; diagnostic supervised DA'),
        target_labels_used=strategy == 'rank_sda',
        target_label_sha256=evaluation.key(labelled.class_index.tolist()) if strategy == 'rank_sda' else None,
        target_class_counts=labelled.class_index.value_counts().to_dict() if strategy == 'rank_sda' else None,
        target_sampling='same GT-free video-uniform replacement; no class balancing; 6000 exposures/epoch',
        target_views='observed normal/weak/medium/strong uniformly; same scheduled image and flip across methods; student complementary flip',
        source_sampling='same CE+Rank rotating source sample, balanced 1:1 batches, paired clean/random smoothing with identical flip',
        selection='maximum FF++ four-condition mean raw-margin AUC; exact AUC tie minimum BER; epoch0 eligible; no target/test selection',
        comparison_limit='SDA uses all labels, TS confidence gates; difference includes supervision coverage, not just label correctness. Early stopping and computation can differ.',
        bn_statistics='frozen', inference_model='student',
        precision='BF16 if supported else FP32 student; FP32 EMA/validation/test',
        ce_rank_reference=control, hard_reference_manifest=args._hard_reference_identity,
        versions={p:engine.metadata.version(p) for p in ['torch','torchvision','Pillow','numpy','pandas','scikit-learn']},
        torch=str(torch.__version__), code_sha256=code_hashes())


def target_objective(student, teacher, labels, strategy, threshold):
    if strategy == 'rank_ts':
        return engine.pseudo_loss(student, teacher, threshold)
    if strategy != 'rank_sda' or labels is None:
        raise ValueError('Actual non-test labels are required only for rank_sda')
    loss = nn.functional.cross_entropy(student.float(), labels)
    counts = torch.stack([(labels == k).sum() for k in [0, 1]])
    return loss, torch.cat([counts, counts]), student.new_tensor(float(len(labels)))


def prepare(args):
    _, _, _, labelled, cfg, data = inputs(args)
    for name in args.models:
        initial, initial_hash = evaluation.checkpoint(argparse.Namespace(runs_root=args.baseline_runs_root, seed=args.seed), name)
        control = verify_ce_reference(args, name, data, cfg, initial, initial_hash)
        print(f'Ready {name}: reuse CE+Rank epoch {control["selected_epoch"]}; train {args.strategies}', flush=True)
        del initial
    if labelled is not None:
        print('SDA diagnostic target class counts:', labelled.class_index.value_counts().to_dict(), flush=True)


def train(args):
    pool, val, target, labelled, cfg, data = inputs(args)
    common.prepare_pixel_cache(args, cfg, data)
    common.prepare_validation(args, val, cfg, data={k: data[k] for k in
        ['source_trainval_sha256', 'source_protocol_sha256', 'val_paths_sha256', 'validation_images']})
    args._target_cache = common.pixel_cache.BilateralCache(args.target_train_cache/'observed', cfg['bilateral'],
        dict(target_paths_sha256=data['target_paths_sha256'], smoothing=cfg,
             opencv=evaluation.cv2.__version__, code_sha256=rank.ORIGINAL_HASHES()))
    label_lookup = labelled.set_index('crop_path').class_index.astype(int).to_dict() if labelled is not None else None
    print(f'Source={args.source_images}/epoch; target={args.source_images} exposures/epoch; FF++ val={len(val)} x4; best=mean AUC', flush=True)
    for name in args.models:
        initial, initial_hash = evaluation.checkpoint(argparse.Namespace(runs_root=args.baseline_runs_root, seed=args.seed), name)
        control = verify_ce_reference(args, name, data, cfg, initial, initial_hash)
        if (initial['config']['data_hashes']['trainval'] != data['source_trainval_sha256']
                or initial['config']['data_hashes']['protocol'] != data['source_protocol_sha256']):
            raise ValueError('Baseline/source mismatch')
        for strategy in args.strategies:
            config = configuration(args, name, strategy, data, cfg, initial, initial_hash, control, labelled)
            identity = evaluation.key(config)
            run = args.runs_root/run_name(name, strategy, args.seed)
            run.mkdir(parents=True, exist_ok=True)
            engine.freeze_json(run/'config.json', config)
            if (run/'completed.json').is_file():
                done = json.loads((run/'completed.json').read_text())
                if done['identity'] != identity or not (run/'best.pt').is_file():
                    raise ValueError('Completed run mismatch')
                print('Already completed:', name, strategy, flush=True)
                continue
            base.common.seed_all(args.seed)
            model, transform = base.build_model(name, pretrained=False)
            model.load_state_dict(initial['model_state_dict'], strict=True)
            model.to(args.device)
            blocks = 2 if name == 'xception' else initial['config']['train_blocks']
            common.training_mode(model, name, blocks)
            teacher = copy.deepcopy(model).requires_grad_(False).eval() if strategy == 'rank_ts' else None
            head = base.head(model, name)
            head_ids = {id(p) for p in head.parameters()}
            groups = [{'params':[p for p in model.parameters() if p.requires_grad and id(p) not in head_ids], 'lr':args.encoder_lr},
                      {'params':list(head.parameters()), 'lr':args.head_lr}]
            optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
            parameters = [p for group in groups for p in group['params']]
            start, stale, history = 0, 0, []
            if (run/'last.pt').is_file():
                state = torch.load(run/'last.pt', map_location='cpu', weights_only=True)
                if state['identity'] != identity:
                    raise ValueError('Resume identity differs')
                model.load_state_dict(state['model_state_dict'], strict=True)
                if teacher is not None:
                    teacher.load_state_dict(state['teacher_state_dict'], strict=True)
                optimizer.load_state_dict(state['optimizer'])
                start, stale, history, best, anchor = [state[k] for k in ['epoch', 'stale', 'history', 'best', 'anchor']]
                del state
                print('Resume completed epoch:', start, flush=True)
            else:
                values = engine.validate(args, model, val, transform)
                best = dict(auc=values['mean_auc'], ber=values['mean_ber'])
                anchor = best['auc']
                base.common.json_write(run/'epoch0_validation.json', values)
                base.common.save(run/'best.pt', dict(identity=identity, config=config, model_state_dict=model.state_dict(), epoch=0, best=best, validation=values))
            for epoch in range(start, args.epochs):
                if stale >= args.patience:
                    break
                seed = args.seed+epoch*100003
                base.common.seed_all(seed)
                common.training_mode(model, name, blocks)
                sampled = adaptation.balanced_source(pool, args.source_images, seed)
                batches = rank.balanced_batches(torch.tensor(sampled.class_index.astype(int).to_numpy()), args.batch_size, seed)
                source_loader = DataLoader(common.PairedSource(sampled, args, transform, cfg, epoch), batch_sampler=batches,
                    num_workers=args.workers, pin_memory=args.device == 'cuda', persistent_workers=False,
                    generator=torch.Generator().manual_seed(seed))
                engine.freeze_csv(args.manifest_root/f'source_epoch{epoch+1:02d}.csv', sampled)
                plan = engine.target_plan(target, len(sampled), seed)
                engine.freeze_csv(args.manifest_root/f'target_exposures_epoch{epoch+1:02d}.csv', pd.DataFrame(plan))
                target_loader = iter(base.common.make_loader(engine.TargetViews(plan, args, transform, cfg, False), args, False, seed))
                offset, totals, counts, confidence_sum = 0, np.zeros(5), np.zeros(4, dtype=np.int64), 0.
                for step, (clean, smooth, labels, _, _) in enumerate(tqdm(source_loader,
                        desc=f'{name} {strategy} {epoch+1}/{args.epochs}', mininterval=15, dynamic_ncols=True)):
                    base.common.seed_all(seed+step)
                    clean, smooth, labels = clean.to(args.device), smooth.to(args.device), labels.to(args.device)
                    n = len(labels)
                    observed, student_view = next(target_loader)
                    if len(observed) != n:
                        raise ValueError('Source/target batch size differs')
                    target_labels = None
                    teacher_logits = None
                    if teacher is not None:
                        with torch.no_grad():
                            teacher_logits = teacher(observed.to(args.device)).float()
                        if not torch.isfinite(teacher_logits).all():
                            raise RuntimeError('Nonfinite teacher logits; last completed epoch preserved')
                    else:
                        target_labels = torch.tensor([label_lookup[p['crop_path']] for p in plan[offset:offset+n]], device=args.device)
                    optimizer.zero_grad(set_to_none=True)
                    with base.amp_context(args):
                        source_logits, _ = adaptation.outputs(name, model, torch.cat([clean, smooth]))
                        logits = torch.stack([source_logits[:n], source_logits[n:]], dim=1)
                        source_loss, ce, pair_rank, _ = rank.classification_and_ranking(logits, labels, args.ranking_weight)
                        target_logits = model(student_view.to(args.device))
                        target_loss, stats, confidence = target_objective(target_logits, teacher_logits, target_labels, strategy, args.pseudo_threshold)
                        weight = args.target_weight*min(1., (epoch+(step+1)/len(source_loader))/args.ramp_epochs)
                        loss = source_loss+weight*target_loss
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite loss; last completed epoch preserved')
                    loss.backward()
                    nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                    optimizer.step()
                    if teacher is not None:
                        engine.update_teacher(teacher, model, args.ema_decay)
                    totals += n*np.array([loss.item(), ce.item(), pair_rank.item(), target_loss.item(), weight])
                    counts += stats.detach().cpu().numpy()
                    confidence_sum += confidence.item()
                    offset += n
                if offset != len(plan):
                    raise ValueError('Target/source exposure count differs')
                values = engine.validate(args, model, val, transform)
                improved = rank.is_better(values, best)
                if improved:
                    best = dict(auc=values['mean_auc'], ber=values['mean_ber'])
                if values['mean_auc'] > anchor+.0001:
                    anchor, stale = values['mean_auc'], 0
                else:
                    stale += 1
                entry = dict(epoch=epoch+1, loss=float(totals[0]/offset), source_ce=float(totals[1]/offset),
                    source_rank=float(totals[2]/offset), target_loss=float(totals[3]/offset), target_weight_mean=float(totals[4]/offset),
                    target_selected_counts={'real':int(counts[2]), 'fake':int(counts[3])},
                    target_selected_fraction=float(counts[2:].sum()/offset),
                    teacher_mean_confidence=float(confidence_sum/offset) if teacher is not None else None,
                    target_exposures=offset, target_condition_counts=dict(Counter(p['condition'] for p in plan)),
                    source_paths_sha256=evaluation.key(sampled.crop_path.tolist()), source_batches_sha256=evaluation.key(batches),
                    target_plan_sha256=evaluation.key(plan), validation=values)
                history.append(entry)
                state = dict(identity=identity, config=config, model_state_dict=model.state_dict(),
                    teacher_state_dict=teacher.state_dict() if teacher is not None else None, optimizer=optimizer.state_dict(),
                    epoch=epoch+1, stale=stale, history=history, best=best, anchor=anchor)
                if improved:
                    base.common.save(run/'best.pt', {k:state[k] for k in ['identity', 'config', 'model_state_dict', 'epoch', 'best']} | {'validation':values})
                base.common.save(run/'last.pt', state)
                del state
                base.common.json_write(run/'history.json', history)
                print(f'CE={entry["source_ce"]:.4f} rank={entry["source_rank"]:.4f} target={entry["target_loss"]:.4f} '
                    f'used={entry["target_selected_fraction"]:.1%} | val AUC={values["mean_auc"]:.4f} BER={values["mean_ber"]:.2%} '
                    f'best AUC={best["auc"]:.4f} patience={stale}/{args.patience}', flush=True)
            selected = torch.load(run/'best.pt', map_location='cpu', weights_only=True)
            base.common.json_write(run/'completed.json', dict(identity=identity, epochs_completed=len(history),
                selected_epoch=selected['epoch'], best_mean_auc=best['auc'], selected_mean_ber=best['ber'],
                inference_model='student', target_labels_used=strategy == 'rank_sda'))
            del selected, model, teacher, optimizer, parameters, groups, head
            if args.device == 'cuda':
                torch.cuda.empty_cache()
        del initial


def ce_records(args):
    ea = argparse.Namespace(data_root=args.target_root, manifest=args.target_root/'manifests/celebdf_v2_roi16_test_v1.csv',
        protocol=args.protocol, cache_root=args.cache_root, output_root=args.output_root, workers=args.workers)
    frame, cfg, provenance = evaluation.inputs(ea)
    _, _, source_data = common.shared.source_inputs(args)
    _, _, _, _, target_data, _ = adaptation.dataset_inputs(args)
    records, sources = [], []
    reader_args = copy.copy(args)
    reader_args.hard_evaluations = args.ce_rank_evaluations
    for name in args.models:
        initial, initial_hash = evaluation.checkpoint(argparse.Namespace(runs_root=args.baseline_runs_root, seed=args.seed), name)
        control = verify_ce_reference(args, name, {**source_data, **target_data}, cfg, initial, initial_hash)
        del initial
        for condition in CONDITIONS:
            metrics, report, checkpoint_hash = reference.prior_predictions(reader_args, name, 'ce_rank', condition, frame, provenance, initial_hash)
            if checkpoint_hash != control['checkpoint_sha256']:
                raise ValueError(f'CE+Rank predictions use another checkpoint: {report}')
            records.append(dict(model=name, strategy='ce_rank', condition=condition, **{k:metrics[k] for k in reference.METRICS}))
            sources.append(dict(report=str(report), report_sha256=base.common.digest(report),
                prediction_sha256=base.common.digest(report.with_suffix('.csv')), checkpoint_sha256=checkpoint_hash))
    return records, sources


def evaluate(args):
    records, sources = ce_records(args)  # Detect missing reusable results before GPU inference.
    available = [s for s in STRATEGIES if all((args.runs_root/run_name(n, s, args.seed)/'completed.json').is_file() for n in args.models)]
    if not set(args.strategies).issubset(available):
        raise FileNotFoundError('Requested adaptation run is not completed')
    copied = copy.copy(args)
    copied.strategies = available
    copied.batch_size = getattr(args, 'eval_batch_size', 32)
    copied.reference_evaluations = args.ce_rank_evaluations
    with protocol_context():
        engine.evaluate(copied)
    combined = pd.concat([pd.read_csv(args.output_root/'comparison.csv'), pd.DataFrame(records)], ignore_index=True)
    keys = ['model', 'strategy', 'condition']
    expected = set((m, s, c) for m in args.models for s in ['baseline', 'ce_rank', *available] for c in CONDITIONS)
    if combined.duplicated(keys).any() or set(map(tuple, combined[keys].to_numpy())) != expected:
        raise ValueError('Incomplete/duplicate matched comparison')
    combined.to_csv(args.output_root/'comparison_with_ce_rank.csv', index=False)
    means = combined.groupby(['model', 'strategy'], sort=False)[reference.METRICS].mean().reset_index()
    means.to_csv(args.output_root/'condition_means_with_ce_rank.csv', index=False)
    differences, changes = [], []
    for _, row in means[means.strategy.isin(available)].iterrows():
        control = means[means.model.eq(row.model) & means.strategy.eq('ce_rank')].iloc[0]
        differences.append(dict(model=row.model, strategy=row.strategy,
            **{'delta_'+metric:row[metric]-control[metric] for metric in reference.METRICS}))
    for (model, strategy), group in combined.groupby(['model', 'strategy'], sort=False):
        by = group.set_index('condition')
        changes.append(dict(model=model, strategy=strategy,
            **{'strong_minus_normal_'+metric:by.loc['smooth_strong', metric]-by.loc['normal', metric] for metric in reference.METRICS}))
    pd.DataFrame(differences).to_csv(args.output_root/'deltas_vs_ce_rank.csv', index=False)
    pd.DataFrame(changes).to_csv(args.output_root/'smoothing_changes.csv', index=False)
    base.common.json_write(args.output_root/'comparison_protocol.json', dict(protocol=PROTOCOL,
        reused_ce_rank_predictions=sources, checkpoint_selection='FF++ four-view mean raw-margin AUC maximum',
        sda_diagnostic_only=True, sda_uses_all_labels_ts_confidence_gates=True,
        test_used_for_selection=False, completed_strategies=available))
    print('Saved:', args.output_root/'condition_means_with_ce_rank.csv', flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'train', 'evaluate'])
    for field in ['source-root', 'target-root', 'baseline-runs-root', 'runs-root', 'manifest-root', 'validation-cache',
                  'output-root', 'hard-manifest-root', 'ce-rank-runs-root', 'ce-rank-evaluations']:
        p.add_argument('--'+field, type=Path, required=True)
    p.add_argument('--source-validation', type=Path)
    for field, default in [('train-cache', '/content/smoothing_pivot_bilateral_cache_v1'),
                          ('target-train-cache', '/content/smoothing_mean_teacher_target_cache_v1'),
                          ('cache-root', '/content/celebdf_official_smoothing_cache_v1'),
                          ('evaluation-pixel-cache', '/content/smoothing_ranking_cdf_bilateral_cache_v1')]:
        p.add_argument('--'+field, type=Path, default=Path(default))
    p.add_argument('--protocol', type=Path, default=ROOT/'configs/ffpp_c23_celebdf_smoothing_v1.json')
    p.add_argument('--models', nargs='+', choices=list(base.PREPROCESS), default=['dinov2_b14'])
    p.add_argument('--strategies', nargs='+', choices=STRATEGIES, default=STRATEGIES)
    for field, default in [('seed', 42), ('epochs', 10), ('patience', 3), ('source-images', 6000), ('val-images', 1200),
                          ('target-images', 5000), ('batch-size', 16), ('eval-batch-size', 32), ('workers', 2)]:
        p.add_argument('--'+field, type=int, default=default)
    for field, default in [('encoder-lr', 1e-5), ('head-lr', 1e-5), ('weight-decay', 1e-4), ('ranking-weight', .25),
                          ('ema-decay', .999), ('pseudo-threshold', .9), ('target-weight', .5), ('ramp-epochs', 2.)]:
        p.add_argument('--'+field, type=float, default=default)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    a = p.parse_args()
    numbers = [a.encoder_lr, a.head_lr, a.weight_decay, a.ranking_weight, a.ema_decay, a.pseudo_threshold, a.target_weight, a.ramp_epochs]
    if (not np.isfinite(numbers).all() or min(a.encoder_lr, a.head_lr, a.ramp_epochs) <= 0
            or min(a.weight_decay, a.ranking_weight, a.target_weight) < 0 or not 0 <= a.ema_decay < 1
            or not .5 <= a.pseudo_threshold <= 1 or min(a.epochs, a.patience, a.source_images, a.val_images, a.target_images, a.batch_size) < 1
            or a.batch_size % 2 or a.eval_batch_size < 1 or a.source_images % 8 or a.val_images % 8 or a.workers < 0
            or len(set(a.models)) != len(a.models) or len(set(a.strategies)) != len(a.strategies)):
        p.error('Invalid settings or duplicate models/strategies')
    if a.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime')
    return a


if __name__ == '__main__':
    arguments = parse_args()
    {'prepare': prepare, 'train': train, 'evaluate': evaluate}[arguments.action](arguments)
