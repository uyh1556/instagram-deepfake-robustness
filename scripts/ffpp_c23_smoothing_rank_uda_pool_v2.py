"""Pool v2 UDA: CE+Rank source loss + label-free CDF self-training with an EMA teacher.

ts_confidence     : hard pseudo labels where teacher softmax confidence >= 0.9 (previous TS rule).
ts_rank_quantile  : class-balanced rank pseudo labels. At each epoch start the EMA teacher scores
                    that epoch's planned CDF exposures; the top `quantile` of raw margins become fake,
                    the bottom `quantile` real, the middle is ignored. No class-prior estimate is used.

Training reads only crop_path/video_id/roi_status of pool v2 adapt-train. Actual labels are read
ONLY to log pseudo-label accuracy and the BBSE prior estimate error after each step; they never
reach the loss, sampling, thresholds, early stopping or checkpoint selection.
Every result folder is model-specific so several Colab runtimes (one per model) can run at once.
"""
import argparse
import copy
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import ffpp_c23_smoothing_rank_sda_pool_v2 as pool

rank, engine, common, base = pool.rank, pool.engine, pool.common, pool.base
adaptation, evaluation = pool.adaptation, pool.evaluation
ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'ffpp_c23_smoothing_rank_uda_pool_v2'
STRATEGIES = ['ts_confidence', 'ts_rank_quantile']
REFERENCES = ['ce_rank_pooled', 'sda_new_balanced']
CONDITIONS = pool.CONDITIONS


def code_hashes():
    return {**pool.code_hashes(), Path(__file__).name: base.common.digest(Path(__file__))}


def run_name(model, strategy, seed):
    return f'{model}_{strategy}_{PROTOCOL}_seed{seed}'


def confidence_loss(student, teacher, threshold):
    """Same as the previous TS rule (engine.pseudo_loss), also returning labels/mask."""
    confidence, labels = teacher.detach().float().softmax(1).max(1)
    selected = confidence >= threshold
    per_image = nn.functional.cross_entropy(student.float(), labels, reduction='none')
    return (per_image*selected).mean(), labels, selected


def rank_quantile_loss(student, teacher, low, high):
    """Fake if teacher margin >= high, real if <= low; whole-batch denominator like TS."""
    margin = (teacher[:, 1]-teacher[:, 0]).detach().float()
    fake, real = margin >= high, margin <= low
    labels, selected = fake.long(), fake | real
    per_image = nn.functional.cross_entropy(student.float(), labels, reduction='none')
    return (per_image*selected).mean(), labels, selected


def quantile_thresholds(margins, fraction):
    low, high = float(np.quantile(margins, fraction)), float(np.quantile(margins, 1-fraction))
    if not low < high:
        raise ValueError('Teacher margins too concentrated for distinct rank quantiles')
    return low, high


def truth_lookup(membership):
    """Logging only. Kept separate so tests can prove labels do not affect training."""
    return membership.set_index('crop_path').class_index.astype(int).to_dict()


def bbse_prior(rate_fake, validation):
    """Black-box shift estimate pi = (q - FPR)/(TPR - FPR) with FF++ val rates at margin 0.
    Logged only: its label-shift assumption is violated by the FF++->CDF score shift."""
    records = validation['ffpp']['conditions'].values()
    fpr = float(np.mean([r['real_fpr'] for r in records]))
    tpr = 1.-float(np.mean([r['fake_fnr'] for r in records]))
    raw = (rate_fake-fpr)/(tpr-fpr) if tpr > fpr else float('nan')
    return dict(teacher_fake_rate=rate_fake, ffpp_fpr=fpr, ffpp_tpr=tpr,
                pi_hat=float(np.clip(raw, 0., 1.)) if np.isfinite(raw) else None)


@torch.no_grad()
def teacher_margins(args, teacher, plan, transform, cfg, seed):
    teacher.eval()
    loader = base.common.make_loader(engine.TargetViews(plan, args, transform, cfg, False), args, False, seed)
    margins = []
    for observed, _ in tqdm(loader, desc='Teacher rank pass', mininterval=15, dynamic_ncols=True):
        logits = teacher(observed.to(args.device)).float()
        if not torch.isfinite(logits).all():
            raise RuntimeError('Nonfinite teacher logits')
        margins.append((logits[:, 1]-logits[:, 0]).cpu().numpy())
    return np.concatenate(margins)


def configuration(args, name, strategy, data, cfg, initial, initial_hash, summary):
    return dict(protocol=PROTOCOL, model=name, strategy=strategy, seed=args.seed, data=data, smoothing=cfg,
        initial_checkpoint_sha256=initial_hash, preprocessing=base.PREPROCESS[name],
        train_blocks=initial['config']['train_blocks'], epochs=args.epochs, patience=args.patience, min_delta=.0001,
        source_images=args.source_images, val_images=args.val_images, batch_size=args.batch_size,
        encoder_lr=args.encoder_lr, head_lr=args.head_lr, weight_decay=args.weight_decay,
        ranking_weight=args.ranking_weight, target_weight=args.target_weight, ramp_epochs=args.ramp_epochs,
        ema_decay=args.ema_decay,
        pseudo_rule=(dict(kind='confidence', threshold=args.pseudo_threshold) if strategy == 'ts_confidence'
                     else dict(kind='rank_quantile', quantile=args.quantile, thresholds='recomputed each epoch from EMA teacher margins on that epoch\'s planned exposures')),
        source_loss='CE + ranking_weight*mean softplus(real_margin-fake_margin), all cross-view pairs (CE+Rank)',
        target_loss='pseudo-label CE, whole target batch denominator, linear ramp to target_weight',
        target_pool='pool v2 adapt-train, label-free columns only', pool_v2_membership_sha256=summary['membership_sha256'],
        target_sampling='GT-free video-uniform replacement, natural class mix; source-matched exposures',
        target_views='normal/weak/medium/strong uniformly; teacher observed view, student complementary flip',
        labels_used_for='logging pseudo-label accuracy and BBSE error only',
        selection='maximum FF++ val pooled raw-margin AUC; tie higher minimum cross-condition AUC; epoch0 eligible',
        adapt_val='CDF pool-v2 adapt-val metrics logged each epoch; NOT used for selection or stopping',
        bn_statistics='frozen', inference_model='student',
        precision='BF16 if supported else FP32 student; FP32 teacher/validation/test',
        versions={p: engine.metadata.version(p) for p in ['torch', 'torchvision', 'Pillow', 'numpy', 'pandas', 'scikit-learn']},
        torch=str(torch.__version__), code_sha256=code_hashes())


def inputs(args):
    source, val, adapt_val, targets, cfg, data, summary = pool.inputs(args)
    if args.pool_reference is not None:
        reference = json.loads(args.pool_reference.read_text())
        if reference['membership_sha256'] != summary['membership_sha256']:
            raise ValueError('Pool v2 differs from the SDA/CE+Rank reference run; same seed/family-cap/adapt-val-fraction required')
    membership = targets['sda_new_balanced']
    unlabelled = membership[['crop_path', 'video_id', 'roi_status']].reset_index(drop=True)
    return source, val, adapt_val, unlabelled, membership, cfg, data, summary


def prepare(args):
    _, _, adapt_val, unlabelled, _, _, _, summary = inputs(args)
    for name in args.models:
        evaluation.checkpoint(argparse.Namespace(runs_root=args.baseline_runs_root, seed=args.seed), name)
    print('Pool v2 adapt-train (label-free for training):', len(unlabelled), 'images /', unlabelled.video_id.nunique(), 'videos', flush=True)
    print('adapt-val (logging only):', len(adapt_val), 'images', flush=True)
    print('Membership sha256:', summary['membership_sha256'], flush=True)


def train(args):
    source, val, adapt_val, unlabelled, membership, cfg, data, summary = inputs(args)
    truth = truth_lookup(membership)
    common.prepare_pixel_cache(args, cfg, {k: data[k] for k in ['source_trainval_sha256']})
    common.prepare_validation(args, val, cfg, data={k: data[k] for k in
        ['source_trainval_sha256', 'source_protocol_sha256', 'val_paths_sha256', 'validation_images']})
    args._pool_cache = args._target_cache = pool.target_cache(args, cfg, 'pool_v2_observed', data['pool_v2_membership_sha256'])
    print(f'Source={args.source_images}/epoch; CDF exposures={args.source_images}/epoch (label-free); best=FF++ pooled AUC', flush=True)
    for name in args.models:
        initial, initial_hash = evaluation.checkpoint(argparse.Namespace(runs_root=args.baseline_runs_root, seed=args.seed), name)
        if (initial['config']['data_hashes']['trainval'] != data['source_trainval_sha256']
                or initial['config']['data_hashes']['protocol'] != data['source_protocol_sha256']):
            raise ValueError('Baseline/source mismatch')
        for strategy in args.strategies:
            config = configuration(args, name, strategy, data, cfg, initial, initial_hash, summary)
            identity = evaluation.key(config)
            run = args.runs_root/run_name(name, strategy, args.seed)
            run.mkdir(parents=True, exist_ok=True)
            engine.freeze_json(run/'config.json', config)
            if (run/'completed.json').is_file():
                done = json.loads((run/'completed.json').read_text())
                if done['identity'] != identity or not (run/'best.pt').is_file():
                    raise ValueError('Completed run mismatch')
                print(f'===== ALREADY COMPLETED: {name} / {strategy} (skip) =====', flush=True)
                continue
            base.common.seed_all(args.seed)
            model, transform = base.build_model(name, pretrained=False)
            model.load_state_dict(initial['model_state_dict'], strict=True)
            model.to(args.device)
            blocks = 2 if name == 'xception' else initial['config']['train_blocks']
            common.training_mode(model, name, blocks)
            teacher = copy.deepcopy(model).requires_grad_(False).eval()
            head = base.head(model, name)
            head_ids = {id(p) for p in head.parameters()}
            groups = [{'params': [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids], 'lr': args.encoder_lr},
                      {'params': list(head.parameters()), 'lr': args.head_lr}]
            optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
            parameters = [p for group in groups for p in group['params']]
            start, stale, history = 0, 0, []
            if (run/'last.pt').is_file():
                state = torch.load(run/'last.pt', map_location='cpu', weights_only=True)
                if state['identity'] != identity:
                    raise ValueError('Resume identity differs')
                model.load_state_dict(state['model_state_dict'], strict=True)
                teacher.load_state_dict(state['teacher_state_dict'], strict=True)
                optimizer.load_state_dict(state['optimizer'])
                start, stale, history, best, anchor, previous = [state[k] for k in
                    ['epoch', 'stale', 'history', 'best', 'anchor', 'last_validation']]
                del state
                print(f'===== RESUME: {name} / {strategy} from completed epoch {start} | {run} =====', flush=True)
            else:
                print(f'===== START NEW: {name} / {strategy} (no last.pt) | {run} =====', flush=True)
                previous = pool.validate(args, model, val, adapt_val, transform, cfg)
                best, anchor = pool.best_of(previous['ffpp']), previous['ffpp']['pooled_auc']
                base.common.json_write(run/'epoch0_validation.json', previous)
                base.common.save(run/'best.pt', dict(identity=identity, config=config, model_state_dict=model.state_dict(),
                    epoch=0, best=best, validation=previous))
            for epoch in range(start, args.epochs):
                if stale >= args.patience:
                    break
                seed = args.seed+epoch*100003
                base.common.seed_all(seed)
                common.training_mode(model, name, blocks)
                sampled = adaptation.balanced_source(source, args.source_images, seed)
                batches = rank.balanced_batches(torch.tensor(sampled.class_index.astype(int).to_numpy()), args.batch_size, seed)
                source_loader = DataLoader(common.PairedSource(sampled, args, transform, cfg, epoch), batch_sampler=batches,
                    num_workers=args.workers, pin_memory=args.device == 'cuda', persistent_workers=False,
                    generator=torch.Generator().manual_seed(seed))
                engine.freeze_csv(args.manifest_root/f'source_epoch{epoch+1:02d}.csv', sampled)
                plan = engine.target_plan(unlabelled, len(sampled), seed)
                engine.freeze_csv(args.manifest_root/'target_exposures'/f'epoch{epoch+1:02d}.csv', pd.DataFrame(plan))
                thresholds = None
                if strategy == 'ts_rank_quantile':
                    thresholds = quantile_thresholds(teacher_margins(args, teacher, plan, transform, cfg, seed), args.quantile)
                    print(f'Rank thresholds epoch {epoch+1}: real <= {thresholds[0]:.3f}, fake >= {thresholds[1]:.3f}', flush=True)
                base.common.seed_all(seed)
                target_loader = iter(base.common.make_loader(engine.TargetViews(plan, args, transform, cfg, False), args, False, seed))
                offset, totals, confusion, online = 0, np.zeros(5), np.zeros((2, 2), dtype=np.int64), []
                for step, (clean, smooth, labels_s, _, _) in enumerate(tqdm(source_loader,
                        desc=f'{name} {strategy} {epoch+1}/{args.epochs}', mininterval=15, dynamic_ncols=True)):
                    base.common.seed_all(seed+step)
                    clean, smooth, labels_s = clean.to(args.device), smooth.to(args.device), labels_s.to(args.device)
                    n = len(labels_s)
                    observed, student_view = next(target_loader)
                    if len(observed) != n:
                        raise ValueError('Source/target batch size differs')
                    with torch.no_grad():
                        teacher_logits = teacher(observed.to(args.device)).float()
                    if not torch.isfinite(teacher_logits).all():
                        raise RuntimeError('Nonfinite teacher logits; last completed epoch preserved')
                    optimizer.zero_grad(set_to_none=True)
                    with base.amp_context(args):
                        source_logits, _ = adaptation.outputs(name, model, torch.cat([clean, smooth]))
                        logits = torch.stack([source_logits[:n], source_logits[n:]], dim=1)
                        source_loss, ce, pair_rank, _ = rank.classification_and_ranking(logits, labels_s, args.ranking_weight)
                        student_logits = model(student_view.to(args.device))
                        if strategy == 'ts_confidence':
                            target_loss, pseudo, selected = confidence_loss(student_logits, teacher_logits, args.pseudo_threshold)
                        else:
                            target_loss, pseudo, selected = rank_quantile_loss(student_logits, teacher_logits, *thresholds)
                        weight = args.target_weight*min(1., (epoch+(step+1)/len(source_loader))/args.ramp_epochs)
                        loss = source_loss+weight*target_loss
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite loss; last completed epoch preserved')
                    loss.backward()
                    nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                    optimizer.step()
                    engine.update_teacher(teacher, model, args.ema_decay)
                    # Logging only: actual labels never enter the objective above.
                    rows = plan[offset:offset+n]
                    for row, p, s in zip(rows, pseudo.tolist(), selected.tolist()):
                        if s:
                            confusion[truth[row['crop_path']], p] += 1
                    online.append((teacher_logits[:, 1]-teacher_logits[:, 0]).cpu().numpy())
                    totals += n*np.array([loss.item(), ce.item(), pair_rank.item(), target_loss.item(), weight])
                    offset += n
                if offset != len(plan):
                    raise ValueError('Target/source exposure count differs')
                prior = bbse_prior(float((np.concatenate(online) >= 0).mean()), previous)
                prior['pi_true_exposures'] = float(np.mean([truth[r['crop_path']] for r in plan]))
                values = pool.validate(args, model, val, adapt_val, transform, cfg)
                improved = pool.is_better(values['ffpp'], best)
                if improved:
                    best = pool.best_of(values['ffpp'])
                if values['ffpp']['pooled_auc'] > anchor+.0001:
                    anchor, stale = values['ffpp']['pooled_auc'], 0
                else:
                    stale += 1
                chosen = int(confusion.sum())
                entry = dict(epoch=epoch+1, loss=float(totals[0]/offset), source_ce=float(totals[1]/offset),
                    source_rank=float(totals[2]/offset), target_ce=float(totals[3]/offset), target_weight_mean=float(totals[4]/offset),
                    rank_thresholds=list(thresholds) if thresholds else None,
                    pseudo_selected=dict(real=int(confusion[:, 0].sum()), fake=int(confusion[:, 1].sum()), fraction=chosen/offset),
                    pseudo_confusion_true_by_pseudo=confusion.tolist(),
                    pseudo_accuracy=float(np.trace(confusion)/chosen) if chosen else None,
                    pseudo_wrong_fake_given_true_real=int(confusion[0, 1]), pseudo_wrong_real_given_true_fake=int(confusion[1, 0]),
                    bbse=prior, target_exposures=offset, target_condition_counts=dict(Counter(p['condition'] for p in plan)),
                    source_paths_sha256=evaluation.key(sampled.crop_path.tolist()), source_batches_sha256=evaluation.key(batches),
                    target_plan_sha256=evaluation.key(plan), validation=values)
                history.append(entry)
                previous = values
                state = dict(identity=identity, config=config, model_state_dict=model.state_dict(),
                    teacher_state_dict=teacher.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch+1, stale=stale,
                    history=history, best=best, anchor=anchor, last_validation=values)
                if improved:
                    base.common.save(run/'best.pt', {k: state[k] for k in ['identity', 'config', 'model_state_dict', 'epoch', 'best']}
                        | dict(validation=values))
                base.common.save(run/'last.pt', state)
                del state
                base.common.json_write(run/'history.json', history)
                f, t = values['ffpp'], values['adapt_val']
                accuracy = f'{entry["pseudo_accuracy"]:.1%}' if chosen else 'n/a'
                print(f'CE={entry["source_ce"]:.4f} targetCE={entry["target_ce"]:.4f} | pseudo used={entry["pseudo_selected"]} '
                    f'acc={accuracy} | BBSE pi_hat={prior["pi_hat"]} true={prior["pi_true_exposures"]:.3f} | '
                    f'FF++ pooled={f["pooled_auc"]:.4f} | CDF adapt-val pooled={t["pooled_auc"]:.4f} (log only) | '
                    f'best={best["pooled_auc"]:.4f} patience={stale}/{args.patience}', flush=True)
            selected = torch.load(run/'best.pt', map_location='cpu', weights_only=True)
            base.common.json_write(run/'completed.json', dict(identity=identity, epochs_completed=len(history),
                selected_epoch=selected['epoch'], best_ffpp=best, inference_model='student', target_labels_used_for_training=False))
            del selected, model, teacher, optimizer, parameters, groups, head
            if args.device == 'cuda':
                torch.cuda.empty_cache()
        del initial


def find_reference(roots, model, label, frame, provenance):
    """Saved FP32 raw-margin predictions of the pool-v2 CE+Rank/SDA runs; read only."""
    for root in roots:
        banks = []
        for condition in CONDITIONS:
            hits = sorted(Path(root).rglob(f'{model}_{label}_{condition}.csv'))
            match = None
            for path in hits:
                protocol = path.parent/'evaluation_protocol.json'
                if not protocol.is_file() or json.loads(protocol.read_text()).get('manifest_sha256') != provenance['manifest_sha256']:
                    continue
                saved = pd.read_csv(path, dtype={'crop_path': str, 'video_id': str, 'class_index': str})
                if saved.crop_path.tolist() == frame.crop_path.tolist() and saved.class_index.tolist() == frame.class_index.tolist():
                    match = (path, saved.raw_margin.to_numpy())
                    break
            if match is None:
                break
            banks.append(match)
        if len(banks) == len(CONDITIONS):
            return [b for _, b in banks], [str(p) for p, _ in banks]
    return None, None


def identity_groups(frame, membership):
    ids = lambda v: set(re.findall(r'id\d+', v)) if not v.startswith('YouTube-real') else set()
    seen = set().union(*[ids(v) for v in membership.video_id.unique()])
    return frame.video_id.map(lambda v: 'youtube' if v.startswith('YouTube-real') else ('unseen' if not ids(v) & seen else 'seen')).to_numpy()


def evaluate(args):
    ea = argparse.Namespace(data_root=args.target_root, manifest=args.target_root/'manifests/celebdf_v2_roi16_test_v1.csv',
        protocol=args.protocol, cache_root=args.cache_root, output_root=args.output_root, workers=args.workers)
    frame, cfg, provenance = evaluation.inputs(ea)
    _, _, _, _, membership, _, _, _ = inputs(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    engine.freeze_json(args.output_root/'evaluation_protocol.json', dict(provenance, used_for_selection=False))
    engine.freeze_csv(args.output_root/'evaluation_manifest.csv', frame)
    view_cache = args.cache_root if (args.cache_root/'config.json').is_file() and json.loads((args.cache_root/'config.json').read_text()) == provenance else None
    native_cache = common.pixel_cache.BilateralCache(args.evaluation_pixel_cache, cfg['bilateral'],
        dict(data=provenance, cache_code_sha256=base.common.digest(Path(common.pixel_cache.__file__)),
             storage='native uint8 RGB bilateral only; lossless PNG'))
    labels = frame.class_index.astype(int).to_numpy()
    identity = identity_groups(frame, membership[membership.partition.eq('adapt_train')] if 'partition' in membership else membership)
    eval_args = copy.copy(args); eval_args.batch_size = args.eval_batch_size
    records, mixtures, cross, groups, bootstrap, pseudo, sources = [], [], [], [], [], [], []
    for name in args.models:
        initial, initial_hash = evaluation.checkpoint(argparse.Namespace(runs_root=args.baseline_runs_root, seed=args.seed), name)
        states = [('baseline', initial, initial_hash)]
        for strategy in args.strategies:
            run = args.runs_root/run_name(name, strategy, args.seed)
            config = json.loads((run/'config.json').read_text())
            done = json.loads((run/'completed.json').read_text())
            state = torch.load(run/'best.pt', map_location='cpu', weights_only=True)
            if (done['identity'] != evaluation.key(config) or state['identity'] != done['identity'] or config['model'] != name
                    or config['initial_checkpoint_sha256'] != initial_hash
                    or config['data']['test_manifest_sha256'] != provenance['manifest_sha256'] or config['code_sha256'] != code_hashes()):
                raise ValueError(f'Checkpoint/evaluation mismatch: {run}')
            states.append((strategy, state, base.common.digest(run/'best.pt')))
            for entry in json.loads((run/'history.json').read_text()):
                pseudo.append(dict(model=name, strategy=strategy, epoch=entry['epoch'], selected_epoch=entry['epoch'] == state['epoch'],
                    pseudo_accuracy=entry['pseudo_accuracy'], selected_fraction=entry['pseudo_selected']['fraction'],
                    selected_real=entry['pseudo_selected']['real'], selected_fake=entry['pseudo_selected']['fake'],
                    wrong_fake_given_true_real=entry['pseudo_wrong_fake_given_true_real'],
                    wrong_real_given_true_fake=entry['pseudo_wrong_real_given_true_fake'],
                    bbse_pi_hat=entry['bbse']['pi_hat'], true_pi=entry['bbse']['pi_true_exposures'],
                    ffpp_pooled_auc=entry['validation']['ffpp']['pooled_auc'], adapt_val_pooled_auc=entry['validation']['adapt_val']['pooled_auc']))
        banks_by = {}
        for label, state, checkpoint_hash in states:
            key = evaluation.key({'data': provenance, 'checkpoint': checkpoint_hash, 'precision': 'FP32',
                                  'code_sha256': code_hashes(), 'torch': str(torch.__version__)})
            model, banks, rows = None, [], []
            for condition in CONDITIONS:
                prefix = args.output_root/f'{name}_{label}_{condition}'
                report, pred = prefix.with_suffix('.json'), prefix.with_suffix('.csv')
                if report.is_file() and pred.is_file():
                    values = json.loads(report.read_text())
                    saved = pd.read_csv(pred, dtype={'crop_path': str, 'video_id': str, 'class_index': str})
                    if values['identity'] != key or saved.crop_path.tolist() != frame.crop_path.tolist():
                        raise ValueError('Saved evaluation differs; use a new output root')
                    print('Reuse:', name, label, condition, flush=True)
                else:
                    previous = None
                    if label == 'baseline':
                        for root in args.reference_roots:
                            previous = engine.previous_baseline(argparse.Namespace(reference_evaluations=root), name, condition, frame, provenance, checkpoint_hash)
                            if previous is not None:
                                break
                    elif state['epoch'] == 0:
                        path = args.output_root/f'{name}_baseline_{condition}.csv'
                        previous = (pd.read_csv(path), path)
                    if previous is not None:
                        old, _ = previous
                        probabilities, margins = old.probability_fake.to_numpy(), old.raw_margin.to_numpy()
                    else:
                        if model is None:
                            model, transform = base.build_model(name, pretrained=False)
                            model.load_state_dict(state['model_state_dict'], strict=True)
                            model.to(args.device).eval()
                        probabilities, margins = engine.infer(eval_args, model, frame,
                            engine.TestViews(frame, args, transform, cfg, condition, native_cache, view_cache), f'{name} {label} {condition}')
                    values = engine.raw_metrics(frame, probabilities, margins)
                    values.update(identity=key, images=len(frame), precision='FP32', threshold=.5, checkpoint_sha256=checkpoint_hash,
                        manifest_sha256=provenance['manifest_sha256'], selected_epoch=state.get('epoch', 0), unit='frame')
                    saved = frame[['crop_path', 'video_id', 'class_index', 'roi_status']].copy()
                    saved['probability_fake'], saved['raw_margin'] = probabilities, margins
                    temp = pred.with_suffix('.tmp.csv'); saved.to_csv(temp, index=False); temp.replace(pred)
                    base.common.json_write(report, values)
                banks.append(saved.raw_margin.to_numpy())
                rows.append(dict(model=name, strategy=label, condition=condition,
                    **{k: values[k] for k in ['real_fpr', 'fake_fnr', 'balanced_error_rate', 'auc']}))
            del model
            banks_by[label] = (banks, rows)
        for label in REFERENCES:
            banks, paths = find_reference(args.reference_roots, name, label, frame, provenance)
            if banks is None:
                print(f'Reference {label} for {name} not found; tables skip it', flush=True)
                continue
            metrics = [engine.raw_metrics(frame, 1/(1+np.exp(-np.clip(b, -50, 50))), b) for b in banks]
            banks_by[label] = (banks, [dict(model=name, strategy=label, condition=c, **{k: m[k] for k in ['real_fpr', 'fake_fnr', 'balanced_error_rate', 'auc']})
                                       for c, m in zip(CONDITIONS, metrics)])
            sources.append(dict(model=name, strategy=label, predictions=paths))
        pooled = {}
        for label, (banks, rows) in banks_by.items():
            records += rows
            summary = pool.summarize(labels, banks, {r['condition']: r for r in rows})
            pooled[label] = summary['pooled_auc']
            row = dict(model=name, strategy=label, **{k: summary[k] for k in ['pooled_auc', 'mean_auc', 'min_cross_auc', 'strong_real_vs_normal_fake_auc']},
                       mean_fpr=float(np.mean([r['real_fpr'] for r in rows])), mean_fnr=float(np.mean([r['fake_fnr'] for r in rows])))
            row['gap_closure_vs_sda'] = None
            for group in ['seen', 'unseen']:
                fake, real = (labels == 1) & (identity == group), (labels == 0) & (identity == group)
                row[f'{group}_identity_pooled_auc'] = None
                if fake.any() and real.any():
                    row[f'{group}_identity_pooled_auc'] = pool.fast_auc(np.r_[np.zeros(4*real.sum()), np.ones(4*fake.sum())],
                        np.r_[np.concatenate([b[real] for b in banks]), np.concatenate([b[fake] for b in banks])])
            mixtures.append(row)
            for i, real_condition in enumerate(CONDITIONS):
                for j, fake_condition in enumerate(CONDITIONS):
                    cross.append(dict(model=name, strategy=label, real_condition=real_condition, fake_condition=fake_condition, auc=summary['cross_auc'][i][j]))
            for condition, margins in zip(CONDITIONS, banks):
                groups.append(dict(model=name, strategy=label, condition=condition, **pool.subgroup_rows(frame, margins)))
        if set(REFERENCES) <= set(pooled) and pooled['sda_new_balanced'] != pooled['ce_rank_pooled']:
            for row in mixtures:
                if row['model'] == name and row['strategy'] in args.strategies:
                    row['gap_closure_vs_sda'] = (pooled[row['strategy']]-pooled['ce_rank_pooled'])/(pooled['sda_new_balanced']-pooled['ce_rank_pooled'])
        pairs = [(s, 'ce_rank_pooled') for s in args.strategies if 'ce_rank_pooled' in banks_by]
        pairs += [('ts_rank_quantile', 'ts_confidence')] if set(STRATEGIES) <= set(banks_by) else []
        for a, b in pairs:
            diffs = pool.video_bootstrap(frame, banks_by[a][0], banks_by[b][0], args.bootstrap, args.seed) if args.bootstrap else None
            for k, metric in enumerate(['pooled_auc', 'mean_auc']):
                observed = (pool.fast_auc(np.tile(labels, 4), np.concatenate(banks_by[a][0]))-pool.fast_auc(np.tile(labels, 4), np.concatenate(banks_by[b][0]))
                            if metric == 'pooled_auc' else float(np.mean([pool.fast_auc(labels, x) for x in banks_by[a][0]])-np.mean([pool.fast_auc(labels, x) for x in banks_by[b][0]])))
                bootstrap.append(dict(model=name, strategy=a, reference=b, metric=metric, observed=observed,
                    ci_low=float(np.quantile(diffs[:, k], .025)) if diffs is not None else None,
                    ci_high=float(np.quantile(diffs[:, k], .975)) if diffs is not None else None, repeats=args.bootstrap))
        del states, initial
    pd.DataFrame(records).to_csv(args.output_root/'comparison.csv', index=False)
    pd.DataFrame(mixtures).to_csv(args.output_root/'mixture_summary.csv', index=False)
    pd.DataFrame(cross).to_csv(args.output_root/'cross_condition_auc.csv', index=False)
    pd.DataFrame(groups).to_csv(args.output_root/'subgroups.csv', index=False)
    pd.DataFrame(bootstrap).to_csv(args.output_root/'bootstrap.csv', index=False)
    pd.DataFrame(pseudo).to_csv(args.output_root/'pseudo_labels.csv', index=False)
    base.common.json_write(args.output_root/'comparison_protocol.json', dict(protocol=PROTOCOL, selection='FF++ val pooled AUC',
        references_read_only=sources, gap_closure='(method - CE+Rank)/(SDA new - CE+Rank), pooled AUC',
        identity_groups='test Celeb videos whose filename ids never/ever appear in pool-v2 adapt-train',
        test_used_for_selection=False, bootstrap='test video sampling only; not seed variance'))
    print('Saved:', args.output_root, flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'train', 'evaluate'])
    for field in ['source-root', 'target-root', 'baseline-runs-root', 'runs-root', 'manifest-root', 'validation-cache', 'output-root']:
        p.add_argument('--'+field, type=Path, required=True)
    for field in ['source-validation', 'pool-reference']:
        p.add_argument('--'+field, type=Path)
    p.add_argument('--reference-roots', type=Path, nargs='*', default=[])
    for field, default in [('train-cache', '/content/smoothing_pivot_bilateral_cache_v1'),
                          ('target-train-cache', '/content/smoothing_rank_sda_pool_v2_target_cache'),
                          ('cache-root', '/content/celebdf_official_smoothing_cache_v1'),
                          ('evaluation-pixel-cache', '/content/smoothing_ranking_cdf_bilateral_cache_v1')]:
        p.add_argument('--'+field, type=Path, default=Path(default))
    p.add_argument('--protocol', type=Path, default=ROOT/'configs/ffpp_c23_celebdf_smoothing_v1.json')
    p.add_argument('--models', nargs='+', choices=list(base.PREPROCESS), default=['clip_b16'])
    p.add_argument('--strategies', nargs='+', choices=STRATEGIES, default=STRATEGIES)
    for field, default in [('seed', 42), ('epochs', 10), ('patience', 3), ('source-images', 6000), ('val-images', 1200),
                          ('target-images', 5000), ('batch-size', 16), ('eval-batch-size', 32), ('workers', 2),
                          ('family-cap', 3), ('bootstrap', 1000)]:
        p.add_argument('--'+field, type=int, default=default)
    for field, default in [('encoder-lr', 1e-5), ('head-lr', 1e-5), ('weight-decay', 1e-4), ('ranking-weight', .25),
                          ('target-weight', .5), ('ramp-epochs', 2.), ('adapt-val-fraction', .2), ('ema-decay', .999),
                          ('pseudo-threshold', .9), ('quantile', .2)]:
        p.add_argument('--'+field, type=float, default=default)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    a = p.parse_args(argv)
    a.hard_manifest_root = None
    numbers = [a.encoder_lr, a.head_lr, a.weight_decay, a.ranking_weight, a.target_weight, a.ramp_epochs,
               a.adapt_val_fraction, a.ema_decay, a.pseudo_threshold, a.quantile]
    if (not np.isfinite(numbers).all() or min(a.encoder_lr, a.head_lr, a.ramp_epochs) <= 0
            or min(a.weight_decay, a.ranking_weight, a.target_weight) < 0 or not 0 < a.adapt_val_fraction < 1
            or not 0 <= a.ema_decay < 1 or not .5 <= a.pseudo_threshold <= 1 or not 0 < a.quantile < .5
            or min(a.epochs, a.patience, a.source_images, a.val_images, a.target_images, a.batch_size, a.family_cap) < 1
            or a.batch_size % 2 or a.eval_batch_size < 1 or a.source_images % 8 or a.val_images % 8
            or a.source_images % a.batch_size or a.workers < 0 or a.bootstrap < 0
            or len(set(a.models)) != len(a.models) or len(set(a.strategies)) != len(a.strategies)):
        p.error('Invalid settings or duplicate models/strategies')
    if a.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime')
    return a


if __name__ == '__main__':
    arguments = parse_args()
    {'prepare': prepare, 'train': train, 'evaluate': evaluate}[arguments.action](arguments)
