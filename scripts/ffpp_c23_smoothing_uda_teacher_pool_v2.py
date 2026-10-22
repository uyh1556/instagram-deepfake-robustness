"""Teacher-protection UDA on pool v2 with a FIXED training length (no early stopping).

Strategies (all start from the FF++ baseline; source loss = CE+Rank on clean+smoothed pairs):
  ce_rank_fixed      source only; matched control for the fixed schedule
  ts_ema             TS baseline: EMA teacher, confidence >= 0.9 hard pseudo labels, one head
  ts_frozen_teacher  same, but the teacher stays the FF++ baseline (no EMA update)
  ts_decoupled_head  EMA teacher labels from the MAIN head; pseudo labels train only an
                     auxiliary head on shared features (decoupled generation/use, the DST
                     idea of Chen et al., NeurIPS 2022, without its worst-case module)

Length: ramp 2 epochs + 4 epochs at full target weight = 6 epochs, fixed before results.
Primary checkpoint = final epoch; best_ffpp.pt (FF++ pooled AUC) kept as a secondary view.
CDF labels are read only to log pseudo-label quality and adapt-val metrics; label-free
validators (entropy, teacher-student agreement, SND) are logged for later study only.
"""
import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import ffpp_c23_smoothing_rank_uda_pool_v2 as uda

pool, rank, engine, common, base = uda.pool, uda.rank, uda.engine, uda.common, uda.base
adaptation, evaluation = uda.adaptation, uda.evaluation
ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'ffpp_c23_smoothing_uda_teacher_pool_v2'
STRATEGIES = ['ce_rank_fixed', 'ts_ema', 'ts_frozen_teacher', 'ts_decoupled_head']
REFERENCES = ['ce_rank_pooled', 'sda_new_balanced', 'ts_confidence']
CONDITIONS = pool.CONDITIONS
CHECKPOINTS = {'final': 'final.pt', 'ffpp': 'best_ffpp.pt'}


def code_hashes():
    return {**uda.code_hashes(), Path(__file__).name: base.common.digest(Path(__file__))}


def run_name(model, strategy, seed):
    return f'{model}_{strategy}_{PROTOCOL}_seed{seed}'


class HeadView(nn.Module):
    """Model whose logits come from `head` on the shared features (auxiliary-head view)."""
    def __init__(self, name, model, head):
        super().__init__()
        self.name, self.model, self.head = name, model, head

    def forward(self, images):
        return self.head(adaptation.outputs(self.name, self.model, images)[1])


def snd(probabilities, temperature=.05, size=2000, seed=0):
    """Soft Neighborhood Density (Saito et al., ICCV 2021) on prediction vectors; label-free."""
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(probabilities), min(size, len(probabilities)), replace=False)
    x = torch.tensor(np.stack([1-probabilities[pick], probabilities[pick]], 1), dtype=torch.float32)
    x = nn.functional.normalize(x, dim=1)
    similarity = x@x.T
    similarity.fill_diagonal_(-float('inf'))
    q = (similarity/temperature).softmax(1)
    return float(-(q*torch.log(q+1e-12)).sum(1).mean())


def score(args, model, frame, dataset, title):
    banks, probabilities, records = [], [], {}
    for condition in CONDITIONS:
        p, m = engine.infer(args, model, frame, dataset(condition), f'{title} {condition}')
        records[condition] = engine.raw_metrics(frame, p, m)
        banks.append(m); probabilities.append(p)
    return pool.summarize(frame.class_index.astype(int).to_numpy(), banks, records), np.concatenate(probabilities)


def validate(args, name, model, teacher, aux, val, adapt_val, transform, cfg):
    model.eval()
    adapt_views = lambda c: engine.TestViews(adapt_val, args, transform, cfg, c, args._pool_cache, None)
    values = dict(ffpp=score(args, model, val, lambda c: common.ValidationViews(val, args, transform, c), 'FF++ val')[0])
    values['adapt_val'], student_p = score(args, model, adapt_val, adapt_views, 'CDF adapt-val')
    if aux is not None:
        values['adapt_val_aux_head'] = score(args, HeadView(name, model, aux).eval(), adapt_val, adapt_views, 'CDF adapt-val aux')[0]
    free = dict(mean_entropy=float(np.mean(-(student_p*np.log(student_p+1e-12)+(1-student_p)*np.log(1-student_p+1e-12)))),
                snd=snd(student_p))
    if teacher is not None:
        _, teacher_p = score(args, teacher.eval(), adapt_val, adapt_views, 'CDF adapt-val teacher')
        free['teacher_student_agreement'] = float(((student_p >= .5) == (teacher_p >= .5)).mean())
    values['label_free'] = free
    return values


def configuration(args, name, strategy, data, cfg, initial, initial_hash, summary):
    return dict(protocol=PROTOCOL, model=name, strategy=strategy, seed=args.seed, data=data, smoothing=cfg,
        initial_checkpoint_sha256=initial_hash, preprocessing=base.PREPROCESS[name],
        train_blocks=initial['config']['train_blocks'], epochs=args.epochs, early_stopping=None,
        source_images=args.source_images, val_images=args.val_images, batch_size=args.batch_size,
        encoder_lr=args.encoder_lr, head_lr=args.head_lr, weight_decay=args.weight_decay,
        ranking_weight=args.ranking_weight, target_weight=args.target_weight if strategy != 'ce_rank_fixed' else 0.,
        ramp_epochs=args.ramp_epochs, ema_decay=args.ema_decay if strategy in ('ts_ema', 'ts_decoupled_head') else None,
        pseudo_threshold=args.pseudo_threshold if strategy != 'ce_rank_fixed' else None,
        teacher={'ce_rank_fixed': None, 'ts_ema': 'EMA of student', 'ts_frozen_teacher': 'FF++ baseline, never updated',
                 'ts_decoupled_head': 'EMA of student encoder+MAIN head'}[strategy],
        pseudo_label_head={'ts_decoupled_head': 'auxiliary head (copy of baseline head) on shared features'}.get(strategy,
                           'main head' if strategy != 'ce_rank_fixed' else None),
        source_loss='CE + ranking_weight*mean softplus(real_margin-fake_margin) on clean+smoothed FF++ pairs, main head',
        target_pool='pool v2 adapt-train, label-free columns', pool_v2_membership_sha256=summary['membership_sha256'],
        schedule='fixed epochs (ramp + full-weight epochs, set before results); primary = final epoch; best_ffpp.pt secondary',
        labels_used_for='logging pseudo-label quality and adapt-val metrics only',
        bn_statistics='frozen', inference_head='main',
        versions={p: engine.metadata.version(p) for p in ['torch', 'torchvision', 'Pillow', 'numpy', 'pandas', 'scikit-learn']},
        torch=str(torch.__version__), code_sha256=code_hashes())


def prepare(args):
    uda.prepare(args)
    print(f'Fixed schedule: {args.epochs} epochs (ramp {args.ramp_epochs:g}); strategies {args.strategies}', flush=True)


def train(args):
    source, val, adapt_val, unlabelled, membership, cfg, data, summary = uda.inputs(args)
    truth = uda.truth_lookup(membership)
    common.prepare_pixel_cache(args, cfg, {k: data[k] for k in ['source_trainval_sha256']})
    common.prepare_validation(args, val, cfg, data={k: data[k] for k in
        ['source_trainval_sha256', 'source_protocol_sha256', 'val_paths_sha256', 'validation_images']})
    args._pool_cache = args._target_cache = pool.target_cache(args, cfg, 'pool_v2_observed', data['pool_v2_membership_sha256'])
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
                if done['identity'] != identity or not all((run/f).is_file() for f in CHECKPOINTS.values()):
                    raise ValueError('Completed run mismatch')
                print(f'===== ALREADY COMPLETED: {name} / {strategy} (skip) =====', flush=True)
                continue
            base.common.seed_all(args.seed)
            model, transform = base.build_model(name, pretrained=False)
            model.load_state_dict(initial['model_state_dict'], strict=True)
            model.to(args.device)
            blocks = 2 if name == 'xception' else initial['config']['train_blocks']
            common.training_mode(model, name, blocks)
            head = base.head(model, name)
            teacher = None if strategy == 'ce_rank_fixed' else copy.deepcopy(model).requires_grad_(False).eval()
            aux = copy.deepcopy(head).to(args.device).requires_grad_(True).train() if strategy == 'ts_decoupled_head' else None
            head_ids = {id(p) for p in head.parameters()}
            groups = [{'params': [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids], 'lr': args.encoder_lr},
                      {'params': list(head.parameters())+(list(aux.parameters()) if aux is not None else []), 'lr': args.head_lr}]
            optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
            parameters = [p for group in groups for p in group['params']]
            start, history = 0, []
            if (run/'last.pt').is_file():
                state = torch.load(run/'last.pt', map_location='cpu', weights_only=True)
                if state['identity'] != identity:
                    raise ValueError('Resume identity differs')
                model.load_state_dict(state['model_state_dict'], strict=True)
                if teacher is not None:
                    teacher.load_state_dict(state['teacher_state_dict'], strict=True)
                if aux is not None:
                    aux.load_state_dict(state['aux_state_dict'], strict=True)
                optimizer.load_state_dict(state['optimizer'])
                start, history, best = state['epoch'], state['history'], state['best_ffpp']
                del state
                print(f'===== RESUME: {name} / {strategy} from completed epoch {start}/{args.epochs} | {run} =====', flush=True)
            else:
                print(f'===== START NEW: {name} / {strategy} (no last.pt) | {run} =====', flush=True)
                values = validate(args, name, model, teacher, aux, val, adapt_val, transform, cfg)
                best = pool.best_of(values['ffpp'])
                base.common.json_write(run/'epoch0_validation.json', values)
                base.common.save(run/'best_ffpp.pt', dict(identity=identity, config=config, model_state_dict=model.state_dict(),
                    epoch=0, best=best, validation=values))
            for epoch in range(start, args.epochs):
                seed = args.seed+epoch*100003
                base.common.seed_all(seed)
                common.training_mode(model, name, blocks)
                if aux is not None:
                    aux.train()
                sampled = adaptation.balanced_source(source, args.source_images, seed)
                batches = rank.balanced_batches(torch.tensor(sampled.class_index.astype(int).to_numpy()), args.batch_size, seed)
                source_loader = DataLoader(common.PairedSource(sampled, args, transform, cfg, epoch), batch_sampler=batches,
                    num_workers=args.workers, pin_memory=args.device == 'cuda', persistent_workers=False,
                    generator=torch.Generator().manual_seed(seed))
                engine.freeze_csv(args.manifest_root/f'source_epoch{epoch+1:02d}.csv', sampled)
                plan = engine.target_plan(unlabelled, len(sampled), seed)
                engine.freeze_csv(args.manifest_root/'target_exposures'/f'epoch{epoch+1:02d}.csv', pd.DataFrame(plan))
                target_loader = None if teacher is None else iter(base.common.make_loader(engine.TargetViews(plan, args, transform, cfg, False), args, False, seed))
                offset, totals, confusion = 0, np.zeros(4), np.zeros((2, 2), dtype=np.int64)
                for step, (clean, smooth, labels_s, _, _) in enumerate(tqdm(source_loader,
                        desc=f'{name} {strategy} {epoch+1}/{args.epochs}', mininterval=15, dynamic_ncols=True)):
                    base.common.seed_all(seed+step)
                    clean, smooth, labels_s = clean.to(args.device), smooth.to(args.device), labels_s.to(args.device)
                    n = len(labels_s)
                    teacher_logits = None
                    if target_loader is not None:
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
                        target_loss, weight, loss = source_loss.new_zeros(()), 0., source_loss
                        if teacher_logits is not None:
                            student_logits, features = adaptation.outputs(name, model, student_view.to(args.device))
                            if aux is not None:
                                student_logits = aux(features)
                            target_loss, pseudo, selected = uda.confidence_loss(student_logits, teacher_logits, args.pseudo_threshold)
                            weight = args.target_weight*min(1., (epoch+(step+1)/len(source_loader))/args.ramp_epochs)
                            loss = source_loss+weight*target_loss
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite loss; last completed epoch preserved')
                    loss.backward()
                    nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                    optimizer.step()
                    if strategy in ('ts_ema', 'ts_decoupled_head'):
                        engine.update_teacher(teacher, model, args.ema_decay)
                    if teacher_logits is not None:  # logging only: labels never enter the objective
                        for row, p, s in zip(plan[offset:offset+n], pseudo.tolist(), selected.tolist()):
                            if s:
                                confusion[truth[row['crop_path']], p] += 1
                    totals += n*np.array([loss.item(), ce.item(), pair_rank.item(), target_loss.item()])
                    offset += n
                values = validate(args, name, model, teacher, aux, val, adapt_val, transform, cfg)
                improved = pool.is_better(values['ffpp'], best)
                if improved:
                    best = pool.best_of(values['ffpp'])
                chosen = int(confusion.sum())
                entry = dict(epoch=epoch+1, loss=float(totals[0]/offset), source_ce=float(totals[1]/offset),
                    source_rank=float(totals[2]/offset), target_ce=float(totals[3]/offset),
                    pseudo_selected=dict(real=int(confusion[:, 0].sum()), fake=int(confusion[:, 1].sum()), fraction=chosen/offset),
                    pseudo_accuracy=float(np.trace(confusion)/chosen) if chosen else None,
                    teacher_false_fake=int(confusion[0, 1]), teacher_false_real=int(confusion[1, 0]),
                    target_condition_counts=dict(Counter(p['condition'] for p in plan)) if teacher is not None else {},
                    source_paths_sha256=evaluation.key(sampled.crop_path.tolist()), source_batches_sha256=evaluation.key(batches),
                    target_plan_sha256=evaluation.key(plan) if teacher is not None else None, validation=values)
                history.append(entry)
                state = dict(identity=identity, config=config, model_state_dict=model.state_dict(),
                    teacher_state_dict=teacher.state_dict() if teacher is not None else None,
                    aux_state_dict=aux.state_dict() if aux is not None else None, optimizer=optimizer.state_dict(),
                    epoch=epoch+1, history=history, best_ffpp=best)
                if improved:
                    base.common.save(run/'best_ffpp.pt', {k: state[k] for k in ['identity', 'config', 'model_state_dict', 'epoch']}
                        | dict(best=best, validation=values))
                base.common.save(run/'last.pt', state)
                del state
                base.common.json_write(run/'history.json', history)
                f, t, free = values['ffpp'], values['adapt_val'], values['label_free']
                accuracy = f'{entry["pseudo_accuracy"]:.1%}' if chosen else 'n/a'
                print(f'epoch {epoch+1}/{args.epochs} CE={entry["source_ce"]:.4f} targetCE={entry["target_ce"]:.4f} | '
                    f'pseudo acc={accuracy} false fake={entry["teacher_false_fake"]} false real={entry["teacher_false_real"]} | '
                    f'FF++ pooled={f["pooled_auc"]:.4f} | adapt-val pooled={t["pooled_auc"]:.4f} (log) | '
                    f'entropy={free["mean_entropy"]:.3f} SND={free["snd"]:.3f}', flush=True)
            final = torch.load(run/'last.pt', map_location='cpu', weights_only=True)
            base.common.save(run/'final.pt', {k: final[k] for k in ['identity', 'config', 'model_state_dict', 'epoch']}
                | dict(validation=final['history'][-1]['validation']))
            selected = torch.load(run/'best_ffpp.pt', map_location='cpu', weights_only=True)
            base.common.json_write(run/'completed.json', dict(identity=identity, epochs_completed=len(history),
                final_epoch=final['epoch'], best_ffpp_epoch=selected['epoch'], best_ffpp=best,
                inference_head='main', target_labels_used_for_training=False))
            del final, selected, model, teacher, aux, optimizer, parameters, groups, head
            if args.device == 'cuda':
                torch.cuda.empty_cache()
        del initial


def evaluate(args):
    ea = argparse.Namespace(data_root=args.target_root, manifest=args.target_root/'manifests/celebdf_v2_roi16_test_v1.csv',
        protocol=args.protocol, cache_root=args.cache_root, output_root=args.output_root, workers=args.workers)
    frame, cfg, provenance = evaluation.inputs(ea)
    _, _, _, _, membership, _, _, _ = uda.inputs(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    engine.freeze_json(args.output_root/'evaluation_protocol.json', dict(provenance, used_for_selection=False))
    engine.freeze_csv(args.output_root/'evaluation_manifest.csv', frame)
    view_cache = args.cache_root if (args.cache_root/'config.json').is_file() and json.loads((args.cache_root/'config.json').read_text()) == provenance else None
    native_cache = common.pixel_cache.BilateralCache(args.evaluation_pixel_cache, cfg['bilateral'],
        dict(data=provenance, cache_code_sha256=base.common.digest(Path(common.pixel_cache.__file__)),
             storage='native uint8 RGB bilateral only; lossless PNG'))
    labels = frame.class_index.astype(int).to_numpy()
    identity = uda.identity_groups(frame, membership)
    eval_args = copy.copy(args); eval_args.batch_size = args.eval_batch_size
    records, mixtures, cross, groups, bootstrap, pseudo, selections = [], [], [], [], [], [], []
    for name in args.models:
        initial, initial_hash = evaluation.checkpoint(argparse.Namespace(runs_root=args.baseline_runs_root, seed=args.seed), name)
        states = [('baseline', initial, initial_hash)]
        for strategy in args.strategies:
            run = args.runs_root/run_name(name, strategy, args.seed)
            config = json.loads((run/'config.json').read_text())
            done = json.loads((run/'completed.json').read_text())
            if (done['identity'] != evaluation.key(config) or config['model'] != name or config['initial_checkpoint_sha256'] != initial_hash
                    or config['data']['test_manifest_sha256'] != provenance['manifest_sha256'] or config['code_sha256'] != code_hashes()):
                raise ValueError(f'Checkpoint/evaluation mismatch: {run}')
            for key, file in CHECKPOINTS.items():
                state = torch.load(run/file, map_location='cpu', weights_only=True)
                if state['identity'] != done['identity']:
                    raise ValueError(f'Checkpoint identity differs: {run/file}')
                label = strategy if key == 'final' else f'{strategy}@ffpp'
                states.append((label, state, base.common.digest(run/file)))
                selections.append(dict(model=name, strategy=label, epoch=state['epoch'], epochs_completed=done['epochs_completed']))
            for entry in json.loads((run/'history.json').read_text()):
                v = entry['validation']
                pseudo.append(dict(model=name, strategy=strategy, epoch=entry['epoch'], pseudo_accuracy=entry['pseudo_accuracy'],
                    selected_real=entry['pseudo_selected']['real'], selected_fake=entry['pseudo_selected']['fake'],
                    teacher_false_fake=entry['teacher_false_fake'], teacher_false_real=entry['teacher_false_real'],
                    ffpp_pooled_auc=v['ffpp']['pooled_auc'], adapt_val_pooled_auc=v['adapt_val']['pooled_auc'],
                    adapt_val_aux_pooled_auc=v.get('adapt_val_aux_head', {}).get('pooled_auc'),
                    **{f'free_{k}': x for k, x in v['label_free'].items()}))
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
                        probabilities, margins = previous[0].probability_fake.to_numpy(), previous[0].raw_margin.to_numpy()
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
            banks, _ = uda.find_reference(args.reference_roots, name, label, frame, provenance)
            if banks is None:
                print(f'Reference {label} for {name} not found; tables skip it', flush=True)
                continue
            metrics = [engine.raw_metrics(frame, 1/(1+np.exp(-np.clip(b, -50, 50))), b) for b in banks]
            banks_by[f'{label}(ref)'] = (banks, [dict(model=name, strategy=f'{label}(ref)', condition=c,
                **{k: m[k] for k in ['real_fpr', 'fake_fnr', 'balanced_error_rate', 'auc']}) for c, m in zip(CONDITIONS, metrics)])
        pooled = {}
        for label, (banks, rows) in banks_by.items():
            records += rows
            summary = pool.summarize(labels, banks, {r['condition']: r for r in rows})
            pooled[label] = summary['pooled_auc']
            row = dict(model=name, strategy=label, **{k: summary[k] for k in ['pooled_auc', 'mean_auc', 'min_cross_auc', 'strong_real_vs_normal_fake_auc']},
                       mean_fpr=float(np.mean([r['real_fpr'] for r in rows])), mean_fnr=float(np.mean([r['fake_fnr'] for r in rows])),
                       gap_closure_vs_sda=None)
            for group in ['seen', 'unseen']:
                fake, real = (labels == 1) & (identity == group), (labels == 0) & (identity == group)
                row[f'{group}_identity_pooled_auc'] = (pool.fast_auc(np.r_[np.zeros(4*real.sum()), np.ones(4*fake.sum())],
                    np.r_[np.concatenate([b[real] for b in banks]), np.concatenate([b[fake] for b in banks])]) if fake.any() and real.any() else None)
            mixtures.append(row)
            for i, real_condition in enumerate(CONDITIONS):
                for j, fake_condition in enumerate(CONDITIONS):
                    cross.append(dict(model=name, strategy=label, real_condition=real_condition, fake_condition=fake_condition, auc=summary['cross_auc'][i][j]))
            for condition, margins in zip(CONDITIONS, banks):
                groups.append(dict(model=name, strategy=label, condition=condition, **pool.subgroup_rows(frame, margins)))
        ceiling, control = pooled.get('sda_new_balanced(ref)'), pooled.get('ce_rank_fixed')
        if ceiling is not None and control is not None and ceiling != control:
            for row in mixtures:
                if row['model'] == name and row['strategy'] in args.strategies:
                    row['gap_closure_vs_sda'] = (pooled[row['strategy']]-control)/(ceiling-control)
        pairs = [(s, 'ce_rank_fixed') for s in args.strategies if s != 'ce_rank_fixed' and 'ce_rank_fixed' in banks_by]
        pairs += [(s, 'ts_ema') for s in ['ts_frozen_teacher', 'ts_decoupled_head'] if s in banks_by and 'ts_ema' in banks_by]
        for a, b in pairs:
            diffs = pool.video_bootstrap(frame, banks_by[a][0], banks_by[b][0], args.bootstrap, args.seed) if args.bootstrap else None
            for k, metric in enumerate(['pooled_auc', 'mean_auc']):
                observed = (pool.fast_auc(np.tile(labels, 4), np.concatenate(banks_by[a][0]))-pool.fast_auc(np.tile(labels, 4), np.concatenate(banks_by[b][0]))
                            if metric == 'pooled_auc' else float(np.mean([pool.fast_auc(labels, x) for x in banks_by[a][0]])-np.mean([pool.fast_auc(labels, x) for x in banks_by[b][0]])))
                bootstrap.append(dict(model=name, strategy=a, reference=b, metric=metric, observed=observed,
                    ci_low=float(np.quantile(diffs[:, k], .025)) if diffs is not None else None,
                    ci_high=float(np.quantile(diffs[:, k], .975)) if diffs is not None else None, repeats=args.bootstrap))
        del states, initial
    for filename, rows in [('comparison.csv', records), ('mixture_summary.csv', mixtures), ('cross_condition_auc.csv', cross),
                           ('subgroups.csv', groups), ('bootstrap.csv', bootstrap), ('pseudo_labels.csv', pseudo), ('selections.csv', selections)]:
        pd.DataFrame(rows).to_csv(args.output_root/filename, index=False)
    base.common.json_write(args.output_root/'comparison_protocol.json', dict(protocol=PROTOCOL,
        primary='final epoch of a fixed schedule', secondary='@ffpp = best FF++ val pooled AUC',
        references='(ref) rows are read-only saved predictions from earlier experiments',
        gap_closure='(method - ce_rank_fixed)/(sda_new_balanced(ref) - ce_rank_fixed)', test_used_for_selection=False,
        bootstrap='test video sampling only; not seed variance'))
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
    for field, default in [('seed', 42), ('epochs', 6), ('source-images', 6000), ('val-images', 1200), ('target-images', 5000),
                          ('batch-size', 16), ('eval-batch-size', 32), ('workers', 2), ('family-cap', 3), ('bootstrap', 1000)]:
        p.add_argument('--'+field, type=int, default=default)
    for field, default in [('encoder-lr', 1e-5), ('head-lr', 1e-5), ('weight-decay', 1e-4), ('ranking-weight', .25),
                          ('target-weight', .5), ('ramp-epochs', 2.), ('adapt-val-fraction', .2), ('ema-decay', .999),
                          ('pseudo-threshold', .9)]:
        p.add_argument('--'+field, type=float, default=default)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    a = p.parse_args(argv)
    a.hard_manifest_root = None
    numbers = [a.encoder_lr, a.head_lr, a.weight_decay, a.ranking_weight, a.target_weight, a.ramp_epochs,
               a.adapt_val_fraction, a.ema_decay, a.pseudo_threshold]
    if (not np.isfinite(numbers).all() or min(a.encoder_lr, a.head_lr, a.ramp_epochs) <= 0
            or min(a.weight_decay, a.ranking_weight, a.target_weight) < 0 or not 0 < a.adapt_val_fraction < 1
            or not 0 <= a.ema_decay < 1 or not .5 <= a.pseudo_threshold <= 1 or a.ramp_epochs > a.epochs
            or min(a.epochs, a.source_images, a.val_images, a.target_images, a.batch_size, a.family_cap) < 1
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
