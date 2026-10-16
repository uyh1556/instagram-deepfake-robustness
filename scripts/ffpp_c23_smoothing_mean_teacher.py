"""Matched source Aug and EMA teacher UDA; no target labels in sampling or loss."""
import argparse
import copy
import json
import random
from importlib import metadata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

import ffpp_c23_smoothing_pivot as common

base, adaptation, evaluation = common.base, common.adaptation, common.evaluation
ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'ffpp_c23_smoothing_mean_teacher_v1'
STRATEGIES = ['augmentation', 'mean_teacher', 'mean_teacher_smoothing']
CONDITIONS = common.CONDITIONS


def run_name(name, strategy, seed):
    return f'{name}_{strategy}_{PROTOCOL}_seed{seed}'


def code_hashes():
    modules = [Path(__file__), Path(common.__file__), Path(common.shared.__file__),
               Path(base.__file__), Path(base.common.__file__), Path(adaptation.__file__),
               Path(evaluation.__file__), Path(common.pixel_cache.__file__),
               ROOT/'scripts/smoothing_roi_pilot.py']
    return {p.name: base.common.digest(p) for p in modules}


def freeze_json(path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f'Existing protocol differs; use a new experiment directory: {path}')
    if not path.exists():
        base.common.json_write(path, value)


def freeze_csv(path, frame):
    content = frame.to_csv(index=False)
    if path.exists() and path.read_text() != content:
        raise ValueError(f'Frozen membership differs: {path}')
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def target_plan(target, n, seed):
    """Video-uniform replacement sampling, using ONLY these three columns."""
    records = target[['crop_path', 'video_id', 'roi_status']].to_dict('records')
    banks = {}
    for row in records:
        banks.setdefault(row['video_id'], []).append(row)
    videos = sorted(banks)
    if not videos:
        raise ValueError('No non-test target videos')
    rng = random.Random(seed+810001)
    plan = []
    for _ in range(n):
        row = rng.choice(banks[rng.choice(videos)])
        plan.append({**row, 'condition': rng.choice(CONDITIONS), 'flip': rng.random() < .5})
    return plan


class TargetViews(Dataset):
    def __init__(self, plan, args, transform, cfg, extra_smoothing):
        self.plan, self.args, self.transform, self.cfg = plan, args, transform, cfg
        self.extra_smoothing = extra_smoothing

    def __len__(self):
        return len(self.plan)

    def __getitem__(self, index):
        row = self.plan[index]
        with Image.open(self.args.target_root/row['crop_path']) as opened:
            image = opened.convert('RGB')
        observed = image
        if row['roi_status'] == 'detected' and row['condition'] != 'normal':
            observed = self.args._target_cache.smooth(image, row['crop_path'],
                self.cfg['conditions'][row['condition']])
        student = observed
        if self.extra_smoothing and row['roi_status'] == 'detected':
            # Reapply weak smoothing to the OBSERVED image, never look up its clean pair.
            student = self.args._target_probe_cache.smooth(observed,
                str(Path(row['condition'])/row['crop_path']), self.cfg['conditions']['smooth_weak'])
        teacher = observed
        if row['flip']:
            teacher = teacher.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        # The student uses the complementary flip. Strength/labels are not model inputs.
        if not row['flip']:
            student = student.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return self.transform(teacher), self.transform(student)


def pseudo_loss(student_logits, teacher_logits, threshold):
    probabilities = teacher_logits.detach().float().softmax(1)
    confidence, labels = probabilities.max(1)
    selected = confidence >= threshold
    per_image = nn.functional.cross_entropy(student_logits.float(), labels, reduction='none')
    # Divide by ALL target examples: few accepted labels cannot dominate the source loss.
    loss = (per_image*selected).mean()
    stats = torch.stack([(labels == y).sum() for y in [0, 1]] +
                        [((labels == y) & selected).sum() for y in [0, 1]])
    return loss, stats, confidence.sum()


@torch.no_grad()
def update_teacher(teacher, student, decay):
    for target, source in zip(teacher.parameters(), student.parameters()):
        if source.requires_grad:
            target.mul_(decay).add_(source.detach(), alpha=1.-decay)
        else:
            target.copy_(source)
    # BN statistics are frozen in student; copy buffers without averaging integer counters.
    for target, source in zip(teacher.buffers(), student.buffers()):
        target.copy_(source)
    teacher.eval()


def raw_metrics(frame, probabilities, margins):
    values = common.metrics(frame, probabilities)
    values['auc_probability'] = values['auc']
    values['auc'] = float(roc_auc_score(frame.class_index.astype(int), margins))
    # Make the fixed decision rule explicit even at FP32 softmax rounding boundaries.
    y, pred = frame.class_index.astype(int).to_numpy(), margins >= 0
    values['real_fpr'] = float(pred[y == 0].mean())
    values['fake_fnr'] = float((~pred[y == 1]).mean())
    values['balanced_error_rate'] = .5*(values['real_fpr']+values['fake_fnr'])
    return values


def validate(args, model, frame, transform):
    model.eval()
    records = {}
    for condition in CONDITIONS:
        probabilities, margins = infer(args, model, frame,
            common.ValidationViews(frame, args, transform, condition), f'Val {condition}')
        records[condition] = raw_metrics(frame, probabilities, margins)
    return {'mean_ber': float(np.mean([r['balanced_error_rate'] for r in records.values()])),
            'mean_auc': float(np.mean([r['auc'] for r in records.values()])), 'conditions': records}


def infer(args, model, frame, dataset, title):
    probabilities, margins = np.empty(len(frame)), np.empty(len(frame))
    loader = base.common.make_loader(dataset, args)
    with torch.inference_mode():
        for images, indexes in tqdm(loader, desc=title, mininterval=15, dynamic_ncols=True):
            logits = model(images.to(args.device)).float()
            if not torch.isfinite(logits).all():
                raise RuntimeError(f'Nonfinite logits: {title}')
            probabilities[indexes.numpy()] = logits.softmax(1)[:, 1].cpu().numpy()
            margins[indexes.numpy()] = (logits[:, 1]-logits[:, 0]).cpu().numpy()
    return probabilities, margins


def configuration(args, name, strategy, data, cfg, initial, initial_hash):
    return {'protocol': PROTOCOL, 'model': name, 'strategy': strategy, 'seed': args.seed,
        'data': data, 'smoothing': cfg, 'initial_checkpoint_sha256': initial_hash,
        'preprocessing': base.PREPROCESS[name], 'train_blocks': initial['config']['train_blocks'],
        'epochs': args.epochs, 'patience': args.patience, 'min_delta': .0001,
        'source_images': args.source_images, 'val_images': args.val_images,
        'target_images': args.target_images, 'batch_size': args.batch_size,
        'encoder_lr': args.encoder_lr, 'head_lr': args.head_lr, 'weight_decay': args.weight_decay,
        'ema_decay': args.ema_decay, 'pseudo_threshold': args.pseudo_threshold,
        'target_weight': args.target_weight, 'ramp_epochs': args.ramp_epochs,
        'loss': 'mean CE on source clean+random weak/medium/strong; UDA adds masked hard-pseudo-label CE divided by whole target batch, linear ramp',
        'target_sampling': 'GT-free video-uniform replacement from frozen non-test subset; no forced class prior',
        'target_views': 'teacher: observed normal/weak/medium/strong uniformly; student: complementary horizontal flip; smoothing strategy additionally re-smooths observed image at weak strength',
        'selection': 'STUDENT minimum four-condition FF++ mean BER @margin>=0; epoch0 eligible; no AUC guard; no target/test selection',
        'inference_model': 'student', 'bn_statistics': 'frozen',
        'precision': 'BF16 if supported else FP32 training; FP32 teacher/validation/test',
        'torch': str(torch.__version__), 'opencv': evaluation.cv2.__version__,
        'versions': {p: metadata.version(p) for p in ['torch','torchvision','Pillow','numpy','pandas','scikit-learn']},
        'baseline_versions': initial['config'].get('versions', {}), 'code_sha256': code_hashes()}


def train(args):
    pool, val, source_data = common.shared.source_inputs(args)
    _, other_val, target, cfg, target_data, blocked = adaptation.dataset_inputs(args)
    if list(cfg['conditions']) != CONDITIONS or val.crop_path.tolist() != other_val.crop_path.tolist():
        raise ValueError('Source validation or smoothing differs')
    target = target[['crop_path', 'video_id', 'roi_status']]
    data = {**source_data, **target_data}
    args.manifest_root.mkdir(parents=True, exist_ok=True)
    freeze_json(args.manifest_root/'protocol.json', {'protocol': PROTOCOL, 'data': data,
        'source_images': args.source_images, 'target_images': args.target_images,
        'seed': args.seed, 'target_labels_used': False, 'excluded_video_ids': blocked})
    freeze_csv(args.manifest_root/'source_validation.csv', val)
    freeze_csv(args.manifest_root/'target_unlabelled.csv', target)
    common.prepare_pixel_cache(args, cfg, source_data)
    target_identity = {'target_paths_sha256': data['target_paths_sha256'], 'smoothing': cfg,
        'opencv': evaluation.cv2.__version__, 'code_sha256': code_hashes()}
    args._target_cache = common.pixel_cache.BilateralCache(args.target_train_cache/'observed', cfg['bilateral'], target_identity)
    args._target_probe_cache = common.pixel_cache.BilateralCache(args.target_train_cache/'extra_weak', cfg['bilateral'], target_identity)
    common.prepare_validation(args, val, cfg, source_data)
    print(f'Source {args.source_images}/epoch; target pool {len(target)} images/{target.video_id.nunique()} videos; GT-free sampling', flush=True)
    for name in args.models:
        initial, initial_hash = evaluation.checkpoint(argparse.Namespace(runs_root=args.baseline_runs_root, seed=args.seed), name)
        if (initial['config']['data_hashes']['trainval'] != data['source_trainval_sha256']
                or initial['config']['data_hashes']['protocol'] != data['source_protocol_sha256']):
            raise ValueError('Baseline source manifest mismatch')
        for strategy in args.strategies:
            config = configuration(args, name, strategy, data, cfg, initial, initial_hash)
            identity = evaluation.key(config)
            run = args.runs_root/run_name(name, strategy, args.seed)
            run.mkdir(parents=True, exist_ok=True)
            freeze_json(run/'config.json', config)
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
            teacher = None if strategy == 'augmentation' else copy.deepcopy(model).requires_grad_(False).eval()
            head_params = list(base.head(model, name).parameters())
            ids = {id(p) for p in head_params}
            groups = [{'params': [p for p in model.parameters() if p.requires_grad and id(p) not in ids], 'lr': args.encoder_lr},
                      {'params': head_params, 'lr': args.head_lr}]
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
                start, stale, history, best, anchor = [state[k] for k in ['epoch','stale','history','best','anchor']]
                del state
                print('Resume completed epoch:', start, flush=True)
            else:
                reference = validate(args, model, val, transform)
                best = anchor = reference['mean_ber']
                base.common.json_write(run/'epoch0_validation.json', reference)
                base.common.save(run/'best.pt', {'identity': identity, 'config': config,
                    'model_state_dict': model.state_dict(), 'epoch': 0, 'best': best, 'validation': reference})
            print(f'===== {name} / {strategy} / student selected by FF++ mean BER =====', flush=True)
            for epoch in range(start, args.epochs):
                if stale >= args.patience:
                    break
                seed = args.seed+epoch*100003
                base.common.seed_all(seed)
                common.training_mode(model, name, blocks)
                sampled = adaptation.balanced_source(pool, args.source_images, seed)
                source_loader = base.common.make_loader(common.PairedSource(sampled, args, transform, cfg, epoch), args, True, seed)
                plan = target_plan(target, len(sampled), seed)
                freeze_csv(args.manifest_root/f'target_exposures_epoch{epoch+1:02d}.csv', pd.DataFrame(plan))
                target_loader = None
                if teacher is not None:
                    target_loader = iter(base.common.make_loader(TargetViews(plan, args, transform, cfg,
                        strategy == 'mean_teacher_smoothing'), args, True, seed))
                totals, stats, count = np.zeros(4), np.zeros(4, dtype=np.int64), 0
                for step, (clean, changed, labels, _, _) in enumerate(tqdm(source_loader,
                        desc=f'{name} {strategy} epoch{epoch+1}/{args.epochs}', mininterval=15, dynamic_ncols=True)):
                    # Target forwards must not change the source dropout RNG in the next batch.
                    base.common.seed_all(seed+step)
                    clean, changed, labels = clean.to(args.device), changed.to(args.device), labels.to(args.device)
                    optimizer.zero_grad(set_to_none=True)
                    with base.amp_context(args):
                        logits = model(torch.cat([clean, changed])).float()
                        ce = nn.functional.cross_entropy(logits, labels.repeat(2))
                    target_loss, confidence_sum = ce.new_zeros(()), ce.new_zeros(())
                    ramp = min(1., (epoch+(step+1)/len(source_loader))/args.ramp_epochs)
                    if teacher is not None:
                        xt, xs = next(target_loader)
                        with torch.no_grad():
                            teacher_logits = teacher(xt.to(args.device)).float()
                        with base.amp_context(args):
                            student_logits = model(xs.to(args.device)).float()
                        if not torch.isfinite(teacher_logits).all() or not torch.isfinite(student_logits).all():
                            raise RuntimeError('Nonfinite target logits; last completed epoch preserved')
                        target_loss, batch_stats, confidence_sum = pseudo_loss(student_logits, teacher_logits, args.pseudo_threshold)
                        stats += batch_stats.cpu().numpy()
                    loss = ce+args.target_weight*ramp*target_loss
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite loss; last completed epoch preserved')
                    loss.backward()
                    nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                    optimizer.step()
                    if teacher is not None:
                        update_teacher(teacher, model, args.ema_decay)
                    n = len(labels)
                    totals += [float(ce.detach())*n, float(target_loss.detach())*n,
                               float(confidence_sum.detach()), args.target_weight*ramp*n]
                    count += n
                values = validate(args, model, val, transform)
                score, improved = values['mean_ber'], values['mean_ber'] < best
                best = min(best, score)
                if score < anchor-.0001:
                    anchor, stale = score, 0
                else:
                    stale += 1
                entry = {'epoch': epoch+1, 'source_ce': float(totals[0]/count),
                    'target_ce': float(totals[1]/count), 'mean_target_weight': float(totals[3]/count),
                    'teacher_mean_confidence': float(totals[2]/count),
                    'teacher_predicted_counts': dict(zip(['real','fake'], stats[:2].tolist())),
                    'pseudo_selected_counts': dict(zip(['real','fake'], stats[2:].tolist())),
                    'pseudo_selected_fraction': float(stats[2:].sum()/count),
                    'target_condition_counts': dict(Counter(p['condition'] for p in plan)),
                    'target_exposures': count if teacher is not None else 0,
                    'source_paths_sha256': evaluation.key(sampled.crop_path.tolist()),
                    'target_plan_sha256': evaluation.key(plan) if teacher is not None else None, 'validation': values}
                history.append(entry)
                state = {'identity': identity, 'config': config, 'model_state_dict': model.state_dict(),
                    'teacher_state_dict': teacher.state_dict() if teacher is not None else None,
                    'optimizer': optimizer.state_dict(), 'epoch': epoch+1, 'stale': stale,
                    'best': best, 'anchor': anchor, 'history': history}
                if improved:
                    base.common.save(run/'best.pt', {k: state[k] for k in ['identity','config','model_state_dict','epoch','best']} | {'validation': values})
                base.common.save(run/'last.pt', state)
                del state
                base.common.json_write(run/'history.json', history)
                print(f'CE={entry["source_ce"]:.4f} targetCE={entry["target_ce"]:.4f} | '
                    f'pseudo selected={entry["pseudo_selected_counts"]} ({entry["pseudo_selected_fraction"]:.1%}) | '
                    f'val BER={score:.2%} AUC={values["mean_auc"]:.4f} best={best:.2%} patience={stale}/{args.patience}', flush=True)
            selected = torch.load(run/'best.pt', map_location='cpu', weights_only=True)
            base.common.json_write(run/'completed.json', {'identity': identity, 'epochs_completed': len(history),
                'selected_epoch': selected['epoch'], 'best_mean_ber': best, 'inference_model': 'student'})
            del selected, teacher, model, optimizer, parameters, groups, head_params
            if args.device == 'cuda':
                torch.cuda.empty_cache()
        del initial


class TestViews(Dataset):
    def __init__(self, frame, args, transform, cfg, condition, native_cache, view_cache):
        self.rows, self.args, self.transform, self.cfg = frame.to_dict('records'), args, transform, cfg
        self.condition, self.native_cache, self.view_cache = condition, native_cache, view_cache

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.args.target_root/row['crop_path']) as opened:
            image = opened.convert('RGB')
        if self.condition != 'normal' and row['roi_status'] == 'detected':
            cached = self.view_cache/self.condition/row['crop_path'] if self.view_cache else None
            if cached is not None and cached.is_file():
                with Image.open(cached) as opened:
                    image = opened.convert('RGB')
            else:
                image = self.native_cache.smooth(image, row['crop_path'], self.cfg['conditions'][self.condition])
        return self.transform(image), index


def previous_baseline(args, name, condition, frame, provenance, checkpoint_hash):
    root = getattr(args, 'reference_evaluations', None)
    if root is None or not root.is_dir():
        return None
    for report in sorted(root.rglob(f'{name}_baseline_{condition}.json')):
        pred, protocol = report.with_suffix('.csv'), report.parent/'evaluation_protocol.json'
        if not pred.is_file() or not protocol.is_file():
            continue
        saved_report = json.loads(report.read_text())
        old = json.loads(protocol.read_text())
        if (any(old.get(k) != v for k, v in provenance.items())
                or saved_report.get('checkpoint_sha256') != checkpoint_hash
                or saved_report.get('manifest_sha256') != provenance['manifest_sha256']
                or saved_report.get('precision') != 'FP32'):
            continue
        saved = pd.read_csv(pred, dtype={'crop_path':str, 'class_index':str, 'video_id':str})
        if ('raw_margin' not in saved or saved.crop_path.tolist() != frame.crop_path.tolist()
                or saved.class_index.tolist() != frame.class_index.tolist()
                or saved.video_id.tolist() != frame.video_id.tolist()
                or not np.isfinite(saved.raw_margin).all()
                or not np.isfinite(saved.probability_fake).all()
                or not saved.probability_fake.between(0, 1).all()):
            continue
        print('Reuse baseline predictions:', pred, flush=True)
        return saved, pred
    return None


def evaluate(args):
    ea = argparse.Namespace(data_root=args.target_root, manifest=args.target_root/'manifests/celebdf_v2_roi16_test_v1.csv',
        protocol=args.protocol, cache_root=args.cache_root, output_root=args.output_root, workers=args.workers)
    frame, cfg, provenance = evaluation.inputs(ea)
    args.output_root.mkdir(parents=True, exist_ok=True)
    freeze_json(args.output_root/'evaluation_protocol.json', dict(provenance,
        excluded_from_adaptation_video_ids=sorted(frame.video_id.unique().tolist()),
        identity_overlap_allowed=True, source_parent_audit_complete=False))
    freeze_csv(args.output_root/'evaluation_manifest.csv', frame)
    view_cache = None
    if (args.cache_root/'config.json').is_file():
        if json.loads((args.cache_root/'config.json').read_text()) != provenance:
            raise ValueError('CDF view cache provenance differs')
        view_cache = args.cache_root
    # Same identity as Ranking's cache: reuse the expensive native bilateral operation.
    native_cache = common.pixel_cache.BilateralCache(args.evaluation_pixel_cache, cfg['bilateral'],
        dict(data=provenance, cache_code_sha256=base.common.digest(Path(common.pixel_cache.__file__)),
             storage='native uint8 RGB bilateral only; lossless PNG'))
    records, mixtures, cross = [], [], []
    for name in args.models:
        initial, initial_hash = evaluation.checkpoint(argparse.Namespace(runs_root=args.baseline_runs_root, seed=args.seed), name)
        states = [('baseline', initial, initial_hash)]
        for strategy in args.strategies:
            run = args.runs_root/run_name(name, strategy, args.seed)
            config = json.loads((run/'config.json').read_text())
            done = json.loads((run/'completed.json').read_text())
            path = run/'best.pt'
            state = torch.load(path, map_location='cpu', weights_only=True)
            if (state['config'] != config or state['identity'] != evaluation.key(config)
                    or done['identity'] != state['identity'] or config['protocol'] != PROTOCOL
                    or config['model'] != name or config['strategy'] != strategy or config['seed'] != args.seed
                    or config['initial_checkpoint_sha256'] != initial_hash or config['smoothing'] != cfg
                    or config['preprocessing'] != base.PREPROCESS[name]
                    or config['data']['test_manifest_sha256'] != provenance['manifest_sha256']
                    or config['code_sha256'] != code_hashes()):
                raise ValueError(f'Checkpoint/evaluation mismatch: {run}')
            states.append((strategy, state, base.common.digest(path)))
        for strategy, state, checkpoint_hash in states:
            identity = evaluation.key({'data': provenance, 'checkpoint': checkpoint_hash,
                'precision': 'FP32', 'code_sha256': code_hashes(), 'torch': str(torch.__version__)})
            model, transform = None, None
            banks = []
            for condition in CONDITIONS:
                prefix = args.output_root/f'{name}_{strategy}_{condition}'
                report, pred = prefix.with_suffix('.json'), prefix.with_suffix('.csv')
                if report.is_file() and pred.is_file():
                    values = json.loads(report.read_text())
                    saved = pd.read_csv(pred, dtype={'crop_path':str,'video_id':str,'class_index':str})
                    if (values['identity'] != identity or saved.crop_path.tolist() != frame.crop_path.tolist()
                            or saved.video_id.tolist() != frame.video_id.tolist()
                            or saved.class_index.tolist() != frame.class_index.tolist()
                            or not np.isfinite(saved.raw_margin).all()):
                        raise ValueError('Saved evaluation differs; use a new output root')
                    print('Reuse:', name, strategy, condition, flush=True)
                else:
                    previous = previous_baseline(args, name, condition, frame, provenance, checkpoint_hash) if strategy == 'baseline' else None
                    if strategy != 'baseline' and state['epoch'] == 0:
                        baseline_path = args.output_root/f'{name}_baseline_{condition}.csv'
                        previous = (pd.read_csv(baseline_path), baseline_path)
                    if previous is not None:
                        old_predictions, old_path = previous
                        probabilities, margins = old_predictions.probability_fake.to_numpy(), old_predictions.raw_margin.to_numpy()
                    else:
                        if model is None:
                            model, transform = base.build_model(name, pretrained=False)
                            model.load_state_dict(state['model_state_dict'], strict=True)
                            model.to(args.device).eval()
                        probabilities, margins = infer(args, model, frame,
                            TestViews(frame, args, transform, cfg, condition, native_cache, view_cache), f'{name} {strategy} {condition}')
                    values = raw_metrics(frame, probabilities, margins)
                    values.update(identity=identity, images=len(frame), precision='FP32', threshold=.5,
                        checkpoint_sha256=checkpoint_hash, manifest_sha256=provenance['manifest_sha256'],
                        selected_epoch=state.get('epoch', 0), unit='frame', inference_model='student' if strategy != 'baseline' else 'baseline')
                    if previous is not None:
                        values['reused_predictions'] = str(old_path)
                        values['reused_predictions_sha256'] = base.common.digest(old_path)
                    saved = frame[['crop_path','video_id','class_index','roi_status']].copy()
                    saved['probability_fake'], saved['raw_margin'] = probabilities, margins
                    temp = pred.with_suffix('.tmp.csv'); saved.to_csv(temp,index=False); temp.replace(pred)
                    base.common.json_write(report, values)
                banks.append(saved.raw_margin.to_numpy())
                records.append({'model':name, 'strategy':strategy, 'condition':condition,
                    **{k: values[k] for k in ['real_fpr','fake_fnr','balanced_error_rate','auc','auc_probability']}})
                print(f'{name} {strategy} {condition}: FPR={values["real_fpr"]:.2%} FNR={values["fake_fnr"]:.2%} '
                    f'BER={values["balanced_error_rate"]:.2%} AUC={values["auc"]:.4f}', flush=True)
            labels = frame.class_index.astype(int).to_numpy()
            mixtures.append({'model': name, 'strategy': strategy,
                'condition_mean_auc': float(np.mean([roc_auc_score(labels, bank) for bank in banks])),
                'pooled_auc': float(roc_auc_score(np.tile(labels,4), np.concatenate(banks))),
                'balanced_error_rate': float(np.mean([r['balanced_error_rate'] for r in records
                    if r['model'] == name and r['strategy'] == strategy]))})
            for i, real_condition in enumerate(CONDITIONS):
                for j, fake_condition in enumerate(CONDITIONS):
                    margins = np.concatenate([banks[i][labels == 0], banks[j][labels == 1]])
                    y = np.concatenate([np.zeros((labels == 0).sum()), np.ones((labels == 1).sum())])
                    cross.append({'model':name,'strategy':strategy,'real_condition':real_condition,
                        'fake_condition':fake_condition,'auc':float(roc_auc_score(y,margins))})
            del model
            if args.device == 'cuda':
                torch.cuda.empty_cache()
        del states, initial
    table = pd.DataFrame(records)
    table.to_csv(args.output_root/'comparison.csv', index=False)
    pd.DataFrame(mixtures).to_csv(args.output_root/'mixture_summary.csv', index=False)
    pd.DataFrame(cross).to_csv(args.output_root/'cross_condition_auc.csv', index=False)
    table.groupby(['model','strategy'], sort=False)[['real_fpr','fake_fnr','balanced_error_rate','auc','auc_probability']].mean().reset_index().to_csv(
        args.output_root/'condition_means.csv', index=False)
    print('Saved:', args.output_root, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['train','evaluate'])
    for field in ['source-root','target-root','baseline-runs-root','runs-root','manifest-root','validation-cache','output-root']:
        p.add_argument('--'+field, type=Path, required=True)
    p.add_argument('--source-validation', type=Path)
    p.add_argument('--reference-evaluations', type=Path, help='Reuse matching raw-margin baseline predictions from Ranking')
    p.add_argument('--train-cache', type=Path, default=Path('/content/smoothing_pivot_bilateral_cache_v1'))
    p.add_argument('--target-train-cache', type=Path, default=Path('/content/smoothing_mean_teacher_target_cache_v1'))
    p.add_argument('--cache-root', type=Path, default=Path('/content/celebdf_official_smoothing_cache_v1'))
    p.add_argument('--evaluation-pixel-cache', type=Path, default=Path('/content/smoothing_ranking_cdf_bilateral_cache_v1'))
    p.add_argument('--protocol', type=Path, default=ROOT/'configs/ffpp_c23_celebdf_smoothing_v1.json')
    p.add_argument('--models', nargs='+', choices=list(base.PREPROCESS), default=['clip_b16'])
    p.add_argument('--strategies', nargs='+', choices=STRATEGIES, default=STRATEGIES)
    for field, default in [('seed',42),('epochs',10),('patience',3),('source-images',6000),('val-images',1200),
                           ('target-images',5000),('batch-size',16),('workers',2)]:
        p.add_argument('--'+field, type=int, default=default)
    for field, default in [('encoder-lr',1e-5),('head-lr',1e-5),('weight-decay',1e-4),
                           ('ema-decay',.999),('pseudo-threshold',.9),('target-weight',.5),('ramp-epochs',2.)]:
        p.add_argument('--'+field, type=float, default=default)
    p.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    args = p.parse_args()
    if (min(args.epochs,args.patience,args.source_images,args.val_images,args.target_images,args.batch_size) < 1
            or args.source_images%8 or args.val_images%8 or args.workers<0
            or not 0 <= args.ema_decay < 1 or not .5 <= args.pseudo_threshold <= 1
            or args.target_weight<0 or args.ramp_epochs<=0 or min(args.encoder_lr,args.head_lr)<=0
            or args.weight_decay<0):
        p.error('Invalid counts or optimization settings')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime')
    (train if args.action == 'train' else evaluate)(args)


if __name__ == '__main__':
    main()
