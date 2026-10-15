"""C: continue trained B correction with bounded unlabeled-target consistency."""
import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import ffpp_c23_smoothing_preservation as b
import ffpp_c23_mixed_balanced_uda as mixed

base, evaluation = b.base, b.evaluation
PROTOCOL = 'ffpp_c23_smoothing_consistency_c_v1'
STRATEGY = 'preservation_c'


def run_name(name, seed):
    return f'{name}_{PROTOCOL}_seed{seed}'


def b_start(args, name):
    run = args.b_runs_root/b.run_name(name, args.seed)
    config = json.loads((run/'config.json').read_text())
    done = json.loads((run/'completed.json').read_text())
    best = torch.load(run/'best.pt', map_location='cpu', weights_only=True)
    if (best['epoch'] != done['selected_epoch'] or best['config'] != config
            or best['identity'] != evaluation.key(config) or done['identity'] != best['identity']):
        raise ValueError('B best/completion epoch mismatch')
    filename = 'last.pt' if best['epoch'] == 0 else 'best.pt'
    state = torch.load(run/filename, map_location='cpu', weights_only=True) if filename == 'last.pt' else best
    if (state['config'] != config or state['identity'] != evaluation.key(config)
            or done['identity'] != state['identity'] or config['model'] != name or config['seed'] != args.seed
            or config['protocol'] != b.PROTOCOL or config['strategy'] != b.STRATEGY
            or config['code_sha256'] != base.common.digest(Path(b.__file__))
            or config['model_code_sha256'] != base.common.digest(Path(base.__file__))
            or config['shared_code_sha256'] != base.common.digest(Path(base.common.__file__))
            or config['smoothing_code_sha256'] != base.common.digest(b.ROOT/'scripts/smoothing_roi_pilot.py')
            or state['epoch'] <= 0
            or (filename == 'last.pt' and state['epoch'] != done['epochs_completed'])):
        raise ValueError(f'Invalid trained B starting checkpoint: {run/filename}')
    metadata = {'checkpoint_file': filename, 'epoch': state['epoch'], 'b_selected_epoch': best['epoch'],
                'checkpoint_sha256': base.common.digest(run/filename), 'b_config': config}
    return state, metadata


class TargetPairs(torch.utils.data.Dataset):
    def __init__(self, source, target, args, transform, epoch, cfg):
        self.args, self.transform, self.cfg = args, transform, cfg
        self.sampler = mixed.MixedViews(source, target, args, transform, 'uda_aug_mixed_balanced', epoch, cfg)

    def __len__(self):
        return len(self.sampler.plan)

    def __getitem__(self, index):
        row, condition, flip = self.sampler.plan[index]
        with Image.open(self.args.target_root/row['crop_path']) as image:
            observed = image.convert('RGB')
            detected = row['roi_status'] == 'detected'
            if detected and condition != 'normal':
                observed = b.adaptation.smooth(observed, self.cfg['conditions'][condition], self.cfg['bilateral'])
            extra = b.adaptation.smooth(observed, self.args.extra_strength, self.cfg['bilateral']) if detected else observed.copy()
            if flip:
                observed = observed.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                extra = extra.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            # No real/fake label, original counterpart, or smoothing metadata returned.
            return self.transform(observed), self.transform(extra)

    def metadata(self):
        rows = self.sampler.exposure_metadata()
        for record, (_, _, flip) in zip(rows, self.sampler.plan):
            record['flip'] = flip
            record['extra_strength'] = self.args.extra_strength
        return rows


def target_features(args, name, model, dataset, initial_hash, epoch):
    plan = dataset.metadata()
    identity = {'checkpoint': initial_hash, 'plan': evaluation.key(plan), 'epoch': epoch,
                'torch': str(torch.__version__), 'device': args.device, 'precision': 'FP32',
                'smoothing': dataset.cfg, 'opencv': evaluation.cv2.__version__,
                'code': base.common.digest(Path(__file__)), 'b_code': base.common.digest(Path(b.__file__))}
    path = args.feature_cache_root/name/f'c_target_{evaluation.key(identity)}.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved['identity'] != identity:
            raise ValueError('C target feature cache mismatch')
        print('Reuse target features:', name, epoch+1, flush=True)
        return saved['features'], saved['margins']
    loader = base.common.make_loader(dataset, args, False, args.seed)
    fs, ms = [], []
    with torch.no_grad():
        for first, second in tqdm(loader, desc=f'{name} C target features {epoch+1}', mininterval=10):
            n = len(first)
            logits, f = b.adaptation.outputs(name, model, torch.cat([first, second]).to(args.device))
            m = logits[:, 1]-logits[:, 0]
            fs.append(torch.stack([f[:n], f[n:]], 1).float().cpu())
            ms.append(torch.stack([m[:n], m[n:]], 1).float().cpu())
    features, margins = torch.cat(fs), torch.cat(ms)
    if not torch.isfinite(features).all() or not torch.isfinite(margins).all():
        raise RuntimeError('Nonfinite C target features')
    base.common.save(path, {'identity': identity, 'features': features, 'margins': margins})
    return features, margins


def loss_for(student, teacher, sf, sm, labels, tf, tm, args):
    source, cls, restore, clean = b.loss_for(student, sf, sm, labels, args.pair_weight, args.clean_weight)
    observed = student(tf[:, 0], tm[:, 0])
    extra = student(tf[:, 1], tm[:, 1])
    consistency = nn.functional.smooth_l1_loss(observed/student.scale, extra/student.scale)
    with torch.no_grad():
        reference = teacher(tf[:, 0], tm[:, 0])
    trust = nn.functional.smooth_l1_loss(observed/student.scale, reference/student.scale)
    return source + args.target_weight*consistency + args.trust_weight*trust, cls, restore, clean, consistency, trust


def train(args):
    pool, val, target, smoothing, data = mixed.sda.inputs(args)
    args.manifest_root.mkdir(parents=True, exist_ok=True)
    info = {**data, 'target_label_usage': 'ground truth only for exact 1:1 exposure sampling; never a target task loss',
            'observed_views': 'normal/weak/medium/strong each25%; no per-image condition supplied to model',
            'pair': 'observed target + additional weak smoothing; not a clean target counterpart',
            'extra_strength': args.extra_strength}
    path = args.manifest_root/'protocol.json'
    if path.exists() and json.loads(path.read_text()) != info:
        raise ValueError('C manifest protocol changed; use a new output root')
    base.common.json_write(path, info)
    target.drop(columns='class_index').to_csv(args.manifest_root/'target_unlabelled.csv', index=False)
    val.to_csv(args.manifest_root/'source_validation.csv', index=False)
    # Verify every requested B checkpoint before expensive feature extraction.
    for name in args.models:
        state, meta = b_start(args, name)
        if (meta['b_config']['data']['val_paths_sha256'] != data['val_paths_sha256']
                or meta['b_config']['data']['source_trainval_sha256'] != data['source_trainval_sha256']
                or meta['b_config']['smoothing'] != smoothing):
            raise ValueError('B/C source validation or smoothing protocol differs')
        print(f'C start {name}: B {meta["checkpoint_file"]} epoch{meta["epoch"]}', flush=True)
        del state
    for name in args.models:
        start_state, start_meta = b_start(args, name)
        bc = start_meta['b_config']
        initial, initial_hash = evaluation.checkpoint(argparse.Namespace(runs_root=args.baseline_runs_root, seed=args.seed), name)
        if bc['initial_checkpoint_sha256'] != initial_hash:
            raise ValueError('B/C original baseline differs')
        config = {'protocol': PROTOCOL, 'strategy': STRATEGY, 'model': name, 'seed': args.seed,
                  'data': info, 'b_start': start_meta, 'initial_checkpoint_sha256': initial_hash,
                  'preprocessing': base.PREPROCESS[name], 'smoothing': smoothing,
                  'source_images_per_epoch': args.source_images, 'target_exposures_per_epoch': args.source_images,
                  'image_views_per_epoch': 4*args.source_images,
                  'epochs': args.epochs, 'patience': args.patience, 'batch_size': args.batch_size,
                  'learning_rate': args.learning_rate, 'weight_decay': args.weight_decay,
                  'pair_weight': args.pair_weight, 'clean_weight': args.clean_weight,
                  'target_weight': args.target_weight, 'trust_weight': args.trust_weight,
                  'extra_strength': args.extra_strength, 'backbone': 'frozen encoder and original classifier; FP32',
                  'loss': 'B source loss + target_weight*normalized Huber(target pair) + trust_weight*normalized Huber(observed target, fixed B observed score)',
                  'selection': {'reference': 'chosen trained B initializer, not zero-correction baseline',
                                'metric': 'source four-condition minimum BER, C epoch0 eligible',
                                'clean_ber_tolerance': args.clean_ber_tolerance,
                                'clean_error_tolerance': args.clean_error_tolerance, 'auc_tolerance': args.auc_tolerance},
                  'optimizer': 'new AdamW for C stage; B optimizer not resumed',
                  'code_sha256': base.common.digest(Path(__file__)), 'b_code_sha256': base.common.digest(Path(b.__file__)),
                  'mixed_code_sha256': base.common.digest(Path(mixed.__file__)),
                  'torch': str(torch.__version__), 'opencv': evaluation.cv2.__version__,
                  'versions': {p: b.importlib.metadata.version(p) for p in bc.get('versions', {}) if bc['versions'][p] is not None}}
        identity = evaluation.key(config)
        run = args.runs_root/run_name(name, args.seed)
        run.mkdir(parents=True, exist_ok=True)
        if (run/'config.json').exists() and json.loads((run/'config.json').read_text()) != config:
            raise ValueError('C run configuration differs')
        if (run/'completed.json').exists():
            done = json.loads((run/'completed.json').read_text())
            if done['identity'] != identity or not (run/'best.pt').is_file():
                raise ValueError('C completion mismatch')
            print('Skip completed C:', name, flush=True)
            del initial, start_state
            continue
        base.common.json_write(run/'config.json', config)
        base.common.seed_all(args.seed)
        model, transform = b.frozen_model(args, name, initial)
        del initial
        student = b.Correction(b.adaptation.FEATURES[name], bc['head']['hidden']).to(args.device)
        student.load_state_dict(start_state['correction_state_dict'], strict=True)
        teacher = copy.deepcopy(student).requires_grad_(False).eval()
        del start_state
        optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        vf, vm = b.features_for(args, name, model, transform, val, smoothing, initial_hash)
        reference = b.validate(teacher, vf, vm, val, smoothing, args.device)
        start, stale, history, best, anchor = 0, 0, [], reference['mean_ber'], reference['mean_ber']
        if (run/'last.pt').exists():
            saved = torch.load(run/'last.pt', map_location='cpu', weights_only=True)
            if saved['identity'] != identity:
                raise ValueError('C resume identity mismatch')
            student.load_state_dict(saved['correction_state_dict'], strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            start, stale, history, best, anchor = [saved[k] for k in ['epoch', 'stale', 'history', 'best', 'anchor']]
            print('Resume C after epoch:', start, flush=True)
            del saved
        else:
            base.common.save(run/'best.pt', {'identity': identity, 'config': config,
                'correction_state_dict': student.state_dict(), 'epoch': 0, 'best': best,
                'reference': reference, 'validation': reference})
            base.common.json_write(run/'b_start_source_validation.json', reference)
        print(f'{name} C epoch0 = trained B epoch{start_meta["epoch"]}; source BER={reference["mean_ber"]:.2%}', flush=True)
        for epoch in range(start, args.epochs):
            if stale >= args.patience:
                break
            seed = args.seed+epoch*100003
            base.common.seed_all(seed)
            sampled = b.adaptation.balanced_source(pool, args.source_images, seed)
            sf, sm = b.features_for(args, name, model, transform, sampled, smoothing, initial_hash, epoch)
            target_views = TargetPairs(sampled, target, args, transform, epoch, smoothing)
            plan = target_views.metadata()
            pd.DataFrame(plan).to_csv(run/f'target_exposures_epoch{epoch+1:02d}.csv', index=False)
            sampled.to_csv(run/f'source_epoch{epoch+1:02d}.csv', index=False)
            counts = Counter(r['sampler_class_index'] for r in plan)
            print(f'Target exposures real={counts[0]}, fake={counts[1]} | {dict(Counter(r["condition"] for r in plan))}', flush=True)
            tf, tm = target_features(args, name, model, target_views, initial_hash, epoch)
            labels = torch.tensor(sampled.class_index.astype(int).to_numpy(), dtype=torch.long)
            loader = base.common.make_loader(TensorDataset(sf, sm, labels, tf, tm), args, False, seed)
            student.train()
            totals, count = np.zeros(6), 0
            for values in tqdm(loader, desc=f'{name} C correction {epoch+1}/{args.epochs}', mininterval=10):
                values = [v.to(args.device) for v in values]
                optimizer.zero_grad(set_to_none=True)
                losses = loss_for(student, teacher, *values, args)
                if not torch.isfinite(losses[0]):
                    raise RuntimeError('Nonfinite C loss; last completed epoch preserved')
                losses[0].backward()
                nn.utils.clip_grad_norm_(student.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                totals += np.array([float(v.detach()) for v in losses])*len(values[2])
                count += len(values[2])
            validation = b.validate(student, vf, vm, val, smoothing, args.device)
            allowed = b.eligible(validation, reference, args)
            improved = allowed and validation['mean_ber'] < best-1e-12
            if improved:
                best = validation['mean_ber']
            if allowed and validation['mean_ber'] < anchor-1e-4:
                stale, anchor = 0, validation['mean_ber']
            else:
                stale += 1
            history.append({'epoch': epoch+1, 'loss': float(totals[0]/count),
                **{k: float(totals[i]/count) for i, k in enumerate(
                    ['classification_loss', 'restoration_loss', 'clean_anchor_loss', 'target_consistency_loss', 'target_trust_loss'], 1)},
                'eligible': allowed, 'validation': validation, 'target_real_exposures': counts[0], 'target_fake_exposures': counts[1],
                'source_paths_sha256': evaluation.key(sampled.crop_path.tolist()), 'target_plan_sha256': evaluation.key(plan)})
            saved = {'identity': identity, 'config': config, 'correction_state_dict': student.state_dict(),
                     'epoch': epoch+1, 'best': best, 'reference': reference, 'validation': validation}
            if improved:
                base.common.save(run/'best.pt', saved)
            base.common.save(run/'last.pt', {**saved, 'optimizer': optimizer.state_dict(), 'stale': stale, 'anchor': anchor, 'history': history})
            base.common.json_write(run/'history.json', history)
            clean = validation['normal']['frame']
            print(f'{name} C epoch{epoch+1}: BER={validation["mean_ber"]:.2%} best={best:.2%} '
                  f'clean FPR/FNR={clean["real_fpr"]:.2%}/{clean["fake_fnr"]:.2%} eligible={allowed} patience={stale}/{args.patience}', flush=True)
            del sf, sm, tf, tm, loader, target_views
        best_state = torch.load(run/'best.pt', map_location='cpu', weights_only=True)
        base.common.json_write(run/'completed.json', {'identity': identity, 'epochs_completed': len(history),
            'selected_epoch': best_state['epoch'], 'selected_source_mean_ber': best, 'b_start': start_meta})
        del model, student, teacher, optimizer, vf, vm, best_state
        if args.device == 'cuda':
            torch.cuda.empty_cache()


def evaluate(args):
    ea = argparse.Namespace(data_root=args.target_root, manifest=args.target_root/'manifests/celebdf_v2_roi16_test_v1.csv',
        protocol=args.protocol, cache_root=args.cache_root, output_root=args.output_root, workers=args.workers)
    evaluation.prepare(ea)
    frame, cfg, provenance = evaluation.inputs(ea)
    previous = json.loads((args.baseline_evaluation_root/'evaluation_protocol.json').read_text())
    if any(previous.get(k) != v for k, v in provenance.items()):
        raise ValueError('C and baseline test protocol differ')
    records = []
    for name in args.models:
        bs, meta = b_start(args, name)
        run = args.runs_root/run_name(name, args.seed)
        config = json.loads((run/'config.json').read_text())
        saved = torch.load(run/'best.pt', map_location='cpu', weights_only=True)
        done = json.loads((run/'completed.json').read_text())
        if (saved['config'] != config or saved['identity'] != evaluation.key(config) or done['identity'] != saved['identity']
                or config['model'] != name or config['seed'] != args.seed or config['protocol'] != PROTOCOL
                or config['b_start'] != meta or config['smoothing'] != cfg
                or config['code_sha256'] != base.common.digest(Path(__file__))):
            raise ValueError('C evaluation/checkpoint mismatch')
        initial, initial_hash = evaluation.checkpoint(argparse.Namespace(runs_root=args.baseline_runs_root, seed=args.seed), name)
        if initial_hash != config['initial_checkpoint_sha256']:
            raise ValueError('C initial baseline differs')
        identity = evaluation.key({'checkpoint': base.common.digest(run/'best.pt'), 'b_start': meta['checkpoint_sha256'],
                                   'test': provenance, 'code': config['code_sha256'], 'precision': 'FP32', 'torch': str(torch.__version__)})
        model, transform, student, teacher = None, None, None, None
        for condition in cfg['conditions']:
            original = json.loads((args.baseline_evaluation_root/f'{name}_{condition}.json').read_text())
            if (original['checkpoint_sha256'] != initial_hash or original['manifest_sha256'] != provenance['manifest_sha256']
                    or original['precision'] != 'FP32' or original['images'] != len(frame)):
                raise ValueError('Baseline test reference differs')
            records.append({'model': name, 'strategy': 'baseline', 'condition': condition,
                **{k: original[k] for k in b.METRICS if k != 'balanced_error_rate'},
                'balanced_error_rate': .5*(original['real_fpr']+original['fake_fnr'])})
            paths = {s: args.output_root/f'{name}_{s}_{condition}.json' for s in ['b_start', STRATEGY]}
            ready = all(p.is_file() and p.with_suffix('.csv').is_file() for p in paths.values())
            reports = {s: json.loads(p.read_text()) for s, p in paths.items()} if ready else {}
            if ready and any(r['identity'] != identity for r in reports.values()):
                raise ValueError('C saved evaluation identity differs')
            if ready:
                for path in paths.values():
                    pred = pd.read_csv(path.with_suffix('.csv'), dtype={'crop_path': str, 'class_index': str})
                    ready = ready and pred.crop_path.tolist() == frame.crop_path.tolist() and pred.class_index.tolist() == frame.class_index.tolist()
                    ready = ready and np.isfinite(pred.probability_fake).all() and pred.probability_fake.between(0, 1).all()
            if not ready:
                if model is None:
                    model, transform = b.frozen_model(args, name, initial)
                    student = b.Correction(b.adaptation.FEATURES[name], meta['b_config']['head']['hidden']).to(args.device)
                    student.load_state_dict(saved['correction_state_dict'], strict=True); student.eval()
                    teacher = copy.deepcopy(student)
                    teacher.load_state_dict(bs['correction_state_dict'], strict=True); teacher.eval()
                loader = DataLoader(evaluation.Views(frame, ea, transform, condition), batch_size=args.batch_size,
                                    num_workers=args.workers, shuffle=False, pin_memory=args.device == 'cuda')
                scores = {s: np.empty(len(frame)) for s in paths}
                with torch.no_grad():
                    for images, indexes in tqdm(loader, desc=f'{name} B-start vs C / {condition}', mininterval=10):
                        logits, features = b.adaptation.outputs(name, model, images.to(args.device))
                        margin = logits[:, 1]-logits[:, 0]
                        for s, corrector in [('b_start', teacher), (STRATEGY, student)]:
                            corrected = corrector(features.float(), margin.float())
                            result = logits.float().clone(); result[:, 1] += corrected-margin
                            scores[s][indexes.numpy()] = result.softmax(1)[:, 1].cpu().numpy()
                reports = {}
                for s, path in paths.items():
                    if not np.isfinite(scores[s]).all():
                        raise RuntimeError('Nonfinite C evaluation prediction')
                    pred = frame[['crop_path', 'video_id', 'class_index', 'roi_status']].copy()
                    pred['probability_fake'] = scores[s]
                    temp = path.with_suffix('.tmp.csv'); pred.to_csv(temp, index=False); temp.replace(path.with_suffix('.csv'))
                    values = base.binary_metrics(frame.class_index.astype(int).to_numpy(), scores[s])
                    values.update(identity=identity, model=name, strategy=s, condition=condition, images=len(frame),
                        threshold=.5, precision='FP32', unit='frame', manifest_sha256=provenance['manifest_sha256'],
                        checkpoint_sha256=base.common.digest(run/'best.pt'), b_start_sha256=meta['checkpoint_sha256'],
                        initial_checkpoint_sha256=initial_hash,
                        b_start_epoch=meta['epoch'], b_start_file=meta['checkpoint_file'], c_selected_epoch=saved['epoch'],
                        balanced_error_rate=.5*(values['real_fpr']+values['fake_fnr']))
                    base.common.json_write(path, values); reports[s] = values
            else:
                print('Reuse C evaluation:', name, condition, flush=True)
            for s, values in reports.items():
                records.append({'model': name, 'strategy': s, 'condition': condition, **{k: values[k] for k in b.METRICS}})
                print(f'{name} {s} {condition}: FPR={values["real_fpr"]:.2%} FNR={values["fake_fnr"]:.2%} BER={values["balanced_error_rate"]:.2%} AUC={values["auc"]:.4f}', flush=True)
        del model, student, teacher, initial, bs, saved
        if args.device == 'cuda':
            torch.cuda.empty_cache()
    if args.reference_evaluations is not None and args.reference_evaluations.is_dir():
        for name in args.models:
            for path in sorted(args.reference_evaluations.rglob(f'{name}_augmentation_normal.json')):
                old_protocol = path.parent/'evaluation_protocol.json'
                if not old_protocol.is_file() or any(json.loads(old_protocol.read_text()).get(k) != v for k, v in provenance.items()):
                    continue
                paths = [path.parent/f'{name}_augmentation_{c}.json' for c in cfg['conditions']]
                if not all(p.is_file() for p in paths):
                    continue
                for condition, path in zip(cfg['conditions'], paths):
                    old = json.loads(path.read_text())
                    records.append({'model': name, 'strategy': 'augmentation', 'condition': condition,
                        **{k: old[k] for k in b.METRICS if k != 'balanced_error_rate'},
                        'balanced_error_rate': .5*(old['real_fpr']+old['fake_fnr'])})
                break
    table = pd.DataFrame(records)
    table.to_csv(args.output_root/'comparison.csv', index=False)
    table.groupby(['model', 'strategy'])[b.METRICS].mean().reset_index().to_csv(args.output_root/'condition_means.csv', index=False)
    print('Saved C comparison:', args.output_root, flush=True)


def main():
    p = b.parser()
    p.description = __doc__
    p.add_argument('--b-runs-root', type=Path, required=True)
    p.add_argument('--uda-manifest-dir', type=Path, required=True)
    p.add_argument('--target-images', type=int, default=5000)
    p.add_argument('--target-weight', type=float, default=.1)
    p.add_argument('--trust-weight', type=float, default=1.)
    p.add_argument('--extra-strength', type=float, default=.25)
    p.set_defaults(learning_rate=1e-4)
    args = p.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('GPU runtime required')
    if args.target_root is None:
        p.error('--target-root required')
    if args.action == 'evaluate':
        for field in ['cache_root', 'output_root', 'baseline_evaluation_root']:
            if getattr(args, field) is None:
                p.error(f'--{field.replace("_", "-")} required for evaluation')
    if args.source_images % 8 or args.val_images % 8 or args.batch_size % 2:
        p.error('Source/val counts must be divisible by8 and batch size must be even')
    if not 0 < args.extra_strength <= .25:
        p.error('Additional smoothing strength must be in (0, .25]')
    (train if args.action == 'train' else evaluate)(args)


if __name__ == '__main__':
    main()
