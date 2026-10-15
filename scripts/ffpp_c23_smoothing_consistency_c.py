"""C v2: independent FF++ baseline initialization + source preservation + target consistency."""
import argparse
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
PROTOCOL = 'ffpp_c23_smoothing_consistency_c_independent_v2'
STRATEGY = 'preservation_c'


def run_name(name, seed):
    return f'{name}_{PROTOCOL}_seed{seed}'


def source_order(args, count, seed):
    """Match B's shuffled SOURCE batches while preserving balanced TARGET batch order."""
    loader = DataLoader(TensorDataset(torch.arange(count)), batch_size=args.batch_size,
                        shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(seed))
    return torch.cat([indexes for (indexes,) in loader])


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


def loss_for(student, sf, sm, labels, tf, tm, args):
    source, cls, restore, clean = b.loss_for(student, sf, sm, labels, args.pair_weight, args.clean_weight)
    observed = student(tf[:, 0], tm[:, 0])
    extra = student(tf[:, 1], tm[:, 1])
    consistency = nn.functional.smooth_l1_loss(observed/student.scale, extra/student.scale)
    return source + args.target_weight*consistency, cls, restore, clean, consistency


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
    for name in args.models:
        initial, initial_hash = evaluation.checkpoint(argparse.Namespace(runs_root=args.baseline_runs_root, seed=args.seed), name)
        if (initial['config']['data_hashes']['trainval'] != data['source_trainval_sha256']
                or initial['config']['data_hashes']['protocol'] != data['source_protocol_sha256']):
            raise ValueError('C source manifests differ from original FF++ baseline')
        config = b.configuration(args, name, info, smoothing, initial_hash)
        config.update(protocol=PROTOCOL, strategy=STRATEGY,
            initialization='original FF++ baseline + fresh zero-output correction; no B checkpoint',
            target_training=True, source_views_per_epoch=2*args.source_images, views_per_epoch=4*args.source_images,
            target_exposures_per_epoch=args.source_images, image_views_per_epoch=4*args.source_images,
            target_weight=args.target_weight, extra_strength=args.extra_strength,
            target_loss='symmetric normalized Huber(observed target score, additionally smoothed target score); no target anchor/teacher',
            source_batch_order='identical B shuffled source order; target batches remain exactly balanced',
            code_sha256=base.common.digest(Path(__file__)), b_code_sha256=base.common.digest(Path(b.__file__)),
            mixed_code_sha256=base.common.digest(Path(mixed.__file__)))
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
            del initial
            continue
        base.common.json_write(run/'config.json', config)
        base.common.seed_all(args.seed)
        model, transform = b.frozen_model(args, name, initial)
        del initial
        student = b.Correction(b.adaptation.FEATURES[name], args.hidden, bound=args.bound).to(args.device)
        optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        vf, vm = b.features_for(args, name, model, transform, val, smoothing, initial_hash)
        start, stale, history, first_pair = 0, 0, [], None
        if (run/'last.pt').exists():
            saved = torch.load(run/'last.pt', map_location='cpu', weights_only=True)
            if saved['identity'] != identity:
                raise ValueError('C resume identity mismatch')
            student.load_state_dict(saved['correction_state_dict'], strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            start, stale, history, best, anchor, reference = [saved[k] for k in ['epoch', 'stale', 'history', 'best', 'anchor', 'reference']]
            print('Resume C after epoch:', start, flush=True)
            del saved
        else:
            sampled = b.adaptation.balanced_source(pool, args.source_images, args.seed)
            first_pair = b.features_for(args, name, model, transform, sampled, smoothing, initial_hash, epoch=0)
            student.scale.fill_(max(1., float(first_pair[1][:, 0].abs().quantile(.75))))
            reference = b.validate(student, vf, vm, val, smoothing, args.device)
            best = anchor = reference['mean_ber']
            base.common.save(run/'best.pt', {'identity': identity, 'config': config,
                'correction_state_dict': student.state_dict(), 'epoch': 0, 'best': best,
                'reference': reference, 'validation': reference})
            base.common.json_write(run/'baseline_source_validation.json', reference)
        print(f'{name} C starts independently from FF++ baseline; epoch0 BER={reference["mean_ber"]:.2%}', flush=True)
        for epoch in range(start, args.epochs):
            if stale >= args.patience:
                break
            seed = args.seed+epoch*100003
            base.common.seed_all(seed)
            sampled = b.adaptation.balanced_source(pool, args.source_images, seed)
            sf, sm = first_pair if epoch == 0 and first_pair is not None else b.features_for(
                args, name, model, transform, sampled, smoothing, initial_hash, epoch)
            first_pair = None
            target_views = TargetPairs(sampled, target, args, transform, epoch, smoothing)
            plan = target_views.metadata()
            pd.DataFrame(plan).to_csv(run/f'target_exposures_epoch{epoch+1:02d}.csv', index=False)
            sampled.to_csv(run/f'source_epoch{epoch+1:02d}.csv', index=False)
            counts = Counter(r['sampler_class_index'] for r in plan)
            print(f'Target exposures real={counts[0]}, fake={counts[1]} | {dict(Counter(r["condition"] for r in plan))}', flush=True)
            tf, tm = target_features(args, name, model, target_views, initial_hash, epoch)
            labels = torch.tensor(sampled.class_index.astype(int).to_numpy(), dtype=torch.long)
            order = source_order(args, len(sampled), seed)
            pd.DataFrame({'source_row_index': order.numpy()}).to_csv(run/f'source_order_epoch{epoch+1:02d}.csv', index=False)
            loader = base.common.make_loader(TensorDataset(sf[order], sm[order], labels[order], tf, tm), args, False, seed)
            student.train()
            totals, count = np.zeros(5), 0
            for values in tqdm(loader, desc=f'{name} C correction {epoch+1}/{args.epochs}', mininterval=10):
                values = [v.to(args.device) for v in values]
                optimizer.zero_grad(set_to_none=True)
                losses = loss_for(student, *values, args)
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
                    ['classification_loss', 'restoration_loss', 'clean_anchor_loss', 'target_consistency_loss'], 1)},
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
            'selected_epoch': best_state['epoch'], 'selected_source_mean_ber': best,
            'initialization': 'original FF++ baseline; independent of B'})
        del model, student, optimizer, vf, vm, best_state
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
        run = args.runs_root/run_name(name, args.seed)
        config = json.loads((run/'config.json').read_text())
        saved = torch.load(run/'best.pt', map_location='cpu', weights_only=True)
        done = json.loads((run/'completed.json').read_text())
        if (saved['config'] != config or saved['identity'] != evaluation.key(config) or done['identity'] != saved['identity']
                or config['model'] != name or config['seed'] != args.seed or config['protocol'] != PROTOCOL
                or config['smoothing'] != cfg or not config['target_training']
                or config['b_code_sha256'] != base.common.digest(Path(b.__file__))
                or config['code_sha256'] != base.common.digest(Path(__file__))):
            raise ValueError('C evaluation/checkpoint mismatch')
        initial, initial_hash = evaluation.checkpoint(argparse.Namespace(runs_root=args.baseline_runs_root, seed=args.seed), name)
        if initial_hash != config['initial_checkpoint_sha256']:
            raise ValueError('C initial baseline differs')
        states = {STRATEGY: saved}
        # Preserve selected epoch0, but show the actually trained last model too.
        if saved['epoch'] == 0:
            last = torch.load(run/'last.pt', map_location='cpu', weights_only=True)
            if last['identity'] != saved['identity'] or last['config'] != config:
                raise ValueError('C last checkpoint mismatch')
            if last['epoch'] > 0:
                states[f'c_last_epoch{last["epoch"]}_diagnostic'] = last
        hashes = {s: base.common.digest(run/('best.pt' if s == STRATEGY else 'last.pt')) for s in states}
        identity = evaluation.key({'checkpoints': hashes,
                                   'test': provenance, 'code': config['code_sha256'], 'precision': 'FP32', 'torch': str(torch.__version__)})
        model, transform, correctors = None, None, {}
        for condition in cfg['conditions']:
            original = json.loads((args.baseline_evaluation_root/f'{name}_{condition}.json').read_text())
            if (original['checkpoint_sha256'] != initial_hash or original['manifest_sha256'] != provenance['manifest_sha256']
                    or original['precision'] != 'FP32' or original['images'] != len(frame)):
                raise ValueError('Baseline test reference differs')
            records.append({'model': name, 'strategy': 'baseline', 'condition': condition,
                **{k: original[k] for k in b.METRICS if k != 'balanced_error_rate'},
                'balanced_error_rate': .5*(original['real_fpr']+original['fake_fnr'])})
            paths = {s: args.output_root/f'{name}_{s}_{condition}.json' for s in states}
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
                    for s, state in states.items():
                        corrector = b.Correction(b.adaptation.FEATURES[name], config['head']['hidden']).to(args.device)
                        corrector.load_state_dict(state['correction_state_dict'], strict=True)
                        correctors[s] = corrector.eval()
                loader = DataLoader(evaluation.Views(frame, ea, transform, condition), batch_size=args.batch_size,
                                    num_workers=args.workers, shuffle=False, pin_memory=args.device == 'cuda')
                scores = {s: np.empty(len(frame)) for s in paths}
                with torch.no_grad():
                    for images, indexes in tqdm(loader, desc=f'{name} independent C / {condition}', mininterval=10):
                        logits, features = b.adaptation.outputs(name, model, images.to(args.device))
                        margin = logits[:, 1]-logits[:, 0]
                        for s, corrector in correctors.items():
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
                        checkpoint_sha256=hashes[s],
                        initial_checkpoint_sha256=initial_hash,
                        evaluated_epoch=states[s]['epoch'], c_selected_epoch=saved['epoch'],
                        evaluation_role='selected_model' if s == STRATEGY else 'last_epoch_diagnostic',
                        balanced_error_rate=.5*(values['real_fpr']+values['fake_fnr']))
                    base.common.json_write(path, values); reports[s] = values
            else:
                print('Reuse C evaluation:', name, condition, flush=True)
            for s, values in reports.items():
                records.append({'model': name, 'strategy': s, 'condition': condition, **{k: values[k] for k in b.METRICS}})
                print(f'{name} {s} {condition}: FPR={values["real_fpr"]:.2%} FNR={values["fake_fnr"]:.2%} BER={values["balanced_error_rate"]:.2%} AUC={values["auc"]:.4f}', flush=True)
        del model, correctors, initial, states, saved
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
    b_references = []
    if args.b_evaluation_root is not None and args.b_evaluation_root.is_dir():
        for name in args.models:
            for path in sorted(args.b_evaluation_root.rglob(f'{name}_preservation_b_normal.json')):
                if any(p.startswith('diagnostic_epoch') for p in path.parts):
                    continue
                previous = path.parent/'evaluation_protocol.json'
                if not previous.is_file() or any(json.loads(previous.read_text()).get(k) != v for k, v in provenance.items()):
                    continue
                paths = [path.parent/f'{name}_preservation_b_{c}.json' for c in cfg['conditions']]
                if not all(p.is_file() for p in paths):
                    continue
                for condition, path in zip(cfg['conditions'], paths):
                    row = json.loads(path.read_text())
                    records.append({'model': name, 'strategy': 'preservation_b', 'condition': condition,
                                    **{k: row[k] for k in b.METRICS}})
                    b_references.append({'model': name, 'condition': condition, 'file': str(path),
                                         'selected_epoch': row['selected_epoch']})
                break
    base.common.json_write(args.output_root/'b_reference_provenance.json', b_references)
    table = pd.DataFrame(records)
    table.to_csv(args.output_root/'comparison.csv', index=False)
    table.groupby(['model', 'strategy'])[b.METRICS].mean().reset_index().to_csv(args.output_root/'condition_means.csv', index=False)
    print('Saved C comparison:', args.output_root, flush=True)


def main():
    p = b.parser()
    p.description = __doc__
    p.add_argument('--b-evaluation-root', type=Path)
    p.add_argument('--uda-manifest-dir', type=Path, required=True)
    p.add_argument('--target-images', type=int, default=5000)
    p.add_argument('--target-weight', type=float, default=.1)
    p.add_argument('--extra-strength', type=float, default=.25)
    p.set_defaults(learning_rate=1e-3)
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
