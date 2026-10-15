"""B: frozen FF++ detector + bounded residual margin correction; no target training."""
import argparse
import importlib.metadata
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import ffpp_c23_smoothing_adaptation as adaptation

base, evaluation = adaptation.base, adaptation.evaluation
ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'ffpp_c23_smoothing_preservation_b_v1'
STRATEGY = 'preservation_b'
METRICS = ['auc', 'accuracy', 'f1', 'real_fpr', 'fake_fnr', 'balanced_error_rate']


def run_name(name, seed):
    return f'{name}_{PROTOCOL}_seed{seed}'


class Correction(nn.Module):
    def __init__(self, features, hidden=32, scale=1., bound=2.):
        super().__init__()
        self.register_buffer('scale', torch.tensor(float(scale)))
        self.register_buffer('bound', torch.tensor(float(bound)))
        self.norm = nn.LayerNorm(features, elementwise_affine=False)
        self.net = nn.Sequential(nn.Linear(features+1, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features, margin):
        inputs = torch.cat([self.norm(features), (margin/self.scale).unsqueeze(1)], 1)
        delta = self.bound * self.scale * self.net(inputs).squeeze(1).tanh()
        return margin + delta


def loss_for(corrector, features, margins, labels, pair_weight, clean_weight):
    clean = corrector(features[:, 0], margins[:, 0])
    smooth = corrector(features[:, 1], margins[:, 1])
    task = .5*(nn.functional.binary_cross_entropy_with_logits(clean, labels.float())
               + nn.functional.binary_cross_entropy_with_logits(smooth, labels.float()))
    # Frozen clean SOURCE margin is a soft reference, not a target-domain pseudo-label.
    restore = nn.functional.smooth_l1_loss(smooth/corrector.scale, margins[:, 0]/corrector.scale)
    anchor = nn.functional.smooth_l1_loss(clean/corrector.scale, margins[:, 0]/corrector.scale)
    return task + pair_weight*restore + clean_weight*anchor, task, restore, anchor


def source_inputs(args):
    path = args.source_root/'manifests/ffpp_c23_4methods_roi16_trainval_v1.csv'
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {'crop_path', 'video_id', 'method', 'class_index', 'source_ids', 'split', 'roi_status'}
    if not required.issubset(frame) or frame.crop_path.duplicated().any():
        raise ValueError('Expected unique frozen FF++ Train/Val manifest')
    if not frame.split.isin(['train', 'val']).all() or frame.groupby('video_id').split.nunique().gt(1).any():
        raise ValueError('FF++ split/video mismatch')
    if not frame.class_index.astype(int).eq(frame.method.ne('original').astype(int)).all():
        raise ValueError('Source labels/manipulation provenance mismatch')
    membership = {}
    for row in frame[['source_ids', 'split']].drop_duplicates().itertuples(index=False):
        for source in row.source_ids.split('|'):
            if source in membership and membership[source] != row.split:
                raise ValueError('FF++ source-pair split leakage')
            membership[source] = row.split
    for split in ['train', 'val']:
        if set(frame[frame.split.eq(split)].method) != {'original', *base.METHODS}:
            raise ValueError('Source split lacks a manipulation')
    val = adaptation.balanced_source(frame[frame.split.eq('val')], args.val_images, args.seed)
    if args.source_validation is not None and args.source_validation.is_file():
        old = pd.read_csv(args.source_validation, dtype=str, keep_default_na=False)
        if (old.crop_path.tolist() != val.crop_path.tolist()
                or old.class_index.tolist() != val.class_index.tolist()):
            raise ValueError('Validation differs from the previous frozen adaptation subset')
    data = {'source_trainval_sha256': base.common.digest(path),
            'source_protocol_sha256': base.common.digest(args.source_root/'manifests/protocol.json'),
            'val_paths_sha256': evaluation.key(val.crop_path.tolist()), 'validation_images': len(val)}
    return frame[frame.split.eq('train')].reset_index(drop=True), val, data


def frozen_model(args, name, state):
    model, transform = base.build_model(name, pretrained=False)
    model.load_state_dict(state['model_state_dict'], strict=True)
    model.requires_grad_(False)
    model.to(args.device).eval()
    return model, transform


def feature_identity(args, name, checkpoint_hash, frame, cfg, kind, epoch=None):
    return {'model': name, 'checkpoint_sha256': checkpoint_hash, 'kind': kind, 'epoch': epoch,
            'rows': evaluation.key(frame.to_dict('records')), 'smoothing': cfg,
            'precision': 'FP32', 'device': args.device, 'seed': args.seed,
            'torch': str(torch.__version__), 'opencv': evaluation.cv2.__version__,
            'code': base.common.digest(Path(__file__)),
            'model_code': base.common.digest(Path(base.__file__)),
            'helper_code': base.common.digest(Path(base.common.__file__)),
            'adaptation_code': base.common.digest(Path(adaptation.__file__)),
            'smoothing_code': base.common.digest(ROOT/'scripts/smoothing_roi_pilot.py')}


def features_for(args, name, model, transform, frame, cfg, checkpoint_hash, epoch=None):
    """Cache frozen FP32 features locally. Train lists/flip/strength still rotate each epoch."""
    kind = 'validation_four_conditions' if epoch is None else 'training_pair'
    identity = feature_identity(args, name, checkpoint_hash, frame, cfg, kind, epoch)
    cache = args.feature_cache_root/name/f'{kind}_{evaluation.key(identity)}.pt'
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.is_file():
        saved = torch.load(cache, map_location='cpu', weights_only=True)
        if saved['identity'] != identity:
            raise ValueError('Frozen feature cache identity mismatch')
        print('Reuse frozen features:', name, kind, flush=True)
        return saved['features'], saved['margins']
    dataset = adaptation.TrainingViews(frame, pd.DataFrame(), args, transform,
                                       'augmentation', epoch or 0, cfg)
    loader = base.common.make_loader(dataset, args, False, args.seed)
    all_features, all_margins = [], []
    if epoch is None:
        for condition in cfg['conditions']:
            xs, ms = [], []
            # Source validation smoothing is generated using the same native-ROI operator.
            view = SingleViews(frame, args.source_root, transform, cfg, condition)
            view_loader = DataLoader(view, batch_size=args.batch_size*2, num_workers=args.workers,
                                     shuffle=False, pin_memory=args.device == 'cuda')
            with torch.no_grad():
                for images, _ in tqdm(view_loader, desc=f'{name} cache val {condition}', mininterval=10):
                    logits, f = adaptation.outputs(name, model, images.to(args.device))
                    xs.append(f.float().cpu()); ms.append((logits[:, 1]-logits[:, 0]).float().cpu())
            all_features.append(torch.cat(xs)); all_margins.append(torch.cat(ms))
        features, margins = torch.stack(all_features, 1), torch.stack(all_margins, 1)
    else:
        with torch.no_grad():
            for clean, smooth, _ in tqdm(loader, desc=f'{name} frozen features epoch{epoch+1}', mininterval=10):
                n = len(clean)
                logits, f = adaptation.outputs(name, model, torch.cat([clean, smooth]).to(args.device))
                margin = logits[:, 1]-logits[:, 0]
                all_features.append(torch.stack([f[:n], f[n:]], 1).float().cpu())
                all_margins.append(torch.stack([margin[:n], margin[n:]], 1).float().cpu())
        features, margins = torch.cat(all_features), torch.cat(all_margins)
    if not torch.isfinite(features).all() or not torch.isfinite(margins).all():
        raise RuntimeError('Nonfinite frozen features/margins')
    base.common.save(cache, {'identity': identity, 'features': features, 'margins': margins})
    return features, margins


class SingleViews(torch.utils.data.Dataset):
    def __init__(self, frame, root, transform, cfg, condition):
        self.rows, self.root, self.transform = frame.to_dict('records'), root, transform
        self.cfg, self.condition = cfg, condition

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from PIL import Image
        row = self.rows[index]
        with Image.open(self.root/row['crop_path']) as image:
            image = image.convert('RGB')
            if self.condition != 'normal' and row['roi_status'] == 'detected':
                image = adaptation.smooth(image, self.cfg['conditions'][self.condition], self.cfg['bilateral'])
            return self.transform(image), index


def validate(corrector, features, margins, frame, cfg, device):
    corrector.eval()
    result = {}
    with torch.no_grad():
        for i, condition in enumerate(cfg['conditions']):
            chunks = []
            for start in range(0, len(frame), 512):
                score = corrector(features[start:start+512, i].to(device), margins[start:start+512, i].to(device))
                chunks.append(score.sigmoid().cpu().numpy())
            values = base.frame_metrics(frame, np.concatenate(chunks))
            values['frame']['balanced_error_rate'] = .5*(values['frame']['real_fpr']+values['frame']['fake_fnr'])
            result[condition] = values
    result['mean_ber'] = float(np.mean([result[c]['frame']['balanced_error_rate'] for c in cfg['conditions']]))
    result['mean_auc'] = float(np.mean([result[c]['macro_method_auc'] for c in cfg['conditions']]))
    return result


def eligible(values, reference, args):
    clean, old = values['normal'], reference['normal']
    return bool(clean['frame']['balanced_error_rate'] <= old['frame']['balanced_error_rate']+args.clean_ber_tolerance
                and clean['frame']['real_fpr'] <= old['frame']['real_fpr']+args.clean_error_tolerance
                and clean['frame']['fake_fnr'] <= old['frame']['fake_fnr']+args.clean_error_tolerance
                and clean['macro_method_auc'] >= old['macro_method_auc']-args.auc_tolerance
                and values['mean_auc'] >= reference['mean_auc']-args.auc_tolerance)


def configuration(args, name, data, smoothing, initial_hash):
    versions = {}
    for package in ['torch', 'torchvision', 'timm', 'Pillow', 'numpy', 'pandas', 'scikit-learn', 'transformers']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {'protocol': PROTOCOL, 'strategy': STRATEGY, 'model': name, 'seed': args.seed,
            'data': data, 'smoothing': smoothing, 'initial_checkpoint_sha256': initial_hash,
            'preprocessing': base.PREPROCESS[name], 'backbone': 'entire encoder/classifier frozen, eval mode, FP32',
            'head': {'hidden': args.hidden, 'zero_initial_output': True, 'scale': 'max(1, train epoch1 clean absolute margin q75)',
                     'bound_in_scale_units': args.bound},
            'loss': {'task': 'mean clean/smoothed BCE on corrected raw margins',
                     'restoration': 'Huber(corrected smoothed / scale, frozen clean / scale)',
                     'clean_anchor': 'Huber(corrected clean / scale, frozen clean / scale)',
                     'pair_weight': args.pair_weight, 'clean_weight': args.clean_weight},
            'sampling': 'existing deterministic rotating epoch sampler, class1:1/fake four methods equal, paired flip',
            'source_images_per_epoch': args.source_images, 'views_per_epoch': 2*args.source_images,
            'epochs': args.epochs, 'patience': args.patience, 'batch_size': args.batch_size,
            'learning_rate': args.learning_rate, 'weight_decay': args.weight_decay,
            'selection': {'metric': 'minimum four-condition source validation mean BER; epoch0 eligible',
                          'clean_ber_tolerance': args.clean_ber_tolerance,
                          'clean_fpr_fnr_tolerance': args.clean_error_tolerance,
                          'clean_and_mean_auc_tolerance': args.auc_tolerance, 'min_delta': 1e-4},
            'target_training': False, 'code_sha256': base.common.digest(Path(__file__)),
            'model_code_sha256': base.common.digest(Path(base.__file__)),
            'shared_code_sha256': base.common.digest(Path(base.common.__file__)),
            'adaptation_code_sha256': base.common.digest(Path(adaptation.__file__)),
            'smoothing_code_sha256': base.common.digest(ROOT/'scripts/smoothing_roi_pilot.py'),
            'torch': str(torch.__version__), 'opencv': evaluation.cv2.__version__, 'versions': versions}


def train(args):
    pool, val, data = source_inputs(args)
    cfg = json.loads(args.protocol.read_text())
    args.manifest_root.mkdir(parents=True, exist_ok=True)
    frozen = args.manifest_root/'protocol.json'
    manifest_info = {**data, 'seed': args.seed, 'source_images_per_epoch': args.source_images}
    if frozen.exists() and json.loads(frozen.read_text()) != manifest_info:
        raise ValueError('Source protocol differs; use a new manifest root')
    base.common.json_write(frozen, manifest_info)
    val.to_csv(args.manifest_root/'source_validation.csv', index=False)
    print(f'B: source={args.source_images}/epoch; val={len(val)} unique x4 conditions; target training=NONE', flush=True)
    for name in args.models:
        state, initial_hash = evaluation.checkpoint(argparse.Namespace(runs_root=args.baseline_runs_root, seed=args.seed), name)
        if (state['config']['data_hashes']['trainval'] != data['source_trainval_sha256']
                or state['config']['data_hashes']['protocol'] != data['source_protocol_sha256']):
            raise ValueError('Source manifest differs from baseline training')
        config = configuration(args, name, data, cfg, initial_hash)
        identity = evaluation.key(config)
        run = args.runs_root/run_name(name, args.seed)
        run.mkdir(parents=True, exist_ok=True)
        if (run/'config.json').exists() and json.loads((run/'config.json').read_text()) != config:
            raise ValueError(f'Existing B configuration differs: {run}')
        if (run/'completed.json').exists():
            done = json.loads((run/'completed.json').read_text())
            if done['identity'] != identity or not (run/'best.pt').is_file():
                raise ValueError('Completed B run mismatch')
            del state
            print('Skip completed B:', name, flush=True)
            continue
        base.common.json_write(run/'config.json', config)
        base.common.seed_all(args.seed)
        model, transform = frozen_model(args, name, state)
        del state
        correction = Correction(adaptation.FEATURES[name], args.hidden, bound=args.bound).to(args.device)
        optimizer = torch.optim.AdamW(correction.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        vf, vm = features_for(args, name, model, transform, val, cfg, initial_hash)
        start, stale, history, first_pair = 0, 0, [], None
        if (run/'last.pt').exists():
            saved = torch.load(run/'last.pt', map_location='cpu', weights_only=True)
            if saved['identity'] != identity:
                raise ValueError('B resume identity differs')
            correction.load_state_dict(saved['correction_state_dict'], strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            start, stale, history, best, anchor, reference = [saved[k] for k in ['epoch', 'stale', 'history', 'best', 'anchor', 'reference']]
            del saved
            print(f'Resume {name}: after epoch {start}', flush=True)
        else:
            sampled = adaptation.balanced_source(pool, args.source_images, args.seed)
            first_pair = features_for(args, name, model, transform, sampled, cfg, initial_hash, epoch=0)
            correction.scale.fill_(max(1., float(first_pair[1][:, 0].abs().quantile(.75))))
            reference = validate(correction, vf, vm, val, cfg, args.device)
            best = anchor = reference['mean_ber']
            checkpoint = {'identity': identity, 'config': config, 'correction_state_dict': correction.state_dict(),
                          'epoch': 0, 'best': best, 'reference': reference, 'validation': reference}
            base.common.save(run/'best.pt', checkpoint)
            base.common.json_write(run/'baseline_source_validation.json', reference)
            print(f'{name} epoch0: BER={best:.2%}; AUC={reference["mean_auc"]:.4f}; margin scale={float(correction.scale):.3f}', flush=True)
        for epoch in range(start, args.epochs):
            if stale >= args.patience:
                break
            epoch_seed = args.seed+epoch*100003
            base.common.seed_all(epoch_seed)
            sampled = adaptation.balanced_source(pool, args.source_images, epoch_seed)
            plan = sampled[['crop_path', 'video_id', 'method', 'class_index']].copy()
            flips, conditions = [], []
            for i in range(len(plan)):
                rng = random.Random(epoch_seed+i)
                flips.append(rng.random() < .5)
                conditions.append(rng.choice(list(cfg['conditions'])[1:]))
            plan['flip'], plan['smoothing_condition'] = flips, conditions
            plan.to_csv(run/f'source_pairs_epoch{epoch+1:02d}.csv', index=False)
            features, margins = first_pair if epoch == 0 and first_pair is not None else features_for(
                args, name, model, transform, sampled, cfg, initial_hash, epoch=epoch)
            first_pair = None
            labels = torch.tensor(sampled.class_index.astype(int).to_numpy(), dtype=torch.long)
            loader = base.common.make_loader(TensorDataset(features, margins, labels), args, True, epoch_seed)
            correction.train()
            totals, count = np.zeros(4), 0
            for f, m, y in tqdm(loader, desc=f'{name} B head {epoch+1}/{args.epochs}', mininterval=10):
                f, m, y = f.to(args.device), m.to(args.device), y.to(args.device)
                optimizer.zero_grad(set_to_none=True)
                losses = loss_for(correction, f, m, y, args.pair_weight, args.clean_weight)
                if not torch.isfinite(losses[0]):
                    raise RuntimeError('Nonfinite B loss; last completed epoch preserved')
                losses[0].backward()
                nn.utils.clip_grad_norm_(correction.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                totals += np.array([float(x.detach()) for x in losses])*len(y)
                count += len(y)
            values = validate(correction, vf, vm, val, cfg, args.device)
            allowed = eligible(values, reference, args)
            improved = allowed and values['mean_ber'] < best-1e-12
            if improved:
                best = values['mean_ber']
            if allowed and values['mean_ber'] < anchor-1e-4:
                anchor, stale = values['mean_ber'], 0
            else:
                stale += 1
            history.append({'epoch': epoch+1, 'loss': float(totals[0]/count), 'classification_loss': float(totals[1]/count),
                            'restoration_loss': float(totals[2]/count), 'clean_anchor_loss': float(totals[3]/count),
                            'eligible': allowed, 'validation': values,
                            'epoch_sample_sha256': evaluation.key(sampled.crop_path.tolist())})
            checkpoint = {'identity': identity, 'config': config, 'correction_state_dict': correction.state_dict(),
                          'epoch': epoch+1, 'best': best, 'reference': reference, 'validation': values}
            if improved:
                base.common.save(run/'best.pt', checkpoint)
            base.common.save(run/'last.pt', {**checkpoint, 'optimizer': optimizer.state_dict(),
                                            'stale': stale, 'anchor': anchor, 'history': history})
            base.common.json_write(run/'history.json', history)
            clean = values['normal']['frame']
            print(f'{name} epoch{epoch+1}: meanBER={values["mean_ber"]:.2%} best={best:.2%} '
                  f'clean FPR/FNR={clean["real_fpr"]:.2%}/{clean["fake_fnr"]:.2%} '
                  f'eligible={allowed} patience={stale}/{args.patience}', flush=True)
            del features, margins, loader
        selected = torch.load(run/'best.pt', map_location='cpu', weights_only=True)
        base.common.json_write(run/'completed.json', {'identity': identity, 'epochs_completed': len(history),
            'selected_epoch': selected['epoch'], 'selected_source_mean_ber': best,
            'epoch_zero_selected': selected['epoch'] == 0})
        print(f'Completed {name}: selected epoch {selected["epoch"]}; source BER={best:.2%}', flush=True)
        del model, correction, optimizer, vf, vm, selected
        if args.device == 'cuda':
            torch.cuda.empty_cache()


def evaluate(args):
    ea = argparse.Namespace(data_root=args.target_root, manifest=args.target_root/'manifests/celebdf_v2_roi16_test_v1.csv',
                            protocol=args.protocol, cache_root=args.cache_root, output_root=args.output_root, workers=args.workers)
    evaluation.prepare(ea)
    frame, cfg, provenance = evaluation.inputs(ea)
    old_protocol = json.loads((args.baseline_evaluation_root/'evaluation_protocol.json').read_text())
    if any(old_protocol.get(k) != v for k, v in provenance.items()):
        raise ValueError('Existing baseline evaluation protocol differs')
    records = []
    for name in args.models:
        initial, initial_hash = evaluation.checkpoint(argparse.Namespace(runs_root=args.baseline_runs_root, seed=args.seed), name)
        run = args.runs_root/run_name(name, args.seed)
        config = json.loads((run/'config.json').read_text())
        done = json.loads((run/'completed.json').read_text())
        saved = torch.load(run/'best.pt', map_location='cpu', weights_only=True)
        if (config != saved['config'] or saved['identity'] != evaluation.key(config)
                or done['identity'] != saved['identity'] or config['model'] != name or config['seed'] != args.seed
                or config['protocol'] != PROTOCOL or config['strategy'] != STRATEGY
                or config['initial_checkpoint_sha256'] != initial_hash or config['smoothing'] != cfg
                or config['preprocessing'] != base.PREPROCESS[name]
                or config['code_sha256'] != base.common.digest(Path(__file__))):
            raise ValueError('B checkpoint/evaluation mismatch')
        checkpoint_hash = base.common.digest(run/'best.pt')
        identity = evaluation.key({'data': provenance, 'checkpoint': checkpoint_hash, 'initial': initial_hash,
                                   'precision': 'FP32', 'code': config['code_sha256'], 'torch': str(torch.__version__)})
        model, transform, correction = None, None, None
        for condition in cfg['conditions']:
            old_path = args.baseline_evaluation_root/f'{name}_{condition}.json'
            old = json.loads(old_path.read_text())
            if (old['checkpoint_sha256'] != initial_hash or old['manifest_sha256'] != provenance['manifest_sha256']
                    or old['precision'] != 'FP32' or old['images'] != len(frame)):
                raise ValueError('Saved baseline evaluation mismatch')
            records.append({'model': name, 'strategy': 'baseline', 'condition': condition,
                            **{k: old[k] for k in METRICS if k != 'balanced_error_rate'},
                            'balanced_error_rate': .5*(old['real_fpr']+old['fake_fnr'])})
            prefix = args.output_root/f'{name}_{STRATEGY}_{condition}'
            report, predictions = prefix.with_suffix('.json'), prefix.with_suffix('.csv')
            values = None
            if report.is_file() and predictions.is_file():
                existing = json.loads(report.read_text())
                p = pd.read_csv(predictions, dtype={'crop_path': str, 'class_index': str})
                if existing.get('identity') != identity:
                    raise ValueError('Existing B result identity differs; use a new output root')
                if (p.crop_path.tolist() == frame.crop_path.tolist() and p.class_index.tolist() == frame.class_index.tolist()
                        and np.isfinite(p.probability_fake).all() and p.probability_fake.between(0, 1).all()):
                    values = existing
                    print('Reuse evaluation:', name, condition, flush=True)
            if values is None:
                if model is None:
                    model, transform = frozen_model(args, name, initial)
                    correction = Correction(adaptation.FEATURES[name], config['head']['hidden']).to(args.device)
                    correction.load_state_dict(saved['correction_state_dict'], strict=True)
                    correction.eval()
                loader = DataLoader(evaluation.Views(frame, ea, transform, condition), batch_size=args.batch_size,
                                    num_workers=args.workers, shuffle=False, pin_memory=args.device == 'cuda')
                score, before, probability = np.empty(len(frame)), np.empty(len(frame)), np.empty(len(frame))
                with torch.no_grad():
                    for images, indexes in tqdm(loader, desc=f'{name} B / {condition}', mininterval=10):
                        logits, features = adaptation.outputs(name, model, images.to(args.device))
                        margin = logits[:, 1]-logits[:, 0]
                        corrected = correction(features.float(), margin.float())
                        if not torch.isfinite(corrected).all():
                            raise RuntimeError('Nonfinite B prediction')
                        score[indexes.numpy()] = corrected.cpu().numpy()
                        before[indexes.numpy()] = margin.cpu().numpy()
                        corrected_logits = logits.float().clone()
                        corrected_logits[:, 1] += corrected-margin
                        probability[indexes.numpy()] = corrected_logits.softmax(1)[:, 1].cpu().numpy()
                p = frame[['crop_path', 'video_id', 'class_index', 'roi_status']].copy()
                p['probability_fake'], p['baseline_margin'], p['corrected_margin'] = probability, before, score
                p['correction'] = score-before
                values = base.binary_metrics(frame.class_index.astype(int).to_numpy(), probability)
                values.update(identity=identity, model=name, strategy=STRATEGY, condition=condition, images=len(frame),
                              checkpoint_sha256=checkpoint_hash, initial_checkpoint_sha256=initial_hash,
                              manifest_sha256=provenance['manifest_sha256'], threshold=.5, precision='FP32', unit='frame',
                              selected_epoch=saved['epoch'], balanced_error_rate=.5*(values['real_fpr']+values['fake_fnr']))
                temp = predictions.with_suffix('.tmp.csv'); p.to_csv(temp, index=False); temp.replace(predictions)
                base.common.json_write(report, values)
            records.append({'model': name, 'strategy': STRATEGY, 'condition': condition, **{k: values[k] for k in METRICS}})
            print(f'{name} {condition}: FPR={values["real_fpr"]:.2%} FNR={values["fake_fnr"]:.2%} '
                  f'BER={values["balanced_error_rate"]:.2%} AUC={values["auc"]:.4f}', flush=True)
        del model, correction, initial, saved
        if args.device == 'cuda':
            torch.cuda.empty_cache()
    summary = pd.DataFrame(records)
    summary.to_csv(args.output_root/'frame_summary.csv', index=False)
    if args.reference_evaluations is not None and args.reference_evaluations.is_dir():
        for name in args.models:
            for path in sorted(args.reference_evaluations.rglob(f'{name}_augmentation_normal.json')):
                other = path.parent/'evaluation_protocol.json'
                if not other.is_file() or any(json.loads(other.read_text()).get(k) != v for k, v in provenance.items()):
                    continue
                paths = [path.parent/f'{name}_augmentation_{c}.json' for c in cfg['conditions']]
                if not all(p.is_file() for p in paths):
                    continue
                for condition, p in zip(cfg['conditions'], paths):
                    row = json.loads(p.read_text())
                    records.append({'model': name, 'strategy': 'augmentation', 'condition': condition,
                                    **{k: row[k] for k in METRICS if k != 'balanced_error_rate'},
                                    'balanced_error_rate': .5*(row['real_fpr']+row['fake_fnr'])})
                print('Reuse existing Aug:', name, path.parent, flush=True)
                break
    comparison = pd.DataFrame(records)
    comparison.to_csv(args.output_root/'comparison.csv', index=False)
    comparison.groupby(['model', 'strategy'])[METRICS].mean().reset_index().to_csv(
        args.output_root/'condition_means.csv', index=False)
    print('Saved:', args.output_root, flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['train', 'evaluate'])
    p.add_argument('--models', nargs='+', choices=list(base.PREPROCESS), default=list(base.PREPROCESS))
    for field in ['source-root', 'baseline-runs-root', 'runs-root', 'manifest-root', 'feature-cache-root']:
        p.add_argument('--'+field, type=Path, required=True)
    for field in ['target-root', 'cache-root', 'output-root', 'baseline-evaluation-root', 'reference-evaluations', 'source-validation']:
        p.add_argument('--'+field, type=Path)
    p.add_argument('--protocol', type=Path, default=ROOT/'configs/ffpp_c23_celebdf_smoothing_v1.json')
    for field, value in [('seed', 42), ('source-images', 6000), ('val-images', 1200), ('epochs', 10),
                         ('patience', 3), ('batch-size', 16), ('workers', 2), ('hidden', 32)]:
        p.add_argument('--'+field, type=int, default=value)
    for field, value in [('learning-rate', 1e-3), ('weight-decay', 1e-4), ('pair-weight', 1.), ('clean-weight', 1.),
                         ('bound', 2.), ('clean-ber-tolerance', .01), ('clean-error-tolerance', .02), ('auc-tolerance', .005)]:
        p.add_argument('--'+field, type=float, default=value)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    return p


def main():
    args = parser().parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('GPU runtime required')
    if args.action == 'train' and (args.source_images % 8 or args.val_images % 8):
        raise ValueError('Source/validation counts must be divisible by eight')
    if args.action == 'evaluate':
        for field in ['target_root', 'cache_root', 'output_root', 'baseline_evaluation_root']:
            if getattr(args, field) is None:
                raise ValueError(f'--{field.replace("_", "-")} required for evaluation')
    (train if args.action == 'train' else evaluate)(args)


if __name__ == '__main__':
    main()
