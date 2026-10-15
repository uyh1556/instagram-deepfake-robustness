"""Frozen FF++ detectors: matched current-view and smoothing-response classifiers."""
import argparse
import importlib.metadata
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

import ffpp_c23_smoothing_preservation as shared

base, adaptation, evaluation = shared.base, shared.adaptation, shared.evaluation
ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'ffpp_c23_smoothing_response_v1'
STRATEGIES = ('current_only', 'response')
METRICS = shared.METRICS


def run_name(name, strategy, seed):
    return f'{name}_{strategy}_{PROTOCOL}_seed{seed}'


class ResponseClassifier(nn.Module):
    """Same parameter count and initialization; current_only masks the response slots."""
    def __init__(self, features, hidden=32, use_response=True):
        super().__init__()
        self.use_response = use_response
        self.norm = nn.LayerNorm(features, elementwise_affine=False)
        self.register_buffer('feature_scale', torch.tensor(1.))
        self.register_buffer('margin_scale', torch.tensor(1.))
        self.net = nn.Sequential(nn.Linear(2*features+2, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features, margins):
        delta = (features[:, 1]-features[:, 0])/self.feature_scale
        dm = (margins[:, 1]-margins[:, 0])/self.margin_scale
        if not self.use_response:
            delta, dm = torch.zeros_like(delta), torch.zeros_like(dm)
        inputs = torch.cat([self.norm(features[:, 0]), delta,
                            (margins[:, 0]/self.margin_scale)[:, None], dm[:, None]], 1)
        return margins[:, 0] + self.margin_scale*self.net(inputs).squeeze(1)


class ProbeViews(Dataset):
    def __init__(self, plan, root, transform, cfg, extra_strength):
        self.rows, self.root, self.transform = plan.to_dict('records'), root, transform
        self.cfg, self.extra_strength = cfg, extra_strength

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        relative = Path(row['crop_path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'Unsafe image path: {relative}')
        with Image.open(self.root/relative) as image:
            observed = image.convert('RGB')
            probe = observed
            if row['roi_status'] == 'detected':
                strength = self.cfg['conditions'][row['condition']]
                if strength:
                    observed = adaptation.smooth(observed, strength, self.cfg['bilateral'])
                probe = adaptation.smooth(observed, self.extra_strength, self.cfg['bilateral'])
            if row['flip']:
                observed = observed.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                probe = probe.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            return torch.stack([self.transform(observed), self.transform(probe)]), index


def training_plan(frame, seed, cfg):
    rows = []
    for index, row in enumerate(frame.to_dict('records')):
        rng = random.Random(seed+index)
        flip = rng.random() < .5
        changed = rng.choice([c for c in cfg['conditions'] if c != 'normal'])
        rows.extend([{**row, 'condition': c, 'flip': flip} for c in ['normal', changed]])
    return pd.DataFrame(rows)


def fixed_plan(frame, condition):
    return frame.assign(condition=condition, flip=False)


def feature_identity(args, name, plan, cfg, checkpoint_hash):
    return {'protocol': PROTOCOL, 'model': name, 'checkpoint_sha256': checkpoint_hash,
            'rows_sha256': evaluation.key(plan.to_dict('records')), 'smoothing': cfg,
            'extra_strength': args.extra_strength, 'precision': 'FP32', 'device': args.device,
            'torch': str(torch.__version__), 'opencv': evaluation.cv2.__version__,
            'code': base.common.digest(Path(__file__)),
            'adaptation_code': base.common.digest(Path(adaptation.__file__)),
            'model_code': base.common.digest(Path(base.__file__)),
            'helper_code': base.common.digest(Path(base.common.__file__)),
            'smoothing_code': base.common.digest(ROOT/'scripts/smoothing_roi_pilot.py')}


def features_for(args, name, model, transform, plan, root, cfg, checkpoint_hash, tag):
    identity = feature_identity(args, name, plan, cfg, checkpoint_hash)
    cache = args.feature_cache_root/name/f'{tag}_{evaluation.key(identity)}.pt'
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.is_file():
        saved = torch.load(cache, map_location='cpu', weights_only=True)
        if saved['identity'] != identity:
            raise ValueError('Feature cache differs')
        print('Reuse features:', name, tag, flush=True)
        return saved['features'], saved['margins'], saved['probabilities']
    loader = DataLoader(ProbeViews(plan, root, transform, cfg, args.extra_strength),
                        batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=args.device == 'cuda', persistent_workers=False)
    fs, ms, ps = [], [], []
    with torch.inference_mode():
        for views, _ in tqdm(loader, desc=f'{name} features {tag}', mininterval=5):
            n = len(views)
            logits, f = adaptation.outputs(name, model, views.flatten(0, 1).to(args.device))
            fs.append(f.float().reshape(n, 2, -1).cpu())
            ms.append((logits[:, 1]-logits[:, 0]).float().reshape(n, 2).cpu())
            ps.append(logits.float().softmax(1)[:, 1].reshape(n, 2)[:, 0].cpu())
    features, margins, probabilities = torch.cat(fs), torch.cat(ms), torch.cat(ps)
    if not all(torch.isfinite(t).all() for t in [features, margins, probabilities]):
        raise RuntimeError('Nonfinite frozen outputs; feature cache not saved')
    base.common.save(cache, dict(identity=identity, features=features, margins=margins, probabilities=probabilities))
    return features, margins, probabilities


def validation(head, banks, frame, device):
    result = {}
    head.eval()
    with torch.inference_mode():
        for condition, (features, margins, _) in banks.items():
            chunks = [head(features[i:i+512].to(device), margins[i:i+512].to(device)).sigmoid().cpu().numpy()
                      for i in range(0, len(frame), 512)]
            values = base.frame_metrics(frame, np.concatenate(chunks))
            values['frame']['balanced_error_rate'] = .5*(values['frame']['real_fpr']+values['frame']['fake_fnr'])
            result[condition] = values
    result['mean_ber'] = float(np.mean([result[c]['frame']['balanced_error_rate'] for c in banks]))
    result['mean_auc'] = float(np.mean([result[c]['macro_method_auc'] for c in banks]))
    return result


def configuration(args, name, strategy, data, cfg, initial_hash):
    versions = {}
    for package in ['torch', 'torchvision', 'Pillow', 'numpy', 'pandas', 'scikit-learn', 'timm', 'transformers']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {'protocol': PROTOCOL, 'model': name, 'strategy': strategy, 'seed': args.seed,
            'data': data, 'smoothing': cfg, 'extra_strength': args.extra_strength,
            'initial_checkpoint_sha256': initial_hash, 'preprocessing': base.PREPROCESS[name],
            'backbone': 'all parameters and original classifier frozen; eval mode; FP32',
            'head': {'hidden': args.hidden, 'input_features': 2*adaptation.FEATURES[name]+2,
                     'response_slots_masked': strategy == 'current_only', 'zero_initial_residual': True,
                     'form': 'existing raw margin + trainable MLP residual; no clipping/bound',
                     'scales': 'first training plan: RMS current features; max(1,q75 absolute current margin)'},
            'loss': 'BCEWithLogits on both observed source views using actual source labels; no restoration/teacher/consistency',
            'source_images_per_epoch': args.source_images, 'observed_images_per_epoch': 2*args.source_images,
            'feature_forward_views_per_epoch_shared': 4*args.source_images,
            'sampling': 'existing rotating class1:1/method-uniform video-balanced sampler; clean+random weak/medium/strong; paired flip',
            'epochs': args.epochs, 'patience': args.patience, 'batch_size': args.batch_size,
            'hidden': args.hidden, 'learning_rate': args.learning_rate, 'weight_decay': args.weight_decay,
            'selection': 'minimum four-condition FF++ validation mean BER; epoch0 eligible; no clean/AUC guards; patience delta 0.0001',
            'target_training': False, 'versions': versions,
            'code_sha256': base.common.digest(Path(__file__)),
            'shared_source_code_sha256': base.common.digest(Path(shared.__file__)),
            'model_code_sha256': base.common.digest(Path(base.__file__)),
            'adaptation_code_sha256': base.common.digest(Path(adaptation.__file__)),
            'smoothing_code_sha256': base.common.digest(ROOT/'scripts/smoothing_roi_pilot.py')}


def train(args):
    if args.source_validation is not None and not args.source_validation.is_file():
        raise FileNotFoundError(args.source_validation)
    pool, val, data = shared.source_inputs(args)
    cfg = json.loads(args.protocol.read_text())
    args.manifest_root.mkdir(parents=True, exist_ok=True)
    manifest_info = {**data, 'seed': args.seed, 'source_images_per_epoch': args.source_images,
                     'smoothing': cfg, 'extra_strength': args.extra_strength}
    frozen = args.manifest_root/'protocol.json'
    if frozen.exists() and json.loads(frozen.read_text()) != manifest_info:
        raise ValueError('Manifest protocol differs; use a new output root')
    base.common.json_write(frozen, manifest_info)
    val.to_csv(args.manifest_root/'source_validation.csv', index=False)
    print(f'Source={args.source_images}/epoch x2 observed views; val={len(val)} x4; no target training', flush=True)
    for name in args.models:
        initial, initial_hash = evaluation.checkpoint(argparse.Namespace(runs_root=args.baseline_runs_root, seed=args.seed), name)
        if (initial['config']['data_hashes']['trainval'] != data['source_trainval_sha256']
                or initial['config']['data_hashes']['protocol'] != data['source_protocol_sha256']):
            raise ValueError('Source data differs from baseline')
        active = {}
        for strategy in args.strategies:
            config = configuration(args, name, strategy, data, cfg, initial_hash)
            identity = evaluation.key(config)
            run = args.runs_root/run_name(name, strategy, args.seed)
            run.mkdir(parents=True, exist_ok=True)
            if (run/'config.json').exists() and json.loads((run/'config.json').read_text()) != config:
                raise ValueError(f'Existing configuration differs: {run}')
            if (run/'completed.json').exists():
                done = json.loads((run/'completed.json').read_text())
                if done['identity'] != identity or not (run/'best.pt').is_file():
                    raise ValueError('Completed run identity mismatch')
                print('Skip completed:', name, strategy, flush=True)
                continue
            base.common.json_write(run/'config.json', config)
            active[strategy] = dict(run=run, config=config, identity=identity)
        if not active:
            del initial
            continue
        model, transform = shared.frozen_model(args, name, initial)
        del initial
        banks = {c: features_for(args, name, model, transform, fixed_plan(val, c), args.source_root,
                                cfg, initial_hash, f'validation_{c}') for c in cfg['conditions']}
        for strategy, item in active.items():
            base.common.seed_all(args.seed)
            head = ResponseClassifier(adaptation.FEATURES[name], args.hidden, strategy == 'response').to(args.device)
            optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            if (item['run']/'last.pt').exists():
                saved = torch.load(item['run']/'last.pt', map_location='cpu', weights_only=True)
                if saved['identity'] != item['identity']:
                    raise ValueError('Resume identity mismatch')
                head.load_state_dict(saved['head_state_dict'], strict=True)
                optimizer.load_state_dict(saved['optimizer'])
                item.update({k: saved[k] for k in ['epoch', 'best', 'stale', 'anchor', 'history']})
                print('Resume:', name, strategy, 'after epoch', item['epoch'], flush=True)
            else:
                item.update(epoch=0, best=None, stale=0, anchor=None, history=[])
            item.update(head=head, optimizer=optimizer)
        first_epoch = min((v['epoch'] for v in active.values() if v['stale'] < args.patience), default=args.epochs)
        for epoch in range(first_epoch, args.epochs):
            pending = {s: v for s, v in active.items() if v['epoch'] <= epoch and v['stale'] < args.patience}
            if not pending:
                break
            seed = args.seed+epoch*100003
            sampled = adaptation.balanced_source(pool, args.source_images, seed)
            plan = training_plan(sampled, seed, cfg)
            plan_path = args.manifest_root/f'source_plan_epoch{epoch+1:02d}.csv'
            plan.to_csv(plan_path, index=False)
            features, margins, _ = features_for(args, name, model, transform, plan, args.source_root,
                                               cfg, initial_hash, f'train_epoch{epoch+1:02d}')
            labels = torch.tensor(plan.class_index.astype(int).to_numpy(), dtype=torch.float32)
            for strategy, item in pending.items():
                head, optimizer = item['head'], item['optimizer']
                if item['best'] is None:
                    head.feature_scale.fill_(max(1e-6, float(features[:, 0].square().mean().sqrt())))
                    head.margin_scale.fill_(max(1., float(margins[:, 0].abs().quantile(.75))))
                    reference = validation(head, banks, val, args.device)
                    item['best'] = item['anchor'] = reference['mean_ber']
                    base.common.json_write(item['run']/'baseline_source_validation.json', reference)
                    base.common.save(item['run']/'best.pt', dict(identity=item['identity'], config=item['config'],
                        head_state_dict=head.state_dict(), epoch=0, best=item['best'], validation=reference))
                loader = DataLoader(TensorDataset(features, margins, labels), batch_size=args.batch_size,
                                    shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(seed))
                head.train()
                total, count = 0., 0
                for f, m, y in tqdm(loader, desc=f'{name} {strategy} {epoch+1}/{args.epochs}', mininterval=5):
                    optimizer.zero_grad(set_to_none=True)
                    loss = nn.functional.binary_cross_entropy_with_logits(head(f.to(args.device), m.to(args.device)), y.to(args.device))
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite loss; last completed epoch preserved')
                    loss.backward()
                    nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                    total, count = total+float(loss.detach())*len(y), count+len(y)
                values = validation(head, banks, val, args.device)
                improved = values['mean_ber'] < item['best']-1e-12
                if improved:
                    item['best'] = values['mean_ber']
                if values['mean_ber'] < item['anchor']-1e-4:
                    item['anchor'], item['stale'] = values['mean_ber'], 0
                else:
                    item['stale'] += 1
                item['epoch'] = epoch+1
                item['history'].append(dict(epoch=epoch+1, classification_loss=total/count,
                    validation=values, source_plan_sha256=evaluation.key(plan.to_dict('records'))))
                saved = dict(identity=item['identity'], config=item['config'], head_state_dict=head.state_dict(),
                             epoch=epoch+1, best=item['best'], validation=values)
                if improved:
                    base.common.save(item['run']/'best.pt', saved)
                base.common.save(item['run']/'last.pt', {**saved, 'optimizer': optimizer.state_dict(),
                    'stale': item['stale'], 'anchor': item['anchor'], 'history': item['history']})
                base.common.json_write(item['run']/'history.json', item['history'])
                normal = values['normal']['frame']
                print(f'{name} {strategy} epoch{epoch+1}: val BER={values["mean_ber"]:.2%} '
                      f'best={item["best"]:.2%} clean FPR/FNR={normal["real_fpr"]:.2%}/{normal["fake_fnr"]:.2%} '
                      f'patience={item["stale"]}/{args.patience}', flush=True)
            del features, margins, labels
        for strategy, item in active.items():
            best = torch.load(item['run']/'best.pt', map_location='cpu', weights_only=True)
            base.common.json_write(item['run']/'completed.json', dict(identity=item['identity'],
                epochs_completed=item['epoch'], selected_epoch=best['epoch'], selected_source_mean_ber=item['best']))
            print('Completed:', name, strategy, 'selected epoch', best['epoch'], flush=True)
        del model, banks, active
        if args.device == 'cuda':
            torch.cuda.empty_cache()


def metrics(frame, probabilities):
    values = base.binary_metrics(frame.class_index.astype(int).to_numpy(), probabilities)
    values['balanced_error_rate'] = .5*(values['real_fpr']+values['fake_fnr'])
    return values


def evaluate(args):
    ea = argparse.Namespace(data_root=args.target_root, manifest=args.target_root/'manifests/celebdf_v2_roi16_test_v1.csv', protocol=args.protocol)
    frame, cfg, provenance = evaluation.inputs(ea)
    args.output_root.mkdir(parents=True, exist_ok=True)
    protocol = {**provenance, 'excluded_from_adaptation_video_ids': sorted(frame.video_id.unique().tolist()),
                'identity_overlap_allowed': True, 'source_parent_audit_complete': False}
    protocol_path = args.output_root/'evaluation_protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Existing evaluation protocol differs')
    base.common.json_write(protocol_path, protocol)
    if not (args.output_root/'evaluation_manifest.csv').exists():
        (args.output_root/'evaluation_manifest.csv').write_bytes(ea.manifest.read_bytes())
    records = []
    for name in args.models:
        initial, initial_hash = evaluation.checkpoint(argparse.Namespace(runs_root=args.baseline_runs_root, seed=args.seed), name)
        heads, metadata = {}, {}
        for strategy in args.strategies:
            run = args.runs_root/run_name(name, strategy, args.seed)
            config = json.loads((run/'config.json').read_text())
            done = json.loads((run/'completed.json').read_text())
            saved = torch.load(run/'best.pt', map_location='cpu', weights_only=True)
            if (saved['config'] != config or saved['identity'] != evaluation.key(config)
                    or done['identity'] != saved['identity'] or config['initial_checkpoint_sha256'] != initial_hash
                    or config['protocol'] != PROTOCOL or config['model'] != name or config['strategy'] != strategy
                    or config['seed'] != args.seed or config['smoothing'] != cfg
                    or config['extra_strength'] != args.extra_strength
                    or config['code_sha256'] != base.common.digest(Path(__file__))):
                raise ValueError('Evaluation/checkpoint configuration mismatch')
            states = {strategy: (saved, run/'best.pt')}
            if saved['epoch'] == 0:
                last = torch.load(run/'last.pt', map_location='cpu', weights_only=True)
                if last['identity'] != saved['identity']:
                    raise ValueError('Last checkpoint identity mismatch')
                if last['epoch'] > 0:
                    states[strategy+'_last_diagnostic'] = (last, run/'last.pt')
            for label, (state, path) in states.items():
                head = ResponseClassifier(adaptation.FEATURES[name], config['hidden'], strategy == 'response').to(args.device)
                head.load_state_dict(state['head_state_dict'], strict=True)
                heads[label] = head.eval()
                metadata[label] = dict(checkpoint_sha256=base.common.digest(path), selected_epoch=state['epoch'],
                    evaluation_role='selected_model' if label == strategy else 'last_epoch_diagnostic')
        model, transform = shared.frozen_model(args, name, initial)
        del initial
        for condition in cfg['conditions']:
            identity = evaluation.key(dict(data=provenance, heads=metadata, initial=initial_hash,
                code=base.common.digest(Path(__file__)), feature=feature_identity(args, name, fixed_plan(frame, condition), cfg, initial_hash)))
            paths = {label: args.output_root/f'{name}_{label}_{condition}.json' for label in ['baseline', *heads]}
            ready = all(p.is_file() and p.with_suffix('.csv').is_file() for p in paths.values())
            reports = {s: json.loads(p.read_text()) for s, p in paths.items()} if ready else {}
            if ready and any(v.get('identity') != identity for v in reports.values()):
                raise ValueError('Existing evaluation identity differs')
            if ready:
                print('Reuse completed evaluation:', name, condition, flush=True)
            else:
                features, margins, old_prob = features_for(args, name, model, transform, fixed_plan(frame, condition),
                    args.target_root, cfg, initial_hash, f'cdf_test_{condition}')
                probabilities = {'baseline': old_prob.numpy()}
                scores = {'baseline': margins[:, 0].numpy()}
                with torch.inference_mode():
                    for label, head in heads.items():
                        chunks = [head(features[i:i+512].to(args.device), margins[i:i+512].to(args.device)).cpu()
                                  for i in range(0, len(frame), 512)]
                        score = torch.cat(chunks)
                        if not torch.isfinite(score).all():
                            raise RuntimeError('Nonfinite classifier margin')
                        scores[label] = score.numpy()
                        probabilities[label] = old_prob.numpy() if metadata[label]['selected_epoch'] == 0 else score.sigmoid().numpy()
                reports = {}
                for label, path in paths.items():
                    probability = probabilities[label]
                    if not np.isfinite(probability).all():
                        raise RuntimeError('Nonfinite evaluation prediction')
                    predicted = frame[['crop_path', 'video_id', 'class_index', 'roi_status']].copy()
                    predicted['probability_fake'] = probability
                    predicted['baseline_margin'], predicted['response_margin'] = margins[:, 0].numpy(), (margins[:, 1]-margins[:, 0]).numpy()
                    predicted['classified_margin'] = scores[label]
                    temp = path.with_suffix('.tmp.csv'); predicted.to_csv(temp, index=False); temp.replace(path.with_suffix('.csv'))
                    values = metrics(frame, probability)
                    values.update(identity=identity, model=name, strategy=label, condition=condition,
                        images=len(frame), threshold=.5, precision='FP32', unit='frame',
                        initial_checkpoint_sha256=initial_hash, manifest_sha256=provenance['manifest_sha256'],
                        **metadata.get(label, dict(checkpoint_sha256=initial_hash, selected_epoch=None, evaluation_role='baseline')))
                    base.common.json_write(path, values)
                    reports[label] = values
                del features, margins
            for label, values in reports.items():
                records.append(dict(model=name, strategy=label, condition=condition, **{k: values[k] for k in METRICS}))
                print(f'{name} {label} {condition}: FPR={values["real_fpr"]:.2%} FNR={values["fake_fnr"]:.2%} '
                      f'BER={values["balanced_error_rate"]:.2%} AUC={values["auc"]:.4f}', flush=True)
        del model, heads
        if args.device == 'cuda':
            torch.cuda.empty_cache()
    if args.reference_evaluations is not None and args.reference_evaluations.is_dir():
        for name in args.models:
            found = False
            for path in sorted(args.reference_evaluations.rglob(f'{name}_augmentation_normal.json')):
                old_protocol = path.parent/'evaluation_protocol.json'
                paths = [path.parent/f'{name}_augmentation_{c}.json' for c in cfg['conditions']]
                if not old_protocol.is_file() or not all(p.is_file() for p in paths):
                    continue
                if any(json.loads(old_protocol.read_text()).get(k) != v for k, v in provenance.items()):
                    continue
                for condition, path in zip(cfg['conditions'], paths):
                    row = json.loads(path.read_text())
                    if row['images'] != len(frame) or row['precision'] != 'FP32':
                        raise ValueError('Aug reference evaluation differs')
                    records.append(dict(model=name, strategy='augmentation', condition=condition,
                        **{k: row[k] for k in METRICS if k != 'balanced_error_rate'},
                        balanced_error_rate=.5*(row['real_fpr']+row['fake_fnr'])))
                found = True
                print('Reuse Aug reference:', name, path.parent, flush=True)
                break
            if not found:
                print('Aug reference absent or incompatible:', name, flush=True)
    table = pd.DataFrame(records)
    table.to_csv(args.output_root/'comparison.csv', index=False)
    table.groupby(['model', 'strategy'])[METRICS].mean().reset_index().to_csv(args.output_root/'condition_means.csv', index=False)
    print('Saved comparison:', args.output_root, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['train', 'evaluate'])
    p.add_argument('--models', nargs='+', choices=list(base.PREPROCESS), default=['clip_b16'])
    p.add_argument('--strategies', nargs='+', choices=STRATEGIES, default=list(STRATEGIES))
    for field in ['source-root', 'baseline-runs-root', 'runs-root', 'manifest-root', 'feature-cache-root']:
        p.add_argument('--'+field, type=Path, required=True)
    for field in ['target-root', 'output-root', 'reference-evaluations', 'source-validation']:
        p.add_argument('--'+field, type=Path)
    p.add_argument('--protocol', type=Path, default=ROOT/'configs/ffpp_c23_celebdf_smoothing_v1.json')
    for field, default in [('seed', 42), ('source-images', 6000), ('val-images', 1200), ('epochs', 10),
                           ('patience', 3), ('batch-size', 16), ('workers', 2), ('hidden', 32)]:
        p.add_argument('--'+field, type=int, default=default)
    p.add_argument('--learning-rate', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--extra-strength', type=float, default=.25)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = p.parse_args()
    if min(args.source_images, args.val_images, args.epochs, args.patience, args.batch_size, args.hidden) <= 0 or args.workers < 0:
        p.error('Counts must be positive and workers nonnegative')
    if args.source_images % 8 or args.val_images % 8 or not 0 < args.extra_strength <= .25:
        p.error('Source/val counts must be divisible by8; extra strength in (0,.25]')
    if len(set(args.models)) != len(args.models) or len(set(args.strategies)) != len(args.strategies):
        p.error('Duplicate model/strategy')
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('GPU runtime required')
    if args.action == 'evaluate' and (args.target_root is None or args.output_root is None):
        p.error('Evaluation needs --target-root and --output-root')
    (train if args.action == 'train' else evaluate)(args)


if __name__ == '__main__':
    main()
