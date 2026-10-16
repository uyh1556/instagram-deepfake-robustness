"""Frozen detector: matched BCE vs BCE + cross-quality real/fake ranking."""
import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

import ffpp_c23_smoothing_response as response
import ffpp_c23_smoothing_pivot as pixel_helpers

shared, base, adaptation, evaluation = response.shared, response.base, response.adaptation, response.evaluation
ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'ffpp_c23_smoothing_ranking_v1'
STRATEGIES = ['current_only', 'quality_ranking']
CONDITIONS = ['normal', 'smooth_weak', 'smooth_medium', 'smooth_strong']
METRICS = ['real_fpr', 'fake_fnr', 'balanced_error_rate', 'auc', 'auc_probability', 'accuracy', 'f1']


class CurrentClassifier(response.ResponseClassifier):
    """Exactly the old Current-only head/state layout, without a probe forward."""
    def __init__(self, features, hidden=32):
        super().__init__(features, hidden, use_response=False)

    def forward(self, features, margins):
        return super().forward(torch.stack([features, features], 1), torch.stack([margins, margins], 1))


def pair_scores(head, features, margins):
    n = len(features)
    return head(features.flatten(0, 1), margins.flatten()).reshape(n, 2)


def losses(scores, labels, scale, weight, gap):
    """Worst of TWO sampled views per image, then all real/fake pairs in batch."""
    ce = nn.functional.binary_cross_entropy_with_logits(scores, labels[:, None].expand_as(scores).float())
    real, fake = scores[labels == 0], scores[labels == 1]
    if len(real) == 0 or len(fake) == 0:
        raise ValueError('Ranking batches must contain both classes')
    real = real.max(1).values / scale
    fake = fake.min(1).values / scale
    rank = nn.functional.softplus(gap + real[:, None] - fake[None, :]).mean()
    return ce + weight * rank, ce, rank


def balanced_batches(labels, batch_size, seed):
    """Keep source-image pairs together and make every batch class balanced."""
    real, fake = [torch.where(labels == y)[0] for y in [0, 1]]
    if len(real) != len(fake) or not len(real) or batch_size % 2:
        raise ValueError('Equal nonempty classes and an even source-image batch size required')
    generator = torch.Generator().manual_seed(seed)
    real, fake = [ix[torch.randperm(len(ix), generator=generator)] for ix in [real, fake]]
    half = batch_size // 2
    result = []
    for start in range(0, len(real), half):
        indexes = torch.cat([real[start:start+half], fake[start:start+half]])
        result.append(indexes[torch.randperm(len(indexes), generator=generator)].tolist())
    return result


def code_hashes():
    modules = [Path(__file__), Path(response.__file__), Path(shared.__file__), Path(base.__file__),
               Path(base.common.__file__), Path(adaptation.__file__), Path(evaluation.__file__),
               Path(pixel_helpers.__file__), Path(pixel_helpers.pixel_cache.__file__), ROOT/'scripts/smoothing_roi_pilot.py']
    return {p.name: base.common.digest(p) for p in modules}


def run_name(name, strategy, seed):
    return f'{name}_{strategy}_{PROTOCOL}_seed{seed}'


class ObservedViews(Dataset):
    def __init__(self, plan, root, transform, cfg, pixel_cache=None, view_cache_root=None):
        self.rows, self.root, self.transform = plan.to_dict('records'), root, transform
        self.cfg, self.cache, self.view_cache_root = cfg, pixel_cache, view_cache_root

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        relative = Path(row['crop_path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'Unsafe frozen crop path: {relative}')
        with Image.open(self.root/relative) as opened:
            image = opened.convert('RGB')
        strength = self.cfg['conditions'][row['condition']]
        if strength and row['roi_status'] == 'detected':
            cached = self.view_cache_root/row['condition']/relative if self.view_cache_root is not None else None
            if cached is not None and cached.is_file():
                with Image.open(cached) as opened:
                    image = opened.convert('RGB')
            else:
                image = (self.cache.smooth(image, relative, strength) if self.cache is not None
                         else adaptation.smooth(image, strength, self.cfg['bilateral']))
        if row['flip']:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return self.transform(image), index


def feature_identity(args, name, plan, cfg, initial_hash):
    return dict(model=name, initial_checkpoint_sha256=initial_hash, rows=evaluation.key(plan.to_dict('records')),
        smoothing=cfg, preprocessing=base.PREPROCESS[name], precision='FP32', device=args.device,
        torch=str(torch.__version__), opencv=evaluation.cv2.__version__, code=code_hashes())


def features_for(args, name, model, transform, plan, root, cfg, initial_hash, tag, native_cache=None, view_cache_root=None):
    identity = feature_identity(args, name, plan, cfg, initial_hash)
    path = args.feature_cache_root/name/f'{tag}_{evaluation.key(identity)}.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved['identity'] != identity or len(saved['margins']) != len(plan):
            raise ValueError('Frozen feature cache identity/length differs')
        print('Reuse features:', name, tag, flush=True)
        return saved['features'], saved['margins']
    loader = DataLoader(ObservedViews(plan, root, transform, cfg, native_cache, view_cache_root),
        batch_size=args.feature_batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=args.device == 'cuda', persistent_workers=False)
    fs, ms = [], []
    with torch.inference_mode():
        for images, _ in tqdm(loader, desc=f'{name} features {tag}', mininterval=10, dynamic_ncols=True):
            logits, features = adaptation.outputs(name, model, images.to(args.device))
            fs.append(features.float().cpu())
            ms.append((logits[:, 1]-logits[:, 0]).float().cpu())
    features, margins = torch.cat(fs), torch.cat(ms)
    if not torch.isfinite(features).all() or not torch.isfinite(margins).all():
        raise RuntimeError('Nonfinite frozen outputs; cache not saved')
    base.common.save(path, dict(identity=identity, features=features, margins=margins))
    return features, margins


def metrics(frame, margins):
    margins = np.asarray(margins, dtype=np.float64)
    if margins.shape != (len(frame),) or not np.isfinite(margins).all():
        raise ValueError('Invalid raw margins')
    labels = frame.class_index.astype(int).to_numpy()
    probabilities = torch.as_tensor(margins, dtype=torch.float32).sigmoid().numpy()
    # Decisions from margin>=0, AUC from raw margins: no artificial FP32 softmax ties.
    values = base.binary_metrics(labels, (margins >= 0).astype(float))
    values.update(auc=float(roc_auc_score(labels, margins)),
        auc_probability=float(roc_auc_score(labels, probabilities)),
        balanced_error_rate=.5*(values['real_fpr']+values['fake_fnr']))
    return values


def validation(head, banks, frame, device):
    head.eval()
    result = {}
    with torch.inference_mode():
        for condition, (features, margins) in banks.items():
            scores = torch.cat([head(features[i:i+512].to(device), margins[i:i+512].to(device)).cpu()
                                for i in range(0, len(frame), 512)]).numpy()
            result[condition] = metrics(frame, scores)
    result['mean_ber'] = float(np.mean([result[c]['balanced_error_rate'] for c in CONDITIONS]))
    result['mean_auc'] = float(np.mean([result[c]['auc'] for c in CONDITIONS]))
    return result


def configuration(args, name, strategy, data, cfg, initial_hash):
    versions = {}
    for package in ['torch', 'torchvision', 'Pillow', 'numpy', 'pandas', 'scikit-learn', 'timm', 'transformers', 'clip']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return dict(protocol=PROTOCOL, model=name, strategy=strategy, seed=args.seed, data=data,
        smoothing=cfg, initial_checkpoint_sha256=initial_hash, preprocessing=base.PREPROCESS[name],
        backbone='entire baseline encoder/classifier frozen, eval mode, FP32',
        head=dict(hidden=args.hidden, form='same old Current-only residual MLP; response slots zero',
                  zero_initial_output=True, scale='max(1, epoch1 observed absolute raw margin q75)',
                  feature_scale='max(1e-6, epoch1 observed feature RMS); unused response input scale'),
        loss=dict(classification='mean BCE over clean and random weak/medium/strong source views',
                  ranking='mean softplus(gap + max(two real view margins)/scale - min(two fake view margins)/scale)',
                  ranking_weight=args.ranking_weight if strategy == 'quality_ranking' else 0., gap=args.ranking_gap,
                  labels='FF++ source GT only; no target, teacher, adversary or consistency loss'),
        sampling='existing rotating source plan; paired flip; balanced real/fake SOURCE-PAIR batches shared by strategies',
        source_images_per_epoch=args.source_images, observed_views_per_epoch=2*args.source_images,
        batch_size=args.batch_size, epochs=args.epochs, patience=args.patience, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, selection='four-condition FF++ val mean BER minimum; epoch0 eligible; no guards; patience delta1e-4',
        target_training=False, precision='FP32', versions=versions, code=code_hashes())


def freeze_json(path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f'Existing settings differ; use a new output root: {path}')
    if not path.exists():
        base.common.json_write(path, value)


def train(args):
    pool, val, data = shared.source_inputs(args)
    cfg = json.loads(args.protocol.read_text())
    if list(cfg['conditions']) != CONDITIONS:
        raise ValueError('Expected frozen normal/weak/medium/strong smoothing protocol')
    args.manifest_root.mkdir(parents=True, exist_ok=True)
    freeze_json(args.manifest_root/'protocol.json', dict(data=data, smoothing=cfg, seed=args.seed,
        source_images_per_epoch=args.source_images, source_pair_batch_size=args.batch_size))
    val_path = args.manifest_root/'source_validation.csv'
    if val_path.exists():
        old = pd.read_csv(val_path, dtype=str, keep_default_na=False)
        if old.to_dict('records') != val.to_dict('records'):
            raise ValueError('Saved source validation differs')
    else:
        val.to_csv(val_path, index=False)
    pixel_helpers.prepare_pixel_cache(args, cfg, data)
    print(f'Source={args.source_images}/epoch x2 views; val={len(val)} x4; frozen backbone; no CDF training', flush=True)
    for name in args.models:
        initial, initial_hash = evaluation.checkpoint(argparse.Namespace(runs_root=args.baseline_runs_root, seed=args.seed), name)
        if (initial['config']['data_hashes']['trainval'] != data['source_trainval_sha256']
                or initial['config']['data_hashes']['protocol'] != data['source_protocol_sha256']):
            raise ValueError('FF++ source differs from initial baseline')
        active = {}
        for strategy in args.strategies:
            config = configuration(args, name, strategy, data, cfg, initial_hash)
            identity = evaluation.key(config)
            run = args.runs_root/run_name(name, strategy, args.seed)
            run.mkdir(parents=True, exist_ok=True)
            freeze_json(run/'config.json', config)
            if (run/'completed.json').is_file():
                if json.loads((run/'completed.json').read_text())['identity'] != identity or not (run/'best.pt').is_file():
                    raise ValueError('Completed checkpoint identity mismatch')
                print('Skip completed:', name, strategy, flush=True)
                continue
            base.common.seed_all(args.seed)
            head = CurrentClassifier(adaptation.FEATURES[name], args.hidden).to(args.device)
            optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            item = dict(run=run, config=config, identity=identity, head=head, optimizer=optimizer,
                        epoch=0, stale=0, best=None, anchor=None, history=[])
            if (run/'last.pt').is_file():
                saved = torch.load(run/'last.pt', map_location='cpu', weights_only=True)
                if saved['identity'] != identity or saved['config'] != config:
                    raise ValueError('Resume identity mismatch')
                head.load_state_dict(saved['head_state_dict'], strict=True)
                optimizer.load_state_dict(saved['optimizer'])
                item.update({k: saved[k] for k in ['epoch', 'stale', 'best', 'anchor', 'history']})
                print('Resume:', name, strategy, 'after epoch', item['epoch'], flush=True)
            active[strategy] = item
        if not active:
            del initial
            continue
        model, transform = shared.frozen_model(args, name, initial)
        del initial
        banks = {c: features_for(args, name, model, transform, response.fixed_plan(val, c), args.source_root,
            cfg, initial_hash, f'validation_{c}', args._pixel_cache) for c in CONDITIONS}
        first_epoch = min((v['epoch'] for v in active.values() if v['stale'] < args.patience), default=args.epochs)
        for epoch in range(first_epoch, args.epochs):
            pending = {s: v for s, v in active.items() if v['epoch'] <= epoch and v['stale'] < args.patience}
            if not pending:
                break
            seed = args.seed + epoch*100003
            sampled = adaptation.balanced_source(pool, args.source_images, seed)
            plan = response.training_plan(sampled, seed, cfg)
            plan_hash = evaluation.key(plan.to_dict('records'))
            plan_path = args.manifest_root/f'source_plan_epoch{epoch+1:02d}.csv'
            if not plan_path.is_file():
                plan.to_csv(plan_path, index=False)
            elif evaluation.key(pd.read_csv(plan_path, dtype=str, keep_default_na=False).assign(flip=lambda x: x.flip.eq('True')).to_dict('records')) != plan_hash:
                raise ValueError('Saved epoch sampling plan differs')
            features, margins = features_for(args, name, model, transform, plan, args.source_root,
                cfg, initial_hash, f'train_epoch{epoch+1:02d}', args._pixel_cache)
            features, margins = features.reshape(len(sampled), 2, -1), margins.reshape(len(sampled), 2)
            labels = torch.tensor(sampled.class_index.astype(int).to_numpy())
            batches = balanced_batches(labels, args.batch_size, seed)
            batch_hash = evaluation.key(batches)
            for strategy, item in pending.items():
                head, optimizer = item['head'], item['optimizer']
                if item['best'] is None:
                    head.feature_scale.fill_(max(1e-6, float(features.square().mean().sqrt())))
                    head.margin_scale.fill_(max(1., float(margins.abs().quantile(.75))))
                    reference = validation(head, banks, val, args.device)
                    item['best'] = item['anchor'] = reference['mean_ber']
                    base.common.json_write(item['run']/'epoch0_validation.json', reference)
                    base.common.save(item['run']/'best.pt', dict(identity=item['identity'], config=item['config'],
                        head_state_dict=head.state_dict(), epoch=0, best=item['best'], validation=reference))
                loader = DataLoader(TensorDataset(features, margins, labels), batch_sampler=batches, num_workers=0,
                    generator=torch.Generator().manual_seed(seed))
                head.train()
                totals, count = np.zeros(3), 0
                weight = item['config']['loss']['ranking_weight']
                for f, m, y in tqdm(loader, desc=f'{name} {strategy} epoch{epoch+1}/{args.epochs}', mininterval=10, dynamic_ncols=True):
                    optimizer.zero_grad(set_to_none=True)
                    scores = pair_scores(head, f.to(args.device), m.to(args.device))
                    total, ce, rank = losses(scores, y.to(args.device), head.margin_scale, weight, args.ranking_gap)
                    if not torch.isfinite(total):
                        raise RuntimeError('Nonfinite classifier loss; last completed epoch preserved')
                    total.backward()
                    nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                    totals += np.array([total.item(), ce.item(), rank.item()])*len(y)
                    count += len(y)
                values = validation(head, banks, val, args.device)
                improved = values['mean_ber'] < item['best']-1e-12
                if improved:
                    item['best'] = values['mean_ber']
                if values['mean_ber'] < item['anchor']-1e-4:
                    item['anchor'], item['stale'] = values['mean_ber'], 0
                else:
                    item['stale'] += 1
                item['epoch'] = epoch+1
                item['history'].append(dict(epoch=epoch+1, loss=float(totals[0]/count), classification_loss=float(totals[1]/count),
                    ranking_loss=float(totals[2]/count), ranking_weight=weight, validation=values,
                    source_plan_sha256=plan_hash, batch_order_sha256=batch_hash))
                saved = dict(identity=item['identity'], config=item['config'], head_state_dict=head.state_dict(),
                    epoch=epoch+1, best=item['best'], validation=values)
                if improved:
                    base.common.save(item['run']/'best.pt', saved)
                base.common.save(item['run']/'last.pt', dict(saved, optimizer=optimizer.state_dict(),
                    stale=item['stale'], anchor=item['anchor'], history=item['history']))
                base.common.json_write(item['run']/'history.json', item['history'])
                print(f'{name} {strategy}: BCE={totals[1]/count:.4f} rank={totals[2]/count:.4f} '
                    f'val BER={values["mean_ber"]:.2%} best={item["best"]:.2%} patience={item["stale"]}/{args.patience}', flush=True)
        for strategy, item in active.items():
            best = torch.load(item['run']/'best.pt', map_location='cpu', weights_only=True)
            base.common.json_write(item['run']/'completed.json', dict(identity=item['identity'],
                epochs_completed=item['epoch'], selected_epoch=best['epoch'], selected_mean_ber=item['best']))
            print('Completed:', name, strategy, 'selected epoch', best['epoch'], flush=True)
        del model, banks, active
        if args.device == 'cuda':
            torch.cuda.empty_cache()


def evaluate(args):
    ea = argparse.Namespace(data_root=args.target_root,
        manifest=args.target_root/'manifests/celebdf_v2_roi16_test_v1.csv', protocol=args.protocol)
    frame, cfg, provenance = evaluation.inputs(ea)
    view_cache_root = None
    if args.cache_root is not None and (args.cache_root/'config.json').is_file():
        if json.loads((args.cache_root/'config.json').read_text()) != provenance:
            raise ValueError('Existing CDF smoothing cache provenance differs')
        view_cache_root = args.cache_root
    native_cache = pixel_helpers.pixel_cache.BilateralCache(args.evaluation_pixel_cache, cfg['bilateral'],
        dict(data=provenance, cache_code_sha256=base.common.digest(Path(pixel_helpers.pixel_cache.__file__)),
             storage='native uint8 RGB bilateral only; lossless PNG'))
    args.output_root.mkdir(parents=True, exist_ok=True)
    freeze_json(args.output_root/'evaluation_protocol.json', dict(provenance,
        excluded_from_adaptation_video_ids=sorted(frame.video_id.unique().tolist()),
        identity_overlap_allowed=True, source_parent_audit_complete=False))
    manifest = args.output_root/'evaluation_manifest.csv'
    if manifest.exists() and manifest.read_bytes() != ea.manifest.read_bytes():
        raise ValueError('Existing evaluation membership differs')
    if not manifest.exists():
        manifest.write_bytes(ea.manifest.read_bytes())
    records, pooled, cross = [], [], []
    for name in args.models:
        initial, initial_hash = evaluation.checkpoint(argparse.Namespace(runs_root=args.baseline_runs_root, seed=args.seed), name)
        heads, metadata = {}, {'baseline': dict(checkpoint_sha256=initial_hash, selected_epoch=0, role='baseline')}
        for strategy in args.strategies:
            run = args.runs_root/run_name(name, strategy, args.seed)
            config = json.loads((run/'config.json').read_text())
            done = json.loads((run/'completed.json').read_text())
            saved = torch.load(run/'best.pt', map_location='cpu', weights_only=True)
            if (saved['config'] != config or saved['identity'] != evaluation.key(config) or done['identity'] != saved['identity']
                    or config['protocol'] != PROTOCOL or config['strategy'] != strategy or config['model'] != name
                    or config['seed'] != args.seed or config['initial_checkpoint_sha256'] != initial_hash
                    or config['smoothing'] != cfg or config['code'] != code_hashes()):
                raise ValueError('Checkpoint/evaluation configuration mismatch')
            variants = {strategy: (saved, run/'best.pt')}
            if saved['epoch'] == 0:
                last = torch.load(run/'last.pt', map_location='cpu', weights_only=True)
                if last['identity'] != saved['identity'] or last['config'] != config:
                    raise ValueError('Last diagnostic identity mismatch')
                if last['epoch'] > 0:
                    variants[strategy+'_last_diagnostic'] = (last, run/'last.pt')
            for label, (state, path) in variants.items():
                head = CurrentClassifier(adaptation.FEATURES[name], config['head']['hidden']).to(args.device)
                head.load_state_dict(state['head_state_dict'], strict=True)
                heads[label] = head.eval()
                metadata[label] = dict(checkpoint_sha256=base.common.digest(path), selected_epoch=state['epoch'],
                    role='selected_model' if label == strategy else 'last_epoch_diagnostic')
        model, transform = None, None
        score_banks = {label: [] for label in metadata}
        for condition in CONDITIONS:
            identity = evaluation.key(dict(data=provenance, checkpoints=metadata, initial=initial_hash,
                code=code_hashes(), condition=condition, precision='FP32', torch=str(torch.__version__)))
            reports = {}
            for label in metadata:
                path = args.output_root/f'{name}_{label}_{condition}.json'
                if path.is_file() and path.with_suffix('.csv').is_file():
                    report = json.loads(path.read_text())
                    if report['identity'] != identity:
                        raise ValueError('Existing evaluation differs; use a new output root')
                    reports[label] = report
            if len(reports) != len(metadata):
                if model is None:
                    model, transform = shared.frozen_model(args, name, initial)
                plan = response.fixed_plan(frame, condition)
                features, margins = features_for(args, name, model, transform, plan, args.target_root,
                    cfg, initial_hash, f'cdf_test_{condition}', native_cache, view_cache_root)
                scores = {'baseline': margins.numpy()}
                with torch.inference_mode():
                    for label, head in heads.items():
                        scores[label] = torch.cat([head(features[i:i+512].to(args.device), margins[i:i+512].to(args.device)).cpu()
                            for i in range(0, len(frame), 512)]).numpy()
                for label in metadata:
                    if label in reports:
                        continue
                    path = args.output_root/f'{name}_{label}_{condition}.json'
                    raw = scores[label]
                    report = dict(metrics(frame, raw), identity=identity, model=name, strategy=label, condition=condition,
                        manifest_sha256=provenance['manifest_sha256'], initial_checkpoint_sha256=initial_hash,
                        images=len(frame), precision='FP32', unit='frame', threshold=.5, auc_score='raw_margin', **metadata[label])
                    pred = frame[['crop_path', 'video_id', 'class_index', 'roi_status']].copy()
                    pred['raw_margin'], pred['baseline_margin'] = raw, margins.numpy()
                    pred['probability_fake'] = torch.as_tensor(raw).sigmoid().numpy()
                    tmp = path.with_suffix('.tmp.csv'); pred.to_csv(tmp, index=False); tmp.replace(path.with_suffix('.csv'))
                    base.common.json_write(path, report)
                    reports[label] = report
            else:
                print('Reuse completed evaluation:', name, condition, flush=True)
            for label, report in reports.items():
                path = args.output_root/f'{name}_{label}_{condition}.csv'
                pred = pd.read_csv(path, dtype={'crop_path': str, 'class_index': str, 'video_id': str})
                if (pred.crop_path.tolist() != frame.crop_path.tolist() or pred.video_id.tolist() != frame.video_id.tolist()
                        or pred.class_index.tolist() != frame.class_index.tolist() or not np.isfinite(pred.raw_margin).all()):
                    raise ValueError('Saved prediction membership/raw margin differs')
                score_banks[label].append(pred.raw_margin.to_numpy())
                records.append(dict(model=name, strategy=label, condition=condition, **{k: report[k] for k in METRICS}))
                print(f'{name} {label} {condition}: FPR={report["real_fpr"]:.2%} FNR={report["fake_fnr"]:.2%} '
                      f'BER={report["balanced_error_rate"]:.2%} AUC={report["auc"]:.4f}', flush=True)
        labels = frame.class_index.astype(int).to_numpy()
        for label, bank in score_banks.items():
            mean_auc = float(np.mean([roc_auc_score(labels, s) for s in bank]))
            pooled.append(dict(model=name, strategy=label, condition_mean_auc=mean_auc,
                pooled_auc=float(roc_auc_score(np.tile(labels, 4), np.concatenate(bank)))))
            for i, real_condition in enumerate(CONDITIONS):
                for j, fake_condition in enumerate(CONDITIONS):
                    raw = np.concatenate([bank[i][labels == 0], bank[j][labels == 1]])
                    cross.append(dict(model=name, strategy=label, real_condition=real_condition, fake_condition=fake_condition,
                        auc=float(roc_auc_score(np.r_[np.zeros((labels == 0).sum()), np.ones((labels == 1).sum())], raw))))
        del model, initial, heads
        if args.device == 'cuda':
            torch.cuda.empty_cache()
    table = pd.DataFrame(records)
    table.to_csv(args.output_root/'comparison.csv', index=False)
    table.groupby(['model', 'strategy'], sort=False)[METRICS].mean().reset_index().to_csv(args.output_root/'condition_means.csv', index=False)
    pd.DataFrame(pooled).to_csv(args.output_root/'mixture_summary.csv', index=False)
    pd.DataFrame(cross).to_csv(args.output_root/'cross_condition_auc.csv', index=False)
    print('Saved:', args.output_root, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['train', 'evaluate'])
    p.add_argument('--models', nargs='+', choices=list(base.PREPROCESS), default=['clip_b16'])
    p.add_argument('--strategies', nargs='+', choices=STRATEGIES, default=STRATEGIES)
    for field in ['source-root', 'baseline-runs-root', 'runs-root', 'manifest-root', 'feature-cache-root']:
        p.add_argument('--'+field, type=Path, required=True)
    for field in ['target-root', 'output-root', 'source-validation']:
        p.add_argument('--'+field, type=Path)
    p.add_argument('--train-cache', type=Path, default=Path('/content/smoothing_pivot_bilateral_cache_v1'))
    p.add_argument('--cache-root', type=Path, default=Path('/content/celebdf_official_smoothing_cache_v1'))
    p.add_argument('--evaluation-pixel-cache', type=Path, default=Path('/content/smoothing_ranking_cdf_bilateral_cache_v1'))
    p.add_argument('--protocol', type=Path, default=ROOT/'configs/ffpp_c23_celebdf_smoothing_v1.json')
    for field, default in [('seed', 42), ('epochs', 10), ('patience', 3), ('source-images', 6000), ('val-images', 1200),
                           ('batch-size', 16), ('feature-batch-size', 32), ('hidden', 32), ('workers', 2)]:
        p.add_argument('--'+field, type=int, default=default)
    for field, default in [('learning-rate', .001), ('weight-decay', .0001), ('ranking-weight', .5), ('ranking-gap', 1.)]:
        p.add_argument('--'+field, type=float, default=default)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = p.parse_args()
    if (min(args.source_images, args.val_images, args.epochs, args.patience, args.batch_size, args.feature_batch_size, args.hidden) < 1
            or args.source_images % 8 or args.val_images % 8 or args.batch_size % 2 or args.workers < 0):
        p.error('Positive counts, source/val divisible by8, even batch size, nonnegative workers required')
    if (not all(np.isfinite(v) for v in [args.learning_rate, args.weight_decay, args.ranking_weight, args.ranking_gap])
            or args.learning_rate <= 0 or min(args.weight_decay, args.ranking_weight, args.ranking_gap) < 0):
        p.error('Finite nonnegative loss/optimizer settings and positive LR required')
    if len(set(args.models)) != len(args.models) or len(set(args.strategies)) != len(args.strategies):
        p.error('Duplicate model/strategy')
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('GPU runtime required')
    if args.action == 'evaluate' and (args.target_root is None or args.output_root is None):
        p.error('Evaluation needs --target-root and --output-root')
    (train if args.action == 'train' else evaluate)(args)


if __name__ == '__main__':
    main()
