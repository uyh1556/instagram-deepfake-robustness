"""Target class-weighted SDA smoothing, with frozen target rows and video sampling."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import roc_curve
from tqdm import tqdm

import ffpp_c23_supervised_da as sda

base, adaptation, evaluation = sda.base, sda.adaptation, sda.evaluation
ROOT = sda.ROOT
STRATEGY = 'sda_smoothing_balanced'


def target_balance(target):
    """p(c): select a video uniformly, then select one of its frozen rows uniformly."""
    if target.empty or not target.class_index.astype(int).isin([0, 1]).all():
        raise ValueError('Expected nonempty labelled target pool')
    labels = target.class_index.astype(int)
    if labels.groupby(target.video_id).nunique().gt(1).any():
        raise ValueError('Target video has inconsistent labels')
    fake_prior = float(labels.groupby(target.video_id, sort=False).mean().mean())
    prior = np.array([1-fake_prior, fake_prior], dtype=float)
    if (prior <= 0).any():
        raise ValueError('Both target classes must be present')
    weights = (0.5/prior).tolist()
    return {'method': 'inverse video-uniform class prior',
            'sampler_class_prior': prior.tolist(), 'weights_real_fake': weights,
            'reduction': 'mean(weight[label] * per-image CE); fixed normalization, not batch weight sum',
            'expected_weight': float(np.dot(prior, weights)),
            'scope': 'target CE only; source CE unchanged'}


def batch_loss(model, xs, xt, ys, yt, weights):
    logits = model(torch.cat([xs, xt])).float()
    source = nn.functional.cross_entropy(logits[:len(ys)], ys)
    per_image = nn.functional.cross_entropy(logits[len(ys):], yt, reduction='none')
    target = (per_image * weights[yt]).mean()
    return source+target, source, target


def train(args):
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('GPU runtime required')
    pool, val, target, smoothing, data = sda.inputs(args)
    balance = target_balance(target)
    data = {**data, 'supervision': 'same frozen UDA target rows/order and restored non-test labels; target class-weighted CE',
            'target_balance': balance}
    args.manifest_root.mkdir(parents=True, exist_ok=True)
    frozen = args.manifest_root/'protocol.json'
    if frozen.exists() and json.loads(frozen.read_text()) != data:
        raise ValueError('Balanced SDA data changed; use a new manifest root')
    base.common.json_write(frozen, data)
    target.to_csv(args.manifest_root/'target_labelled.csv', index=False)
    val.to_csv(args.manifest_root/'source_validation.csv', index=False)
    initial_args = argparse.Namespace(runs_root=args.baseline_runs_root, seed=args.seed)
    # Read data provenance once; do not resample or relabel the frozen target pool.
    for name in args.models:
        initial, _ = evaluation.checkpoint(initial_args, name)
        if (initial['config']['data_hashes']['trainval'] != data['source_trainval_sha256']
                or initial['config']['data_hashes']['protocol'] != data['source_protocol_sha256']):
            raise ValueError('Initial baseline source data differ')
        del initial
    print(f'Source/epoch={args.source_images}; validation={len(val)}; target={len(target)}; labels={data["target_class_counts"]}', flush=True)
    print('Target video-sampler prior [real,fake]:', balance['sampler_class_prior'], flush=True)
    print('Target CE weights [real,fake]:', balance['weights_real_fake'], flush=True)
    weights = torch.tensor(balance['weights_real_fake'], dtype=torch.float32, device=args.device)
    for name in args.models:
        initial, initial_hash = evaluation.checkpoint(initial_args, name)
        cfg = {'protocol': 'ffpp_c23_balanced_supervised_da_v1', 'model': name, 'strategy': STRATEGY,
               'seed': args.seed, 'data': data, 'initial_checkpoint_sha256': initial_hash,
               'preprocessing': base.PREPROCESS[name], 'smoothing': smoothing,
               'epochs': args.epochs, 'patience': args.patience, 'min_delta': 1e-4,
               'source_images_per_epoch': args.source_images, 'batch_size': args.batch_size,
               'encoder_lr': args.encoder_lr, 'head_lr': args.head_lr, 'weight_decay': args.weight_decay,
               'train_blocks': initial['config']['train_blocks'],
               'loss': 'source CE + mean(target CE * inverse video-prior class weight); each domain weight 1',
               'target_balance': balance, 'batchnorm': 'running statistics frozen; identical to previous SDA',
               'sampling': 'same source epoch sampler and video-uniform target sampling with replacement as SDA',
               'checkpoint_selection': f'max clean FF++ fixed validation{len(val)} macro4 frame AUC; no target/test selection',
               'precision': 'BF16 if supported else FP32; no FP16',
               'code_sha256': base.common.digest(Path(__file__)),
               'sda_helper_sha256': base.common.digest(Path(sda.__file__)),
               'adaptation_helper_sha256': base.common.digest(Path(adaptation.__file__)),
               'model_code_sha256': base.common.digest(Path(base.__file__)),
               'shared_code_sha256': base.common.digest(Path(base.common.__file__)),
               'smoothing_code_sha256': base.common.digest(ROOT/'scripts/smoothing_roi_pilot.py'),
               'opencv': evaluation.cv2.__version__, 'torch': str(torch.__version__)}
        identity = evaluation.key(cfg)
        run = args.runs_root/sda.run_name(name, STRATEGY, args.seed)
        run.mkdir(parents=True, exist_ok=True)
        if (run/'config.json').exists() and json.loads((run/'config.json').read_text()) != cfg:
            raise ValueError(f'Existing balanced SDA run differs: {run}')
        if (run/'completed.json').exists():
            if json.loads((run/'completed.json').read_text())['identity'] != identity or not (run/'best.pt').is_file():
                raise ValueError('Completed balanced SDA checkpoint mismatch')
            print('Skip completed:', name, STRATEGY, flush=True)
            del initial
            continue
        base.common.json_write(run/'config.json', cfg)
        base.common.seed_all(args.seed)
        model, transform = base.build_model(name, pretrained=False)
        model.load_state_dict(initial['model_state_dict'], strict=True)
        del initial
        model.to(args.device)
        blocks = cfg['train_blocks'] if name != 'xception' else 2
        base.stage(model, name, blocks, False)
        heads = list(base.head(model, name).parameters())
        head_ids = {id(p) for p in heads}
        encoder = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
        parameters = encoder+heads
        optimizer = torch.optim.AdamW([{'params': encoder, 'lr': args.encoder_lr},
                                       {'params': heads, 'lr': args.head_lr}], weight_decay=args.weight_decay)
        start, best, anchor, stale, history = 0, -1., -1., 0, []
        if (run/'last.pt').exists():
            state = torch.load(run/'last.pt', map_location='cpu', weights_only=True)
            if state['identity'] != identity:
                raise ValueError('Balanced SDA resume identity differs')
            model.load_state_dict(state['model_state_dict'], strict=True)
            optimizer.load_state_dict(state['optimizer'])
            start, best, anchor, stale, history = (state[k] for k in ['epoch', 'best', 'anchor', 'stale', 'history'])
            del state
            print('Resume completed epochs:', start, flush=True)
        print(f'===== {name} / {STRATEGY} =====', flush=True)
        for epoch in range(start, args.epochs):
            if stale >= args.patience:
                break
            epoch_seed = args.seed+epoch*100003
            base.common.seed_all(epoch_seed)
            base.stage(model, name, blocks, False)
            for layer in model.modules():
                if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                    layer.eval()
            sampled = adaptation.balanced_source(pool, args.source_images, epoch_seed)
            dataset = sda.SupervisedViews(sampled, target, args, transform, STRATEGY, epoch, smoothing)
            loader = base.common.make_loader(dataset, args, True, epoch_seed)
            totals, count, target_counts = np.zeros(3), 0, np.zeros(2, dtype=np.int64)
            for xs, xt, ys, yt in tqdm(loader, desc=f'{name} balanced SDA {epoch+1}/{args.epochs}', mininterval=10, dynamic_ncols=True):
                xs, xt, ys, yt = [x.to(args.device) for x in [xs, xt, ys, yt]]
                optimizer.zero_grad(set_to_none=True)
                with base.amp_context(args):
                    loss, source_loss, target_loss = batch_loss(model, xs, xt, ys, yt, weights)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss; last completed epoch preserved')
                loss.backward()
                nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                optimizer.step()
                totals += np.array([float(v.detach()) for v in [loss, source_loss, target_loss]])*len(ys)
                target_counts += torch.bincount(yt.detach().cpu(), minlength=2).numpy()
                count += len(ys)
            val_args = argparse.Namespace(data_root=args.source_root, batch_size=args.batch_size,
                                           workers=args.workers, device=args.device, seed=args.seed)
            values, _ = base.evaluate(model, val, transform, val_args, 'FF++ clean validation')
            auc = values['macro_method_auc']
            improved = auc > best
            best = max(best, auc)
            if auc > anchor+1e-4:
                anchor, stale = auc, 0
            else:
                stale += 1
            history.append({'epoch': epoch+1, 'loss': float(totals[0]/count),
                            'source_loss': float(totals[1]/count), 'target_loss': float(totals[2]/count),
                            'target_real_exposures': int(target_counts[0]), 'target_fake_exposures': int(target_counts[1]),
                            'val_macro_auc': auc, 'val_real_fpr': values['frame']['real_fpr'],
                            'epoch_sample_sha256': evaluation.key(sampled.crop_path.tolist())})
            state = {'identity': identity, 'config': cfg, 'model_state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict(), 'epoch': epoch+1, 'best': best,
                     'anchor': anchor, 'stale': stale, 'history': history}
            if improved:
                base.common.save(run/'best.pt', {k: state[k] for k in ['identity', 'config', 'model_state_dict', 'epoch', 'best']})
            base.common.save(run/'last.pt', state)
            del state
            base.common.json_write(run/'history.json', history)
            pd.DataFrame(history).to_csv(run/'history.csv', index=False)
            print(f'Val AUC={auc:.4f} | best={best:.4f} | patience={stale}/{args.patience}', flush=True)
        base.common.json_write(run/'completed.json', {'identity': identity, 'best_source_val_macro_auc': best,
                                                     'epochs_completed': len(history)})
        del model, optimizer, parameters, heads, encoder
        if args.device == 'cuda':
            torch.cuda.empty_cache()


def roc_diagnostic(frame):
    labels = frame.class_index.astype(int).to_numpy()
    scores = frame.probability_fake.to_numpy(dtype=float)
    if set(labels) != {0, 1} or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError('Invalid predictions for ROC comparison')
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    # Descriptive ROC point, no test-selected deployment threshold and no interpolation.
    return {'tpr_at_fpr_5pct': float(tpr[fpr <= .05].max())}


def reference_predictions(root, name, condition, provenance):
    stem = f'{name}_sda_smoothing_{condition}'
    candidates = [root/f'{stem}.json', *sorted(root.glob(f'*/{stem}.json'))]
    valid = []
    for report in candidates:
        if report.is_file() and report.with_suffix('.csv').is_file():
            data = json.loads(report.read_text())
            if (data.get('model') == name and data.get('strategy') == 'sda_smoothing'
                    and data.get('condition') == condition and data.get('threshold') == .5
                    and data.get('manifest_sha256') == provenance['manifest_sha256']):
                valid.append(report)
    if len(valid) > 1:
        raise ValueError(f'Multiple matching SDA references: {valid}; point to the intended group folder')
    return valid[0] if valid else None


def compare(args):
    records, differences = [], []
    for name in args.models:
        for condition in json.loads(args.protocol.read_text())['conditions']:
            report = args.output_root/f'{name}_{STRATEGY}_{condition}.json'
            data = json.loads(report.read_text())
            weighted = pd.read_csv(report.with_suffix('.csv'), dtype={'crop_path': str, 'class_index': str})
            ref = reference_predictions(args.reference_output_root, name, condition, data) if args.reference_output_root else None
            metrics = {k: data[k] for k in ['auc', 'accuracy', 'f1', 'real_fpr', 'fake_fnr']}
            metrics.update(roc_diagnostic(weighted))
            balanced = {'model': name, 'condition': condition, 'training': 'SDA smoothing class-balanced', **metrics}
            if ref is not None:
                original = json.loads(ref.read_text())
                frame = pd.read_csv(ref.with_suffix('.csv'), dtype={'crop_path': str, 'class_index': str})
                if (frame.crop_path.duplicated().any() or weighted.crop_path.duplicated().any()
                        or set(frame.crop_path) != set(weighted.crop_path)):
                    raise ValueError('Reference and balanced test images differ')
                aligned = frame.set_index('crop_path').loc[weighted.crop_path]
                if aligned.class_index.tolist() != weighted.class_index.tolist():
                    raise ValueError('Reference and balanced test labels differ')
                raw_metrics = {k: original[k] for k in metrics if k != 'tpr_at_fpr_5pct'}
                raw_metrics.update(roc_diagnostic(frame))
                records.append({'model': name, 'condition': condition, 'training': 'SDA smoothing original', **raw_metrics})
                differences.append({'model': name, 'condition': condition, **{f'delta_{k}': metrics[k]-raw_metrics[k] for k in metrics}})
            else:
                print(f'Existing SDA reference not found: {name} / {condition}; new results are saved', flush=True)
            records.append(balanced)
    result = pd.DataFrame(records)
    result.to_csv(args.output_root/'class_balance_comparison.csv', index=False)
    pd.DataFrame(differences).to_csv(args.output_root/'class_balance_deltas.csv', index=False)
    print(result.to_string(index=False, float_format=lambda x: f'{x:.4f}'), flush=True)
    print('TPR@FPR<=5% is a descriptive test ROC metric; no operational threshold selected.', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['train', 'evaluate', 'compare'])
    for field in ['source-root', 'target-root', 'baseline-runs-root', 'uda-manifest-dir',
                  'runs-root', 'manifest-root', 'output-root', 'cache-root']:
        p.add_argument('--'+field, type=Path, required=True)
    p.add_argument('--reference-output-root', type=Path)
    p.add_argument('--protocol', type=Path, default=ROOT/'configs/ffpp_c23_celebdf_smoothing_v1.json')
    p.add_argument('--models', nargs='+', choices=list(base.PREPROCESS), default=list(base.PREPROCESS))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--source-images', type=int, default=6000)
    p.add_argument('--val-images', type=int, default=1200)
    p.add_argument('--target-images', type=int, default=5000)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--encoder-lr', type=float, default=1e-5)
    p.add_argument('--head-lr', type=float, default=1e-5)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = p.parse_args()
    if min(args.epochs, args.patience, args.source_images, args.val_images, args.target_images, args.batch_size) < 1:
        p.error('Counts and duration must be positive')
    if len(set(args.models)) != len(args.models):
        p.error('Duplicate models')
    args.strategies = [STRATEGY]
    if args.action == 'evaluate':
        # Existing SDA evaluator handles protocol checks, FP32 inference and result reuse.
        sda.evaluate(args)
    else:
        (train if args.action == 'train' else compare)(args)


if __name__ == '__main__':
    main()
