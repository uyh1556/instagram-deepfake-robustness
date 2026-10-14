"""Supervised target-label reference; only SDA normal/smoothing, no other runs."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

import ffpp_c23_smoothing_adaptation as adaptation
import evaluate_ffpp_baselines_celebdf_smoothing as evaluation
import train_ffpp_c23_baseline as base

ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = ['sda_normal', 'sda_smoothing']


def inputs(args):
    pool, val, target, cfg, data, _ = adaptation.dataset_inputs(args)
    frozen = json.loads((args.uda_manifest_dir/'protocol.json').read_text())
    saved_target = pd.read_csv(args.uda_manifest_dir/'target_unlabelled.csv', dtype=str, keep_default_na=False)
    saved_val = pd.read_csv(args.uda_manifest_dir/'source_validation.csv', dtype=str, keep_default_na=False)
    if (frozen['data'] != data or frozen['seed'] != args.seed
            or frozen['source_images_per_epoch'] != args.source_images or frozen['val_images'] != args.val_images
            or frozen['target_images'] != args.target_images
            or not saved_target.equals(target)
            or saved_val.crop_path.tolist() != val.crop_path.tolist()
            or saved_val.class_index.tolist() != val.class_index.tolist()):
        raise ValueError('Source/validation/target differ from frozen UDA protocol; do not resplit')
    catalogs = [pd.read_csv(args.target_root/'manifests'/f'celebdf_v2_roi16_non_test_{kind}_v1.csv',
                           dtype=str, keep_default_na=False) for kind in ['real', 'fake']]
    catalog = pd.concat(catalogs, ignore_index=True)
    if catalog.crop_path.duplicated().any():
        raise ValueError('Duplicate target label lookup')
    # Restore ground truth only from training-pool metadata, never the test manifest.
    lookup = catalog.set_index('crop_path')['class_index']
    target = saved_target.copy()
    target['class_index'] = target.crop_path.map(lookup)
    if target.class_index.isna().any() or not target.class_index.isin(['0', '1']).all():
        raise ValueError('Missing/invalid target training labels')
    data = {**data, 'uda_protocol_sha256': base.common.digest(args.uda_manifest_dir/'protocol.json'),
            'target_labels_sha256': evaluation.key(target.class_index.tolist()),
            'target_class_counts': {str(k): int(v) for k, v in target.class_index.value_counts().items()},
            'supervision': 'same frozen UDA target rows/order, restored non-test labels; no class balancing'}
    return pool, val, target, cfg, data


class SupervisedViews(adaptation.TrainingViews):
    def __getitem__(self, index):
        # Identical random sequence / target video sampler to uda_normal/uda_smoothing.
        rng = random.Random(self.args.seed+self.epoch*100003+index)
        source = self.source[index]
        xs = self.view(source, self.args.source_root, 'normal', rng.random() < .5)
        target = rng.choice(self.video_rows[rng.choice(self.videos)])
        condition = 'normal' if self.strategy == 'sda_normal' else rng.choice(list(self.cfg['conditions'])[1:])
        xt = self.view(target, self.args.target_root, condition, rng.random() < .5)
        return xs, xt, int(source['class_index']), int(target['class_index'])


def run_name(name, strategy, seed):
    return f'{name}_{strategy}_ffpp_c23_celebdf_smoothing_v1_seed{seed}'


def batch_loss(model, xs, xt, ys, yt):
    logits = model(torch.cat([xs, xt])).float()
    source = nn.functional.cross_entropy(logits[:len(ys)], ys)
    target = nn.functional.cross_entropy(logits[len(ys):], yt)
    return source+target, source, target


def train(args):
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('GPU runtime required')
    pool, val, target, smoothing, data = inputs(args)
    args.manifest_root.mkdir(parents=True, exist_ok=True)
    frozen = args.manifest_root/'protocol.json'
    if frozen.exists() and json.loads(frozen.read_text()) != data:
        raise ValueError('SDA data changed; preserve old output and use new manifest root')
    base.common.json_write(frozen, data)
    target.to_csv(args.manifest_root/'target_labelled.csv', index=False)
    val.to_csv(args.manifest_root/'source_validation.csv', index=False)
    initial_args = argparse.Namespace(runs_root=args.baseline_runs_root, seed=args.seed)
    for name in args.models:
        state, _ = evaluation.checkpoint(initial_args, name)
        if (state['config']['data_hashes']['trainval'] != data['source_trainval_sha256']
                or state['config']['data_hashes']['protocol'] != data['source_protocol_sha256']):
            raise ValueError('Initial baseline source data differ')
        del state
    print(f'Source/epoch={args.source_images}; validation={len(val)}; target={len(target)}; labels={data["target_class_counts"]}', flush=True)
    for name in args.models:
        initial, initial_hash = evaluation.checkpoint(initial_args, name)
        for strategy in args.strategies:
            cfg = {'protocol': 'ffpp_c23_supervised_da_v1', 'model': name, 'strategy': strategy, 'seed': args.seed,
                   'data': data, 'initial_checkpoint_sha256': initial_hash, 'preprocessing': base.PREPROCESS[name],
                   'smoothing': smoothing, 'epochs': args.epochs, 'patience': args.patience, 'min_delta': 1e-4,
                   'source_images_per_epoch': args.source_images, 'batch_size': args.batch_size,
                   'encoder_lr': args.encoder_lr, 'head_lr': args.head_lr, 'weight_decay': args.weight_decay,
                   'train_blocks': initial['config']['train_blocks'],
                   'loss': 'source CE + target CE, unit weight each; no DANN, no class reweighting, no pseudo-labels',
                   'batchnorm': 'running statistics frozen; identical to previous adaptation',
                   'sampling': 'same source epoch sampler and video-uniform target sampling with replacement as UDA',
                   'checkpoint_selection': f'max clean FF++ fixed validation{len(val)} macro4 frame AUC after an SDA epoch; no target/test selection',
                   'precision': 'BF16 if supported else FP32; no FP16',
                   'code_sha256': base.common.digest(Path(__file__)),
                   'adaptation_helper_sha256': base.common.digest(Path(adaptation.__file__)),
                   'model_code_sha256': base.common.digest(Path(base.__file__)),
                   'shared_code_sha256': base.common.digest(Path(base.common.__file__)),
                   'smoothing_code_sha256': base.common.digest(ROOT/'scripts/smoothing_roi_pilot.py'),
                   'opencv': evaluation.cv2.__version__, 'torch': str(torch.__version__)}
            identity = evaluation.key(cfg)
            run = args.runs_root/run_name(name, strategy, args.seed)
            run.mkdir(parents=True, exist_ok=True)
            if (run/'config.json').exists() and json.loads((run/'config.json').read_text()) != cfg:
                raise ValueError(f'Existing SDA run differs: {run}')
            if (run/'completed.json').exists():
                if json.loads((run/'completed.json').read_text())['identity'] != identity or not (run/'best.pt').is_file():
                    raise ValueError('Completed SDA checkpoint mismatch')
                print('Skip completed:', name, strategy, flush=True)
                continue
            base.common.json_write(run/'config.json', cfg)
            base.common.seed_all(args.seed)
            model, transform = base.build_model(name, pretrained=False)
            model.load_state_dict(initial['model_state_dict'], strict=True)
            model.to(args.device)
            blocks = initial['config']['train_blocks'] if name != 'xception' else 2
            base.stage(model, name, blocks, False)
            heads = list(base.head(model, name).parameters())
            head_ids = {id(p) for p in heads}
            encoder = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
            optimizer = torch.optim.AdamW([{'params': encoder, 'lr': args.encoder_lr},
                                           {'params': heads, 'lr': args.head_lr}], weight_decay=args.weight_decay)
            parameters = encoder+heads
            start, best, anchor, stale, history = 0, -1., -1., 0, []
            if (run/'last.pt').exists():
                state = torch.load(run/'last.pt', map_location='cpu', weights_only=True)
                if state['identity'] != identity:
                    raise ValueError('SDA resume identity differs')
                model.load_state_dict(state['model_state_dict'], strict=True)
                optimizer.load_state_dict(state['optimizer'])
                start, best, anchor, stale, history = (state[k] for k in ['epoch', 'best', 'anchor', 'stale', 'history'])
                del state
                print('Resume completed epochs:', start, flush=True)
            print(f'===== {name} / {strategy} =====', flush=True)
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
                dataset = SupervisedViews(sampled, target, args, transform, strategy, epoch, smoothing)
                loader = base.common.make_loader(dataset, args, True, epoch_seed)
                totals, count = np.zeros(3), 0
                for xs, xt, ys, yt in tqdm(loader, desc=f'{name} {strategy} {epoch+1}/{args.epochs}',
                                           mininterval=10, dynamic_ncols=True):
                    xs, xt, ys, yt = [x.to(args.device) for x in [xs, xt, ys, yt]]
                    optimizer.zero_grad(set_to_none=True)
                    with base.amp_context(args):
                        loss, source_loss, target_loss = batch_loss(model, xs, xt, ys, yt)
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite loss; last completed epoch preserved, no batch skipped')
                    loss.backward()
                    nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                    optimizer.step()
                    totals += np.array([float(v.detach()) for v in [loss, source_loss, target_loss]])*len(ys)
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
        del initial


def evaluate(args):
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('GPU runtime required')
    eval_args = argparse.Namespace(data_root=args.target_root,
        manifest=args.target_root/'manifests/celebdf_v2_roi16_test_v1.csv', protocol=args.protocol,
        cache_root=args.cache_root, output_root=args.output_root, workers=args.workers)
    evaluation.prepare(eval_args)
    frame, smoothing, provenance = evaluation.inputs(eval_args)
    records = []
    for name in args.models:
        # No baseline result read/inference; only initial checkpoint provenance is checked.
        initial, initial_hash = evaluation.checkpoint(argparse.Namespace(runs_root=args.baseline_runs_root, seed=args.seed), name)
        del initial
        for strategy in args.strategies:
            run = args.runs_root/run_name(name, strategy, args.seed)
            completed = json.loads((run/'completed.json').read_text())
            cfg = json.loads((run/'config.json').read_text())
            state = torch.load(run/'best.pt', map_location='cpu', weights_only=True)
            if (cfg != state['config'] or state['identity'] != evaluation.key(cfg)
                    or completed['identity'] != state['identity'] or cfg['model'] != name
                    or cfg['strategy'] != strategy or cfg['seed'] != args.seed
                    or cfg['initial_checkpoint_sha256'] != initial_hash or cfg['smoothing'] != smoothing
                    or cfg['data']['test_manifest_sha256'] != provenance['manifest_sha256']
                    or cfg['preprocessing'] != base.PREPROCESS[name]):
                raise ValueError('SDA checkpoint/evaluation protocol mismatch')
            checkpoint_hash = base.common.digest(run/'best.pt')
            identity = evaluation.key({'data': provenance, 'checkpoint': checkpoint_hash,
                'precision': 'FP32', 'torch': str(torch.__version__),
                'code_sha256': base.common.digest(Path(__file__)),
                'model_code_sha256': base.common.digest(Path(base.__file__)),
                'shared_code_sha256': base.common.digest(Path(base.common.__file__))})
            model, transform = None, None
            for condition in smoothing['conditions']:
                prefix = args.output_root/f'{name}_{strategy}_{condition}'
                report, predictions = prefix.with_suffix('.json'), prefix.with_suffix('.csv')
                values = None
                if report.exists() and predictions.exists():
                    saved = json.loads(report.read_text())
                    predicted = pd.read_csv(predictions, dtype={'crop_path': str, 'class_index': str})
                    if saved.get('identity') != identity:
                        raise ValueError('Stored SDA evaluation differs; use new output root')
                    if (predicted.crop_path.tolist() == frame.crop_path.tolist()
                            and predicted.class_index.tolist() == frame.class_index.tolist()
                            and np.isfinite(predicted.probability_fake).all()
                            and predicted.probability_fake.between(0, 1).all()):
                        values = saved
                        print('Reuse completed:', name, strategy, condition, flush=True)
                if values is None:
                    if model is None:
                        model, transform = base.build_model(name, pretrained=False)
                        model.load_state_dict(state['model_state_dict'], strict=True)
                        model.to(args.device).eval()
                        del state
                    loader = base.common.make_loader(evaluation.Views(frame, eval_args, transform, condition), args)
                    scores = np.empty(len(frame))
                    with torch.inference_mode():
                        for images, indexes in tqdm(loader, desc=f'{name} {strategy} {condition}', mininterval=10, dynamic_ncols=True):
                            probabilities = model(images.to(args.device)).float().softmax(1)[:, 1].cpu().numpy()
                            if not np.isfinite(probabilities).all():
                                raise RuntimeError('Nonfinite SDA prediction')
                            scores[indexes.numpy()] = probabilities
                    predicted = frame[['crop_path', 'video_id', 'class_index', 'roi_status']].copy()
                    predicted['probability_fake'] = scores
                    temp = predictions.with_suffix('.tmp.csv')
                    predicted.to_csv(temp, index=False)
                    temp.replace(predictions)
                    values = base.binary_metrics(frame.class_index.astype(int).to_numpy(), scores)
                    values.update(identity=identity, model=name, strategy=strategy, condition=condition, images=len(frame),
                                  checkpoint_sha256=checkpoint_hash, manifest_sha256=provenance['manifest_sha256'],
                                  unit='frame', threshold=.5, precision='FP32')
                    base.common.json_write(report, values)
                records.append({k: values[k] for k in ['model', 'strategy', 'condition', 'images', 'auc', 'accuracy', 'f1', 'real_fpr', 'fake_fnr']})
                print(f'{name} {strategy} {condition}: AUC={values["auc"]:.4f}, FPR={values["real_fpr"]:.2%}, FNR={values["fake_fnr"]:.2%}', flush=True)
            del model
            if 'state' in locals():
                del state
            if args.device == 'cuda':
                torch.cuda.empty_cache()
    pd.DataFrame(records).to_csv(args.output_root/'sda_frame_summary.csv', index=False)
    print('SDA-only results:', args.output_root, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['train', 'evaluate'])
    for field in ['source-root', 'target-root', 'baseline-runs-root', 'uda-manifest-dir',
                  'runs-root', 'manifest-root', 'output-root', 'cache-root']:
        p.add_argument('--'+field, type=Path, required=True)
    p.add_argument('--protocol', type=Path, default=ROOT/'configs/ffpp_c23_celebdf_smoothing_v1.json')
    p.add_argument('--models', nargs='+', choices=list(base.PREPROCESS), default=list(base.PREPROCESS))
    p.add_argument('--strategies', nargs='+', choices=STRATEGIES, default=STRATEGIES)
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
    (train if args.action == 'train' else evaluate)(args)


if __name__ == '__main__':
    main()
