"""Matched Aug, mixed-target UDA and Aug+UDA; target labels used ONLY for sampling."""
import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

import ffpp_c23_supervised_da as sda

base, adaptation, evaluation = sda.base, sda.adaptation, sda.evaluation
ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = ['uda_mixed_balanced', 'uda_aug_mixed_balanced']
CONDITIONS = ['normal', 'smooth_weak', 'smooth_medium', 'smooth_strong']
PROTOCOL = 'ffpp_c23_mixed_balanced_uda_v1'


def run_name(name, strategy, seed):
    return f'{name}_{strategy}_{PROTOCOL}_seed{seed}'


class MixedViews(adaptation.TrainingViews):
    """A fixed-order source epoch and an independently seeded target exposure plan.

    Each contiguous batch contains exactly half real/half fake targets. The loader
    MUST keep this order. Target classes/conditions are never returned to the loss.
    Target video and condition draws match between UDA and Aug+UDA.
    """
    def __init__(self, source, target, args, transform, strategy, epoch, cfg):
        super().__init__(source, target, args, transform, strategy, epoch, cfg)
        self.plan = []
        if strategy == 'augmentation':
            return
        videos = {0: [], 1: []}
        for video, rows in self.video_rows.items():
            labels = {int(row['class_index']) for row in rows}
            if len(labels) != 1 or next(iter(labels)) not in videos:
                raise ValueError('Target video labels inconsistent')
            videos[next(iter(labels))].append(video)
        if not all(videos.values()):
            raise ValueError('Target sampler requires both classes')
        rng = random.Random(args.seed + epoch*100003 + 810001)
        for start in range(0, len(self.source), args.batch_size):
            size = min(args.batch_size, len(self.source)-start)
            if size % 2:
                raise ValueError('Every target batch, including final batch, must be even')
            classes = [0]*(size//2) + [1]*(size//2)
            rng.shuffle(classes)
            for label in classes:
                row = rng.choice(self.video_rows[rng.choice(videos[label])])
                self.plan.append((row, rng.choice(CONDITIONS), rng.random() < .5))

    def __getitem__(self, index):
        row = self.source[index]
        rng = random.Random(self.args.seed + self.epoch*100003 + index)
        flip = rng.random() < .5
        xs = self.view(row, self.args.source_root, 'normal', flip)
        use_aug = self.strategy != 'uda_mixed_balanced'
        xa = self.view(row, self.args.source_root, rng.choice(CONDITIONS[1:]), flip) if use_aug else xs
        if self.plan:
            target, condition, target_flip = self.plan[index]
            xt = self.view(target, self.args.target_root, condition, target_flip)
        else:
            xt = xs
        return xs, xa, xt, int(row['class_index'])

    def exposure_metadata(self):
        return [{'crop_path': row['crop_path'], 'video_id': row['video_id'],
                 'sampler_class_index': int(row['class_index']), 'condition': condition,
                 'smoothing_applied': condition != 'normal' and row['roi_status'] == 'detected'}
                for row, condition, _ in self.plan]


def batch_loss(name, model, domain, xs, xa, xt, labels, strategy, grl, weight):
    n = len(labels)
    use_aug, uda = strategy != 'uda_mixed_balanced', strategy != 'augmentation'
    images = [xs] + ([xa] if use_aug else []) + ([xt] if uda else [])
    logits, features = adaptation.outputs(name, model, torch.cat(images))
    source_n = n*(2 if use_aug else 1)
    cls = nn.functional.cross_entropy(logits[:source_n].float(), labels.repeat(2 if use_aug else 1))
    dom = cls.new_zeros(())
    if uda:
        domain_logits = domain(adaptation.Reverse.apply(features.float(), grl))
        # Equal domain contribution even though Aug+UDA has two source views.
        ds = nn.functional.cross_entropy(domain_logits[:source_n], torch.zeros(source_n, dtype=torch.long, device=xs.device))
        dt = nn.functional.cross_entropy(domain_logits[source_n:], torch.ones(n, dtype=torch.long, device=xs.device))
        dom = .5*(ds+dt)
    return cls + weight*dom, cls, dom


def train(args):
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('GPU runtime required')
    # Reuse the exact old target rows and validation, restoring only non-test labels.
    pool, val, target, smoothing, data = sda.inputs(args)
    if list(smoothing['conditions']) != CONDITIONS:
        raise ValueError('Expected original + three frozen smoothing conditions')
    if args.batch_size % 2 or args.source_images % 8:
        raise ValueError('Batch size must be even; source count divisible by eight')
    data = {**data, 'supervision': 'same frozen target pool; target ground truth controls sampling ONLY, not the task loss',
            'target_label_usage': 'ground truth used ONLY to balance sampling; no target classification loss',
            'target_sampler': 'each batch 1:1 classes, video-uniform within class, with replacement',
            'target_condition_probabilities': dict.fromkeys(CONDITIONS, .25),
            'target_pair_usage': 'one target view per exposure; no clean/smoothed target pair supplied'}
    args.manifest_root.mkdir(parents=True, exist_ok=True)
    frozen = args.manifest_root/'protocol.json'
    if frozen.exists() and json.loads(frozen.read_text()) != data:
        raise ValueError('Existing mixed UDA manifest differs; use a new output root')
    base.common.json_write(frozen, data)
    target.drop(columns='class_index').to_csv(args.manifest_root/'target_unlabelled.csv', index=False)
    target.rename(columns={'class_index': 'sampler_class_index'}).to_csv(args.manifest_root/'target_sampling_metadata.csv', index=False)
    val.to_csv(args.manifest_root/'source_validation.csv', index=False)
    initial_args = argparse.Namespace(runs_root=args.baseline_runs_root, seed=args.seed)
    for name in args.models:
        initial, _ = evaluation.checkpoint(initial_args, name)
        if (initial['config']['data_hashes']['trainval'] != data['source_trainval_sha256']
                or initial['config']['data_hashes']['protocol'] != data['source_protocol_sha256']):
            raise ValueError('Baseline source manifests differ')
        del initial
    print(f'Source={args.source_images}/epoch; clean FF++ val={len(val)}; frozen target={len(target)}', flush=True)
    print('Target labels: sampler only; exposure Real:Fake=1:1; original/weak/medium/strong=25% each', flush=True)
    for name in args.models:
        initial, initial_hash = evaluation.checkpoint(initial_args, name)
        for strategy in args.strategies:
            uda = strategy != 'augmentation'
            cfg = {'protocol': PROTOCOL, 'model': name, 'strategy': strategy, 'seed': args.seed,
                   'data': data, 'initial_checkpoint_sha256': initial_hash, 'preprocessing': base.PREPROCESS[name],
                   'smoothing': smoothing, 'epochs': args.epochs, 'patience': args.patience, 'min_delta': 1e-4,
                   'source_images_per_epoch': args.source_images, 'batch_size': args.batch_size,
                   'encoder_lr': args.encoder_lr, 'head_lr': args.head_lr, 'weight_decay': args.weight_decay,
                   'domain_weight': args.domain_weight, 'train_blocks': initial['config']['train_blocks'],
                   'loss': 'mean source-view CE + domain_weight * balanced-domain CE (UDA only); target task labels never passed to loss',
                   'source_views': 'clean only' if strategy == 'uda_mixed_balanced' else 'clean + random weak/medium/strong',
                   'target_views_per_epoch': args.source_images if uda else 0,
                   'total_image_views_per_epoch': args.source_images*(3 if strategy == 'uda_aug_mixed_balanced' else 2),
                   'domain_loss_reduction': '0.5*mean(source domain CE) + 0.5*mean(target domain CE)',
                   'batchnorm': 'running statistics frozen', 'domain_head': 'LayerNorm + Linear128/ReLU/Linear2; progressive GRL',
                   'checkpoint_selection': 'max clean FF++ validation macro4 frame AUC; no target/test selection',
                   'precision': 'BF16 if supported else FP32; no FP16',
                   'code_sha256': base.common.digest(Path(__file__)),
                   'adaptation_helper_sha256': base.common.digest(Path(adaptation.__file__)),
                   'sda_helper_sha256': base.common.digest(Path(sda.__file__)),
                   'model_code_sha256': base.common.digest(Path(base.__file__)),
                   'shared_code_sha256': base.common.digest(Path(base.common.__file__)),
                   'smoothing_code_sha256': base.common.digest(ROOT/'scripts/smoothing_roi_pilot.py'),
                   'opencv': evaluation.cv2.__version__, 'torch': str(torch.__version__),
                   'baseline_package_versions': initial['config'].get('versions', {})}
            identity = evaluation.key(cfg)
            run = args.runs_root/run_name(name, strategy, args.seed)
            run.mkdir(parents=True, exist_ok=True)
            if (run/'config.json').exists() and json.loads((run/'config.json').read_text()) != cfg:
                raise ValueError(f'Run configuration differs: {run}')
            if (run/'completed.json').exists():
                if json.loads((run/'completed.json').read_text())['identity'] != identity or not (run/'best.pt').is_file():
                    raise ValueError('Completed checkpoint mismatch')
                print('Skip completed:', name, strategy, flush=True)
                continue
            base.common.json_write(run/'config.json', cfg)
            base.common.seed_all(args.seed)
            model, transform = base.build_model(name, pretrained=False)
            model.load_state_dict(initial['model_state_dict'], strict=True)
            model.to(args.device)
            blocks = initial['config']['train_blocks'] if name != 'xception' else 2
            base.stage(model, name, blocks, False)
            domain = nn.Sequential(nn.LayerNorm(adaptation.FEATURES[name]), nn.Linear(adaptation.FEATURES[name], 128),
                                   nn.ReLU(), nn.Linear(128, 2)).to(args.device)
            heads = list(base.head(model, name).parameters())
            head_ids = {id(p) for p in heads}
            groups = [{'params': [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids], 'lr': args.encoder_lr},
                      {'params': heads, 'lr': args.head_lr}]
            if uda:
                groups.append({'params': list(domain.parameters()), 'lr': args.head_lr})
            optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
            parameters = [p for group in groups for p in group['params']]
            start, best, anchor, stale, history = 0, -1., -1., 0, []
            if (run/'last.pt').exists():
                state = torch.load(run/'last.pt', map_location='cpu', weights_only=True)
                if state['identity'] != identity:
                    raise ValueError('Resume configuration differs')
                model.load_state_dict(state['model_state_dict'], strict=True)
                domain.load_state_dict(state['domain_state_dict'], strict=True)
                optimizer.load_state_dict(state['optimizer'])
                start, best, anchor, stale, history = (state[k] for k in ['epoch', 'best', 'anchor', 'stale', 'history'])
                del state
                print('Resume after epoch:', start, flush=True)
            print(f'===== {name} / {strategy} =====', flush=True)
            for epoch in range(start, args.epochs):
                if stale >= args.patience:
                    break
                epoch_seed = args.seed + epoch*100003
                base.common.seed_all(epoch_seed)
                base.stage(model, name, blocks, False)
                for layer in model.modules():
                    if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                        layer.eval()
                domain.train()
                sampled = adaptation.balanced_source(pool, args.source_images, epoch_seed)
                dataset = MixedViews(sampled, target, args, transform, strategy, epoch, smoothing)
                # Source was already shuffled above. Additional loader shuffling breaks target batch balance.
                loader = base.common.make_loader(dataset, args, False, epoch_seed)
                exposure = dataset.exposure_metadata()
                classes = Counter(row['sampler_class_index'] for row in exposure)
                conditions = Counter(row['condition'] for row in exposure)
                if uda:
                    pd.DataFrame(exposure).to_csv(run/f'target_exposures_epoch{epoch+1:02d}.csv', index=False)
                    print(f'Target exposures: real={classes[0]}, fake={classes[1]} | {dict(conditions)}', flush=True)
                totals, count = np.zeros(3), 0
                for step, (xs, xa, xt, labels) in enumerate(tqdm(loader, desc=f'{name} {strategy} {epoch+1}/{args.epochs}',
                                                                 mininterval=15, dynamic_ncols=True), 1):
                    xs, xa, xt, labels = (v.to(args.device) for v in [xs, xa, xt, labels])
                    optimizer.zero_grad(set_to_none=True)
                    progress = (epoch*len(loader)+step)/(args.epochs*len(loader))
                    grl = 2/(1+math.exp(-10*progress))-1
                    with base.amp_context(args):
                        loss, cls, dom = batch_loss(name, model, domain, xs, xa, xt, labels, strategy, grl, args.domain_weight)
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite loss; last completed epoch preserved')
                    loss.backward()
                    nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                    optimizer.step()
                    totals += np.array([float(v.detach()) for v in [loss, cls, dom]])*len(labels)
                    count += len(labels)
                val_args = argparse.Namespace(data_root=args.source_root, batch_size=args.batch_size,
                                               workers=args.workers, device=args.device, seed=args.seed)
                values, _ = base.evaluate(model, val, transform, val_args, 'FF++ clean validation')
                auc = values['macro_method_auc']
                improved = auc > best
                best = max(best, auc)
                if auc > anchor + 1e-4:
                    anchor, stale = auc, 0
                else:
                    stale += 1
                history.append({'epoch': epoch+1, 'loss': float(totals[0]/count),
                                'classification_loss': float(totals[1]/count), 'domain_loss': float(totals[2]/count),
                                'val_macro_auc': auc, 'val_real_fpr': values['frame']['real_fpr'],
                                'val_fake_fnr': values['frame']['fake_fnr'],
                                'target_real_exposures': classes[0], 'target_fake_exposures': classes[1],
                                'target_condition_counts': dict(conditions),
                                'target_smoothing_applied': sum(row['smoothing_applied'] for row in exposure),
                                'target_exposure_sha256': evaluation.key(exposure),
                                'epoch_sample_sha256': evaluation.key(sampled.crop_path.tolist())})
                state = {'identity': identity, 'config': cfg, 'model_state_dict': model.state_dict(),
                         'domain_state_dict': domain.state_dict(), 'optimizer': optimizer.state_dict(),
                         'epoch': epoch+1, 'best': best, 'anchor': anchor, 'stale': stale, 'history': history}
                if improved:
                    base.common.save(run/'best.pt', {k: state[k] for k in ['identity', 'config', 'model_state_dict', 'epoch', 'best']})
                base.common.save(run/'last.pt', state)
                base.common.json_write(run/'history.json', history)
                pd.DataFrame(history).to_csv(run/'history.csv', index=False)
                del state
                print(f'Val AUC={auc:.4f} | FPR={values["frame"]["real_fpr"]:.2%} FNR={values["frame"]["fake_fnr"]:.2%} | patience={stale}/{args.patience}', flush=True)
            base.common.json_write(run/'completed.json', {'identity': identity, 'best_source_val_macro_auc': best,
                                                         'epochs_completed': len(history)})
            del model, domain, optimizer, parameters, groups, heads
            if args.device == 'cuda':
                torch.cuda.empty_cache()
        del initial


def evaluate(args):
    # Reuse all original official-test, cache, checkpoint, and prediction checks.
    original = adaptation.run_name
    adaptation.run_name = run_name
    try:
        adaptation.evaluate(args)
    finally:
        adaptation.run_name = original
    path = args.output_root/'frame_summary.csv'
    frame = pd.read_csv(path)
    frame['balanced_error_rate'] = .5*(frame.real_fpr + frame.fake_fnr)
    frame.to_csv(path, index=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['train', 'evaluate'])
    for field in ['source-root', 'target-root', 'baseline-runs-root', 'runs-root', 'manifest-root',
                  'output-root', 'cache-root', 'baseline-evaluation-root', 'uda-manifest-dir']:
        p.add_argument('--'+field, type=Path, required=True)
    p.add_argument('--protocol', type=Path, default=ROOT/'configs/ffpp_c23_celebdf_smoothing_v1.json')
    p.add_argument('--models', nargs='+', choices=list(base.PREPROCESS), default=list(base.PREPROCESS))
    p.add_argument('--strategies', nargs='+', choices=STRATEGIES, default=STRATEGIES)
    for field, default in [('seed', 42), ('epochs', 10), ('patience', 3), ('source-images', 6000),
                           ('val-images', 1200), ('target-images', 5000), ('batch-size', 16), ('workers', 2)]:
        p.add_argument('--'+field, type=int, default=default)
    for field, default in [('encoder-lr', 1e-5), ('head-lr', 1e-5), ('weight-decay', 1e-4), ('domain-weight', .1)]:
        p.add_argument('--'+field, type=float, default=default)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = p.parse_args()
    if min(args.epochs, args.patience, args.source_images, args.val_images, args.target_images, args.batch_size) < 1 or args.workers < 0:
        p.error('Counts must be positive and workers nonnegative')
    (train if args.action == 'train' else evaluate)(args)


if __name__ == '__main__':
    main()
