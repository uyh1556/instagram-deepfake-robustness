"""Matched source fine-tuning, smoothing augmentation and DANN for FF++ baselines."""
import argparse
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

import evaluate_ffpp_baselines_celebdf_smoothing as evaluation
import train_ffpp_c23_baseline as base
from pilgram2_roi_pilot import Reverse
from smoothing_roi_pilot import smooth

ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = ['source_ft', 'augmentation', 'uda_normal', 'uda_smoothing']
FEATURES = {'xception': 2048, 'clip_b16': 512, 'dinov2_b14': 768}


def connected_exclusions(videos, test_ids):
    """Conservative filename candidates, NOT verified manipulation direction metadata."""
    ids = set(videos)
    edges = {v: set() for v in ids}
    for video in ids:
        match = re.fullmatch(r'Celeb-synthesis/(id\d+)_(id\d+)_(\d+)', video)
        if match:
            # Consider both identities as possible parent at the same suffix.
            for identity in match.group(1, 2):
                parent = f'Celeb-real/{identity}_{match.group(3)}'
                if parent in ids:
                    edges[video].add(parent)
                    edges[parent].add(video)
    blocked, pending = set(test_ids), list(test_ids)
    while pending:
        for neighbour in edges[pending.pop()]-blocked:
            blocked.add(neighbour)
            pending.append(neighbour)
    return blocked


def balanced_source(frame, n, seed):
    if n % 8:
        raise ValueError('Source/validation count must be divisible by 8')
    parts = [base.video_balanced(frame[frame.method.eq('original')], n//2, seed)]
    for i, method in enumerate(base.METHODS):
        parts.append(base.video_balanced(frame[frame.method.eq(method)], n//8, seed+i+1))
    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def dataset_inputs(args):
    source_path = args.source_root/'manifests/ffpp_c23_4methods_roi16_trainval_v1.csv'
    source = pd.read_csv(source_path, dtype=str, keep_default_na=False)
    if not source.split.isin(['train', 'val']).all() or source.crop_path.duplicated().any():
        raise ValueError('Expected fixed FF++ Train/Val manifest')
    if not source.class_index.astype(int).eq(source.method.ne('original').astype(int)).all():
        raise ValueError('FF++ labels/manipulation provenance mismatch')
    membership = {}
    for row in source[['source_ids', 'split']].drop_duplicates().itertuples(index=False):
        for identity in row.source_ids.split('|'):
            if identity in membership and membership[identity] != row.split:
                raise ValueError('FF++ source-pair overlap')
            membership[identity] = row.split
    target_parts = [pd.read_csv(args.target_root/'manifests'/f'celebdf_v2_roi16_non_test_{kind}_v1.csv',
                               dtype=str, keep_default_na=False) for kind in ['real', 'fake']]
    # Drop all class/method metadata BEFORE target sampling and any training.
    target = pd.concat(target_parts, ignore_index=True)[['crop_path', 'video_id', 'roi_status', 'split']]
    if not target.split.eq('non_test_pool').all() or target.crop_path.duplicated().any():
        raise ValueError('Expected unique non-test target pool')
    test_path = args.target_root/'manifests/celebdf_v2_roi16_test_v1.csv'
    test = pd.read_csv(test_path, dtype=str, keep_default_na=False)
    cfg = json.loads(args.protocol.read_text())
    if len(test) != cfg['test_images'] or test.video_id.nunique() != cfg['test_videos'] or not test.split.eq('test').all():
        raise ValueError('Official test membership missing')
    test_ids = set(test.video_id)
    if test_ids & set(target.video_id):
        raise ValueError('Target/test video overlap')
    blocked = connected_exclusions(set(target.video_id) | test_ids, test_ids)
    eligible = target[~target.video_id.isin(blocked)].reset_index(drop=True)
    if len(eligible) < args.target_images:
        raise ValueError(f'Only {len(eligible)} images after conservative exclusions; target request too large')
    target = base.video_balanced(eligible, args.target_images, args.seed)[['crop_path', 'video_id', 'roi_status']].reset_index(drop=True)
    val = balanced_source(source[source.split.eq('val')], args.val_images, args.seed)
    data = {'source_trainval_sha256': base.common.digest(source_path),
            'source_protocol_sha256': base.common.digest(args.source_root/'manifests/protocol.json'),
            'test_manifest_sha256': base.common.digest(test_path),
            'non_test_manifest_sha256': [base.common.digest(args.target_root/'manifests'/f'celebdf_v2_roi16_non_test_{kind}_v1.csv') for kind in ['real', 'fake']],
            'target_paths_sha256': evaluation.key(target.crop_path.tolist()),
            'val_paths_sha256': evaluation.key(val.crop_path.tolist()),
            'eligible_target_images': len(eligible), 'eligible_target_videos': eligible.video_id.nunique(),
            'selected_target_images': len(target), 'selected_target_videos': target.video_id.nunique(),
            'test_exclusion': 'official test video/frame exclusion plus conservative same-suffix filename parent connected components; identities not partitioned',
            'parent_mapping_limit': 'Filename candidates, not verified source/driver direction; unknown relationships cannot be guaranteed absent.'}
    for root, frame in [(args.source_root, source), (args.target_root, target)]:
        for path in frame.crop_path:
            rel = Path(path)
            if rel.is_absolute() or '..' in rel.parts or not (root/rel).is_file():
                raise FileNotFoundError(root/rel)
    return source[source.split.eq('train')].reset_index(drop=True), val, target, cfg, data, sorted(blocked)


class TrainingViews(Dataset):
    def __init__(self, source, target, args, transform, strategy, epoch, cfg):
        self.source, self.target = source.to_dict('records'), target.to_dict('records')
        self.args, self.transform, self.strategy, self.epoch, self.cfg = args, transform, strategy, epoch, cfg
        self.video_rows = {}
        for row in self.target:
            self.video_rows.setdefault(row['video_id'], []).append(row)
        self.videos = list(self.video_rows)

    def __len__(self):
        return len(self.source)

    def view(self, row, root, condition, flip):
        with Image.open(root/row['crop_path']) as f:
            image = f.convert('RGB')
            if condition != 'normal' and row['roi_status'] == 'detected':
                image = smooth(image, self.cfg['conditions'][condition], self.cfg['bilateral'])
            if flip:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            return self.transform(image)

    def __getitem__(self, index):
        rng = random.Random(self.args.seed+self.epoch*100003+index)
        row = self.source[index]
        flip = rng.random() < .5
        xs = self.view(row, self.args.source_root, 'normal', flip)
        if self.strategy == 'source_ft':
            second = xs.clone()
        elif self.strategy == 'augmentation':
            second = self.view(row, self.args.source_root, rng.choice(list(self.cfg['conditions'])[1:]), flip)
        else:
            target = rng.choice(self.video_rows[rng.choice(self.videos)])
            condition = 'normal' if self.strategy == 'uda_normal' else rng.choice(list(self.cfg['conditions'])[1:])
            second = self.view(target, self.args.target_root, condition, rng.random() < .5)
        return xs, second, int(row['class_index'])


def outputs(name, model, images):
    if name == 'xception':
        features = model.forward_head(model.forward_features(images), pre_logits=True)
        return model.fc(features), features
    if name == 'clip_b16':
        features = model.visual(images)
    else:
        features = model.encoder(pixel_values=images).last_hidden_state[:, 0]
    return model.head(features), features


def loss_for(name, model, domain, xs, second, labels, uda, grl, weight):
    logits, features = outputs(name, model, torch.cat([xs, second]))
    n = len(labels)
    if uda:
        cls = nn.functional.cross_entropy(logits[:n].float(), labels)
        domain_labels = torch.cat([torch.zeros_like(labels), torch.ones_like(labels)])
        dom = nn.functional.cross_entropy(domain(Reverse.apply(features.float(), grl)), domain_labels)
        return cls+weight*dom, cls, dom
    cls = nn.functional.cross_entropy(logits.float(), torch.cat([labels, labels]))
    return cls, cls, cls.new_zeros(())


def run_name(name, strategy, seed):
    return f'{name}_{strategy}_ffpp_c23_celebdf_smoothing_v1_seed{seed}'


def train(args):
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('GPU runtime required')
    pool, val, target, cfg, data, blocked = dataset_inputs(args)
    args.manifest_root.mkdir(parents=True, exist_ok=True)
    frozen = args.manifest_root/'protocol.json'
    frozen_value = {'data': data, 'seed': args.seed, 'source_images_per_epoch': args.source_images,
                    'val_images': args.val_images, 'target_images': args.target_images}
    if frozen.exists() and json.loads(frozen.read_text()) != frozen_value:
        raise ValueError('Frozen adaptation data differs; use a new manifest root')
    base.common.json_write(frozen, frozen_value)
    target.to_csv(args.manifest_root/'target_unlabelled.csv', index=False)
    val.to_csv(args.manifest_root/'source_validation.csv', index=False)
    base.common.json_write(args.manifest_root/'excluded_target_video_ids.json', blocked)
    print(f'Source/epoch={args.source_images}, validation={len(val)}, target={len(target)} from {target.video_id.nunique()} videos', flush=True)
    print('Target pool after conservative exclusion:', data['eligible_target_images'], 'images;', data['eligible_target_videos'], 'videos', flush=True)
    # Check all requested baseline completions before training any strategy.
    initial_args = argparse.Namespace(runs_root=args.baseline_runs_root, seed=args.seed)
    for name in args.models:
        initial, _ = evaluation.checkpoint(initial_args, name)
        if (initial['config']['data_hashes']['trainval'] != data['source_trainval_sha256']
                or initial['config']['data_hashes']['protocol'] != data['source_protocol_sha256']):
            raise ValueError('Source data differs from initial baseline')
        del initial
    for name in args.models:
        initial, initial_hash = evaluation.checkpoint(initial_args, name)
        for strategy in args.strategies:
            configuration = {'protocol': 'ffpp_c23_celebdf_smoothing_adaptation_v1', 'model': name,
                'strategy': strategy, 'seed': args.seed, 'data': data,
                'initial_checkpoint_sha256': initial_hash, 'preprocessing': base.PREPROCESS[name],
                'epochs': args.epochs, 'patience': args.patience, 'min_delta': 1e-4,
                'source_images_per_epoch': args.source_images, 'batch_size': args.batch_size,
                'encoder_lr': args.encoder_lr, 'head_lr': args.head_lr, 'weight_decay': args.weight_decay,
                'domain_weight': args.domain_weight, 'smoothing': cfg, 'train_blocks': initial['config']['train_blocks'],
                'batchnorm': 'running statistics frozen for every strategy',
                'loss': 'source_ft/augmentation: mean CE on clean+second labelled view; UDA: clean source CE + weighted domain CE, target class labels never used',
                'domain_head': 'LayerNorm + Linear128/ReLU/Linear2, progressive GRL',
                'sampling': 'same source epoch list, Real:Fake1:1, equal four Fake methods; target video-uniform with replacement from frozen unlabelled pool',
                'checkpoint_selection': 'maximum clean FF++ fixed validation subset macro4 frame AUC after an adaptation epoch; not target/test',
                'precision': 'BF16 if supported otherwise FP32; finite checks, no FP16',
                'code_sha256': base.common.digest(Path(__file__)),
                'model_code_sha256': base.common.digest(Path(base.__file__)),
                'shared_code_sha256': base.common.digest(Path(base.common.__file__)),
                'smoothing_code_sha256': base.common.digest(ROOT/'scripts/smoothing_roi_pilot.py'),
                'opencv': evaluation.cv2.__version__,
                'baseline_package_versions': initial['config'].get('versions', {}),
                'torch': str(torch.__version__)}
            identity = evaluation.key(configuration)
            run = args.runs_root/run_name(name, strategy, args.seed)
            run.mkdir(parents=True, exist_ok=True)
            if (run/'config.json').exists() and json.loads((run/'config.json').read_text()) != configuration:
                raise ValueError(f'Run config differs: {run}')
            if (run/'completed.json').exists():
                if json.loads((run/'completed.json').read_text())['identity'] != identity or not (run/'best.pt').is_file():
                    raise ValueError('Completed run mismatch')
                print('Skip completed:', name, strategy, flush=True)
                continue
            base.common.json_write(run/'config.json', configuration)
            base.common.seed_all(args.seed)
            model, transform = base.build_model(name, pretrained=False)
            model.load_state_dict(initial['model_state_dict'], strict=True)
            model.to(args.device)
            blocks = initial['config']['train_blocks'] if name != 'xception' else 2
            base.stage(model, name, blocks, False)
            domain = nn.Sequential(nn.LayerNorm(FEATURES[name]), nn.Linear(FEATURES[name], 128),
                                   nn.ReLU(), nn.Linear(128, 2)).to(args.device)
            uda = strategy.startswith('uda_')
            head_ids = {id(p) for p in base.head(model, name).parameters()}
            encoder = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
            groups = [{'params': encoder, 'lr': args.encoder_lr},
                      {'params': list(base.head(model, name).parameters()), 'lr': args.head_lr}]
            if uda:
                groups.append({'params': list(domain.parameters()), 'lr': args.head_lr})
            optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
            parameters = [p for group in groups for p in group['params']]
            start, best, anchor, stale, history = 0, -1., -1., 0, []
            if (run/'last.pt').exists():
                state = torch.load(run/'last.pt', map_location='cpu', weights_only=True)
                if state['identity'] != identity:
                    raise ValueError('Resume protocol mismatch')
                model.load_state_dict(state['model_state_dict'], strict=True)
                domain.load_state_dict(state['domain_state_dict'], strict=True)
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
                domain.train()
                sampled = balanced_source(pool, args.source_images, epoch_seed)
                dataset = TrainingViews(sampled, target, args, transform, strategy, epoch, cfg)
                loader = base.common.make_loader(dataset, args, True, epoch_seed)
                totals, count = np.zeros(3), 0
                for step, (xs, second, labels) in enumerate(tqdm(loader, desc=f'{name} {strategy} {epoch+1}/{args.epochs}',
                                                                mininterval=10, dynamic_ncols=True), 1):
                    xs, second, labels = xs.to(args.device), second.to(args.device), labels.to(args.device)
                    optimizer.zero_grad(set_to_none=True)
                    progress = (epoch*len(loader)+step)/(args.epochs*len(loader))
                    grl = 2/(1+math.exp(-10*progress))-1
                    with base.amp_context(args):
                        loss, cls, dom = loss_for(name, model, domain, xs, second, labels, uda, grl, args.domain_weight)
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite loss; no batch skipped, last completed epoch preserved')
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
                if auc > anchor+1e-4:
                    anchor, stale = auc, 0
                else:
                    stale += 1
                history.append({'epoch': epoch+1, 'loss': float(totals[0]/count), 'classification_loss': float(totals[1]/count),
                                'domain_loss': float(totals[2]/count), 'val_macro_auc': auc,
                                'val_real_fpr': values['frame']['real_fpr'],
                                'epoch_sample_sha256': evaluation.key(sampled.crop_path.tolist())})
                state = {'identity': identity, 'config': configuration, 'model_state_dict': model.state_dict(),
                         'domain_state_dict': domain.state_dict(), 'optimizer': optimizer.state_dict(),
                         'epoch': epoch+1, 'best': best, 'anchor': anchor, 'stale': stale, 'history': history}
                if improved:
                    base.common.save(run/'best.pt', {k: state[k] for k in ['identity', 'config', 'model_state_dict', 'epoch', 'best']})
                base.common.save(run/'last.pt', state)
                del state
                base.common.json_write(run/'history.json', history)
                pd.DataFrame(history).to_csv(run/'history.csv', index=False)
                print(f'Validation AUC={auc:.4f} | best={best:.4f} | patience={stale}/{args.patience}', flush=True)
            base.common.json_write(run/'completed.json', {'identity': identity, 'best_source_val_macro_auc': best,
                                                         'epochs_completed': len(history)})
            del model, domain, optimizer, parameters, groups, encoder
            if args.device == 'cuda':
                torch.cuda.empty_cache()
        del initial


def evaluate(args):
    eval_args = argparse.Namespace(data_root=args.target_root,
        manifest=args.target_root/'manifests/celebdf_v2_roi16_test_v1.csv', protocol=args.protocol,
        cache_root=args.cache_root, output_root=args.output_root, workers=args.workers)
    evaluation.prepare(eval_args)
    frame, cfg, provenance = evaluation.inputs(eval_args)
    baseline_protocol = json.loads((args.baseline_evaluation_root/'evaluation_protocol.json').read_text())
    if any(baseline_protocol.get(k) != v for k, v in provenance.items()):
        raise ValueError('Baseline smoothing/manifest/OpenCV protocol differs')
    records = []
    for name in args.models:
        initial_args = argparse.Namespace(runs_root=args.baseline_runs_root, seed=args.seed)
        initial, initial_hash = evaluation.checkpoint(initial_args, name)
        del initial
        # Reuse baseline predictions/results; never re-run the baseline here.
        for condition in cfg['conditions']:
            path = args.baseline_evaluation_root/f'{name}_{condition}.json'
            if not path.is_file():
                raise FileNotFoundError(f'Finish the baseline evaluation Colab first: {path}')
            saved = json.loads(path.read_text())
            if (saved['checkpoint_sha256'] != initial_hash or saved['manifest_sha256'] != provenance['manifest_sha256']
                    or saved['precision'] != 'FP32' or saved['images'] != len(frame)):
                raise ValueError('Baseline evaluation protocol/checkpoint differs')
            records.append({'model': name, 'strategy': 'baseline', 'condition': condition,
                            **{k: saved[k] for k in ['auc', 'accuracy', 'f1', 'real_fpr', 'fake_fnr']}})
        for strategy in args.strategies:
            run = args.runs_root/run_name(name, strategy, args.seed)
            completed = json.loads((run/'completed.json').read_text())
            state = torch.load(run/'best.pt', map_location='cpu', weights_only=True)
            configuration = json.loads((run/'config.json').read_text())
            if (configuration != state['config'] or state['identity'] != evaluation.key(configuration)
                    or completed['identity'] != state['identity'] or configuration['model'] != name
                    or configuration['strategy'] != strategy or configuration['seed'] != args.seed
                    or configuration['initial_checkpoint_sha256'] != initial_hash
                    or configuration['data']['test_manifest_sha256'] != provenance['manifest_sha256']
                    or configuration['preprocessing'] != base.PREPROCESS[name]
                    or configuration['smoothing'] != cfg):
                raise ValueError('Adapted checkpoint/evaluation mismatch')
            checkpoint_hash = base.common.digest(run/'best.pt')
            result_identity = evaluation.key({'data': provenance, 'checkpoint': checkpoint_hash,
                'precision': 'FP32', 'torch': str(torch.__version__),
                'code_sha256': base.common.digest(Path(__file__)),
                'model_code_sha256': base.common.digest(Path(base.__file__)),
                'shared_code_sha256': base.common.digest(Path(base.common.__file__))})
            model, transform = None, None
            for condition in cfg['conditions']:
                prefix = args.output_root/f'{name}_{strategy}_{condition}'
                report, predictions = prefix.with_suffix('.json'), prefix.with_suffix('.csv')
                values = None
                if report.exists() and predictions.exists():
                    saved = json.loads(report.read_text())
                    predicted = pd.read_csv(predictions, dtype={'crop_path': str, 'class_index': str})
                    if saved.get('identity') != result_identity:
                        raise ValueError('Previous evaluation differs; use new output root')
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
                    dataset = evaluation.Views(frame, eval_args, transform, condition)
                    loader = base.common.make_loader(dataset, args)
                    scores = np.empty(len(frame))
                    with torch.inference_mode():
                        for images, indexes in tqdm(loader, desc=f'{name} {strategy} {condition}', mininterval=10, dynamic_ncols=True):
                            p = model(images.to(args.device)).float().softmax(1)[:, 1].cpu().numpy()
                            if not np.isfinite(p).all():
                                raise RuntimeError('Nonfinite evaluation')
                            scores[indexes.numpy()] = p
                    predicted = frame[['crop_path', 'video_id', 'class_index', 'roi_status']].copy()
                    predicted['probability_fake'] = scores
                    temp = predictions.with_suffix('.tmp.csv')
                    predicted.to_csv(temp, index=False)
                    temp.replace(predictions)
                    values = base.binary_metrics(frame.class_index.astype(int).to_numpy(), scores)
                    values.update(identity=result_identity, checkpoint_sha256=checkpoint_hash, unit='frame', threshold=.5)
                    base.common.json_write(report, values)
                records.append({'model': name, 'strategy': strategy, 'condition': condition,
                                **{k: values[k] for k in ['auc', 'accuracy', 'f1', 'real_fpr', 'fake_fnr']}})
                print(f'{name} {strategy} {condition}: AUC={values["auc"]:.4f}, FPR={values["real_fpr"]:.2%}, FNR={values["fake_fnr"]:.2%}', flush=True)
            del model
            if 'state' in locals():
                del state
            if args.device == 'cuda':
                torch.cuda.empty_cache()
    pd.DataFrame(records).to_csv(args.output_root/'frame_summary.csv', index=False)
    print('Saved:', args.output_root, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['train', 'evaluate'])
    for field in ['source-root', 'target-root', 'baseline-runs-root', 'runs-root', 'manifest-root',
                  'output-root', 'cache-root', 'baseline-evaluation-root']:
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
    p.add_argument('--domain-weight', type=float, default=.1)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = p.parse_args()
    if min(args.epochs, args.patience, args.source_images, args.val_images, args.target_images, args.batch_size) < 1:
        p.error('Counts and learning duration must be positive')
    (train if args.action == 'train' else evaluate)(args)


if __name__ == '__main__':
    main()
