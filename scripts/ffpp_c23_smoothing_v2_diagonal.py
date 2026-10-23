"""Diagonal experiment under the v2 beauty-smoothing protocol: train with filter k, evaluate on all four filters.

Every run starts from the same FF++ c23 baseline and keeps the source-ablation recipe (same epoch samples, batches,
flips and random weak/medium/strong strength per frame, trainable blocks, optimiser, max 10 epochs, patience 3); only
the smoothing changes to v2 (face-parsing skin mask + one edge-preserving filter, precomputed ffpp_trainval cache):
  aug_single        CE on ONE view: original with p=1/4, else the filter-k view
  aug_pair          CE on the original and the filter-k view
  aug_pair_rank     aug_pair + 0.25 x all real/fake softplus ranking
  aug_pair_rank_fc  aug_pair_rank + lambda x anchored feature consistency: 1 - cos(f(smoothed), stopgrad f(original))
                    on the classifier-head input, detected frames only. Score losses can be met by subtracting a fixed
                    amount whenever the smoothing trace is seen (D1, candidate 3); this pulls the smoothed features onto
                    the original ones so the trace itself disappears, while the original features stay the anchor.
                    One run per --fc-weights value (lambda), named aug_pair_rank_fc<lambda>, so the pilot can compare lambdas.
Selection (best.pt): FF++ val (1,200 frames) pooled raw-margin AUC over original + filter-k weak/medium/strong; no
Celeb-DF frame is used for training or selection. Evaluation lives in evaluate_beauty_smoothing_v2.py.
"""
import argparse
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import beauty_smoothing_v2 as v2
import ffpp_c23_smoothing_source_ablation as ablation

pool, rank, engine, common, base = ablation.pool, ablation.rank, ablation.engine, ablation.common, ablation.base
adaptation, evaluation = ablation.adaptation, ablation.evaluation
ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'ffpp_c23_smoothing_v2_diagonal_v1'
STRATEGIES = ['aug_single', 'aug_pair', 'aug_pair_rank', 'aug_pair_rank_fc']
CONDITIONS = engine.CONDITIONS          # normal, smooth_weak, smooth_medium, smooth_strong
CLEAN_PROBABILITY = ablation.CLEAN_PROBABILITY


def code_hashes():
    return {**pool.code_hashes(), Path(ablation.__file__).name: base.common.digest(Path(ablation.__file__)),
            Path(v2.__file__).name: base.common.digest(Path(v2.__file__)), Path(__file__).name: base.common.digest(Path(__file__))}


def run_name(model, train_filter, label, seed):
    return f'{model}_{train_filter}_{label}_{PROTOCOL}_seed{seed}'


def variants(strategies, fc_weights):
    """(label, strategy, lambda): aug_pair_rank_fc expands to one run per lambda (label aug_pair_rank_fc<lambda>)."""
    out = []
    for strategy in strategies:
        out += ([(f'{strategy}{w:g}', strategy, float(w)) for w in fc_weights] if strategy == 'aug_pair_rank_fc'
                else [(strategy, strategy, 0.)])
    return out


def restore_cache(cache_root, archive_dir, dataset, parts):
    """Unpack the requested v2 cache parts from their Drive tars (once) and check they share one identity."""
    cache = v2.SmoothingCache(cache_root, dataset)
    for part in parts:
        if (cache.root/part/'complete.json').is_file():
            continue
        archive = Path(archive_dir)/f'{dataset}_{part}.tar'
        if not archive.is_file():
            raise FileNotFoundError(f'Build the {dataset} v2 cache first: {archive}')
        print('Restoring', archive, flush=True)
        cache.root.mkdir(parents=True, exist_ok=True)
        subprocess.run(['tar', '-xf', str(archive), '-C', str(cache.root)], check=True)
    ident = cache.check(parts)
    if ident['protocol'] != v2.PROTOCOL or ident['dataset'] != dataset:
        raise ValueError(f'Not a v2 {dataset} cache')
    return cache


class PairedV2(Dataset):
    """Original + filter-k view with the source-ablation per-frame RNG (same flip and strength draw)."""
    def __init__(self, frame, args, transform, cache, epoch):
        self.rows, self.args, self.transform, self.cache, self.epoch = frame.to_dict('records'), args, transform, cache, epoch

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        rng = random.Random(self.args.seed+self.epoch*100003+index)
        flip, strength = rng.random() < .5, rng.randrange(1, 4)
        with Image.open(self.args.source_root/row['crop_path']) as opened:
            clean = opened.convert('RGB')
        detected = row['roi_status'] == 'detected'
        changed = self.cache.view(clean, row['crop_path'], self.args.train_filter, v2.LEVELS[CONDITIONS[strength]], detected)
        if flip:
            clean, changed = clean.transpose(Image.Transpose.FLIP_LEFT_RIGHT), changed.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return self.transform(clean), self.transform(changed), int(row['class_index']), strength, detected


class ValidationV2(Dataset):
    def __init__(self, frame, args, transform, cache, condition):
        self.rows, self.args, self.transform, self.cache = frame.to_dict('records'), args, transform, cache
        self.strength = 0 if condition == 'normal' else v2.LEVELS[condition]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.args.source_root/row['crop_path']) as opened:
            image = opened.convert('RGB')
        image = self.cache.view(image, row['crop_path'], self.args.train_filter, self.strength, row['roi_status'] == 'detected')
        return self.transform(image), index


def feature_consistency(features, n, detected):
    """1 - cos(smoothed, stopgrad original) on the head input, averaged over detected frames (fallbacks are identical)."""
    clean, smooth = features[:n].float().detach(), features[n:].float()
    cos = nn.functional.cosine_similarity(smooth, clean, dim=1)
    mask = detected.float()
    return ((1-cos)*mask).sum()/mask.sum().clamp_min(1.)


def source_loss(strategy, name, model, clean, smooth, labels, detected, args, seed):
    """Returns (loss, ce, rank, fc). The first three strategies reuse the source-ablation losses unchanged."""
    if strategy != 'aug_pair_rank_fc':
        loss, ce, rk = ablation.source_loss(strategy, name, model, clean, smooth, labels, args.ranking_weight, seed)
        return loss, ce, rk, ce.new_zeros(())
    n = len(labels)
    logits, features = adaptation.outputs(name, model, torch.cat([clean, smooth]))
    logits = torch.stack([logits[:n], logits[n:]], dim=1)
    loss, ce, rk, _ = rank.classification_and_ranking(logits, labels, args.ranking_weight)
    fc = feature_consistency(features, n, detected)
    return loss+args.fc_weight*fc, ce, rk, fc


def configuration(args, name, strategy, data, cache_identity, initial, initial_hash):
    weights = dict(ranking_weight=args.ranking_weight if strategy in ('aug_pair_rank', 'aug_pair_rank_fc') else 0.,
                   fc_weight=args.fc_weight if strategy == 'aug_pair_rank_fc' else 0.)
    return dict(protocol=PROTOCOL, model=name, train_filter=args.train_filter, strategy=strategy, seed=args.seed,
        baseline_seed=args.baseline_seed, data=data,
        smoothing=dict(cache_identity, levels=v2.LEVELS), initial_checkpoint_sha256=initial_hash,
        preprocessing=base.PREPROCESS[name], train_blocks=initial['config']['train_blocks'], epochs=args.epochs,
        patience=args.patience, min_delta=.0001, source_images=args.source_images, val_images=args.val_images,
        batch_size=args.batch_size, encoder_lr=args.encoder_lr, head_lr=args.head_lr, weight_decay=args.weight_decay,
        clean_probability=CLEAN_PROBABILITY, **weights,
        selection='best.pt: max FF++ val pooled raw-margin AUC over original + train-filter weak/medium/strong; tie: higher min cross AUC; epoch0 eligible',
        target_data='none', bn_statistics='frozen', precision='BF16 if supported else FP32 training; FP32 validation',
        versions={p: engine.metadata.version(p) for p in ['torch', 'torchvision', 'Pillow', 'numpy', 'pandas', 'scikit-learn']},
        torch=str(torch.__version__), code_sha256=code_hashes())


def comparable(config):
    """What must match for a COMPLETED run to be reused: everything except this file's digest (later evaluation/printing-only
    edits) and the package versions (a newer Colab image must not block reuse of finished pilot/earlier runs). An unfinished run
    still resumes only under the exact identity."""
    hashes = {k: v for k, v in config['code_sha256'].items() if k != Path(__file__).name}
    return {k: v for k, v in config.items() if k not in ('versions', 'torch')} | {'code_sha256': hashes}


def validate(args, model, val, transform, cache):
    model.eval()
    return pool.score_conditions(args, model, val, lambda c: ValidationV2(val, args, transform, cache, c), f'FF++ val {args.train_filter}')


def train(args):
    source, val, data = common.shared.source_inputs(args)
    cache = restore_cache(args.cache_root, args.archive_dir, 'ffpp_trainval', ['mask', args.train_filter])
    cache_identity = {k: cache.identity[k] for k in ['protocol', 'values', 'color_sigma', 'parser', 'manifest_sha256', 'calibration_sha256']}
    if cache.identity['manifest_sha256'] != data['source_trainval_sha256']:
        raise ValueError('v2 cache was built from a different FF++ trainval manifest')
    for name in args.models:
        # All seeds fine-tune the SAME baseline (seed 42); --seed only changes sampling/order/initial randomness.
        initial, initial_hash = evaluation.checkpoint(argparse.Namespace(runs_root=args.baseline_runs_root, seed=args.baseline_seed), name)
        if (initial['config']['data_hashes']['trainval'] != data['source_trainval_sha256']
                or initial['config']['data_hashes']['protocol'] != data['source_protocol_sha256']):
            raise ValueError('Baseline/source mismatch')
        for label, strategy, weight in variants(args.strategies, args.fc_weights):
            args.fc_weight = weight
            config = configuration(args, name, strategy, data, cache_identity, initial, initial_hash)
            identity = evaluation.key(config)
            run = args.runs_root/run_name(name, args.train_filter, label, args.seed)
            run.mkdir(parents=True, exist_ok=True)
            if (run/'completed.json').is_file() and (run/'config.json').is_file():
                saved = json.loads((run/'config.json').read_text())
                if saved != config and comparable(saved) == comparable(config):
                    if json.loads((run/'completed.json').read_text())['identity'] != evaluation.key(saved) or not (run/'best.pt').is_file():
                        raise ValueError(f'Completed run mismatch: {run}')
                    print('=== ALREADY COMPLETED', name, args.train_filter, label, '(reused: only non-training code or package versions differ) ===', flush=True)
                    continue
            engine.freeze_json(run/'config.json', config)
            if (run/'completed.json').is_file():
                if json.loads((run/'completed.json').read_text())['identity'] != identity or not (run/'best.pt').is_file():
                    raise ValueError(f'Completed run mismatch: {run}')
                print('=== ALREADY COMPLETED', name, args.train_filter, label, '===', flush=True)
                continue
            base.common.seed_all(args.seed)
            model, transform = base.build_model(name, pretrained=False)
            model.load_state_dict(initial['model_state_dict'], strict=True)
            model.to(args.device)
            blocks = 2 if name == 'xception' else initial['config']['train_blocks']
            common.training_mode(model, name, blocks)
            head = base.head(model, name)
            head_ids = {id(p) for p in head.parameters()}
            groups = [{'params': [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids], 'lr': args.encoder_lr},
                      {'params': list(head.parameters()), 'lr': args.head_lr}]
            optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
            parameters = [p for group in groups for p in group['params']]
            start, stale, history = 0, 0, []
            if (run/'last.pt').is_file():
                state = torch.load(run/'last.pt', map_location='cpu', weights_only=True)
                if state['identity'] != identity:
                    raise ValueError('Resume identity differs')
                model.load_state_dict(state['model_state_dict'], strict=True)
                optimizer.load_state_dict(state['optimizer'])
                start, stale, history, best, anchor = [state[k] for k in ['epoch', 'stale', 'history', 'best', 'anchor']]
                del state
                print(f'=== RESUME {name} {args.train_filter} {strategy} from completed epoch {start} ===', flush=True)
            else:
                print(f'=== START NEW {name} {args.train_filter} {strategy} ===', flush=True)
                values = validate(args, model, val, transform, cache)
                best, anchor = pool.best_of(values), values['pooled_auc']
                base.common.json_write(run/'epoch0_validation.json', values)
                base.common.save(run/'best.pt', dict(identity=identity, config=config, model_state_dict=model.state_dict(),
                    epoch=0, best=best, validation=values))
            for epoch in range(start, args.epochs):
                if stale >= args.patience:
                    break
                seed = args.seed+epoch*100003
                base.common.seed_all(seed)
                common.training_mode(model, name, blocks)
                sampled = adaptation.balanced_source(source, args.source_images, seed)
                batches = rank.balanced_batches(torch.tensor(sampled.class_index.astype(int).to_numpy()), args.batch_size, seed)
                loader = DataLoader(PairedV2(sampled, args, transform, cache, epoch), batch_sampler=batches,
                    num_workers=args.workers, pin_memory=args.device == 'cuda', persistent_workers=False,
                    generator=torch.Generator().manual_seed(seed))
                # Epoch samples depend only on the seed (shared by every filter/strategy/model of that seed).
                engine.freeze_csv(args.manifest_root/f'seed{args.seed}'/f'source_epoch{epoch+1:02d}.csv', sampled)
                totals, seen = np.zeros(4), 0
                for step, (clean, smooth, labels, _, detected) in enumerate(tqdm(loader,
                        desc=f'{name} {args.train_filter} {strategy} {epoch+1}/{args.epochs}', mininterval=15, dynamic_ncols=True)):
                    base.common.seed_all(seed+step)
                    clean, smooth, labels, detected = (clean.to(args.device), smooth.to(args.device), labels.to(args.device),
                                                       detected.to(args.device))
                    optimizer.zero_grad(set_to_none=True)
                    with base.amp_context(args):
                        loss, ce, rk, fc = source_loss(strategy, name, model, clean, smooth, labels, detected, args, seed+step)
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite loss; last completed epoch preserved')
                    loss.backward()
                    nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                    optimizer.step()
                    totals += len(labels)*np.array([loss.item(), ce.item(), rk.item(), fc.item()])
                    seen += len(labels)
                values = validate(args, model, val, transform, cache)
                improved = pool.is_better(values, best)
                if improved:
                    best = pool.best_of(values)
                if values['pooled_auc'] > anchor+.0001:
                    anchor, stale = values['pooled_auc'], 0
                else:
                    stale += 1
                history.append(dict(epoch=epoch+1, loss=float(totals[0]/seen), source_ce=float(totals[1]/seen),
                    source_rank=float(totals[2]/seen), feature_consistency=float(totals[3]/seen),
                    source_paths_sha256=evaluation.key(sampled.crop_path.tolist()), source_batches_sha256=evaluation.key(batches),
                    validation=values))
                state = dict(identity=identity, config=config, model_state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                    epoch=epoch+1, stale=stale, history=history, best=best, anchor=anchor)
                if improved:
                    base.common.save(run/'best.pt', {k: state[k] for k in ['identity', 'config', 'model_state_dict', 'epoch']}
                        | dict(best=best, validation=values))
                base.common.save(run/'last.pt', state)
                del state
                base.common.json_write(run/'history.json', history)
                print(f'CE={history[-1]["source_ce"]:.4f} rank={history[-1]["source_rank"]:.4f} fc={history[-1]["feature_consistency"]:.4f} '
                      f'| FF++ val pooled={values["pooled_auc"]:.4f} strong real/normal fake={values["strong_real_vs_normal_fake_auc"]:.4f} '
                      f'| best={best["pooled_auc"]:.4f} patience={stale}/{args.patience}', flush=True)
            selected = torch.load(run/'best.pt', map_location='cpu', weights_only=True)['epoch']
            base.common.json_write(run/'completed.json', dict(identity=identity, epochs_completed=len(history),
                selected_epoch=selected, best=best))
            del model, optimizer, parameters, groups, head
            if args.device == 'cuda':
                torch.cuda.empty_cache()
        del initial


def completed_checkpoint(runs_root, name, train_filter, label, seed, initial_hash):
    """(state_dict, best.pt digest, selected epoch) of a completed run started from this baseline."""
    run = Path(runs_root)/run_name(name, train_filter, label, seed)
    done = json.loads((run/'completed.json').read_text())
    state = torch.load(run/'best.pt', map_location='cpu', weights_only=True)
    if state['identity'] != done['identity'] or state['config']['initial_checkpoint_sha256'] != initial_hash:
        raise ValueError(f'Checkpoint mismatch: {run}')
    return state['model_state_dict'], base.common.digest(run/'best.pt'), state['epoch']


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for field in ['source-root', 'baseline-runs-root', 'runs-root', 'manifest-root', 'archive-dir']:
        p.add_argument('--'+field, type=Path, required=True)
    p.add_argument('--source-validation', type=Path)
    p.add_argument('--cache-root', type=Path, default=Path('/content/beauty_smoothing_v2_cache'))
    p.add_argument('--train-filter', choices=v2.FILTERS, required=True)
    p.add_argument('--models', nargs='+', choices=list(base.PREPROCESS), default=['clip_b16'])
    p.add_argument('--strategies', nargs='+', choices=STRATEGIES, default=STRATEGIES)
    for field, default in [('seed', 42), ('baseline-seed', 42), ('epochs', 10), ('patience', 3), ('source-images', 6000), ('val-images', 1200),
                          ('batch-size', 16), ('workers', 2)]:
        p.add_argument('--'+field, type=int, default=default)
    for field, default in [('encoder-lr', 1e-5), ('head-lr', 1e-5), ('weight-decay', 1e-4), ('ranking-weight', .25)]:
        p.add_argument('--'+field, type=float, default=default)
    p.add_argument('--fc-weights', nargs='+', type=float, default=[1.], help='lambda(s) of the feature-consistency term')
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    a = p.parse_args(argv)
    if (min(a.epochs, a.patience, a.source_images, a.val_images, a.batch_size) < 1 or a.batch_size % 2 or a.source_images % 8
            or a.val_images % 8 or a.source_images % a.batch_size or a.workers < 0 or min(a.fc_weights) < 0 or len(set(a.fc_weights)) != len(a.fc_weights)
            or len(set(a.models)) != len(a.models) or len(set(a.strategies)) != len(a.strategies)):
        p.error('Invalid settings or duplicate models/strategies')
    if a.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime')
    return a


if __name__ == '__main__':
    train(parse_args())
