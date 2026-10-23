"""Source-only ablation: what does the ranking term add over plain smoothing augmentation?

Every condition starts from the same FF++ baseline and uses the SAME epoch samples, batches, seeds,
trainable blocks, optimiser and selection (FF++ val pooled AUC over 4 smoothing conditions,
max 10 epochs, patience 3). Only the source loss differs:
  clean_only     CE on the original frame only
  aug_single     CE on ONE view per frame: original with p=1/4, else the random-strength smoothed view
  aug_pair       CE on BOTH the original and the smoothed view (ranking weight 0)
  aug_pair_rank  CE on both views + 0.25 x all real/fake softplus ranking (identical to pool v2 ce_rank_pooled)
No Celeb-DF image is used for training or selection. Evaluation: CDF official test x4 conditions,
reporting the smoothing-only metrics (fp_gap / fn_gap, skew, fp/fn flips, paired score shifts) beside
the clean AUC, plus a video bootstrap of both gaps against aug_pair_rank.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import analyze_smoothing_threshold_shift as shift
import ffpp_c23_smoothing_rank_sda_pool_v2 as pool

rank, engine, common, base = pool.rank, pool.engine, pool.common, pool.base
adaptation, evaluation = pool.adaptation, pool.evaluation
ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'ffpp_c23_smoothing_source_ablation_v1'
STRATEGIES = ['clean_only', 'aug_single', 'aug_pair', 'aug_pair_rank']
CONDITIONS = engine.CONDITIONS
CLEAN_PROBABILITY = .25


def code_hashes():
    return {**pool.code_hashes(), Path(__file__).name: base.common.digest(Path(__file__))}


def run_name(model, strategy, seed):
    return f'{model}_{strategy}_{PROTOCOL}_seed{seed}'


def source_loss(strategy, name, model, clean, smooth, labels, ranking_weight, seed):
    """Returns (loss, ce, rank). aug_pair_rank reproduces pool v2 ce_rank_pooled exactly."""
    if strategy in ('aug_pair', 'aug_pair_rank'):
        n = len(labels)
        logits, _ = adaptation.outputs(name, model, torch.cat([clean, smooth]))
        logits = torch.stack([logits[:n], logits[n:]], dim=1)
        loss, ce, aug_pair_rank, _ = rank.classification_and_ranking(logits, labels, ranking_weight if strategy == 'aug_pair_rank' else 0.)
        return loss, ce, aug_pair_rank
    if strategy == 'clean_only':
        x = clean
    else:
        keep = torch.rand(len(labels), generator=torch.Generator().manual_seed(seed)) < CLEAN_PROBABILITY
        x = torch.where(keep.to(clean.device)[:, None, None, None], clean, smooth)
    logits, _ = adaptation.outputs(name, model, x)
    ce = nn.functional.cross_entropy(logits.float(), labels)
    return ce, ce, ce.new_zeros(())


def inputs(args):
    cfg = json.loads(args.protocol.read_text())
    if list(cfg['conditions']) != CONDITIONS:
        raise ValueError('Smoothing condition mismatch')
    source, val, data = common.shared.source_inputs(args)
    return source, val, cfg, data


def configuration(args, name, strategy, data, cfg, initial, initial_hash):
    return dict(protocol=PROTOCOL, model=name, strategy=strategy, seed=args.seed, data=data, smoothing=cfg,
        initial_checkpoint_sha256=initial_hash, preprocessing=base.PREPROCESS[name],
        train_blocks=initial['config']['train_blocks'], epochs=args.epochs, patience=args.patience, min_delta=.0001,
        source_images=args.source_images, val_images=args.val_images, batch_size=args.batch_size,
        encoder_lr=args.encoder_lr, head_lr=args.head_lr, weight_decay=args.weight_decay,
        ranking_weight=args.ranking_weight if strategy == 'aug_pair_rank' else 0., clean_probability=CLEAN_PROBABILITY,
        source_loss={'clean_only': 'CE on original only', 'aug_single': 'CE on one view: original p=0.25 else smoothed (random strength)',
                     'aug_pair': 'CE on original and smoothed views', 'aug_pair_rank': 'CE on both views + ranking_weight*softplus ranking (pool v2 ce_rank_pooled)'}[strategy],
        selection='best.pt: maximum FF++ val pooled raw-margin AUC over 4 conditions; tie: higher minimum cross-condition AUC; epoch0 eligible',
        early_stopping='FF++ pooled AUC, patience/min_delta', target_data='none', bn_statistics='frozen',
        precision='BF16 if supported else FP32 training; FP32 validation/test',
        versions={p: engine.metadata.version(p) for p in ['torch', 'torchvision', 'Pillow', 'numpy', 'pandas', 'scikit-learn']},
        torch=str(torch.__version__), code_sha256=code_hashes())


def without_own_hash(config):
    """Config minus this file's digest: later evaluation-only edits must not invalidate completed runs."""
    hashes = {k: v for k, v in config['code_sha256'].items() if k != Path(__file__).name}
    return {**config, 'code_sha256': hashes}


def validate(args, model, val, transform):
    model.eval()
    return pool.score_conditions(args, model, val, lambda c: common.ValidationViews(val, args, transform, c), 'FF++ val')


def train(args):
    source, val, cfg, data = inputs(args)
    common.prepare_pixel_cache(args, cfg, {k: data[k] for k in ['source_trainval_sha256']})
    common.prepare_validation(args, val, cfg, data={k: data[k] for k in
        ['source_trainval_sha256', 'source_protocol_sha256', 'val_paths_sha256', 'validation_images']})
    engine.freeze_csv(args.manifest_root/'source_validation.csv', val)
    for name in args.models:
        initial, initial_hash = evaluation.checkpoint(argparse.Namespace(runs_root=args.baseline_runs_root, seed=args.seed), name)
        if (initial['config']['data_hashes']['trainval'] != data['source_trainval_sha256']
                or initial['config']['data_hashes']['protocol'] != data['source_protocol_sha256']):
            raise ValueError('Baseline/source mismatch')
        for strategy in args.strategies:
            config = configuration(args, name, strategy, data, cfg, initial, initial_hash)
            identity = evaluation.key(config)
            run = args.runs_root/run_name(name, strategy, args.seed)
            run.mkdir(parents=True, exist_ok=True)
            if (run/'completed.json').is_file() and (run/'config.json').is_file():
                saved = json.loads((run/'config.json').read_text())
                if saved != config and without_own_hash(saved) == without_own_hash(config):
                    if json.loads((run/'completed.json').read_text())['identity'] != evaluation.key(saved) or not (run/'best.pt').is_file():
                        raise ValueError(f'Completed run mismatch: {run}')
                    print('=== ALREADY COMPLETED', name, strategy, '(trained with an earlier revision of this file; only evaluation code changed) ===', flush=True)
                    continue
            engine.freeze_json(run/'config.json', config)
            if (run/'completed.json').is_file():
                if json.loads((run/'completed.json').read_text())['identity'] != identity or not (run/'best.pt').is_file():
                    raise ValueError(f'Completed run mismatch: {run}')
                print('=== ALREADY COMPLETED', name, strategy, '===', flush=True)
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
                print(f'=== RESUME {name} {strategy} from completed epoch {start} ===', flush=True)
            else:
                print(f'=== START NEW {name} {strategy} ===', flush=True)
                values = validate(args, model, val, transform)
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
                loader = DataLoader(common.PairedSource(sampled, args, transform, cfg, epoch), batch_sampler=batches,
                    num_workers=args.workers, pin_memory=args.device == 'cuda', persistent_workers=False,
                    generator=torch.Generator().manual_seed(seed))
                engine.freeze_csv(args.manifest_root/f'source_epoch{epoch+1:02d}.csv', sampled)
                totals, seen = np.zeros(3), 0
                for step, (clean, smooth, labels, _, _) in enumerate(tqdm(loader,
                        desc=f'{name} {strategy} {epoch+1}/{args.epochs}', mininterval=15, dynamic_ncols=True)):
                    base.common.seed_all(seed+step)
                    clean, smooth, labels = clean.to(args.device), smooth.to(args.device), labels.to(args.device)
                    optimizer.zero_grad(set_to_none=True)
                    with base.amp_context(args):
                        loss, ce, aug_pair_rank = source_loss(strategy, name, model, clean, smooth, labels, args.ranking_weight, seed+step)
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite loss; last completed epoch preserved')
                    loss.backward()
                    nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                    optimizer.step()
                    totals += len(labels)*np.array([loss.item(), ce.item(), aug_pair_rank.item()])
                    seen += len(labels)
                values = validate(args, model, val, transform)
                improved = pool.is_better(values, best)
                if improved:
                    best = pool.best_of(values)
                if values['pooled_auc'] > anchor+.0001:
                    anchor, stale = values['pooled_auc'], 0
                else:
                    stale += 1
                history.append(dict(epoch=epoch+1, loss=float(totals[0]/seen), source_ce=float(totals[1]/seen),
                    source_rank=float(totals[2]/seen), source_paths_sha256=evaluation.key(sampled.crop_path.tolist()),
                    source_batches_sha256=evaluation.key(batches), validation=values))
                state = dict(identity=identity, config=config, model_state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                    epoch=epoch+1, stale=stale, history=history, best=best, anchor=anchor)
                if improved:
                    base.common.save(run/'best.pt', {k: state[k] for k in ['identity', 'config', 'model_state_dict', 'epoch']}
                        | dict(best=best, validation=values))
                base.common.save(run/'last.pt', state)
                del state
                base.common.json_write(run/'history.json', history)
                print(f'CE={history[-1]["source_ce"]:.4f} rank={history[-1]["source_rank"]:.4f} | FF++ pooled={values["pooled_auc"]:.4f} '
                      f'strong real/normal fake={values["strong_real_vs_normal_fake_auc"]:.4f} | best={best["pooled_auc"]:.4f} '
                      f'patience={stale}/{args.patience}', flush=True)
            selected = torch.load(run/'best.pt', map_location='cpu', weights_only=True)['epoch']
            base.common.json_write(run/'completed.json', dict(identity=identity, epochs_completed=len(history),
                selected_epoch=selected, best=best))
            del model, optimizer, parameters, groups, head
            if args.device == 'cuda':
                torch.cuda.empty_cache()
        del initial


def auc_gap(labels, normal, strong, rows=None, side='fp'):
    """Clean AUC minus the cross AUC with one class strongly smoothed: fp_gap (side='fp') smooths the
    reals (false-positive direction), fn_gap (side='fn') smooths the fakes (miss direction)."""
    if rows is not None:
        labels, normal, strong = labels[rows], normal[rows], strong[rows]
    real, fake = labels == 0, labels == 1
    scores = np.r_[strong[real], normal[fake]] if side == 'fp' else np.r_[normal[real], strong[fake]]
    cross = pool.fast_auc(np.r_[np.zeros(real.sum()), np.ones(fake.sum())], scores)
    return pool.fast_auc(labels, normal)-cross


def gap_bootstrap(frame, margins_a, margins_b, repeats, seed, side='fp'):
    """Video-cluster bootstrap of gap(a) - gap(b); frames of one video stay together."""
    y = frame.class_index.astype(int).to_numpy()
    index = {}
    for i, video in enumerate(frame.video_id):
        index.setdefault(video, []).append(i)
    pools = [[np.array(index[v]) for v in sorted(index) if y[index[v][0]] == c] for c in [0, 1]]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(repeats):
        rows = np.concatenate([p[j] for p in pools for j in rng.integers(0, len(p), len(p))])
        out.append(auc_gap(y, margins_a['normal'], margins_a['smooth_strong'], rows, side)
                   - auc_gap(y, margins_b['normal'], margins_b['smooth_strong'], rows, side))
    return np.array(out)


def bootstrap_rows(frame, labels, banks, reference, args, extra):
    """Both gap directions of every other condition minus the reference, video-cluster bootstrap."""
    out = []
    for label in banks:
        if label == reference:
            continue
        for side in ['fp', 'fn']:
            metric = f'{side}_gap'
            diff = gap_bootstrap(frame, banks[label], banks[reference], args.bootstrap, args.seed, side)
            observed = (auc_gap(labels, banks[label]['normal'], banks[label]['smooth_strong'], side=side)
                        - auc_gap(labels, banks[reference]['normal'], banks[reference]['smooth_strong'], side=side))
            out.append(dict(**extra, strategy=label, reference=reference, metric=metric, observed=observed,
                            ci_low=float(np.quantile(diff, .025)), ci_high=float(np.quantile(diff, .975)), repeats=args.bootstrap))
    return out


def evaluate(args):
    ea = argparse.Namespace(data_root=args.target_root, manifest=args.target_root/'manifests/celebdf_v2_roi16_test_v1.csv',
        protocol=args.protocol, cache_root=args.cache_root, output_root=args.output_root, workers=args.workers)
    frame, cfg, provenance = evaluation.inputs(ea)
    args.output_root.mkdir(parents=True, exist_ok=True)
    engine.freeze_json(args.output_root/'evaluation_protocol.json', dict(provenance, used_for_selection=False))
    view_cache = args.cache_root if (args.cache_root/'config.json').is_file() and json.loads((args.cache_root/'config.json').read_text()) == provenance else None
    native_cache = common.pixel_cache.BilateralCache(args.evaluation_pixel_cache, cfg['bilateral'],
        dict(data=provenance, cache_code_sha256=base.common.digest(Path(common.pixel_cache.__file__)),
             storage='native uint8 RGB bilateral only; lossless PNG'))
    labels = frame.class_index.astype(int).to_numpy()
    eval_args = argparse.Namespace(**{**vars(args), 'batch_size': args.eval_batch_size})
    rows, boots = [], []
    for name in args.models:
        initial, initial_hash = evaluation.checkpoint(argparse.Namespace(runs_root=args.baseline_runs_root, seed=args.seed), name)
        states = [('baseline', initial['model_state_dict'], initial_hash, 0)]
        for strategy in args.strategies:
            run = args.runs_root/run_name(name, strategy, args.seed)
            done = json.loads((run/'completed.json').read_text())
            state = torch.load(run/'best.pt', map_location='cpu', weights_only=True)
            if state['identity'] != done['identity'] or state['config']['initial_checkpoint_sha256'] != initial_hash:
                raise ValueError(f'Checkpoint mismatch: {run}')
            states.append((strategy, state['model_state_dict'], base.common.digest(run/'best.pt'), state['epoch']))
        banks = {}
        for label, weights, checkpoint_hash, epoch in states:
            model, margins = None, {}
            for condition in CONDITIONS:
                pred = args.output_root/f'{name}_{label}_{condition}.csv'
                if pred.is_file():
                    saved = pd.read_csv(pred, dtype={'crop_path': str})
                    if saved.crop_path.tolist() != frame.crop_path.tolist() or saved.checkpoint_sha256.iloc[0] != checkpoint_hash:
                        raise ValueError(f'Saved predictions differ: {pred}')
                    print('Reuse:', pred.name, flush=True)
                else:
                    if model is None:
                        model, transform = base.build_model(name, pretrained=False)
                        model.load_state_dict(weights, strict=True)
                        model.to(args.device).eval()
                    _, raw = engine.infer(eval_args, model, frame, engine.TestViews(frame, args, transform, cfg, condition, native_cache, view_cache),
                                          f'{name} {label} {condition}')
                    saved = frame[['crop_path', 'video_id', 'class_index']].copy()
                    saved['raw_margin'], saved['checkpoint_sha256'] = raw, checkpoint_hash
                    temp = pred.with_suffix('.tmp.csv'); saved.to_csv(temp, index=False); temp.replace(pred)
                margins[condition] = saved.raw_margin.to_numpy()
            banks[label] = margins
            measured = shift.measure(labels, margins)
            grid = [[pool.fast_auc(np.r_[np.zeros((labels == 0).sum()), np.ones((labels == 1).sum())],
                                   np.r_[margins[r][labels == 0], margins[f][labels == 1]]) for f in CONDITIONS] for r in CONDITIONS]
            rows.append(dict(model=name, strategy=label, selected_epoch=epoch, matched_auc=float(np.mean(np.diag(grid))),
                             **measured))
            print(f"{name} {label}: clean AUC={measured['normal_auc']:.4f} gap={rows[-1]['fp_gap']:.4f} "
                  f"skew={measured['skew']:.4f} flip={measured['fp_flip@normal5']:.2%}", flush=True)
            del model
            if args.device == 'cuda':
                torch.cuda.empty_cache()
        if 'aug_pair_rank' in banks and args.bootstrap:
            boots += bootstrap_rows(frame, labels, banks, 'aug_pair_rank', args, dict(model=name))
    table = pd.DataFrame(rows)
    table.to_csv(args.output_root/'summary.csv', index=False)
    pd.DataFrame(boots).to_csv(args.output_root/'bootstrap_gaps_vs_aug_pair_rank.csv', index=False)
    base.common.json_write(args.output_root/'comparison_protocol.json', dict(protocol=PROTOCOL, strategies=args.strategies,
        primary='fp_gap = clean AUC - AUC(strong real vs clean fake) [false-positive direction]; fn_gap = clean AUC - AUC(clean real vs strong fake) [miss direction]', selection='FF++ val pooled AUC',
        test_used_for_selection=False, normal5_threshold='test-real-label oracle diagnostic only'))
    columns = ['model', 'strategy', 'selected_epoch', 'normal_auc', 'fp_gap', 'fn_gap', 'skew', 'fp_flip@normal5',
               'fn_flip@normal5', 'real_shift', 'fake_shift', 'matched_auc']
    with pd.option_context('display.width', 250, 'display.max_columns', 20):
        print(table[columns].round(4).to_string(index=False))
        if boots:
            print(pd.DataFrame(boots).round(4).to_string(index=False))
    print('Saved:', args.output_root, flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=['train', 'evaluate'])
    for field in ['source-root', 'target-root', 'baseline-runs-root', 'runs-root', 'manifest-root', 'validation-cache', 'output-root']:
        p.add_argument('--'+field, type=Path, required=True)
    p.add_argument('--source-validation', type=Path)
    for field, default in [('train-cache', '/content/smoothing_pivot_bilateral_cache_v1'),
                          ('cache-root', '/content/celebdf_official_smoothing_cache_v1'),
                          ('evaluation-pixel-cache', '/content/smoothing_ranking_cdf_bilateral_cache_v1')]:
        p.add_argument('--'+field, type=Path, default=Path(default))
    p.add_argument('--protocol', type=Path, default=ROOT/'configs/ffpp_c23_celebdf_smoothing_v1.json')
    p.add_argument('--models', nargs='+', choices=list(base.PREPROCESS), default=['dinov2_b14'])
    p.add_argument('--strategies', nargs='+', choices=STRATEGIES, default=STRATEGIES)
    for field, default in [('seed', 42), ('epochs', 10), ('patience', 3), ('source-images', 6000), ('val-images', 1200),
                          ('batch-size', 16), ('eval-batch-size', 64), ('workers', 2), ('bootstrap', 1000)]:
        p.add_argument('--'+field, type=int, default=default)
    for field, default in [('encoder-lr', 1e-5), ('head-lr', 1e-5), ('weight-decay', 1e-4), ('ranking-weight', .25)]:
        p.add_argument('--'+field, type=float, default=default)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    a = p.parse_args(argv)
    if (min(a.epochs, a.patience, a.source_images, a.val_images, a.batch_size, a.eval_batch_size) < 1 or a.batch_size % 2
            or a.source_images % 8 or a.val_images % 8 or a.source_images % a.batch_size or a.workers < 0 or a.bootstrap < 0
            or len(set(a.models)) != len(a.models) or len(set(a.strategies)) != len(a.strategies)):
        p.error('Invalid settings or duplicate models/strategies')
    if a.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime')
    return a


if __name__ == '__main__':
    arguments = parse_args()
    {'train': train, 'evaluate': evaluate}[arguments.action](arguments)
