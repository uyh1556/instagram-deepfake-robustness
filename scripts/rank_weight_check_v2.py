"""Rank-weight sensitivity of the cross-ranking baseline (교차 순위), seed 42, one backbone per call.

Question (defense): was the ranking term (alpha = .25 in the thesis) simply too weak to help? Training is the
unchanged v2 diagonal trainer (ffpp_c23_smoothing_v2_diagonal.py, strategy aug_pair_rank) with --ranking-weight
alpha (default 1, 2, 4); every other setting is the thesis one (start from baseline seed 42, same epoch samples,
batches, flips and strength draws, max 10 epochs / patience 3, best.pt by FF++ val pooled AUC, epoch 0 eligible).
Each alpha has its own runs root (runs/ffpp_c23_rank_weight_v2_<model>/alpha<a>) so the thesis runs stay untouched.
The epoch-sample manifests are shared across alphas, and this script checks that every new run saw exactly the
samples/batches of the thesis alpha=.25 run of the same filter and that its config differs only in the weight.

collect (inference only; Celeb-DF v2 official test, never used for training or selection):
  * new runs: raw margins on the original and the four filters at strong strength (.75, the thesis main metric),
    saved like evaluate_beauty_smoothing_v2.py (same file names), and the FF++ val mean/std of each run (the 1,200
    frames of the NSF scorer) for the eye/mouth fusion. Also the FF++ val statistics of the thesis alpha=.25 run.
analyze (no inference; seed 42 only for every row):
  * references read from the thesis outputs: 동시 증강 (aug_pair) and 교차 순위 alpha=.25 predictions, and the
    pair+NSF calibration (the proposed method).
  * every method also gets "+ 눈·입" with the thesis fusion: w = .2, the original scorer, score B computed on the same
    input image as the detector (nsf_smoothed_b_v2 scores: clean for the original, smooth_all for smoothed views).
  * reproduction checks: aug_pair / alpha=.25 cross AUCs must equal the saved diagonal summary and pair+NSF must equal
    the saved smoothed-B result (same predictions and statistics), otherwise it stops.
Report per method: original AUC, worst cross AUC (seen 4 / unseen 12 / all 16), cells above 동시 증강 and above the
proposed method, selected epochs, mean training CE / rank loss and alpha x rank / CE. All alphas are reported; nothing
is chosen on Celeb-DF.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import evaluate_beauty_smoothing_v2 as ebs
import evaluate_pair_nsf_v2 as pair_nsf
import nonskin_fusion_v2 as nsf
import nsf_region_controls_v2 as rc

diagonal, v2, base, evaluation = ebs.diagonal, ebs.v2, ebs.base, ebs.evaluation
FILTERS = list(v2.FILTERS)
VIEWS = ['normal', *FILTERS]
STRONG = v2.LEVELS['smooth_strong']
REFERENCE_ALPHA = .25
SEED = 42
PROTOCOL = 'rank_weight_check_v2_v1'
RANK, PAIR = 'aug_pair_rank', 'aug_pair'
IGNORED = ('ranking_weight', 'code_sha256', 'versions', 'torch')   # may differ between the thesis run and a new alpha
NAMES = {PAIR: '동시 증강'}


def tag(alpha):
    return f'alpha{alpha:g}'


def method(alpha):
    return PAIR if alpha is None else f'{RANK}_{tag(alpha)}'


def display_name(alpha, fused=False):
    name = NAMES[PAIR] if alpha is None else f'교차 순위 α={alpha:g}'
    if fused:
        return '동시 증강 + 눈·입 (제안 방법)' if alpha is None else name+' + 눈·입'
    return name


def stem(view):
    return 'normal' if view == 'normal' else f'{view}_smooth_strong'


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    base.common.json_write(path, value)


def run_dir(runs_root, model, filt, label=RANK):
    return Path(runs_root)/diagonal.run_name(model, filt, label, SEED)


def runs_root(args, alpha):
    return args.original_runs if alpha == REFERENCE_ALPHA else args.runs_base/tag(alpha)


def prediction_root(args, alpha):
    """Saved CDF test margins: thesis evaluation folder for the references, this experiment's folder otherwise."""
    if alpha is None or alpha == REFERENCE_ALPHA:
        return args.original_eval/f'seed{SEED}'/'cdf_test'
    return args.output_root/tag(alpha)/'cdf_test'


def prediction_path(args, alpha, filt, view):
    label = PAIR if alpha is None else RANK
    return prediction_root(args, alpha)/f'{args.model}_{label}@{filt}_{stem(view)}.csv'


def calibration_path(args, alpha, filt):
    if alpha is None:
        return args.pair_nsf_root/'calibration'/f'seed{SEED}_{filt}.json'
    return args.output_root/'calibration'/f'{tag(alpha)}_{filt}.json'


def same_samples(history, reference, run):
    """Every epoch both runs trained must have used the same epoch sample and batch order (paired comparison)."""
    for a, b in zip(history, reference):
        if (a['epoch'] != b['epoch'] or a['source_paths_sha256'] != b['source_paths_sha256']
                or a['source_batches_sha256'] != b['source_batches_sha256']):
            raise ValueError(f'Epoch {a["epoch"]} samples/batches differ from the thesis alpha=.25 run: {run}')
    return min(len(history), len(reference))


def check_run(args, alpha, filt, initial_hash):
    """Config/provenance of one rank run; new alphas must equal the thesis run except for the weight."""
    run = run_dir(runs_root(args, alpha), args.model, filt)
    for part in ['config.json', 'completed.json', 'history.json', 'best.pt']:
        if not (run/part).is_file():
            raise FileNotFoundError(f'Train this run first: {run/part}')
    config = read_json(run/'config.json')
    if (config['strategy'] != RANK or config['model'] != args.model or config['train_filter'] != filt
            or int(config['seed']) != SEED or abs(config['ranking_weight']-alpha) > 1e-12
            or config['initial_checkpoint_sha256'] != initial_hash):
        raise ValueError(f'Unexpected run settings (strategy/model/filter/seed/weight/baseline): {run}')
    if alpha != REFERENCE_ALPHA:
        ref = run_dir(args.original_runs, args.model, filt)
        old = read_json(ref/'config.json')
        a = {k: v for k, v in config.items() if k not in IGNORED}
        b = {k: v for k, v in old.items() if k not in IGNORED}
        if a != b:
            raise ValueError(f'Settings other than the ranking weight differ from the thesis run: '
                             f'{sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))}')
        same_samples(read_json(run/'history.json'), read_json(ref/'history.json'), run)
    return run


def training_summary(run, alpha):
    history, done = read_json(run/'history.json'), read_json(run/'completed.json')
    ce = float(np.mean([h['source_ce'] for h in history])) if history else float('nan')
    rk = float(np.mean([h['source_rank'] for h in history])) if history else float('nan')
    return dict(selected_epoch=int(done['selected_epoch']), epochs_trained=len(history), train_ce=ce,
                train_rank=rk if alpha is not None else float('nan'),
                weighted_rank_over_ce=alpha*rk/ce if alpha is not None and history else float('nan'))


# ---------------------------------------------------------------- collect (GPU)
def prediction_ready(path, digest, order):
    if not path.is_file():
        return False
    saved = pd.read_csv(path, dtype={'crop_path': str})
    if saved.crop_path.tolist() != order:
        raise ValueError(f'Saved predictions have another frame order: {path}')
    return bool(saved.checkpoint_sha256.eq(digest).all())


def calibration_identity(args, alpha, filt, digest, completed, info):
    return dict(protocol=PROTOCOL, model=args.model, seed=SEED, train_filter=filt, source_method=RANK,
                ranking_weight=alpha, source_checkpoint_sha256=digest, source_identity=completed['identity'],
                settings=nsf.SETTINGS, scorer_pickle_sha256=info['pickle_sha256'], val_paths_sha256=info['val_paths_sha256'])


def collect(args):
    info = read_json(args.original_runs/'nonskin_probe'/f'{args.model}_face_inner_seed42.json')
    if info['settings'] != nsf.SETTINGS:
        raise ValueError('Original NSF scorer settings differ')
    _, initial_hash = evaluation.checkpoint(argparse.Namespace(runs_root=args.baseline_runs_root, seed=SEED), args.model)
    if initial_hash != info['baseline_sha256']:
        raise ValueError('Baseline checkpoint differs from the one behind the NSF scorer')
    ea = argparse.Namespace(dataset='cdf_test', data_root=args.cdf_root, protocol=args.protocol, output_root=args.output_root,
                            workers=args.workers)
    frame, _ = ebs.test_inputs(ea)
    order = frame.crop_path.tolist()
    reference = pair_nsf.read_prediction(prediction_path(args, None, FILTERS[0], 'normal'))
    if reference.crop_path.tolist() != order:
        raise ValueError('CDF test order differs from the thesis predictions')
    val = pair_nsf.val_frame(argparse.Namespace(source_root=args.source_root, runs_root=args.original_runs, model=args.model))
    infer_args = argparse.Namespace(batch_size=args.batch_size, workers=args.workers, device=args.device)
    predict_args = argparse.Namespace(output_root=None, data_root=args.cdf_root, eval_batch_size=args.batch_size,
                                      workers=args.workers, device=args.device, batch_size=args.batch_size)
    cache = None
    for alpha in [REFERENCE_ALPHA, *args.alphas]:
        for i, filt in enumerate(FILTERS, 1):
            run = check_run(args, alpha, filt, initial_hash)
            digest, completed = base.common.digest(run/'best.pt'), read_json(run/'completed.json')
            epoch = completed['selected_epoch']
            identity = calibration_identity(args, alpha, filt, digest, completed, info)
            cal_path = calibration_path(args, alpha, filt)
            need_cal = not cal_path.is_file() or read_json(cal_path)['identity'] != identity
            if alpha == REFERENCE_ALPHA:     # thesis predictions are reused; they must belong to this best.pt
                for view in VIEWS:
                    if not prediction_ready(prediction_path(args, alpha, filt, view), digest, order):
                        raise ValueError(f'Thesis alpha=.25 predictions do not match its best.pt: {filt} {view}')
                need_pred = []
            else:
                need_pred = [v for v in VIEWS if not prediction_ready(prediction_path(args, alpha, filt, v), digest, order)]
            print(f'[{tag(alpha)} {i}/4 {filt}] selected epoch {epoch} | predictions to infer: {len(need_pred)} | '
                  f'val statistics: {"infer" if need_cal else "cached"}', flush=True)
            if not need_pred and not need_cal:
                continue
            weights, verified, epoch = diagonal.completed_checkpoint(runs_root(args, alpha), args.model, filt, RANK, SEED, initial_hash)
            if verified != digest:
                raise ValueError(f'best.pt changed while reading: {run}')
            model, transform = nsf.load(args, args.model, weights)
            state = dict(model=model, transform=transform)
            if need_pred:
                if cache is None:
                    cache = diagonal.restore_cache(args.cache_root, args.archive_dir, 'cdf_test', v2.SmoothingCache.PARTS)
                predict_args.output_root = prediction_root(args, alpha)
                predict_args.output_root.mkdir(parents=True, exist_ok=True)
                for view in need_pred:
                    ebs.predict(predict_args, frame, args.model, f'{RANK}@{filt}', weights, digest, cache,
                                None if view == 'normal' else view, 'normal' if view == 'normal' else 'smooth_strong',
                                0 if view == 'normal' else STRONG, state)
            if need_cal:
                dataset = nsf.fusion.Inputs(val, args.source_root, None, 'full', transform, args.work_dir)
                _, raw = diagonal.engine.infer(infer_args, model, val, dataset, f'{args.model} {tag(alpha)} {filt}: FF++ val 1200')
                raw = np.asarray(raw, dtype=np.float32)       # same precision as the thesis NSF statistics
                if not np.isfinite(raw).all() or raw.std() <= 0:
                    raise ValueError('Invalid validation statistics')
                write_json(cal_path, dict(identity=identity, selected_epoch=epoch, rank_val_mean=float(raw.mean()),
                                          rank_val_std=float(raw.std()), scorer_val_mean=info['val_mean'],
                                          scorer_val_std=info['val_std'], weight=nsf.SETTINGS['weight']))
            state.clear()
            del model, weights
            if args.device == 'cuda':
                torch.cuda.empty_cache()
    write_json(args.output_root/'protocol.json', dict(protocol=PROTOCOL, model=args.model, seed=SEED, alphas=args.alphas,
        reference_alpha=REFERENCE_ALPHA, strength=STRONG, weight=nsf.SETTINGS['weight'], b_input='same image as the detector',
        target_used_for_training_or_selection=False))
    analyze(args)


# ---------------------------------------------------------------- analyze (CPU, reads saved files)
def b_scores(args, order):
    """z-scored eye/mouth score B per view, computed on the same input image as the detector."""
    info = read_json(args.original_runs/'nonskin_probe'/f'{args.model}_face_inner_seed42.json')
    identity = read_json(args.smoothed_b_root/'scores'/'identity.json')['identity']
    if identity['scorer_pickle_sha256'] != info['pickle_sha256'] or identity['model'] != args.model:
        raise ValueError('Smoothed-B scores were made with another scorer/backbone')
    out = {}
    for view in VIEWS:
        t = pd.read_csv(args.smoothed_b_root/'scores'/f'cdf_test_{view}.csv', dtype={'crop_path': str})
        if t.crop_path.tolist() != order:
            raise ValueError(f'B score order differs: {view}')
        raw = t['clean' if view == 'normal' else 'smooth_all'].to_numpy()
        out[view] = (raw-info['val_mean'])/info['val_std']
    return out


def margins(args, alpha, filt, order, digest=None):
    out = {}
    for view in VIEWS:
        path = prediction_path(args, alpha, filt, view)
        pred = pair_nsf.read_prediction(path)
        if pred.crop_path.tolist() != order:
            raise ValueError(f'Prediction order differs: {path}')
        if digest is not None and not pred.checkpoint_sha256.eq(digest).all():
            raise ValueError(f'Prediction does not belong to the checkpoint behind its statistics: {path}')
        out[view] = pred.raw_margin.to_numpy()
    return out


def method_rows(args, alpha, order, y, b, w):
    rows, train = [], []
    for filt in FILTERS:
        cal = read_json(calibration_path(args, alpha, filt))
        a = margins(args, alpha, filt, order, cal['identity']['source_checkpoint_sha256'])
        z = {v: (a[v]-cal['rank_val_mean'])/cal['rank_val_std'] for v in VIEWS}
        run = run_dir(args.original_runs if alpha in (None, REFERENCE_ALPHA) else args.runs_base/tag(alpha), args.model, filt,
                      PAIR if alpha is None else RANK)
        train.append(dict(method=method(alpha), train_filter=filt, **training_summary(run, alpha)))
        for ev in FILTERS:
            common = dict(model=args.model, seed=SEED, alpha=alpha, train_filter=filt, filter=ev, held_out=ev != filt)
            rows.append(dict(common, method=method(alpha), name=display_name(alpha), fused=False, **rc.metrics(y, a['normal'], a[ev])))
            rows.append(dict(common, method=method(alpha)+'+nsf', name=display_name(alpha, True), fused=True,
                             **rc.metrics(y, (1-w)*z['normal']+w*b['normal'], (1-w)*z[ev]+w*b[ev])))
    return rows, train


def reproduce(args, rows):
    """Reference rows must equal the saved thesis numbers (same predictions; pair+NSF also same statistics and B)."""
    t = pd.DataFrame(rows)
    old = pd.read_csv(prediction_root(args, None)/'summary.csv')
    for m, label in [(PAIR, PAIR), (method(REFERENCE_ALPHA), RANK)]:
        for r in t[t.method.eq(m)].itertuples():
            ref = old[old.method.eq(label) & old.train_filter.eq(r.train_filter) & old['filter'].eq(r.filter)]
            if len(ref) != 1 or abs(ref.fp_cross_auc.iloc[0]-r.fp_cross_auc) > 1e-9 or abs(ref.fn_cross_auc.iloc[0]-r.fn_cross_auc) > 1e-9:
                raise ValueError(f'{label} does not reproduce the saved diagonal summary: {r.train_filter}->{r.filter}')
    saved = pd.read_csv(args.smoothed_b_root/f'seed{SEED}'/'summary.csv')
    saved = saved[saved.variant.eq('smooth_all')]
    for r in t[t.method.eq(PAIR+'+nsf')].itertuples():
        ref = saved[saved.train_filter.eq(r.train_filter) & saved['filter'].eq(r.filter)]
        if len(ref) != 1 or abs(ref.fp_cross_auc.iloc[0]-r.fp_cross_auc) > 1e-9 or abs(ref.fn_cross_auc.iloc[0]-r.fn_cross_auc) > 1e-9:
            raise ValueError(f'pair+NSF does not reproduce the saved thesis result: {r.train_filter}->{r.filter}')
    print('Reproduced: 동시 증강 / 교차 순위 α=0.25 (diagonal summary) and 제안 방법 (pair+NSF, B on the input), 16/16 cells each', flush=True)


def headline(rows, train):
    t, tr = pd.DataFrame(rows), pd.DataFrame(train)
    cell = lambda m: t[t.method.eq(m)].set_index(['train_filter', 'filter']).worse_cross_auc
    pair, proposed = cell(PAIR), cell(PAIR+'+nsf')
    out = []
    for m, g in t.groupby('method', sort=False):
        base_m = m.removesuffix('+nsf')
        info = tr[tr.method.eq(base_m)]
        out.append(dict(method=m, name=g.name.iloc[0], alpha=g.alpha.iloc[0], fused=bool(g.fused.iloc[0]),
                        clean_auc=g.groupby('train_filter').clean_auc.first().mean(),
                        seen4=g[~g.held_out].worse_cross_auc.mean(), unseen12=g[g.held_out].worse_cross_auc.mean(),
                        all16=g.worse_cross_auc.mean(), cells_over_pair=int((cell(m) > pair).sum()),
                        cells_over_proposed=int((cell(m) > proposed).sum()),
                        selected_epochs='/'.join(str(int(e)) for e in info.selected_epoch), epoch0_runs=int(info.selected_epoch.eq(0).sum()),
                        train_ce=info.train_ce.mean(), train_rank=info.train_rank.mean(),
                        weighted_rank_over_ce=info.weighted_rank_over_ce.mean()))
    h = pd.DataFrame(out)
    h['_order'] = h.fused.astype(int)*100+h.alpha.fillna(-1).astype(float)
    return h.sort_values('_order').drop(columns='_order').reset_index(drop=True)


def analyze(args):
    reference = pair_nsf.read_prediction(prediction_path(args, None, FILTERS[0], 'normal'))
    order, y = reference.crop_path.tolist(), reference.class_index.astype(int).to_numpy()
    b, w = b_scores(args, order), nsf.SETTINGS['weight']
    rows, train = [], []
    for alpha in [None, REFERENCE_ALPHA, *args.alphas]:
        r, tr = method_rows(args, alpha, order, y, b, w)
        rows += r
        train += tr
    reproduce(args, rows)
    head = headline(rows, train)
    args.output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_root/'rank_weight_rows.csv', index=False)
    pd.DataFrame(train).to_csv(args.output_root/'rank_weight_training.csv', index=False)
    head.to_csv(args.output_root/'rank_weight_headline.csv', index=False, encoding='utf-8-sig')
    cols = ['name', 'clean_auc', 'seen4', 'unseen12', 'all16', 'cells_over_pair', 'cells_over_proposed', 'selected_epochs',
            'weighted_rank_over_ce']
    print(f'{args.model} | seed {SEED} | worst cross AUC at strong smoothing (seen 4 / unseen 12 / all 16)', flush=True)
    print(head[cols].round(4).to_string(index=False), flush=True)
    print('Results:', args.output_root, flush=True)
    return head


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=['collect', 'analyze'])
    p.add_argument('--model', required=True, choices=['xception', 'clip_b16', 'dinov2_b14'])
    p.add_argument('--alphas', nargs='+', type=float, default=[1., 2., 4.])
    for key in ['runs-base', 'original-runs', 'original-eval', 'pair-nsf-root', 'smoothed-b-root', 'output-root']:
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--baseline-runs-root', type=Path)
    p.add_argument('--source-root', type=Path, default=Path('/content/ffpp_c23_native_roi16_v1'))
    p.add_argument('--cdf-root', type=Path, default=Path('/content/celebdf_v2_face_crops16_v1'))
    p.add_argument('--archive-dir', type=Path)
    p.add_argument('--cache-root', type=Path, default=Path('/content/beauty_smoothing_v2_cache'))
    p.add_argument('--work-dir', type=Path, default=Path('/content/rank_weight_work'))
    p.add_argument('--protocol', type=Path, default=ebs.ROOT/'configs/ffpp_c23_celebdf_smoothing_v1.json')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    a = p.parse_args(argv)
    if REFERENCE_ALPHA in a.alphas or len(set(a.alphas)) != len(a.alphas) or min(a.alphas) <= 0:
        p.error('--alphas: distinct positive weights other than the thesis .25')
    for root in [a.original_runs, a.original_eval, a.pair_nsf_root, a.smoothed_b_root]:
        if a.output_root.resolve() == root.resolve() or a.runs_base.resolve() == root.resolve():
            p.error('Use new folders; thesis runs/evaluations must remain unchanged')
    if a.action == 'collect' and (a.archive_dir is None or a.baseline_runs_root is None):
        p.error('collect needs --archive-dir and --baseline-runs-root')
    return a


if __name__ == '__main__':
    args = parse_args()
    {'collect': collect, 'analyze': analyze}[args.action](args)
