"""Pair + NSF ablation: reuse frozen pair predictions and the original NSF scorer.

Only the whole-face source changes from rank to pair. Weight=.2, scorer, validation
sample, preprocessing, best.pt selection and CDF test membership stay fixed.
prepare extracts only the original FF++ validation sample (not smoothing caches).
evaluate infers that sample once per checkpoint, then fuses saved CDF predictions.
Existing run/evaluation files are read-only. New artifacts live under output-root.
"""
import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import nonskin_fusion_v2 as nsf
import summarize_v2_diagonal as summary

FILTERS = list(nsf.diagonal.v2.FILTERS)
CONDITIONS = ['normal', 'smooth_weak', 'smooth_medium', 'smooth_strong']
LABEL = 'aug_pair_nsf'
PROTOCOL = 'pair_nsf_v2_w02_v1'
MASTER = 'ffpp_c23_4methods_roi16_trainval_v1.csv'
PREFIX = 'ffpp_c23_native_roi16_v1'
SLUGS = {'original': 'real', 'Deepfakes': 'deepfakes', 'Face2Face': 'face2face',
         'FaceSwap': 'faceswap', 'NeuralTextures': 'neuraltextures'}


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    return nsf.base.common.digest(path)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    nsf.base.common.json_write(path, value)


def read_prediction(path):
    frame = pd.read_csv(path, dtype={'crop_path': str, 'video_id': str})
    required = {'crop_path', 'video_id', 'class_index', 'raw_margin', 'checkpoint_sha256'}
    if not required.issubset(frame) or frame.crop_path.duplicated().any():
        raise ValueError(f'Invalid saved predictions: {path}')
    if not np.isfinite(frame.raw_margin.to_numpy()).all() or frame.checkpoint_sha256.nunique() != 1:
        raise ValueError(f'Nonfinite scores or mixed checkpoints: {path}')
    return frame


def same_membership(a, b, path):
    cols = ['crop_path', 'video_id', 'class_index']
    if not a[cols].equals(b[cols]):
        raise ValueError(f'Image order/labels differ: {path}')


def scorer_paths(args):
    folder = args.runs_root/'nonskin_probe'
    return (folder/f'{args.model}_face_inner_seed42.json',
            folder/f'{args.model}_face_inner_seed42.pkl',
            folder/f'{args.model}_cdf_test_face_inner_scores.csv')


def preflight(args):
    info_path, pkl, scores_path = scorer_paths(args)
    for path in [info_path, pkl, scores_path]:
        if not path.is_file():
            raise FileNotFoundError(f'Original rank+NSF artifact required (do not refit): {path}')
    info = read_json(info_path)
    if info['settings'] != nsf.SETTINGS or digest(pkl) != info['pickle_sha256']:
        raise ValueError('Original NSF scorer settings/hash differ')
    scores = pd.read_csv(scores_path, dtype={'crop_path': str})
    if scores.crop_path.duplicated().any() or not np.isfinite(scores.score).all():
        raise ValueError('Invalid cached eye/mouth scores')
    if not scores.scorer_pickle_sha256.eq(info['pickle_sha256']).all():
        raise ValueError('Cached eye/mouth scores use another scorer')
    reference = None
    for seed in args.seeds:
        source = args.evaluation_root/f'seed{seed}'/'cdf_test'
        for filt in args.train_filters:
            run = args.runs_root/nsf.diagonal.run_name(args.model, filt, 'aug_pair', seed)
            for part in ['best.pt', 'completed.json', 'config.json']:
                if not (run/part).is_file():
                    raise FileNotFoundError(run/part)
            path = source/f'{args.model}_aug_pair@{filt}_normal.csv'
            normal = read_prediction(path)
            if reference is None:
                reference = normal
                if len(normal) != 8288 or normal.class_index.value_counts().to_dict() != {1: 5440, 0: 2848}:
                    raise ValueError('Expected original CDF official test: 8288 frames, real2848/fake5440')
                if normal.crop_path.tolist() != scores.crop_path.tolist():
                    raise ValueError('Pair predictions and original NSF scores have different membership')
            same_membership(reference, normal, path)
            # Early existence check only; exact rows/hash are checked when each file is used.
            for ev in FILTERS:
                for c in CONDITIONS[1:]:
                    path = source/f'{args.model}_aug_pair@{filt}_{ev}_{c}.csv'
                    if not path.is_file():
                        raise FileNotFoundError(f'Finish existing pair evaluation first: {path}')
        old = pd.read_csv(source/'summary.csv')
        for method in ['aug_pair', 'aug_pair_rank', nsf.LABEL]:
            part = old[old.method.eq(method) & old.train_filter.isin(args.train_filters)]
            if part.duplicated(['train_filter', 'filter']).any() or len(part) != 4*len(args.train_filters):
                raise ValueError(f'Incomplete reference {method} seed{seed}: {source}')
    print(f'Ready: {args.model}, seeds={args.seeds}, {len(args.train_filters)} training filters; original NSF artifacts reused', flush=True)
    return info, scores


def val_frame(args):
    frame = pd.read_csv(args.source_root/'manifests'/MASTER, dtype=str, keep_default_na=False)
    selected = nsf.occlusion.balanced(frame[frame.split.eq('val')], nsf.SETTINGS['val_images'], 42)
    if len(selected) != 1200:
        raise ValueError('Expected the original balanced FF++ validation sample:1200')
    expected = read_json(scorer_paths(args)[0])['val_paths_sha256']
    if hashlib.sha256('\n'.join(selected.crop_path).encode()).hexdigest() != expected:
        raise ValueError('FF++ validation sample differs from original NSF; frozen manifests must be preserved')
    return selected


def archive_members(path, wanted, root):
    """Read uncompressed TAR directly, write only explicit requested files; no extractall/markers."""
    with tarfile.open(path, 'r:') as tar:
        for rel in tqdm(wanted, desc=path.name, mininterval=10):
            if Path(rel).is_absolute() or '..' in Path(rel).parts:
                raise ValueError(f'Unsafe path: {rel}')
            member = tar.getmember(PREFIX+'/'+rel)
            if not member.isfile():
                raise ValueError('Expected a regular TAR member')
            dest = root/rel
            if dest.is_file() and dest.stat().st_size == member.size:
                continue
            if dest.exists():
                raise ValueError(f'Existing file size differs; preserved: {dest}')
            dest.parent.mkdir(parents=True, exist_ok=True)
            temp = dest.with_suffix(dest.suffix+'.partial')
            with tar.extractfile(member) as src, temp.open('wb') as out:
                shutil.copyfileobj(src, out)
            if temp.stat().st_size != member.size:
                raise RuntimeError(f'Incomplete extraction: {dest}')
            temp.replace(dest)


def prepare(args):
    preflight(args)
    master = args.source_root/'manifests'/MASTER
    method_tar = args.drive_data/'ffpp_c23_real_roi16_trainval_v1.tar'
    combined = args.drive_data/'ffpp_c23_4methods_roi16_trainval_v1.tar'
    if not master.is_file():
        archive_members(method_tar if method_tar.is_file() else combined, ['manifests/'+MASTER], args.source_root)
    selected = val_frame(args)
    for method, group in selected.groupby('method', sort=False):
        needed = [p for p in group.crop_path if not (args.source_root/p).is_file()]
        if needed:
            path = args.drive_data/f'ffpp_c23_{SLUGS[method]}_roi16_trainval_v1.tar'
            archive_members(path if path.is_file() else combined, needed, args.source_root)
    if any(not (args.source_root/p).is_file() for p in selected.crop_path):
        raise RuntimeError('FF++ validation extraction incomplete')
    print('Prepared only1200 FF++ original validation images; no CDF/smoothing TAR needed', flush=True)


def calibration(args, seed, filt, frame, info):
    run = args.runs_root/nsf.diagonal.run_name(args.model, filt, 'aug_pair', seed)
    cfg, completed = read_json(run/'config.json'), read_json(run/'completed.json')
    if (cfg.get('strategy') != 'aug_pair' or cfg.get('model') != args.model
            or cfg.get('train_filter') != filt or int(cfg.get('seed', -1)) != seed):
        raise ValueError('Wrong source model/filter/seed/strategy in pair config')
    if cfg['initial_checkpoint_sha256'] != info['baseline_sha256']:
        raise ValueError('Pair checkpoint and NSF scorer originate from different baselines')
    pair_sha = digest(run/'best.pt')
    identity = dict(protocol=PROTOCOL, model=args.model, seed=seed, train_filter=filt,
                    source_method='aug_pair', source_checkpoint_sha256=pair_sha,
                    source_identity=completed['identity'], settings=nsf.SETTINGS,
                    scorer_pickle_sha256=info['pickle_sha256'], val_paths_sha256=info['val_paths_sha256'])
    dest = args.output_root/'calibration'/f'seed{seed}_{filt}.json'
    if dest.is_file():
        saved = read_json(dest)
        if saved['identity'] != identity:
            raise ValueError(f'Calibration provenance changed; use a new output folder: {dest}')
        print('Reuse validation statistics:', seed, filt, flush=True)
        return saved
    weights, verified_sha, epoch = nsf.diagonal.completed_checkpoint(args.runs_root, args.model, filt, 'aug_pair', seed, info['baseline_sha256'])
    if verified_sha != pair_sha:
        raise ValueError('Pair checkpoint changed while reading')
    model, transform = nsf.load(args, args.model, weights)
    dataset = nsf.fusion.Inputs(frame, args.source_root, None, 'full', transform, args.work_dir)
    _, raw = nsf.diagonal.engine.infer(args, model, frame, dataset,
        f'{args.model} seed{seed} {filt}: FF++ val 1200')
    raw = np.asarray(raw, dtype=np.float32)  # match original NSF validation-statistic precision
    del model, weights
    if args.device == 'cuda':
        torch.cuda.empty_cache()
    sd = float(raw.std())
    if not np.isfinite(raw).all() or sd <= 0:
        raise ValueError('Invalid pair validation statistics')
    # Original fused() keys are retained to use the exact existing NSF arithmetic.
    saved = dict(identity=identity, selected_epoch=epoch, rank_val_mean=float(raw.mean()), rank_val_std=sd,
                 scorer_val_mean=info['val_mean'], scorer_val_std=info['val_std'], weight=.2)
    write_json(dest, saved)
    return saved


def fuse_one(args, seed, filt, done, scores):
    source = args.evaluation_root/f'seed{seed}'/'cdf_test'
    target = args.output_root/f'seed{seed}'/'cdf_test'
    target.mkdir(parents=True, exist_ok=True)
    normal = read_prediction(source/f'{args.model}_aug_pair@{filt}_normal.csv')
    source_sha = done['identity']['source_checkpoint_sha256']
    if not normal.checkpoint_sha256.eq(source_sha).all():
        raise ValueError('Pair predictions do not belong to selected best.pt')
    identity = hashlib.sha256(json.dumps(done, sort_keys=True).encode()).hexdigest()
    rows = []
    for ev in FILTERS:
        margins = {}
        for condition in CONDITIONS:
            stem = 'normal' if condition == 'normal' else ev+'_'+condition
            pred = source/f'{args.model}_aug_pair@{filt}_{stem}.csv'
            raw = normal if condition == 'normal' else read_prediction(pred)
            same_membership(normal, raw, pred)
            if not raw.checkpoint_sha256.eq(source_sha).all():
                raise ValueError(f'Stale prediction checkpoint: {pred}')
            fused = nsf.fused(raw.raw_margin.to_numpy(), scores.score.to_numpy(), done)
            out = raw[['crop_path', 'video_id', 'class_index']].copy()
            out['raw_margin'], out['checkpoint_sha256'] = fused, identity
            dest = target/f'{args.model}_{LABEL}@{filt}_{stem}.csv'
            if condition != 'normal' or ev == FILTERS[0]:
                temp = dest.with_suffix('.tmp.csv'); out.to_csv(temp, index=False); temp.replace(dest)
            margins[condition] = fused
        measured = nsf.fusion.shift.measure(normal.class_index.to_numpy(), margins)
        rows.append(dict(model=args.model, strategy=LABEL+'@'+filt, method=LABEL, train_filter=filt,
                         filter=ev, held_out=ev != filt, seed=seed, **measured,
                         max_gap=max(abs(measured['fp_gap']), abs(measured['fn_gap']))))
    return rows


def evaluate(args):
    info, scores = preflight(args)
    frame = val_frame(args)
    for seed in args.seeds:
        rows = []
        for i, filt in enumerate(args.train_filters, 1):
            print(f'[{i}/{len(args.train_filters)}] {args.model} seed{seed} {filt}: validation statistics + saved-score fusion', flush=True)
            done = calibration(args, seed, filt, frame, info)
            rows += fuse_one(args, seed, filt, done, scores)
        target = args.output_root/f'seed{seed}'/'cdf_test'
        pd.DataFrame(rows).to_csv(target/'summary.csv', index=False)
    write_json(args.output_root/'protocol.json', dict(protocol=PROTOCOL, model=args.model, seeds=args.seeds,
        train_filters=args.train_filters, weight=.2, settings=nsf.SETTINGS, original_evaluation_root=str(args.evaluation_root),
        target_used_for_training_or_selection=False, note='Only source method changes rank->pair; original NSF scorer reused'))
    report(args)


def report(args):
    frames = []
    compare = ['baseline', 'aug_single', 'aug_pair', 'aug_pair_rank', 'aug_pair_rank_fc100_ramp2', nsf.LABEL]
    for seed in args.seeds:
        old = pd.read_csv(args.evaluation_root/f'seed{seed}'/'cdf_test'/'summary.csv')
        old = old[old.method.isin(compare) & (old.method.eq('baseline') | old.train_filter.isin(args.train_filters))].copy()
        new = pd.read_csv(args.output_root/f'seed{seed}'/'cdf_test'/'summary.csv')
        for t in [old, new]:
            t['seed'] = seed
            t['worse_cross_auc'] = np.minimum(t.fp_cross_auc, t.fn_cross_auc)
            frames.append(t)
    all_rows = pd.concat(frames, ignore_index=True)
    means = summary.seed_means(all_rows)
    head = summary.headline(all_rows.assign(epoch0=False))
    head = head.drop(columns=['epoch0_runs'])
    all_rows.to_csv(args.output_root/'comparison_rows.csv', index=False)
    means.to_csv(args.output_root/'pair_nsf_seed_means.csv', index=False)
    head.to_csv(args.output_root/'pair_nsf_headline.csv', index=False)
    print(head[['model', 'method', 'seeds', 'clean_auc_mean', 'diag_worse_mean', 'heldout_worse_mean', 'all_worse_mean']].round(4).to_string(index=False), flush=True)
    export_ppt_tables(args, head, means)
    export_plot(args)
    print('Results:', args.output_root, flush=True)
    return head, means


def export_ppt_tables(args, head, means):
    """Copy/paste values and both ablation references; no automatic winner/cherry-pick."""
    folder = args.output_root/'ppt_tables'
    folder.mkdir(parents=True, exist_ok=True)
    names = {'baseline': '추가 학습 없음', 'aug_single': '무작위 증강',
             'aug_pair': '원본·보정 동시 학습', 'aug_pair_rank': '교차 순위 학습',
             'aug_pair_rank_fc100_ramp2': '특징 일관성 학습',
             'aug_pair_rank_nsf': '교차 순위 학습 + NSF', LABEL: '원본·보정 동시 학습 + NSF'}
    order = list(names)
    h = head.set_index('method').reindex([m for m in order if m in set(head.method)])
    same, general = [], []
    for me, row in h.iterrows():
        # Baseline has no training-filter axis; its four-filter reference is shown only in the same table.
        cross = row.all_worse_mean if me == 'baseline' else row.diag_worse_mean
        same.append({'학습 방법': names[me], '원본 AUC / 학습한 방식 최저 교차 AUC':
                     f"{row.clean_auc_mean:.3f} / {cross:.3f}"})
        if me != 'baseline':
            general.append({'학습 방법': names[me], '학습한 방식 / 처음 보는 방식 최저 교차 AUC':
                            f"{row.diag_worse_mean:.3f} / {row.heldout_worse_mean:.3f}"})
    pd.DataFrame(same).to_csv(folder/'slide25_same_filter.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(general).to_csv(folder/'slide26_seen_unseen.csv', index=False, encoding='utf-8-sig')
    matrices = {}
    for me in ['aug_pair', 'aug_pair_rank', 'aug_pair_rank_nsf', LABEL]:
        a = means[means.method.eq(me)].pivot(index='train_filter', columns='filter', values='worse_cross_auc_mean').reindex(index=FILTERS, columns=FILTERS)
        if a.isna().any().any():
            raise ValueError('Incomplete matrix: '+me)
        matrices[me] = a
        a.to_csv(folder/(me+'_4x4.csv'), encoding='utf-8-sig')
    for name, left, right in [('pair_nsf_minus_pair', LABEL, 'aug_pair'),
                              ('rank_nsf_minus_rank', 'aug_pair_rank_nsf', 'aug_pair_rank'),
                              ('pair_nsf_minus_rank_nsf', LABEL, 'aug_pair_rank_nsf')]:
        (matrices[left]-matrices[right]).to_csv(folder/(name+'_4x4.csv'), encoding='utf-8-sig')
    rows = []
    for label, base, method in [('pair + NSF vs pair', 'aug_pair', LABEL),
                                ('rank + NSF vs rank', 'aug_pair_rank', 'aug_pair_rank_nsf'),
                                ('pair + NSF vs rank + NSF', 'aug_pair_rank_nsf', LABEL)]:
        delta = matrices[method]-matrices[base]
        rows.append(dict(comparison=label, cells_improved=int((delta.to_numpy()>0).sum()), cells_total=16,
                         clean_auc_gain=h.loc[method,'clean_auc_mean']-h.loc[base,'clean_auc_mean'],
                         seen_cross_gain=h.loc[method,'diag_worse_mean']-h.loc[base,'diag_worse_mean'],
                         unseen_cross_gain=h.loc[method,'heldout_worse_mean']-h.loc[base,'heldout_worse_mean'],
                         all_cross_gain=h.loc[method,'all_worse_mean']-h.loc[base,'all_worse_mean']))
    pd.DataFrame(rows).to_csv(folder/'fusion_comparisons.csv', index=False)
    write_json(folder/'definitions.json', dict(model=args.model, seeds=args.seeds, filters=FILTERS,
        rows='training filter', columns='evaluation filter', metric='min(FP cross AUC,FN cross AUC), strong alpha=.75',
        aggregation='mean per-cell metric across seeds; seen4/unseen12/all16, equal weights',
        baseline_note='baseline row uses four evaluation filters, not three independently trained baselines',
        comparison_note='PPT25/26: compare all methods. 4x4 ablation: pair+NSF vs pair OR rank+NSF vs rank; do not mix references.',
        direction_note='Original AUC, seen/unseen AUC and paired differences reported separately; no test-based weight/epoch tuning'))
    print(pd.DataFrame(rows).round(4).to_string(index=False), flush=True)
    # The last completed backbone can collect all three PPT columns, within its own output folder.
    completed = {}
    for model in ['xception','clip_b16','dinov2_b14']:
        parent = args.output_root.parent/f'ffpp_c23_pair_nsf_v2_{model}'/'ppt_tables'
        if all((parent/f).is_file() for f in ['slide25_same_filter.csv','slide26_seen_unseen.csv']):
            completed[model] = parent
    if len(completed) == 3:
        for source, target in [('slide25_same_filter.csv','slide25_three_backbones.csv'),
                               ('slide26_seen_unseen.csv','slide26_three_backbones.csv')]:
            joined = None
            for model, parent in completed.items():
                table = pd.read_csv(parent/source).set_index('학습 방법')
                table.columns = [model]
                joined = table if joined is None else joined.join(table,how='outer')
            joined.to_csv(folder/target,encoding='utf-8-sig')
        print('All three backbone PPT tables collected.',flush=True)


def export_plot(args):
    # Run in the fresh script process; notebook kernels may have an older Pillow loaded.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    files = [('aug_pair_4x4.csv','Pair'),(LABEL+'_4x4.csv','Pair + NSF'),
             ('pair_nsf_minus_pair_4x4.csv','Pair + NSF minus Pair'),
             ('pair_nsf_minus_rank_nsf_4x4.csv','Pair + NSF minus Rank + NSF')]
    fig, axes = plt.subplots(1,4,figsize=(17,4))
    short = ['Bil.','WMF','FGS','DT']
    for i,(file,title) in enumerate(files):
        a = pd.read_csv(args.output_root/'ppt_tables'/file,index_col=0).to_numpy()
        bound = max(.02,float(np.abs(a).max()))
        im = axes[i].imshow(a,vmin=0 if i<2 else -bound,vmax=1 if i<2 else bound,cmap='Blues' if i<2 else 'RdBu')
        for r in range(4):
            for c in range(4):
                axes[i].text(c,r,f'{a[r,c]:+.3f}' if i>=2 else f'{a[r,c]:.3f}',ha='center',va='center',fontsize=8,
                    color='white' if (i<2 and a[r,c]>.65) or (i>=2 and abs(a[r,c])>.65*bound) else 'black',
                    fontweight='bold' if r==c else 'normal')
        axes[i].set(xticks=range(4),yticks=range(4),xticklabels=short,yticklabels=short,title=title,xlabel='Evaluation filter')
        if i==0:axes[i].set_ylabel('Training filter')
        fig.colorbar(im,ax=axes[i],shrink=.7)
    fig.suptitle(args.model+' | min cross AUC | seed mean',y=1.02)
    fig.tight_layout()
    fig.savefig(args.output_root/'pair_nsf_comparison.png',dpi=180,bbox_inches='tight')
    plt.close(fig)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'evaluate', 'report'])
    p.add_argument('--model', required=True, choices=['xception', 'clip_b16', 'dinov2_b14'])
    for key in ['runs-root', 'evaluation-root', 'output-root']:
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--source-root', type=Path, default=Path('/content/pair_nsf_val_only'))
    p.add_argument('--drive-data', type=Path)
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    p.add_argument('--train-filters', nargs='+', choices=FILTERS, default=FILTERS)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--work-dir', type=Path, default=Path('/content/pair_nsf_work'))
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    a = p.parse_args(argv)
    if a.output_root.resolve() == a.evaluation_root.resolve() or a.output_root.resolve() == a.runs_root.resolve():
        p.error('Use a separate output folder; original results must remain unchanged')
    if a.action == 'prepare' and a.drive_data is None:
        p.error('--drive-data required for prepare')
    return a


if __name__ == '__main__':
    args = parse_args()
    {'prepare': prepare, 'evaluate': evaluate, 'report': report}[args.action](args)
