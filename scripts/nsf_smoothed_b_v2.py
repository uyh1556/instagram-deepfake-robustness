"""pair+NSF with score B taken from the smoothed image: does the saved result hold when B also sees the smoothing?

The saved pair+NSF scores B once per CDF frame from the CLEAN image (face_inner mask parsed on the clean crop, clean
pixels) and reuses it for every smoothed condition, so B is unchanged by construction. A deployed detector only has
the smoothed image. Here only the input of B changes; the scorer, w=.2, z statistics and the saved pair predictions of
pair+NSF are reused unchanged. Variants of B (face_inner, same pinned SegFormer parser, rest filled grey):
  saved       clean pixels, clean-image mask    (the original pair+NSF; read from the saved NSF scores)
  smooth_pix  smoothed pixels, clean-image mask  -> effect of the smoothed pixels alone (feathered mask edges)
  smooth_all  smoothed pixels, mask parsed on the smoothed image -> deployment setting (pixels + parser)
The clean view of both new variants is the recomputed clean B, which must reproduce the saved scores. Views are the
clean image and the 4 filters at strong strength (alpha .75), the main metric. B depends only on the backbone, so it
is scored once (cached in output-root/scores) and evaluate covers every seed. Ablation only: nothing is chosen from
these results. No training; original runs/evaluations are read-only.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import nsf_region_controls_v2 as rc

nsf, fusion, occlusion, diagonal, base, evaluation, v2 = rc.nsf, rc.fusion, rc.occlusion, rc.diagonal, rc.base, rc.evaluation, rc.v2
FILTERS, VIEWS, STRONG = rc.FILTERS, rc.VIEWS, rc.STRONG
VARIANTS = ['saved', 'smooth_pix', 'smooth_all']
PROTOCOL = 'nsf_smoothed_b_v2_v1'
SETTINGS = dict(nsf.SETTINGS, variants=VARIANTS, strength=STRONG)
NAMES = {'none': 'pair (B 없음)', 'saved': 'pair+NSF (기존: B=원본 이미지)',
         'smooth_pix': 'B=보정 픽셀 + 원본 마스크', 'smooth_all': 'B=보정 이미지 (픽셀+파싱 모두)'}


def mask_path(work_dir, view, crop_path):
    return work_dir/'inner'/view/(crop_path+'.png')


def as_list(batch):
    return batch


class ViewImages(Dataset):
    """One view (clean or a strong-smoothed filter) of each frame as a uint8 array, for face parsing."""
    def __init__(self, frame, root, cache, view):
        self.rows, self.root, self.cache, self.view = frame.to_dict('records'), root, cache, view

    def __len__(self):
        return len(self.rows)

    def image(self, index):
        row = self.rows[index]
        with Image.open(self.root/row['crop_path']) as opened:
            image = opened.convert('RGB')
        if self.view != 'normal':
            image = self.cache.view(image, row['crop_path'], self.view, STRONG, row.get('roi_status', 'detected') == 'detected')
        return image

    def __getitem__(self, index):
        return np.asarray(self.image(index)), index


class InnerInputs(ViewImages):
    """face_inner input(s) of one view: [clean-image mask] for the clean view, [clean-image mask, own mask] otherwise."""
    def __init__(self, frame, root, cache, view, work_dir, transform):
        super().__init__(frame, root, cache, view)
        self.work_dir, self.transform = work_dir, transform

    def __getitem__(self, index):
        image, path = self.image(index), self.rows[index]['crop_path']
        views = ['normal'] if self.view == 'normal' else ['normal', self.view]
        inputs = []
        for v in views:
            with Image.open(mask_path(self.work_dir, v, path)) as opened:
                keep = v2.image_to_mask(opened)
            if keep.shape != image.size[::-1]:
                raise ValueError(f'Mask size differs: {path}')
            inputs.append(self.transform(occlusion.occlude(image, 1.-keep)))
        return torch.stack(inputs), index


def agreement(clean, own):
    """IoU and area ratio of two feathered keep masks (thresholded at .5); nan when the clean mask is empty."""
    a, b = clean > .5, own > .5
    union, area = int((a | b).sum()), int(a.sum())
    return dict(iou=(a & b).sum()/union if union else np.nan, area_ratio=b.sum()/area if area else np.nan)


def parse_view(args, frame, cache, view):
    """face_inner masks of one view under work-dir (once per Colab session); smoothed views also log mask agreement."""
    done = args.work_dir/'inner'/view/'complete.json'
    stats = args.output_root/'scores'/f'mask_agreement_{view}.csv'
    if done.is_file() and json.loads(done.read_text()).get('frames') == len(frame) and (view == 'normal' or stats.is_file()):
        return
    parser = fusion.InnerParser(args.device)
    loader = DataLoader(ViewImages(frame, args.cdf_root, cache, view), batch_size=args.parse_batch,
                        num_workers=args.workers, collate_fn=as_list)
    rows = []
    for batch in tqdm(loader, desc=f'face parsing {view}', unit='batch', mininterval=10):
        images = [Image.fromarray(a) for a, _ in batch]
        for (_, i), mask in zip(batch, parser.masks(images)):
            path = frame.crop_path.iloc[i]
            out = mask_path(args.work_dir, view, path)
            out.parent.mkdir(parents=True, exist_ok=True)
            v2.mask_to_image(mask).save(out)
            if view != 'normal':
                with Image.open(mask_path(args.work_dir, 'normal', path)) as opened:
                    rows.append(dict(crop_path=path, roi_status=frame.roi_status.iloc[i],
                                     **agreement(v2.image_to_mask(opened), v2.image_to_mask(v2.mask_to_image(mask)))))
    del parser
    if args.device == 'cuda':
        torch.cuda.empty_cache()
    if view != 'normal':
        temp = stats.with_suffix('.tmp.csv'); pd.DataFrame(rows).to_csv(temp, index=False); temp.replace(stats)
    done.write_text(json.dumps(dict(frames=len(frame))))


def inner_features(args, model, name, dataset, k, desc):
    """[n, k, d] classifier-input features; one forward pass carries the k inputs of each frame."""
    captured = {}
    hook = base.head(model, name).register_forward_pre_hook(lambda m, inp: captured.__setitem__('x', inp[0].detach()))
    feats = []
    try:
        with torch.inference_mode():
            for x, _ in tqdm(DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers),
                             desc=desc, unit='batch', mininterval=10):
                b = x.shape[0]
                model(x.flatten(0, 1).to(args.device))
                feats.append(captured['x'].float().flatten(1).cpu().numpy().reshape(b, k, -1))
    finally:
        hook.remove()
    return np.concatenate(feats)


def scorer(args):
    """The original NSF scorer of this backbone (hash-checked; never refitted here)."""
    sub = argparse.Namespace(runs_root=args.runs_root, baseline_runs_root=args.baseline_runs_root, baseline_seed=42, source_root=None)
    pipe, info = nsf.scorer(sub, args.model)
    if info['settings'] != nsf.SETTINGS:
        raise ValueError('Original NSF scorer settings differ')
    return pipe, info


def saved_scores(args, cdf, info):
    inner = pd.read_csv(args.runs_root/'nonskin_probe'/f'{args.model}_cdf_test_face_inner_scores.csv', dtype={'crop_path': str})
    if inner.crop_path.tolist() != cdf.crop_path.tolist() or not inner.scorer_pickle_sha256.eq(info['pickle_sha256']).all():
        raise ValueError('Cached face_inner scores do not match the CDF test frames/scorer')
    return inner.score.to_numpy()


def check_clean(recomputed, saved):
    """The recomputed clean B must be the saved B (same parser, masks, encoder and scorer; GPU noise only)."""
    corr, gap = float(np.corrcoef(recomputed, saved)[0, 1]), float(np.abs(recomputed-saved).max())
    print(f'clean B recomputed vs saved: corr {corr:.6f}, max |diff| {gap:.4f} (saved std {saved.std():.3f})', flush=True)
    if corr < .9999:
        raise ValueError('Recomputed clean B does not reproduce the saved NSF scores; check parser/encoder/scorer versions')
    return dict(corr=corr, max_abs_diff=gap)


def scores(args):
    """Parse every view, then score B of each view with the original scorer; cached under output-root/scores."""
    out = args.output_root/'scores'
    out.mkdir(parents=True, exist_ok=True)
    pipe, info = scorer(args)
    initial, initial_hash = evaluation.checkpoint(argparse.Namespace(runs_root=args.baseline_runs_root, seed=42), args.model)
    if initial_hash != info['baseline_sha256']:
        raise ValueError('Baseline checkpoint differs from the one behind the NSF scorer')
    identity = dict(protocol=PROTOCOL, model=args.model, settings=SETTINGS, baseline_sha256=initial_hash,
                    scorer_pickle_sha256=info['pickle_sha256'])
    info_path = out/'identity.json'
    if info_path.is_file() and json.loads(info_path.read_text())['identity'] != identity:
        raise ValueError(f'Cached scores were made with other settings; use a new output folder: {out}')
    cdf = rc.cdf_frame(args)
    if info_path.is_file() and all((out/f'cdf_test_{v}.csv').is_file() for v in VIEWS):
        print('All B scores cached:', out, flush=True)
        return
    rc.pair_nsf.write_json(info_path, dict(identity=identity))
    cache = diagonal.restore_cache(args.cache_root, args.archive_dir, 'cdf_test', v2.SmoothingCache.PARTS)
    model, transform = nsf.load(args, args.model, initial['model_state_dict'])
    saved = saved_scores(args, cdf, info)
    for i, view in enumerate(VIEWS, 1):
        dest = out/f'cdf_test_{view}.csv'
        if dest.is_file():
            print(f'[{i}/{len(VIEWS)}] cached: {view}', flush=True)
            continue
        parse_view(args, cdf, cache, 'normal')
        parse_view(args, cdf, cache, view)
        k = 1 if view == 'normal' else 2
        f = inner_features(args, model, args.model, InnerInputs(cdf, args.cdf_root, cache, view, args.work_dir, transform),
                           k, f'[{i}/{len(VIEWS)}] CDF test {view}')
        table = cdf[['crop_path']].copy()
        if view == 'normal':
            table['clean'] = pipe.decision_function(f[:, 0])
            check = check_clean(table.clean.to_numpy(), saved)
            rc.pair_nsf.write_json(out/'clean_check.json', check)
        else:
            table['smooth_pix'], table['smooth_all'] = pipe.decision_function(f[:, 0]), pipe.decision_function(f[:, 1])
        temp = dest.with_suffix('.tmp.csv'); table.to_csv(temp, index=False); temp.replace(dest)
    del model
    if args.device == 'cuda':
        torch.cuda.empty_cache()


def view_scores(args, cdf):
    """{view: {variant: z-scored B}}; saved is the same in every view, the new variants use the recomputed clean B."""
    info = json.loads((args.runs_root/'nonskin_probe'/f'{args.model}_face_inner_seed42.json').read_text())
    if info['settings'] != nsf.SETTINGS:
        raise ValueError('Original NSF scorer settings differ')
    z = lambda s: (s-info['val_mean'])/info['val_std']
    saved = z(saved_scores(args, cdf, info))
    tables = {}
    for view in VIEWS:
        t = pd.read_csv(args.output_root/'scores'/f'cdf_test_{view}.csv', dtype={'crop_path': str})
        if t.crop_path.tolist() != cdf.crop_path.tolist():
            raise ValueError(f'B score order differs: {view}')
        tables[view] = t
    clean = z(tables['normal'].clean.to_numpy())
    out = {'normal': dict(saved=saved, smooth_pix=clean, smooth_all=clean)}
    for view in FILTERS:
        out[view] = dict(saved=saved, smooth_pix=z(tables[view].smooth_pix.to_numpy()), smooth_all=z(tables[view].smooth_all.to_numpy()))
    return out


def evaluate(args):
    """Fuse the saved pair predictions of every seed with each B variant; saved must reproduce pair+NSF exactly."""
    cdf = rc.cdf_frame(args, status=False)
    y = cdf.class_index.astype(int).to_numpy()
    b = view_scores(args, cdf)
    w = nsf.SETTINGS['weight']
    for seed in args.seeds:
        source = args.evaluation_root/f'seed{seed}'/'cdf_test'
        reference = pd.read_csv(args.pair_nsf_root/f'seed{seed}'/'cdf_test'/'summary.csv')
        rows = []
        for filt in FILTERS:
            cal = json.loads((args.pair_nsf_root/'calibration'/f'seed{seed}_{filt}.json').read_text())
            sha = cal['identity']['source_checkpoint_sha256']
            a = {}
            for view in VIEWS:
                pred = rc.pair_nsf.read_prediction(source/f'{args.model}_aug_pair@{filt}_{"normal" if view == "normal" else view+"_smooth_strong"}.csv')
                if pred.crop_path.tolist() != cdf.crop_path.tolist() or not pred.checkpoint_sha256.eq(sha).all():
                    raise ValueError(f'Saved pair predictions do not match the pair+NSF calibration: seed{seed} {filt} {view}')
                a[view] = (pred.raw_margin.to_numpy()-cal['rank_val_mean'])/cal['rank_val_std']
            for ev in FILTERS:
                common = dict(model=args.model, seed=seed, train_filter=filt, filter=ev, held_out=ev != filt)
                rows.append(dict(common, variant='none', **rc.metrics(y, a['normal'], a[ev])))
                for v in VARIANTS:
                    rows.append(dict(common, variant=v, **rc.metrics(y, (1-w)*a['normal']+w*b['normal'][v], (1-w)*a[ev]+w*b[ev][v])))
                ref = reference[reference.train_filter.eq(filt) & reference['filter'].eq(ev)].iloc[0]
                mine = rows[-len(VARIANTS)]
                if abs(mine['fp_cross_auc']-ref.fp_cross_auc) > 1e-6 or abs(mine['fn_cross_auc']-ref.fn_cross_auc) > 1e-6:
                    raise ValueError(f'saved B does not reproduce the saved pair+NSF result: seed{seed} {filt}->{ev}')
        target = args.output_root/f'seed{seed}'
        target.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(target/'summary.csv', index=False)
        print(f'seed{seed}: saved B reproduces the saved pair+NSF numbers (16/16 cells)', flush=True)
    b_compare(args, cdf, y, b)
    report(args)


def b_compare(args, cdf, y, b):
    """B on its own per variant and filter: how far the smoothing moves B, and how well the parser agrees."""
    rows = []
    for ev in FILTERS:
        status = None
        stats = args.output_root/'scores'/f'mask_agreement_{ev}.csv'
        if stats.is_file():
            m = pd.read_csv(stats, dtype={'crop_path': str})
            status = m[m.roi_status.eq('detected')]
        for v in VARIANTS:
            delta = b[ev][v]-b['normal'][v]
            row = dict(variant=v, filter=ev, mean_abs_delta_z=float(np.abs(delta).mean()),
                       corr_with_clean=float(np.corrcoef(b[ev][v], b['normal'][v])[0, 1]) if v != 'saved' else 1.,
                       **rc.metrics(y, b['normal'][v], b[ev][v]))
            if v == 'smooth_all' and status is not None:
                row.update(mask_iou_mean=float(status.iou.mean()), mask_iou_p05=float(status.iou.quantile(.05)),
                           mask_area_ratio_mean=float(status.area_ratio.mean()))
            rows.append(row)
    pd.DataFrame(rows).to_csv(args.output_root/'b_compare.csv', index=False)


def report(args):
    """Seed means over every finished seed: clean / seen 4 / unseen 12 / all 16 worse cross AUC per B variant."""
    files = sorted(args.output_root.glob('seed*/summary.csv'))
    rows = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    seeds = sorted(int(s) for s in rows.seed.unique())
    cell = rows.groupby(['variant', 'train_filter', 'filter', 'held_out'], as_index=False)[['clean_auc', 'worse_cross_auc', 'real_shift', 'fake_shift']].mean()
    pivot = cell.pivot_table(index=['train_filter', 'filter'], columns='variant', values='worse_cross_auc')
    head = []
    for v in ['none', *VARIANTS]:
        g = cell[cell.variant.eq(v)]
        head.append(dict(variant=v, name=NAMES[v], seeds=len(seeds), clean_auc=g.clean_auc.mean(),
                         seen_worse=g[~g.held_out].worse_cross_auc.mean(), unseen_worse=g[g.held_out].worse_cross_auc.mean(),
                         all16_worse=g.worse_cross_auc.mean(),
                         cells_over_pair=int((pivot[v] > pivot['none']).sum()) if v != 'none' else None,
                         delta_all16_vs_saved=g.worse_cross_auc.mean()-cell[cell.variant.eq('saved')].worse_cross_auc.mean()))
    head = pd.DataFrame(head)
    head.to_csv(args.output_root/'smoothed_b_headline.csv', index=False, encoding='utf-8-sig')
    cell.to_csv(args.output_root/'smoothed_b_seed_means.csv', index=False)
    print(f'{args.model} | seeds {seeds} | worse cross AUC at strong smoothing', flush=True)
    print(head.round(4).to_string(index=False), flush=True)
    compare = args.output_root/'b_compare.csv'
    if compare.is_file():
        t = pd.read_csv(compare)
        cols = [c for c in ['mean_abs_delta_z', 'corr_with_clean', 'worse_cross_auc', 'real_shift', 'fake_shift', 'mask_iou_mean', 'mask_iou_p05'] if c in t]
        print('B alone per filter (z units; mask IoU on detected frames):', flush=True)
        print(t.set_index(['variant', 'filter'])[cols].round(4).to_string(), flush=True)
    print('Results:', args.output_root, flush=True)
    return head


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=['scores', 'evaluate', 'report'])
    p.add_argument('--model', required=True, choices=['xception', 'clip_b16', 'dinov2_b14'])
    for key in ['runs-root', 'evaluation-root', 'pair-nsf-root', 'output-root', 'baseline-runs-root']:
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--cdf-root', type=Path, default=Path('/content/celebdf_v2_face_crops16_v1'))
    p.add_argument('--archive-dir', type=Path)
    p.add_argument('--cache-root', type=Path, default=Path('/content/beauty_smoothing_v2_cache'))
    p.add_argument('--work-dir', type=Path, default=Path('/content/nsf_smoothed_b_work'))
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    for key, default in [('batch-size', 32), ('workers', 2), ('parse-batch', 16)]:
        p.add_argument('--'+key, type=int, default=default)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    a = p.parse_args(argv)
    for root in [a.evaluation_root, a.pair_nsf_root, a.runs_root]:
        if a.output_root.resolve() == root.resolve():
            p.error('Use a separate output folder; original results must remain unchanged')
    if a.action == 'scores' and a.archive_dir is None:
        p.error('--archive-dir required for scores')
    return a


if __name__ == '__main__':
    args = parse_args()
    {'scores': scores, 'evaluate': evaluate, 'report': report}[args.action](args)
