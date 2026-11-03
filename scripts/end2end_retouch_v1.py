"""End-to-end learned retoucher on the CDF test: does pair+NSF still help when no skin mask is involved?

The advisor (2026-11-03) asked whether the results hold for modern retouching and whether a mask-based training
setup is realistic when real tools output a retouched image directly. Here a public end-to-end face retoucher is
applied to the official CDF test frames and the FINISHED detectors are only evaluated on its output:
  retoucher  GFPGAN v1.4 (TencentARC, Apache-2.0; arch 'clean', upscale 1, centre face, pasted back on the crop).
             It regenerates the whole face (skin, eyes, mouth) with a StyleGAN2 prior; no mask is given to it.
             Used as a face enhancer in consumer apps and toolkits, and by Realism to Deception (BMVC-W 2025).
  build      retouch every detected CDF test frame once (per-frame torch seed: GFPGAN injects noise) and keep it as
             one Drive tar <archive-dir>/cdf_test_gfpgan.tar; fallback frames stay unchanged like the v2 protocol.
             The tar is the reference (never recomputed); it also stores frames.csv (faces restored per frame).
  evaluate   per backbone, with the frozen checkpoints and settings:
             baseline   FF++ c23 baseline seed 42 (the reference model of every diagonal evaluation)
             pair       aug_pair@<4 train filters> x seeds 42/43/44
             pair+NSF   the same pair margins fused with score B (w=.2, original scorer and z statistics), B scored on
                        saved      clean image (B untouched by the retoucher; optimistic)
                        smooth_pix retouched pixels, eyes/mouth mask parsed on the clean image
                        smooth_all retouched pixels, mask parsed on the retouched image (deployment setting)
             The clean predictions are read from the saved evaluations (hash-checked; their clean AUC must
             reproduce the saved summaries); only the retouched view is inferred. Retouched predictions, B scores
             and parser masks are cached for resuming.
Metrics (as for the v2 filters): clean AUC, FP cross AUC (retouched real vs clean fake), FN cross AUC (clean real vs
retouched fake), worse cross AUC (main), retouched AUC (both classes retouched), score shifts in clean-real std, flips at
the clean 5% FPR threshold. Plus how much the retoucher changes skin vs eyes/mouth (mean |RGB difference|).
Evaluation only: nothing is trained, tuned or selected on these results; original runs/evaluations are read-only.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

import build_beauty_smoothing_v2_cache as builder
import evaluate_beauty_smoothing_v2 as ev
import nsf_smoothed_b_v2 as sb

rc, nsf, v2, pair_nsf = sb.rc, sb.nsf, sb.v2, sb.rc.pair_nsf
FILTERS = sb.FILTERS
PROTOCOL = 'end2end_retouch_v1'
VIEW = 'gfpgan'
RETOUCHER = dict(name='GFPGAN', version='1.4', arch='clean', channel_multiplier=2, upscale=1, only_center_face=True,
                 paste_back=True, bg_upsampler=None, noise_seed='sha256(crop_path) per frame',
                 weights_url='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth')
METHODS = ['baseline', 'pair', *(f'pair_nsf_{v}' for v in sb.VARIANTS)]
NAMES = {'baseline': 'Baseline (추가 학습 없음)', 'pair': 'pair (원본·보정 동시 학습)',
         'pair_nsf_saved': 'pair+NSF, B=원본 이미지 (낙관적)', 'pair_nsf_smooth_pix': 'pair+NSF, B=보정 픽셀 + 원본 마스크',
         'pair_nsf_smooth_all': 'pair+NSF, B=보정 이미지 (실제 사용)'}
NORMAL_FPR = .05


class RetouchCache:
    """Reader of the retouched frames: <root>/<dataset>/<part>/<crop_path>; same view() call as the v2 SmoothingCache."""
    def __init__(self, root, dataset='cdf_test', part=VIEW):
        self.root, self.part = Path(root)/dataset, part

    def path(self, relative):
        relative = Path(relative)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Cache requires a relative crop path')
        return self.root/self.part/relative

    def check(self):
        folder = self.root/self.part
        if not (folder/'complete.json').is_file():
            raise FileNotFoundError(f'Retouched cache incomplete: {folder}')
        ident = json.loads((folder/'identity.json').read_text())
        if ident.get('protocol') != PROTOCOL or ident.get('part') != self.part:
            raise ValueError(f'Not a {PROTOCOL} {self.part} cache: {folder}')
        return ident

    def view(self, image, relative, name=VIEW, strength=1., detected=True):
        if name != self.part:
            raise ValueError(f'This cache holds {self.part}, not {name}')
        if not detected:
            return image
        with Image.open(self.path(relative)) as opened:
            out = opened.convert('RGB')
        if out.size != image.size:
            raise ValueError(f'Retouched size differs: {relative}')
        return out


# ---------------------------------------------------------------- build (once, backbone independent)

def frame_seed(relative):
    return int.from_bytes(hashlib.sha256(relative.encode()).digest()[:4], 'little')


def load_retoucher(args):
    """GFPGANer and the sha256 of its weights. basicsr 1.4.2 imports a torchvision module removed in torchvision 0.17;
    its one used function exists in torchvision.transforms.functional, so that module is aliased (no behaviour change)."""
    import torchvision.transforms.functional as functional
    sys.modules.setdefault('torchvision.transforms.functional_tensor', functional)
    from gfpgan import GFPGANer
    weights = args.weights_dir/Path(RETOUCHER['weights_url']).name
    if not weights.is_file():
        weights.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(RETOUCHER['weights_url'], str(weights))
    model = GFPGANer(model_path=str(weights), upscale=RETOUCHER['upscale'], arch=RETOUCHER['arch'],
                     channel_multiplier=RETOUCHER['channel_multiplier'], bg_upsampler=None, device=torch.device(args.device))
    return model, builder.digest(weights)


def retouch(model, array, relative):
    """RGB uint8 crop -> (retouched RGB uint8 of the same size, number of faces GFPGAN restored)."""
    torch.manual_seed(frame_seed(relative))
    _, faces, out = model.enhance(np.ascontiguousarray(array[..., ::-1]), has_aligned=False,
                                  only_center_face=RETOUCHER['only_center_face'], paste_back=RETOUCHER['paste_back'])
    if out is None or out.shape != array.shape:
        raise ValueError(f'GFPGAN output size differs: {relative}')
    return np.ascontiguousarray(out[..., ::-1]), len(faces)


def versions():
    out = {}
    for name in ['gfpgan', 'basicsr', 'facexlib', 'torch', 'torchvision']:
        try:
            from importlib.metadata import version
            out[name] = version(name)
        except Exception:
            out[name] = None
    return out


def preview(cache, rows, data_root, dest, count=8):
    """Clean (top) vs retouched (bottom) for a few evenly spaced frames, for a visual check."""
    picks = rows[::max(1, len(rows)//count)][:count]
    tiles = []
    for p in picks:
        with Image.open(data_root/p) as opened:
            clean = opened.convert('RGB').resize((192, 192))
        with Image.open(cache.path(p)) as opened:
            done = opened.convert('RGB').resize((192, 192))
        tiles.append((clean, done))
    sheet = Image.new('RGB', (192*len(tiles), 384), 'white')
    for i, (a, b) in enumerate(tiles):
        sheet.paste(a, (192*i, 0)); sheet.paste(b, (192*i, 192))
    dest.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dest, quality=92)


def build(args):
    sub = argparse.Namespace(dataset='cdf_test', data_root=args.cdf_root, archive_dir=args.archive_dir)
    rows = builder.frames(sub).crop_path.tolist()
    cache = RetouchCache(args.cache_root)
    folder = cache.root/VIEW
    archive = args.archive_dir/f'cdf_test_{VIEW}.tar'
    if not (folder/'complete.json').is_file() and archive.is_file():
        print('Restoring', archive, flush=True)
        cache.root.mkdir(parents=True, exist_ok=True)
        builder.subprocess.run(['tar', '-xf', str(archive), '-C', str(cache.root)], check=True)
    if not (folder/'complete.json').is_file():
        model, weights_sha = load_retoucher(args)
        ident = dict(protocol=PROTOCOL, dataset='cdf_test', part=VIEW, retoucher=RETOUCHER, weights_sha256=weights_sha,
                     packages=versions(), manifest_sha256=builder.digest(builder.manifest(sub)),
                     code_sha256=builder.digest(Path(__file__)))
        folder.mkdir(parents=True, exist_ok=True)
        saved = folder/'identity.json'
        if saved.is_file() and json.loads(saved.read_text())['weights_sha256'] != weights_sha:
            raise ValueError(f'Local cache was made with other weights: {folder}; delete it or use a new --cache-root')
        saved.write_text(json.dumps(ident, sort_keys=True, indent=1)+'\n')
        log = folder/'faces_partial.csv'
        faces = pd.read_csv(log, dtype={'crop_path': str}).set_index('crop_path').faces.to_dict() if log.is_file() else {}
        todo = [p for p in rows if not cache.path(p).is_file() or p not in faces]
        print(f'{VIEW}: {len(rows)-len(todo)}/{len(rows)} ready; retouching {len(todo)} on {args.device}', flush=True)
        for k, p in enumerate(tqdm(todo, desc='GFPGAN cdf_test', unit='frame', mininterval=15), 1):
            with Image.open(args.cdf_root/p) as opened:
                array = np.asarray(opened.convert('RGB'))
            out, faces[p] = retouch(model, array, p)
            v2.save_png(cache.path(p), Image.fromarray(out))
            if k % 500 == 0 or k == len(todo):
                pd.DataFrame(dict(crop_path=list(faces), faces=list(faces.values()))).to_csv(log, index=False)
        missing = [p for p in rows if not cache.path(p).is_file()]
        if missing:
            raise RuntimeError(f'{len(missing)} frames missing after the pass')
        table = pd.DataFrame(dict(crop_path=rows, faces=[faces[p] for p in rows]))
        table.to_csv(folder/'frames.csv', index=False)
        log.unlink(missing_ok=True)
        (folder/'complete.json').write_text(json.dumps(dict(frames=len(rows), no_face=int(table.faces.eq(0).sum())))+'\n')
    builder.pack(sub, argparse.Namespace(root=cache.root), VIEW)
    ident = cache.check()
    table = pd.read_csv(folder/'frames.csv', dtype={'crop_path': str})
    if table.crop_path.tolist() != rows:
        raise ValueError('Retouched cache frames differ from the CDF test manifest')
    preview(cache, rows, args.cdf_root, args.archive_dir/f'preview_{VIEW}.jpg')
    print(f'{VIEW} ready: {len(rows)} detected frames, GFPGAN found no face in {int(table.faces.eq(0).sum())} '
          f'(left unchanged) | weights {ident["weights_sha256"][:12]} | preview {args.archive_dir/f"preview_{VIEW}.jpg"}', flush=True)


def restore(args):
    cache = RetouchCache(args.cache_root)
    if not (cache.root/VIEW/'complete.json').is_file():
        archive = args.archive_dir/f'cdf_test_{VIEW}.tar'
        if not archive.is_file():
            raise FileNotFoundError(f'Build the retouched cache first (build_end2end_retouch_v1 notebook): {archive}')
        print('Restoring', archive, flush=True)
        cache.root.mkdir(parents=True, exist_ok=True)
        builder.subprocess.run(['tar', '-xf', str(archive), '-C', str(cache.root)], check=True)
    return cache, cache.check()


# ---------------------------------------------------------------- evaluate (per backbone)

def metrics(y, normal, retouched):
    """Main metrics of one score pair (clean view, retouched view)."""
    real, fake = y == 0, y == 1
    row = rc.metrics(y, normal, retouched)
    row['retouched_auc'] = rc.shift.auc(retouched[real], retouched[fake])
    t = np.quantile(normal[real], 1-NORMAL_FPR)
    row['fp_flip'] = float(((normal[real] < t) & (retouched[real] >= t)).mean())
    row['fn_flip'] = float(((normal[fake] >= t) & (retouched[fake] < t)).mean())
    return row


def saved_clean(path, cdf, sha):
    pred = pair_nsf.read_prediction(path)
    if pred.crop_path.tolist() != cdf.crop_path.tolist() or not pred.checkpoint_sha256.eq(sha).all():
        raise ValueError(f'Saved clean predictions do not match the CDF frames/checkpoint: {path}')
    return pred.raw_margin.to_numpy()


def check_reference(summary, strategy, clean_auc):
    t = pd.read_csv(summary)
    ref = t[t.strategy.eq(strategy)].normal_auc
    if ref.empty or abs(float(ref.iloc[0])-clean_auc) > 1e-6:
        raise ValueError(f'Clean AUC does not reproduce the saved evaluation: {strategy} {summary}')


def predictions(args, cdf, cache):
    """{(label, seed): (clean margin, retouched margin)}: clean read from the saved evaluations, retouched inferred once."""
    y = cdf.class_index.astype(int).to_numpy()
    initial, initial_hash = rc.evaluation.checkpoint(argparse.Namespace(runs_root=args.baseline_runs_root, seed=42), args.model)
    jobs = [('baseline', 42, None)] + [(f'aug_pair@{f}', s, f) for s in args.seeds for f in FILTERS]
    out = {}
    for i, (label, seed, filt) in enumerate(jobs, 1):
        source = args.evaluation_root/f'seed{seed}'/'cdf_test'
        if filt is None:
            weights, sha = initial['model_state_dict'], initial_hash
        else:
            weights, sha, _ = rc.diagonal.completed_checkpoint(args.runs_root, args.model, filt, 'aug_pair', seed, initial_hash)
        clean = saved_clean(source/f'{args.model}_{label}_normal.csv', cdf, sha)
        check_reference(source/'summary.csv', label, rc.shift.auc(clean[y == 0], clean[y == 1]))
        sub = argparse.Namespace(output_root=args.output_root/'predictions'/f'seed{seed}', data_root=args.cdf_root,
                                 eval_batch_size=args.batch_size, batch_size=args.batch_size, workers=args.workers, device=args.device)
        sub.output_root.mkdir(parents=True, exist_ok=True)
        print(f'[{i}/{len(jobs)}] {args.model} {label} seed{seed}: {VIEW} view', flush=True)
        state = {}
        retouched = ev.predict(sub, cdf, args.model, label, weights, sha, cache, VIEW, 'retouched', 1., state)
        state.clear()
        out[(label, seed)] = (clean, np.asarray(retouched))
        if args.device == 'cuda':
            torch.cuda.empty_cache()
    return out


def b_scores(args, cdf, cache):
    """z-scored B of the clean and retouched views per variant (smooth_* use masks parsed by the pinned parser)."""
    out = args.output_root/'scores'
    out.mkdir(parents=True, exist_ok=True)
    pipe, info = sb.scorer(args)
    dest = out/f'cdf_test_{VIEW}.csv'
    if not dest.is_file():
        initial, initial_hash = rc.evaluation.checkpoint(argparse.Namespace(runs_root=args.baseline_runs_root, seed=42), args.model)
        if initial_hash != info['baseline_sha256']:
            raise ValueError('Baseline checkpoint differs from the one behind the NSF scorer')
        sb.parse_view(args, cdf, cache, 'normal')
        sb.parse_view(args, cdf, cache, VIEW)
        model, transform = nsf.load(args, args.model, initial['model_state_dict'])
        f = sb.inner_features(args, model, args.model, sb.InnerInputs(cdf, args.cdf_root, cache, VIEW, args.work_dir, transform),
                              2, f'B on {VIEW} view')
        del model
        if args.device == 'cuda':
            torch.cuda.empty_cache()
        table = cdf[['crop_path']].copy()
        table['smooth_pix'], table['smooth_all'] = pipe.decision_function(f[:, 0]), pipe.decision_function(f[:, 1])
        temp = dest.with_suffix('.tmp.csv'); table.to_csv(temp, index=False); temp.replace(dest)
    t = pd.read_csv(dest, dtype={'crop_path': str})
    if t.crop_path.tolist() != cdf.crop_path.tolist():
        raise ValueError(f'B score order differs: {dest}')
    z = lambda s: (np.asarray(s)-info['val_mean'])/info['val_std']
    clean = z(sb.saved_scores(args, cdf, info))
    return {'normal': {v: clean for v in sb.VARIANTS},
            VIEW: dict(saved=clean, smooth_pix=z(t.smooth_pix), smooth_all=z(t.smooth_all))}


def region_change(args, cdf, cache):
    """Mean |RGB change| (0-255) of the retoucher inside the v2 skin mask, the clean eyes/mouth mask and elsewhere."""
    dest = args.output_root/'region_change.csv'
    if dest.is_file():
        return pd.read_csv(dest, dtype={'crop_path': str})
    skin = rc.diagonal.restore_cache(args.v2_cache_root, args.v2_archive_dir, 'cdf_test', ['mask'])
    rows = []
    for r in tqdm(cdf[cdf.roi_status.eq('detected')].to_dict('records'), desc='region change', unit='frame', mininterval=15):
        p = r['crop_path']
        with Image.open(args.cdf_root/p) as opened:
            clean = opened.convert('RGB')
        diff = np.abs(np.asarray(cache.view(clean, p), np.float32)-np.asarray(clean, np.float32)).mean(2)
        s = skin.mask(p, clean.size) > .5
        with Image.open(sb.mask_path(args.work_dir, 'normal', p)) as opened:
            inner = v2.image_to_mask(opened) > .5
        rest = ~(s | inner)
        rows.append(dict(crop_path=p, skin=float(diff[s].mean()) if s.any() else np.nan,
                         eyes_mouth=float(diff[inner].mean()) if inner.any() else np.nan,
                         rest=float(diff[rest].mean()) if rest.any() else np.nan))
    table = pd.DataFrame(rows)
    temp = dest.with_suffix('.tmp.csv'); table.to_csv(temp, index=False); temp.replace(dest)
    return table


def rows_for(y, preds, b, seeds, model, w=nsf.SETTINGS['weight'], calibrations=None):
    """Metric rows of every method; calibrations[(seed, filt)] holds the pair+NSF z statistics of that pair run."""
    rows = [dict(model=model, method='baseline', seed=42, train_filter='none', **metrics(y, *preds[('baseline', 42)]))]
    for seed in seeds:
        for filt in FILTERS:
            clean, retouched = preds[(f'aug_pair@{filt}', seed)]
            common = dict(model=model, seed=seed, train_filter=filt)
            rows.append(dict(common, method='pair', **metrics(y, clean, retouched)))
            cal = calibrations[(seed, filt)]
            a = [(m-cal['rank_val_mean'])/cal['rank_val_std'] for m in (clean, retouched)]
            for v in sb.VARIANTS:
                rows.append(dict(common, method=f'pair_nsf_{v}',
                                 **metrics(y, (1-w)*a[0]+w*b['normal'][v], (1-w)*a[1]+w*b[VIEW][v])))
    return pd.DataFrame(rows)


def evaluate(args):
    cache, ident = restore(args)
    cdf = rc.cdf_frame(args)
    y = cdf.class_index.astype(int).to_numpy()
    preds = predictions(args, cdf, cache)
    b = b_scores(args, cdf, cache)
    calibrations = {}
    for seed in args.seeds:
        reference = pd.read_csv(args.pair_nsf_root/f'seed{seed}'/'cdf_test'/'summary.csv')
        for filt in FILTERS:
            cal = json.loads((args.pair_nsf_root/'calibration'/f'seed{seed}_{filt}.json').read_text())
            if cal['identity']['source_checkpoint_sha256'] != pair_nsf.read_prediction(
                    args.evaluation_root/f'seed{seed}'/'cdf_test'/f'{args.model}_aug_pair@{filt}_normal.csv').checkpoint_sha256.iloc[0]:
                raise ValueError(f'pair+NSF calibration belongs to another pair checkpoint: seed{seed} {filt}')
            calibrations[(seed, filt)] = cal
            clean = preds[(f'aug_pair@{filt}', seed)][0]
            a = (clean-cal['rank_val_mean'])/cal['rank_val_std']
            fused = (1-cal['weight'])*a+cal['weight']*b['normal']['saved']
            ref = reference[reference.train_filter.eq(filt) & reference['filter'].eq(filt)].normal_auc.iloc[0]
            if abs(rc.shift.auc(fused[y == 0], fused[y == 1])-ref) > 1e-6:
                raise ValueError(f'Clean pair+NSF AUC does not reproduce the saved result: seed{seed} {filt}')
    print('Clean baseline / pair / pair+NSF AUCs reproduce the saved evaluations', flush=True)
    rows = rows_for(y, preds, b, args.seeds, args.model, calibrations=calibrations)
    rows.to_csv(args.output_root/'summary.csv', index=False)
    b_compare(args, y, b)
    region_change(args, cdf, cache)
    pair_nsf.write_json(args.output_root/'protocol.json', dict(protocol=PROTOCOL, model=args.model, seeds=args.seeds,
        retoucher_identity=ident, nsf_settings=nsf.SETTINGS, test_used_for_selection=False,
        note='Evaluation only; frozen checkpoints, scorer, w and z statistics'))
    report(args)


def b_compare(args, y, b):
    rows = []
    stats = args.output_root/'scores'/f'mask_agreement_{VIEW}.csv'
    m = pd.read_csv(stats, dtype={'crop_path': str})
    m = m[m.roi_status.eq('detected')]
    for v in sb.VARIANTS:
        delta = b[VIEW][v]-b['normal'][v]
        rows.append(dict(variant=v, mean_abs_delta_z=float(np.abs(delta).mean()), **metrics(y, b['normal'][v], b[VIEW][v]),
                         **(dict(mask_iou_mean=float(m.iou.mean()), mask_iou_p05=float(m.iou.quantile(.05))) if v == 'smooth_all' else {})))
    pd.DataFrame(rows).to_csv(args.output_root/'b_compare.csv', index=False)


def report(args):
    rows = pd.read_csv(args.output_root/'summary.csv')
    cols = ['clean_auc', 'fp_cross_auc', 'fn_cross_auc', 'worse_cross_auc', 'retouched_auc', 'real_shift', 'fake_shift', 'fp_flip', 'fn_flip']
    head = rows.groupby('method', sort=False)[cols].mean().reindex(METHODS)
    head.insert(0, 'runs', rows.groupby('method').size().reindex(METHODS))
    pair = rows[rows.method.eq('pair')].set_index(['seed', 'train_filter']).worse_cross_auc
    head['runs_over_pair'] = [None if m in ['baseline', 'pair'] else
                              int((rows[rows.method.eq(m)].set_index(['seed', 'train_filter']).worse_cross_auc > pair).sum()) for m in METHODS]
    head.insert(0, 'name', [NAMES[m] for m in METHODS])
    head.reset_index().to_csv(args.output_root/'headline.csv', index=False, encoding='utf-8-sig')
    by_filter = rows[rows.method.ne('baseline')].groupby(['method', 'train_filter'], sort=False)[['clean_auc', 'worse_cross_auc']].mean()
    by_filter.reset_index().to_csv(args.output_root/'by_train_filter.csv', index=False)
    print(f'{args.model} | {VIEW} (end-to-end) | baseline seed42, pair/pair+NSF mean of {len(pair)} runs (4 filters x seeds)', flush=True)
    with pd.option_context('display.width', 250, 'display.max_columns', 20):
        print(head.round(4).to_string(), flush=True)
    for name in ['b_compare.csv', 'region_change.csv']:
        path = args.output_root/name
        if path.is_file() and name == 'b_compare.csv':
            print('B alone (z units):', flush=True)
            print(pd.read_csv(path).set_index('variant').round(4).to_string(), flush=True)
        elif path.is_file():
            t = pd.read_csv(path)
            print(f'Mean |RGB change| by region (0-255, {len(t)} detected frames): skin {t.skin.mean():.2f} | '
                  f'eyes/mouth {t.eyes_mouth.mean():.2f} | rest {t.rest.mean():.2f}', flush=True)
    print('Results:', args.output_root, flush=True)
    return head


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=['build', 'evaluate', 'report'])
    p.add_argument('--model', choices=['xception', 'clip_b16', 'dinov2_b14'])
    for key in ['runs-root', 'evaluation-root', 'pair-nsf-root', 'output-root', 'baseline-runs-root', 'v2-archive-dir']:
        p.add_argument('--'+key, type=Path)
    p.add_argument('--archive-dir', type=Path, required=True, help='Drive folder of cdf_test_gfpgan.tar')
    p.add_argument('--cdf-root', type=Path, default=Path('/content/celebdf_v2_face_crops16_v1'))
    p.add_argument('--cache-root', type=Path, default=Path('/content/end2end_retouch_cache'))
    p.add_argument('--v2-cache-root', type=Path, default=Path('/content/beauty_smoothing_v2_cache'))
    p.add_argument('--weights-dir', type=Path, default=Path('/content/gfpgan_weights'))
    p.add_argument('--work-dir', type=Path, default=Path('/content/end2end_retouch_work'))
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    for key, default in [('batch-size', 64), ('workers', 2), ('parse-batch', 16)]:
        p.add_argument('--'+key, type=int, default=default)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    a = p.parse_args(argv)
    if a.action != 'build':
        need = ['model', 'runs_root', 'evaluation_root', 'pair_nsf_root', 'output_root', 'baseline_runs_root', 'v2_archive_dir']
        missing = [k for k in need if getattr(a, k) is None]
        if missing:
            p.error(f'{a.action} needs --'+', --'.join(k.replace('_', '-') for k in missing))
        for root in [a.evaluation_root, a.pair_nsf_root, a.runs_root]:
            if a.output_root.resolve() == root.resolve():
                p.error('Use a separate output folder; original results must remain unchanged')
    return a


if __name__ == '__main__':
    args = parse_args()
    {'build': build, 'evaluate': evaluate, 'report': report}[args.action](args)
