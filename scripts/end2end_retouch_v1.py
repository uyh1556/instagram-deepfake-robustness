"""End-to-end learned retouchers on the CDF test: does pair+NSF still help when no skin mask is given?

The advisor (2026-11-03) asked whether the results hold for modern retouching and whether a mask-based training
setup is realistic when real tools output a retouched image directly. Public end-to-end retouchers are applied to the
official CDF test frames and the FINISHED detectors are only evaluated on their output. Retouchers (one view each):
  abpn    ModelScope skin retouching (iic/cv_unet_skin_retouching_torch, Alibaba, Apache-2.0; an ABPN-style public
          pipeline, CVPR 2022 - not the paper's exact network): RetinaFace face box x1.5 -> UNet predicts a per-pixel
          blend map on the 512 face ROI -> soft-light blend at the ROI resolution. Official torch-pipeline default
          degree .7; whitening and local blemish inpainting OFF (whitening would mix a brightness/colour change into the
          texture edit). No skin mask is given to it; the network decides where to edit. Main view.
  gfpgan  GFPGAN v1.4 (TencentARC, Apache-2.0; arch 'clean', upscale 1, centre face, pasted back on the crop). A face
          restorer that regenerates the whole face (skin, eyes, mouth) with a StyleGAN2 prior; no strength knob
          (its `weight` argument is ignored by the clean arch). Stress test; used by Realism to Deception (BMVC-W 2025).
  sample    a few real/fake frames as clean | retouched | |difference| x4, before the full build (nothing saved to Drive)
  build     retouch every detected CDF test frame once and keep it as one Drive tar <archive-dir>/cdf_test_<view>.tar
            (+ frames.csv: faces edited per frame); fallback frames stay unchanged like the v2 protocol. The tar is the
            reference and is never recomputed.
  evaluate  per backbone, for every built view, with the frozen checkpoints and settings:
            baseline   FF++ c23 baseline seed 42 (the reference model of every diagonal evaluation)
            pair       aug_pair@<4 train filters> x seeds 42/43/44
            pair+NSF   the same pair margins fused with score B (w=.2, original scorer and z statistics), B scored on
                       saved      clean image (B untouched by the retoucher; optimistic)
                       smooth_pix retouched pixels, eyes/mouth mask parsed on the clean image
                       smooth_all retouched pixels, mask parsed on the retouched image (deployment setting)
            Clean predictions are read from the saved evaluations (hash-checked; their clean AUCs must reproduce the
            saved summaries); only the retouched views are inferred. Predictions, B scores and masks are cached.
Metrics (as for the v2 filters): clean AUC, FP cross AUC (retouched real vs clean fake), FN cross AUC (clean real vs
retouched fake), worse cross AUC (main), retouched AUC (both classes retouched), score shifts in clean-real std, flips at
the clean 5% FPR threshold. Plus how much each retoucher changes skin vs eyes/mouth (mean |RGB difference|).
Evaluation only: nothing is trained, tuned or selected on these results; original runs/evaluations are read-only.
"""
import argparse
import hashlib
import json
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
VIEWS = ['abpn', 'gfpgan']
GFPGAN = dict(name='GFPGAN', version='1.4', arch='clean', channel_multiplier=2, upscale=1, only_center_face=True,
              paste_back=True, bg_upsampler=None, noise_seed='sha256(crop_path) per frame',
              weights_url='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth')
BASICSR = dict(url='https://files.pythonhosted.org/packages/86/41/00a6b000f222f0fa4c6d9e1d6dcc9811a374cabb8abb9d408b77de39648c/basicsr-1.4.2.tar.gz',
               sha256='b89b595a87ef964cda9913b4d99380ddb6554c965577c0c10cb7b78e31301e87')
MODELSCOPE = 'https://www.modelscope.cn/api/v1/models/{repo}/repo?Revision={revision}&FilePath={name}'
ABPN = dict(name='ModelScope skin retouching (ABPN-style)', degree=.7, whitening=False, retouch_local=False,
            input_size=512, enlarge=1.5, smooth_border=True,
            generator=dict(repo='iic/cv_unet_skin_retouching_torch', revision='30f890aab89b787aae93c3f396dbf32714bd950e',
                           name='pytorch_model.pt', sha256='760d0c0b95c4e0a692165677fe2b82dbdf8fbbd4a3780ca966599c627b082993'),
            detector=[dict(repo='damo/cv_resnet50_face-detection_retinaface', revision='a3f0e4db2f39fdc1486fca660282f9a18a9753f9',
                           name='pytorch_model.pt', sha256='6d1de9c2944f2ccddca5f5e010ea5ae64a39845a86311af6fdf30841b0a5a16d'),
                      dict(repo='damo/cv_resnet50_face-detection_retinaface', revision='075dac0f8b30c60196855ad10d66e0f03402efa5',
                           name='configuration.json', sha256='d03ba2dd7a34450d7e1b088701e2e4118b31b7f5791697188df0a60952207ecc')])
METHODS = ['baseline', 'pair', *(f'pair_nsf_{v}' for v in sb.VARIANTS)]
NAMES = {'baseline': 'Baseline (추가 학습 없음)', 'pair': 'pair (원본·보정 동시 학습)',
         'pair_nsf_saved': 'pair+NSF, B=원본 이미지 (낙관적)', 'pair_nsf_smooth_pix': 'pair+NSF, B=보정 픽셀 + 원본 마스크',
         'pair_nsf_smooth_all': 'pair+NSF, B=보정 이미지 (실제 사용)'}
NORMAL_FPR = .05


class Retouched:
    """Reader of retouched frames: <root>/<dataset>/<view>/<crop_path>; same view() call as the v2 SmoothingCache."""
    def __init__(self, root, dataset='cdf_test'):
        self.root = Path(root)/dataset

    def path(self, view, relative):
        relative = Path(relative)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Cache requires a relative crop path')
        return self.root/view/relative

    def check(self, view):
        folder = self.root/view
        if not (folder/'complete.json').is_file():
            raise FileNotFoundError(f'Retouched cache incomplete: {folder}')
        ident = json.loads((folder/'identity.json').read_text())
        if ident.get('protocol') != PROTOCOL or ident.get('view') != view:
            raise ValueError(f'Not a {PROTOCOL} {view} cache: {folder}')
        return ident

    def view(self, image, relative, name, strength=1., detected=True):
        if not detected:
            return image
        with Image.open(self.path(name, relative)) as opened:
            out = opened.convert('RGB')
        if out.size != image.size:
            raise ValueError(f'Retouched size differs: {name} {relative}')
        return out


# ---------------------------------------------------------------- retouchers

def fetch(url, dest, sha256):
    """Download once and check the pinned sha256."""
    if not dest.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(url, str(dest))
    if builder.digest(dest) != sha256:
        raise ValueError(f'Downloaded file hash differs from the pinned one: {dest}')
    return dest


def frame_seed(relative):
    return int.from_bytes(hashlib.sha256(relative.encode()).digest()[:4], 'little')


def vendor_basicsr(folder):
    """basicsr 1.4.2 ships only as an sdist whose setup.py cannot run on Python 3.13 (it reads __version__ back through
    locals() after exec, which PEP 667 broke). The pure-Python package (with its generated version.py) is taken from the
    hash-checked PyPI sdist and put on sys.path instead; its optional CUDA ops are not used by GFPGAN."""
    import shutil
    import tarfile
    folder = Path(folder)
    if not (folder/'basicsr'/'version.py').is_file():
        archive = fetch(BASICSR['url'], folder/'basicsr-1.4.2.tar.gz', BASICSR['sha256'])
        prefix = 'basicsr-1.4.2/basicsr/'
        with tarfile.open(archive) as tar:
            members = [m for m in tar.getmembers() if m.name.startswith(prefix) and (m.isfile() or m.isdir())]
            tar.extractall(folder/'sdist', members=members, filter='data')
        shutil.move(str(folder/'sdist'/prefix), str(folder/'basicsr'))
        shutil.rmtree(folder/'sdist'); archive.unlink()
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))


def versions(names):
    from importlib.metadata import version
    out = {}
    for name in ['torch', 'torchvision', *names]:
        try:
            out[name] = version(name)
        except Exception:   # vendored basicsr has no install metadata
            out[name] = getattr(sys.modules.get(name), '__version__', None)
    return out


class Gfpgan:
    """GFPGANer; basicsr 1.4.2 imports a torchvision module removed in torchvision 0.17, whose one used function exists in
    torchvision.transforms.functional, so that module is aliased (no behaviour change)."""
    def __init__(self, args):
        vendor_basicsr(args.vendor_dir)
        import torchvision.transforms.functional as functional
        sys.modules.setdefault('torchvision.transforms.functional_tensor', functional)
        from gfpgan import GFPGANer
        weights = args.weights_dir/Path(GFPGAN['weights_url']).name
        if not weights.is_file():
            weights.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(GFPGAN['weights_url'], str(weights))
        self.model = GFPGANer(model_path=str(weights), upscale=GFPGAN['upscale'], arch=GFPGAN['arch'],
                              channel_multiplier=GFPGAN['channel_multiplier'], bg_upsampler=None, device=torch.device(args.device))
        self.identity = dict(settings=GFPGAN, weights_sha256=builder.digest(weights), packages=versions(['gfpgan', 'basicsr', 'facexlib']))

    def __call__(self, array, relative):
        """RGB uint8 crop -> (retouched RGB uint8 of the same size, faces restored)."""
        torch.manual_seed(frame_seed(relative))
        _, faces, out = self.model.enhance(np.ascontiguousarray(array[..., ::-1]), has_aligned=False,
                                           only_center_face=GFPGAN['only_center_face'], paste_back=GFPGAN['paste_back'])
        if out is None or out.shape != array.shape:
            raise ValueError(f'GFPGAN output size differs: {relative}')
        return np.ascontiguousarray(out[..., ::-1]), len(faces)


class Abpn:
    """The forward pass of the official ModelScope SkinRetouchingTorchPipeline (ms_wrapper.py at the pinned revision)
    with whitening and local inpainting off, built from the modelscope package's own modules instead of its pipeline
    framework. Channel conventions follow the pipeline: the face detector gets the image channel-flipped (its pipeline's
    LoadImage flips an ndarray), the generator gets RGB."""
    def __init__(self, args):
        from modelscope.models.cv.face_detection.retinaface.detection import RetinaFaceDetection
        from modelscope.models.cv.skin_retouching import utils
        from modelscope.models.cv.skin_retouching.unet_deploy import UNet
        self.utils, self.device = utils, torch.device(args.device)
        folder = args.weights_dir/'modelscope'
        g = ABPN['generator']
        gen_path = fetch(MODELSCOPE.format(**g), folder/'skin_retouching'/g['name'], g['sha256'])
        for d in ABPN['detector']:
            det_path = fetch(MODELSCOPE.format(**d), folder/'retinaface'/d['name'], d['sha256'])
        self.generator = UNet(3, 3).to(self.device)
        self.generator.load_state_dict(torch.load(gen_path, map_location='cpu', weights_only=True)['generator'])
        self.generator.eval()
        self.detector = RetinaFaceDetection(model_path=str(det_path.parent/'pytorch_model.pt'), device=args.device)
        self.diffuse = torch.from_numpy(utils.gen_diffuse_mask()).to(self.device).float().permute(2, 0, 1)[None]
        self.identity = dict(settings=ABPN, packages=versions(['modelscope']))

    @torch.no_grad()
    def __call__(self, array, relative):
        U = self.utils
        dets, _ = self.detector({'img': torch.from_numpy(np.ascontiguousarray(array[..., ::-1]).astype(np.float32))})
        boxes = [dict(bbox=np.asarray(d[:4]).astype(np.int32).tolist(), score=float(d[4])) for d in dets]
        out = torch.from_numpy(array.copy()).to(self.device)
        crops = U.get_crop_bbox(boxes)
        for bbox in crops:
            roi, _, (t, b, l, r) = U.get_roi_without_padding(array, bbox)
            roi = U.preprocess_roi(U.roi_to_tensor(np.ascontiguousarray(roi)).to(self.device))
            mg = self.generator(torch.nn.functional.interpolate(roi, (ABPN['input_size'], ABPN['input_size']), mode='bilinear'))
            mg = ((mg-.5)*ABPN['degree']+.5).clamp(0., 1.)
            mg = torch.nn.functional.interpolate(mg, roi.shape[2:], mode='bilinear')[0].permute(1, 2, 0)
            mg = U.smooth_border_mg(self.diffuse, mg)
            image = (roi[0].permute(1, 2, 0)+1.)/2
            out[t:b, l:r] = (((1-2*mg)*image*image+2*mg*image)*255.).byte()
        result = out.cpu().numpy()
        if result.shape != array.shape:
            raise ValueError(f'ABPN output size differs: {relative}')
        return result, len(crops)


def load_retoucher(args, view):
    return {'abpn': Abpn, 'gfpgan': Gfpgan}[view](args)


# ---------------------------------------------------------------- sample / build (once, backbone independent)

def detected_frames(args):
    sub = argparse.Namespace(dataset='cdf_test', data_root=args.cdf_root)
    return builder.frames(sub), builder.digest(builder.manifest(sub))


def sample(args):
    """A few random detected real/fake frames as clean | retouched | |difference| x4; nothing is cached."""
    frame, _ = detected_frames(args)
    picks = pd.concat([frame[frame.class_index.eq(c)].sample(args.count//2, random_state=args.sample_seed) for c in ['0', '1']])
    model = load_retoucher(args, args.view)
    side = 256
    sheet = Image.new('RGB', (3*side, side*len(picks)), 'white')
    for i, r in enumerate(picks.to_dict('records')):
        with Image.open(args.cdf_root/r['crop_path']) as opened:
            clean = np.asarray(opened.convert('RGB'))
        out, faces = model(clean, r['crop_path'])
        change = np.abs(out.astype(np.int16)-clean).mean(2)
        diff = np.clip(change*4, 0, 255).astype(np.uint8)
        for j, a in enumerate([clean, out, np.repeat(diff[..., None], 3, 2)]):
            sheet.paste(Image.fromarray(a).resize((side, side)), (j*side, i*side))
        print(f"{i+1}. {'real' if r['class_index'] == '0' else 'fake'} | faces edited {faces} | "
              f"mean |change| {change.mean():.2f} | {r['crop_path']}", flush=True)
    args.sample_out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.sample_out, quality=92)
    print('Saved:', args.sample_out, f'(columns: clean | {args.view} | |difference| x4)', flush=True)


def preview(cache, view, rows, data_root, dest, count=8):
    """Clean (top) vs retouched (bottom) for a few evenly spaced frames."""
    picks = rows[::max(1, len(rows)//count)][:count]
    sheet = Image.new('RGB', (192*len(picks), 384), 'white')
    for i, p in enumerate(picks):
        for j, path in enumerate([data_root/p, cache.path(view, p)]):
            with Image.open(path) as opened:
                sheet.paste(opened.convert('RGB').resize((192, 192)), (192*i, 192*j))
    dest.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dest, quality=92)


def unpack(cache, archive):
    print('Restoring', archive, flush=True)
    cache.root.mkdir(parents=True, exist_ok=True)
    builder.subprocess.run(['tar', '-xf', str(archive), '-C', str(cache.root)], check=True)


def build(args):
    view = args.view
    frame, manifest_sha = detected_frames(args)
    rows = frame.crop_path.tolist()
    cache = Retouched(args.cache_root)
    folder = cache.root/view
    archive = args.archive_dir/f'cdf_test_{view}.tar'
    if not (folder/'complete.json').is_file() and archive.is_file():
        unpack(cache, archive)
    if not (folder/'complete.json').is_file():
        model = load_retoucher(args, view)
        ident = dict(protocol=PROTOCOL, dataset='cdf_test', view=view, manifest_sha256=manifest_sha,
                     code_sha256=builder.digest(Path(__file__)), **model.identity)
        folder.mkdir(parents=True, exist_ok=True)
        saved = folder/'identity.json'
        if saved.is_file() and json.loads(saved.read_text()).get('settings') != ident['settings']:
            raise ValueError(f'Local cache was made with other settings: {folder}; delete it or use a new --cache-root')
        saved.write_text(json.dumps(ident, sort_keys=True, indent=1)+'\n')
        log = folder/'faces_partial.csv'
        faces = pd.read_csv(log, dtype={'crop_path': str}).set_index('crop_path').faces.to_dict() if log.is_file() else {}
        todo = [p for p in rows if not cache.path(view, p).is_file() or p not in faces]
        print(f'{view}: {len(rows)-len(todo)}/{len(rows)} ready; retouching {len(todo)} on {args.device}', flush=True)
        for k, p in enumerate(tqdm(todo, desc=f'{view} cdf_test', unit='frame', mininterval=15), 1):
            with Image.open(args.cdf_root/p) as opened:
                array = np.asarray(opened.convert('RGB'))
            out, faces[p] = model(array, p)
            v2.save_png(cache.path(view, p), Image.fromarray(out))
            if k % 500 == 0 or k == len(todo):
                pd.DataFrame(dict(crop_path=list(faces), faces=list(faces.values()))).to_csv(log, index=False)
        missing = [p for p in rows if not cache.path(view, p).is_file()]
        if missing:
            raise RuntimeError(f'{len(missing)} frames missing after the pass')
        table = pd.DataFrame(dict(crop_path=rows, faces=[faces[p] for p in rows]))
        table.to_csv(folder/'frames.csv', index=False)
        log.unlink(missing_ok=True)
        (folder/'complete.json').write_text(json.dumps(dict(frames=len(rows), no_face=int(table.faces.eq(0).sum())))+'\n')
    builder.pack(argparse.Namespace(dataset='cdf_test', archive_dir=args.archive_dir), argparse.Namespace(root=cache.root), view)
    cache.check(view)
    table = pd.read_csv(folder/'frames.csv', dtype={'crop_path': str})
    if table.crop_path.tolist() != rows:
        raise ValueError('Retouched cache frames differ from the CDF test manifest')
    preview(cache, view, rows, args.cdf_root, args.archive_dir/f'preview_{view}.jpg')
    print(f'{view} ready: {len(rows)} detected frames, no face found (left unchanged) in {int(table.faces.eq(0).sum())} '
          f'| preview {args.archive_dir/f"preview_{view}.jpg"}', flush=True)


def restore(args, views):
    cache = Retouched(args.cache_root)
    for view in views:
        if not (cache.root/view/'complete.json').is_file():
            archive = args.archive_dir/f'cdf_test_{view}.tar'
            if not archive.is_file():
                raise FileNotFoundError(f'Build the {view} cache first (build_end2end_retouch_v1 notebook): {archive}')
            unpack(cache, archive)
    return cache, {v: cache.check(v) for v in views}


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


def predictions(args, cdf, cache, views):
    """{(label, seed): (clean margin, {view: retouched margin})}: clean from the saved evaluations, retouched inferred."""
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
        state, retouched = {}, {}
        for view in views:
            print(f'[{i}/{len(jobs)}] {args.model} {label} seed{seed}: {view}', flush=True)
            retouched[view] = np.asarray(ev.predict(sub, cdf, args.model, label, weights, sha, cache, view, 'retouched', 1., state))
        state.clear()
        out[(label, seed)] = (clean, retouched)
        if args.device == 'cuda':
            torch.cuda.empty_cache()
    return out


def b_scores(args, cdf, cache, views):
    """{'normal' or view: {variant: z-scored B}}; smooth_* use masks parsed by the pinned parser on each view."""
    out = args.output_root/'scores'
    out.mkdir(parents=True, exist_ok=True)
    pipe, info = sb.scorer(args)
    z = lambda s: (np.asarray(s)-info['val_mean'])/info['val_std']
    clean = z(sb.saved_scores(args, cdf, info))
    result = {'normal': {v: clean for v in sb.VARIANTS}}
    model = None
    for view in views:
        dest = out/f'cdf_test_{view}.csv'
        if not dest.is_file():
            if model is None:
                initial, initial_hash = rc.evaluation.checkpoint(argparse.Namespace(runs_root=args.baseline_runs_root, seed=42), args.model)
                if initial_hash != info['baseline_sha256']:
                    raise ValueError('Baseline checkpoint differs from the one behind the NSF scorer')
                sb.parse_view(args, cdf, cache, 'normal')
                model, transform = nsf.load(args, args.model, initial['model_state_dict'])
            sb.parse_view(args, cdf, cache, view)
            f = sb.inner_features(args, model, args.model, sb.InnerInputs(cdf, args.cdf_root, cache, view, args.work_dir, transform),
                                  2, f'B on {view} view')
            table = cdf[['crop_path']].copy()
            table['smooth_pix'], table['smooth_all'] = pipe.decision_function(f[:, 0]), pipe.decision_function(f[:, 1])
            temp = dest.with_suffix('.tmp.csv'); table.to_csv(temp, index=False); temp.replace(dest)
        t = pd.read_csv(dest, dtype={'crop_path': str})
        if t.crop_path.tolist() != cdf.crop_path.tolist():
            raise ValueError(f'B score order differs: {dest}')
        result[view] = dict(saved=clean, smooth_pix=z(t.smooth_pix), smooth_all=z(t.smooth_all))
    del model
    if args.device == 'cuda':
        torch.cuda.empty_cache()
    return result


def region_change(args, cdf, cache, views):
    """Mean |RGB change| (0-255) of each retoucher inside the v2 skin mask, the clean eyes/mouth mask and elsewhere."""
    dest = args.output_root/'region_change.csv'
    if dest.is_file() and set(pd.read_csv(dest).view) >= set(views):
        return
    skin = rc.diagonal.restore_cache(args.v2_cache_root, args.v2_archive_dir, 'cdf_test', ['mask'])
    rows = []
    for r in tqdm(cdf[cdf.roi_status.eq('detected')].to_dict('records'), desc='region change', unit='frame', mininterval=15):
        p = r['crop_path']
        with Image.open(args.cdf_root/p) as opened:
            clean = opened.convert('RGB')
        s = skin.mask(p, clean.size) > .5
        with Image.open(sb.mask_path(args.work_dir, 'normal', p)) as opened:
            inner = v2.image_to_mask(opened) > .5
        rest = ~(s | inner)
        for view in views:
            diff = np.abs(np.asarray(cache.view(clean, p, view), np.float32)-np.asarray(clean, np.float32)).mean(2)
            rows.append(dict(view=view, crop_path=p, skin=float(diff[s].mean()) if s.any() else np.nan,
                             eyes_mouth=float(diff[inner].mean()) if inner.any() else np.nan,
                             rest=float(diff[rest].mean()) if rest.any() else np.nan))
    table = pd.DataFrame(rows)
    temp = dest.with_suffix('.tmp.csv'); table.to_csv(temp, index=False); temp.replace(dest)


def rows_for(y, preds, b, seeds, model, views, w=nsf.SETTINGS['weight'], calibrations=None):
    """Metric rows of every view and method; calibrations[(seed, filt)] holds the pair+NSF z statistics of that run."""
    rows = []
    for view in views:
        clean, retouched = preds[('baseline', 42)]
        rows.append(dict(model=model, view=view, method='baseline', seed=42, train_filter='none', **metrics(y, clean, retouched[view])))
        for seed in seeds:
            for filt in FILTERS:
                clean, retouched = preds[(f'aug_pair@{filt}', seed)]
                common = dict(model=model, view=view, seed=seed, train_filter=filt)
                rows.append(dict(common, method='pair', **metrics(y, clean, retouched[view])))
                cal = calibrations[(seed, filt)]
                a = [(m-cal['rank_val_mean'])/cal['rank_val_std'] for m in (clean, retouched[view])]
                for v in sb.VARIANTS:
                    rows.append(dict(common, method=f'pair_nsf_{v}',
                                     **metrics(y, (1-w)*a[0]+w*b['normal'][v], (1-w)*a[1]+w*b[view][v])))
    return pd.DataFrame(rows)


def evaluate(args):
    views = [v for v in args.views if (args.archive_dir/f'cdf_test_{v}.tar').is_file() or
             (Retouched(args.cache_root).root/v/'complete.json').is_file()]
    if not views:
        raise FileNotFoundError(f'No retouched cache found for {args.views} in {args.archive_dir}')
    print('Views:', views, '| not built yet:', [v for v in args.views if v not in views], flush=True)
    cache, idents = restore(args, views)
    cdf = rc.cdf_frame(args)
    y = cdf.class_index.astype(int).to_numpy()
    preds = predictions(args, cdf, cache, views)
    b = b_scores(args, cdf, cache, views)
    calibrations = {}
    for seed in args.seeds:
        reference = pd.read_csv(args.pair_nsf_root/f'seed{seed}'/'cdf_test'/'summary.csv')
        for filt in FILTERS:
            cal = json.loads((args.pair_nsf_root/'calibration'/f'seed{seed}_{filt}.json').read_text())
            if cal['identity']['source_checkpoint_sha256'] != pair_nsf.read_prediction(
                    args.evaluation_root/f'seed{seed}'/'cdf_test'/f'{args.model}_aug_pair@{filt}_normal.csv').checkpoint_sha256.iloc[0]:
                raise ValueError(f'pair+NSF calibration belongs to another pair checkpoint: seed{seed} {filt}')
            calibrations[(seed, filt)] = cal
            a = (preds[(f'aug_pair@{filt}', seed)][0]-cal['rank_val_mean'])/cal['rank_val_std']
            fused = (1-cal['weight'])*a+cal['weight']*b['normal']['saved']
            ref = reference[reference.train_filter.eq(filt) & reference['filter'].eq(filt)].normal_auc.iloc[0]
            if abs(rc.shift.auc(fused[y == 0], fused[y == 1])-ref) > 1e-6:
                raise ValueError(f'Clean pair+NSF AUC does not reproduce the saved result: seed{seed} {filt}')
    print('Clean baseline / pair / pair+NSF AUCs reproduce the saved evaluations', flush=True)
    rows = rows_for(y, preds, b, args.seeds, args.model, views, calibrations=calibrations)
    rows.to_csv(args.output_root/'summary.csv', index=False)
    b_compare(args, y, b, views)
    region_change(args, cdf, cache, views)
    pair_nsf.write_json(args.output_root/'protocol.json', dict(protocol=PROTOCOL, model=args.model, seeds=args.seeds, views=views,
        retoucher_identities=idents, nsf_settings=nsf.SETTINGS, test_used_for_selection=False,
        note='Evaluation only; frozen checkpoints, scorer, w and z statistics'))
    report(args)


def b_compare(args, y, b, views):
    rows = []
    for view in views:
        m = pd.read_csv(args.output_root/'scores'/f'mask_agreement_{view}.csv', dtype={'crop_path': str})
        m = m[m.roi_status.eq('detected')]
        for v in sb.VARIANTS:
            rows.append(dict(view=view, variant=v, mean_abs_delta_z=float(np.abs(b[view][v]-b['normal'][v]).mean()),
                             **metrics(y, b['normal'][v], b[view][v]),
                             **(dict(mask_iou_mean=float(m.iou.mean()), mask_iou_p05=float(m.iou.quantile(.05))) if v == 'smooth_all' else {})))
    pd.DataFrame(rows).to_csv(args.output_root/'b_compare.csv', index=False)


def report(args):
    rows = pd.read_csv(args.output_root/'summary.csv')
    cols = ['clean_auc', 'fp_cross_auc', 'fn_cross_auc', 'worse_cross_auc', 'retouched_auc', 'real_shift', 'fake_shift', 'fp_flip', 'fn_flip']
    heads = []
    for view, g in rows.groupby('view', sort=False):
        head = g.groupby('method', sort=False)[cols].mean().reindex(METHODS)
        head.insert(0, 'runs', g.groupby('method').size().reindex(METHODS))
        pair = g[g.method.eq('pair')].set_index(['seed', 'train_filter']).worse_cross_auc
        head['runs_over_pair'] = [None if m in ['baseline', 'pair'] else
                                  int((g[g.method.eq(m)].set_index(['seed', 'train_filter']).worse_cross_auc > pair).sum()) for m in METHODS]
        head.insert(0, 'name', [NAMES[m] for m in METHODS])
        head.insert(0, 'view', view)
        heads.append(head.reset_index())
    head = pd.concat(heads, ignore_index=True)
    head.to_csv(args.output_root/'headline.csv', index=False, encoding='utf-8-sig')
    by_filter = rows[rows.method.ne('baseline')].groupby(['view', 'method', 'train_filter'], sort=False)[['clean_auc', 'worse_cross_auc']].mean()
    by_filter.reset_index().to_csv(args.output_root/'by_train_filter.csv', index=False)
    print(f'{args.model} | end-to-end retouchers | baseline seed42, pair/pair+NSF mean of 4 filters x seeds', flush=True)
    with pd.option_context('display.width', 250, 'display.max_columns', 20):
        print(head.set_index(['view', 'method']).drop(columns='name').round(4).to_string(), flush=True)
    compare = args.output_root/'b_compare.csv'
    if compare.is_file():
        print('B alone (z units):', flush=True)
        print(pd.read_csv(compare).set_index(['view', 'variant']).round(4).to_string(), flush=True)
    change = args.output_root/'region_change.csv'
    if change.is_file():
        t = pd.read_csv(change).groupby('view')[['skin', 'eyes_mouth', 'rest']].mean()
        print('Mean |RGB change| by region (0-255, detected frames):', flush=True)
        print(t.round(2).to_string(), flush=True)
    print('Results:', args.output_root, flush=True)
    return head


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=['sample', 'build', 'evaluate', 'report'])
    p.add_argument('--view', choices=VIEWS, help='sample/build: which retoucher')
    p.add_argument('--views', nargs='+', choices=VIEWS, default=VIEWS, help='evaluate: retouchers to evaluate (built ones only)')
    p.add_argument('--model', choices=['xception', 'clip_b16', 'dinov2_b14'])
    for key in ['runs-root', 'evaluation-root', 'pair-nsf-root', 'output-root', 'baseline-runs-root', 'v2-archive-dir']:
        p.add_argument('--'+key, type=Path)
    p.add_argument('--archive-dir', type=Path, help='Drive folder of cdf_test_<view>.tar')
    p.add_argument('--cdf-root', type=Path, default=Path('/content/celebdf_v2_face_crops16_v1'))
    p.add_argument('--cache-root', type=Path, default=Path('/content/end2end_retouch_cache'))
    p.add_argument('--v2-cache-root', type=Path, default=Path('/content/beauty_smoothing_v2_cache'))
    p.add_argument('--weights-dir', type=Path, default=Path('/content/retoucher_weights'))
    p.add_argument('--vendor-dir', type=Path, default=Path('/content/gfpgan_vendor'), help='basicsr taken from its PyPI sdist')
    p.add_argument('--work-dir', type=Path, default=Path('/content/end2end_retouch_work'))
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    for key, default in [('batch-size', 64), ('workers', 2), ('parse-batch', 16)]:
        p.add_argument('--'+key, type=int, default=default)
    p.add_argument('--count', type=int, default=6, help='sample: frames shown (half real, half fake)')
    p.add_argument('--sample-seed', type=int, default=0)
    p.add_argument('--sample-out', type=Path, default=Path('/content/retouch_sample.jpg'))
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    a = p.parse_args(argv)
    if a.action in ['sample', 'build'] and a.view is None:
        p.error(f'{a.action} needs --view')
    if a.action != 'sample' and a.archive_dir is None:
        p.error(f'{a.action} needs --archive-dir')
    if a.action in ['evaluate', 'report']:
        need = ['model', 'output_root'] if a.action == 'report' else \
               ['model', 'runs_root', 'evaluation_root', 'pair_nsf_root', 'output_root', 'baseline_runs_root', 'v2_archive_dir']
        missing = [k for k in need if getattr(a, k) is None]
        if missing:
            p.error(f'{a.action} needs --'+', --'.join(k.replace('_', '-') for k in missing))
        for root in [a.evaluation_root, a.pair_nsf_root, a.runs_root]:
            if root is not None and a.output_root.resolve() == root.resolve():
                p.error('Use a separate output folder; original results must remain unchanged')
    return a


if __name__ == '__main__':
    args = parse_args()
    {'sample': sample, 'build': build, 'evaluate': evaluate, 'report': report}[args.action](args)
