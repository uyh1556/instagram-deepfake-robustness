"""Face-parser ablation for pair+NSF: does the gain depend on the SegFormer parser that draws the eyes/mouth mask?

pair+NSF fuses the pair detector score A with a score B from the face-internal non-skin parts (eyes, brows, lips,
mouth; rest filled grey). Here only the parser behind B's mask changes. Variants, in report order:
  segformer   the pair+NSF parser (jonathandinu/face-parsing SegFormer, CelebAMask-HQ); original scorer reused
  erode1 / erode05 / dilate05 / dilate1
              the SegFormer eyes/mouth mask shrunk / grown by 1% / 0.5% of the crop side before feathering
              (parser boundary error; dilation lets smoothed skin into B). On a test crop 0.5% keeps .61x / 1.45x of
              the mask area and 1% .36x / 2.1x (2% left .19x: eyes and brows gone), while BiSeNet and MediaPipe
              disagree with SegFormer at IoU ~.85, about the 0.5% level
  bisenet     BiSeNet face parser (facexlib v0.2.0 weights, CelebAMask-HQ; different network)
  mediapipe   MediaPipe Face Landmarker (478-point mesh): filled eye, brow and outer-lip polygons (different
              training data and method; no glasses class; frames without a detected face get an empty mask)
B is computed on the same image the detector sees: the clean view on the clean image, each strong-smoothed view on
that smoothed image, with the mask parsed on that same image. Everything else is the fixed pair+NSF protocol: w=.2,
Baseline seed 42 encoder, standardized logistic probe C=1 on the original 2,000 FF++ train frames, z statistics from
the original 1,200 FF++ val frames, saved pair cdf_test predictions and their pair+NSF z statistics. segformer uses
the original scorer (its recomputed clean B must reproduce the saved one); every other variant fits a new probe on
the same FF++ sample with its own masks (FF++ masks parsed on the clean images, as for the original).
Pre-registered on 2026-11-03 before any result: a variant holds on a backbone if pair+NSF beats pair in >= 10/12 runs
(seed x train filter; mean worse cross AUC over the 4 filters at strong strength) and its all-16 worse cross AUC is
within .01 of segformer. Ablation only: nothing is chosen from these results.
Actions:
  masks     (once per parser family, shared by every backbone) parse the FF++ probe sample and the CDF test clean and
            4 strong views; hard masks packed into <mask-dir>/parser_masks_<family>.tar
  prepare   extract the FF++ probe sample;  scores / evaluate / report  per backbone
"""
import argparse
import hashlib
import json
import pickle
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import nsf_smoothed_b_v2 as sb

rc, nsf, fusion, occlusion, diagonal, base, evaluation, v2 = sb.rc, sb.nsf, sb.fusion, sb.occlusion, sb.diagonal, sb.base, sb.evaluation, sb.v2
FILTERS, VIEWS, STRONG = sb.FILTERS, sb.VIEWS, sb.STRONG
FAMILIES = ['segformer', 'bisenet', 'mediapipe']
MORPH = {'erode1': -.01, 'erode05': -.005, 'dilate05': .005, 'dilate1': .01}
VARIANTS = ['segformer', *MORPH, 'bisenet', 'mediapipe']
FAMILY = {v: 'segformer' if v in MORPH else v for v in VARIANTS}
NAMES = {'segformer': 'SegFormer (원본, 제안)', 'erode1': 'SegFormer 마스크 축소 1%', 'erode05': 'SegFormer 마스크 축소 0.5%',
         'dilate05': 'SegFormer 마스크 확대 0.5%', 'dilate1': 'SegFormer 마스크 확대 1%', 'bisenet': 'BiSeNet',
         'mediapipe': 'MediaPipe Face Mesh', 'none': 'pair (B 없음, 참고)'}
PROTOCOL = 'nsf_parser_ablation_v1'
SETTINGS = dict(nsf.SETTINGS, variants=VARIANTS, morph=MORPH, strength=STRONG, blur_fraction=v2.PARSER['blur_fraction'])
CRITERION = dict(runs_over_pair=10, runs=12, max_gap_to_segformer=.01)
BISENET = dict(url='https://github.com/xinntao/facexlib/releases/download/v0.2.0/parsing_bisenet.pth',
               sha256='468e13ca13a9b43cc0881a9f99083a430e9c0a38abd935431d1c28ee94b26567', facexlib='0.3.0',
               input_size=512, mean=[.485, .456, .406], std=[.229, .224, .225],
               labels=dict(l_brow=2, r_brow=3, l_eye=4, r_eye=5, eye_g=6, mouth=11, u_lip=12, l_lip=13))
# Closed outlines of the Face Landmarker mesh, ordered from its FaceLandmarksConnections edges (eyes and outer lips are
# loops; each brow is its lower edge followed by its upper edge reversed). The outer-lip polygon covers lips + mouth.
MEDIAPIPE = dict(version='1.1.1', num_faces=1,
                 url='https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
                 sha256='64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff',
                 polygons=dict(
                     right_eye=[33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7],
                     left_eye=[263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249],
                     right_brow=[46, 53, 52, 65, 55, 107, 66, 105, 63, 70],
                     left_brow=[276, 283, 282, 295, 285, 336, 296, 334, 293, 300],
                     lips=[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]))
FAMILY_SETTINGS = dict(segformer=dict(model=v2.PARSER['model'], revision=v2.PARSER['revision'], labels=fusion.INNER),
                       bisenet=BISENET, mediapipe=MEDIAPIPE)


def mask_path(work_dir, family, view, crop_path):
    """view: 'ffpp' (FF++ probe sample, clean) or a CDF test view."""
    return work_dir/family/view/(crop_path+'.png')


def read_hard(path):
    with Image.open(path) as opened:
        return np.asarray(opened) > 127


def morphed(hard, morph=0.):
    """Hard mask after an optional erosion (morph < 0) or dilation (morph > 0) by |morph| of the crop side."""
    if not morph:
        return hard
    r = max(1, int(round(abs(morph)*min(hard.shape))))
    kernel = v2.cv2.getStructuringElement(v2.cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
    return (v2.cv2.dilate if morph > 0 else v2.cv2.erode)(hard.astype(np.uint8), kernel) > 0


def keep_mask(hard, morph=0.):
    """Soft keep mask exactly as pair+NSF builds it (feather, then the 8-bit PNG round trip) of the morphed mask."""
    side = min(hard.shape)
    soft = v2.cv2.GaussianBlur(morphed(hard, morph).astype(np.float32), (0, 0), max(1., v2.PARSER['blur_fraction']*side)).clip(0, 1)
    return v2.image_to_mask(v2.mask_to_image(soft))


def landmark_mask(points, shape):
    """Filled eye, brow and outer-lip polygons of one face; points are normalized (x, y) of the mesh landmarks."""
    h, w = shape
    out = np.zeros((h, w), np.uint8)
    if points is None:
        return out.astype(bool)
    xy = np.asarray(points, np.float64)*[w, h]
    for ring in MEDIAPIPE['polygons'].values():
        v2.cv2.fillPoly(out, [np.rint(xy[ring]).astype(np.int32)], 1)
    return out.astype(bool)


class SegformerInner(rc.LabelParser):
    """The pinned pair+NSF parser; hard eyes/mouth masks (before feathering)."""
    def __init__(self, args):
        super().__init__(args.device)
        names = {v: int(k) for k, v in self.model.config.id2label.items()}
        self.keep = [names[n] for n in fusion.INNER]

    def hard(self, arrays):
        return [np.isin(label, self.keep) for label in self.labels([Image.fromarray(a) for a in arrays])]


class BisenetInner:
    """BiSeNet (facexlib) at 512x512 with ImageNet normalization; logits resized back to the crop before argmax."""
    def __init__(self, args):
        from facexlib.parsing.bisenet import BiSeNet
        path = fetch(BISENET['url'], args.weights_dir/'parsing_bisenet.pth', BISENET['sha256'])
        self.net = BiSeNet(num_class=19)
        self.net.load_state_dict(torch.load(path, map_location='cpu'), strict=True)
        self.net.to(args.device).eval()
        self.device, self.keep = args.device, list(BISENET['labels'].values())
        self.mean = torch.tensor(BISENET['mean'], device=args.device).view(1, 3, 1, 1)
        self.std = torch.tensor(BISENET['std'], device=args.device).view(1, 3, 1, 1)

    def hard(self, arrays):
        size = BISENET['input_size']
        x = np.stack([v2.cv2.resize(a, (size, size), interpolation=v2.cv2.INTER_LINEAR) for a in arrays])
        x = torch.from_numpy(x).to(self.device).permute(0, 3, 1, 2).float()/255.
        with torch.inference_mode():
            logits = self.net((x-self.mean)/self.std)[0].float()
        return [np.isin(torch.nn.functional.interpolate(logit[None], size=a.shape[:2], mode='bilinear', align_corners=False)
                        .argmax(1)[0].cpu().numpy(), self.keep) for logit, a in zip(logits, arrays)]


class MediapipeInner:
    """MediaPipe Face Landmarker (CPU, image mode, one face) -> filled eye/brow/lip polygons."""
    def __init__(self, args):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        if mp.__version__ != MEDIAPIPE['version']:
            raise ValueError(f"mediapipe {mp.__version__} installed; pinned {MEDIAPIPE['version']}")
        path = fetch(MEDIAPIPE['url'], args.weights_dir/'face_landmarker.task', MEDIAPIPE['sha256'])
        options = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(path)),
                                               running_mode=vision.RunningMode.IMAGE, num_faces=MEDIAPIPE['num_faces'])
        self.mp, self.landmarker = mp, vision.FaceLandmarker.create_from_options(options)

    def points(self, array):
        found = self.landmarker.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(array)))
        return [(p.x, p.y) for p in found.face_landmarks[0]] if found.face_landmarks else None

    def hard(self, arrays):
        return [landmark_mask(self.points(a), a.shape[:2]) for a in arrays]


PARSERS = dict(segformer=SegformerInner, bisenet=BisenetInner, mediapipe=MediapipeInner)


def fetch(url, dest, sha256):
    """Download once and check the pinned sha256."""
    if not dest.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(url, str(dest))
    if hashlib.sha256(dest.read_bytes()).hexdigest() != sha256:
        raise ValueError(f'Downloaded file hash differs from the pinned one: {dest}')
    return dest


def family_identity(family):
    return dict(protocol=PROTOCOL, family=family, settings=FAMILY_SETTINGS[family], strength=STRONG)


def mask_sets(args):
    """[(view, frame, data root)]: the FF++ probe sample (clean) and the 5 CDF test views."""
    train, val = rc.ffpp_frames(args)
    cdf = rc.cdf_frame(args)
    return [('ffpp', pd.concat([train, val], ignore_index=True), args.source_root), *((v, cdf, args.cdf_root) for v in VIEWS)]


def masks(args):
    """Parse every set with one parser family and pack the hard masks into one Drive tar (skipped when it exists)."""
    family = args.family
    archive = args.mask_dir/f'parser_masks_{family}.tar'
    if archive.is_file():
        print('Already built:', archive, flush=True)
        return
    folder = args.work_dir/family
    sets = mask_sets(args)
    cache = diagonal.restore_cache(args.cache_root, args.archive_dir, 'cdf_test', v2.SmoothingCache.PARTS)
    parser = PARSERS[family](args)
    stats = []
    for i, (view, frame, root) in enumerate(sets, 1):
        dataset = sb.ViewImages(frame, root, cache if view in FILTERS else None, view if view in FILTERS else 'normal')
        loader = DataLoader(dataset, batch_size=args.parse_batch, num_workers=args.workers, collate_fn=sb.as_list)
        for batch in tqdm(loader, desc=f'[{i}/{len(sets)}] {family} {view}', unit='batch', mininterval=10):
            for (array, j), hard in zip(batch, parser.hard([a for a, _ in batch])):
                path = frame.crop_path.iloc[j]
                out = mask_path(args.work_dir, family, view, path)
                out.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(hard.astype(np.uint8)*255, mode='L').save(out)
                stats.append(dict(view=view, crop_path=path, roi_status=frame.roi_status.iloc[j] if 'roi_status' in frame else 'detected',
                                  area=float(hard.mean())))
    table = pd.DataFrame(stats)
    table.to_csv(folder/'mask_area.csv', index=False)
    summary = table.groupby('view', sort=False).agg(frames=('area', 'size'), mean_area=('area', 'mean'),
                                                    empty=('area', lambda a: int((a == 0).sum()))).reset_index()
    print(summary.to_string(index=False), flush=True)
    sb.rc.pair_nsf.write_json(folder/'identity.json', family_identity(family))
    sb.rc.pair_nsf.write_json(folder/'complete.json', dict(frames={r.view: int(r.frames) for r in summary.itertuples()}))
    args.mask_dir.mkdir(parents=True, exist_ok=True)
    temp = archive.with_suffix('.tmp.tar')
    subprocess.run(['tar', '-cf', str(temp), '-C', str(args.work_dir), family], check=True)
    temp.replace(archive)
    print('Saved:', archive, flush=True)


def restore_masks(args, sets):
    """Unpack every family's mask tar (once) and check its identity and frame counts."""
    for family in FAMILIES:
        folder = args.work_dir/family
        if not (folder/'complete.json').is_file():
            archive = args.mask_dir/f'parser_masks_{family}.tar'
            if not archive.is_file():
                raise FileNotFoundError(f'Build the {family} masks first (build_nsf_parser_masks_v1): {archive}')
            print('Restoring', archive, flush=True)
            args.work_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(['tar', '-xf', str(archive), '-C', str(args.work_dir)], check=True)
        if json.loads((folder/'identity.json').read_text()) != family_identity(family):
            raise ValueError(f'{family} masks were built with other settings: {folder}')
        counts = json.loads((folder/'complete.json').read_text())['frames']
        if any(counts.get(view) != len(frame) for view, frame, _ in sets):
            raise ValueError(f'{family} masks do not cover the expected frames: {counts}')


class VariantInputs(Dataset):
    """One view of each frame as the face_inner inputs of every variant (masks parsed on that same view), plus the
    agreement of each variant's hard mask with the SegFormer one (IoU, area ratio)."""
    def __init__(self, frame, root, view, work_dir, transform, cache=None):
        self.rows, self.root, self.view, self.work_dir, self.transform, self.cache = frame.to_dict('records'), root, view, work_dir, transform, cache

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = row['crop_path']
        with Image.open(self.root/path) as opened:
            image = opened.convert('RGB')
        if self.view in FILTERS:
            image = self.cache.view(image, path, self.view, STRONG, row.get('roi_status', 'detected') == 'detected')
        hard = {f: read_hard(mask_path(self.work_dir, f, self.view, path)) for f in FAMILIES}
        if any(h.shape != image.size[::-1] for h in hard.values()):
            raise ValueError(f'Mask size differs: {path}')
        inputs, agree = [], []
        for v in VARIANTS:
            inputs.append(self.transform(occlusion.occlude(image, 1.-keep_mask(hard[FAMILY[v]], MORPH.get(v, 0.)))))
            a, b = hard['segformer'], morphed(hard[FAMILY[v]], MORPH.get(v, 0.))
            union, area = (a | b).sum(), a.sum()
            agree.append([(a & b).sum()/union if union else np.nan, b.sum()/area if area else np.nan, float(b.sum() == 0)])
        return torch.stack(inputs), index, torch.tensor(agree, dtype=torch.float64)


def variant_features(args, model, dataset, desc):
    """([n, variants, d] classifier-input features, [n, variants, 3] mask agreement); one forward pass per frame."""
    captured = {}
    hook = base.head(model, args.model).register_forward_pre_hook(lambda m, inp: captured.__setitem__('x', inp[0].detach()))
    feats, agree = [], []
    try:
        with torch.inference_mode():
            for x, _, a in tqdm(DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers),
                                desc=desc, unit='batch', mininterval=10):
                b = x.shape[0]
                model(x.flatten(0, 1).to(args.device))
                feats.append(captured['x'].float().flatten(1).cpu().numpy().reshape(b, len(VARIANTS), -1))
                agree.append(a.numpy())
    finally:
        hook.remove()
    return np.concatenate(feats), np.concatenate(agree)


def agreement_rows(view, frame, agree):
    detected = (frame.roi_status.eq('detected') if 'roi_status' in frame else pd.Series(True, index=frame.index)).to_numpy()
    rows = []
    for i, v in enumerate(VARIANTS):
        iou, ratio, empty = agree[detected, i, 0], agree[detected, i, 1], agree[detected, i, 2]
        rows.append(dict(view=view, variant=v, frames=int(detected.sum()), iou_vs_segformer=float(np.nanmean(iou)),
                         iou_p05=float(np.nanquantile(iou, .05)), area_ratio=float(np.nanmean(ratio)), empty_rate=float(empty.mean())))
    return rows


def fit_probes(args, model, transform, out, original):
    """Probes of every variant on the FF++ sample (segformer = the original scorer); stats give the z of each B."""
    train, val = rc.ffpp_frames(args)
    f_train, _ = variant_features(args, model, VariantInputs(train, args.source_root, 'ffpp', args.work_dir, transform), 'FF++ train 2000')
    f_val, agree = variant_features(args, model, VariantInputs(val, args.source_root, 'ffpp', args.work_dir, transform), 'FF++ val 1200')
    y_train, y_val = train.class_index.astype(int).to_numpy(), val.class_index.astype(int).to_numpy()
    pipe, info = original
    probes, stats = {}, {}
    for i, v in enumerate(VARIANTS):
        if v == 'segformer':
            probes[v] = pipe
            s = pipe.decision_function(f_val[:, i])
            gap = abs(float(s.mean())-info['val_mean'])/info['val_std']
            print(f'segformer: original scorer on recomputed FF++ val masks, mean differs by {gap:.4f} sd', flush=True)
            if gap > .02:
                raise ValueError('Recomputed SegFormer FF++ masks do not reproduce the original NSF val scores')
            stats[v] = dict(val_mean=info['val_mean'], val_std=info['val_std'], ffpp_val_auc=info['ffpp_val_auc'])
        else:
            probes[v] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=nsf.SETTINGS['probe_c'])).fit(f_train[:, i], y_train)
            s = probes[v].decision_function(f_val[:, i])
            stats[v] = dict(val_mean=float(s.mean()), val_std=float(s.std()),
                            ffpp_val_auc=float(fusion.shift.auc(s[y_val == 0], s[y_val == 1])))
        print(f'{args.model} {v}: probe FF++ val AUC {stats[v]["ffpp_val_auc"]:.3f}', flush=True)
    with open(out/'probes.pkl', 'wb') as f:
        pickle.dump(probes, f)
    pd.DataFrame(agreement_rows('ffpp', val, agree)).to_csv(out/'mask_agreement_ffpp.csv', index=False)
    return probes, stats


def scores(args):
    """Fit the variant probes (once) and score B of every variant on each CDF view; cached under output-root/scores."""
    out = args.output_root/'scores'
    out.mkdir(parents=True, exist_ok=True)
    info_path = out/'probes.json'
    original = sb.scorer(args)
    initial, initial_hash = evaluation.checkpoint(argparse.Namespace(runs_root=args.baseline_runs_root, seed=42), args.model)
    if initial_hash != original[1]['baseline_sha256']:
        raise ValueError('Baseline checkpoint differs from the one behind the NSF scorer')
    identity = dict(protocol=PROTOCOL, model=args.model, settings=SETTINGS, baseline_sha256=initial_hash,
                    scorer_pickle_sha256=original[1]['pickle_sha256'], families={f: family_identity(f) for f in FAMILIES})
    cdf = rc.cdf_frame(args)
    done = info_path.is_file() and json.loads(info_path.read_text())['identity'] == identity
    if done and all((out/f'cdf_test_{v}.csv').is_file() for v in VIEWS):
        print('All B scores cached:', out, flush=True)
        return
    if info_path.is_file() and not done:
        raise ValueError(f'Cached scores were made with other settings; use a new output folder: {out}')
    restore_masks(args, mask_sets(args))
    model, transform = nsf.load(args, args.model, initial['model_state_dict'])
    if done:
        saved = json.loads(info_path.read_text())
        if hashlib.sha256((out/'probes.pkl').read_bytes()).hexdigest() != saved['pickle_sha256']:
            raise ValueError('Probe file changed')
        with open(out/'probes.pkl', 'rb') as f:
            probes = pickle.load(f)
    else:
        probes, stats = fit_probes(args, model, transform, out, original)
        rc.pair_nsf.write_json(info_path, dict(identity=identity, stats=stats,
                                               pickle_sha256=hashlib.sha256((out/'probes.pkl').read_bytes()).hexdigest()))
    cache = diagonal.restore_cache(args.cache_root, args.archive_dir, 'cdf_test', v2.SmoothingCache.PARTS)
    for i, view in enumerate(VIEWS, 1):
        dest = out/f'cdf_test_{view}.csv'
        if dest.is_file():
            print(f'[{i}/{len(VIEWS)}] cached: {view}', flush=True)
            continue
        f, agree = variant_features(args, model, VariantInputs(cdf, args.cdf_root, view, args.work_dir, transform, cache),
                                    f'[{i}/{len(VIEWS)}] CDF test {view}')
        table = cdf[['crop_path']].copy()
        for j, v in enumerate(VARIANTS):
            table[v] = probes[v].decision_function(f[:, j])
        if view == 'normal':
            rc.pair_nsf.write_json(out/'clean_check.json', sb.check_clean(table.segformer.to_numpy(), sb.saved_scores(args, cdf, original[1])))
        pd.DataFrame(agreement_rows(view, cdf, agree)).to_csv(out/f'mask_agreement_{view}.csv', index=False)
        temp = dest.with_suffix('.tmp.csv'); table.to_csv(temp, index=False); temp.replace(dest)
    del model
    if args.device == 'cuda':
        torch.cuda.empty_cache()


def view_scores(args, cdf):
    """{view: {variant: z-scored B}}."""
    stats = json.loads((args.output_root/'scores'/'probes.json').read_text())['stats']
    out = {}
    for view in VIEWS:
        t = pd.read_csv(args.output_root/'scores'/f'cdf_test_{view}.csv', dtype={'crop_path': str})
        if t.crop_path.tolist() != cdf.crop_path.tolist():
            raise ValueError(f'B score order differs: {view}')
        out[view] = {v: (t[v].to_numpy()-stats[v]['val_mean'])/stats[v]['val_std'] for v in VARIANTS}
    return out


def evaluate(args):
    """Fuse the saved pair predictions of every seed with each variant's B (A and B on the same view)."""
    cdf = rc.cdf_frame(args, status=False)
    y = cdf.class_index.astype(int).to_numpy()
    b = view_scores(args, cdf)
    w = nsf.SETTINGS['weight']
    for seed in args.seeds:
        source = args.evaluation_root/f'seed{seed}'/'cdf_test'
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
        target = args.output_root/f'seed{seed}'
        target.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(target/'summary.csv', index=False)
        print(f'seed{seed}: {len(rows)} rows', flush=True)
    b_alone(args, y, b)
    report(args)


def b_alone(args, y, b):
    """B on its own (no fusion) per variant and filter."""
    stats = json.loads((args.output_root/'scores'/'probes.json').read_text())['stats']
    rows = [dict(variant=v, filter=ev, ffpp_val_auc=stats[v]['ffpp_val_auc'], **rc.metrics(y, b['normal'][v], b[ev][v]))
            for v in VARIANTS for ev in FILTERS]
    pd.DataFrame(rows).to_csv(args.output_root/'b_alone.csv', index=False)


def headline(rows):
    """Per variant (report order, pair last): seed-mean worse cross AUC, runs over pair (/12), cells over pair (/16),
    gap to segformer and the pre-registered verdict."""
    run = rows.groupby(['variant', 'seed', 'train_filter'], as_index=False).worse_cross_auc.mean()
    run = run.pivot_table(index=['seed', 'train_filter'], columns='variant', values='worse_cross_auc')
    cell = rows.groupby(['variant', 'train_filter', 'filter', 'held_out'], as_index=False)[['clean_auc', 'worse_cross_auc', 'real_shift', 'fake_shift']].mean()
    pivot = cell.pivot_table(index=['train_filter', 'filter'], columns='variant', values='worse_cross_auc')
    reference = cell[cell.variant.eq('segformer')].worse_cross_auc.mean()
    head = []
    for v in [*VARIANTS, 'none']:
        g = cell[cell.variant.eq(v)]
        row = dict(variant=v, name=NAMES[v], seeds=rows.seed.nunique(), clean_auc=g.clean_auc.mean(),
                   seen_worse=g[~g.held_out].worse_cross_auc.mean(), unseen_worse=g[g.held_out].worse_cross_auc.mean(),
                   all16_worse=g.worse_cross_auc.mean(), real_shift=g.real_shift.mean(), fake_shift=g.fake_shift.mean())
        if v != 'none':
            row.update(gap_to_segformer=row['all16_worse']-reference, runs_over_pair=int((run[v] > run['none']).sum()),
                       runs=len(run), cells_over_pair=int((pivot[v] > pivot['none']).sum()))
            row['holds'] = bool(row['runs_over_pair'] >= CRITERION['runs_over_pair'] and len(run) == CRITERION['runs']
                                and abs(row['gap_to_segformer']) <= CRITERION['max_gap_to_segformer'])
        head.append(row)
    return pd.DataFrame(head), cell


def report(args):
    files = sorted(args.output_root.glob('seed*/summary.csv'))
    rows = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    head, cell = headline(rows)
    head.to_csv(args.output_root/'parser_headline.csv', index=False, encoding='utf-8-sig')
    cell.to_csv(args.output_root/'parser_seed_means.csv', index=False)
    agreement = pd.concat([pd.read_csv(f) for f in sorted((args.output_root/'scores').glob('mask_agreement_*.csv'))], ignore_index=True)
    agreement.to_csv(args.output_root/'mask_agreement.csv', index=False)
    print(f'{args.model} | seeds {sorted(rows.seed.unique())} | worse cross AUC at strong smoothing', flush=True)
    print(head.drop(columns='name').round(4).to_string(index=False), flush=True)
    clean = agreement[agreement.view.eq('normal')].set_index('variant').loc[VARIANTS]
    print('Mask vs SegFormer on clean CDF test (detected frames):', flush=True)
    print(clean[['iou_vs_segformer', 'iou_p05', 'area_ratio', 'empty_rate']].round(3).to_string(), flush=True)
    alone = args.output_root/'b_alone.csv'
    if alone.is_file():
        t = pd.read_csv(alone).groupby('variant', sort=False)[['ffpp_val_auc', 'clean_auc', 'worse_cross_auc', 'real_shift', 'fake_shift']].mean()
        print('B alone (mean over 4 filters):', flush=True)
        print(t.round(4).to_string(), flush=True)
    print('Results:', args.output_root, flush=True)
    return head


REQUIRED = dict(masks=['runs-root', 'evaluation-root', 'source-root', 'archive-dir', 'mask-dir'],
                prepare=['runs-root', 'source-root', 'drive-data'],
                scores=['runs-root', 'evaluation-root', 'output-root', 'baseline-runs-root', 'source-root', 'archive-dir', 'mask-dir'],
                evaluate=['evaluation-root', 'pair-nsf-root', 'output-root'], report=['output-root'])


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=list(REQUIRED))
    p.add_argument('--model', required=True, choices=['xception', 'clip_b16', 'dinov2_b14'])
    p.add_argument('--family', choices=FAMILIES)
    for key in ['runs-root', 'evaluation-root', 'pair-nsf-root', 'output-root', 'baseline-runs-root', 'source-root',
                'archive-dir', 'mask-dir', 'drive-data']:
        p.add_argument('--'+key, type=Path)
    p.add_argument('--cdf-root', type=Path, default=Path('/content/celebdf_v2_face_crops16_v1'))
    p.add_argument('--cache-root', type=Path, default=Path('/content/beauty_smoothing_v2_cache'))
    p.add_argument('--work-dir', type=Path, default=Path('/content/nsf_parser_masks'))
    p.add_argument('--weights-dir', type=Path, default=Path('/content/parser_weights'))
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    for key, default in [('batch-size', 16), ('workers', 2), ('parse-batch', 16)]:
        p.add_argument('--'+key, type=int, default=default)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    a = p.parse_args(argv)
    missing = [k for k in REQUIRED[a.action] if getattr(a, k.replace('-', '_')) is None]
    if missing:
        p.error(f'{a.action} needs --' + ', --'.join(missing))
    if a.action == 'masks' and a.family is None:
        p.error('masks needs --family')
    if a.output_root is not None:
        for root in [a.evaluation_root, a.pair_nsf_root, a.runs_root]:
            if root is not None and a.output_root.resolve() == root.resolve():
                p.error('Use a separate output folder; original results must remain unchanged')
    return a


if __name__ == '__main__':
    args = parse_args()
    {'masks': masks, 'prepare': rc.prepare, 'scores': scores, 'evaluate': evaluate, 'report': report}[args.action](args)
