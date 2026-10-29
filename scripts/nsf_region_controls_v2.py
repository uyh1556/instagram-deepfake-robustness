"""Region controls for pair+NSF: is the gain from the eyes/mouth, or from fusing any second score?

pair+NSF fuses the pair detector score A with a score B from the face-internal non-skin parts (eyes, brows, lips,
mouth). Here only the region of B changes; everything else is the pair+NSF protocol (w=.2, Baseline seed 42 encoder,
standardized logistic probe C=1 on 2,000 balanced FF++ train frames, z statistics from 1,200 FF++ val frames, saved
pair cdf_test predictions and their pair+NSF z statistics). Regions of B (rest filled grey like face_inner):
  full    the whole crop, no mask                                   -> is it just a two-model ensemble?
  face    the whole face (skin, nose, eyes, brows, glasses, lips, mouth; no ears/hair/neck/background)
                                                                    -> does removing the skin matter?
  random  a square of the eyes/mouth area at a random face position -> does the eyes/mouth position matter?
  face_inner (reference) the original pair+NSF score, read from the saved NSF artifacts
Unlike face_inner, these regions contain smoothed skin, so B is scored on the clean view AND each strong view
(normal + 4 filters at alpha .75); masks come from parsing the clean image (smoothing only edits the skin).
B depends only on the backbone, so it is scored once (cached in output-root/scores) and every --seed reuses it.
Metrics are the main ones at strong strength: clean AUC, both cross AUCs, the worse cross AUC, score shifts.
Ablation only: no weight, region or epoch is chosen from these results. No training; original runs/evaluations
are read-only.
"""
import argparse
import hashlib
import json
import pickle
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

import evaluate_pair_nsf_v2 as pair_nsf
import nonskin_fusion_v2 as nsf

fusion, occlusion, diagonal, base, evaluation = nsf.fusion, nsf.occlusion, nsf.diagonal, nsf.base, nsf.evaluation
v2, shift = fusion.v2, fusion.shift
FILTERS = list(v2.FILTERS)
STRONG = v2.LEVELS['smooth_strong']
VIEWS = ['normal', *FILTERS]
REGIONS = ['full', 'face', 'random']
FACE = ['skin', 'nose', *fusion.INNER]
PROTOCOL = 'nsf_region_controls_v2_v1'
SETTINGS = dict(nsf.SETTINGS, regions=REGIONS, face_labels=FACE, random_seed=42, strength=STRONG)
CDF_MANIFEST = 'manifests/celebdf_v2_roi16_test_v1.csv'


def label_path(work_dir, dataset, crop_path):
    return work_dir/'labels'/dataset/(crop_path+'.png')


class LabelParser(v2.SkinParser):
    """Same pinned face parser as the v2 skin mask; returns the raw label map of each image."""
    def labels(self, images):
        with self.torch.inference_mode():
            pixels = self.processor(images=images, return_tensors='pt')['pixel_values'].to(self.device)
            logits = self.model(pixel_values=pixels).logits.float()
        return [self.torch.nn.functional.interpolate(logit[None], size=image.size[::-1], mode='bilinear', align_corners=False)
                .argmax(1)[0].cpu().numpy().astype(np.uint8) for logit, image in zip(logits, images)]


def parse_labels(args, dataset, frame, root):
    """Label maps of every frame (PNG under work-dir/labels/<dataset>), parsed once per Colab session."""
    todo = [p for p in frame.crop_path if not label_path(args.work_dir, dataset, p).is_file()]
    if not todo:
        return
    parser = LabelParser(args.device)
    for start in tqdm(range(0, len(todo), args.parse_batch), desc=f'face parsing {dataset}', unit='batch', mininterval=10):
        chunk = todo[start:start+args.parse_batch]
        images = []
        for p in chunk:
            with Image.open(root/p) as opened:
                images.append(opened.convert('RGB'))
        for p, label in zip(chunk, parser.labels(images)):
            out = label_path(args.work_dir, dataset, p)
            out.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(label, mode='L').save(out)
    del parser
    if args.device == 'cuda':
        torch.cuda.empty_cache()


def label_ids():
    """{label name: id} of the pinned face parser (read from its config, no weights loaded)."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(v2.PARSER['model'], revision=v2.PARSER['revision'])
    ids = {v: int(k) for k, v in cfg.id2label.items()}
    missing = [n for n in FACE if n not in ids]
    if missing:
        raise ValueError(f'Face parser lacks labels {missing}')
    return ids


def feather(hard, side):
    """Same soft edge as the face_inner keep mask."""
    return v2.cv2.GaussianBlur(hard.astype(np.float32), (0, 0), max(1., v2.PARSER['blur_fraction']*side)).clip(0, 1)


def random_square(face, area, key):
    """Hard square of ~area pixels centred on a face pixel drawn by a per-image RNG (fixed for every view/run)."""
    h, w = face.shape
    out = np.zeros_like(face, dtype=bool)
    if area <= 0:
        return out
    side = int(min(round(np.sqrt(area)), h, w))
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little'))
    ys, xs = np.nonzero(face)
    if len(ys) == 0:
        ys, xs = np.mgrid[:h, :w].reshape(2, -1)
    i = rng.integers(len(ys))
    top = int(np.clip(ys[i]-side//2, 0, h-side))
    left = int(np.clip(xs[i]-side//2, 0, w-side))
    out[top:top+side, left:left+side] = True
    return out


def keep_masks(label, ids, key):
    """{'face', 'random'} soft keep masks of one label map (full keeps everything)."""
    face = np.isin(label, [ids[n] for n in FACE])
    inner = np.isin(label, [ids[n] for n in fusion.INNER])
    side = min(label.shape)
    return dict(face=feather(face, side), random=feather(random_square(face, int(inner.sum()), key), side))


class RegionViews(Dataset):
    """One view (clean or a strong-smoothed filter) of each frame, returned as the three region inputs at once."""
    def __init__(self, frame, root, dataset, args, ids, transform, cache=None, filt=None):
        self.rows, self.root, self.dataset, self.args = frame.to_dict('records'), root, dataset, args
        self.ids, self.transform, self.cache, self.filt = ids, transform, cache, filt

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = row['crop_path']
        with Image.open(self.root/path) as opened:
            image = opened.convert('RGB')
        if self.filt is not None:
            image = self.cache.view(image, path, self.filt, STRONG, row.get('roi_status', 'detected') == 'detected')
        with Image.open(label_path(self.args.work_dir, self.dataset, path)) as opened:
            label = np.asarray(opened)
        if label.shape != image.size[::-1]:
            raise ValueError(f'Label map size differs: {path}')
        keep = keep_masks(label, self.ids, f"{SETTINGS['random_seed']}|{path}")
        views = [image, *(occlusion.occlude(image, 1.-keep[r]) for r in REGIONS[1:])]
        return torch.stack([self.transform(v) for v in views]), index


def region_features(args, model, name, dataset, desc):
    """{region: classifier-input features} with a progress bar; one forward pass carries all three regions."""
    captured = {}
    hook = base.head(model, name).register_forward_pre_hook(lambda m, inp: captured.__setitem__('x', inp[0].detach()))
    feats = []
    try:
        with torch.inference_mode():
            loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)
            for x, _ in tqdm(loader, desc=desc, unit='batch', mininterval=10):
                b = x.shape[0]
                model(x.flatten(0, 1).to(args.device))
                feats.append(captured['x'].float().flatten(1).cpu().numpy().reshape(b, len(REGIONS), -1))
    finally:
        hook.remove()
    f = np.concatenate(feats)
    return {r: f[:, i] for i, r in enumerate(REGIONS)}


def nsf_info(args):
    folder = args.runs_root/'nonskin_probe'
    info = json.loads((folder/f'{args.model}_face_inner_seed42.json').read_text())
    if info['settings'] != nsf.SETTINGS:
        raise ValueError('Original NSF scorer settings differ')
    return info


def ffpp_frames(args):
    sub = argparse.Namespace(source_root=args.source_root)
    train, val = nsf.ffpp_samples(sub)
    info = nsf_info(args)
    for frame, key in [(train, 'train_paths_sha256'), (val, 'val_paths_sha256')]:
        if hashlib.sha256('\n'.join(frame.crop_path).encode()).hexdigest() != info[key]:
            raise ValueError('FF++ probe sample differs from the original NSF sample; frozen manifests must be preserved')
    return train, val


def prepare(args):
    """Extract only the 2,000 train + 1,200 val FF++ frames of the original NSF sample from the Drive TARs."""
    master = args.source_root/'manifests'/pair_nsf.MASTER
    method_tar = args.drive_data/'ffpp_c23_real_roi16_trainval_v1.tar'
    combined = args.drive_data/'ffpp_c23_4methods_roi16_trainval_v1.tar'
    if not master.is_file():
        pair_nsf.archive_members(method_tar if method_tar.is_file() else combined, ['manifests/'+pair_nsf.MASTER], args.source_root)
    train, val = ffpp_frames(args)
    selected = pd.concat([train, val])
    for method, group in selected.groupby('method', sort=False):
        needed = [p for p in group.crop_path if not (args.source_root/p).is_file()]
        if needed:
            path = args.drive_data/f'ffpp_c23_{pair_nsf.SLUGS[method]}_roi16_trainval_v1.tar'
            pair_nsf.archive_members(path if path.is_file() else combined, needed, args.source_root)
    if any(not (args.source_root/p).is_file() for p in selected.crop_path):
        raise RuntimeError('FF++ probe-sample extraction incomplete')
    print(f'Prepared {len(selected)} FF++ frames (original NSF train/val sample)', flush=True)


def cdf_frame(args, status=True):
    """Official CDF test frames in the saved-prediction order (seed 42 pair normal CSV); status adds roi_status."""
    ref = pair_nsf.read_prediction(args.evaluation_root/'seed42'/'cdf_test'/f'{args.model}_aug_pair@{FILTERS[0]}_normal.csv')
    if not status:
        return ref[['crop_path', 'video_id', 'class_index']]
    manifest = pd.read_csv(args.cdf_root/CDF_MANIFEST, dtype=str, keep_default_na=False)[['crop_path', 'roi_status']]
    frame = ref[['crop_path', 'video_id', 'class_index']].merge(manifest, on='crop_path', how='left', validate='one_to_one')
    if len(frame) != 8288 or frame.roi_status.isna().any():
        raise ValueError('Expected the 8,288 official CDF test frames in the CDF manifest')
    return frame


def scores(args):
    """Fit the three region probes (once) and score FF++ val + every CDF view; cached under output-root/scores."""
    out = args.output_root/'scores'
    out.mkdir(parents=True, exist_ok=True)
    info_path = out/'probes.json'
    initial, initial_hash = evaluation.checkpoint(argparse.Namespace(runs_root=args.baseline_runs_root, seed=42), args.model)
    if initial_hash != nsf_info(args)['baseline_sha256']:
        raise ValueError('Baseline checkpoint differs from the one behind the NSF scorer')
    identity = dict(protocol=PROTOCOL, model=args.model, settings=SETTINGS, baseline_sha256=initial_hash)
    cdf = cdf_frame(args)
    done = info_path.is_file() and json.loads(info_path.read_text())['identity'] == identity
    if done and all((out/f'cdf_test_{v}.csv').is_file() for v in VIEWS):
        print('All region scores cached:', out, flush=True)
        return
    ids = label_ids()
    model, transform = nsf.load(args, args.model, initial['model_state_dict'])
    if done:
        saved = json.loads(info_path.read_text())
        with open(out/'probes.pkl', 'rb') as f:
            probes = pickle.load(f)
        if hashlib.sha256((out/'probes.pkl').read_bytes()).hexdigest() != saved['pickle_sha256']:
            raise ValueError('Region probe file changed')
    else:
        train, val = ffpp_frames(args)
        parse_labels(args, 'ffpp', pd.concat([train, val]), args.source_root)
        f_train = region_features(args, model, args.model, RegionViews(train, args.source_root, 'ffpp', args, ids, transform), 'FF++ train 2000')
        f_val = region_features(args, model, args.model, RegionViews(val, args.source_root, 'ffpp', args, ids, transform), 'FF++ val 1200')
        y_train, y_val = train.class_index.astype(int).to_numpy(), val.class_index.astype(int).to_numpy()
        probes, stats = {}, {}
        for r in REGIONS:
            probes[r] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=nsf.SETTINGS['probe_c'])).fit(f_train[r], y_train)
            s = probes[r].decision_function(f_val[r])
            stats[r] = dict(val_mean=float(s.mean()), val_std=float(s.std()),
                            ffpp_val_auc=float(shift.auc(s[y_val == 0], s[y_val == 1])))
            print(f'{args.model} {r}: probe FF++ val AUC {stats[r]["ffpp_val_auc"]:.3f}', flush=True)
        with open(out/'probes.pkl', 'wb') as f:
            pickle.dump(probes, f)
        pair_nsf.write_json(info_path, dict(identity=identity, stats=stats,
                                            pickle_sha256=hashlib.sha256((out/'probes.pkl').read_bytes()).hexdigest()))
    parse_labels(args, 'cdf_test', cdf, args.cdf_root)
    cache = diagonal.restore_cache(args.cache_root, args.archive_dir, 'cdf_test', v2.SmoothingCache.PARTS)
    for i, view in enumerate(VIEWS, 1):
        dest = out/f'cdf_test_{view}.csv'
        if dest.is_file():
            print(f'[{i}/{len(VIEWS)}] cached: {view}', flush=True)
            continue
        dataset = RegionViews(cdf, args.cdf_root, 'cdf_test', args, ids, transform, cache, None if view == 'normal' else view)
        feats = region_features(args, model, args.model, dataset, f'[{i}/{len(VIEWS)}] CDF test {view}')
        table = cdf[['crop_path']].copy()
        for r in REGIONS:
            table[r] = probes[r].decision_function(feats[r])
        temp = dest.with_suffix('.tmp.csv'); table.to_csv(temp, index=False); temp.replace(dest)
    del model
    if args.device == 'cuda':
        torch.cuda.empty_cache()


def view_scores(args, cdf):
    """{view: {region: z-scored B}} including the face_inner reference (identical in every view)."""
    stats = json.loads((args.output_root/'scores'/'probes.json').read_text())['stats']
    info = nsf_info(args)
    inner = pd.read_csv(args.runs_root/'nonskin_probe'/f'{args.model}_cdf_test_face_inner_scores.csv', dtype={'crop_path': str})
    if inner.crop_path.tolist() != cdf.crop_path.tolist() or not inner.scorer_pickle_sha256.eq(info['pickle_sha256']).all():
        raise ValueError('Cached face_inner scores do not match the CDF test frames/scorer')
    z_inner = (inner.score.to_numpy()-info['val_mean'])/info['val_std']
    out = {}
    for view in VIEWS:
        t = pd.read_csv(args.output_root/'scores'/f'cdf_test_{view}.csv', dtype={'crop_path': str})
        if t.crop_path.tolist() != cdf.crop_path.tolist():
            raise ValueError(f'Region score order differs: {view}')
        out[view] = {r: (t[r].to_numpy()-stats[r]['val_mean'])/stats[r]['val_std'] for r in REGIONS}
        out[view]['face_inner'] = z_inner
    return out


def metrics(y, normal, strong):
    real, fake = y == 0, y == 1
    scale = normal[real].std()
    row = dict(clean_auc=shift.auc(normal[real], normal[fake]), fp_cross_auc=shift.auc(strong[real], normal[fake]),
               fn_cross_auc=shift.auc(normal[real], strong[fake]),
               real_shift=float((strong[real]-normal[real]).mean()/scale), fake_shift=float((strong[fake]-normal[fake]).mean()/scale))
    row['worse_cross_auc'] = min(row['fp_cross_auc'], row['fn_cross_auc'])
    return row


def evaluate(args):
    """Fuse the saved pair predictions of --seed with every region score; checks face_inner against pair+NSF."""
    cdf = cdf_frame(args, status=False)
    y = cdf.class_index.astype(int).to_numpy()
    b = view_scores(args, cdf)
    w = nsf.SETTINGS['weight']
    source = args.evaluation_root/f'seed{args.seed}'/'cdf_test'
    reference = pd.read_csv(args.pair_nsf_root/f'seed{args.seed}'/'cdf_test'/'summary.csv')
    rows = []
    for filt in FILTERS:
        cal = json.loads((args.pair_nsf_root/'calibration'/f'seed{args.seed}_{filt}.json').read_text())
        sha = cal['identity']['source_checkpoint_sha256']
        a = {}
        for view in VIEWS:
            pred = pair_nsf.read_prediction(source/f'{args.model}_aug_pair@{filt}_{"normal" if view == "normal" else view+"_smooth_strong"}.csv')
            if pred.crop_path.tolist() != cdf.crop_path.tolist() or not pred.checkpoint_sha256.eq(sha).all():
                raise ValueError(f'Saved pair predictions do not match the pair+NSF calibration: seed{args.seed} {filt} {view}')
            a[view] = (pred.raw_margin.to_numpy()-cal['rank_val_mean'])/cal['rank_val_std']
        for ev in FILTERS:
            common = dict(model=args.model, seed=args.seed, train_filter=filt, filter=ev, held_out=ev != filt)
            rows.append(dict(common, region='none', **metrics(y, a['normal'], a[ev])))
            for r in ['face_inner', *REGIONS]:
                rows.append(dict(common, region=r, **metrics(y, (1-w)*a['normal']+w*b['normal'][r], (1-w)*a[ev]+w*b[ev][r])))
            ref = reference[reference.train_filter.eq(filt) & reference['filter'].eq(ev)].iloc[0]
            mine = rows[-len(REGIONS)-1]
            if abs(mine['fp_cross_auc']-ref.fp_cross_auc) > 1e-6 or abs(mine['fn_cross_auc']-ref.fn_cross_auc) > 1e-6:
                raise ValueError(f'face_inner fusion does not reproduce the saved pair+NSF result: {filt}->{ev}')
    target = args.output_root/f'seed{args.seed}'
    target.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(target/'summary.csv', index=False)
    print(f'seed{args.seed}: face_inner reproduces the saved pair+NSF numbers (16/16 cells)', flush=True)
    b_alone(args, cdf, y, b)
    report(args)


def b_alone(args, cdf, y, b):
    """B on its own (no fusion) per region and filter: how strong and how smoothing-sensitive each region is."""
    stats = json.loads((args.output_root/'scores'/'probes.json').read_text())['stats']
    rows = []
    for r in ['face_inner', *REGIONS]:
        for ev in FILTERS:
            rows.append(dict(region=r, filter=ev, ffpp_val_auc=stats[r]['ffpp_val_auc'] if r in stats else nsf_info(args)['ffpp_val_auc'],
                             **metrics(y, b['normal'][r], b[ev][r])))
    pd.DataFrame(rows).to_csv(args.output_root/'b_alone.csv', index=False)


def report(args):
    """Seed means over every finished seed: clean / seen 4 / unseen 12 / all 16 worse cross AUC, cells vs pair and face_inner."""
    files = sorted(args.output_root.glob('seed*/summary.csv'))
    rows = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    seeds = sorted(int(s) for s in rows.seed.unique())
    cell = rows.groupby(['region', 'train_filter', 'filter', 'held_out'], as_index=False)[['clean_auc', 'worse_cross_auc', 'real_shift', 'fake_shift']].mean()
    order = ['none', 'face_inner', *REGIONS]
    pivot = cell.pivot_table(index=['train_filter', 'filter'], columns='region', values='worse_cross_auc')
    head = []
    for r in order:
        g = cell[cell.region.eq(r)]
        head.append(dict(region=r, seeds=len(seeds), clean_auc=g.clean_auc.mean(), seen_worse=g[~g.held_out].worse_cross_auc.mean(),
                         unseen_worse=g[g.held_out].worse_cross_auc.mean(), all16_worse=g.worse_cross_auc.mean(),
                         unseen_real_shift=g[g.held_out].real_shift.mean(),
                         cells_over_pair=int((pivot[r] > pivot['none']).sum()) if r != 'none' else None,
                         cells_over_face_inner=int((pivot[r] > pivot['face_inner']).sum()) if r not in ['none', 'face_inner'] else None))
    head = pd.DataFrame(head)
    names = {'none': 'pair (B 없음)', 'face_inner': '눈·입 (제안)', 'full': '이미지 전체', 'face': '얼굴 전체', 'random': '랜덤 패치'}
    head.insert(1, 'name', head.region.map(names))
    head.to_csv(args.output_root/'region_headline.csv', index=False, encoding='utf-8-sig')
    cell.to_csv(args.output_root/'region_seed_means.csv', index=False)
    print(f'{args.model} | seeds {seeds} | worse cross AUC at strong smoothing', flush=True)
    print(head.round(4).to_string(index=False), flush=True)
    alone = args.output_root/'b_alone.csv'
    if alone.is_file():
        t = pd.read_csv(alone).groupby('region', sort=False)[['ffpp_val_auc', 'clean_auc', 'worse_cross_auc', 'real_shift', 'fake_shift']].mean()
        print('B alone (mean over 4 filters):', flush=True)
        print(t.round(4).to_string(), flush=True)
    print('Results:', args.output_root, flush=True)
    return head



def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=['prepare', 'scores', 'evaluate', 'report'])
    p.add_argument('--model', required=True, choices=['xception', 'clip_b16', 'dinov2_b14'])
    for key in ['runs-root', 'evaluation-root', 'pair-nsf-root', 'output-root', 'baseline-runs-root', 'source-root']:
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--cdf-root', type=Path, default=Path('/content/celebdf_v2_face_crops16_v1'))
    p.add_argument('--archive-dir', type=Path)
    p.add_argument('--drive-data', type=Path)
    p.add_argument('--cache-root', type=Path, default=Path('/content/beauty_smoothing_v2_cache'))
    p.add_argument('--work-dir', type=Path, default=Path('/content/nsf_region_work'))
    for key, default in [('seed', 42), ('batch-size', 32), ('workers', 2), ('parse-batch', 16)]:
        p.add_argument('--'+key, type=int, default=default)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    a = p.parse_args(argv)
    for root in [a.evaluation_root, a.pair_nsf_root, a.runs_root]:
        if a.output_root.resolve() == root.resolve():
            p.error('Use a separate output folder; original results must remain unchanged')
    if a.action == 'prepare' and a.drive_data is None:
        p.error('--drive-data required for prepare')
    if a.action == 'scores' and a.archive_dir is None:
        p.error('--archive-dir required for scores')
    return a


if __name__ == '__main__':
    args = parse_args()
    {'prepare': prepare, 'scores': scores, 'evaluate': evaluate, 'report': report}[args.action](args)
