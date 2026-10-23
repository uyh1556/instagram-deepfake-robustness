"""Precompute the v2 smoothing cache once and keep it as one Drive tar per (dataset, part).

Parts: `mask` (face-parsing skin mask, 8-bit PNG, GPU) and one part per filter (full unblended output, lossless
PNG, CPU workers). Only roi_status=detected frames are cached; fallback frames are never smoothed.
  cdf_test       Celeb-DF v2 official test (8,288 frames) -> v2 evaluation, cross-dataset
  ffpp_test      FF++ c23 4-method test frames             -> v2 evaluation, in-domain (no CDF domain gap)
  cdf_dev        CDF pool-v2 labelled NON-test frames      -> method development (never the official test);
                 --membership = pool_v2/membership_labelled.csv, video overlap with the official test refused
  ffpp_trainval  FF++ c23 4-method train+val frames        -> needed by v2 training (one filter part per run)
Every part directory holds identity.json (protocol, calibrated values, manifest/calibration/code hashes) and
complete.json. A part already complete locally is kept, a Drive tar is restored instead of recomputed, and an
interrupted run resumes frame by frame (atomic writes). Blending to weak/medium/strong happens at use time.
"""
import argparse
import hashlib
import json
import os
import subprocess
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import beauty_smoothing_v2 as v2

MANIFESTS = {'cdf_test': 'manifests/celebdf_v2_roi16_test_v1.csv',
             'ffpp_test': 'manifests/ffpp_c23_4methods_roi16_test_v1.csv',
             'ffpp_trainval': 'manifests/ffpp_c23_4methods_roi16_trainval_v1.csv',
             'cdf_dev': None}   # --membership
TEST_SETS = {'cdf_test', 'ffpp_test'}


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def manifest(args):
    if args.dataset == 'cdf_dev':
        if not getattr(args, 'membership', None):
            raise ValueError('cdf_dev needs --membership (pool v2 membership_labelled.csv)')
        return Path(args.membership)
    return args.data_root/MANIFESTS[args.dataset]


def dev_frame(path, data_root):
    """CDF pool-v2 labelled non-test frames; refuses any video of the official test."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if frame.crop_path.duplicated().any() or set(frame.class_index) != {'0', '1'}:
        raise ValueError(f'Unique two-class dev membership required: {path}')
    test = Path(data_root)/MANIFESTS['cdf_test']
    if test.is_file():
        overlap = set(frame.video_id) & set(pd.read_csv(test, dtype=str, keep_default_na=False).video_id)
        if overlap:
            raise ValueError(f'cdf_dev overlaps the official test: {sorted(overlap)[:3]}')
    return frame


def frames(args):
    if args.dataset == 'cdf_dev':
        return dev_frame(manifest(args), args.data_root).query('roi_status == "detected"').reset_index(drop=True)
    frame = pd.read_csv(manifest(args), dtype=str, keep_default_na=False)
    if args.dataset in TEST_SETS and not frame.split.eq('test').all():
        raise ValueError(f'{args.dataset} manifest must contain test frames only')
    if args.dataset == 'ffpp_trainval' and frame.split.eq('test').any():
        raise ValueError('ffpp_trainval manifest must not contain test frames')
    return frame[frame.roi_status.eq('detected')].reset_index(drop=True)


def identity(args):
    return dict(protocol=v2.PROTOCOL, dataset=args.dataset, values=v2.calibrated_values(args.calibration),
                color_sigma=v2.COLOR_SIGMA, parser=v2.PARSER, opencv=cv2.__version__,
                manifest_sha256=digest(manifest(args)),
                calibration_sha256=digest(args.calibration), protocol_code_sha256=digest(Path(v2.__file__)))


def restore(args, cache, part, ident):
    archive = args.archive_dir/f'{args.dataset}_{part}.tar'
    if (cache.root/part/'complete.json').is_file() or not archive.is_file():
        return
    print('Restoring', archive, flush=True)
    cache.root.mkdir(parents=True, exist_ok=True)
    subprocess.run(['tar', '-xf', str(archive), '-C', str(cache.root)], check=True)
    if json.loads((cache.root/part/'identity.json').read_text()) != ident:
        raise ValueError(f'Archived {part} provenance differs; use a new --archive-dir')


def masks(args, cache, rows):
    parser = v2.SkinParser(args.device)
    todo = [p for p in rows if not cache.path('mask', p).is_file()]
    print(f'mask: {len(rows)-len(todo)}/{len(rows)} ready; parsing {len(todo)} on {parser.device}', flush=True)
    empty = 0
    for start in range(0, len(todo), args.parse_batch):
        batch = todo[start:start+args.parse_batch]
        images = []
        for p in batch:
            with Image.open(args.data_root/p) as opened:
                images.append(opened.convert('RGB'))
        for p, mask in zip(batch, parser.masks(images)):
            empty += int(mask.max() == 0)
            v2.save_png(cache.path('mask', p), v2.mask_to_image(mask))
        done = start+len(batch)
        if done % (args.parse_batch*50) < args.parse_batch or done == len(todo):
            print(f'  parsed {done}/{len(todo)} (no skin so far: {empty})', flush=True)


def _fill(job):
    data_root, root, part, value, relative = job
    dest = Path(root)/part/relative
    if not dest.is_file():
        with Image.open(Path(data_root)/relative) as opened:
            array = np.asarray(opened.convert('RGB'))
        v2.save_png(dest, Image.fromarray(v2.full(part, array, value)))
    return relative


def filtered(args, cache, rows, part, value):
    todo = [p for p in rows if not cache.path(part, p).is_file()]
    print(f'{part}: {len(rows)-len(todo)}/{len(rows)} ready; filtering {len(todo)} with {args.workers} CPU workers', flush=True)
    jobs = [(str(args.data_root), str(cache.root), part, value, p) for p in todo]
    pool = Pool(args.workers) if args.workers > 1 else None
    done = pool.imap_unordered(_fill, jobs, chunksize=16) if pool else map(_fill, jobs)
    for k, _ in enumerate(done, 1):
        if k % 2000 == 0 or k == len(jobs):
            print(f'  {part} {k}/{len(jobs)}', flush=True)
    if pool:
        pool.close(); pool.join()


def pack(args, cache, part):
    archive = args.archive_dir/f'{args.dataset}_{part}.tar'
    if archive.is_file():
        return
    archive.parent.mkdir(parents=True, exist_ok=True)
    temp = archive.with_suffix(f'.partial{os.getpid()}.tar')
    print('Packing', archive, flush=True)
    subprocess.run(['tar', '-cf', str(temp), '-C', str(cache.root), part], check=True)
    os.replace(temp, archive)


def build(args):
    ident = identity(args)
    cache = v2.SmoothingCache(args.cache_root, args.dataset, ident)
    rows = frames(args).crop_path.tolist()
    print(f'{args.dataset}: {len(rows)} detected frames | parts {args.parts}', flush=True)
    for part in args.parts:
        restore(args, cache, part, ident)
        part_dir = cache.root/part
        if not (part_dir/'complete.json').is_file():
            part_dir.mkdir(parents=True, exist_ok=True)
            saved = part_dir/'identity.json'
            if saved.is_file() and json.loads(saved.read_text()) != ident:
                raise ValueError(f'Local {part} cache provenance differs: {part_dir}; delete it or use a new --cache-root')
            saved.write_text(json.dumps(ident, sort_keys=True, indent=1)+'\n')
            if part == 'mask':
                masks(args, cache, rows)
            else:
                filtered(args, cache, rows, part, ident['values'][part])
            missing = [p for p in rows if not cache.path(part, p).is_file()]
            if missing:
                raise RuntimeError(f'{part}: {len(missing)} frames missing after the pass')
            (part_dir/'complete.json').write_text(json.dumps(dict(frames=len(rows)))+'\n')
        pack(args, cache, part)
        print(f'{part}: ready ({len(rows)} frames)', flush=True)
    cache.check(args.parts)
    print('Cache ready:', cache.root, flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', choices=list(MANIFESTS), required=True)
    for field in ['data-root', 'calibration', 'archive-dir']:
        p.add_argument('--'+field, type=Path, required=True)
    p.add_argument('--cache-root', type=Path, default=Path('/content/beauty_smoothing_v2_cache'))
    p.add_argument('--membership', type=Path, help='cdf_dev only: pool v2 membership_labelled.csv')
    p.add_argument('--parts', nargs='+', choices=v2.SmoothingCache.PARTS, default=v2.SmoothingCache.PARTS)
    p.add_argument('--workers', type=int, default=os.cpu_count() or 2)
    p.add_argument('--parse-batch', type=int, default=32)
    p.add_argument('--device')
    return p.parse_args(argv)


if __name__ == '__main__':
    build(parse_args())
