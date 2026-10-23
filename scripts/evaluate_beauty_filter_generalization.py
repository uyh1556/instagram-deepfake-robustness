"""Do the source-ablation models stay robust to beauty smoothing filters they were NOT trained on?

Inference only. The five source-ablation checkpoints of one model (baseline, clean_only, aug_single,
aug_pair, aug_pair_rank; all trained with bilateral pairs only) are scored on the CDF official test with
three held-out skin-smoothing filters (weighted median, fast global smoother, domain transform) at the
calibrated strength (calibrate_beauty_filters.py, FF++ train only) and the same mask/blend levels
(weak/medium/strong = .25/.5/.75). Original and bilateral predictions are REUSED from the ablation
evaluation (checkpoint hashes verified). Each full filter output is computed once per frame into a
lossless local cache shared by every checkpoint and strength (optionally packed to / restored from one
tar on Drive so parallel model notebooks do not repeat it). No training, no selection; test labels
only score the metrics. Per (checkpoint, filter): fp_gap / fn_gap, skew, fp/fn flips,
paired shifts, plus a video bootstrap of each condition's gaps minus aug_pair_rank's.
"""
import argparse
import json
import os
import shutil
import subprocess
import tempfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

import analyze_smoothing_threshold_shift as shift
import beauty_smoothing_filters as beauty
import ffpp_c23_smoothing_source_ablation as ablation

engine, base, evaluation = ablation.engine, ablation.base, ablation.evaluation
ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'beauty_filter_generalization_v1'
LEVELS = {'smooth_weak': .25, 'smooth_medium': .5, 'smooth_strong': .75}
LABELS = ['baseline', *ablation.STRATEGIES]


class FilterCache:
    """Full (unblended) held-out filter output per frame, lossless PNG, keyed by filter/crop_path."""
    def __init__(self, root, identity):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root/'identity.json'
        if path.is_file() and json.loads(path.read_text()) != identity:
            raise ValueError(f'Filter cache provenance differs: {self.root}')
        path.write_text(json.dumps(identity, sort_keys=True, indent=1)+'\n')
        self.values = identity['values']

    def path(self, name, relative):
        relative = Path(relative)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Cache requires a relative crop path')
        return self.root/name/relative

    def full(self, image, name, relative):
        dest = self.path(name, relative)
        if dest.is_file():
            with Image.open(dest) as opened:
                if opened.size != image.size:
                    raise ValueError(f'Cached size differs: {dest}')
                return np.asarray(opened.convert('RGB'))
        filtered = beauty.full(name, np.asarray(image), self.values[name])
        dest.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temp = tempfile.mkstemp(prefix=dest.name+'.', suffix='.tmp.png', dir=dest.parent)
        os.close(descriptor)
        Image.fromarray(filtered).save(temp, compress_level=1)
        os.replace(temp, dest)
        return filtered


class FilterViews(Dataset):
    def __init__(self, frame, args, transform, cache, name, strength):
        self.rows, self.args, self.transform = frame.to_dict('records'), args, transform
        self.cache, self.name, self.strength = cache, name, strength

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.args.target_root/row['crop_path']) as opened:
            image = opened.convert('RGB')
        if row['roi_status'] == 'detected':  # fallback frames stay unchanged, exactly like the bilateral views
            image = beauty.pixel_cache.blend(image, self.cache.full(image, self.name, row['crop_path']), self.strength)
        return self.transform(image), index


def calibration_values(path):
    data = json.loads(path.read_text())
    if data.get('protocol') != 'beauty_filter_calibration_v1' or set(data['filters']) != set(beauty.FILTERS):
        raise ValueError(f'Unexpected calibration file: {path}')
    return {name: float(data['filters'][name]['value']) for name in beauty.FILTERS}


def build_cache(args, frame):
    values = calibration_values(args.calibration)
    import cv2
    identity = dict(protocol=PROTOCOL, values=values, calibration_sha256=base.common.digest(args.calibration),
                    manifest_sha256=base.common.digest(args.target_root/'manifests/celebdf_v2_roi16_test_v1.csv'),
                    opencv=cv2.__version__, filters_code_sha256=base.common.digest(Path(beauty.__file__)))
    return FilterCache(args.filter_cache, identity)


def _fill(job):
    root, identity, target_root, crop_path = job
    cache = FilterCache.__new__(FilterCache)
    cache.root, cache.values = Path(root), identity['values']
    with Image.open(Path(target_root)/crop_path) as opened:
        image = opened.convert('RGB')
    for name in beauty.FILTERS:
        cache.full(image, name, crop_path)
    return crop_path


def prepare(args):
    frame, _, _ = test_inputs(args)
    cache = build_cache(args, frame)
    identity = json.loads((cache.root/'identity.json').read_text())
    if args.cache_archive and args.cache_archive.is_file() and not (cache.root/'complete.json').is_file():
        print('Restoring filter cache from', args.cache_archive, flush=True)
        subprocess.run(['tar', '-xf', str(args.cache_archive), '-C', str(cache.root.parent)], check=True)
        if json.loads((cache.root/'identity.json').read_text()) != identity:
            raise ValueError('Archived filter cache provenance differs; delete it or use a new archive path')
    detected = frame[frame.roi_status.eq('detected')].crop_path.tolist()
    missing = [p for p in detected if not all(cache.path(n, p).is_file() for n in beauty.FILTERS)]
    print(f'Filter cache: {len(detected)-len(missing)}/{len(detected)} detected frames ready; computing {len(missing)} '
          f'with {args.cache_workers} CPU workers', flush=True)
    if missing:
        jobs = [(str(cache.root), identity, str(args.target_root), p) for p in missing]
        workers = Pool(args.cache_workers) if args.cache_workers > 1 else None
        done = workers.imap_unordered(_fill, jobs, chunksize=8) if workers else map(_fill, jobs)
        for k, _ in enumerate(done, 1):
            if k % 500 == 0 or k == len(jobs):
                print(f'  filtered {k}/{len(jobs)}', flush=True)
        if workers:
            workers.close(); workers.join()
    (cache.root/'complete.json').write_text(json.dumps(dict(frames=len(detected)))+'\n')
    if args.cache_archive and not args.cache_archive.is_file():
        args.cache_archive.parent.mkdir(parents=True, exist_ok=True)
        temp = args.cache_archive.with_suffix(f'.partial{os.getpid()}.tar')  # parallel notebooks never share a temp
        print('Packing filter cache to', args.cache_archive, flush=True)
        subprocess.run(['tar', '-cf', str(temp), '-C', str(cache.root.parent), cache.root.name], check=True)
        os.replace(temp, args.cache_archive)
    print('Filter cache ready:', cache.root, flush=True)


def test_inputs(args):
    ea = argparse.Namespace(data_root=args.target_root, manifest=args.target_root/'manifests/celebdf_v2_roi16_test_v1.csv',
        protocol=args.protocol, cache_root=args.target_root, output_root=args.output_root, workers=args.workers)
    return evaluation.inputs(ea)


def reused(args, name, label, condition, frame, checkpoint_hash):
    path = args.ablation_output_root/f'{name}_{label}_{condition}.csv'
    if not path.is_file():
        raise FileNotFoundError(f'Run the source-ablation evaluation first: {path}')
    saved = pd.read_csv(path, dtype={'crop_path': str})
    if saved.crop_path.tolist() != frame.crop_path.tolist() or saved.checkpoint_sha256.iloc[0] != checkpoint_hash:
        raise ValueError(f'Ablation predictions do not match this checkpoint: {path}')
    return saved.raw_margin.to_numpy()


def checkpoints(args, name):
    initial, initial_hash = evaluation.checkpoint(argparse.Namespace(runs_root=args.baseline_runs_root, seed=args.seed), name)
    states = [('baseline', initial['model_state_dict'], initial_hash)]
    for strategy in ablation.STRATEGIES:
        run = args.runs_root/ablation.run_name(name, strategy, args.seed)
        state = torch.load(run/'best.pt', map_location='cpu', weights_only=True)
        if state['config']['initial_checkpoint_sha256'] != initial_hash:
            raise ValueError(f'Checkpoint does not start from this baseline: {run}')
        states.append((strategy, state['model_state_dict'], base.common.digest(run/'best.pt')))
    return states


def evaluate(args):
    frame, _, provenance = test_inputs(args)
    cache = build_cache(args, frame)
    args.output_root.mkdir(parents=True, exist_ok=True)
    labels = frame.class_index.astype(int).to_numpy()
    eval_args = argparse.Namespace(**{**vars(args), 'batch_size': args.eval_batch_size})
    rows, boots = [], []
    for name in args.models:
        banks = {}
        for label, weights, checkpoint_hash in checkpoints(args, name):
            model = None
            banks[label] = {'bilateral': {c: reused(args, name, label, c, frame, checkpoint_hash) for c in ablation.CONDITIONS}}
            normal = banks[label]['bilateral']['normal']
            for filt in beauty.FILTERS:
                margins = {'normal': normal}
                for condition, strength in LEVELS.items():
                    pred = args.output_root/f'{name}_{label}_{filt}_{condition}.csv'
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
                        _, raw = engine.infer(eval_args, model, frame, FilterViews(frame, args, transform, cache, filt, strength),
                                              f'{name} {label} {filt} {condition}')
                        saved = frame[['crop_path', 'video_id', 'class_index']].copy()
                        saved['raw_margin'], saved['checkpoint_sha256'] = raw, checkpoint_hash
                        temp = pred.with_suffix('.tmp.csv'); saved.to_csv(temp, index=False); temp.replace(pred)
                    margins[condition] = saved.raw_margin.to_numpy()
                banks[label][filt] = margins
            for filt, margins in banks[label].items():
                measured = shift.measure(labels, margins)
                rows.append(dict(model=name, strategy=label, filter=filt, held_out=filt != 'bilateral',
                                 **measured))
            del model
            if args.device == 'cuda':
                torch.cuda.empty_cache()
        if args.bootstrap:
            for filt in ['bilateral', *beauty.FILTERS]:
                boots += ablation.bootstrap_rows(frame, labels, {k: v[filt] for k, v in banks.items()}, 'aug_pair_rank', args,
                                                 dict(model=name, filter=filt))
    table = pd.DataFrame(rows)
    table.to_csv(args.output_root/'summary.csv', index=False)
    pd.DataFrame(boots).to_csv(args.output_root/'bootstrap_gaps_vs_aug_pair_rank.csv', index=False)
    base.common.json_write(args.output_root/'protocol.json', dict(protocol=PROTOCOL, manifest_sha256=provenance['manifest_sha256'],
        calibration=calibration_values(args.calibration), levels=LEVELS, trained_with='bilateral pairs only',
        reused_from=str(args.ablation_output_root), test_used_for_selection=False))
    with pd.option_context('display.width', 250, 'display.max_columns', 20):
        for metric, title in [('fp_gap', 'Real-side gap (clean AUC - AUC(strong real vs clean fake))'),
                              ('fn_gap', 'Fake-side gap (clean AUC - AUC(clean real vs strong fake))')]:
            print(title, 'by evaluation filter:')
            print(table.pivot_table(index=['model', 'strategy'], columns='filter', values=metric, sort=False).round(4).to_string())
        if boots:
            print(pd.DataFrame(boots).round(4).to_string(index=False))
    print('Saved:', args.output_root, flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=['prepare', 'evaluate'])
    for field in ['target-root', 'baseline-runs-root', 'runs-root', 'ablation-output-root', 'calibration', 'output-root']:
        p.add_argument('--'+field, type=Path, required=True)
    p.add_argument('--filter-cache', type=Path, default=Path('/content/beauty_filter_cache_v1'))
    p.add_argument('--cache-archive', type=Path)
    p.add_argument('--protocol', type=Path, default=ROOT/'configs/ffpp_c23_celebdf_smoothing_v1.json')
    p.add_argument('--models', nargs='+', choices=list(base.PREPROCESS), default=['dinov2_b14'])
    for field, default in [('seed', 42), ('eval-batch-size', 64), ('workers', 2), ('cache-workers', os.cpu_count() or 2), ('bootstrap', 1000)]:
        p.add_argument('--'+field, type=int, default=default)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    a = p.parse_args(argv)
    if a.action == 'evaluate' and a.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime')
    return a


if __name__ == '__main__':
    arguments = parse_args()
    {'prepare': prepare, 'evaluate': evaluate}[arguments.action](arguments)
