"""Score checkpoints on a test set under the v2 beauty-smoothing protocol (inference only).

--dataset cdf_test (Celeb-DF v2 official test, cross-dataset), ffpp_test (FF++ c23 4-method test, in-domain: the
smoothing shift without the CDF domain gap) or cdf_dev (CDF pool-v2 labelled non-test frames, for method development so
the official test is looked at only once, after the method is fixed).

For every v2 filter (bilateral, weighted median, fast global, domain transform) and strength (weak/medium/strong
= .25/.5/.75 inside the face-parsing skin mask), from the precomputed cdf_test cache (build_beauty_smoothing_v2_cache.py,
restored from its Drive tars) of that dataset. Currently the FF++ c23 baselines; the original (normal) condition is inferred once per
checkpoint. Per (checkpoint, filter): fp_gap / fn_gap / max|gap|, cross AUCs, skew, flips and paired shifts
(analyze_smoothing_threshold_shift.measure). Test labels only score; nothing is selected on the test.
"""
import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

import analyze_smoothing_threshold_shift as shift
import beauty_smoothing_v2 as v2
import build_beauty_smoothing_v2_cache as builder
import ffpp_c23_smoothing_source_ablation as ablation

engine, base, evaluation = ablation.engine, ablation.base, ablation.evaluation
ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = 'beauty_smoothing_v2_baseline'


class V2Views(Dataset):
    def __init__(self, frame, args, transform, cache, name, strength):
        self.rows, self.args, self.transform = frame.to_dict('records'), args, transform
        self.cache, self.name, self.strength = cache, name, strength

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.args.data_root/row['crop_path']) as opened:
            image = opened.convert('RGB')
        image = self.cache.view(image, row['crop_path'], self.name, self.strength, row['roi_status'] == 'detected')
        return self.transform(image), index


def ready_cache(args):
    cache = v2.SmoothingCache(args.cache_root, args.dataset)
    for part in cache.PARTS:
        archive = args.archive_dir/f'{args.dataset}_{part}.tar'
        if not (cache.root/part/'complete.json').is_file():
            if not archive.is_file():
                raise FileNotFoundError(f'Build the {args.dataset} v2 cache first: {archive}')
            print('Restoring', archive, flush=True)
            cache.root.mkdir(parents=True, exist_ok=True)
            subprocess.run(['tar', '-xf', str(archive), '-C', str(cache.root)], check=True)
    ident = cache.check()
    if ident['protocol'] != v2.PROTOCOL or ident['dataset'] != args.dataset:
        raise ValueError(f'Not a v2 {args.dataset} cache')
    return cache


def test_inputs(args):
    if args.dataset == 'cdf_test':   # official-test count/uniqueness checks of the v1 evaluator
        ea = argparse.Namespace(data_root=args.data_root, manifest=args.data_root/'manifests/celebdf_v2_roi16_test_v1.csv',
            protocol=args.protocol, cache_root=args.data_root, output_root=args.output_root, workers=args.workers)
        frame, _, provenance = evaluation.inputs(ea)
        return frame, provenance
    if args.dataset == 'cdf_dev':
        frame = builder.dev_frame(args.membership, args.data_root).reset_index(drop=True)
        return frame, dict(manifest_sha256=base.common.digest(args.membership))
    manifest = args.data_root/'manifests/ffpp_c23_4methods_roi16_test_v1.csv'
    frame = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    if not frame.split.eq('test').all() or frame.crop_path.duplicated().any() or set(frame.class_index) != {'0', '1'}:
        raise ValueError(f'FF++ test-only, unique, two-class manifest required: {manifest}')
    missing = [p for p in frame.crop_path if not (args.data_root/p).is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} FF++ test images missing, e.g. {missing[0]}')
    return frame.reset_index(drop=True), dict(manifest_sha256=base.common.digest(manifest))


def checkpoints(args, name):
    initial, initial_hash = evaluation.checkpoint(argparse.Namespace(runs_root=args.baseline_runs_root, seed=args.seed), name)
    return [('baseline', initial['model_state_dict'], initial_hash)]


def predict(args, frame, name, label, weights, checkpoint_hash, cache, filt, condition, strength, state):
    pred = args.output_root/(f'{name}_{label}_normal.csv' if filt is None else f'{name}_{label}_{filt}_{condition}.csv')
    if pred.is_file():
        saved = pd.read_csv(pred, dtype={'crop_path': str})
        if saved.crop_path.tolist() != frame.crop_path.tolist() or saved.checkpoint_sha256.iloc[0] != checkpoint_hash:
            raise ValueError(f'Saved predictions differ: {pred}')
        print('Reuse:', pred.name, flush=True)
        return saved.raw_margin.to_numpy()
    if state.get('model') is None:
        model, transform = base.build_model(name, pretrained=False)
        model.load_state_dict(weights, strict=True)
        state['model'], state['transform'] = model.to(args.device).eval(), transform
    eval_args = argparse.Namespace(**{**vars(args), 'batch_size': args.eval_batch_size})
    _, raw = engine.infer(eval_args, state['model'], frame, V2Views(frame, args, state['transform'], cache, filt, strength),
                          f'{name} {label} {filt} {condition}')
    saved = frame[['crop_path', 'video_id', 'class_index']].copy()
    saved['raw_margin'], saved['checkpoint_sha256'] = raw, checkpoint_hash
    temp = pred.with_suffix('.tmp.csv'); saved.to_csv(temp, index=False); temp.replace(pred)
    return raw


def evaluate(args):
    frame, provenance = test_inputs(args)
    cache = ready_cache(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    labels = frame.class_index.astype(int).to_numpy()
    rows = []
    for name in args.models:
        for label, weights, checkpoint_hash in checkpoints(args, name):
            state = {}
            normal = predict(args, frame, name, label, weights, checkpoint_hash, cache, None, 'normal', 0, state)
            for filt in v2.FILTERS:
                margins = {'normal': normal}
                for condition, strength in v2.LEVELS.items():
                    margins[condition] = predict(args, frame, name, label, weights, checkpoint_hash, cache, filt, condition, strength, state)
                measured = shift.measure(labels, margins)
                rows.append(dict(model=name, strategy=label, filter=filt, **measured,
                                 max_gap=max(abs(measured['fp_gap']), abs(measured['fn_gap']))))
            state.clear()
            if args.device == 'cuda':
                torch.cuda.empty_cache()
    table = pd.DataFrame(rows)
    table.to_csv(args.output_root/'summary.csv', index=False)
    base.common.json_write(args.output_root/'protocol.json', dict(protocol=v2.PROTOCOL, experiment=EXPERIMENT,
        dataset=args.dataset, manifest_sha256=provenance['manifest_sha256'], cache_identity=cache.identity, levels=v2.LEVELS,
        test_used_for_selection=False))
    with pd.option_context('display.width', 250, 'display.max_columns', 20):
        for metric in ['normal_auc', 'fp_gap', 'fn_gap', 'max_gap', 'fp_flip@normal5', 'fn_flip@normal5']:
            print(metric, 'by filter:')
            print(table.pivot_table(index=['model', 'strategy'], columns='filter', values=metric, sort=False)[v2.FILTERS].round(4).to_string())
    print('Saved:', args.output_root, flush=True)
    return table


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', choices=['cdf_test', 'ffpp_test', 'cdf_dev'], default='cdf_test')
    p.add_argument('--membership', type=Path, help='cdf_dev only: pool v2 membership_labelled.csv')
    for field in ['data-root', 'baseline-runs-root', 'archive-dir', 'output-root']:
        p.add_argument('--'+field, type=Path, required=True)
    p.add_argument('--cache-root', type=Path, default=Path('/content/beauty_smoothing_v2_cache'))
    p.add_argument('--protocol', type=Path, default=ROOT/'configs/ffpp_c23_celebdf_smoothing_v1.json')  # CDF test counts only
    p.add_argument('--models', nargs='+', choices=list(base.PREPROCESS), default=['dinov2_b14'])
    for field, default in [('seed', 42), ('eval-batch-size', 64), ('workers', 2)]:
        p.add_argument('--'+field, type=int, default=default)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    a = p.parse_args(argv)
    if a.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime')
    return a


if __name__ == '__main__':
    evaluate(parse_args())
