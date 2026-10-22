"""Does low-frequency image style explain the FF++ -> Celeb-DF gap? Inference-only diagnostic.

CDF official test frames (normal and strongly smoothed views) are resized to a fixed square, and
the centred low-frequency Fourier AMPLITUDE of each channel is replaced by a reference mean
amplitude (FDA, Yang & Soatto, CVPR 2020); phase and high frequencies stay untouched.
Variants per model/checkpoint:
  resize        resize to the square and back through the model transform only (control)
  ffpp_b<beta>  low-frequency amplitude from FF++ train frames (FF++ "style")
  cdf_b<beta>   same operation with the CDF test frames' OWN mean amplitude (controls for the
                averaging itself; uses test images without labels)
If ffpp_b raises CDF AUC well above resize and cdf_b, style is a large part of the domain gap and
target-style pairs are worth training; otherwise the gap lies in the manipulation traces.
No training, no checkpoint selection. CDF test labels only score the diagnostic.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import rankdata

import ffpp_c23_smoothing_delta_transfer as rank

engine, base, evaluation, common = rank.engine, rank.base, rank.evaluation, rank.common
ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'cdf_style_gap_diagnostic_v1'
CONDITIONS = ['normal', 'smooth_strong']
STRATEGIES = ['baseline', 'ce_rank']


def resized(image, size):
    return np.asarray(image.convert('RGB').resize((size, size), Image.BICUBIC), dtype=np.float64)


def amplitude(array):
    return np.abs(np.fft.fftshift(np.fft.fft2(array, axes=(0, 1)), axes=(0, 1)))


def mean_amplitude(paths, size):
    total = None
    for path in paths:
        with Image.open(path) as opened:
            value = amplitude(resized(opened, size))
        total = value if total is None else total+value
    return total/len(paths)


def swap_low_frequency(array, reference, beta):
    """FDA: replace the centred (2b+1)^2 amplitude block per channel, keep the phase."""
    spectrum = np.fft.fftshift(np.fft.fft2(array, axes=(0, 1)), axes=(0, 1))
    amp, phase = np.abs(spectrum), np.angle(spectrum)
    size = array.shape[0]
    b, c = int(np.floor(size*beta)), size//2
    amp[c-b:c+b+1, c-b:c+b+1] = reference[c-b:c+b+1, c-b:c+b+1]
    restored = np.fft.ifft2(np.fft.ifftshift(amp*np.exp(1j*phase), axes=(0, 1)), axes=(0, 1)).real
    return np.clip(np.rint(restored), 0, 255).astype(np.uint8)


class Styled:
    """Picklable PIL -> tensor transform: fixed resize, optional FDA swap, then the model transform."""
    def __init__(self, transform, size, reference=None, beta=0.):
        self.transform, self.size, self.reference, self.beta = transform, size, reference, beta

    def __call__(self, image):
        array = resized(image, self.size)
        styled = (np.clip(np.rint(array), 0, 255).astype(np.uint8) if self.reference is None
                  else swap_low_frequency(array, self.reference, self.beta))
        return self.transform(Image.fromarray(styled))


def auc(labels, scores):
    ranks = rankdata(scores)
    positive = labels == 1
    n1, n0 = positive.sum(), (~positive).sum()
    return float((ranks[positive].sum()-n1*(n1+1)/2)/(n1*n0))


def summarize(labels, margins):
    normal, strong = margins['normal'], margins['smooth_strong']
    real, fake = labels == 0, labels == 1
    return dict(normal_auc=auc(labels, normal), strong_auc=auc(labels, strong),
                strong_real_vs_normal_fake=auc(np.r_[np.zeros(real.sum()), np.ones(fake.sum())], np.r_[strong[real], normal[fake]]),
                fpr_normal=float((normal[real] >= 0).mean()), fpr_strong=float((strong[real] >= 0).mean()),
                fnr_normal=float((normal[fake] < 0).mean()))


def checkpoints(args, name):
    initial, initial_hash = evaluation.checkpoint(argparse.Namespace(runs_root=args.baseline_runs_root, seed=args.seed), name)
    states = {'baseline': (initial['model_state_dict'], initial_hash)}
    if 'ce_rank' in args.strategies:
        path = args.ce_rank_runs_root/rank.run_name(name, 'ce_rank', args.seed)/'best.pt'
        state = torch.load(path, map_location='cpu', weights_only=True)
        if state['config']['initial_checkpoint_sha256'] != initial_hash or state['config']['strategy'] != 'ce_rank':
            raise ValueError(f'CE+Rank checkpoint does not start from this baseline: {path}')
        states['ce_rank'] = (state['model_state_dict'], base.common.digest(path))
    return {k: v for k, v in states.items() if k in args.strategies}


def references(args, frame):
    folder = args.output_root/'references'
    folder.mkdir(parents=True, exist_ok=True)
    out = {}
    source = pd.read_csv(args.source_root/'manifests/ffpp_c23_4methods_roi16_trainval_v1.csv', dtype=str, keep_default_na=False)
    source = source[source.split.eq('train')]
    for key, root, pool in [('ffpp', args.source_root, source), ('cdf', args.target_root, frame)]:
        path = folder/f'{key}_amplitude_{args.size}_{args.reference_images}.npy'
        if not path.is_file():
            rows = pool.sample(n=min(args.reference_images, len(pool)), random_state=args.seed)
            print(f'Reference amplitude {key}: {len(rows)} frames', flush=True)
            np.save(path, mean_amplitude([root/p for p in rows.crop_path], args.size))
        out[key] = np.load(path)
    return out


def evaluate(args):
    ea = argparse.Namespace(data_root=args.target_root, manifest=args.target_root/'manifests/celebdf_v2_roi16_test_v1.csv',
                            protocol=args.protocol, cache_root=args.cache_root, output_root=args.output_root, workers=args.workers)
    frame, cfg, provenance = evaluation.inputs(ea)
    args.output_root.mkdir(parents=True, exist_ok=True)
    view_cache = args.cache_root if (args.cache_root/'config.json').is_file() and json.loads((args.cache_root/'config.json').read_text()) == provenance else None
    native_cache = common.pixel_cache.BilateralCache(args.evaluation_pixel_cache, cfg['bilateral'],
        dict(data=provenance, cache_code_sha256=base.common.digest(Path(common.pixel_cache.__file__)),
             storage='native uint8 RGB bilateral only; lossless PNG'))
    refs = references(args, frame)
    variants = [('resize', None, 0.)]+[(f'{k}_b{beta:g}', refs[k], beta) for beta in args.betas for k in ['ffpp', 'cdf']]
    labels = frame.class_index.astype(int).to_numpy()
    rows = []
    for name in args.models:
        for strategy, (weights, checkpoint_hash) in checkpoints(args, name).items():
            model = None
            for variant, reference, beta in variants:
                margins = {}
                for condition in CONDITIONS:
                    pred = args.output_root/f'{name}_{strategy}_{variant}_{condition}.csv'
                    if pred.is_file():
                        saved = pd.read_csv(pred, dtype={'crop_path': str})
                        if saved.crop_path.tolist() != frame.crop_path.tolist():
                            raise ValueError(f'Saved predictions differ: {pred}')
                        print('Reuse:', pred.name, flush=True)
                    else:
                        if model is None:
                            model, transform = base.build_model(name, pretrained=False)
                            model.load_state_dict(weights, strict=True)
                            model.to(args.device).eval()
                        styled = Styled(transform, args.size, reference, beta)
                        _, raw = engine.infer(args, model, frame, engine.TestViews(frame, args, styled, cfg, condition, native_cache, view_cache),
                                              f'{name} {strategy} {variant} {condition}')
                        saved = frame[['crop_path', 'class_index']].copy()
                        saved['raw_margin'] = raw
                        temp = pred.with_suffix('.tmp.csv'); saved.to_csv(temp, index=False); temp.replace(pred)
                    margins[condition] = saved.raw_margin.to_numpy()
                row = dict(model=name, strategy=strategy, variant=variant, beta=beta, checkpoint_sha256=checkpoint_hash, **summarize(labels, margins))
                rows.append(row)
                print(f"{name} {strategy} {variant}: normal AUC={row['normal_auc']:.4f} strong AUC={row['strong_auc']:.4f} "
                      f"FPR normal={row['fpr_normal']:.1%}", flush=True)
            del model
            if args.device == 'cuda':
                torch.cuda.empty_cache()
    table = pd.DataFrame(rows)
    control = table[table.variant.eq('resize')].set_index(['model', 'strategy'])
    keys = pd.MultiIndex.from_frame(table[['model', 'strategy']])
    for column in ['normal_auc', 'strong_auc', 'fpr_normal']:
        table[f'{column}_vs_resize'] = table[column].to_numpy()-control[column].reindex(keys).to_numpy()
    table.to_csv(args.output_root/'summary.csv', index=False)
    base.common.json_write(args.output_root/'protocol.json', dict(protocol=PROTOCOL, size=args.size, betas=args.betas,
        reference_images=args.reference_images, seed=args.seed, manifest_sha256=provenance['manifest_sha256'],
        ffpp_reference='random FF++ train frames, both classes', cdf_reference='random CDF official test frames, labels unused',
        used_for_selection=False))
    with pd.option_context('display.width', 250, 'display.max_columns', 20):
        print(table.drop(columns='checkpoint_sha256').round(4).to_string(index=False))
    print('Saved:', args.output_root/'summary.csv', flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for field in ['source-root', 'target-root', 'baseline-runs-root', 'ce-rank-runs-root', 'output-root']:
        p.add_argument('--'+field, type=Path, required=True)
    p.add_argument('--cache-root', type=Path, default=Path('/content/celebdf_official_smoothing_cache_v1'))
    p.add_argument('--evaluation-pixel-cache', type=Path, default=Path('/content/smoothing_ranking_cdf_bilateral_cache_v1'))
    p.add_argument('--protocol', type=Path, default=ROOT/'configs/ffpp_c23_celebdf_smoothing_v1.json')
    p.add_argument('--models', nargs='+', choices=list(base.PREPROCESS), default=['clip_b16', 'dinov2_b14', 'xception'])
    p.add_argument('--strategies', nargs='+', choices=STRATEGIES, default=STRATEGIES)
    p.add_argument('--betas', nargs='+', type=float, default=[.01, .03])
    for field, default in [('size', 256), ('reference-images', 2000), ('seed', 42), ('batch-size', 64), ('workers', 2)]:
        p.add_argument('--'+field, type=int, default=default)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    a = p.parse_args(argv)
    if not all(0 < b < .5 for b in a.betas) or a.size < 32 or a.reference_images < 1:
        p.error('betas must be in (0, 0.5); size >= 32')
    if a.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime')
    return a


if __name__ == '__main__':
    evaluate(parse_args())
