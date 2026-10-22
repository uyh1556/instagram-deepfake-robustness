"""Match each held-out filter's strength to the training bilateral filter on FF++ TRAIN frames only.

Target: median skin-mask mean absolute RGB change (0-255) of the full (unblended) training bilateral
filter, i.e. how much it edits the skin. (A high-frequency-energy target was tried first: on FF++ c23 it is
dominated by compression/structure edges, ~0.04 for the bilateral, and not monotone for the weighted median;
it is still reported per filter as a descriptive measure.) For each held-out filter its one PARAMETER (beauty_smoothing_filters.PARAMETER) is searched on a log grid, refined by
log-linear interpolation between the bracketing grid points and re-measured. Blend strengths
(weak/medium/strong = 0.25/0.5/0.75 of the full filter inside the same mask) are then shared, so the
held-out views differ from the bilateral views only by the filter algorithm. CPU only; no Celeb-DF.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import beauty_smoothing_filters as beauty

ROOT = Path(__file__).resolve().parents[1]


def sample(args):
    frame = pd.read_csv(args.source_root/'manifests/ffpp_c23_4methods_roi16_trainval_v1.csv', dtype=str, keep_default_na=False)
    frame = frame[frame.split.eq('train') & frame.roi_status.eq('detected')]
    half = args.images//2
    parts = [frame[frame.class_index.eq(c)].sample(n=min(half, frame.class_index.eq(c).sum()), random_state=args.seed) for c in ['0', '1']]
    return pd.concat(parts).sort_values('crop_path').reset_index(drop=True)


def load(args, rows):
    out = []
    for path in rows.crop_path:
        with Image.open(args.source_root/path) as opened:
            image = opened.convert('RGB')
        out.append((image, np.asarray(image), beauty.mask(image)))
    return out


def changes(images, name, value):
    return np.array([beauty.skin_change(array, beauty.full(name, array, value), mask) for _, array, mask in images])


def hf(images, name, value):
    return float(np.median([beauty.hf_reduction(array, beauty.full(name, array, value), mask) for _, array, mask in images]))


def summary(values):
    return dict(median=float(np.median(values)), q25=float(np.quantile(values, .25)), q75=float(np.quantile(values, .75)))


def calibrate(images, name, target, grid=None):
    grid = beauty.GRIDS[beauty.PARAMETER[name]] if grid is None else grid
    medians = [float(np.median(changes(images, name, f))) for f in grid]
    order = np.argsort(medians)
    xs, ys = np.log(np.asarray(grid)[order]), np.asarray(medians)[order]
    if not ys[0] <= target <= ys[-1]:
        raise ValueError(f'{name}: bilateral target {target:.4f} outside grid range [{ys[0]:.4f}, {ys[-1]:.4f}]')
    value = float(np.exp(np.interp(target, ys, xs)))
    final = changes(images, name, value)
    return dict(parameter=beauty.PARAMETER[name], value=value, grid=[dict(value=float(f), median_change=m) for f, m in zip(grid, medians)],
                hf_reduction_median=hf(images, name, value), **summary(final))


def preview(images, settings, chosen, path, count=4):
    tiles = []
    for image, array, _ in images[:count]:
        row = [image, beauty.pixel_cache.blend(image, beauty.bilateral_full(image, settings), .75)]
        row += [beauty.smooth(image, name, chosen[name], .75) for name in beauty.FILTERS]
        size = (160, 160)
        tiles.append(np.concatenate([np.asarray(t.resize(size, Image.BICUBIC)) for t in row], axis=1))
    Image.fromarray(np.concatenate(tiles, axis=0)).save(path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--protocol', type=Path, default=ROOT/'configs/ffpp_c23_celebdf_smoothing_v1.json')
    p.add_argument('--images', type=int, default=300)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    settings = json.loads(args.protocol.read_text())['bilateral']
    rows = sample(args)
    images = load(args, rows)
    filtered = [beauty.bilateral_full(i, settings) for i, _, _ in images]
    bilateral = np.array([beauty.skin_change(a, f, m) for (_, a, m), f in zip(images, filtered)])
    bilateral_hf = float(np.median([beauty.hf_reduction(a, f, m) for (_, a, m), f in zip(images, filtered)]))
    target = float(np.median(bilateral))
    print(f'{len(images)} FF++ train frames | bilateral full-strength skin change median={target:.3f} (HF reduction {bilateral_hf:.3f})', flush=True)
    result = dict(protocol='beauty_filter_calibration_v1', images=len(images), seed=args.seed, sigma_color=beauty.SIGMA_COLOR,
                  spatial_fraction=beauty.SPATIAL_FRACTION,
                  hf_sigma_fraction=beauty.HF_SIGMA_FRACTION, target='median skin-mask mean |RGB change| of the full filter', blend_strengths={'smooth_weak': .25, 'smooth_medium': .5, 'smooth_strong': .75},
                  bilateral=dict(settings=settings, hf_reduction_median=bilateral_hf, **summary(bilateral)), filters={}, data='FF++ train frames only, roi_status=detected')
    for name in beauty.FILTERS:
        result['filters'][name] = calibrate(images, name, target)
        r = result['filters'][name]
        print(f"{name}: {r['parameter']}={r['value']:.5g} skin change median={r['median']:.3f} IQR=[{r['q25']:.3f}, {r['q75']:.3f}] "
              f"(HF reduction {r['hf_reduction_median']:.3f})", flush=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root/'calibration.json').write_text(json.dumps(result, indent=1)+'\n')
    preview(images, settings, {k: v['value'] for k, v in result['filters'].items()}, args.output_root/'preview_strong.png')
    rows.to_csv(args.output_root/'calibration_frames.csv', index=False)
    print('Saved:', args.output_root, '| preview columns: original, bilateral, weighted_median, fast_global, domain_transform (all strong 0.75)', flush=True)


if __name__ == '__main__':
    main()
