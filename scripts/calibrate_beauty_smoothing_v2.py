"""Fix the v2 smoothing strengths on FF++ TRAIN frames only (no Celeb-DF).

The anchor (bilateral, visually approved reach) defines the target: its median skin change (mask-weighted mean
|RGB change| of the full filter, 0-255) inside the face-parsing skin mask. Each other filter keeps its fixed colour
sigma and only its spatial reach is searched on a log grid around the visually approved value, refined by
log-linear interpolation and re-measured. Output: calibration.json (values used by the cache) and a preview of
the strong (.75) views of all four filters.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

import beauty_smoothing_v2 as v2
import calibrate_beauty_filters as v1_calibration


def load(args, rows, parser):
    images = []
    for path in rows.crop_path:
        with Image.open(args.source_root/path) as opened:
            images.append(opened.convert('RGB'))
    masks = []
    print(f'Parsing {len(images)} frames on {parser.device}', flush=True)
    for start in range(0, len(images), args.parse_batch):
        masks += parser.masks(images[start:start+args.parse_batch])
        print(f'  parsed {len(masks)}/{len(images)}', flush=True)
    return [(image, np.asarray(image), mask) for image, mask in zip(images, masks)]


def changes(images, name, value):
    return np.array([v2.skin_change(array, v2.full(name, array, value), mask) for _, array, mask in images])


def summary(values):
    return dict(median=float(np.median(values)), q25=float(np.quantile(values, .25)), q75=float(np.quantile(values, .75)))


def calibrate(images, name, target, grid):
    medians = []
    for v in grid:
        medians.append(float(np.median(changes(images, name, v))))
        print(f'  {name} reach {v:.5g}: skin change median {medians[-1]:.3f}', flush=True)
    order = np.argsort(medians)
    xs, ys = np.log(np.asarray(grid)[order]), np.asarray(medians)[order]
    if not ys[0] <= target <= ys[-1]:
        raise ValueError(f'{name}: anchor target {target:.4f} outside grid range [{ys[0]:.4f}, {ys[-1]:.4f}]')
    value = float(np.exp(np.interp(target, ys, xs)))
    return dict(value=value, grid=[dict(value=float(v), median_change=m) for v, m in zip(grid, medians)],
                **summary(changes(images, name, value)))


def preview(images, values, path, count=4, size=200):
    tiles = []
    for image, array, mask in images[:count]:
        row = [image]+[v2.blend(image, v2.full(n, array, values[n]), mask, v2.LEVELS['smooth_strong']) for n in v2.FILTERS]
        tiles.append(np.concatenate([np.asarray(t.resize((size, size), Image.BICUBIC)) for t in row], axis=1))
    Image.fromarray(np.concatenate(tiles, axis=0)).save(path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--images', type=int, default=300)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--parse-batch', type=int, default=16)
    p.add_argument('--device')
    args = p.parse_args(argv)
    rows = v1_calibration.sample(args)            # same FF++ train detected sample rule as v1 (half real / half fake)
    parser = v2.SkinParser(args.device)
    images = [x for x in load(args, rows, parser) if x[2].sum() > 0]
    anchor = changes(images, v2.ANCHOR, v2.APPROVED[v2.ANCHOR])
    target = float(np.median(anchor))
    print(f'{len(images)}/{len(rows)} FF++ train frames with parsed skin | anchor {v2.ANCHOR} reach '
          f'{v2.APPROVED[v2.ANCHOR]} skin change median={target:.3f}', flush=True)
    result = dict(protocol=v2.PROTOCOL, images=len(images), sampled=len(rows), seed=args.seed, parser=v2.PARSER,
                  color_sigma=v2.COLOR_SIGMA, approved=v2.APPROVED, levels=v2.LEVELS, anchor=v2.ANCHOR,
                  target='median parsed-skin mean |RGB change| (0-255) of the full anchor filter',
                  target_value=target, data='FF++ train frames only, roi_status=detected', filters={})
    result['filters'][v2.ANCHOR] = dict(value=v2.APPROVED[v2.ANCHOR], **summary(anchor))
    for name in v2.FILTERS:
        if name == v2.ANCHOR:
            continue
        grid = np.geomspace(v2.APPROVED[name]/3, v2.APPROVED[name]*3, 9)
        result['filters'][name] = calibrate(images, name, target, grid)
        r = result['filters'][name]
        print(f"{name}: reach={r['value']:.5g} (approved {v2.APPROVED[name]}) skin change median={r['median']:.3f} "
              f"IQR=[{r['q25']:.3f}, {r['q75']:.3f}]", flush=True)
    result['values'] = {n: result['filters'][n]['value'] for n in v2.FILTERS}
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root/'calibration.json').write_text(json.dumps(result, indent=1)+'\n')
    rows.to_csv(args.output_root/'calibration_frames.csv', index=False)
    preview(images, result['values'], args.output_root/'preview_strong.png')
    print('Saved:', args.output_root, '| preview columns: original, '+', '.join(v2.FILTERS)+' (all strong .75)', flush=True)
    return result


if __name__ == '__main__':
    main()
