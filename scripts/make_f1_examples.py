"""F1 for the proposal deck: one real frame flipped to 'fake' by skin smoothing and one fake frame flipped to 'real'.

Uses only what the v2 evaluation already saved (per-frame raw margins of the FF++ baseline on Celeb-DF test, seed 42)
plus the v2 cdf_test cache for the images; no model is run. p(fake) = sigmoid(raw margin), threshold 0.5.
  real candidates  detected real frames that ALL three backbones score < 0.5 on the original and > 0.5 after
                   bilateral strong smoothing (false positive caused by smoothing)
  fake candidates  detected fake frames that Xception and DINO score > 0.5 on the original and < 0.5 after domain
                   transform strong smoothing (false negative; CLIP shown but not required)
Candidates are ranked by the smallest margin of the flip across the required backbones (clearest example first) and
one frame per video is kept. Outputs: candidates CSV, one preview PNG per candidate, and F1 for the chosen pair.
These are illustrative examples; the deck must also show the overall flip rates.
"""
import argparse
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import beauty_smoothing_v2 as v2

MODELS = [('clip_b16', 'CLIP'), ('dinov2_b14', 'DINO'), ('xception', 'Xception')]
CASES = {'real': dict(label=0, filter='bilateral', need=['clip_b16', 'dinov2_b14', 'xception'], flip='up'),
         'fake': dict(label=1, filter='domain_transform', need=['dinov2_b14', 'xception'], flip='down')}
FONT = Path('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
TEXT = {True: dict(orig='원본', smooth='보정본 ({f}, 강)', p='가짜 확률', real='진짜 영상', fake='가짜 영상', ex='예시 · Celeb-DF v2 test · FF++ 학습 Baseline'),
        False: dict(orig='Original', smooth='Smoothed ({f}, strong)', p='p(fake)', real='Real video', fake='Fake video', ex='Example · Celeb-DF v2 test · FF++-trained baseline')}
SHORT = {'bilateral': 'bilateral', 'domain_transform': 'DT'}


def sigmoid(x):
    return 1/(1+np.exp(-x))


def scores(args):
    """crop_path, class_index, video_id + p(fake) per backbone for normal and the two strong-smoothing conditions."""
    out = None
    for model, _ in MODELS:
        folder = args.eval_root/f'ffpp_c23_smoothing_v2_diagonal_{model}'/f'seed{args.seed}'/'cdf_test'
        for cond, stem in [('normal', 'normal'), ('bilateral', 'bilateral_smooth_strong'), ('domain_transform', 'domain_transform_smooth_strong')]:
            t = pd.read_csv(folder/f'{model}_baseline_{stem}.csv', dtype={'crop_path': str})
            col = t[['crop_path', 'video_id', 'class_index']].copy() if out is None else t[['crop_path']].copy()
            col[f'p_{model}_{cond}'] = sigmoid(t.raw_margin.to_numpy())
            out = col if out is None else out.merge(col, on='crop_path', validate='one_to_one')
    return out


def candidates(table, manifest, case, top):
    c = CASES[case]
    t = table.merge(manifest[['crop_path', 'roi_status']], on='crop_path')
    t = t[t.roi_status.eq('detected') & t.class_index.astype(int).eq(c['label'])].copy()
    margins = []
    for m in c['need']:
        before, after = t[f'p_{m}_normal'], t[f'p_{m}_{c["filter"]}']
        margins.append(np.minimum(.5-before, after-.5) if c['flip'] == 'up' else np.minimum(before-.5, .5-after))
    t['flip_margin'] = np.min(margins, axis=0)
    t = t[t.flip_margin > 0].sort_values('flip_margin', ascending=False).drop_duplicates('video_id')
    return t.head(top).reset_index(drop=True).assign(case=case, filter=c['filter'])


def restore(args, parts):
    cache = v2.SmoothingCache(args.cache_root, 'cdf_test')
    for part in parts:
        if not (cache.root/part/'complete.json').is_file():
            cache.root.mkdir(parents=True, exist_ok=True)
            print('Restoring', part, flush=True)
            subprocess.run(['tar', '-xf', str(args.archive_dir/f'cdf_test_{part}.tar'), '-C', str(cache.root)], check=True)
    cache.check(parts)
    return cache


def images(args, cache, row):
    with Image.open(args.cdf_root/row.crop_path) as opened:
        image = opened.convert('RGB')
    return image, cache.view(image, row.crop_path, row['filter'], v2.LEVELS['smooth_strong'])


def setup_font():
    if FONT.is_file():
        from matplotlib import font_manager
        font_manager.fontManager.addfont(str(FONT))
        plt.rcParams['font.family'] = 'NanumGothic'
    plt.rcParams['axes.unicode_minus'] = False
    return TEXT[FONT.is_file()]


def panel(fig, spec, image, smooth, row, text, title):
    """[original | smoothed] images above a grouped p(fake) bar chart for the three backbones."""
    sub = spec.subgridspec(2, 2, height_ratios=[3, 1.6], hspace=.25, wspace=.05)
    for k, (img, name) in enumerate([(image, text['orig']), (smooth, text['smooth'].format(f=SHORT[row['filter']]))]):
        a = fig.add_subplot(sub[0, k]); a.imshow(img); a.set_xticks([]); a.set_yticks([]); a.set_title(name, fontsize=10)
    a = fig.add_subplot(sub[1, :])
    x = np.arange(len(MODELS)); w = .36
    before = [row[f'p_{m}_normal'] for m, _ in MODELS]; after = [row[f'p_{m}_{row["filter"]}'] for m, _ in MODELS]
    b1 = a.bar(x-w/2, before, w, color='#9aa5b1', label=text['orig'])
    b2 = a.bar(x+w/2, after, w, color='#e4572e' if row.case == 'real' else '#3a7ca5', label=text['smooth'].format(f=SHORT[row['filter']]))
    for bars in (b1, b2):
        for bar in bars:
            a.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.02, f'{bar.get_height():.2f}', ha='center', fontsize=8.5)
    a.axhline(.5, ls='--', color='#333', lw=1)
    a.set_xticks(x); a.set_xticklabels([n for _, n in MODELS], fontsize=9.5); a.set_ylim(0, 1.12); a.set_ylabel(text['p'], fontsize=9)
    a.spines[['top', 'right']].set_visible(False); a.legend(fontsize=7.5, loc='upper left', frameon=False, ncol=2)
    fig.text(spec.get_position(fig).x0, spec.get_position(fig).y1+.01, title, fontsize=11, weight='bold')


def preview(args, cache, row, text, path):
    image, smooth = images(args, cache, row)
    fig = plt.figure(figsize=(5.2, 5.4))
    spec = fig.add_gridspec(1, 1, top=.9)[0]
    panel(fig, spec, image, smooth, row, text, f"{text[row.case]} #{row.name}  {Path(row.crop_path).parent.name}")
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white'); plt.close(fig)


def final(args, cache, real, fake, text):
    fig = plt.figure(figsize=(10.5, 5.6))
    gs = fig.add_gridspec(1, 2, wspace=.12, top=.88)
    for spec, row in [(gs[0], real), (gs[1], fake)]:
        image, smooth = images(args, cache, row)
        panel(fig, spec, image, smooth, row, text, text[row.case])
    fig.text(.99, .01, text['ex'], ha='right', fontsize=8, color='#666')
    path = args.output_root/'F1_examples.png'
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white'); plt.close(fig)
    return path


def run(args):
    text = setup_font()
    table = scores(args)
    manifest = pd.read_csv(args.cdf_root/'manifests/celebdf_v2_roi16_test_v1.csv', dtype=str, keep_default_na=False)
    cache = restore(args, ['mask', 'bilateral', 'domain_transform'])
    args.output_root.mkdir(parents=True, exist_ok=True)
    picks = {}
    for case in CASES:
        c = candidates(table, manifest, case, args.top)
        c.to_csv(args.output_root/f'F1_candidates_{case}.csv', index=False)
        print(f'{case}: {len(c)} candidates', flush=True)
        for i, row in c.iterrows():
            preview(args, cache, row, text, args.output_root/f'F1_candidate_{case}_{i}.png')
        picks[case] = c
    if len(picks['real']) and len(picks['fake']):
        path = final(args, cache, picks['real'].iloc[args.real_pick], picks['fake'].iloc[args.fake_pick], text)
        print('Saved:', path, flush=True)
    return picks


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for field in ['eval-root', 'cdf-root', 'archive-dir', 'output-root']:
        p.add_argument('--'+field, type=Path, required=True)
    p.add_argument('--cache-root', type=Path, default=Path('/content/beauty_smoothing_v2_cache'))
    for field, default in [('seed', 42), ('top', 8), ('real-pick', 0), ('fake-pick', 0)]:
        p.add_argument('--'+field, type=int, default=default)
    return p.parse_args(argv)


if __name__ == '__main__':
    run(parse_args())
