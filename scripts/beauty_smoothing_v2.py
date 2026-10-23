"""Beauty skin-smoothing protocol v2 (agreed 2026-10-23 after visual review on CDF/FF++ crops).

Replaces the v1 geometric-ellipse bilateral protocol for NEW experiments only (v1 code/results are kept).
  * Mask: face-parsing skin mask (SegFormer jonathandinu/face-parsing, CelebAMask-HQ labels skin+nose,
    pinned revision), eroded by 1% and feathered by a 1.2% Gaussian of the ROI side, so eyes, brows, lips,
    hair, ears, neck and background are never smoothed. A frame with no parsed skin is left unchanged.
  * Four edge-preserving filters, one per family (StyleRetoucher arXiv 2312.14389, Sec. 2.1, + bilateral).
    The colour/range sigma of each is FIXED low so weak facial shading edges (nose, eye sockets, jaw) survive;
    the strength is set ONLY by the spatial reach (fraction of the ROI side), calibrated so every filter makes
    the same median skin change as the anchor (bilateral) on FF++ train (calibrate_beauty_smoothing_v2.py).
  * Weak/medium/strong = .25/.5/.75 blend of the full filter output inside the mask.
The full (unblended) filter outputs and masks are cached once per frame (build_beauty_smoothing_v2_cache.py);
blending is done on the fly by SmoothingCache.view.
"""
import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROTOCOL = 'beauty_smoothing_v2'
FILTERS = ['bilateral', 'weighted_median', 'fast_global', 'domain_transform']
LEVELS = {'smooth_weak': .25, 'smooth_medium': .5, 'smooth_strong': .75}
ANCHOR = 'bilateral'
COLOR_SIGMA = {'bilateral': 35., 'weighted_median': 35., 'fast_global': 8., 'domain_transform': 40.}
# Visually approved spatial reach (fraction of the ROI side). The anchor stays fixed; the others are re-calibrated
# on FF++ train to the anchor's median skin change and the calibrated values are what the cache uses.
APPROVED = {'bilateral': .08225, 'weighted_median': .04465, 'fast_global': .0158, 'domain_transform': .1044}
PARSER = dict(model='jonathandinu/face-parsing', revision='758b82e15a0178c9db39c1ff666a8b56e3a550c8',
              skin_labels=[1, 2], erode_fraction=.01, blur_fraction=.012)


def full(name, array, spatial):
    """Unblended full-strength filter output of a uint8 RGB array; spatial = reach as a fraction of the ROI side."""
    side = min(array.shape[:2])
    sigma = COLOR_SIGMA[name]
    if name == 'bilateral':
        return cv2.bilateralFilter(array, 0, sigma, max(1., spatial*side))
    if name == 'weighted_median':
        return cv2.ximgproc.weightedMedianFilter(array, array, max(1, int(round(spatial*side))), sigma)
    if name == 'fast_global':
        # Smoothing reach grows like sqrt(lambda): lambda = (spatial*side)^2 keeps it proportional to the ROI side.
        return cv2.ximgproc.fastGlobalSmootherFilter(array, array, max(spatial*side, 1e-3)**2, sigma)
    if name == 'domain_transform':
        return cv2.ximgproc.dtFilter(array, array, max(1., spatial*side), sigma, mode=cv2.ximgproc.DTF_NC)
    raise ValueError(f'Unknown filter: {name}')


def finish_mask(skin, side):
    """Binary skin map -> eroded, feathered float mask in [0, 1]."""
    r = max(1, int(round(PARSER['erode_fraction']*side)))
    eroded = cv2.erode(skin.astype(np.uint8), np.ones((2*r+1, 2*r+1), np.uint8))
    return cv2.GaussianBlur(eroded.astype(np.float32), (0, 0), max(1., PARSER['blur_fraction']*side)).clip(0, 1)


class SkinParser:
    """Batched face parsing; returns one float mask per PIL image (same size as the image)."""
    def __init__(self, device=None):
        import torch
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        self.torch = torch
        self.device = device or ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.processor = SegformerImageProcessor.from_pretrained(PARSER['model'], revision=PARSER['revision'])
        self.model = SegformerForSemanticSegmentation.from_pretrained(PARSER['model'], revision=PARSER['revision'])
        self.model.to(self.device).eval()

    def masks(self, images):
        torch = self.torch
        with torch.inference_mode():
            pixels = self.processor(images=images, return_tensors='pt')['pixel_values'].to(self.device)
            logits = self.model(pixel_values=pixels).logits.float()
        out = []
        for logit, image in zip(logits, images):
            label = torch.nn.functional.interpolate(logit[None], size=image.size[::-1], mode='bilinear',
                                                    align_corners=False).argmax(1)[0].cpu().numpy()
            out.append(finish_mask(np.isin(label, PARSER['skin_labels']), min(image.size)))
        return out


def mask_to_image(mask):
    return Image.fromarray(np.rint(mask*255).astype(np.uint8), mode='L')


def image_to_mask(image):
    return np.asarray(image, dtype=np.float32)/255.


def blend(image, filtered, mask, strength):
    array = np.asarray(image.convert('RGB')).astype(np.float32)
    weight = (mask*strength)[..., None]
    return Image.fromarray(np.clip(np.rint(array*(1-weight)+filtered.astype(np.float32)*weight), 0, 255).astype(np.uint8))


def skin_change(original, filtered, mask):
    """Mask-weighted mean absolute RGB change (0-255) of the full filter: how much it edits the skin."""
    diff = np.abs(filtered.astype(np.float32)-original.astype(np.float32)).mean(axis=2)
    return float((mask*diff).sum()/max(float(mask.sum()), 1e-6))


def calibrated_values(path):
    data = json.loads(Path(path).read_text())
    if data.get('protocol') != PROTOCOL or set(data['values']) != set(FILTERS):
        raise ValueError(f'Unexpected v2 calibration file: {path}')
    return {name: float(data['values'][name]) for name in FILTERS}


def save_png(dest, image):
    dest.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp = tempfile.mkstemp(prefix=dest.name+'.', suffix='.tmp.png', dir=dest.parent)
    os.close(descriptor)
    image.save(temp, compress_level=1)
    os.replace(temp, dest)


class SmoothingCache:
    """Reader of one dataset's v2 cache: <root>/<dataset>/{mask,<filter>}/<crop_path>.

    Every part directory carries identity.json; all parts of a dataset must share it."""
    PARTS = ['mask', *FILTERS]

    def __init__(self, root, dataset, identity=None):
        self.root = Path(root)/dataset
        self.identity = identity

    def path(self, part, relative):
        relative = Path(relative)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Cache requires a relative crop path')
        return self.root/part/relative

    def check(self, parts=None):
        for part in parts or self.PARTS:
            path = self.root/part/'identity.json'
            if not (self.root/part/'complete.json').is_file():
                raise FileNotFoundError(f'v2 cache part incomplete: {self.root/part}')
            saved = json.loads(path.read_text())
            if self.identity is None:
                self.identity = saved
            elif saved != self.identity:
                raise ValueError(f'v2 cache provenance differs: {path}')
        return self.identity

    def mask(self, relative, size):
        with Image.open(self.path('mask', relative)) as opened:
            if opened.size != size:
                raise ValueError(f'Cached mask size differs: {relative}')
            return image_to_mask(opened)

    def filtered(self, name, relative, size):
        with Image.open(self.path(name, relative)) as opened:
            if opened.size != size:
                raise ValueError(f'Cached {name} size differs: {relative}')
            return np.asarray(opened.convert('RGB'))

    def view(self, image, relative, name, strength, detected=True):
        """Smoothed view; fallback (non-detected) frames stay unchanged exactly like v1."""
        if not detected or strength == 0:
            return image
        return blend(image, self.filtered(name, relative, image.size), self.mask(relative, image.size), strength)
