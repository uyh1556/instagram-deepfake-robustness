"""Held-out skin-smoothing filters for generalisation tests (never used in training).

One filter per family of filter-based skin retouching (StyleRetoucher, arXiv 2312.14389, Sec. 2.1):
  weighted_median   local edge-preserving   Zhang et al., CVPR 2014 (100+ times faster WMF)
  fast_global       global energy optimisation  Min et al., TIP 2014 (fast global smoother)
  domain_transform  structure-preserving (guided)  Gastal & Oliveira, SIGGRAPH 2011
All use OpenCV contrib (cv2.ximgproc), spatial scales proportional to the ROI side (like the bilateral
sigma_space_roi_fraction) and the SAME geometric skin mask and blend strengths. ONE parameter per filter
is calibrated (PARAMETER): the spatial fraction for weighted_median/fast_global (colour sigma fixed to
the bilateral 35); for domain_transform the colour sigma (its spatial sigma fixed to the bilateral 0.02),
because the domain transform treats fine skin-noise gradients as edges unless the colour sigma grows.
"""
import cv2
import numpy as np
from PIL import Image

import smoothing_pixel_cache as pixel_cache

FILTERS = ['weighted_median', 'fast_global', 'domain_transform']
SIGMA_COLOR = 35.
SPATIAL_FRACTION = .02
HF_SIGMA_FRACTION = .01
PARAMETER = {'weighted_median': 'spatial_fraction', 'fast_global': 'spatial_fraction', 'domain_transform': 'sigma_color'}
GRIDS = {'spatial_fraction': np.geomspace(.002, .08, 13), 'sigma_color': np.geomspace(10., 3000., 13)}


def full(name, array, value):
    """Unblended, full-strength filter output for a uint8 RGB array; value = the calibrated PARAMETER."""
    side = min(array.shape[:2])
    if name == 'weighted_median':
        return cv2.ximgproc.weightedMedianFilter(array, array, max(1, int(round(value*side))), SIGMA_COLOR)
    if name == 'fast_global':
        # Smoothing reach grows like sqrt(lambda): lambda = scale^2 keeps it proportional to the ROI side.
        return cv2.ximgproc.fastGlobalSmootherFilter(array, array, max(value*side, 1e-3)**2, SIGMA_COLOR)
    if name == 'domain_transform':
        return cv2.ximgproc.dtFilter(array, array, max(1., SPATIAL_FRACTION*side), float(value), mode=cv2.ximgproc.DTF_NC)
    if name == 'bilateral':
        raise ValueError('Use the training bilateral settings, not a fraction')
    raise ValueError(f'Unknown filter: {name}')


def smooth(image, name, value, strength):
    """Skin-masked blend exactly like the training bilateral views."""
    filtered = full(name, np.asarray(image.convert('RGB')), value)
    return pixel_cache.blend(image, filtered, strength)


def luminance(array):
    return array.astype(np.float32) @ np.array([.299, .587, .114], np.float32)


def high_frequency_energy(array, mask):
    y = luminance(array)
    sigma = max(1., HF_SIGMA_FRACTION*min(y.shape))
    residual = y-cv2.GaussianBlur(y, (0, 0), sigma)
    return float((mask*residual**2).sum()/mask.sum())


def hf_reduction(original, filtered, mask):
    """1 - E_hf(filtered)/E_hf(original) inside the skin mask; 0 = untouched, 1 = no detail left."""
    return 1-high_frequency_energy(filtered, mask)/max(high_frequency_energy(original, mask), 1e-12)


def bilateral_full(image, settings):
    return pixel_cache.bilateral(image, settings)


def mask(image):
    return pixel_cache.mask_for(image.size)[..., 0]


def as_image(array):
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
