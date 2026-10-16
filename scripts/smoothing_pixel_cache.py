"""Lossless native bilateral cache; identical pixels to smoothing_roi_pilot.smooth."""
import json
import os
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import smoothing_roi_pilot as smoothing


def bilateral(image, settings):
    array = np.asarray(image.convert('RGB'))
    spatial = max(1., min(image.size)*settings['sigma_space_roi_fraction'])
    return cv2.bilateralFilter(array, settings['diameter'], settings['sigma_color_rgb255'], spatial)


@lru_cache(maxsize=8)
def mask_for(size):
    # Exact original mask calculation, only reused by shape within each worker.
    return smoothing.mask_for(size)[..., None]


def blend(image, filtered, strength):
    array = np.asarray(image.convert('RGB'))
    mask = mask_for(image.size)*strength
    result = np.rint(array.astype(np.float32)*(1-mask)+filtered.astype(np.float32)*mask)
    return Image.fromarray(np.clip(result,0,255).astype(np.uint8))


class BilateralCache:
    def __init__(self, root, settings, identity):
        self.root, self.settings = Path(root), settings
        self.identity = identity
        self.space_warning = False
        self.root.mkdir(parents=True,exist_ok=True)
        path = self.root/'protocol.json'
        if path.exists() and json.loads(path.read_text()) != identity:
            raise ValueError('Bilateral cache provenance differs; use a new --train-cache')
        if not path.exists():
            path.write_text(json.dumps(identity,sort_keys=True,indent=2)+'\n')

    def filtered(self, image, relative):
        relative=Path(relative)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Cache requires a relative frozen crop path')
        dest=self.root/'bilateral'/relative
        if dest.is_file():
            with Image.open(dest) as opened:
                if opened.size != image.size or opened.mode != 'RGB':
                    raise ValueError(f'Cached native image dimensions/mode differ: {dest}')
                return np.array(opened)
        filtered=bilateral(image,self.settings)
        if shutil.disk_usage(self.root).free < 2*2**30 + 2*filtered.nbytes:
            if not self.space_warning:
                print('Bilateral cache: free disk below 2GiB; compute uncached misses without deleting data',flush=True)
                self.space_warning=True
            return filtered
        dest.parent.mkdir(parents=True,exist_ok=True)
        descriptor,temporary=tempfile.mkstemp(prefix=dest.name+'.',suffix='.tmp.png',dir=dest.parent)
        os.close(descriptor)
        temp=Path(temporary)
        try:
            Image.fromarray(filtered).save(temp,format='PNG',compress_level=1)
            temp.replace(dest)
        finally:
            temp.unlink(missing_ok=True)
        return filtered

    def smooth(self,image,relative,strength):
        return blend(image,self.filtered(image,relative),strength)
