"""Pothole-600 loader.

Layout: <root>/{training,validation,testing}/{rgb,label,tdisp}/NNNN.png
  rgb   : 400x400 near top-down road crop
  label : binary pothole mask (0/255)
  tdisp : transformed stereo disparity (road plane flattened), stored as a JET colour image;
          decoded back to a scalar 0..255 by nearest-colour lookup.
"""
import functools
import glob
import json
import os
from dataclasses import dataclass

import cv2
import numpy as np

from .segment import SegParams

SPLITS = ("training", "validation", "testing")


@functools.lru_cache(maxsize=1)
def _jet_inverse():
    """6-bit-per-channel BGR cube -> nearest JET index (built once, ~2 s)."""
    lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(-1, 1),
                            cv2.COLORMAP_JET).reshape(256, 3).astype(np.int32)
    q = np.stack(np.meshgrid(np.arange(64), np.arange(64), np.arange(64), indexing="ij"),
                 -1).reshape(-1, 3) * 4 + 2
    inv = np.empty(len(q), np.uint8)
    for a in range(0, len(q), 16384):
        inv[a:a + 16384] = ((q[a:a + 16384, None, :] - lut[None]) ** 2).sum(-1).argmin(1)
    return inv


def decode_jet(img):
    c = img.astype(np.int32) >> 2
    return _jet_inverse()[(c[..., 0] << 12) | (c[..., 1] << 6) | c[..., 2]]


@dataclass
class Sample:
    name: str
    rgb: np.ndarray
    label: np.ndarray       # bool
    disp: np.ndarray        # uint8 scalar disparity (None if missing)


class Pothole600:
    def __init__(self, root, split):
        self.dir = os.path.join(root, split)
        self.names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(self.dir, "rgb", "*.png")))
        if not self.names:
            raise FileNotFoundError(f"no images in {self.dir}/rgb")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        n = self.names[i]
        rgb = cv2.imread(os.path.join(self.dir, "rgb", n))
        lab = cv2.imread(os.path.join(self.dir, "label", n), cv2.IMREAD_GRAYSCALE)
        td = cv2.imread(os.path.join(self.dir, "tdisp", n))
        return Sample(n, rgb, None if lab is None else lab > 127, None if td is None else decode_jet(td))


class PatchPrep:
    """Same conditioning as the video BEV (grey, working resolution, CLAHE, median 3), plus
    robust contrast normalisation: road texture is rescaled to a fixed spread (median/MAD),
    so grey-level thresholds mean "N texture-sigmas" on smooth and coarse surfaces alike."""

    def __init__(self, work=200, target_sigma=16.0):
        self.work = work
        self.target_sigma = target_sigma
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

    def __call__(self, rgb, disp=None):
        g = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY), (self.work, self.work),
                       interpolation=cv2.INTER_AREA)
        g = cv2.medianBlur(self.clahe.apply(g), 3).astype(np.float32)
        med = np.median(g)
        sigma = 1.4826 * np.median(np.abs(g - med)) + 1e-3
        g = np.clip(128 + (g - med) * (self.target_sigma / sigma), 0, 255).astype(np.uint8)
        d = None if disp is None else cv2.resize(disp, (self.work, self.work), interpolation=cv2.INTER_AREA)
        return g, d


# Segmenter defaults for 200x200 working patches; tuned values override them.
BASE_PARAMS = dict(min_area=40, max_area_frac=0.95, circle_rmin=6, circle_rmax=60, close_k=9,
                   dark_weak=12, disp_weak=10, disp_ref=40)
PARAMS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "params")


def params_path(mode):
    return os.path.join(PARAMS_DIR, f"pothole600_{mode}.json")


def load_params(mode):
    """mode: 'rgb' (monocular) or 'rgbd' (with stereo disparity)."""
    p = dict(BASE_PARAMS)
    if os.path.exists(params_path(mode)):
        with open(params_path(mode)) as f:
            p.update(json.load(f))
    return SegParams.from_dict(p)
