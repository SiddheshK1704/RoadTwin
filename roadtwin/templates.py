"""Scale-normalised template matching for pothole verification.

Because candidates are found in the metric bird's-eye view, each candidate patch is
resized to a fixed size and compared with normalised cross-correlation
(TM_CCOEFF_NORMED) against a bank of templates: procedural ones (dark bowl with a
brighter broken rim) plus any real crops dropped into the template directory.
"""
import glob
import os

import cv2
import numpy as np


class TemplateBank:
    def __init__(self, size=40, extra_dir=None):
        self.size = size
        self.templates = self._procedural()
        if extra_dir and os.path.isdir(extra_dir):
            for p in sorted(glob.glob(os.path.join(extra_dir, "*.png")) +
                            glob.glob(os.path.join(extra_dir, "*.jpg"))):
                img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    self.templates.append(cv2.resize(img, (size, size)).astype(np.float32))

    def _procedural(self):
        s = self.size
        out = []
        for ratio in (1.0, 0.7, 0.5):
            for angle in ((0,) if ratio == 1.0 else (0, 90)):
                t = np.full((s, s), 120, np.float32)
                axes = (int(s * 0.32), int(s * 0.32 * ratio))
                cv2.ellipse(t, (s // 2, s // 2), axes, angle, 0, 360, 165, 3)   # rim
                cv2.ellipse(t, (s // 2, s // 2), axes, angle, 0, 360, 50, -1)   # bowl
                out.append(cv2.GaussianBlur(t, (5, 5), 0))
        return out

    @staticmethod
    def crop(img, bbox, out_size):
        """Square crop 1.6x the bbox around its centre, resized; None if too small."""
        x, y, w, h = bbox
        side = int(max(w, h) * 1.6)
        cx, cy = x + w // 2, y + h // 2
        x0, y0 = max(cx - side // 2, 0), max(cy - side // 2, 0)
        patch = img[y0:y0 + side, x0:x0 + side]
        if patch.shape[0] < 6 or patch.shape[1] < 6:
            return None
        return cv2.resize(patch, (out_size, out_size)).astype(np.float32)

    def score(self, img, bbox):
        patch = self.crop(img, bbox, self.size + 8)   # small search margin for centring error
        if patch is None:
            return 0.0
        return float(max(cv2.matchTemplate(patch, t, cv2.TM_CCOEFF_NORMED).max()
                         for t in self.templates))
