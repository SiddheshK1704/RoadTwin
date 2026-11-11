"""Classical pothole segmentation on a top-down road patch.

Shared by the video pipeline (the metric bird's-eye view) and the Pothole-600 dataset
(near top-down road crops). All parameters are in pixels of the patch it runs on.

Cues
  appearance : darkness vs. a large median background (+ Otsu), rim Canny-edge density,
               solidity, Hough-circle support, template matching (NCC)
  depth      : (optional, stereo) |residual| of the disparity from a robustly fitted
               road plane - potholes are depressions, whatever their colour
Regions are grown by hysteresis (weak region must contain a strong seed), cleaned by
morphology, then gated and scored per contour.
"""
from dataclasses import dataclass, field, asdict

import cv2
import numpy as np


@dataclass
class SegParams:
    bg_kernel: int = 61           # median background window (odd, > typical pothole)
    dark_weak: float = 18         # region growing threshold (raised to Otsu if higher)
    dark_strong: float = 45       # seed threshold
    disp_weak: float = 12         # same for the disparity residual
    disp_strong: float = 30
    min_area: int = 48
    max_area_frac: float = 0.9
    max_aspect: float = 4.0
    min_contrast: float = 38      # appearance-only gates
    min_rim_edges: float = 0.06
    min_solidity: float = 0.75
    disp_ref: float = 40          # disparity contrast that counts as "fully deep"
    score_thresh: float = 0.75
    use_circles: bool = True
    circle_rmin: int = 4
    circle_rmax: int = 32
    close_k: int = 7

    @classmethod
    def from_dict(cls, d):
        p = cls()
        for k, v in d.items():
            if hasattr(p, k):
                setattr(p, k, type(getattr(p, k))(v))
        return p

    def to_dict(self):
        return asdict(self)


@dataclass
class Instance:
    contour: np.ndarray
    score: float
    area_px: float
    centroid: tuple
    feats: dict = field(default_factory=dict)


def plane_residual(disp, valid=None, step=4, iters=4):
    """|disparity - robust least-squares plane|. The road is a plane in (u, v, d) space."""
    h, w = disp.shape
    ys, xs = np.mgrid[0:h:step, 0:w:step]
    d = disp[ys, xs].astype(np.float64).ravel()
    A = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)], 1).astype(np.float64)
    m = np.ones(d.size, bool) if valid is None else valid[ys, xs].ravel()
    for _ in range(iters):
        sol = np.linalg.lstsq(A[m], d[m], rcond=None)[0]
        r = d - A @ sol
        s = 1.4826 * np.median(np.abs(r[m])) + 1e-3
        m_new = m & (np.abs(r) < 2.5 * s + 1)
        if m_new.sum() < 50:
            break
        m = m_new
    yy, xx = np.mgrid[0:h, 0:w]
    plane = sol[0] * xx + sol[1] * yy + sol[2]
    return np.clip(np.abs(disp.astype(np.float64) - plane), 0, 255).astype(np.uint8)


class PotholeSegmenter:
    def __init__(self, params=None, templates=None):
        self.p = params or SegParams()
        self.templates = templates
        self.k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # ---------------------------------------------------------- stage 1: cue maps
    def features(self, g, valid=None, disp=None):
        """Parameter-light cue maps (computed once, reusable across a threshold sweep)."""
        p = self.p
        valid = np.ones(g.shape, bool) if valid is None else valid
        bg = cv2.medianBlur(g, p.bg_kernel | 1)
        dark = cv2.subtract(bg, g)
        f = dict(g=g, valid=valid, bg=bg, dark=dark, dark_s=cv2.GaussianBlur(dark, (5, 5), 0),
                 edges=cv2.Canny(g, 40, 120), dres=None, circles=None)
        if disp is not None:
            f["dres"] = cv2.GaussianBlur(plane_residual(disp, valid), (5, 5), 0)
        if p.use_circles:
            c = cv2.HoughCircles(cv2.GaussianBlur(g, (7, 7), 2), cv2.HOUGH_GRADIENT, dp=1.5,
                                 minDist=max(p.circle_rmax // 2, 4), param1=120, param2=22,
                                 minRadius=p.circle_rmin, maxRadius=p.circle_rmax)
            f["circles"] = None if c is None else c.reshape(-1, 3)
        return f

    # ------------------------------------------------------- stage 2: regions + gates
    def decide(self, f):
        p = self.p
        g, valid, dark, d, dres = f["g"], f["valid"], f["dark"], f["dark_s"], f["dres"]
        vals = d[valid]
        otsu = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0] \
            if vals.size else 255
        weak = d > max(p.dark_weak, otsu)
        strong = d > p.dark_strong
        if dres is not None:
            weak |= dres > p.disp_weak
            strong |= dres > p.disp_strong
        weak = ((weak & valid).astype(np.uint8)) * 255
        weak = cv2.morphologyEx(weak, cv2.MORPH_OPEN, self.k3)
        kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.close_k, p.close_k))
        weak = cv2.morphologyEx(weak, cv2.MORPH_CLOSE, kc)
        n, lab = cv2.connectedComponents(weak, connectivity=8)
        seeded = np.zeros(n, bool)
        seeded[np.unique(lab[strong & (weak > 0)])] = True
        seeded[0] = False
        mask = seeded[lab].astype(np.uint8) * 255

        out = []
        final = np.zeros_like(mask)
        max_area = p.max_area_frac * g.size
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (p.min_area <= area <= max_area):
                continue
            (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
            aspect = max(rw, rh) / max(min(rw, rh), 1)
            if aspect > p.max_aspect:
                continue
            solidity = area / max(cv2.contourArea(cv2.convexHull(cnt)), 1)
            if solidity < p.min_solidity:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            x0, y0 = max(x - 4, 0), max(y - 4, 0)
            x1, y1 = min(x + w + 4, g.shape[1]), min(y + h + 4, g.shape[0])
            local = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.drawContours(local, [cnt - [x0, y0]], -1, 255, -1)
            inside = local > 0
            contrast = float(dark[y0:y1, x0:x1][inside].mean())
            ring = (cv2.dilate(local, self.k7) - cv2.erode(local, self.k3)) > 0
            rim = float((f["edges"][y0:y1, x0:x1][ring] > 0).mean()) if ring.any() else 0.0
            dcon = float(dres[y0:y1, x0:x1][inside].mean()) if dres is not None else None

            # appearance-only: potholes are clearly darker AND have a sharp rim
            if dres is None and (contrast < p.min_contrast or rim < p.min_rim_edges):
                continue
            tmpl = max(self.templates.score(g, (x, y, w, h)), 0.0) if self.templates else 0.0
            circ = 0.0
            if f["circles"] is not None:
                for cx, cy, _ in f["circles"]:
                    if cv2.pointPolygonTest(cnt, (float(cx), float(cy)), False) >= 0:
                        circ = 1.0
                        break
            appearance = (0.35 * min(contrast / 50, 1) + 0.15 * solidity +
                          0.25 * min(rim / 0.25, 1) + 0.15 * tmpl + 0.10 * circ)
            score = appearance if dcon is None else 0.4 * appearance + 0.6 * min(dcon / p.disp_ref, 1)
            if score < p.score_thresh:
                continue
            M = cv2.moments(cnt)
            out.append(Instance(cnt, float(min(score, 1)), area,
                                (M["m10"] / M["m00"], M["m01"] / M["m00"]),
                                feats=dict(contrast=contrast, solidity=solidity, edges=rim,
                                           template=tmpl, circle=circ, aspect=aspect,
                                           depth=dcon)))
            cv2.drawContours(final, [cnt], -1, 255, -1)
        return final, out

    def segment(self, g, valid=None, disp=None):
        f = self.features(g, valid, disp)
        mask, inst = self.decide(f)
        return mask, inst, f
