"""Classical road-anomaly detector.

Pipeline per frame
  1. Lanes        : Canny + probabilistic Hough on the camera image.
  2. IPM          : homography to a metric bird's-eye view (BEV) + CLAHE.
  3. Potholes     : shared classical segmenter (roadtwin/segment.py) on the BEV:
                    median-background darkness, hysteresis, morphology, contour gates,
                    rim edges, Hough circles and template matching fused into a score.
  4. Cracks       : thin-dark-structure blackhat -> probabilistic Hough segments ->
                    grid clustering; orientation spread separates linear from alligator cracking.
  5. Bumps/joints : near-horizontal Hough lines spanning most of the road width.
  6. Odometry     : LK optical flow on the BEV (used when no GPS speed is available).
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from .geometry import GroundProjector
from .lanes import LaneDetector
from .odometry import BevOdometry
from .segment import PotholeSegmenter, SegParams
from .templates import TemplateBank


def seg_params_from_config(cfg, proj):
    """Metric config -> pixel parameters of the shared segmenter on the BEV grid."""
    mpp = cfg.bev_mpp
    return SegParams(bg_kernel=int(cfg.bg_kernel_m / mpp) | 1, dark_weak=cfg.dark_thresh,
                     dark_strong=cfg.dark_strong, min_area=int(cfg.min_area_m2 / mpp ** 2),
                     max_area_frac=cfg.max_area_m2 / (mpp ** 2 * proj.bw * proj.bh),
                     max_aspect=cfg.max_aspect, min_contrast=cfg.min_contrast,
                     min_rim_edges=cfg.min_rim_edges, min_solidity=cfg.min_solidity,
                     score_thresh=cfg.pothole_score_thresh, use_circles=cfg.use_hough_circles,
                     circle_rmin=int(0.1 / mpp), circle_rmax=int(0.8 / mpp))


@dataclass
class Detection:
    kind: str             # pothole | crack | alligator | bump
    conf: float           # 0..1
    severity: float       # 0..1
    size_m2: float
    X: float              # lateral offset, metres (+ right of camera)
    Z: float              # forward distance from camera, metres
    bev_poly: np.ndarray
    img_poly: np.ndarray = None
    feats: dict = field(default_factory=dict)


@dataclass
class FrameResult:
    detections: list
    lanes: dict
    bev: np.ndarray
    dark: np.ndarray
    crack_map: np.ndarray
    speed: float


class RoadAnomalyDetector:
    def __init__(self, cfg, frame_w, frame_h):
        self.cfg = cfg
        self.proj = GroundProjector(cfg, frame_w, frame_h)
        self.lanes = LaneDetector(cfg, frame_w, frame_h, self.proj.src)
        self.templates = TemplateBank(extra_dir=cfg.template_dir)
        self.seg = PotholeSegmenter(seg_params_from_config(cfg, self.proj), self.templates)
        self.odo = BevOdometry(cfg.bev_mpp, cfg.max_speed_mps)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 8))
        mpp = cfg.bev_mpp
        k = int(cfg.crack_blackhat_m / mpp) | 1
        self.k_crack = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        self.k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # ------------------------------------------------------------------ main
    def process(self, frame, dt):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lanes = self.lanes.detect(gray)

        bev = self.proj.to_bev(gray)
        g = cv2.medianBlur(self.clahe.apply(bev), 3)
        f = self.seg.features(g, self.proj.valid)
        bright = cv2.subtract(g, f["bg"]) > 40                 # paint / lane markings
        bright = cv2.dilate(bright.astype(np.uint8), self.k7) > 0
        road = self.proj.valid & ~bright
        f["valid"] = road
        dark, edges = f["dark"], f["edges"]

        potholes, pmask = self._potholes(f)
        cracks, crack_map = self._cracks(g, road, pmask)
        bumps = self._bumps(edges, road)
        speed = self.odo.update(g, self.proj.valid, dt)

        dets = potholes + cracks + bumps
        for d in dets:
            d.img_poly = self.proj.bev_to_image(d.bev_poly).astype(np.int32)
        return FrameResult(dets, lanes, g, dark, crack_map, speed)

    # ------------------------------------------------------------ potholes
    def _potholes(self, f):
        mpp = self.cfg.bev_mpp
        pmask, insts = self.seg.decide(f)
        out = []
        for ins in insts:
            X, Z = self.proj.bev_to_ground(*ins.centroid)
            area_m2 = ins.area_px * mpp * mpp
            sev = float(np.clip(0.5 * min(area_m2 / 1.0, 1) + 0.5 * min(ins.feats["contrast"] / 60, 1), 0, 1))
            out.append(Detection("pothole", ins.score, sev, area_m2, X, Z,
                                 ins.contour.reshape(-1, 2).astype(np.float32), feats=ins.feats))
        return out, pmask

    # -------------------------------------------------------------- cracks
    def _cracks(self, g, road, pmask):
        cfg, mpp = self.cfg, self.cfg.bev_mpp
        bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, self.k_crack)
        excl = ~road | (cv2.dilate(pmask, self.k7, iterations=2) > 0)
        bh[excl] = 0
        bw = cv2.threshold(bh, cfg.crack_thresh, 255, cv2.THRESH_BINARY)[1]
        # keep only components that are long relative to their area (thin structures)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
        min_len = cfg.crack_min_len_m / mpp
        keep = np.zeros(n, bool)
        for i in range(1, n):
            ext = max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
            if ext >= 0.6 * min_len and stats[i, cv2.CC_STAT_AREA] < 0.35 * ext * ext:
                keep[i] = True
        bw = (keep[lab]).astype(np.uint8) * 255

        lines = cv2.HoughLinesP(bw, 1, np.pi / 180, threshold=12,
                                minLineLength=int(min_len), maxLineGap=6)
        if lines is None:
            return [], bw
        cell = cfg.crack_cell_m / mpp
        gh, gw = int(np.ceil(g.shape[0] / cell)), int(np.ceil(g.shape[1] / cell))
        grid_len = np.zeros((gh, gw), np.float32)
        members = {}
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            L = float(np.hypot(x2 - x1, y2 - y1))
            r, c = int(((y1 + y2) / 2) // cell), int(((x1 + x2) / 2) // cell)
            grid_len[r, c] += L
            members.setdefault((r, c), []).append((x1, y1, x2, y2, L))
        active = (grid_len * mpp >= cfg.crack_min_total_m).astype(np.uint8)
        n, lab = cv2.connectedComponents(active, connectivity=8)

        out = []
        for i in range(1, n):
            segs = [s for rc in zip(*np.where(lab == i)) for s in members.get(rc, [])]
            pts = np.float32([[s[0], s[1]] for s in segs] + [[s[2], s[3]] for s in segs])
            total_m = sum(s[4] for s in segs) * mpp
            ang = np.array([np.arctan2(s[3] - s[1], s[2] - s[0]) for s in segs])
            coherence = float(np.abs(np.mean(np.exp(2j * ang))))   # 1 = all parallel
            kind = "alligator" if (len(segs) >= 4 and coherence < 0.55) else "crack"
            hull = cv2.convexHull(pts).reshape(-1, 2)
            if len(hull) < 3:
                x, y, w, h = cv2.boundingRect(pts)
                hull = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
            cx, cy = pts.mean(0)
            X, Z = self.proj.bev_to_ground(cx, cy)
            area = max(cv2.contourArea(hull), 1) * mpp * mpp
            sev = min(1.0, (0.55 if kind == "alligator" else 0.2) + 0.1 * total_m)
            out.append(Detection(kind, float(min(1, total_m / 1.5)), sev, area, X, Z,
                                 hull.astype(np.float32),
                                 feats=dict(length_m=total_m, coherence=coherence, segs=len(segs))))
        return out, bw

    # --------------------------------------------------------------- bumps
    def _bumps(self, edges, road):
        cfg, mpp = self.cfg, self.cfg.bev_mpp
        e = edges.copy()
        e[~self.proj.valid] = 0
        W = e.shape[1]
        lines = cv2.HoughLinesP(e, 1, np.pi / 180, threshold=int(W * 0.35),
                                minLineLength=int(W * cfg.bump_min_span), maxLineGap=12)
        if lines is None:
            return []
        ys = []
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            ang = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))
            if ang <= cfg.bump_max_angle_deg:
                ys.append((y1 + y2) / 2)
        ys.sort()
        out, group = [], []
        gap = 0.6 / mpp
        for y in ys + [None]:
            if group and (y is None or y - group[-1] > gap):
                if len(group) >= 2:            # a raised band has (at least) two parallel edges
                    y0, y1 = min(group) - 3, max(group) + 3
                    poly = np.float32([[0, y0], [W - 1, y0], [W - 1, y1], [0, y1]])
                    X, Z = self.proj.bev_to_ground(W / 2, (y0 + y1) / 2)
                    out.append(Detection("bump", min(1.0, len(group) / 4), 0.5,
                                         W * (y1 - y0) * mpp * mpp, X, Z, poly,
                                         feats=dict(edges=len(group))))
                group = []
            if y is not None:
                group.append(y)
        return out
