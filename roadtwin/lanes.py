"""Lane-line detection: Canny edges + probabilistic Hough transform + weighted line fit."""
import cv2
import numpy as np


class LaneDetector:
    def __init__(self, cfg, frame_w, frame_h, src_pts, smooth=0.8, hold_frames=10):
        self.cfg = cfg
        self.w, self.h = frame_w, frame_h
        self.y_top = float(min(src_pts[0][1], src_pts[1][1]))
        self.y_bot = float(frame_h - 1)
        # ROI: the IPM trapezoid widened sideways and stretched to the bottom of the frame
        tl, tr, br, bl = src_pts
        pad = 0.08 * frame_w
        roi = np.int32([[tl[0] - pad, tl[1]], [tr[0] + pad, tr[1]],
                        [frame_w - 1, frame_h - 1], [0, frame_h - 1]])
        self.roi = np.zeros((frame_h, frame_w), np.uint8)
        cv2.fillPoly(self.roi, [roi], 255)
        self.smooth = smooth
        self.hold = hold_frames
        self.state = {"left": None, "right": None}
        self.missing = {"left": 0, "right": 0}

    def detect(self, gray):
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        edges = cv2.bitwise_and(edges, self.roi)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30,
                                minLineLength=int(self.h * 0.04), maxLineGap=int(self.h * 0.05))
        groups = {"left": [], "right": []}
        if lines is not None:
            for x1, y1, x2, y2 in lines.reshape(-1, 4):   # (N,1,4) in OpenCV 4, (N,4) in 5
                if x2 == x1:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < self.cfg.lane_min_slope:
                    continue
                mx = (x1 + x2) / 2
                side = "left" if slope < 0 and mx < self.w * 0.55 else \
                       "right" if slope > 0 and mx > self.w * 0.45 else None
                if side:
                    groups[side].append((x1, y1, x2, y2, np.hypot(x2 - x1, y2 - y1)))

        for side, segs in groups.items():
            fit = self._fit(segs)
            prev = self.state[side]
            if fit is not None:
                self.missing[side] = 0
                self.state[side] = fit if prev is None else self.smooth * prev + (1 - self.smooth) * fit
            else:
                self.missing[side] += 1
                if self.missing[side] > self.hold:
                    self.state[side] = None
        return {k: (None if v is None else self._endpoints(v)) for k, v in self.state.items()}

    def _fit(self, segs):
        """Fit x = a*y + b, weighted by segment length."""
        if len(segs) < 1:
            return None
        ys, xs, ws = [], [], []
        for x1, y1, x2, y2, length in segs:
            ys += [y1, y2]; xs += [x1, x2]; ws += [length, length]
        a, b = np.polyfit(ys, xs, 1, w=np.sqrt(ws))
        return np.array([a, b])

    def _endpoints(self, ab):
        a, b = ab
        return (int(a * self.y_bot + b), int(self.y_bot)), (int(a * self.y_top + b), int(self.y_top))
