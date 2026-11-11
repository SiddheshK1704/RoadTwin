"""Visual odometry on the bird's-eye view.

Shi-Tomasi corners on the road surface are tracked with pyramidal Lucas-Kanade.
In the metric BEV, forward motion shows up as a pure downward shift, so
speed = median(dy) * metres_per_pixel / dt. This replaces GPS speed in the mock demo.
"""
import cv2
import numpy as np


class BevOdometry:
    def __init__(self, mpp, max_speed=45.0, smooth=0.7):
        self.mpp = mpp
        self.max_speed = max_speed
        self.smooth = smooth
        self.prev = None
        self.speed = None

    def update(self, bev_gray, mask, dt):
        prev, self.prev = self.prev, bev_gray
        if prev is None or dt <= 0:
            return self.speed
        m = mask.astype(np.uint8) * 255
        m[: m.shape[0] // 3] = 0          # far rows are smeared by IPM; use the near 2/3
        p0 = cv2.goodFeaturesToTrack(prev, maxCorners=200, qualityLevel=0.01, minDistance=7, mask=m)
        if p0 is None or len(p0) < 10:
            return self.speed
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, bev_gray, p0, None, winSize=(21, 21), maxLevel=3)
        ok = st.ravel() == 1
        if ok.sum() < 10:
            return self.speed
        d = (p1 - p0).reshape(-1, 2)[ok]
        dy = d[:, 1]
        med = np.median(dy)
        inliers = np.abs(dy - med) < max(2.0, 3 * np.median(np.abs(dy - med)))
        if inliers.sum() < 8:
            return self.speed
        v = float(np.clip(np.median(dy[inliers]) * self.mpp / dt, 0, self.max_speed))
        self.speed = v if self.speed is None else self.smooth * self.speed + (1 - self.smooth) * v
        return self.speed
