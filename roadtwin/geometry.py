"""Ground-plane geometry: homography between the camera image and a metric bird's-eye view.

Because the road is (locally) a plane, a single 3x3 homography maps image pixels to
road coordinates exactly. Working in the bird's-eye view (BEV) makes every size
metric and removes perspective scale changes, so thresholds are in metres.
"""
import cv2
import numpy as np


class GroundProjector:
    def __init__(self, cfg, frame_w, frame_h):
        self.cfg = cfg
        self.mpp = cfg.bev_mpp
        self.width_m = cfg.bev_width_m
        self.near_m = cfg.bev_near_m
        self.far_m = cfg.bev_far_m
        self.bw = int(round(cfg.bev_width_m / cfg.bev_mpp))
        self.bh = int(round((cfg.bev_far_m - cfg.bev_near_m) / cfg.bev_mpp))

        self.src = np.float32(cfg.src_norm) * np.float32([frame_w, frame_h])
        dst = np.float32([[0, 0], [self.bw - 1, 0], [self.bw - 1, self.bh - 1], [0, self.bh - 1]])
        self.H = cv2.getPerspectiveTransform(self.src, dst)
        self.Hinv = np.linalg.inv(self.H)

        ones = np.full((frame_h, frame_w), 255, np.uint8)
        valid = cv2.warpPerspective(ones, self.H, (self.bw, self.bh), flags=cv2.INTER_NEAREST)
        self.valid = cv2.erode(valid, np.ones((5, 5), np.uint8)) > 0

    def to_bev(self, img, interp=cv2.INTER_LINEAR):
        return cv2.warpPerspective(img, self.H, (self.bw, self.bh), flags=interp)

    def bev_to_ground(self, x, y):
        """BEV pixel -> (X lateral m, +right; Z forward m from camera)."""
        X = (x + 0.5) * self.mpp - self.width_m / 2
        Z = self.far_m - (y + 0.5) * self.mpp
        return X, Z

    def bev_to_image(self, pts):
        pts = np.asarray(pts, np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.Hinv).reshape(-1, 2)

    def image_to_bev(self, pts):
        pts = np.asarray(pts, np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)
