"""Central configuration. Every tunable lives here so experiments are reproducible.

A calibration JSON (written by tools/calibrate.py or the synthetic generator)
can override any field via Config.load(path).
"""
from dataclasses import dataclass, field, asdict, fields
import json


@dataclass
class Config:
    # ---------------- Inverse perspective mapping (IPM) ----------------
    # Image points (normalised 0..1) of a rectangle lying flat on the road, in order:
    # far-left, far-right, near-right, near-left.
    src_norm: list = field(default_factory=lambda: [[0.40, 0.62], [0.60, 0.62],
                                                    [0.92, 0.95], [0.08, 0.95]])
    bev_width_m: float = 5.5      # lateral extent of that rectangle (m)
    bev_near_m: float = 4.5       # distance from camera to its near edge (m)
    bev_far_m: float = 16.5       # distance from camera to its far edge (m)
    bev_mpp: float = 0.025        # bird's-eye resolution, metres per pixel

    # ---------------- Pothole detector ----------------
    bg_kernel_m: float = 1.5      # median background window (larger than typical pothole)
    dark_thresh: int = 18         # weak darkness vs. local background (region growing)
    dark_strong: int = 45         # a region must contain pixels at least this dark (seed)
    min_contrast: float = 38.0    # mean darkness of the region
    min_rim_edges: float = 0.06   # Canny edge density on the region boundary
    min_solidity: float = 0.75    # area / convex-hull area (crack networks are ragged)
    min_area_m2: float = 0.03
    max_area_m2: float = 3.0
    max_aspect: float = 4.0       # more elongated than this -> treated as crack, not pothole
    pothole_score_thresh: float = 0.75
    use_hough_circles: bool = True
    template_dir: str = "templates"   # optional extra real pothole crops (*.png/*.jpg)

    # ---------------- Crack detector ----------------
    crack_blackhat_m: float = 0.2
    crack_thresh: int = 22
    crack_min_len_m: float = 0.3
    crack_cell_m: float = 1.0
    crack_min_total_m: float = 0.6

    # ---------------- Bump / transverse joint detector ----------------
    bump_min_span: float = 0.6    # fraction of BEV width a horizontal Hough line must span
    bump_max_angle_deg: float = 8.0

    # ---------------- Lanes ----------------
    lane_min_slope: float = 0.35

    # ---------------- Odometry ----------------
    max_speed_mps: float = 45.0

    # ---------------- Digital twin ----------------
    min_hits: int = 6            # frames a hazard must be seen before it is confirmed
    merge_radius_m: dict = field(default_factory=lambda: {"pothole": 1.0, "crack": 1.5,
                                                          "alligator": 2.0, "bump": 2.5})
    candidate_ttl_frames: int = 20
    segment_len_m: float = 25.0

    @classmethod
    def load(cls, path=None):
        cfg = cls()
        if path:
            with open(path) as f:
                data = json.load(f)
            names = {f.name for f in fields(cls)}
            for k, v in data.items():
                if k in names:
                    setattr(cfg, k, v)
        return cfg

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
