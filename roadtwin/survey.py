"""Pothole-600 survey replay: map a dataset split onto the digital twin.

The dataset has no video or GPS, so each 400x400 top-down crop is treated as one
survey sample of the road, taken every `spacing_m` metres along a (mock) route.
Pothole pixels are converted to ground offsets with an assumed patch size, geo-located
with the vehicle pose, and merged into the twin, exactly like the video pipeline.
Ground-truth masks are mapped the same way, so the finished map can be scored.
"""
import json
import os
import time

import cv2
import numpy as np

from .dataset import PatchPrep, Pothole600, load_params
from .segment import PotholeSegmenter
from .templates import TemplateBank
from .twin import RoadTwin
from .viz import TwinPanel, compose, _text

CAMERA_OFFSET_M = 2.0     # distance from vehicle reference point to the near edge of a patch


def _ground(cx, cy, work, mpp):
    """Working-resolution pixel -> (X right, Z ahead) metres; image top = farther away."""
    return (cx - work / 2) * mpp, (work - cy) * mpp + CAMERA_OFFSET_M


def run_survey(a, cfg, loc, geo_path, save_twin, start_web):
    use_disp = a.disparity
    mode = "rgbd" if use_disp else "rgb"
    ds = Pothole600(a.dataset, a.split)
    prep = PatchPrep()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    seg = PotholeSegmenter(load_params(mode), TemplateBank(extra_dir=os.path.join(root, "templates")))
    mpp = a.patch_m / prep.work
    cfg.min_hits = 1                    # every patch is seen exactly once
    twin = RoadTwin(cfg, loc.frame)
    panel = TwinPanel(h=600, px_per_m=4.0)
    gt = []
    start_web()

    writer, pose, delay = None, None, max(int(1000 / a.replay_fps), 1)
    n = min(len(ds), a.max_frames or len(ds))
    for i in range(n):
        s = ds[i]
        odo = i * a.spacing
        pose = loc.pose(odo, i / a.replay_fps)
        g, d = prep(s.rgb, s.disp if use_disp else None)
        t0 = time.perf_counter()
        mask, inst, feats = seg.segment(g, None, d)
        dt = time.perf_counter() - t0

        world = []
        for ins in inst:
            X, Z = _ground(*ins.centroid, prep.work, mpp)
            e, n_ = pose.offset(X, Z)
            area = ins.area_px * mpp * mpp
            sev = float(np.clip(0.5 * min(area / 0.5, 1) + 0.5 * ins.score, 0, 1))
            world.append(("pothole", e, n_, ins.score, sev, area, Z))
        twin.update(world, i, pose, odo)

        lab = cv2.resize(s.label.astype(np.uint8), (prep.work, prep.work), interpolation=cv2.INTER_NEAREST)
        k, _, st, cen = cv2.connectedComponentsWithStats(lab)
        for j in range(1, k):
            if st[j, cv2.CC_STAT_AREA] >= 12:
                X, Z = _ground(*cen[j], prep.work, mpp)
                gt.append(dict(kind="pothole", s=odo + Z, x=X))

        if i % 5 == 0 or i == n - 1:
            save_twin(twin, pose)
        if a.headless and not a.save:
            continue
        view = s.rgb.copy()
        sc = s.rgb.shape[0] / prep.work
        for m, c in ((s.label, (0, 220, 0)), (cv2.resize(mask, s.rgb.shape[1::-1]) > 0, (0, 0, 255))):
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(view, cnts, -1, c, 2)
        for ins in inst:
            _text(view, f"pothole {ins.score:.2f}", (int(ins.centroid[0] * sc) - 30, int(ins.centroid[1] * sc)),
                  0.45, (80, 80, 255))
        if feats["dres"] is not None:
            inset = cv2.applyColorMap(cv2.normalize(feats["dres"], None, 0, 255, cv2.NORM_MINMAX),
                                      cv2.COLORMAP_INFERNO)
            inset = cv2.resize(inset, (120, 120))
            view[6:126, 6:126] = inset
            _text(view, "depth residual", (8, 140), 0.4)
        _text(view, "green: ground truth   red: detected", (8, view.shape[0] - 10), 0.45)
        pimg = panel.render(twin, pose, loc.route_enu(), a.spacing * a.replay_fps,
                            "Pothole-600 replay on mock route", odo)
        status = (f"Pothole-600 {a.split} [{mode}]  sample {i + 1}/{n} ({s.name})  |  "
                  f"{1000 * dt:.1f} ms  |  {len(inst)} detected  |  {len(twin.confirmed)} in twin")
        vis = compose(view, pimg, status)
        if a.save:
            if writer is None:
                writer = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"), a.replay_fps,
                                         (vis.shape[1], vis.shape[0]))
            writer.write(vis)
        if not a.headless:
            cv2.imshow("RoadTwin", vis)
            key = cv2.waitKey(delay) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                cv2.waitKey(0)
    if writer:
        writer.release()

    gt_path = os.path.join(a.out, f"pothole600_{a.split}_gt.json")
    with open(gt_path, "w") as f:
        json.dump(dict(route=[list(loc.frame.to_latlon(e, n)) for e, n in loc.route_enu()],
                       hazards=gt), f)
    return twin, pose, gt_path
