"""Render a synthetic forward-facing dash-cam video of a road with known hazards.

A pinhole camera (height, pitch, focal length) looks at a long top-down road texture
containing potholes, linear and alligator cracks, speed bumps, and distractors
(repair patches and oil stains that should NOT be reported). Because every hazard's
chainage/offset is known, the demo can report precision/recall and localisation error.

Outputs (in --out): synthetic_road.mp4, synthetic_calib.json, synthetic_gt.json,
synthetic_route.geojson
"""
import argparse
import json
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from roadtwin.localization import procedural_route, save_route_geojson  # noqa: E402

W, H, F = 1280, 720, 1000.0
CAM_H, PITCH = 1.5, math.radians(8.0)
PPM = 40                         # texture pixels per metre
HALF = 11.25                     # texture half-width (road 5.25 m + shoulder)
ROAD_HALF = 5.25
FAR = 180.0


def project(X, Z):
    """Ground point -> image pixel for the synthetic camera."""
    y = CAM_H * math.cos(PITCH) - Z * math.sin(PITCH)
    z = CAM_H * math.sin(PITCH) + Z * math.cos(PITCH)
    return W / 2 + F * X / z, H / 2 + F * y / z


def build_texture(rng, length_m, n_potholes, n_cracks, n_alligator, n_bumps, n_patch, n_stain):
    Ht, Wt = int(length_m * PPM), int(2 * HALF * PPM)
    col = lambda X: int((X + HALF) * PPM)
    row = lambda s: int(s * PPM)
    tex = np.full((Ht, Wt), 95, np.float32)
    tex += rng.normal(0, 9, (Ht, Wt)).astype(np.float32)
    coarse = rng.normal(0, 7, (Ht // 24 + 1, Wt // 24 + 1)).astype(np.float32)
    tex += cv2.resize(coarse, (Wt, Ht), interpolation=cv2.INTER_CUBIC)
    # shoulders: lighter, rougher dirt/grass (tinted green when rendered)
    for a, b in ((0, col(-ROAD_HALF - 0.3)), (col(ROAD_HALF + 0.3), Wt)):
        tex[:, a:b] = 120 + rng.normal(0, 18, (Ht, b - a))
    # lane markings: solid edges, dashed centre lines at +-1.75 m
    for X in (-ROAD_HALF, ROAD_HALF):
        tex[:, col(X) - 3:col(X) + 3] = 215
    for X in (-1.75, 1.75):
        for s in np.arange(0, length_m, 9.0):
            tex[row(s):row(s + 3), col(X) - 3:col(X) + 3] = 220

    gt = []
    usable = (25, length_m - 200)
    slots = np.linspace(*usable, n_potholes + n_cracks + n_alligator + n_bumps + n_patch + n_stain)
    rng.shuffle(slots)
    slots = list(slots)

    for _ in range(n_potholes):
        s, x = slots.pop() + rng.uniform(-3, 3), rng.uniform(-2.2, 2.2)
        R = rng.uniform(0.18, 0.55)
        ecc = rng.uniform(0.55, 1.0)
        th = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        r = R * (1 + 0.15 * np.sin(3 * th + rng.uniform(0, 6)) + 0.08 * np.sin(5 * th + rng.uniform(0, 6)))
        pts = np.stack([(x + r * np.cos(th) + HALF) * PPM, (s + ecc * r * np.sin(th)) * PPM], 1).astype(np.int32)
        mask = np.zeros((Ht, Wt), np.uint8)
        x0, y0 = pts.min(0) - 10
        x1, y1 = pts.max(0) + 10
        sub = mask[y0:y1, x0:x1]
        cv2.fillPoly(sub, [pts - [x0, y0]], 255)
        inside = sub > 0
        patch = tex[y0:y1, x0:x1]
        depth = cv2.GaussianBlur(inside.astype(np.float32), (0, 0), R * PPM * 0.25)
        patch[inside] = (45 + rng.normal(0, 10, inside.sum()) + 25 * (1 - depth[inside])).astype(np.float32)
        cv2.polylines(patch, [pts - [x0, y0]], True, 160, 3)            # broken bright rim
        gt.append(dict(kind="pothole", s=float(s), x=float(x), radius=float(R)))

    def walk(s, x, length, ang, jitter=0.35, step=0.08):
        p = [(x, s)]
        for _ in range(int(length / step)):
            ang += rng.normal(0, jitter)
            x, s = x + step * math.cos(ang), s + step * math.sin(ang)
            p.append((x, s))
        return np.int32([[(px + HALF) * PPM, py * PPM] for px, py in p])

    for _ in range(n_cracks):
        s, x = slots.pop(), rng.uniform(-2.0, 2.0)
        L = rng.uniform(1.5, 4.0)
        cv2.polylines(tex, [walk(s - L / 2 * 0.7, x, L, rng.uniform(0, np.pi))], False, 40, 2)
        gt.append(dict(kind="crack", s=float(s), x=float(x), length=float(L)))

    for _ in range(n_alligator):
        s, x = slots.pop(), rng.uniform(-1.8, 1.8)
        for _ in range(16):
            cs, cx = s + rng.uniform(-0.8, 0.8), x + rng.uniform(-0.8, 0.8)
            cv2.polylines(tex, [walk(cs, cx, rng.uniform(0.4, 0.9), rng.uniform(0, 2 * np.pi), 0.6)],
                          False, 42, 2)
        gt.append(dict(kind="alligator", s=float(s), x=float(x)))

    for _ in range(n_bumps):
        s = slots.pop()
        tex[row(s - 0.25):row(s + 0.25), col(-ROAD_HALF):col(ROAD_HALF)] = 185
        tex[row(s - 0.27):row(s - 0.25), col(-ROAD_HALF):col(ROAD_HALF)] = 50
        gt.append(dict(kind="bump", s=float(s), x=0.0))

    for _ in range(n_patch):     # distractor: smooth repaired-asphalt rectangle
        s, x = slots.pop(), rng.uniform(-1.5, 1.5)
        w, l = rng.uniform(1.0, 1.6), rng.uniform(1.5, 2.5)
        a, b = row(s - l / 2), row(s + l / 2)
        c, d = col(x - w / 2), col(x + w / 2)
        tex[a:b, c:d] = 78 + rng.normal(0, 5, (b - a, d - c))
        gt.append(dict(kind="patch", s=float(s), x=float(x)))

    for _ in range(n_stain):     # distractor: soft oil stain
        s, x = slots.pop(), rng.uniform(-2.0, 2.0)
        sig = rng.uniform(0.15, 0.3) * PPM
        a, c = row(s), col(x)
        rr = int(3 * sig)
        yy, xx = np.mgrid[-rr:rr, -rr:rr]
        tex[a - rr:a + rr, c - rr:c + rr] -= 30 * np.exp(-(xx ** 2 + yy ** 2) / (2 * sig ** 2))
        gt.append(dict(kind="stain", s=float(s), x=float(x)))

    return np.clip(tex, 0, 255).astype(np.uint8), sorted(gt, key=lambda h: h["s"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--length", type=float, default=300.0, help="metres driven")
    ap.add_argument("--speed-kmh", type=float, default=36.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--origin", default="18.5204,73.8567")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    tex_len = a.length + FAR + 20
    print("building road texture...")
    tex, gt = build_texture(rng, tex_len, n_potholes=12, n_cracks=6, n_alligator=3, n_bumps=2,
                            n_patch=3, n_stain=3)

    # Mip levels to avoid aliasing in the distance (a real lens integrates the footprint).
    levels = [tex, cv2.blur(tex, (1, 3)), cv2.blur(tex, (3, 9)), cv2.blur(tex, (5, 27))]

    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    xn, yn = (u - W / 2) / F, (v - H / 2) / F
    Yr = yn * math.cos(PITCH) + math.sin(PITCH)
    Zr = -yn * math.sin(PITCH) + math.cos(PITCH)
    ground = Yr > 1e-4
    t = np.where(ground, CAM_H / np.maximum(Yr, 1e-4), 0)
    X, Z = t * xn, t * Zr
    ground &= Z < FAR
    map_x = np.where(ground, (X + HALF) * PPM, -10).astype(np.float32)
    base_y = np.where(ground, Z * PPM, -10).astype(np.float32)
    footprint = PPM * Z * Z / (F * CAM_H)
    lvl = np.digitize(footprint, [1.5, 4, 12])
    haze = np.clip((Z - 40) / 140, 0, 1)[..., None].astype(np.float32)
    haze[~ground] = 1
    shoulder = (np.abs(X) > ROAD_HALF + 0.3) & ground
    tint = np.ones((H, W, 3), np.float32)
    tint[shoulder] = (0.55, 0.95, 0.7)
    sky = np.zeros((H, W, 3), np.float32)
    sky[:] = np.linspace(235, 190, H)[:, None, None] * np.float32([1.0, 0.9, 0.75])

    vid = os.path.join(a.out, "synthetic_road.mp4")
    wr = cv2.VideoWriter(vid, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, H))
    speed = a.speed_kmh / 3.6
    n = int(a.length / speed * a.fps)
    for i in range(n):
        s_cam = i * speed / a.fps
        my = base_y + np.float32(s_cam * PPM)
        g = np.zeros((H, W), np.float32)
        for k, L in enumerate(levels):
            r = cv2.remap(L, map_x, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=120)
            g = np.where(lvl == k, r.astype(np.float32), g)
        img = g[..., None] * tint
        img = img * (1 - haze) + sky * haze
        jitter = 1.5 * math.sin(2 * math.pi * 1.3 * i / a.fps) + rng.normal(0, 0.5)
        img = cv2.warpAffine(img, np.float32([[1, 0, 0], [0, 1, jitter]]), (W, H),
                             borderMode=cv2.BORDER_REPLICATE)
        img = cv2.GaussianBlur(img, (3, 3), 0) + rng.normal(0, 2, img.shape).astype(np.float32)
        wr.write(np.clip(img, 0, 255).astype(np.uint8))
        if i % 100 == 0:
            print(f"  frame {i}/{n}")
    wr.release()

    w2 = 2.75
    corners = [project(-w2, 16.5), project(w2, 16.5), project(w2, 4.5), project(-w2, 4.5)]
    calib = dict(src_norm=[[x / W, y / H] for x, y in corners],
                 bev_width_m=2 * w2, bev_near_m=4.5, bev_far_m=16.5)
    with open(os.path.join(a.out, "synthetic_calib.json"), "w") as f:
        json.dump(calib, f, indent=2)

    lat, lon = map(float, a.origin.split(","))
    route = procedural_route(lat, lon, tex_len)
    save_route_geojson(route, os.path.join(a.out, "synthetic_route.geojson"))
    with open(os.path.join(a.out, "synthetic_gt.json"), "w") as f:
        json.dump(dict(route=route, speed_mps=speed, hazards=gt), f, indent=1)
    print(f"wrote {vid} ({n} frames) + calib, route and ground truth "
          f"({sum(h['kind'] in ('pothole', 'crack', 'alligator', 'bump') for h in gt)} hazards, "
          f"{sum(h['kind'] in ('patch', 'stain') for h in gt)} distractors)")


if __name__ == "__main__":
    main()
