"""Pothole-600: build templates, tune the classical segmenter, and evaluate it.

    python tools/pothole600.py templates                      # k-means templates from training masks
    python tools/pothole600.py tune --mode rgb                # grid search on training, check on validation
    python tools/pothole600.py tune --mode rgbd               # same, with the stereo disparity cue
    python tools/pothole600.py eval --mode rgbd --split testing --vis out/p600_vis

Modes: rgb  = monocular camera only (what the vehicle pipeline uses)
       rgbd = rgb + transformed disparity (stereo camera upgrade path)
Tuned parameters are written to params/pothole600_<mode>.json.
"""
import argparse
import itertools
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from roadtwin.dataset import BASE_PARAMS as BASE, Pothole600, PatchPrep, load_params, params_path  # noqa: E402
from roadtwin.segment import PotholeSegmenter, SegParams  # noqa: E402
from roadtwin.templates import TemplateBank  # noqa: E402

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = 200
GRID = {
    "rgb": dict(bg_kernel=[41, 61, 91], dark_strong=[35, 45, 55], min_contrast=[25, 38],
                min_rim_edges=[0.03, 0.06], min_solidity=[0.5, 0.7], score_thresh=[0.6, 0.7, 0.8]),
    "rgbd": dict(bg_kernel=[41, 61, 91], dark_strong=[40, 60], disp_strong=[20, 30, 45],
                 disp_weak=[8, 14], min_solidity=[0.4, 0.6], score_thresh=[0.5, 0.6, 0.7, 0.8]),
}
FEATURE_KEYS = ("bg_kernel",)


# ------------------------------------------------------------------ metrics
class Metrics:
    def __init__(self):
        self.tp = self.fp = self.fn = 0
        self.ious = []
        self.inst_gt = self.inst_gt_hit = self.inst_pred = self.inst_pred_ok = 0

    def add(self, pred, gt):
        tp = int((pred & gt).sum()); fp = int((pred & ~gt).sum()); fn = int((~pred & gt).sum())
        self.tp += tp; self.fp += fp; self.fn += fn
        self.ious.append(tp / max(tp + fp + fn, 1))
        ng, lg = cv2.connectedComponents(gt.astype(np.uint8))
        npd, lp = cv2.connectedComponents(pred.astype(np.uint8))
        for i in range(1, ng):
            comp = lg == i
            if comp.sum() < 50:
                continue
            self.inst_gt += 1
            self.inst_gt_hit += (pred[comp].mean() >= 0.3)          # >=30% of the pothole found
        for i in range(1, npd):
            comp = lp == i
            self.inst_pred += 1
            self.inst_pred_ok += (gt[comp].mean() >= 0.3)           # >=30% of the region is pothole

    def summary(self):
        p = self.tp / max(self.tp + self.fp, 1)
        r = self.tp / max(self.tp + self.fn, 1)
        return dict(precision=p, recall=r, f1=2 * p * r / max(p + r, 1e-9),
                    iou=self.tp / max(self.tp + self.fp + self.fn, 1), mean_img_iou=float(np.mean(self.ious)),
                    inst_recall=self.inst_gt_hit / max(self.inst_gt, 1),
                    inst_precision=self.inst_pred_ok / max(self.inst_pred, 1),
                    inst_gt=self.inst_gt, inst_pred=self.inst_pred)


def fmt(s):
    return (f"pixel P {s['precision']:.3f}  R {s['recall']:.3f}  F1 {s['f1']:.3f}  IoU {s['iou']:.3f}  "
            f"mIoU/img {s['mean_img_iou']:.3f} | instances: recall {s['inst_recall']:.3f} "
            f"({s['inst_gt']} GT)  precision {s['inst_precision']:.3f} ({s['inst_pred']} pred)")


# ------------------------------------------------------------------ data
def load_split(root, split, mode):
    ds, prep, out = Pothole600(root, split), PatchPrep(WORK), []
    for i in range(len(ds)):
        s = ds[i]
        g, d = prep(s.rgb, s.disp if mode == "rgbd" else None)
        out.append((s, g, d))
    return out


def run(data, params, templates, keep=False):
    seg = PotholeSegmenter(params, templates)
    m, preds = Metrics(), []
    for s, g, d in data:
        mask, inst, _ = seg.segment(g, None, d)
        pred = cv2.resize(mask, s.label.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
        m.add(pred, s.label)
        if keep:
            preds.append((pred, inst))
    return m.summary(), preds


# ------------------------------------------------------------------ commands
def cmd_templates(a):
    ds, prep = Pothole600(a.root, "training"), PatchPrep(WORK)
    size, crops = 40, []
    for i in range(len(ds)):
        s = ds[i]
        g, _ = prep(s.rgb)
        lab = cv2.resize(s.label.astype(np.uint8), (WORK, WORK), interpolation=cv2.INTER_NEAREST)
        n, _, stats, _ = cv2.connectedComponentsWithStats(lab)
        for j in range(1, n):
            if stats[j, cv2.CC_STAT_AREA] < 60:
                continue
            c = TemplateBank.crop(g, tuple(stats[j, :4]), size)
            if c is not None:
                crops.append((c - c.mean()) / (c.std() + 1e-6))
    X = np.float32([c.ravel() for c in crops])
    k = min(a.k, len(X))
    _, lab, centers = cv2.kmeans(X, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3),
                                 5, cv2.KMEANS_PP_CENTERS)
    out = os.path.join(ROOT_DIR, "templates")
    os.makedirs(out, exist_ok=True)
    for j, c in enumerate(centers):
        t = c.reshape(size, size)
        t = cv2.normalize(t, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(os.path.join(out, f"p600_{j:02d}.png"), t)
    print(f"{len(crops)} pothole crops -> {k} k-means templates in {out}")


def cmd_tune(a):
    t0 = time.time()
    templates = TemplateBank(extra_dir=os.path.join(ROOT_DIR, "templates"))
    train = load_split(a.root, "training", a.mode)
    val = load_split(a.root, "validation", a.mode)
    grid = GRID[a.mode]
    keys = list(grid)
    feat_keys = [k for k in keys if k in FEATURE_KEYS]
    dec_keys = [k for k in keys if k not in FEATURE_KEYS]
    results = []
    for fvals in itertools.product(*[grid[k] for k in feat_keys]):
        fp = dict(BASE, **dict(zip(feat_keys, fvals)))
        seg = PotholeSegmenter(SegParams.from_dict(fp), templates)
        feats = [(s, seg.features(g, None, d)) for s, g, d in train]    # cue maps computed once
        for dvals in itertools.product(*[grid[k] for k in dec_keys]):
            p = dict(fp, **dict(zip(dec_keys, dvals)))
            seg.p = SegParams.from_dict(p)
            m = Metrics()
            for s, f in feats:
                mask, _ = seg.decide(f)
                m.add(cv2.resize(mask, s.label.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0, s.label)
            results.append((m.summary()["f1"], p))
        print(f"  {fvals}: {len(results)} configs evaluated ({time.time() - t0:.0f}s)")
    results.sort(key=lambda r: -r[0])
    print(f"top-{a.top} on training (pixel F1): " + ", ".join(f"{r[0]:.3f}" for r in results[:a.top]))
    best, best_f1 = None, -1
    for f1_train, p in results[:a.top]:
        s, _ = run(val, SegParams.from_dict(p), templates)
        if s["f1"] > best_f1:
            best, best_f1, best_s = p, s["f1"], s
    os.makedirs(os.path.dirname(params_path(a.mode)), exist_ok=True)
    with open(params_path(a.mode), "w") as f:
        json.dump({k: best[k] for k in keys}, f, indent=2)
    print(f"selected on validation -> {params_path(a.mode)}")
    print({k: best[k] for k in keys})
    print("validation:", fmt(best_s))


def cmd_eval(a):
    templates = TemplateBank(extra_dir=os.path.join(ROOT_DIR, "templates"))
    data = load_split(a.root, a.split, a.mode)
    t0 = time.perf_counter()
    s, preds = run(data, load_params(a.mode), templates, keep=bool(a.vis))
    dt = (time.perf_counter() - t0) / len(data)
    print(f"[{a.mode}] {a.split} ({len(data)} images, {1000 * dt:.1f} ms/img): {fmt(s)}")
    if a.vis:
        os.makedirs(a.vis, exist_ok=True)
        for (smp, _, d), (pred, inst) in zip(data, preds):
            v = smp.rgb.copy()
            for m, c in ((smp.label, (0, 220, 0)), (pred, (0, 0, 255))):
                cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(v, cnts, -1, c, 2)
            sc = WORK and smp.rgb.shape[0] / WORK
            for ins in inst:
                cx, cy = ins.centroid
                cv2.putText(v, f"{ins.score:.2f}", (int(cx * sc), int(cy * sc)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(os.path.join(a.vis, smp.name), v)
        print(f"overlays (green = ground truth, red = prediction) -> {a.vis}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.join(ROOT_DIR, "pothole600"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("templates"); t.add_argument("--k", type=int, default=12)
    t = sub.add_parser("tune"); t.add_argument("--mode", choices=GRID, default="rgb")
    t.add_argument("--top", type=int, default=10)
    t = sub.add_parser("eval"); t.add_argument("--mode", choices=GRID, default="rgb")
    t.add_argument("--split", default="testing"); t.add_argument("--vis")
    a = ap.parse_args()
    {"templates": cmd_templates, "tune": cmd_tune, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
