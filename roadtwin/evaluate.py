"""Score the digital twin against ground truth (synthetic video or a hand-labelled survey).

Ground-truth JSON: {"route": [[lat, lon], ...], "hazards": [{"kind", "s", "x"}, ...]}
where s is chainage along the route (m) and x the lateral offset (m, + right).
Kinds other than pothole/crack/alligator/bump (e.g. 'patch', 'stain') are distractors.
"""
import json
import math

from .localization import LocalFrame, _Polyline
from .twin import KIND_GROUP


def evaluate(twin, gt_path, radius_m=4.0):
    with open(gt_path) as f:
        gt = json.load(f)
    route = _Polyline(twin.frame, [tuple(p) for p in gt["route"]])
    truth = []
    for h in gt["hazards"]:
        if h["kind"] not in KIND_GROUP:
            continue
        e, n = route.pose(h["s"]).offset(h["x"], 0.0)
        truth.append((KIND_GROUP[h["kind"]], e, n))

    report = {}
    for group in ("pothole", "crack", "bump"):
        T = [(e, n) for g, e, n in truth if g == group]
        P = [(h.e, h.n) for h in twin.confirmed if KIND_GROUP[h.kind] == group]
        pairs = sorted((math.hypot(p[0] - t[0], p[1] - t[1]), i, j)
                       for i, p in enumerate(P) for j, t in enumerate(T))
        used_p, used_t, errs = set(), set(), []
        for d, i, j in pairs:
            if d > radius_m or i in used_p or j in used_t:
                continue
            used_p.add(i); used_t.add(j); errs.append(d)
        tp = len(errs)
        report[group] = dict(gt=len(T), pred=len(P), tp=tp,
                             precision=tp / len(P) if P else float("nan"),
                             recall=tp / len(T) if T else float("nan"),
                             mean_err_m=sum(errs) / tp if tp else float("nan"))
    return report


def print_report(report):
    print("\n=== Evaluation vs ground truth ===")
    print(f"{'class':<9}{'GT':>5}{'pred':>6}{'TP':>5}{'prec':>8}{'recall':>8}{'loc err':>9}")
    for k, r in report.items():
        print(f"{k:<9}{r['gt']:>5}{r['pred']:>6}{r['tp']:>5}{r['precision']:>8.2f}"
              f"{r['recall']:>8.2f}{r['mean_err_m']:>8.2f}m")
