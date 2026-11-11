"""OpenCV rendering: annotated camera view, BEV debug inset, and the live digital-twin panel."""
import math

import cv2
import numpy as np

COLORS = {"pothole": (40, 40, 230), "crack": (0, 150, 255), "alligator": (200, 0, 200),
          "bump": (0, 220, 255)}
COND_COLORS = {"good": (90, 190, 60), "fair": (0, 210, 230), "poor": (0, 130, 255),
               "bad": (40, 40, 220)}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(img, s, org, scale=0.5, color=(255, 255, 255), thick=1):
    cv2.putText(img, s, (org[0] + 1, org[1] + 1), FONT, scale, (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def draw_camera(frame, res, proj, show_bev=True):
    out = frame.copy()
    cv2.polylines(out, [proj.src.astype(np.int32)], True, (200, 200, 200), 1, cv2.LINE_AA)
    for side, ln in res.lanes.items():
        if ln:
            cv2.line(out, ln[0], ln[1], (60, 220, 60), 3, cv2.LINE_AA)
    for d in res.detections:
        c = COLORS[d.kind]
        cv2.polylines(out, [d.img_poly], True, c, 2, cv2.LINE_AA)
        x, y = d.img_poly.min(0)
        _text(out, f"{d.kind} {d.conf:.2f} @{d.Z:.0f}m", (int(x), int(y) - 6), 0.5, c)

    if show_bev:
        bev = cv2.cvtColor(res.bev, cv2.COLOR_GRAY2BGR)
        bev[res.crack_map > 0] = (0, 150, 255)
        for d in res.detections:
            cv2.polylines(bev, [d.bev_poly.astype(np.int32)], True, COLORS[d.kind], 2)
        h = int(out.shape[0] * 0.55)
        w = int(bev.shape[1] * h / bev.shape[0])
        bev = cv2.resize(bev, (w, h))
        cv2.rectangle(bev, (0, 0), (w - 1, h - 1), (255, 255, 255), 1)
        out[8:8 + h, 8:8 + w] = bev
        _text(out, "bird's-eye (IPM)", (12, 8 + h + 16), 0.45)
    return out


class TwinPanel:
    """Heading-up top-down view of the digital twin around the vehicle."""

    def __init__(self, w=420, h=600, px_per_m=2.6, road_width_m=10.5):
        self.w, self.h, self.s = w, h, px_per_m
        self.road_w = road_width_m
        self.vehicle = (w // 2, int(h * 0.35))   # most of the panel shows the road already mapped

    def _tf(self, pts, pose):
        pts = np.asarray(pts, np.float64).reshape(-1, 2)
        de, dn = pts[:, 0] - pose.e, pts[:, 1] - pose.n
        sh, ch = math.sin(pose.heading), math.cos(pose.heading)
        fwd = de * sh + dn * ch
        right = de * ch - dn * sh
        return np.stack([self.vehicle[0] + right * self.s, self.vehicle[1] - fwd * self.s], 1)

    def render(self, twin, pose, route_enu, speed, localizer_name, odometer):
        img = np.full((self.h, self.w, 3), (38, 52, 40), np.uint8)   # grass
        road = route_enu if route_enu is not None and len(route_enu) > 1 else \
            np.array([(e, n) for e, n, _ in twin.trajectory])
        if pose is None:
            _text(img, "waiting for GPS fix...", (20, 40), 0.6)
            return img
        if len(road) > 1:
            p = self._tf(road, pose).astype(np.int32)
            cv2.polylines(img, [p], False, (70, 70, 70), int(self.road_w * self.s), cv2.LINE_AA)
            cv2.polylines(img, [p], False, (200, 200, 200), 1, cv2.LINE_AA)

        conds = twin.segment_conditions()
        L = twin.cfg.segment_len_m
        traj = twin.trajectory
        for (e0, n0, o0), (e1, n1, o1) in zip(traj, traj[1:]):
            c = COND_COLORS[conds.get(int(o1 // L), "good")]
            a, b = self._tf([(e0, n0), (e1, n1)], pose).astype(np.int32)
            cv2.line(img, tuple(a), tuple(b), c, 5, cv2.LINE_AA)

        for hz in twin.hazards:
            x, y = self._tf([(hz.e, hz.n)], pose)[0].astype(int)
            if not (-20 < x < self.w + 20 and -20 < y < self.h + 20):
                continue
            c = COLORS[hz.kind]
            r = int(4 + 6 * hz.severity)
            if not hz.confirmed:
                cv2.circle(img, (x, y), r, c, 1, cv2.LINE_AA)
            elif hz.kind == "pothole":
                cv2.circle(img, (x, y), r, c, -1, cv2.LINE_AA)
                cv2.circle(img, (x, y), r, (255, 255, 255), 1, cv2.LINE_AA)
            elif hz.kind == "bump":
                cv2.rectangle(img, (x - 14, y - 3), (x + 14, y + 3), c, -1)
            else:
                cv2.drawMarker(img, (x, y), c, cv2.MARKER_TILTED_CROSS, 2 * r, 2, cv2.LINE_AA)

        vx, vy = self.vehicle
        tri = np.int32([[vx, vy - 12], [vx - 8, vy + 9], [vx + 8, vy + 9]])
        cv2.fillPoly(img, [tri], (255, 200, 0))
        cv2.polylines(img, [tri], True, (255, 255, 255), 1, cv2.LINE_AA)

        # header + stats
        cv2.rectangle(img, (0, 0), (self.w, 74), (25, 25, 25), -1)
        _text(img, "ROAD DIGITAL TWIN", (10, 22), 0.6, (255, 220, 120), 2)
        counts = {}
        for hz in twin.confirmed:
            counts[hz.kind] = counts.get(hz.kind, 0) + 1
        _text(img, f"{pose.lat:.6f}, {pose.lon:.6f}   {speed * 3.6:5.1f} km/h", (10, 44), 0.45)
        _text(img, f"chainage {odometer:6.1f} m   src: {localizer_name}", (10, 64), 0.4,
              (190, 190, 190))
        y = self.h - 12
        for k in ("pothole", "crack", "alligator", "bump"):
            _text(img, f"{k}: {counts.get(k, 0)}", (10, y), 0.45, COLORS[k])
            y -= 18
        x = self.w - 80
        for k, c in COND_COLORS.items():
            cv2.line(img, (x, y + 14), (x + 18, y + 14), c, 5)
            _text(img, k, (x + 24, y + 18), 0.4)
            y += 18
        # scale bar (10 m)
        cv2.line(img, (self.w - 20 - int(10 * self.s), 90), (self.w - 20, 90), (255, 255, 255), 2)
        _text(img, "10 m", (self.w - 20 - int(10 * self.s), 84), 0.4)
        return img


def compose(cam, panel, status):
    h = panel.shape[0]
    cam = cv2.resize(cam, (int(cam.shape[1] * h / cam.shape[0]), h))
    out = np.hstack([cam, panel])
    bar = np.full((28, out.shape[1], 3), 20, np.uint8)
    _text(bar, status, (10, 19), 0.5, (220, 220, 220))
    return np.vstack([out, bar])
