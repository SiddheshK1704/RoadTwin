"""Interactive IPM calibration for a real camera.

Click 4 points on the road that form a rectangle on the ground, in order:
far-left, far-right, near-right, near-left. (Easiest: two lane-line points far away and
two close by; lane width ~3.5 m is a good known width, dashed lane markings give length.)

    python tools/calibrate.py --video myroad.mp4 --width 3.5 --near 6 --far 18 --out my_calib.json

Keys: r reset · s save · q quit
"""
import argparse
import json

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--width", type=float, required=True, help="rectangle width on the road (m)")
    ap.add_argument("--near", type=float, required=True, help="distance camera -> near edge (m)")
    ap.add_argument("--far", type=float, required=True, help="distance camera -> far edge (m)")
    ap.add_argument("--out", default="calib.json")
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.frame)
    ok, img = cap.read()
    if not ok:
        raise SystemExit("cannot read frame")
    h, w = img.shape[:2]
    pts = []
    names = ["far-left", "far-right", "near-right", "near-left"]

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))

    cv2.namedWindow("calibrate")
    cv2.setMouseCallback("calibrate", on_mouse)
    while True:
        vis = img.copy()
        for i, p in enumerate(pts):
            cv2.circle(vis, p, 5, (0, 0, 255), -1)
            cv2.putText(vis, names[i], (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        if len(pts) == 4:
            for i in range(4):
                cv2.line(vis, pts[i], pts[(i + 1) % 4], (0, 255, 0), 2)
        msg = f"click {names[len(pts)]}" if len(pts) < 4 else "s = save, r = reset"
        cv2.putText(vis, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("calibrate", vis)
        k = cv2.waitKey(20) & 0xFF
        if k == ord("r"):
            pts.clear()
        elif k == ord("s") and len(pts) == 4:
            calib = dict(src_norm=[[x / w, y / h] for x, y in pts], bev_width_m=a.width,
                         bev_near_m=a.near, bev_far_m=a.far)
            with open(a.out, "w") as f:
                json.dump(calib, f, indent=2)
            print(f"saved {a.out}")
            break
        elif k in (ord("q"), 27):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
