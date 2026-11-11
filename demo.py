"""RoadTwin demo / runner.

Mock demo (no GPS; position = visual odometry along a route):
    python demo.py --video data/synthetic_road.mp4 --calib data/synthetic_calib.json \
                   --route data/synthetic_route.geojson --gt data/synthetic_gt.json

Your own dash-cam video (calibrate first with tools/calibrate.py):
    python demo.py --video myroad.mp4 --calib my_calib.json [--route road.geojson | --origin LAT,LON]
Video + GPS log:      python demo.py --video myroad.mp4 --gps-csv track.csv
Live camera + GPS:    python demo.py --camera 0 --gps-serial COM5

Pothole-600 survey replay (dataset crops laid along a mock route, scored at the end):
    python demo.py --dataset pothole600 --split testing [--disparity]

Keys: q quit · space pause · b toggle bird's-eye inset
"""
import argparse
import os
import time
import webbrowser

import cv2

from roadtwin.config import Config
from roadtwin.detector import RoadAnomalyDetector
from roadtwin.evaluate import evaluate, print_report
from roadtwin.localization import (CsvGpsLocalizer, NmeaSerialLocalizer, RouteLocalizer,
                                   load_route, procedural_route, save_route_geojson)
from roadtwin.twin import RoadTwin
from roadtwin.viz import TwinPanel, compose, draw_camera
from roadtwin.webmap import start_server


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video")
    src.add_argument("--camera", type=int)
    src.add_argument("--dataset", help="Pothole-600 root folder (survey replay mode)")
    ap.add_argument("--split", default="testing", help="dataset split for --dataset")
    ap.add_argument("--disparity", action="store_true", help="--dataset: also use stereo disparity")
    ap.add_argument("--patch-m", type=float, default=1.5, help="--dataset: assumed ground size of a crop (m)")
    ap.add_argument("--spacing", type=float, default=5.0, help="--dataset: metres between samples")
    ap.add_argument("--replay-fps", type=float, default=4.0, help="--dataset: samples shown per second")
    ap.add_argument("--calib", help="calibration/config JSON")
    loc = ap.add_mutually_exclusive_group()
    loc.add_argument("--route", help="GeoJSON LineString the mock vehicle drives along")
    loc.add_argument("--gps-csv", help="GPS log CSV (t,lat,lon)")
    loc.add_argument("--gps-serial", help="serial port of an NMEA GPS, e.g. COM5")
    ap.add_argument("--origin", default="18.5204,73.8567",
                    help="lat,lon start of the procedural route when no route/GPS is given")
    ap.add_argument("--speed-kmh", type=float,
                    help="assume constant speed instead of visual odometry")
    ap.add_argument("--gt", help="ground-truth JSON to score the twin against")
    ap.add_argument("--out", default="out")
    ap.add_argument("--save", help="write the composite visualisation to this .mp4")
    ap.add_argument("--no-web", action="store_true", help="don't start the browser map")
    ap.add_argument("--headless", action="store_true", help="no OpenCV window")
    ap.add_argument("--max-frames", type=int)
    return ap.parse_args()


def make_localizer(a):
    if a.gps_serial:
        return NmeaSerialLocalizer(a.gps_serial)
    if a.gps_csv:
        return CsvGpsLocalizer(a.gps_csv)
    if a.route:
        return RouteLocalizer(load_route(a.route))
    lat, lon = map(float, a.origin.split(","))
    pts = procedural_route(lat, lon, 3000)
    save_route_geojson(pts, os.path.join(a.out, "route.geojson"))
    return RouteLocalizer(pts)


def start_web(a):
    if not a.no_web:
        url = start_server(a.out)
        print(f"live map: {url}")
        webbrowser.open(url)


def report(twin, odometer, geo_path):
    counts = {}
    for h in twin.confirmed:
        counts[h.kind] = counts.get(h.kind, 0) + 1
    print(f"\ndistance {odometer:.1f} m, confirmed hazards: {counts}")
    print(f"digital twin saved to {geo_path}")


def main_dataset(a, cfg):
    from roadtwin.survey import run_survey
    if a.gps_csv or a.gps_serial:
        raise SystemExit("--dataset uses a route (--route or --origin), not live/logged GPS")
    loc = make_localizer(a)
    geo_path = os.path.join(a.out, "twin.geojson")
    twin, pose, gt_path = run_survey(a, cfg, loc, geo_path, lambda t, p: t.save(geo_path, p),
                                     lambda: start_web(a))
    twin.save(geo_path, pose)
    report(twin, twin.trajectory[-1][2] if twin.trajectory else 0.0, geo_path)
    r = evaluate(twin, gt_path, radius_m=1.0)
    print("\n=== Map vs ground-truth masks (match radius 1 m) ===")
    g = r["pothole"]
    print(f"potholes: GT {g['gt']}  mapped {g['pred']}  TP {g['tp']}  precision {g['precision']:.2f}  "
          f"recall {g['recall']:.2f}  loc err {g['mean_err_m']:.2f} m")
    if not a.headless:
        cv2.destroyAllWindows()
    if not a.no_web:
        input("Map server still running - press Enter to exit.")


def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = Config.load(a.calib)
    if a.dataset:
        return main_dataset(a, cfg)
    cap = cv2.VideoCapture(a.video if a.video else a.camera)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {a.video or a.camera}")
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("no frames")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    live = a.video is None
    H, W = frame.shape[:2]

    det = RoadAnomalyDetector(cfg, W, H)
    loc = make_localizer(a)
    twin = None
    panel = TwinPanel(h=600)
    geo_path = os.path.join(a.out, "twin.geojson")

    start_web(a)

    writer = None
    idx, odometer, speed, t_video = 0, 0.0, 0.0, 0.0
    last_wall = time.time()
    show_bev, paused = True, False
    t_proc = 0.0
    pose = None
    while ok:
        now = time.time()
        dt = (now - last_wall) if live else 1.0 / fps
        last_wall = now
        t0 = time.perf_counter()
        res = det.process(frame, dt)

        if a.speed_kmh is not None:
            speed = a.speed_kmh / 3.6
        elif res.speed is not None:
            speed = res.speed
        if idx > 0:
            odometer += speed * dt
            t_video += dt

        pose = loc.pose(odometer, t_video)
        if pose is not None:
            if twin is None:
                twin = RoadTwin(cfg, loc.frame)
            world = []
            for d in res.detections:
                e, n = pose.offset(float(d.X), float(d.Z))
                world.append((d.kind, e, n, d.conf, d.severity, float(d.size_m2), float(d.Z)))
            twin.update(world, idx, pose, odometer)
        t_proc = 0.9 * t_proc + 0.1 * (time.perf_counter() - t0) if idx else time.perf_counter() - t0

        if twin is not None and idx % 10 == 0:
            twin.save(geo_path, pose)

        if not a.headless or a.save:
            cam = draw_camera(frame, res, det.proj, show_bev)
            pimg = panel.render(twin, pose, loc.route_enu(), speed, loc.name, odometer) \
                if twin is not None else panel.render(None, None, None, speed, loc.name, odometer)
            n_conf = len(twin.confirmed) if twin else 0
            status = (f"frame {idx}  |  {1 / max(t_proc, 1e-6):5.1f} FPS processing  |  "
                      f"{len(res.detections)} detections in frame  |  {n_conf} hazards in twin")
            vis = compose(cam, pimg, status)
            if a.save:
                if writer is None:
                    writer = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                             (vis.shape[1], vis.shape[0]))
                writer.write(vis)
            if not a.headless:
                cv2.imshow("RoadTwin", vis)
                k = cv2.waitKey(0 if paused else 1) & 0xFF
                if k == ord("q") or k == 27:
                    break
                if k == ord(" "):
                    paused = not paused
                if k == ord("b"):
                    show_bev = not show_bev

        idx += 1
        if a.max_frames and idx >= a.max_frames:
            break
        ok, frame = cap.read()

    cap.release()
    if writer:
        writer.release()
    if twin is not None:
        twin.save(geo_path, pose)
        print(f"\nprocessed {idx} frames, avg processing {1 / max(t_proc, 1e-6):.1f} FPS")
        report(twin, odometer, geo_path)
        if a.gt:
            print_report(evaluate(twin, a.gt))
    if not a.headless:
        cv2.destroyAllWindows()
    if not a.no_web:
        input("Map server still running - press Enter to exit.")


if __name__ == "__main__":
    main()
