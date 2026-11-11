# RoadTwin: pothole and road-irregularity mapping with classical CV

A forward-facing vehicle camera detects potholes, cracks, alligator cracking and bumps/joints. Each detection is placed on a geo-referenced **digital twin** of the road, using GPS or, in the mock demo, visual odometry along a known route. There is no deep learning: every stage is a classical, explainable CV technique, so you can measure it and then decide what to scale up.

## Quick start (mock demo, no GPS)

```bash
pip install -r requirements.txt
python tools/make_synthetic_video.py            # -> data/ (video, calib, route, ground truth)
python demo.py --video data/synthetic_road.mp4 --calib data/synthetic_calib.json \
               --route data/synthetic_route.geojson --gt data/synthetic_gt.json
```

You get an OpenCV window with the annotated camera view, a bird's-eye inset and a live heading-up twin panel. A browser map (Leaflet/OpenStreetMap) also opens and updates every second from `out/twin.geojson`. Keys: `q` quit, `space` pause, `b` toggle bird's-eye inset. Add `--save out/demo.mp4` to record the composite view.

To run on your own road video, calibrate once and then run:

```bash
python tools/calibrate.py --video myroad.mp4 --width 3.5 --near 6 --far 18 --out my_calib.json
python demo.py --video myroad.mp4 --calib my_calib.json --route my_road.geojson  # mock GPS
python demo.py --video myroad.mp4 --calib my_calib.json --gps-csv track.csv      # GPS log (t,lat,lon)
python demo.py --camera 0 --calib my_calib.json --gps-serial COM5                # live, NMEA GPS
```

`my_road.geojson` is a LineString you can draw at geojson.io along the real road. If you leave it out, the demo builds a procedural route starting at `--origin LAT,LON`.

## Pothole-600 dataset

`pothole600/{training,validation,testing}/{rgb,label,tdisp}` holds 240/180/180 near top-down 400×400 road crops. Each crop has a binary pothole mask and a transformed stereo disparity map (`tdisp`, stored as a JET colour image and decoded back to a scalar by `roadtwin/dataset.py`). The pothole segmenter is shared with the video pipeline, so improvements made on the dataset carry over to the car.

```bash
python tools/pothole600.py templates                  # 12 k-means pothole templates from training masks -> templates/
python tools/pothole600.py tune --mode rgb            # grid search on training, pick on validation -> params/
python tools/pothole600.py tune --mode rgbd           # same, with the disparity (depth) cue
python tools/pothole600.py eval --mode rgbd --split testing --vis out/p600_vis_rgbd
python demo.py --dataset pothole600 --split testing --disparity   # survey replay onto the live map
```

The two modes:
- **rgb** uses the camera image only, as the current vehicle pipeline does.
- **rgbd** adds the depth cue: a road plane is fitted robustly to the disparity, and the residual marks depressions whatever their colour.

In the **survey replay**, each crop is treated as one sample of road every `--spacing` m (default 5) along a mock route, with an assumed crop size of `--patch-m` (default 1.5 m). Detections and ground-truth masks are placed on the map the same way, and the finished map is scored.

**Results on the held-out testing split.** Parameters were chosen on training and validation only.

| Mode | Pixel P | Pixel R | Pixel F1 | IoU | Pothole recall | Pothole precision | Time per image |
|---|---|---|---|---|---|---|---|
| rgb | 0.80 | 0.07 | 0.14 | 0.07 | 0.26 | 0.88 | ~6 ms |
| rgbd | 0.40 | 0.91 | 0.56 | 0.39 | **0.985** | 0.82 | ~10 ms |

On validation, rgb reaches F1 0.60 and rgbd 0.54.

A pothole counts as found if at least 30% of its pixels are detected. A detected region counts as real if at least 30% of it is pothole.

In the survey replay, the rgbd map places 186 of 196 test potholes within 1 m. The 1 m radius is generous, so use the pothole-level numbers above as the stricter measure.

What the numbers mean:
- **The image-only cue doesn't transfer to the testing split.** Testing roads have coarse aggregate, higher contrast, and large shallow potholes that are not darker than the road. Per-image contrast normalisation (median/MAD) helped on validation but not on testing.
- **The depth cue transfers.** With stereo disparity, almost every pothole is found. The remaining error is over-segmentation: the detected region spreads beyond the rim, which keeps pixel precision near 0.40. Tightening region growing on the residual is the next thing to tune.
- **For the vehicle,** a stereo camera (or another depth source) is the most valuable upgrade, more than any further image-only tuning.

The dataset templates in `templates/` are also loaded by the video detector. On the synthetic video this changed pothole precision from 0.82 to 0.75 (one extra false positive), with recall unchanged. Delete `templates/` to go back to the procedural templates.

## Pipeline

| Stage | Technique | File |
|---|---|---|
| Lane lines | Canny → probabilistic **Hough** → length-weighted line fit + temporal smoothing | `roadtwin/lanes.py` |
| Bird's-eye view | Ground-plane **homography** (IPM) to a metric grid; CLAHE | `roadtwin/geometry.py` |
| Potholes | Median-background darkness, plus an optional **disparity road-plane residual** → **hysteresis** segmentation → morphology → contour geometry (area, solidity, aspect), rim **Canny edge density**, **Hough circles**, scale-normalised **template matching** (NCC) against k-means templates learned from Pothole-600; scored and gated | `roadtwin/segment.py`, `templates.py` |
| Cracks | Morphological **black-hat** → thin-component filter → **HoughLinesP** → grid clustering. Orientation coherence separates linear cracks from alligator cracking. | `roadtwin/detector.py` |
| Bumps / joints | Near-horizontal Hough lines spanning most of the lane, paired edges | `roadtwin/detector.py` |
| Speed (no GPS) | Shi-Tomasi corners + pyramidal **Lucas-Kanade** flow on the bird's-eye view, median shift × m/px | `roadtwin/odometry.py` |
| Localisation | Mock route + odometry / GPS-log replay / live NMEA with dead reckoning | `roadtwin/localization.py` |
| Digital twin | Tracking in world coordinates: merge radius, running-mean position, confirmed after N sightings, per-25 m condition score, GeoJSON export | `roadtwin/twin.py` |

Working in the metric bird's-eye view is the key design choice. Every threshold is in metres, so it doesn't depend on how far away a pothole is, and a hazard's position follows directly from its pixel position.

## Current results (synthetic benchmark, 300 m at 36 km/h, 900 frames)

| Class | GT | Predicted | Precision | Recall | Location error |
|---|---|---|---|---|---|
| Pothole | 12 | 12 | 0.75 | 0.75 | 0.26 m |
| Crack (incl. alligator) | 9 | 9 | 1.00 | 1.00 | 1.1 m |
| Bump | 2 | 2 | 1.00 | 1.00 | 0.03 m |

- **Odometry:** 299.1 m measured vs. 300 m true, with no GPS.
- **Speed:** about 40 FPS on a laptop CPU.
- **Distractors:** repair patches are rejected. One oil stain and one alligator patch were initially misread as potholes, and the 6-frame confirmation filters these out.
- **Misses:** potholes smaller than about 0.3 m radius are missed.

Reproduce with `--gt`. The evaluator also works on real footage if you label hazards as `{kind, s (chainage m), x (lateral m)}`.

## Tuning and limitations

All thresholds live in `roadtwin/config.py` and can be overridden from the calibration JSON.

- **Calibration matters.** Wrong IPM points give wrong metric sizes and wrong odometry. Re-run `tools/calibrate.py` for every camera mount.
- **Lighting.** The detectors assume potholes are darker than the surrounding asphalt and have a sharp rim. Hard shadows, wet patches and night scenes will produce false positives; the confirmation step only partly absorbs them.
- **Templates.** Drop real pothole crops (`*.png`/`*.jpg`) into `templates/` to extend the procedural template bank.

## Scaling path

1. **Real data first.** Record dash-cam video with a phone GPS log, label a few km, and measure with the evaluator.
2. **Learned classifier.** Keep the classical stage as the proposal generator and add a small classifier (HOG+SVM, then a small CNN) on candidate patches. This is the cheapest accuracy gain.
3. **Segmentation network.** Replace detection with a segmentation net (e.g. trained on RDD2022) once you have data. The IPM, odometry, twin and map stay the same.
4. **Fleet-scale twin.** Aggregate twins from many vehicles server-side, with multi-pass voting and hazard ageing and repair detection. IMU vibration can be added as a roughness signal.
