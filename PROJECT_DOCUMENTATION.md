# RoadTwin: Complete Project Documentation

**Road-hazard detection with classical computer vision, mapped onto a geo-referenced digital twin of the road**

This document explains the whole project from start to end: what problem it solves, the technology it uses, how the system is put together, how every algorithm works and why it was chosen, how it was tested, what the results are, and how to run and extend it.

---

## Table of contents

1. [Project overview](#1-project-overview)
2. [Technology stack](#2-technology-stack)
3. [Repository layout](#3-repository-layout)
4. [System architecture](#4-system-architecture)
5. [Methodology: the processing pipeline in detail](#5-methodology-the-processing-pipeline-in-detail)
6. [Localisation: putting detections on the map](#6-localisation-putting-detections-on-the-map)
7. [The digital twin](#7-the-digital-twin)
8. [Visualisation and the live web map](#8-visualisation-and-the-live-web-map)
9. [Synthetic test-video generator](#9-synthetic-test-video-generator)
10. [The Pothole-600 dataset integration](#10-the-pothole-600-dataset-integration)
11. [Tuning and evaluation methodology](#11-tuning-and-evaluation-methodology)
12. [Results](#12-results)
13. [How to run everything](#13-how-to-run-everything)
14. [Configuration reference](#14-configuration-reference)
15. [Design decisions and rationale](#15-design-decisions-and-rationale)
16. [Limitations](#16-limitations)
17. [Roadmap for scaling up](#17-roadmap-for-scaling-up)
18. [Glossary](#18-glossary)

---

## 1. Project overview

### 1.1 The problem

Potholes and surface defects damage vehicles, cause accidents and are expensive to find by manual inspection. Vehicles already drive every road every day. A forward-facing camera on a car or truck, together with GPS, could survey the road continuously and keep an up-to-date map of where the damage is.

### 1.2 What the system does

1. **Reads a forward-facing camera feed** (live camera or recorded video).
2. **Detects road irregularities** in each frame:
   - **potholes**
   - **linear cracks**
   - **alligator (network) cracking**
   - **speed bumps and transverse joints**
3. **Estimates where each hazard is on the ground**, in metres relative to the vehicle, using a road-plane homography.
4. **Locates the vehicle** with GPS. Without GPS, it uses **visual odometry** along a known route.
5. **Builds a digital twin** of the road: a persistent, geo-referenced map of confirmed hazards plus a condition rating for each road segment.
6. **Shows it live** in an OpenCV dashboard and in a browser map (Leaflet + OpenStreetMap). The twin is exported as GeoJSON.

### 1.3 Design philosophy

The detection is deliberately built only from **classical computer vision**: homographies, morphology, thresholding, Canny edges, Hough transforms, template matching and optical flow. There is no deep learning. This has three benefits:

- **Explainable:** every decision can be traced to a measurable feature such as "darker than the road by 40 grey levels, with a sharp rim".
- **Light:** it runs at roughly 35–40 FPS on a laptop CPU with no GPU.
- **A measurable baseline:** it gives a clear reference point before scaling up to learned models.

### 1.4 Three ways to run it

| Mode | Input | Localisation | Purpose |
|---|---|---|---|
| **Mock video demo** | Synthetic or recorded road video | Visual odometry along a route (no GPS) | Live demo without hardware |
| **Real deployment** | Live camera or dash-cam video | Live NMEA GPS, or a recorded GPS log | Real surveying |
| **Dataset survey replay** | Pothole-600 image crops | Crops laid along a mock route | Benchmarking on real potholes |

---

## 2. Technology stack

| Layer | Technology | Used for |
|---|---|---|
| Language | **Python 3.13** (works on 3.9+) | Everything |
| Computer vision | **OpenCV 5.0** (4.8+ also supported) | Image processing, homography, Hough, morphology, optical flow, k-means, video I/O, drawing |
| Numerics | **NumPy** | Array maths, least squares, statistics |
| Web map (front end) | **Leaflet 1.9.4** (from cdnjs) + **OpenStreetMap** tiles | Interactive live map in the browser |
| Web map (back end) | Python standard library `http.server` (`ThreadingHTTPServer`) running in a thread | Serves `map.html` and `twin.geojson` locally |
| Data formats | **GeoJSON** (twin, routes), **JSON** (config, calibration, ground truth, parameters), **CSV** (GPS logs), **PNG** (dataset, templates), **MP4** (`mp4v` codec) | Interchange |
| GPS (optional) | **pyserial** + a hand-written NMEA `$xxRMC` parser | Live GPS receiver on a serial/USB port |
| Concurrency | `threading` | Web server and GPS reader run in the background |

**Dependencies** (`requirements.txt`): `opencv-python`, `numpy`. `pyserial` is needed only for live GPS. There are no deep-learning frameworks.

---

## 3. Repository layout

```
CV_proj/
├── demo.py                      Main entry point: video, camera and dataset-replay modes
├── requirements.txt
├── README.md                    Quick start
├── PROJECT_DOCUMENTATION.md     This document
│
├── roadtwin/                    The library
│   ├── config.py                All tunable parameters (dataclass, JSON-overridable)
│   ├── geometry.py              Ground-plane homography / bird's-eye view (IPM)
│   ├── lanes.py                 Lane-line detection (Canny + Hough)
│   ├── segment.py               Shared classical pothole segmenter (RGB + optional depth)
│   ├── templates.py             Template bank and NCC template matching
│   ├── detector.py              Per-frame pipeline: lanes, potholes, cracks, bumps, odometry
│   ├── odometry.py              Visual odometry (Lucas-Kanade flow on the bird's-eye view)
│   ├── localization.py          Local ENU frame, poses, mock route / GPS log / live NMEA
│   ├── twin.py                  Digital twin: association, confirmation, segments, GeoJSON
│   ├── viz.py                   OpenCV rendering: camera overlay, inset, twin panel
│   ├── webmap.py                Leaflet page and local HTTP server
│   ├── evaluate.py              Scores the twin against geo-located ground truth
│   ├── dataset.py               Pothole-600 loader, disparity decoding, patch preparation
│   └── survey.py                Dataset survey replay onto the twin
│
├── tools/
│   ├── make_synthetic_video.py  Renders a synthetic dash-cam video with known hazards
│   ├── calibrate.py             Click-4-points IPM calibration for a real camera
│   └── pothole600.py            Dataset: build templates, tune, evaluate
│
├── data/                        Generated synthetic video, calibration, route, ground truth
├── params/                      Tuned segmenter parameters (pothole600_rgb/rgbd.json)
├── templates/                   12 k-means pothole templates learned from Pothole-600
├── pothole600/                  The dataset (training / validation / testing)
└── out/                         Run outputs: twin.geojson, map.html, videos, overlays
```

---

## 4. System architecture

### 4.1 Layered view

```
                ┌──────────────────────────────────────────────────────────────┐
  INPUT         │ Camera / video file          Pothole-600 crops (+ disparity) │
                └──────────────┬──────────────────────────────┬────────────────┘
                               │ frame                        │ crop
                ┌──────────────▼──────────────┐   ┌───────────▼───────────────┐
  PERCEPTION    │ detector.py                 │   │ survey.py + dataset.py     │
                │  lanes.py      (Hough)      │   │  PatchPrep (normalise)     │
                │  geometry.py   (IPM / BEV)  │   │                            │
                │  segment.py ◄──────── shared pothole segmenter ──────────────┤
                │  cracks / bumps (Hough)     │   │                            │
                │  odometry.py   (LK flow)    │   │                            │
                └──────────────┬──────────────┘   └───────────┬───────────────┘
                               │ detections in metres (X right, Z ahead)
                ┌──────────────▼──────────────────────────────▼────────────────┐
  LOCALISATION  │ localization.py: Pose(e, n, heading) from                     │
                │   RouteLocalizer (odometry) | CsvGpsLocalizer | NmeaSerial    │
                │   pose.offset(X, Z) → world east/north (m) → lat/lon          │
                └──────────────┬───────────────────────────────────────────────┘
                ┌──────────────▼───────────────────────────────────────────────┐
  MAPPING       │ twin.py: associate → merge → confirm → segment conditions     │
                └──────────────┬───────────────────────────────────────────────┘
                ┌──────────────▼───────────────┐   ┌───────────────────────────┐
  OUTPUT        │ viz.py (OpenCV dashboard)    │   │ twin.geojson → webmap.py  │
                │ optional MP4 recording       │   │ Leaflet live map (browser)│
                └──────────────────────────────┘   └───────────────────────────┘
```

### 4.2 What happens in one frame (video mode)

```
frame (BGR)
  ├─► grey ─► LaneDetector ─────────────────────────────► lane lines (display)
  └─► grey ─► warpPerspective(H) ─► CLAHE ─► median 3 ─► g (bird's-eye view, 0.025 m/px)
                                                          │
            PotholeSegmenter.features(g) ◄────────────────┤  median background, darkness,
                 │                                        │  Canny, Hough circles
                 ├─► bright paint mask ─► road mask       │
                 ├─► decide() ─► pothole instances        │
                 │                                        ├─► cracks (black-hat + HoughLinesP)
                 │                                        ├─► bumps  (horizontal HoughLinesP)
                 │                                        └─► BevOdometry ─► speed (m/s)
                 ▼
  detections (kind, confidence, severity, size, X, Z, polygon)
                 │  odometer += speed·dt ;  pose = localizer.pose(odometer, t)
                 ▼
  world (e, n) = pose.offset(X, Z) ─► RoadTwin.update() ─► every 10 frames: twin.geojson
                 ▼
  draw camera overlay + bird's-eye inset + twin panel ─► window / MP4
```

### 4.3 Key interfaces

| Interface | Contract |
|---|---|
| `RoadAnomalyDetector.process(frame, dt) → FrameResult` | Detections, lanes, BEV image, darkness map, crack map, speed |
| `Detection` | `kind, conf, severity, size_m2, X, Z, bev_poly, img_poly, feats` |
| `PotholeSegmenter.features(g, valid, disp)` / `.decide(f)` | Two stages: cue maps, then regions and gates. The split lets threshold sweeps reuse the expensive maps. |
| `localizer.pose(odometer_m, t_s) → Pose` | Every position source has the same interface |
| `Pose.offset(X, Z) → (e, n)` | Vehicle-relative ground point to world east/north |
| `RoadTwin.update(world_dets, frame_idx, pose, odometer)` | Feeds the map |

---

## 5. Methodology: the processing pipeline in detail

### 5.1 Inverse perspective mapping (bird's-eye view)

**Idea.** The road surface near the vehicle is approximately a **plane**. Any plane seen by a pinhole camera is related to its image by a single 3×3 **homography** H. Mark four image points that form a known rectangle on the road (width W, from distance Z_near to Z_far). `cv2.getPerspectiveTransform` then gives the H that maps the image to a top-down grid where every pixel is a fixed size on the ground.

```
image point (u, v)  ──H──►  BEV pixel (x, y)  ──►  ground (X, Z)
X = (x + 0.5)·mpp − W/2          (lateral, + right of camera)
Z = Z_far − (y + 0.5)·mpp        (forward distance; top row = far)
```

- **Default BEV:** 5.5 m wide, 4.5–16.5 m ahead, 0.025 m/px, giving a 220 × 480 pixel image.
- **Valid mask:** a white image is warped once to record which BEV pixels actually come from the camera, then eroded 5 px.
- `bev_to_image` uses H⁻¹, so detections can be drawn back on the camera frame.

**Why it matters.** In the BEV, a 30 cm pothole is always the same number of pixels, whether it is 5 m or 15 m away. All thresholds can therefore be in **metres** (areas in m², kernel sizes in m), and a hazard's position on the ground comes straight from its pixel coordinates.

**Calibration.** `tools/calibrate.py` shows a frame. You click far-left, far-right, near-right, near-left, give the real width, near and far distances, and it saves normalised `src_norm` points. For the synthetic camera, the generator computes these points exactly by projecting the ground rectangle through its camera model.

### 5.2 Image conditioning

- **Grey conversion.** Colour carries little pothole information (measured on the dataset: the chroma difference between pothole and road is about 1.6 vs. 1.3 units).
- **CLAHE** (Contrast-Limited Adaptive Histogram Equalisation: clip limit 2.0, 4×8 tiles on the BEV, 4×4 on dataset crops). It evens out illumination and shadows locally.
- **3×3 median blur** removes speckle from asphalt grain.
- **Dataset only: robust contrast normalisation.** `g' = 128 + (g − median)·16 / (1.4826·MAD)`. This rescales every crop so its road texture has the same spread, so grey-level thresholds mean "N texture standard deviations".

### 5.3 Lane-line detection (`lanes.py`)

1. **Edges:** Gaussian blur 5×5, then **Canny** (50/150).
2. **Region of interest:** the IPM trapezoid widened by 8% of the frame width and extended to the bottom of the image.
3. **Probabilistic Hough transform** (`HoughLinesP`: ρ = 1 px, θ = 1°, threshold 30, minimum length 4% of the frame height, maximum gap 5%).
4. **Filter and classify:** segments with |slope| < 0.35 are rejected. Negative slope in the left part of the frame is the left lane line; positive slope in the right part is the right lane line.
5. **Fit:** x = a·y + b by weighted least squares (weight = √length).
6. **Smoothing:** exponential moving average (factor 0.8). A line is kept for up to 10 frames when it isn't detected.

Lanes are used for **display and context**. Detection itself runs on the whole BEV, so hazards in adjacent lanes are still found.

### 5.4 Pothole segmentation (`segment.py`, shared)

This is the core detector. It is shared by the video pipeline (on the BEV) and the dataset (on crops), so improvements made on real data carry straight over to the vehicle.

#### Stage 1: cue maps (`features`)

| Cue | Computation | Captures |
|---|---|---|
| Background | `medianBlur(g, k)` with k larger than a pothole (1.5 m on the BEV) | Local road appearance, ignoring small defects and lane paint |
| **Darkness** | `dark = max(bg − g, 0)` | Potholes are shadowed or wet depressions, darker than the road |
| Edges | Canny 40/120 | Sharp, broken pothole rims |
| **Hough circles** | `HoughCircles` (gradient method, dp 1.5, param2 22, radius 0.1–0.8 m) | Roughly round depressions |
| **Depth residual** (optional) | Robust plane fit to disparity, then the absolute residual | Physical depressions, whatever their colour |

**Robust road-plane fit.** The disparity d(u, v) of a flat road is linear in the image coordinates: d ≈ a·u + b·v + c. The plane is fitted by least squares on a 4-pixel grid and repeated 4 times. Each time, points whose residual exceeds 2.5σ + 1 are discarded, with σ estimated robustly as 1.4826 × the median absolute deviation (MAD). The final |d − plane| highlights potholes and also works for raw, untransformed disparity.

#### Stage 2: regions and decisions (`decide`)

1. **Thresholds.**
   - *Weak:* darkness > max(`dark_weak`, Otsu threshold of the road pixels).
   - *Strong:* darkness > `dark_strong`.
   - With depth, these are OR-ed with residual > `disp_weak` (weak) and > `disp_strong` (strong).
2. **Morphology:** opening with a 3×3 ellipse removes speckle; closing with a 7×7 (video) or 9×9 (dataset) ellipse fills holes.
3. **Hysteresis:** a connected weak region is kept only if it contains at least one strong pixel. Faint blotches are dropped, but a pothole's full extent is kept.
4. **Per-contour gates:**
   - area between `min_area` and `max_area`
   - aspect ratio (from `minAreaRect`) ≤ 4, because thinner shapes are cracks
   - **solidity** = area / convex-hull area ≥ `min_solidity`, because crack networks are ragged
5. **Per-contour features:**
   - `contrast` = mean darkness inside the region
   - `rim` = fraction of Canny edge pixels in the boundary band (7×7 dilation minus 3×3 erosion)
   - `template` = best normalised cross-correlation (NCC) against the template bank
   - `circle` = 1 if a Hough-circle centre falls inside the region
   - `depth` = mean depth residual inside the region
6. **Image-only hard gates:** `contrast ≥ min_contrast` **and** `rim ≥ min_rim_edges`. Oil stains and soft blotches are dark but have no sharp rim.
7. **Score** (weights chosen from feature distributions on labelled data):

```
appearance = 0.35·min(contrast/50, 1) + 0.15·solidity + 0.25·min(rim/0.25, 1)
           + 0.15·template + 0.10·circle
score      = appearance                                          (image only)
score      = 0.4·appearance + 0.6·min(depth/disp_ref, 1)         (with depth)
keep if score ≥ score_thresh
```

8. **Severity** (video): `0.5·min(area/1 m², 1) + 0.5·min(contrast/60, 1)`.

**How the score was designed.** On the synthetic video, every candidate region was labelled as true pothole or distractor, and the feature distributions were compared:

| Candidates | Contrast (median) | Rim edges (median) |
|---|---|---|
| True potholes | 44 | 0.21 |
| False regions | 23 | 0 |

That evidence is why contrast and rim edges get the largest weights and act as hard gates.

### 5.5 Template matching (`templates.py`)

- **Scale normalisation:** each candidate is cropped as a square 1.6× its bounding box and resized to 48×48. Templates are 40×40, so `matchTemplate(TM_CCOEFF_NORMED)` searches a ±4 px margin. Because the crop is resized, a single template works at any pothole size; no image pyramid is needed.
- **Procedural templates:** a dark elliptical bowl with a brighter rim, at aspect ratios 1.0, 0.7 and 0.5, rotated 0° and 90°, with a Gaussian blur.
- **Learned templates:** `tools/pothole600.py templates` cuts out every labelled training pothole with the **same crop rule**, z-normalises each crop, and clusters them with **k-means** (k = 12, k-means++ initialisation). The 12 cluster centres are saved as `templates/p600_XX.png`. Any image placed in `templates/` is loaded automatically.

### 5.6 Crack detection (`detector.py`)

1. **Black-hat** morphology (closing − image) with a 0.2 m ellipse picks out thin dark structures narrower than the kernel.
2. Pixels outside the road mask and near detected potholes are suppressed. The rest are thresholded at 22.
3. **Thin-component filter:** a component is kept only if its longest side is ≥ 0.6 × the minimum crack length **and** its area is < 0.35 × (longest side)², which means it is line-like.
4. **HoughLinesP** (threshold 12, minimum length 0.3 m, maximum gap 6 px) returns crack segments.
5. **Grid clustering:** segments are binned into 1 m cells. Cells with ≥ 0.6 m of total crack length are active, and adjacent active cells are merged by connected components.
6. **Linear vs. alligator:** orientation coherence is R = |mean(e^{2iθ})| over the segment angles θ. R = 1 means all segments are parallel. **Alligator cracking** if there are ≥ 4 segments and R < 0.55; otherwise a **linear crack**.
7. **Confidence** = min(1, length/1.5 m). **Severity**: 0.2 (crack) or 0.55 (alligator), plus 0.1 per metre, capped at 1.

### 5.7 Speed-bump and joint detection

On the BEV edge map, `HoughLinesP` looks for long lines that span at least 60% of the BEV width and are within 8° of horizontal. Lines are grouped when they are within 0.6 m of each other along the road. A group with **at least two parallel edges** (the front and back of a raised band) is a bump or transverse joint.

### 5.8 Visual odometry (`odometry.py`)

Without GPS, the vehicle still needs to know how far it has travelled.

1. **Shi-Tomasi corners** (`goodFeaturesToTrack`: up to 200, quality 0.01, minimum distance 7) are found on the near two-thirds of the BEV. The far third is smeared by the IPM.
2. **Pyramidal Lucas-Kanade** optical flow (21×21 window, 3 pyramid levels) tracks them into the next frame.
3. In a metric top-down view, driving forward moves the road **straight down**. So **speed = median(dy) × mpp / dt**.
4. **Robustness:** outliers with |dy − median| > max(2, 3·MAD) are rejected, at least 8 inliers are required, speed is clamped to 0–45 m/s, and an exponential moving average (0.7) smooths it.
5. Distance travelled: `odometer += speed · dt`.

On the synthetic benchmark this measured **299.1 m against 300 m true (0.3% error)**.

---

## 6. Localisation: putting detections on the map

### 6.1 Local coordinate frame

GPS gives latitude and longitude. Geometry is done in a local **East-North (ENU) tangent plane** in metres (equirectangular approximation, accurate to centimetres over a few km):

```
e = Δlon (rad) · R · cos(lat0)       n = Δlat (rad) · R        R = 6 378 137 m
```

### 6.2 Vehicle pose and hazard position

`Pose = (e, n, heading, lat, lon)`, with heading in radians clockwise from north. A detection X metres to the right and Z metres ahead is placed at:

```
e_h = e + Z·sin(h) + X·cos(h)
n_h = n + Z·cos(h) − X·sin(h)
```

### 6.3 Position sources (all implement `pose(odometer_m, t_s)`)

| Class | When | How |
|---|---|---|
| `RouteLocalizer` | **Mock demo (no GPS)** | Interpolates along a route polyline at the odometry distance. Heading comes from the points 3 m behind and ahead. |
| `CsvGpsLocalizer` | Video plus a recorded GPS log (`t,lat,lon`) | Interpolates by video time. Heading comes from positions ±1 s apart. |
| `NmeaSerialLocalizer` | Live GPS receiver (`--gps-serial COM5`) | A background thread parses `$xxRMC` sentences (status, lat/lon in degrees-minutes, course). Between 1 Hz fixes, position is **dead-reckoned** with visual odometry along the course. |

Routes are GeoJSON LineStrings. You can draw one at geojson.io along a real road. Without one, `procedural_route()` builds a plausible road (straight, gentle bend, straight) from `--origin`.

---

## 7. The digital twin (`twin.py`)

The twin is the persistent, geo-referenced model of the road.

### 7.1 Tracking in world coordinates

Instead of tracking objects in the image (which move and change scale), every detection is converted to **world coordinates** first and associated there:

1. **Association:** each detection is matched to the nearest existing hazard of the same group within a **merge radius**: pothole 1.0 m, crack 1.5 m, alligator 2.0 m, bump 2.5 m. The groups are pothole, crack (linear + alligator) and bump.
2. **Update:** the position is refined by a running mean (weight 1/(hits+1)). Confidence, severity and size keep their maximum. The hazard's kind is decided by majority vote of the kinds it was seen as.
3. **New candidate:** if nothing is within the merge radius, a new hazard starts.
4. **Confirmation (temporal voting):** a hazard is **confirmed** after `min_hits = 6` sightings. At 36 km/h a hazard stays in view for about 36 frames; at 100 km/h for about 13.
5. **Expiry:** unconfirmed candidates not seen for 20 frames are deleted. This removes single-frame false positives.

### 7.2 Road-segment condition

The travelled road is split into **25 m segments** by chainage (distance along the road). Each segment's score is

```
score = Σ over confirmed hazards in the segment of  w(kind) · (0.5 + severity)
w: pothole 3.0, alligator 2.0, crack 1.0, bump 0.5
good < 1 ≤ fair < 4 ≤ poor < 8 ≤ bad
```

### 7.3 Export format (`twin.geojson`)

A GeoJSON `FeatureCollection` with three layers:

- `segment`: LineString of the travelled path per segment, with its `condition`
- `hazard`: Point with `id, kind, confidence, severity, size_m2, hits, chainage_m`
- `vehicle`: Point with `heading_deg`

The file is written atomically (to a temporary file, then `os.replace`, retried if Windows has it locked) every 10 frames and at the end. It opens directly in QGIS, geojson.io or any GIS tool.

---

## 8. Visualisation and the live web map

### 8.1 OpenCV dashboard (`viz.py`)

- **Camera view:**
  - IPM trapezoid (grey)
  - lane lines (green)
  - detection outlines projected back from the BEV with H⁻¹, coloured by kind and labelled with confidence and distance
- **Bird's-eye inset:** CLAHE BEV with the crack map in orange and detection outlines. Toggle with `b`.
- **Twin panel** (heading-up, 2.6 px/m, vehicle near the top so the mapped road trails behind it):
  - road band drawn from the route, travelled path coloured by segment condition
  - hazard icons: filled circle for a pothole, cross for a crack, bar for a bump; hollow when not yet confirmed
  - header with lat/lon, speed, chainage and position source
  - hazard counts, condition legend and a 10 m scale bar
- **Status bar:** frame number, processing FPS, detections in this frame, hazards in the twin.
- **Keys:** `q` quit, `space` pause, `b` toggle inset. `--save` records the dashboard as MP4.

### 8.2 Browser map (`webmap.py`)

- A local `ThreadingHTTPServer` serves the output folder on `127.0.0.1:8765`, trying the next 9 ports if that one is busy. It sends `Cache-Control: no-store`.
- `map.html` (Leaflet + OSM tiles) fetches `twin.geojson` **every second** and redraws:
  - segments coloured by condition
  - hazard markers sized by severity, with popups (kind, confidence, severity, size, hits, chainage, coordinates)
  - the vehicle marker, with a "follow vehicle" toggle
- The map tiles need an internet connection. The OpenCV dashboard works offline.

---

## 9. Synthetic test-video generator (`tools/make_synthetic_video.py`)

Real annotated dash-cam video with exact hazard positions is hard to get. The generator renders a **physically consistent** dash-cam video where every hazard's position is known exactly, so the whole pipeline (detection, odometry, mapping) can be scored.

### 9.1 Camera model

The camera is a pinhole with 1280×720 resolution, focal length f = 1000 px, height 1.5 m and a pitch of 8° downwards. Each image pixel's viewing ray is rotated by the pitch and intersected with the ground plane:

```
xn = (u − cx)/f,  yn = (v − cy)/f
Yr = yn·cos θ + sin θ,   Zr = −yn·sin θ + cos θ
t  = h / Yr   →   X = t·xn,  Z = t·Zr        (pixels with Yr > 0 see the ground)
```

These per-pixel (X, Z) maps are computed once. Each frame is then a `cv2.remap` of a long top-down road texture, shifted by the distance travelled.

### 9.2 Road texture (40 px/m)

- **Asphalt:** grey level 95 + fine Gaussian noise (σ 9) + coarse cubic-upsampled noise (σ 7) for stains and wear.
- **Road layout:** shoulders tinted green; solid edge lines at ±5.25 m; dashed lane lines at ±1.75 m (3 m dash, 6 m gap).
- **Hazards (ground truth):**
  - 12 potholes: irregular polar-harmonic outlines, radius 0.18–0.55 m, dark textured interior with depth shading, broken bright rim
  - 6 linear cracks: random walks 1.5–4 m long
  - 3 alligator patches: 16 short random walks each
  - 2 speed bumps: bright 0.5 m band with a dark leading edge
- **Distractors** that must *not* be reported: 3 smooth repair patches and 3 soft oil stains.

### 9.3 Realism details

- **Anti-aliasing:** four mip levels (box blurs of 1×3, 3×9, 5×27) are chosen per pixel by the ground footprint `PPM·Z²/(f·h)`. A real lens averages over that footprint too.
- **Atmosphere:** distance haze from 40 m, and a sky gradient above the horizon.
- **Camera artefacts:** a vertical jitter of 1.5 px at 1.3 Hz plus noise to simulate suspension vibration, 3×3 lens blur, and σ = 2 sensor noise.
- **Drive:** 36 km/h, 30 FPS, 300 m, 900 frames.

### 9.4 Outputs

- `synthetic_road.mp4`
- `synthetic_calib.json`: the exact IPM points, from projecting the ground rectangle ±2.75 m × 4.5–16.5 m
- `synthetic_route.geojson`
- `synthetic_gt.json`: each hazard's chainage `s`, lateral offset `x` and kind

---

## 10. The Pothole-600 dataset integration

### 10.1 The dataset

`pothole600/{training, validation, testing}/{rgb, label, tdisp}/NNNN.png` holds **240 / 180 / 180** images.

| Folder | Contents |
|---|---|
| `rgb` | 400×400 near top-down road crops (stereo camera images). Every crop contains at least one pothole, covering 1.5–76% of its pixels. |
| `label` | Binary pothole mask (0/255) |
| `tdisp` | **Transformed disparity**: stereo depth with the road plane flattened, stored as a **JET colour image** |

### 10.2 Decoding the disparity (`dataset.py`)

`tdisp` is a picture of the data, not the data itself. It is turned back into a scalar 0–255 by inverting the JET colour map:

1. The 256 JET colours are generated with `cv2.applyColorMap`.
2. A lookup table is built over the whole 64×64×64 colour cube (6 bits per channel), mapping each cell to the nearest JET index. This takes about 2 s once and is cached.
3. Each pixel is decoded with a single array lookup. The mean decoding error is 3.4 out of a possible 441.

### 10.3 Cue analysis (why depth matters)

| Cue (median over images) | Pothole | Road |
|---|---|---|
| Disparity deviation | 68 | 4 |
| Darkness | 28.6 | about 0 |
| Chroma (colour) deviation | 1.6 | 1.3 (useless) |

### 10.4 Patch preparation (`PatchPrep`)

Grey, resized to a **200×200 working resolution** (INTER_AREA), CLAHE 4×4, median 3, robust contrast normalisation (section 5.2). Disparity is resized the same way.

### 10.5 Survey replay (`survey.py`)

The dataset has no video or GPS, so each crop is treated as **one road sample every `--spacing` m** (default 5 m) along a mock route.

- **Assumed crop size:** `--patch-m` (default 1.5 m), so mpp = 1.5/200.
- **Pixel to ground:** a pixel (cx, cy) is at `X = (cx − 100)·mpp`, `Z = (200 − cy)·mpp + 2.0 m`, then goes through `pose.offset` like the video pipeline.
- **Confirmation:** `min_hits = 1`, because each crop is seen exactly once.
- **Ground truth:** the label masks are placed on the map the same way, so the final map is scored.
- **Display:** the crop with ground truth (green) and detection (red) outlines, a depth-residual inset in rgbd mode, and the twin panel.

---

## 11. Tuning and evaluation methodology

### 11.1 Synthetic video: map-level evaluation (`evaluate.py`)

1. Ground-truth hazards (chainage s, offset x) are converted to world coordinates through the same route.
2. Confirmed twin hazards and ground-truth hazards are matched **one to one, greedily, by distance**, within 4 m.
3. Reported per class: precision, recall and mean location error. Distractors (patches, stains) are never positives, so reporting them counts against precision.

### 11.2 Pothole-600: segmentation metrics (`tools/pothole600.py`)

- **Pixel level** (accumulated over the split): precision, recall, F1, IoU, plus mean IoU per image.
- **Instance level** (each connected region):
  - a **ground-truth pothole is found** if at least 30% of its pixels are detected (components under 50 px ignored)
  - a **detected region is correct** if at least 30% of it lies on ground-truth pothole

### 11.3 Tuning protocol (avoiding test leakage)

1. **Templates** are learned from **training** masks only.
2. **Grid search** on **training**. Image-only mode has 216 configurations: median window {41, 61, 91}, seed strength, contrast gate, rim gate, solidity gate, score threshold. Depth mode has 288. Ranked by pixel F1.
3. The **top 10** are re-scored on **validation**, and the best is saved to `params/pothole600_<mode>.json`.
4. **Testing** is evaluated **once**, with the chosen parameters.
5. **Efficiency:** the expensive cue maps (`features`) are computed once per median-window size and reused across all threshold combinations (`decide`).

### 11.4 How the detector was refined (summary)

| Step | Change | Evidence |
|---|---|---|
| 1 | Hysteresis seeds, contrast and rim gates | False regions had contrast ≈ 23 and rim edges ≈ 0, against 44 and 0.21 for real potholes |
| 2 | Solidity gate, score threshold 0.75, 6-frame confirmation | Remaining false positives were alligator patches and stains seen in only 4–6 frames |
| 3 | Wider tuning grid | The first grid excluded small median windows and was worse than the untuned default |
| 4 | Contrast normalisation | Testing images have about 60% more contrast (σ 25 vs. 16) and coarser aggregate |

---

## 12. Results

### 12.1 Synthetic dash-cam benchmark (300 m, 36 km/h, 900 frames)

| Class | Ground truth | Predicted | Precision | Recall | Location error |
|---|---|---|---|---|---|
| Pothole | 12 | 12 | 0.75 | 0.75 | 0.26 m |
| Crack (incl. alligator) | 9 | 9 | 1.00 | 1.00 | 1.13 m |
| Bump | 2 | 2 | 1.00 | 1.00 | 0.03 m |

- **Odometry:** 299.1 m measured vs. 300 m true, with no GPS.
- **Speed:** about 35–42 FPS for the full pipeline on a laptop CPU.
- **Distractors:** repair patches are always rejected.
- **Misses:** the potholes the system misses are the smallest ones (radius 0.23–0.31 m).

### 12.2 Pothole-600, held-out **testing** split

| Mode | Pixel P | Pixel R | Pixel F1 | IoU | Potholes found | Detections correct | Time per image |
|---|---|---|---|---|---|---|---|
| Image only (rgb) | 0.80 | 0.07 | 0.14 | 0.07 | 0.26 | 0.88 | ~6 ms |
| Image + depth (rgbd) | 0.40 | 0.91 | 0.56 | 0.39 | **0.985** | 0.82 | ~10 ms |

On validation, image-only reaches pixel F1 0.60 and image + depth 0.54.

**Survey replay of the testing split onto the map (rgbd):** 186 of 196 potholes mapped within 1 m, with mean location error 0.08 m. In image-only mode, 51 of 196 are mapped. The 1 m match radius is generous, so the instance-level numbers above are the stricter measure.

### 12.3 What the results mean

1. **Image-only detection doesn't generalise across road types.** It works on validation-like roads, but the testing split has coarse aggregate, large shallow potholes that are *not darker* than the road, and different lighting. Darkness and rim cues aren't reliable there, and contrast normalisation doesn't fix it.
2. **Depth is what transfers.** A plane-fit residual on stereo disparity finds 98.5% of test potholes. Its remaining weakness is **over-segmentation**: the detected region spreads beyond the rim, which keeps pixel precision at about 0.40.
3. **For deployment, a stereo camera** (or another depth source) is worth more than any further image-only tuning.

---

## 13. How to run everything

### 13.1 Setup

```bash
pip install -r requirements.txt          # opencv-python, numpy  (+ pyserial for live GPS)
```

### 13.2 Mock demo (synthetic video, no GPS)

```bash
python tools/make_synthetic_video.py                 # creates data/ (about 1 minute)
python demo.py --video data/synthetic_road.mp4 --calib data/synthetic_calib.json \
               --route data/synthetic_route.geojson --gt data/synthetic_gt.json
```

### 13.3 Real footage

```bash
python tools/calibrate.py --video myroad.mp4 --width 3.5 --near 6 --far 18 --out my_calib.json
python demo.py --video myroad.mp4 --calib my_calib.json --route my_road.geojson   # mock GPS
python demo.py --video myroad.mp4 --calib my_calib.json --gps-csv track.csv       # GPS log
python demo.py --camera 0 --calib my_calib.json --gps-serial COM5                 # live
```

### 13.4 Pothole-600

```bash
python tools/pothole600.py templates                     # learn 12 templates
python tools/pothole600.py tune --mode rgb               # about 6 min
python tools/pothole600.py tune --mode rgbd              # about 8 min
python tools/pothole600.py eval --mode rgbd --split testing --vis out/p600_vis_rgbd
python demo.py --dataset pothole600 --split testing --disparity
```

### 13.5 Useful `demo.py` options

| Option | Effect |
|---|---|
| `--speed-kmh V` | Use a constant speed instead of visual odometry |
| `--origin LAT,LON` | Start of the procedural route |
| `--save file.mp4` | Record the dashboard |
| `--headless` / `--no-web` | No window / no browser map (for batch runs) |
| `--max-frames N` | Stop early |
| `--out DIR` | Output folder (default `out/`) |
| `--split`, `--disparity`, `--patch-m`, `--spacing`, `--replay-fps` | Dataset replay settings |

### 13.6 Outputs

- `out/twin.geojson`: the digital twin
- `out/map.html`: the live map page
- `out/*.mp4`: recordings
- `out/p600_vis_*/`: per-image overlays (green = ground truth, red = detection)
- `out/pothole600_<split>_gt.json`: geo-located ground truth from the replay

---

## 14. Configuration reference

All parameters are in `roadtwin/config.py` (video pipeline). Any of them can be overridden by a calibration JSON.

| Group | Parameter | Default | Meaning |
|---|---|---|---|
| IPM | `src_norm` | trapezoid | Image points of the ground rectangle (normalised) |
| | `bev_width_m`, `bev_near_m`, `bev_far_m` | 5.5, 4.5, 16.5 | Ground rectangle size (m) |
| | `bev_mpp` | 0.025 | BEV resolution (m/px) |
| Potholes | `bg_kernel_m` | 1.5 | Median background window |
| | `dark_thresh` / `dark_strong` | 18 / 45 | Weak / seed darkness |
| | `min_contrast`, `min_rim_edges`, `min_solidity` | 38, 0.06, 0.75 | Hard gates |
| | `min_area_m2`, `max_area_m2`, `max_aspect` | 0.03, 3.0, 4.0 | Geometry gates |
| | `pothole_score_thresh` | 0.75 | Score cut-off |
| | `template_dir` | `templates` | Extra templates |
| Cracks | `crack_blackhat_m`, `crack_thresh` | 0.2, 22 | Thin dark structure extraction |
| | `crack_min_len_m`, `crack_cell_m`, `crack_min_total_m` | 0.3, 1.0, 0.6 | Hough and clustering |
| Bumps | `bump_min_span`, `bump_max_angle_deg` | 0.6, 8 | Horizontal-line criteria |
| Lanes | `lane_min_slope` | 0.35 | Reject near-horizontal lines |
| Odometry | `max_speed_mps` | 45 | Clamp |
| Twin | `min_hits` | 6 | Confirmation |
| | `merge_radius_m` | 1.0 / 1.5 / 2.0 / 2.5 | Pothole / crack / alligator / bump |
| | `candidate_ttl_frames`, `segment_len_m` | 20, 25 | Expiry, segment length |

**Tuned dataset parameters** (`params/`, 200 px working crops):

- **rgb:** `bg_kernel 91, dark_strong 55, min_contrast 25, min_rim_edges 0.03, min_solidity 0.5, score_thresh 0.6`
- **rgbd:** `bg_kernel 41, dark_strong 60, disp_strong 45, disp_weak 14, min_solidity 0.4, score_thresh 0.5`

---

## 15. Design decisions and rationale

| Decision | Why |
|---|---|
| **Detect in the metric bird's-eye view** | Removes perspective, so thresholds are in metres, pothole size doesn't depend on distance, and ground position comes free |
| **Classical CV, no deep learning** | Explainable, CPU-real-time, no training data needed to start, and a measurable baseline before scaling up |
| **Several weak cues fused into a score** | No single cue (darkness, edges, shape, template, circles) is reliable alone. Combining them with hard gates on the strongest ones cuts false positives. |
| **Hysteresis segmentation** | Keeps a pothole's full extent while rejecting faint blotches |
| **Track in world coordinates, not the image** | Hazards are static in the world, so association is a simple radius test. The same code works for video, GPS and dataset replay. |
| **Temporal confirmation (N hits)** | Cheap and effective removal of single-frame false positives |
| **Visual odometry for the mock demo** | Gives a credible no-GPS demo, and doubles as dead reckoning between GPS fixes |
| **One segmenter shared by video and dataset** | Tuning on real labelled potholes improves the in-vehicle detector directly |
| **Two-stage segmenter (features / decide)** | Makes large parameter sweeps fast |
| **Train / validation / test discipline** | Test results are honest. That is how the image-only generalisation gap was discovered. |
| **Synthetic generator with ground truth** | End-to-end scoring (detection, odometry, map) before any field data exists |
| **GeoJSON output** | Opens in standard GIS tools and is easy to send to a server |

---

## 16. Limitations

1. **Image-only cues fail on some road types:** coarse aggregate, shallow or same-coloured potholes, hard shadows, wet patches and night scenes.
2. **Over-segmentation with depth:** detected regions are larger than the labelled pothole (pixel precision about 0.40).
3. **Calibration-dependent:** a wrong IPM gives wrong sizes, positions and odometry speed. Recalibrate for every camera mount.
4. **Flat-road assumption:** the IPM assumes a planar road; steep grades, crests and camera pitch changes under braking add error.
5. **Lane fit:** the displayed lane line can be pulled by the road edge line. This affects display only.
6. **Dataset scale is assumed** (`--patch-m`), because Pothole-600 gives no metric scale.
7. **Synthetic results are optimistic:** the synthetic potholes were tuned against. Real-road performance must be measured on real data.
8. **The web map needs internet** for OSM tiles.

---

## 17. Roadmap for scaling up

1. **Collect real data:** dash-cam video plus a phone GPS log (`--gps-csv`), with a few km labelled as {kind, chainage, offset}. Score it with the existing evaluator.
2. **Add depth:** a stereo camera, or monocular depth estimation, to bring the rgbd cue to the vehicle. This is the largest measured gain.
3. **Learned verification:** keep the classical stage as a high-recall proposal generator and add a small classifier on candidate patches (HOG + SVM, then a small CNN). Pothole-600 and RDD2022 are training sources.
4. **Segmentation network:** replace the segmenter with a learned model once there is enough data. IPM, odometry, the twin and the map are unchanged.
5. **Better localisation:** fuse GPS, IMU and visual odometry with a Kalman filter, and snap to the road network (map matching).
6. **Fleet-scale twin:** a server aggregating twins from many vehicles, with multi-pass voting, hazard ageing and repair detection. IMU vibration can be added as a ride-quality (IRI-like) signal.

---

## 18. Glossary

| Term | Meaning |
|---|---|
| **IPM** | Inverse Perspective Mapping: warping the camera image into a top-down view of the road plane |
| **BEV** | Bird's-Eye View, the result of IPM |
| **Homography** | A 3×3 projective transform relating two views of a plane |
| **CLAHE** | Contrast-Limited Adaptive Histogram Equalisation |
| **Otsu** | Automatic threshold that best separates a two-class histogram |
| **Black-hat** | Morphological closing minus the image; highlights thin dark structures |
| **Hough transform** | Voting method for finding lines (`HoughLinesP`) or circles (`HoughCircles`) |
| **NCC** | Normalised cross-correlation, the template-matching score (`TM_CCOEFF_NORMED`) |
| **Solidity** | Region area divided by its convex-hull area |
| **Hysteresis** | Two-threshold segmentation: weak regions are kept only if they touch a strong seed |
| **Lucas-Kanade** | Optical-flow method that tracks feature points between frames |
| **Visual odometry** | Estimating motion from camera images |
| **Disparity** | Horizontal pixel shift between stereo images; inversely related to depth |
| **MAD** | Median absolute deviation, a robust spread estimate (σ ≈ 1.4826·MAD) |
| **ENU** | Local East-North-Up coordinate frame in metres |
| **Chainage** | Distance along the road from the start of the survey |
| **NMEA / RMC** | Standard GPS serial sentences; RMC carries position, speed and course |
| **GeoJSON** | JSON format for geographic features |
| **Digital twin** | A persistent virtual model of the physical road, kept up to date from sensor data |
| **IoU / F1** | Intersection-over-union, and the harmonic mean of precision and recall |
