"""Vehicle localisation sources.

All sources expose  pose(odometer_m, t_s) -> Pose  in a local East-North (metres) frame.

  RouteLocalizer      mock GPS for demos: vehicle position = distance travelled
                      (from visual odometry) along a known route polyline.
  CsvGpsLocalizer     replay a GPS log recorded alongside the video (t,lat,lon).
  NmeaSerialLocalizer live GPS receiver on a serial port (NMEA RMC sentences),
                      dead-reckoned with visual odometry between fixes.
"""
import csv
import json
import math
import threading
import time
from dataclasses import dataclass

import numpy as np

R_EARTH = 6378137.0


class LocalFrame:
    """Equirectangular tangent-plane approximation; accurate to cm over a few km."""

    def __init__(self, lat0, lon0):
        self.lat0, self.lon0 = lat0, lon0
        self.k = math.cos(math.radians(lat0))

    def to_enu(self, lat, lon):
        return (math.radians(lon - self.lon0) * R_EARTH * self.k,
                math.radians(lat - self.lat0) * R_EARTH)

    def to_latlon(self, e, n):
        return (self.lat0 + math.degrees(n / R_EARTH),
                self.lon0 + math.degrees(e / (R_EARTH * self.k)))


@dataclass
class Pose:
    e: float
    n: float
    heading: float      # radians, clockwise from north
    lat: float
    lon: float

    def offset(self, X, Z):
        """Ground point X metres right and Z metres ahead of the vehicle -> (e, n)."""
        s, c = math.sin(self.heading), math.cos(self.heading)
        return self.e + Z * s + X * c, self.n + Z * c - X * s


def load_route(path):
    """GeoJSON LineString (Feature/FeatureCollection) or JSON list of [lat, lon]."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return [tuple(p) for p in data]
    feats = data.get("features", [data])
    for ft in feats:
        geom = ft.get("geometry", ft)
        if geom.get("type") == "LineString":
            return [(lat, lon) for lon, lat in geom["coordinates"]]
    raise ValueError(f"No LineString in {path}")


def procedural_route(lat0, lon0, length_m=600.0, step=5.0):
    """A plausible road: straight, gentle left bend, straight. Used when no route is given."""
    frame = LocalFrame(lat0, lon0)
    pts, e, n, h = [], 0.0, 0.0, math.radians(70)
    for i in range(int(length_m / step) + 1):
        s = i * step
        pts.append(frame.to_latlon(e, n))
        if 150 < s < 300:
            h -= math.radians(0.5) * step / 5
        e += step * math.sin(h)
        n += step * math.cos(h)
    return pts


def save_route_geojson(pts, path):
    with open(path, "w") as f:
        json.dump({"type": "Feature", "properties": {"name": "route"},
                   "geometry": {"type": "LineString",
                                "coordinates": [[lon, lat] for lat, lon in pts]}}, f)


class _Polyline:
    def __init__(self, frame, latlon):
        self.frame = frame
        self.xy = np.array([frame.to_enu(lat, lon) for lat, lon in latlon])
        seg = np.diff(self.xy, axis=0)
        self.cum = np.concatenate([[0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))])

    @property
    def length(self):
        return float(self.cum[-1])

    def point(self, s):
        s = float(np.clip(s, 0, self.cum[-1]))
        return np.interp(s, self.cum, self.xy[:, 0]), np.interp(s, self.cum, self.xy[:, 1])

    def pose(self, s, look=3.0):
        e, n = self.point(s)
        e0, n0 = self.point(s - look)
        e1, n1 = self.point(s + look)
        heading = math.atan2(e1 - e0, n1 - n0)
        lat, lon = self.frame.to_latlon(e, n)
        return Pose(e, n, heading, lat, lon)


class RouteLocalizer:
    """Mock GPS: no satellite fix, position comes from odometry along a known road."""
    name = "mock route + visual odometry"

    def __init__(self, route_latlon):
        self.frame = LocalFrame(*route_latlon[0])
        self.route = _Polyline(self.frame, route_latlon)

    def pose(self, odometer_m, t_s):
        return self.route.pose(odometer_m)

    def route_enu(self):
        return self.route.xy


class CsvGpsLocalizer:
    """Replay a GPS track (CSV with columns t,lat,lon; t in seconds from video start)."""
    name = "GPS log replay"

    def __init__(self, path):
        rows = []
        with open(path) as f:
            for r in csv.DictReader(f):
                rows.append((float(r["t"]), float(r["lat"]), float(r["lon"])))
        rows.sort()
        self.t = np.array([r[0] for r in rows])
        self.frame = LocalFrame(rows[0][1], rows[0][2])
        self.xy = np.array([self.frame.to_enu(r[1], r[2]) for r in rows])

    def _at(self, t):
        return np.interp(t, self.t, self.xy[:, 0]), np.interp(t, self.t, self.xy[:, 1])

    def pose(self, odometer_m, t_s):
        e, n = self._at(t_s)
        e0, n0 = self._at(t_s - 1.0)
        e1, n1 = self._at(t_s + 1.0)
        h = math.atan2(e1 - e0, n1 - n0) if (e1 - e0) ** 2 + (n1 - n0) ** 2 > 0.25 else 0.0
        return Pose(e, n, h, *self.frame.to_latlon(e, n))

    def route_enu(self):
        return self.xy


class NmeaSerialLocalizer:
    """Live GPS on a serial port (needs `pip install pyserial`). Parses $xxRMC sentences."""
    name = "live GPS (NMEA)"

    def __init__(self, port, baud=9600):
        import serial  # optional dependency
        self.ser = serial.Serial(port, baud, timeout=1)
        self.frame = None
        self.fix = None          # (lat, lon, course_rad)
        self.fix_odo = None
        self.last_odo = 0.0
        self.lock = threading.Lock()
        threading.Thread(target=self._reader, daemon=True).start()

    @staticmethod
    def _deg(v, hemi):
        d = int(float(v) / 100)
        val = d + (float(v) - d * 100) / 60
        return -val if hemi in "SW" else val

    def _reader(self):
        while True:
            try:
                line = self.ser.readline().decode("ascii", "ignore").strip()
            except Exception:
                time.sleep(0.5)
                continue
            p = line.split(",")
            if len(p) > 8 and p[0].endswith("RMC") and p[2] == "A":
                lat, lon = self._deg(p[3], p[4]), self._deg(p[5], p[6])
                course = math.radians(float(p[8])) if p[8] else None
                with self.lock:
                    if self.frame is None:
                        self.frame = LocalFrame(lat, lon)
                    if course is None and self.fix:
                        course = self.fix[2]
                    self.fix = (lat, lon, course or 0.0)
                    self.fix_odo = self.last_odo

    def pose(self, odometer_m, t_s):
        with self.lock:
            self.last_odo = odometer_m
            if self.fix is None:
                return None
            lat, lon, h = self.fix
            e, n = self.frame.to_enu(lat, lon)
            d = odometer_m - self.fix_odo          # dead-reckon since last fix
            e, n = e + d * math.sin(h), n + d * math.cos(h)
            return Pose(e, n, h, *self.frame.to_latlon(e, n))

    def route_enu(self):
        return None
