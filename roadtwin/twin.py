"""Road digital twin: a persistent, geo-referenced map of road hazards and condition.

Detections are tracked directly in world coordinates: each new detection is merged
with an existing hazard of a compatible kind within a merge radius (its position is
refined by running average), otherwise it starts a new candidate. A candidate becomes
a confirmed hazard after `min_hits` sightings; unconfirmed candidates that stop being
seen expire. This temporal voting suppresses single-frame false positives.
"""
import json
import math
import os
from dataclasses import dataclass, field

KIND_GROUP = {"pothole": "pothole", "crack": "crack", "alligator": "crack", "bump": "bump"}
SEVERITY_WEIGHT = {"pothole": 3.0, "alligator": 2.0, "crack": 1.0, "bump": 0.5}
CONDITIONS = [(0.0, "good"), (1.0, "fair"), (4.0, "poor"), (8.0, "bad")]


@dataclass
class Hazard:
    id: int
    kind: str
    e: float
    n: float
    hits: int = 1
    conf: float = 0.0
    severity: float = 0.0
    size_m2: float = 0.0
    first_frame: int = 0
    last_frame: int = 0
    odometer: float = 0.0
    confirmed: bool = False
    kinds: dict = field(default_factory=dict)


class RoadTwin:
    def __init__(self, cfg, frame):
        self.cfg = cfg
        self.frame = frame              # LocalFrame for lat/lon conversion
        self.hazards = []
        self.trajectory = []            # (e, n, odometer)
        self._next = 1

    def update(self, dets_world, frame_idx, pose, odometer):
        if not self.trajectory or odometer - self.trajectory[-1][2] > 1.0:
            self.trajectory.append((pose.e, pose.n, odometer))
        for kind, e, n, conf, sev, size, Z in dets_world:
            group = KIND_GROUP[kind]
            r = self.cfg.merge_radius_m.get(kind, 1.5)
            best, bd = None, r
            for h in self.hazards:
                if KIND_GROUP[h.kind] != group:
                    continue
                d = math.hypot(h.e - e, h.n - n)
                if d < bd:
                    best, bd = h, d
            if best is None:
                best = Hazard(self._next, kind, e, n, conf=conf, severity=sev, size_m2=size,
                              first_frame=frame_idx, last_frame=frame_idx, odometer=odometer + Z)
                best.kinds[kind] = 1
                self._next += 1
                self.hazards.append(best)
            else:
                w = 1.0 / (best.hits + 1)
                best.e += w * (e - best.e)
                best.n += w * (n - best.n)
                best.hits += 1
                best.conf = max(best.conf, conf)
                best.severity = max(best.severity, sev)
                best.size_m2 = max(best.size_m2, size)
                best.last_frame = frame_idx
                best.kinds[kind] = best.kinds.get(kind, 0) + 1
                best.kind = max(best.kinds, key=best.kinds.get)
            if best.hits >= self.cfg.min_hits:
                best.confirmed = True
        ttl = self.cfg.candidate_ttl_frames
        self.hazards = [h for h in self.hazards if h.confirmed or frame_idx - h.last_frame <= ttl]

    @property
    def confirmed(self):
        return [h for h in self.hazards if h.confirmed]

    def segment_conditions(self):
        """Condition label per segment of travelled road, from confirmed hazard severities."""
        L = self.cfg.segment_len_m
        score = {}
        for h in self.confirmed:
            k = int(h.odometer // L)
            score[k] = score.get(k, 0.0) + SEVERITY_WEIGHT[h.kind] * (0.5 + h.severity)
        out = {}
        for k in range(int(self.trajectory[-1][2] // L) + 1 if self.trajectory else 0):
            s = score.get(k, 0.0)
            out[k] = [lbl for th, lbl in CONDITIONS if s >= th][-1]
        return out

    def to_geojson(self, pose=None):
        feats = []
        conds = self.segment_conditions()
        L = self.cfg.segment_len_m
        segs = {}
        for e, n, odo in self.trajectory:
            segs.setdefault(int(odo // L), []).append((e, n))
        prev_last = None
        for k in sorted(segs):
            pts = ([prev_last] if prev_last else []) + segs[k]
            prev_last = segs[k][-1]
            if len(pts) < 2:
                continue
            coords = [list(reversed(self.frame.to_latlon(e, n))) for e, n in pts]
            feats.append({"type": "Feature",
                          "properties": {"layer": "segment", "segment": k,
                                         "condition": conds.get(k, "good")},
                          "geometry": {"type": "LineString", "coordinates": coords}})
        for h in self.confirmed:
            lat, lon = self.frame.to_latlon(h.e, h.n)
            feats.append({"type": "Feature",
                          "properties": {"layer": "hazard", "id": h.id, "kind": h.kind,
                                         "confidence": round(h.conf, 2),
                                         "severity": round(h.severity, 2),
                                         "size_m2": round(h.size_m2, 2), "hits": h.hits,
                                         "chainage_m": round(h.odometer, 1)},
                          "geometry": {"type": "Point", "coordinates": [lon, lat]}})
        if pose is not None:
            feats.append({"type": "Feature",
                          "properties": {"layer": "vehicle",
                                         "heading_deg": round(math.degrees(pose.heading), 1)},
                          "geometry": {"type": "Point", "coordinates": [pose.lon, pose.lat]}})
        return {"type": "FeatureCollection", "features": feats}

    def save(self, path, pose=None):
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.to_geojson(pose), f)
        for _ in range(5):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:     # Windows: file momentarily open by the web server
                pass
