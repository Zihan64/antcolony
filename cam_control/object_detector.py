#!/usr/bin/env python3
"""
object_detector.py

Detect objects WITHOUT AprilTags, class-agnostically, and locate them on the floor plane built
by arena_map.py.

  pip install ultralytics          (first use downloads the weights)

What "class-agnostic" means here
  Nothing is predefined: no list of things to look for, no text prompts, no stored
  appearance of objects seen before. Every frame, a prompt-free YOLOE model proposes regions
  that look like distinct objects, and every proposal is just "an object". Whether it counts
  is decided by geometry on the floor plane, not by what the model thinks it is:
    * the box must touch the floor at a plausible place (its contact edge projects onto the
      plane within range) and have a plausible footprint;
    * boxes covering a large part of the image (the scene itself, the floor) are dropped;
    * boxes that are really an AprilTag are dropped (the tag is tracked far more precisely);
    * overlapping proposals are merged (one object, one entry, whatever the label);
    * a new object must show up in 2 detector cycles before it is reported, so one-frame
      flickers never become obstacles.
  Tracks are matched by floor position only. A never-seen object is picked up the first time
  it appears; the model's own class label is discarded.

Models (prompt-free, no text encoder needed)
  yoloe-26s-seg-pf.pt   default: segments everything that looks like an object
  yoloe-26n-seg-pf.pt   smaller and faster
  your_model.pt         any Ultralytics detection / segmentation model (labels are ignored)

From image box to floor coordinates
  The floor plane is known, so a pixel where an object touches the floor maps to an exact
  (x, y). The contact edge is the lowest part of the segmentation mask (or the box bottom).
  Projected onto the plane it gives the object's near edge and its width; the far side is not
  visible, so depth is assumed equal to width. Height comes from the top of the box, measured
  on the vertical line above the contact point. The footprint polygon (near edge + assumed
  depth) is what A* avoids.

Limits (honest)
  * Position is the floor contact point plus half the assumed depth: good to ~1-3 cm for
    compact objects, rougher for long objects seen end-on.
  * Things not standing on the floor (on a table, held in a hand) get wrong floor positions.
  * Flat things (paper, tape) are proposed too; they become low obstacles.
  * Detection runs at the model's own rate (CPU: a few per second; GPU: real time); positions
    use the newest detections, which can be a few hundred ms old.
"""

import math
import threading
import time

import cv2
import numpy as np

MODEL_CHOICES = [
    ("yoloe-26s-seg-pf.pt", "YOLOE-26s prompt-free: any object, no class list (default)"),
    ("yoloe-26n-seg-pf.pt", "YOLOE-26n prompt-free: any object, faster"),
]
DET_ID_BASE = 1000          # detected objects get ids 1000+ (tags use 0-101)
MAX_BOX_FRAC = 0.25         # a box covering more of the image than this is the scene, not an object


class ObjectDetector:
    """Runs a prompt-free YOLO model on the newest submitted frame in a background thread.
    Every detection is a generic object; class labels are discarded."""

    def __init__(self, model_name, conf=0.15, imgsz=640, device=None):
        self.model_name, self.conf, self.imgsz, self.device = model_name, conf, imgsz, device
        self.model = None
        self.status = "loading model…"
        self.error = None
        self.infer_ms = 0.0
        self._frame = None
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._latest = (0.0, [])
        self.running = True
        self._reconfig = None
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    # ---- control
    def submit(self, frame, stamp):
        with self._lock:
            self._frame = (frame, stamp)
        self._event.set()

    def latest(self):
        with self._lock:
            return self._latest

    def configure(self, conf=None, imgsz=None):
        """Change confidence / image size without reloading the model."""
        self._reconfig = (conf, imgsz)

    def stop(self):
        self.running = False
        self._event.set()

    # ---- worker
    def _load(self):
        from ultralytics import YOLO, YOLOE
        return YOLOE(self.model_name) if "yoloe" in self.model_name.lower() else YOLO(self.model_name)

    @staticmethod
    def _cap_cpu_threads():
        """The tag detector already uses several cores. Torch (ultralytics sets cores - 1 when
        the predictor is set up) then oversubscribes the CPU and inference got ~5-10x slower
        (1.3 s instead of ~0.15-0.3 s per frame). Use half the cores. Called before every
        predict because ultralytics resets the count while setting up its predictor."""
        import os
        import torch
        n = max(1, (os.cpu_count() or 4) // 2)
        if not torch.cuda.is_available() and torch.get_num_threads() != n:
            torch.set_num_threads(n)

    def _loop(self):
        try:
            self.model = self._load()
            self.status = "ready"
        except Exception as e:      # missing package, no internet for weights, bad file ...
            self.error = f"{type(e).__name__}: {e}"
            self.status = "failed to load"
            return
        while self.running:
            self._event.wait(0.5)
            self._event.clear()
            if not self.running:
                break
            if self._reconfig is not None:
                conf, imgsz = self._reconfig
                self._reconfig = None
                if conf is not None:
                    self.conf = conf
                if imgsz is not None:
                    self.imgsz = imgsz
            with self._lock:
                item, self._frame = self._frame, None
            if item is None:
                continue
            frame, stamp = item
            t0 = time.perf_counter()
            try:
                # agnostic_nms: overlapping boxes are merged whatever label the model gave them
                kw = dict(conf=self.conf, imgsz=self.imgsz, verbose=False, agnostic_nms=True)
                if self.device:
                    kw["device"] = self.device
                self._cap_cpu_threads()
                r = self.model.predict(frame, **kw)[0]
                dets = self._parse(r, frame.shape)
            except Exception as e:
                self.error = f"{type(e).__name__}: {e}"
                continue
            self.infer_ms = (time.perf_counter() - t0) * 1e3
            self.status = f"{len(dets)} proposals, {self.infer_ms:.0f} ms/frame"
            with self._lock:
                self._latest = (stamp, dets)

    @staticmethod
    def _parse(r, shape):
        out = []
        if r.boxes is None:
            return out
        img_area = float(shape[0] * shape[1])
        polys = r.masks.xy if getattr(r, "masks", None) is not None else None
        for k, (b, p) in enumerate(zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())):
            x1, y1, x2, y2 = [float(v) for v in b]
            if (x2 - x1) * (y2 - y1) > MAX_BOX_FRAC * img_area:
                continue
            d = {"name": "object", "conf": float(p), "box": (x1, y1, x2, y2)}
            poly = None
            if polys is not None and k < len(polys) and len(polys[k]) >= 3:
                poly = np.asarray(polys[k], np.float64)
            d["bottom"] = bottom_edge(poly, (x1, y1, x2, y2))
            d["poly"] = poly
            out.append(d)
        return out


def drop_tag_boxes(dets, tag_quads):
    """Remove proposals that are really an AprilTag: a box around a tag's center that is not
    much bigger than the tag itself. (A box much bigger than a tag inside it is a tagged
    object, e.g. a crate with a tag on top, and is kept.)"""
    if not tag_quads:
        return dets
    tags = [(np.asarray(q, np.float64).mean(0), cv2.contourArea(np.asarray(q, np.float32))) for q in tag_quads]
    keep = []
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        area = max((x2 - x1) * (y2 - y1), 1.0)
        if any(x1 <= c[0] <= x2 and y1 <= c[1] <= y2 and area < 4.0 * max(a, 1.0) for c, a in tags):
            continue
        keep.append(d)
    return keep


def bottom_edge(poly, box, bins=12):
    """Image points along the object's lowest outline (where it meets the floor): for each of
    `bins` vertical strips, the lowest point of the mask outline. Falls back to the box bottom."""
    x1, y1, x2, y2 = box
    if poly is None:
        xs = np.linspace(x1, x2, bins)
        return np.column_stack([xs, np.full(bins, y2)])
    # densify the outline (~2 px steps) so straight bottom edges are sampled, not just corners
    closed = np.vstack([poly, poly[:1]])
    dense = []
    for a_, b_ in zip(closed, closed[1:]):
        n = max(2, int(np.hypot(*(b_ - a_)) / 2))
        dense.append(a_ + np.linspace(0, 1, n, endpoint=False)[:, None] * (b_ - a_))
    poly = np.vstack(dense)
    edges = np.linspace(poly[:, 0].min(), poly[:, 0].max(), bins + 1)
    pts = []
    for a, b in zip(edges, edges[1:]):
        sel = poly[(poly[:, 0] >= a) & (poly[:, 0] <= b)]
        if len(sel):
            pts.append(sel[np.argmax(sel[:, 1])])
    if len(pts) < 2:
        xs = np.linspace(x1, x2, bins)
        return np.column_stack([xs, np.full(bins, y2)])
    return np.array(pts)


def _ray(floor, K, D, u, v):
    n = cv2.undistortPoints(np.array([[[u, v]]], np.float64), K, D).reshape(2)
    return floor.R.T @ np.array([n[0], n[1], 1.0])


def localize(dets, floor, K, D, max_range=8.0):
    """Turn image detections into floor objects (arena meters). Returns a list of dicts with
    x, y (estimated center), contact (near-edge center), width, depth, height, footprint."""
    if not floor.valid:
        return []
    C = floor.camera_position
    out = []
    for d in dets:
        pts = []
        for u, v in d["bottom"]:
            p = floor.pixel_to_floor(u, v, K, D, 0.0)
            if p is not None and np.hypot(p[0] - C[0], p[1] - C[1]) <= max_range:
                pts.append(p[:2])
        if len(pts) < 2:
            continue
        B = np.array(pts)
        away = B.mean(0) - C[:2]
        nrm = np.linalg.norm(away)
        if nrm < 1e-6:
            continue
        away /= nrm
        side = np.array([-away[1], away[0]])
        a_c, s_c = (B - C[:2]) @ away, (B - C[:2]) @ side
        width = float(np.clip(s_c.max() - s_c.min(), 0.02, 3.0))
        seen_depth = float(a_c.max() - a_c.min())     # visible side edges on the floor
        # the far side is hidden: if the side edges are visible use them, otherwise assume the
        # object is about as deep as it is wide
        depth = float(np.clip(max(seen_depth, width), 0.03, 1.5))
        # Near edge: the box's bottom edge is sharper than the (low-res) mask outline.
        x1_, _, x2_, y2_ = d["box"]
        pb = floor.pixel_to_floor((x1_ + x2_) / 2, y2_, K, D, 0.0)
        a_near = a_c.min()
        if pb is not None:
            a_near = float((pb[:2] - C[:2]) @ away)
        near = C[:2] + away * a_near + side * (s_c.max() + s_c.min()) / 2
        center = near + away * depth / 2
        far = near + away * depth
        # height: top of the box, on the vertical line above the far edge (where a box's top
        # edge sits; for round objects this slightly overestimates, which is the safe side)
        x1, y1, x2, y2 = d["box"]
        r_ = _ray(floor, K, D, (x1 + x2) / 2, y1)
        w0 = C - np.array([far[0], far[1], 0.0])
        a_, b_, c_ = r_ @ r_, r_[2], 1.0
        dd, ee = r_ @ w0, w0[2]
        den = a_ * c_ - b_ * b_
        height = float(max(0.0, (a_ * ee - b_ * dd) / den)) if abs(den) > 1e-9 else 0.0
        hull = cv2.convexHull(np.vstack([B, near + side * (s_c.min() - (s_c.max() + s_c.min()) / 2),
                                         near + side * (s_c.max() - (s_c.max() + s_c.min()) / 2),
                                         far + side * (s_c.min() - (s_c.max() + s_c.min()) / 2),
                                         far + side * (s_c.max() - (s_c.max() + s_c.min()) / 2)])
                              .astype(np.float32)).reshape(-1, 2)
        out.append({"name": d["name"], "conf": d["conf"], "box": d["box"],
                    "x": float(center[0]), "y": float(center[1]),
                    "contact": (float(near[0]), float(near[1])),
                    "width_m": width, "depth_m": depth, "height_m": min(height, 3.0),
                    "footprint": hull.astype(np.float64)})
    return out


def dedupe(objs, tag_points=()):
    """Clean up one frame of floor objects:
    * the same thing proposed twice (whole object and a face / shadow of it, two overlapping
      boxes) -> keep the more confident one. Decided on the floor: centers close together, or
      footprints overlapping by more than 30 % of the smaller one;
    * a detection that contains an AprilTag object -> dropped, the tag is far more precise."""
    keep = []
    for o in sorted(objs, key=lambda o: -o["conf"]):
        hull = o["footprint"].astype(np.float32)
        if any(cv2.pointPolygonTest(hull, (float(x), float(y)), True) > -0.02 for x, y in tag_points):
            continue
        area = cv2.contourArea(hull)
        dup = False
        for k in keep:
            dist = math.hypot(k["x"] - o["x"], k["y"] - o["y"])
            if dist < 0.5 * max(min(k["width_m"], o["width_m"]), 0.05):
                dup = True
                break
            khull = k["footprint"].astype(np.float32)
            inter, _ = cv2.intersectConvexConvex(khull, hull)
            if inter > 0.3 * max(min(area, cv2.contourArea(khull)), 1e-6):
                dup = True
                break
        if not dup:
            keep.append(o)
    return keep


ELONGATED = 2.5     # length / width above which an object keeps a rectangle instead of a square


def box_geometry(footprint):
    """(cx, cy, length, width, angle_rad) of the smallest rotated rectangle around a footprint."""
    (cx, cy), (w, h), ang = cv2.minAreaRect(np.asarray(footprint, np.float32))
    a = math.radians(ang)
    if h > w:                                   # angle of the LONG side
        w, h, a = h, w, a + math.pi / 2
    return cx, cy, w, h, a


def box_polygon(cx, cy, length, width, angle):
    """Square (side = length) unless the object is clearly elongated, then a rectangle."""
    if length < ELONGATED * width:
        width = length
    c, s_ = math.cos(angle), math.sin(angle)
    hl, hw = length / 2, width / 2
    loc = np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]])
    return loc @ np.array([[c, s_], [-s_, c]]) + [cx, cy]


class FloorTracker:
    """Keeps detected objects stable. Matching is by floor position alone (labels play no
    part). Each track holds a smoothed box (center, size, orientation) rather than the raw
    outline of the latest frame, so the drawn bounds do not jitter:
      * center and size are exponentially smoothed (size more slowly than position);
      * orientation is smoothed on the circle with the box's symmetry (90 deg for a square,
        180 deg for a rectangle), so 0 / 90 deg flips of a square never make it spin;
      * a new track is reported only once seen `min_hits` times, and a track survives short
        gaps (`ttl`), so one missed or extra frame never makes a box blink."""

    def __init__(self, gate_m=0.15, ttl=1.5, alpha=0.35, size_alpha=0.15, min_hits=3):
        self.gate, self.ttl, self.alpha, self.size_alpha, self.min_hits = gate_m, ttl, alpha, size_alpha, min_hits
        self.tracks = {}
        self.next_id = DET_ID_BASE

    @staticmethod
    def _sym(t):
        return 4 if t["length"] < ELONGATED * t["width"] else 2      # 90 or 180 deg symmetry

    def update(self, objs, now):
        used = set()
        for o in sorted(objs, key=lambda o: -o["conf"]):
            cx, cy, L, Wd, ang = box_geometry(o["footprint"])
            best, bd = None, None
            for tid, t in self.tracks.items():
                if tid in used:
                    continue
                dist = math.hypot(t["x"] - cx, t["y"] - cy)
                if dist < max(self.gate, 0.6 * max(L, t["length"])) and (bd is None or dist < bd):
                    best, bd = tid, dist
            if best is None:
                best = self.next_id
                self.next_id += 1
                self.tracks[best] = dict(o, x=cx, y=cy, length=L, width=Wd, angle=ang, first=now, hits=0)
            else:
                t = self.tracks[best]
                a, sa = self.alpha, self.size_alpha
                t["x"] += a * (cx - t["x"])
                t["y"] += a * (cy - t["y"])
                t["length"] += sa * (L - t["length"])
                t["width"] += sa * (Wd - t["width"])
                n = self._sym(t)                      # average n*angle on the circle
                v = (1 - a) * np.exp(1j * n * t["angle"]) + a * np.exp(1j * n * ang)
                t["angle"] = float(np.angle(v)) / n
                for k in ("height_m", "conf"):
                    t[k] = (1 - a) * t.get(k, 0.0) + a * o.get(k, 0.0)
                t["contact"], t["box"] = o.get("contact"), o.get("box")
            t = self.tracks[best]
            t["width_m"], t["depth_m"] = t["length"], t["width"] if t["length"] >= ELONGATED * t["width"] else t["length"]
            t["footprint"] = box_polygon(t["x"], t["y"], t["length"], t["width"], t["angle"])
            t["seen"] = now
            t["hits"] += 1
            used.add(best)
        for tid in [t for t, v in self.tracks.items() if now - v["seen"] > self.ttl]:
            del self.tracks[tid]
        return {tid: t for tid, t in self.tracks.items()
                if t["hits"] >= self.min_hits and (tid in used or now - t["seen"] <= self.ttl)}


def as_result_objects(tracks, now):
    """Detected objects in the same format as tag objects (so tables, map, A* use them)."""
    out = []
    for tid, t in tracks.items():
        out.append({"id": tid, "name": "object", "frame": "world", "method": "detect",
                    "x": t["x"], "y": t["y"], "z": 0.0, "yaw_deg": 0.0, "tilt_deg": 0.0,
                    "reproj_px": 0.0, "conf": round(t["conf"], 2), "width_m": round(t["width_m"], 3),
                    "yaw_box_deg": round(math.degrees(t.get("angle", 0.0)), 1),
                    "depth_m": round(t["depth_m"], 3), "height_m": round(t["height_m"], 3),
                    "contact": t["contact"], "box": t["box"], "footprint_poly": t["footprint"],
                    "age_s": round(now - t["seen"], 2), "family": "detector", "native_id": None})
    return out


def draw_detections(frame, objs, color=(255, 0, 255)):
    for o in objs:
        x1, y1, x2, y2 = [int(v) for v in o["box"]]
        stale = o.get("age_s", 0) > 0.6
        col = (150, 100, 150) if stale else color
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        txt = (f"obj {o['id']}  ({o['x']:+.2f}, {o['y']:+.2f})  "
               f"{o.get('width_m', 0) * 100:.0f}x{o.get('depth_m', 0) * 100:.0f} cm, h {o.get('height_m', 0) * 100:.0f} cm")
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 0, 0), -1)
        cv2.putText(frame, txt, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
