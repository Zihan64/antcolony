#!/usr/bin/env python3
"""
apriltag_tracker_gui.py

Real time testing GUI for apriltag_floor_mapper.py (keep both files in the same folder).

Panels
  Video      annotated camera feed with floor grid. Hover = floor x, y in the status bar.
             Left click drops a floor pin, right click clears pins.
  Map        top-down view: floor tags, camera, objects with motion trails, pins,
             and ground truth (synthetic mode).
  Plot       x, y, z over time for the selected tag (click a row in the table).
  Table      per tag pose, jitter (std dev over a rolling window), update rate,
             reprojection error, and error vs ground truth when available.

  Arena/A*   bird's-eye metric map between the 4 pivot tags, obstacles, occupancy grid and
             A* path. Left click = goal, right click = start.
             Pivots: the 4 LARGEST tags on the print sheet (29-32, 112 mm) are the only tags that
             define the floor plane. Lay them flat at the corners (any order). Other tags may be
             in view at any time: they are objects and never shift the plane. When all 4 pivots
             are in view the plane is built automatically and drawn on the camera image.
             The log then names the two distances to tape-measure (origin -> +x, origin -> +y);
             enter them and press "Apply tape measurements" (also estimates the focal length
             when no calibration file is loaded). Clear the floor, press "Capture empty floor".
             "Tags lie flat on the floor" locates tags by ray-plane intersection (mm precision).
             The map covers all visible floor out to "Map range" (beyond the tags too), grows
             as new floor is seen, accumulates obstacle evidence over frames, and re-locks the
             pose after the camera moves.
             "UDP out" streams robot pose, path and a steering command as JSON every frame.

Testing tools
  Measure    average the selected tag over N seconds and compare against a
             tape-measured position you type in ("x, y, z" in meters).
  Record     log every processed frame to CSV.
  Synthetic  a rendered scene with moving tags and known true positions, so the
             whole pipeline can be tested without the camera.

Run
  pip install opencv-contrib-python pupil-apriltags PyQt6 pyqtgraph pygrabber
  python apriltag_tracker_gui.py                       # pick the source in the GUI
  python apriltag_tracker_gui.py --synthetic --autostart
  python apriltag_tracker_gui.py --calib calib.npz --autostart

Keep apriltag_sheet_a4.json next to the scripts: it holds the printed marker sizes.
AprilTags and ArUco markers are tracked together. IDs are global: AprilTag 0-86 keep their
number, ArUco 0-14 become 87-101 (the table's Name column shows e.g. "aruco3" for id 90).
Layout: the Run tab has what is needed to operate (camera, arena setup, A*, object detection,
log); calibration and tuning live in the Settings tab.
Objects without tags: a prompt-free YOLOE model finds any object (no class list, nothing
learned from earlier sightings) and places it on the floor map.
"""

import argparse
import csv
import json
import math
import os
import sys
import threading
import time
import traceback
from collections import deque
from types import SimpleNamespace

import cv2
import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, pyqtSignal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import apriltag_floor_mapper as m  # noqa: E402
import arena_map as am  # noqa: E402
import object_detector as od  # noqa: E402

RESOLUTIONS = ["1920x1080", "3840x2160", "2560x1440", "1280x720"]

# pupil_apriltags can segfault when a Detector is freed (its __del__), especially from
# another thread. Detectors replaced at runtime are parked here instead of being freed.
_DETECTOR_KEEPALIVE = []


# --------------------------------------------------------------------------
# Frame sources (same interface as m.CameraStream)
# --------------------------------------------------------------------------

class PacedSource:
    """Plays frames from read_fn() at a fixed rate on a background thread.
    read_fn returns (frame, meta) or None to stop. meta travels with its frame."""

    def __init__(self, read_fn, fps):
        self.read_fn, self.period = read_fn, 1.0 / max(fps, 1.0)
        first = read_fn()
        if first is None:
            raise RuntimeError("Source returned no frames.")
        self.frame, self.meta = first
        self.height, self.width = self.frame.shape[:2]
        self.frame_id, self.stamp, self.grab_fps = 1, time.perf_counter(), 0.0
        self.last_meta = self.meta
        self.cond = threading.Condition()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        t_next, n, t_last = time.perf_counter(), 0, time.perf_counter()
        while self.running:
            t_next += self.period
            item = self.read_fn()
            if item is None:
                self.running = False
                break
            now = time.perf_counter()
            with self.cond:
                self.frame, self.meta = item
                self.stamp = now
                self.frame_id += 1
                self.cond.notify_all()
            n += 1
            if now - t_last >= 1.0:
                self.grab_fps, n, t_last = n / (now - t_last), 0, now
            delay = t_next - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                t_next = time.perf_counter()

    def latest(self, after_id=0, timeout=1.0):
        with self.cond:
            self.cond.wait_for(lambda: self.frame_id > after_id or not self.running, timeout=timeout)
            if self.frame_id <= after_id:
                return None, after_id, 0.0
            self.last_meta = self.meta
            return self.frame.copy(), self.frame_id, self.stamp

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)


def video_file_reader(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    def read():
        ok, frame = cap.read()
        if not ok:  # loop the file
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        return (frame, None) if ok else None
    return read, fps


class SyntheticScene:
    """Rendered 1.2 x 0.8 m arena with known ground truth, laid out like the real setup:
    corner AprilTags 29-32 (140 mm sheet tags), a robot carrying ArUco 3 (= global 90)
    driving a loop, a crate with tag 47 on top, an untagged box obstacle, and tag 60 lying
    still on the floor for jitter tests."""
    W, H = 1280, 720
    AW, AH = 1.2, 0.8
    ROBOT_R, ROBOT_H = 0.07, 0.05

    def __init__(self, raw=False):
        """raw=True: nothing preset (like a fresh real setup): corners in the same order as the
        user's floor (29 far-right, 30 near-right, 31 far-left, 32 near-left), no arena, and the
        app gets a 15% wrong focal length, as when no calibration file is loaded."""
        self.raw = raw
        self.K = np.array([[1000.0, 0, self.W / 2], [0, 1000.0, self.H / 2], [0, 0, 1]])
        self.D = np.zeros(5)
        self.C = np.array([0.6, -0.75, 1.3])
        target = np.array([0.6, 0.4, 0.0])
        z = (target - self.C) / np.linalg.norm(target - self.C)
        x = np.cross(z, [0, 0, 1.0])
        x /= np.linalg.norm(x)
        self.R = np.vstack([x, np.cross(z, x), z])
        self.rvec, _ = cv2.Rodrigues(self.R)
        self.tvec = -self.R @ self.C

        cfg = m.load_config(None)
        # The robot carries ArUco DICT_4X4_50 #3 (= global 90). The print sheet only lists the
        # AprilTags 0-86, so the scene declares that family and the marker's size itself.
        cfg["families"] = m.normalize_families([{"name": "tag36h11", "offset": 0},
                                                {"name": "DICT_4X4_50", "offset": 87, "ids": [0, 14]}])
        cfg["tag_sizes"].setdefault(90, 0.05)
        if raw:
            self.corners = {29: (self.AW, self.AH, 90.0), 30: (self.AW, 0.0, 0.0),
                            31: (0.0, self.AH, 180.0), 32: (0.0, 0.0, -90.0)}
        else:
            self.corners = {29: (0.0, 0.0, 0.0), 30: (self.AW, 0.0, 90.0),
                            31: (self.AW, self.AH, 180.0), 32: (0.0, self.AH, 0.0)}
        cfg["floor_tags"] = {k: {"x": v[0], "y": v[1], "yaw_deg": v[2]} for k, v in self.corners.items()}
        cfg["objects"] = {90: {"name": "robot", "orientation": "flat"},
                          47: {"name": "crate", "footprint": [0.12, 0.12], "offset": [0, 0, -0.04],
                               "orientation": "flat", "height_m": 0.08},
                          60: {"name": "static", "orientation": "flat", "obstacle": False}}
        cfg["arena"] = {"corner_ids": [29, 30, 31, 32], "width_m": self.AW, "height_m": self.AH,
                        "robot_id": 90, "robot_radius_m": 0.09}
        if raw:
            cfg["floor_tags"] = {}
            cfg["arena"] = {"corner_ids": [29, 30, 31, 32], "robot_id": 90, "robot_radius_m": 0.09}
            self.K_app = self.K.copy()
            self.K_app[0, 0] = self.K_app[1, 1] = 850.0
        else:
            self.K_app = self.K
        self.cfg = cfg
        self.sizes = cfg["tag_sizes"]
        dA = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        dR = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.tiles = {}
        for gid in (29, 30, 31, 32, 47, 60, 90):
            mk = (cv2.aruco.generateImageMarker(dR, gid - 87, 240) if gid >= 87
                  else cv2.aruco.generateImageMarker(dA, gid, 240))
            pad = 40
            tile = np.full((mk.shape[0] + 2 * pad,) * 2, 255, np.uint8)
            tile[pad:pad + mk.shape[0], pad:pad + mk.shape[0]] = mk
            self.tiles[gid] = (cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR), mk.shape[0])
        rng = np.random.default_rng(5)
        tex = cv2.resize(rng.normal(0, 1, (24, 40)), (self.W, self.H), interpolation=cv2.INTER_CUBIC)
        g = np.clip(172 + 6 * tex, 0, 255).astype(np.uint8)
        self.floor_bgr = cv2.merge([g, (g * 0.97).astype(np.uint8), (g * 0.92).astype(np.uint8)])
        self.t0 = time.perf_counter()
        self.rng = np.random.default_rng(0)
        self.empty_frames = 0          # >0: render without objects (for "Capture empty floor")

    def _proj(self, P):
        return cv2.projectPoints(np.asarray(P, float).reshape(-1, 3), self.rvec, self.tvec,
                                 self.K, self.D)[0].reshape(-1, 2)

    def _tag(self, img, gid, yaw, pos):
        tile, n = self.tiles[gid]
        h = self.sizes[gid] * tile.shape[0] / n / 2
        loc = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
        px = self._proj((m.rot_z(yaw) @ loc.T).T + pos)
        x0, y0 = np.maximum(np.floor(px.min(0)).astype(int) - 1, 0)
        x1, y1 = np.minimum(np.ceil(px.max(0)).astype(int) + 1, [self.W, self.H])
        if x1 <= x0 or y1 <= y0:
            return
        s = tile.shape[0] - 1
        Hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [s, 0], [s, s], [0, s]]),
                                         (px - [x0, y0]).astype(np.float32))
        wh = (int(x1 - x0), int(y1 - y0))
        warped = cv2.warpPerspective(tile, Hm, wh)
        mask = cv2.warpPerspective(np.full(tile.shape[:2], 255, np.uint8), Hm, wh) > 128
        img[y0:y1, x0:x1][mask] = warped[mask]

    def _box(self, img, x0, y0, x1, y1, h, top, side):
        faces = [[(x0, y0, h), (x1, y0, h), (x1, y1, h), (x0, y1, h)],
                 [(x0, y0, 0), (x1, y0, 0), (x1, y0, h), (x0, y0, h)],
                 [(x0, y0, 0), (x0, y1, 0), (x0, y1, h), (x0, y0, h)],
                 [(x1, y0, 0), (x1, y1, 0), (x1, y1, h), (x1, y0, h)],
                 [(x0, y1, 0), (x1, y1, 0), (x1, y1, h), (x0, y1, h)]]
        for k in sorted(range(5), key=lambda k: -np.linalg.norm(np.mean(faces[k], 0) - self.C)):
            cv2.fillConvexPoly(img, np.round(self._proj(faces[k])).astype(np.int32),
                               top if k == 0 else side, cv2.LINE_AA)

    def robot_pose(self, t):
        w = 2 * math.pi / 16.0
        x = 0.6 + 0.38 * math.cos(w * t)
        y = 0.4 + 0.24 * math.sin(w * t)
        heading = math.degrees(math.atan2(0.24 * math.cos(w * t), -0.38 * math.sin(w * t)))
        return x, y, heading

    def read(self):
        t = time.perf_counter() - self.t0
        img = self.floor_bgr.copy()
        for gid, (x, y, yaw) in self.corners.items():
            self._tag(img, gid, yaw, np.array([x, y, 0.0]))
        truth = {}
        if self.empty_frames > 0:
            self.empty_frames -= 1
        else:
            self._box(img, 0.48, 0.30, 0.66, 0.50, 0.15, (60, 70, 160), (40, 50, 120))   # untagged
            self._box(img, 1.32, 0.15, 1.50, 0.33, 0.06, (40, 130, 60), (30, 90, 40))    # outside the tags
            self._box(img, 0.92, 0.56, 1.04, 0.68, 0.08, (205, 205, 205), (120, 140, 120))
            self._tag(img, 47, 20.0, np.array([0.98, 0.62, 0.08]))
            truth[47] = [0.98, 0.62, 0.04]
            self._tag(img, 60, 15.0, np.array([0.22, 0.62, 0.0]))
            truth[60] = [0.22, 0.62, 0.0]
            x, y, hd = self.robot_pose(t)
            th = np.linspace(0, 2 * np.pi, 48)
            for zz, col in ((0.0, (45, 45, 45)), (self.ROBOT_H, (70, 70, 70))):
                ring = np.stack([x + self.ROBOT_R * np.cos(th), y + self.ROBOT_R * np.sin(th),
                                 np.full(48, zz)], 1)
                cv2.fillPoly(img, [np.round(self._proj(ring)).astype(np.int32)], col, cv2.LINE_AA)
            self._tag(img, 90, hd, np.array([x, y, self.ROBOT_H]))
            truth[90] = [x, y, self.ROBOT_H]
        if self.raw:
            truth = {k: [self.AW - v[0], self.AH - v[1], v[2]] for k, v in truth.items()}
        img = cv2.GaussianBlur(img, (3, 3), 0.7)
        noise = self.rng.integers(-5, 6, (self.H, self.W, 1), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img, truth


# --------------------------------------------------------------------------
# Worker thread
# --------------------------------------------------------------------------

class Worker(QtCore.QThread):
    frame_ready = pyqtSignal(QtGui.QImage, float)
    arena_ready = pyqtSignal(QtGui.QImage, float, dict)
    result_ready = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, source, mapper, max_display_w=1280):
        super().__init__()
        self.source, self.mapper = source, mapper
        self.max_display_w = max_display_w
        self.pending_detector = None
        self.pending_floor = None
        self.show_grid = True
        self.pins = []
        self.display_free = True
        self.last_frame = None
        self.rec_lock = threading.Lock()
        self.rec_file = self.rec_writer = None
        # arena / planning state (set from the GUI thread, used here)
        self.amap = None
        self.pending_amap = None
        self.capture_bg = 0            # frames to wait, then store the empty-floor reference
        self.goal = None               # (x, y) arena meters
        self.manual_start = None
        self.auto_replan = True
        self.replan_period = 0.3
        self.robot_body = (0.08, None) # radius, height (None = from the robot tag)
        self.arena_free = True
        self.arena_last = {}
        self.udp = None                # (socket, (host, port))
        self._last_plan_t = 0.0
        self._plan = None
        self.objdet = None             # object_detector.ObjectDetector (optional YOLO proposals)
        self.appearance = False        # objects = floor that does not look like floor (no model)
        self.app_tracker = od.FloorTracker(gate_m=0.15, ttl=1.0, min_hits=3)
        self._app_objs = []
        self.tracker = od.FloorTracker()
        self.det_obstacles = True
        self._det_stamp = None
        self._det_objs = []
        self.pair = None               # (start tag id, goal tag id) for tag-to-tag paths
        self._pair_last = {}
        self._pair_key = None
        # intermediate stops between start and goal, in order: {"kind": "tag", "id"} (a tag or a
        # detected object, followed live) or {"kind": "point", "xy"}; like a route with stops
        self.stops = []
        self._next_stop = 0            # with a robot: stops before this index are reached
        self._route_draw = []          # (number, xy, done) for drawing

    def start_recording(self, path):
        with self.rec_lock:
            self.rec_file = open(path, "w", newline="")
            self.rec_writer = csv.writer(self.rec_file)
            self.rec_writer.writerow(["t", "id", "name", "frame", "x", "y", "z", "yaw_deg", "tilt_deg",
                                      "reproj_px", "floor_valid", "floor_reproj_px", "cam_x", "cam_y",
                                      "cam_z", "truth_x", "truth_y", "truth_z"])

    def stop_recording(self):
        with self.rec_lock:
            if self.rec_file:
                self.rec_file.close()
            self.rec_file = self.rec_writer = None

    def _record(self, r):
        with self.rec_lock:
            if not self.rec_writer:
                return
            cam = r["camera"] or {}
            for o in r["objects"]:
                tr = (r.get("truth") or {}).get(o["id"], [None] * 3)
                self.rec_writer.writerow([f"{r['t']:.4f}", o["id"], o["name"], o["frame"],
                                          f"{o['x']:.5f}", f"{o['y']:.5f}", f"{o['z']:.5f}",
                                          f"{o.get('yaw_deg', 0):.2f}", f"{o.get('tilt_deg', 0):.2f}",
                                          o["reproj_px"], int(r["floor_valid"]), r["floor_reproj_px"],
                                          cam.get("x"), cam.get("y"), cam.get("z"), *tr])

    def _merge_detections(self, r, stamp):
        """Add detected (untagged) objects to this frame's results, located on the floor."""
        floor, now = self.mapper.floor, time.perf_counter()
        st, dets = self.objdet.latest()
        if st != self._det_stamp:
            self._det_stamp = st
            # a proposal that is really an AprilTag is dropped: the tag itself is tracked precisely
            dets = od.drop_tag_boxes(dets, [c for _, c in r.get("_dets", [])])
            objs = od.localize(dets, floor, self.mapper.K, self.mapper.D) if floor.valid else []
            tag_pts = [(o["x"], o["y"]) for o in r["objects"]
                       if o.get("frame") == "world" and o.get("method") != "detect"]
            objs = od.dedupe(objs, tag_pts)
            # keep tracks alive for a few detector cycles, however slow the detector is
            self.tracker.ttl = max(1.5, 3.5 * self.objdet.infer_ms / 1000.0)
            tracks = self.tracker.update(objs, now)
            self._det_objs = od.as_result_objects(tracks, now)
            if not floor.valid:      # no floor yet: still show the boxes
                self._det_objs = [{"id": -1 - i, "name": d["name"], "conf": d["conf"], "box": d["box"],
                                   "x": float("nan"), "y": float("nan"), "frame": "image"}
                                  for i, d in enumerate(dets)]
        r["objects"].extend(o for o in self._det_objs if o.get("frame") == "world")
        r["detections"] = list(self._det_objs)
        if not self.det_obstacles:
            for o in r["objects"]:
                if o.get("method") == "detect":
                    o["obstacle"] = False

    def _arena_step(self, raw, r, now):
        """Global floor map: fuse this frame, learn anchors, plan, publish, draw."""
        gm, floor, cfg = self.amap, self.mapper.floor, self.mapper.cfg
        K, D = self.mapper.K, self.mapper.D
        a = gm.arena
        info = {"valid": False, "events": floor.pop_events()}
        if not floor.valid:
            self.arena_last = info
            if info["events"]:
                self.arena_ready.emit(QtGui.QImage(), 1.0, info)
            return
        objs = {o["id"]: o for o in r["objects"] if o.get("frame") == "world"}
        robot = objs.get(a.get("robot_id"))
        # Tag-to-tag mode: the start tag is the robot, the goal tag is the target. Both are
        # followed live; if one drops out of view its last seen position is kept.
        pair = self.pair
        goal = self.goal
        pair_info = None
        skip = {a.get("robot_id")}
        if pair:
            sid, gid = pair
            for t in (sid, gid):
                if t in objs:
                    self._pair_last[t] = (objs[t], now)
            s_ = self._pair_last.get(sid)
            g_ = self._pair_last.get(gid)
            robot = s_[0] if s_ else None
            goal = (g_[0]["x"], g_[0]["y"]) if g_ else None
            skip |= {sid, gid}
            pair_info = {"start_id": sid, "goal_id": gid,
                         "start_seen": sid in objs, "goal_seen": gid in objs,
                         "start_age_s": None if not s_ else round(now - s_[1], 2),
                         "goal_age_s": None if not g_ else round(now - g_[1], 2)}
        clear = []
        if robot is not None:
            rad, hgt = self.robot_body
            hgt = robot["z"] + 0.01 if hgt is None else hgt
            clear.append(am.ArenaMap.body_outline(robot["x"], robot["y"], rad + 0.01, max(hgt, 0.0),
                                                  floor.camera_position))
        if pair and goal is not None:
            g_obj = self._pair_last[pair[1]][0]
            spec = cfg.get("objects", {}).get(pair[1], {})
            g_rad = spec.get("radius") or 0.75 * max(spec.get("footprint", [self.mapper.tag_size(pair[1])]))
            clear.append(am.ArenaMap.body_outline(goal[0], goal[1], g_rad + 0.01,
                                                  max(g_obj.get("z", 0.0), 0.0) + 0.01, floor.camera_position))
        # Flat tags on the floor (pivots, markers) are paper, not obstacles: their outline, with
        # the white border, is cleared from the obstacle evidence. Pivots are cleared at their known
        # floor position whether or not they were detected in this frame.
        for tid, wc in floor.world_corners.items():
            ctr = wc[:, :2].mean(0)
            clear.append(ctr + (wc[:, :2] - ctr) * 1.45)
        for tid, c in r.get("_dets", []):
            if tid in floor.world_corners:
                continue
            o_ = objs.get(tid)
            if tid not in floor.world_corners and not (o_ and o_.get("method") == "plane"):
                continue
            pts = [floor.pixel_to_floor(u, v, K, D) for u, v in c]
            if any(p_ is None for p_ in pts):
                continue
            pts = np.array(pts)[:, :2]
            ctr = pts.mean(0)
            clear.append(ctr + (pts - ctr) * 1.45)
        cap = False
        if self.capture_bg > 0:
            self.capture_bg -= 1
            cap = self.capture_bg == 0
        gm.detect_appearance = self.appearance
        # Fuse into the map at most ~10x per second; tag tracking keeps running every frame.
        fuse_ok = floor.locked or not floor.auto_lock   # pause while the camera is moving
        info["map_paused"] = not fuse_ok
        if fuse_ok and (cap or now - getattr(self, "_fuse_t", 0) >= 0.1):
            gm.update(raw, floor, K, D, now, clear_polys=clear, max_range=float(a["max_range_m"]),
                      min_px_per_cell=float(a["min_px_per_cell"]), capture_bg=cap)
            self._fuse_t = now
            if self.appearance:
                found = gm.objects_in_view()
                tag_pts = [(o_["x"], o_["y"]) for o_ in objs.values() if o_.get("method") != "detect"]
                tracks = self.app_tracker.update(od.dedupe(found, tag_pts), now)
                self._app_objs = od.as_result_objects(tracks, now)
                for o_ in self._app_objs:
                    o_["method"] = "appearance"   # its square bounds are an A* obstacle too
        if not self.appearance:
            self._app_objs = []
        r["objects"].extend(self._app_objs)
        for o_ in self._app_objs:
            objs.setdefault(o_["id"], o_)
        if cap:
            info["background_captured"] = True

        start = (robot["x"], robot["y"]) if robot is not None else (None if pair else self.manual_start)

        # Intermediate stops: tags / detected objects are followed live (last seen position kept),
        # points are fixed. With a robot, a stop it has reached drops out of the route, so the plan
        # always runs robot -> next stops -> goal.
        stop_xy, stop_info = [], []
        for k, st in enumerate(self.stops):
            xy, seen = None, True
            if st["kind"] == "point":
                xy = tuple(st["xy"])
            else:
                tid = st["id"]
                if tid in objs:
                    self._pair_last[tid] = (objs[tid], now)
                last = self._pair_last.get(tid)
                seen = tid in objs
                xy = (last[0]["x"], last[0]["y"]) if last else None
                if tid < od.DET_ID_BASE:
                    skip.add(tid)                 # a tag stop is a place to go, not an obstacle
            stop_xy.append(xy)
            stop_info.append({"n": k + 1, "label": st.get("label", ""), "xy": xy, "seen": seen,
                              "done": robot is not None and k < self._next_stop})
        arrive = max(0.05, 0.6 * float(a.get("robot_radius_m", 0.1)))
        if robot is not None and self._next_stop < len(self.stops):
            nxt = stop_xy[self._next_stop]
            if nxt is not None and math.hypot(robot["x"] - nxt[0], robot["y"] - nxt[1]) < arrive:
                info["events"].append(f"reached stop {self._next_stop + 1} "
                                      f"({self.stops[self._next_stop].get('label', '')})")
                stop_info[self._next_stop]["done"] = True
                self._next_stop += 1
        first = self._next_stop if robot is not None else 0
        route_stops = [xy for xy in stop_xy[first:] if xy is not None]
        missing = [i + 1 for i, xy in enumerate(stop_xy) if xy is None and i >= first]
        self._route_draw = [(i["n"], i["xy"], i["done"]) for i in stop_info if i["xy"] is not None]
        points = [start] + route_stops + ([goal] if goal is not None else [])
        have_route = start is not None and len(points) >= 2
        # replan when any point moved more than 1 cm (at most ~7x per second), or periodically
        key = tuple((round(p_[0], 2), round(p_[1], 2)) for p_ in points) if have_route else None
        replan = have_route and (self._plan is None or (
            key != self._pair_key and now - self._last_plan_t > 0.15) or
            (self.auto_replan and now - self._last_plan_t > self.replan_period))
        if replan:
            self._pair_key = key
        draw = self.arena_free and now - getattr(self, "_draw_t", 0) > 0.2
        foot = blocked = None
        cell = gm.plan_resolution()
        if replan or draw:
            # a detected object that contains the robot or a path end is that thing itself
            ends = [p_ for p_ in ((robot["x"], robot["y"]) if robot is not None else None, goal, *route_stops) if p_]
            for o in r["objects"]:
                if o.get("footprint_poly") is not None and ends:
                    hull = o["footprint_poly"].astype(np.float32)
                    if any(cv2.pointPolygonTest(hull, (float(e[0]), float(e[1])), True) > -0.05 for e in ends):
                        o["obstacle"] = False
            foot = gm.footprint_mask(r["objects"], cfg, skip_ids=tuple(skip))
            occ = gm.occupancy(foot, bool(a.get("unknown_blocked", True)))
            blocked, cell = am.planning_grid(occ, gm, plan_res=cell)
            self.arena_data = (gm, occ, blocked, cell)
        if replan:
            self._plan = am.plan_route(gm, points, plan_res=cell, blocked=blocked)
            self._plan["goal"] = points[-1]
            if missing:
                self._plan["notes"] = [f"stop(s) {missing} not seen yet: skipped"] + self._plan["notes"]
            self._last_plan_t = now
        if not have_route:
            self._plan = None
        plan = self._plan
        cmd = None
        if plan and plan["status"] == "ok" and robot is not None:
            # steer along the current leg only: a route that comes back along the same corridor
            # must not make the robot skip ahead to the return part
            leg = plan["legs"][0]["waypoints"] if plan.get("legs") else plan["waypoints"]
            cmd = am.pure_pursuit((robot["x"], robot["y"]), robot.get("yaw_deg", 0.0), leg)

        info.update({"valid": True, "robot": None if robot is None else
                     {k: robot[k] for k in ("x", "y", "z", "yaw_deg")},
                     "start": start, "goal": goal, "pair": pair_info,
                     "stops": stop_info, "next_stop": self._next_stop if robot is not None else None,
                     "plan": None if plan is None else {k: plan[k] for k in
                                                        ("status", "waypoints", "length_m", "notes")
                                                        if k in plan} | {"ms": plan.get("ms"),
                                                                         "leg_lengths_m": [lg["length_m"] for lg in plan.get("legs", [])]},
                     "command": cmd, "has_background": gm.bg_enabled,
                     "map": {"res_mm": round(gm.res * 1000, 1), "x0": gm.x0, "x1": gm.x1, "y0": gm.y0,
                             "y1": gm.y1, "known_m2": round(float(gm.seen.sum()) * gm.res ** 2, 2),
                             "obstacle_m2": round(float(gm.obstacles().sum()) * gm.res ** 2, 3),
                             "anchors": sorted(floor.world_corners), "locked": floor.locked}})
        self.arena_last = info
        if self.udp:
            try:
                msg = {"t": r["t"], "frame": "arena", "objects": m.public(r)["objects"],
                       "camera": r["camera"], "robot": info["robot"], "goal": goal,
                       "pair": pair_info, "stops": stop_info,
                       "path": info["plan"], "command": cmd}
                self.udp[0].sendto(json.dumps(msg, default=float).encode(), self.udp[1])
            except OSError:
                pass

        if not draw:
            if info["events"] or info.get("background_captured"):
                self.arena_ready.emit(QtGui.QImage(), 1.0, info)
            return
        self._draw_t = now
        self.arena_free = False
        vis = am.render_global(gm, blocked, foot, cell, now)
        if gm.view is not None:
            r0, r1, c0, c1, _ = gm.view
            cv2.rectangle(vis, (c0, r0), (c1 - 1, r1 - 1), (255, 255, 255), 1)
        if plan and plan.get("waypoints"):
            pts = np.round(gm.w2p(plan["waypoints"])).astype(np.int32)
            cv2.polylines(vis, [pts], False, (0, 220, 0), 2, cv2.LINE_AA)
        for o in objs.values():
            p = tuple(np.round(gm.w2p([(o["x"], o["y"])])[0]).astype(int))
            h = math.radians(o.get("yaw_deg", 0))
            col = (255, 0, 255) if robot is not None and o is robot else (255, 255, 0)
            cv2.circle(vis, p, 5, col, -1)
            cv2.line(vis, p, (int(p[0] + 14 * math.cos(h)), int(p[1] - 14 * math.sin(h))), col, 2)
        for o in r["objects"]:
            if o.get("footprint_poly") is not None:
                col = (255, 0, 255) if o.get("obstacle") is not False else (180, 120, 180)
                cv2.polylines(vis, [np.round(gm.w2p(o["footprint_poly"])).astype(np.int32)], True, col, 2)
                p_ = tuple(np.round(gm.w2p([(o["x"], o["y"])])[0]).astype(int))
                cv2.putText(vis, f"{o['id']} {o['name']}", (p_[0] + 6, p_[1] - 6), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, col, 1, cv2.LINE_AA)
        if robot is not None:
            p = tuple(np.round(gm.w2p([(robot["x"], robot["y"])])[0]).astype(int))
            cv2.circle(vis, p, int(round(a["robot_radius_m"] / gm.res)), (255, 0, 255), 1)
        if start is not None and robot is None:
            p = tuple(np.round(gm.w2p([start])[0]).astype(int))
            cv2.drawMarker(vis, p, (255, 0, 255), cv2.MARKER_SQUARE, 12, 2)
        for n_, xy, done in self._route_draw:
            p = tuple(np.round(gm.w2p([xy])[0]).astype(int))
            col = (150, 150, 150) if done else (0, 200, 255)
            cv2.circle(vis, p, 9, col, 2, cv2.LINE_AA)
            cv2.putText(vis, str(n_), (p[0] - 4, p[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        if goal is not None:
            p = tuple(np.round(gm.w2p([goal])[0]).astype(int))
            cv2.drawMarker(vis, p, (0, 255, 0), cv2.MARKER_STAR, 18, 2)
            if pair:
                cv2.putText(vis, f"goal tag {pair[1]}", (p[0] + 10, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1, cv2.LINE_AA)
        if pair and robot is not None:
            p = tuple(np.round(gm.w2p([(robot["x"], robot["y"])])[0]).astype(int))
            cv2.putText(vis, f"start tag {pair[0]}", (p[0] + 10, p[1] + 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 255), 1, cv2.LINE_AA)
        if cmd:
            p = tuple(np.round(gm.w2p([cmd["target_xy"]])[0]).astype(int))
            cv2.circle(vis, p, 6, (0, 255, 255), 2)
        s_ = min(1.0, 1000 / max(vis.shape[1], 1), 800 / max(vis.shape[0], 1))
        if s_ < 1:
            vis = cv2.resize(vis, None, fx=s_, fy=s_, interpolation=cv2.INTER_AREA)
        rgb = np.ascontiguousarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        hh, ww = rgb.shape[:2]
        qimg = QtGui.QImage(rgb.data, ww, hh, 3 * ww, QtGui.QImage.Format.Format_RGB888).copy()
        self.arena_ready.emit(qimg, s_, info)

    def run(self):
        try:
            fid, n, t_fps, fps = 0, 0, time.perf_counter(), 0.0
            while not self.isInterruptionRequested():
                K, D = self.mapper.K, self.mapper.D
                frame, fid, stamp = self.source.latest(fid, timeout=0.5)
                if frame is None:
                    if not self.source.running:
                        raise RuntimeError("Source stopped delivering frames.")
                    continue
                meta = getattr(self.source, "last_meta", None)
                if self.pending_detector is not None:
                    _DETECTOR_KEEPALIVE.append(self.mapper.detector)
                    self.mapper.detector, self.pending_detector = self.pending_detector, None
                if self.pending_floor is not None:
                    self.mapper.floor, self.pending_floor = self.pending_floor, None
                if self.pending_amap is not None:
                    self.amap, self.pending_amap = self.pending_amap, None
                    self._plan = None

                raw = frame.copy() if (self.amap is not None or self.objdet is not None) else None
                if self.objdet is not None:
                    self.objdet.submit(raw, stamp)
                r = self.mapper.process(frame, stamp)
                if self.objdet is not None:
                    self._merge_detections(r, stamp)
                now = time.perf_counter()
                n += 1
                if now - t_fps >= 1.0:
                    fps, n, t_fps = n / (now - t_fps), 0, now
                floor = self.mapper.floor
                r.update(latency_ms=(now - stamp) * 1e3, proc_fps=fps, grab_fps=self.source.grab_fps,
                         truth=meta, wall=now)
                r["_floor"] = (floor.R.copy(), floor.t.copy()) if floor.valid else None
                self._record(r)
                if raw is not None and self.amap is not None:
                    self._arena_step(raw, r, now)
                self.result_ready.emit(r)

                if self.display_free:
                    self.display_free = False
                    if self.show_grid and floor.valid:
                        if self.amap is not None:
                            ids = self.amap.arena["corner_ids"]
                            am.draw_plane(frame, self.amap.corners, [f"tag {i}" for i in ids], floor, K, D,
                                          max_range=float(self.amap.arena.get("max_range_m", 4.0)))
                        else:
                            m.draw_floor_grid(frame, floor, K, D)
                    if floor.valid:
                        for o in self._app_objs:
                            poly = np.column_stack([o["footprint_poly"], np.zeros(len(o["footprint_poly"]))])
                            pp = np.round(floor.world_to_pixel(poly, K, D)).astype(np.int32)
                            cv2.polylines(frame, [pp], True, (255, 0, 255), 2, cv2.LINE_AA)
                            m._put(frame, f"obj {o['id']} ({o['x']:+.2f}, {o['y']:+.2f})",
                                   (int(pp[:, 0].min()), int(pp[:, 1].min()) - 6), (255, 0, 255))
                    if floor.valid and self.objdet is not None:
                        for o in self._det_objs:
                            if o.get("frame") != "world" or o.get("footprint_poly") is None:
                                continue
                            poly = np.column_stack([o["footprint_poly"], np.zeros(len(o["footprint_poly"]))])
                            pp = np.round(floor.world_to_pixel(poly, K, D)).astype(np.int32)
                            cv2.polylines(frame, [pp], True, (255, 0, 255), 2, cv2.LINE_AA)
                        for o in self._det_objs:
                            if o.get("frame") == "image":
                                x1, y1, x2, y2 = [int(v) for v in o["box"]]
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 0, 200), 2)
                                m._put(frame, f"{o['name']} {o['conf']:.2f} (no floor yet)", (x1, y1 - 6), (200, 0, 200))
                    plan = self._plan
                    if floor.valid and plan and plan.get("status") == "ok" and len(plan["waypoints"]) >= 2:
                        # the A* path drawn on the camera image (on the floor plane)
                        pts = []
                        for (ax, ay), (bx, by) in zip(plan["waypoints"], plan["waypoints"][1:]):
                            for t_ in np.linspace(0, 1, 12, endpoint=False):
                                pts.append([ax + t_ * (bx - ax), ay + t_ * (by - ay), 0.0])
                        pts.append([*plan["waypoints"][-1], 0.0])
                        px = np.round(floor.world_to_pixel(pts, K, D)).astype(np.int32)
                        cv2.polylines(frame, [px], False, (0, 0, 0), 7, cv2.LINE_AA)
                        cv2.polylines(frame, [px], False, (0, 255, 0), 4, cv2.LINE_AA)
                        for wp in floor.world_to_pixel([[x_, y_, 0] for x_, y_ in plan["waypoints"]], K, D):
                            cv2.circle(frame, (int(wp[0]), int(wp[1])), 6, (0, 120, 0), -1)
                        g_ = floor.world_to_pixel([[*plan["waypoints"][-1], 0]], K, D)[0]
                        cv2.drawMarker(frame, (int(g_[0]), int(g_[1])), (0, 255, 0), cv2.MARKER_STAR, 26, 3)
                        m._put(frame, f"A* {plan['length_m']:.3f} m", (int(g_[0]) + 14, int(g_[1]) - 10), (0, 255, 0))
                    if floor.valid:
                        for n_, xy, done in list(self._route_draw):
                            u_, v_ = floor.world_to_pixel([[xy[0], xy[1], 0.0]], K, D)[0]
                            if np.isfinite(u_) and np.isfinite(v_):
                                col = (150, 150, 150) if done else (0, 200, 255)
                                cv2.circle(frame, (int(u_), int(v_)), 14, (0, 0, 0), 5, cv2.LINE_AA)
                                cv2.circle(frame, (int(u_), int(v_)), 14, col, 3, cv2.LINE_AA)
                                m._put(frame, f"stop {n_}" + (" (done)" if done else ""), (int(u_) + 16, int(v_) - 12), col)
                    if floor.valid:
                        for p in list(self.pins):
                            u, v = floor.world_to_pixel([p], K, D)[0]
                            if np.isfinite(u) and np.isfinite(v):
                                cv2.drawMarker(frame, (int(u), int(v)), (0, 140, 255), cv2.MARKER_CROSS, 22, 2)
                    self.last_frame = frame
                    s = min(1.0, self.max_display_w / frame.shape[1])
                    disp = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else frame
                    rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
                    h, w = rgb.shape[:2]
                    qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888).copy()
                    self.frame_ready.emit(qimg, s)
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            self.stop_recording()
            if self.objdet is not None:
                self.objdet.stop()


# --------------------------------------------------------------------------
# Widgets
# --------------------------------------------------------------------------

class VideoView(QtWidgets.QLabel):
    hovered = pyqtSignal(float, float)
    clicked = pyqtSignal(float, float, int)

    def __init__(self):
        super().__init__("No source running")
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Ignored)
        self.setMinimumSize(480, 270)
        self.setStyleSheet("background:#111; color:#888;")
        self._img, self._scale, self._pm = None, 1.0, None

    def set_frame(self, qimg, scale):
        self._img, self._scale = qimg, scale
        self._refresh()

    def _refresh(self):
        if self._img is None:
            return
        self._pm = QtGui.QPixmap.fromImage(self._img).scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
        self.setPixmap(self._pm)

    def resizeEvent(self, e):
        self._refresh()
        super().resizeEvent(e)

    def _to_frame(self, pos):
        if self._pm is None:
            return None
        ox = (self.width() - self._pm.width()) / 2
        oy = (self.height() - self._pm.height()) / 2
        x, y = pos.x() - ox, pos.y() - oy
        if not (0 <= x < self._pm.width() and 0 <= y < self._pm.height()):
            return None
        k = self._img.width() / self._pm.width() / self._scale
        return x * k, y * k

    def mouseMoveEvent(self, e):
        p = self._to_frame(e.position())
        if p:
            self.hovered.emit(*p)

    def mousePressEvent(self, e):
        p = self._to_frame(e.position())
        if p:
            b = e.button()
            shift = bool(e.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            code = (3 if (b == Qt.MouseButton.MiddleButton or (b == Qt.MouseButton.LeftButton and shift))
                    else 1 if b == Qt.MouseButton.LeftButton else 2)
            self.clicked.emit(p[0], p[1], code)


class ArenaView(VideoView):
    """Bird's-eye arena map. Left click = goal, right click = manual start."""

    def __init__(self):
        super().__init__()
        self.setText("Arena map appears here once the 4 corner tags are set up")
        self.setMinimumSize(300, 200)


class TagStats:
    def __init__(self, maxlen=3000):
        self.hist = deque(maxlen=maxlen)   # (wall, x, y, z, tx, ty, tz)
        self.last = None

    def add(self, wall, o, truth):
        tr = truth if truth is not None else (np.nan,) * 3
        self.hist.append((wall, o["x"], o["y"], o["z"], *tr))
        self.last = (wall, o)

    def window(self, now, seconds):
        if not self.hist:
            return np.empty((0, 7))
        a = np.asarray(self.hist)
        return a[a[:, 0] >= now - seconds]


def pixel_to_floor(Rt, K, D, u, v):
    if Rt is None:
        return None
    R, t = Rt
    n = cv2.undistortPoints(np.array([[[u, v]]], np.float64), K, D).reshape(2)
    d = R.T @ np.array([n[0], n[1], 1.0])
    C = -R.T @ t
    if abs(d[2]) < 1e-9:
        return None
    s = -C[2] / d[2]
    return None if s <= 0 else C + s * d


class MainWindow(QtWidgets.QMainWindow):
    COLS = ["ID", "Name", "X m", "Y m", "Z m", "Heading°", "Tilt°", "σX mm", "σY mm", "σZ mm",
            "Hz", "Reproj px", "Err mm"]

    def __init__(self, args):
        super().__init__()
        self.setWindowTitle("AprilTag tracker test GUI")
        self.resize(1700, 950)
        self.args = args
        self.worker = self.source = self.mapper = None
        self.cfg = None
        self.stats = {}
        self.latest = None
        self.selected = None
        self.pins = []
        self.measure = None
        self.arena_setup = None
        self.fitted = False
        self._build_ui()
        self.ui_timer = QtCore.QTimer(self, interval=66, timeout=self._refresh_panels)
        self.ui_timer.start()

    # ---------------- layout ----------------
    def _build_ui(self):
        a = self.args

        def group(title, parent_layout):
            box = QtWidgets.QGroupBox(title)
            form = QtWidgets.QFormLayout(box)
            parent_layout.addWidget(box)
            return form

        def hrow(*widgets):
            row = QtWidgets.QHBoxLayout()
            for w_ in widgets:
                row.addWidget(w_)
            return row

        # Two tabs on the left: "Run" holds only what is needed to operate (camera, arena,
        # A*, objects, log); "Settings" holds calibration and tuning.
        run_page = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(run_page)
        set_page = QtWidgets.QWidget()
        sl = QtWidgets.QVBoxLayout(set_page)

        # ---------------- widgets ----------------
        self.src_combo = QtWidgets.QComboBox()
        self.src_combo.addItems(["Camera", "Video file", "Synthetic"])
        self.cam_combo = QtWidgets.QComboBox(editable=True)
        self.cam_combo.setToolTip("Pick the Sony. You can also type an index or part of a name.")
        self.conn_combo = QtWidgets.QComboBox()
        self.conn_combo.addItem("Auto (USB-C, then HDMI)", "auto")
        self.conn_combo.addItem("USB-C to C (USB Streaming)", "usb")
        self.conn_combo.addItem("HDMI to C (capture card)", "hdmi")
        self.conn_combo.setCurrentIndex(max(0, self.conn_combo.findData(a.connection)))
        self.conn_combo.currentIndexChanged.connect(lambda _: self.refresh_cameras(force_auto=True))
        self.backend_combo = QtWidgets.QComboBox()
        for label, key in (("Auto", "auto"), ("DirectShow", "dshow"), ("Media Foundation", "msmf")):
            self.backend_combo.addItem(label, key)
        self.backend_combo.setCurrentIndex(max(0, self.backend_combo.findData(a.backend)))
        self.backend_combo.setToolTip("Windows: DirectShow (Auto) matches the names in the list. "
                                      "If a device stays black, try Media Foundation and use Preview all.")
        cam_refresh = QtWidgets.QToolButton(text="⟳")
        cam_refresh.setToolTip("Rescan cameras")
        cam_refresh.clicked.connect(self.refresh_cameras)
        cam_row = QtWidgets.QWidget()
        crl = QtWidgets.QHBoxLayout(cam_row)
        crl.setContentsMargins(0, 0, 0, 0)
        crl.addWidget(self.cam_combo, 1)
        crl.addWidget(cam_refresh)
        self.res_combo = QtWidgets.QComboBox()
        self.res_combo.addItems(RESOLUTIONS)
        self.fps_spin = QtWidgets.QSpinBox(minimum=1, maximum=120, value=a.fps)
        self.video_edit = QtWidgets.QLineEdit(a.video or "")
        self.calib_edit = QtWidgets.QLineEdit(a.calib or "")
        self.config_edit = QtWidgets.QLineEdit(a.config or "")
        self.focal_spin = QtWidgets.QDoubleSpinBox(minimum=8, maximum=800, value=a.focal_mm, suffix=" mm")
        prev_btn = QtWidgets.QPushButton("Preview all")
        prev_btn.setToolTip("Grab one frame from every camera index so you can see which one is the Sony")
        prev_btn.clicked.connect(self.preview_all)
        diag_btn = QtWidgets.QPushButton("Diagnose")
        diag_btn.setToolTip("Explain why the A7 IV or capture card is not found")
        diag_btn.clicked.connect(self.diagnose)
        self.cam_tool_btns = (prev_btn, diag_btn)
        self.start_btn = QtWidgets.QPushButton("Start")
        self.start_btn.clicked.connect(self.toggle_run)
        if a.synthetic or getattr(a, "synthetic_raw", False):
            self.src_combo.setCurrentText("Synthetic")
        elif a.video:
            self.src_combo.setCurrentText("Video file")
        res = f"{a.width}x{a.height}"
        if res not in RESOLUTIONS:
            self.res_combo.addItem(res)
        self.res_combo.setCurrentText(res)

        # tag detection
        self.engine_combo = QtWidgets.QComboBox()
        self.engine_combo.addItems(["auto", "pupil", "aruco"])
        self.scale_spin = QtWidgets.QDoubleSpinBox(minimum=0.25, maximum=1.0, singleStep=0.05, value=1.0)
        self.decim_spin = QtWidgets.QDoubleSpinBox(minimum=1.0, maximum=4.0, singleStep=0.25, value=1.5)
        self.decim_spin.setToolTip("Fast pass only. Small / far tags are still found: a full-resolution "
                                   "background pass discovers them and they are tracked in upscaled crops.")
        self.smooth_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal, minimum=5, maximum=100, value=50)
        self.smooth_label = QtWidgets.QLabel("0.50")
        sm = QtWidgets.QHBoxLayout()
        sm.addWidget(self.smooth_slider)
        sm.addWidget(self.smooth_label)
        self.lock_chk = QtWidgets.QCheckBox("Lock floor pose")
        self.grid_chk = QtWidgets.QCheckBox("Floor grid overlay", checked=True)
        for w in (self.engine_combo,):
            w.currentTextChanged.connect(self._detector_changed)
        for w in (self.scale_spin, self.decim_spin):
            w.valueChanged.connect(self._detector_changed)
        self.smooth_slider.valueChanged.connect(self._smoothing_changed)
        self.lock_chk.toggled.connect(self._lock_toggled)
        self.grid_chk.toggled.connect(lambda v: self.worker and setattr(self.worker, "show_grid", v))

        # arena calibration
        self.ar_w = QtWidgets.QDoubleSpinBox(minimum=0, maximum=20, decimals=3, singleStep=0.001, suffix=" m")
        self.ar_h = QtWidgets.QDoubleSpinBox(minimum=0, maximum=20, decimals=3, singleStep=0.001, suffix=" m")
        for w_ in (self.ar_w, self.ar_h):
            w_.setSpecialValueText("not measured")
            w_.setToolTip("Tape-measured distance between tag CENTERS. Needed for a precise plane.")
        self.ar_w_label = QtWidgets.QLabel("Tape origin→?")
        self.ar_h_label = QtWidgets.QLabel("Tape origin→?")
        self.ar_layout = None
        self.ar_apply_btn = QtWidgets.QPushButton("Apply tape measurements")
        self.ar_apply_btn.setToolTip("Recompute the plane from the frames already collected, using the tape "
                                     "distances (and estimate the focal length if there is no calibration).")
        self.ar_apply_btn.clicked.connect(self.apply_tape)
        self.ar_floor_chk = QtWidgets.QCheckBox("Tags lie flat on the floor (plane mode)", checked=True)
        self.ar_floor_chk.setToolTip("Locate tags by intersecting camera rays with the floor plane: "
                                     "millimeter precision for anything lying on the floor. Turn off for "
                                     "tags at unknown heights (then full 3D pose is used).")
        self.ar_floor_chk.toggled.connect(self._plane_mode_changed)
        self.ar_robot_h = QtWidgets.QDoubleSpinBox(minimum=0, maximum=2, decimals=3, singleStep=0.005, suffix=" m")
        self.ar_robot_h.setSpecialValueText("auto")
        self.ar_robot_h.setToolTip("Height of the robot's tag above the floor. With it, plane mode "
                                   "locates the robot to ~mm. auto = 3D pose if the tag is above the floor.")
        self.ar_robot_h.valueChanged.connect(self._plane_mode_changed)
        self.ar_adaptive = QtWidgets.QCheckBox("Adaptive pose (auto lock / re-lock)", checked=True)
        self.ar_adaptive.setToolTip("Locks the floor pose when the camera is still, unlocks and re-locks "
                                    "automatically when it is moved or bumped.")
        self.ar_adaptive.toggled.connect(self._adaptive_changed)

        # map and A*
        self.ar_robot = QtWidgets.QLineEdit(placeholderText="global ID, e.g. 90")
        self.ar_radius = QtWidgets.QDoubleSpinBox(minimum=0.0, maximum=1.0, decimals=3, singleStep=0.005,
                                                  value=0.10, suffix=" m")
        self.ar_res = QtWidgets.QDoubleSpinBox(minimum=0, maximum=20, decimals=1, value=0, suffix=" mm/px")
        self.ar_res.setSpecialValueText("auto")
        self.ar_res.setToolTip("Map cell size. auto = matched to how finely the camera sees the arena.")
        self.ar_range = QtWidgets.QDoubleSpinBox(minimum=0.5, maximum=20, decimals=1, value=4.0, suffix=" m")
        self.ar_range.setToolTip("Map the floor out to this horizontal distance from the camera, "
                                 "including beyond the corner tags.")
        self.ar_unknown = QtWidgets.QCheckBox("Unseen floor is blocked for A*", checked=True)
        self.ar_unknown.toggled.connect(self._arena_params_changed)
        self.ar_range.valueChanged.connect(self._arena_params_changed)
        self.ar_plan = QtWidgets.QDoubleSpinBox(minimum=0.5, maximum=10, decimals=1, value=1.0, suffix=" cm")
        self.ar_thr = QtWidgets.QSpinBox(minimum=5, maximum=150, value=30)
        self.ar_thr.setToolTip("Color difference from the empty floor that counts as an obstacle")
        reset_map = QtWidgets.QPushButton("Reset map")
        reset_map.clicked.connect(self.reset_global_map)
        save_map = QtWidgets.QPushButton("Save map")
        save_map.clicked.connect(self.save_arena_map)
        self.ar_auto = QtWidgets.QCheckBox("Replan continuously", checked=True)
        self.ar_auto.toggled.connect(lambda v: self.worker and setattr(self.worker, "auto_replan", v))
        self.ar_udp = QtWidgets.QLineEdit(placeholderText="robot UDP out, e.g. 127.0.0.1:5005")
        self.ar_udp.editingFinished.connect(self._udp_changed)
        for w_ in (self.ar_radius, self.ar_res, self.ar_plan, self.ar_thr):
            w_.valueChanged.connect(self._arena_params_changed)
        self.ar_robot.editingFinished.connect(self._arena_params_changed)

        # ================= Run tab =================
        f = group("Camera", cl)
        f.addRow("Type", self.src_combo)
        f.addRow("Connection", self.conn_combo)
        f.addRow("Camera", cam_row)
        f.addRow(self.start_btn)

        f = group("Floor plane (4 pivot tags)", cl)
        self.pivot_label = QtWidgets.QLabel("Pivots: the 4 largest tags on the print sheet.")
        self.pivot_label.setWordWrap(True)
        self.pivot_label.setToolTip("Only these 4 tags define the floor plane. Any other tag in view is "
                                    "tracked as an object and can never shift the floor.")
        self.ar_setup_btn = QtWidgets.QPushButton("Set up arena (2 s) / redo")
        self.ar_setup_btn.setToolTip("Keep the camera still. Uses the 4 pivot tags only (other tags may be in "
                                     "view), builds the arena frame, locks the pose, saves arena.json. "
                                     "Runs by itself when all 4 pivots are in view.")
        self.ar_setup_btn.clicked.connect(self.start_arena_setup)
        self.ar_bg_btn = QtWidgets.QPushButton("Capture empty floor")
        self.ar_bg_btn.setToolTip("Optional. Clear the arena (robot may stay), then press: anything that later "
                                  "differs from this image also becomes an obstacle.")
        self.ar_bg_btn.clicked.connect(self.capture_background)
        f.addRow(self.pivot_label)
        f.addRow(self.ar_setup_btn)
        f.addRow(self.ar_bg_btn)

        f = group("A* path", cl)
        self.pair_start = QtWidgets.QComboBox(editable=True)
        self.pair_goal = QtWidgets.QComboBox(editable=True)
        for cb in (self.pair_start, self.pair_goal):
            cb.setToolTip("Any tag (0-101) or detected object (1000+) seen on the floor.")
        self.pair_btn = QtWidgets.QPushButton("Plan path", checkable=True)
        self.pair_btn.setToolTip("A* from the start to the goal, replanned live as either moves. The start "
                                 "is treated as the robot: obstacles are inflated by the robot radius, and a "
                                 "steering command is computed from its heading.")
        self.pair_btn.toggled.connect(self.toggle_pair)
        swap = QtWidgets.QPushButton("Swap")
        swap.clicked.connect(self._swap_pair)
        clear_goal = QtWidgets.QPushButton("Clear goal")
        clear_goal.clicked.connect(self.clear_goal)
        hint = QtWidgets.QLabel("Or click the Arena / A* view: left = goal, right = start.")
        hint.setWordWrap(True)
        self.pair_info = QtWidgets.QLabel("")
        self.pair_info.setWordWrap(True)
        self.stops_list = QtWidgets.QListWidget()
        self.stops_list.setToolTip("Stops in order, between start and goal. Drag to reorder.")
        self.stops_list.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self.stops_list.setMaximumHeight(78)
        self.stops_list.model().rowsMoved.connect(lambda *_: self._sync_stops())
        self.stop_combo = QtWidgets.QComboBox(editable=True)
        self.stop_combo.setToolTip("A tag or detected object to stop at (followed live).")
        add_stop = QtWidgets.QPushButton("Add stop")
        add_stop.clicked.connect(self.add_stop)
        del_stop = QtWidgets.QPushButton("Remove")
        del_stop.setToolTip("Remove the selected stop (or the last one)")
        del_stop.clicked.connect(self.remove_stop)
        clr_stop = QtWidgets.QPushButton("Clear")
        clr_stop.clicked.connect(self.clear_stops)
        f.addRow("Start (robot)", self.pair_start)
        f.addRow("Stops", self.stops_list)
        f.addRow(self.stop_combo, hrow(add_stop, del_stop, clr_stop))
        f.addRow("Goal", self.pair_goal)
        f.addRow(self.pair_btn)
        f.addRow(hrow(swap, clear_goal))
        hint.setText("Arena / A* view: left click = goal, right = start, shift+click (or middle) = add stop.")
        f.addRow(hint)
        f.addRow(self.pair_info)

        f = group("Objects without tags", cl)
        self.det_chk = QtWidgets.QCheckBox("Detect objects (any object, no class list)", checked=True)
        self.det_chk.setToolTip("Every part of the floor that does not look like the floor (compared with the "
                                "rest of the floor in the same frame) becomes an object and an A* obstacle. "
                                "No model, no class list, no stored reference. Optional YOLO in Settings. "
                                "and places it on the floor map. Needs: pip install ultralytics.")
        self.det_chk.toggled.connect(self.toggle_detection)
        self.det_status = QtWidgets.QLabel("off")
        self.det_status.setWordWrap(True)
        f.addRow(self.det_chk)
        f.addRow(self.det_status)

        self.log = QtWidgets.QPlainTextEdit(readOnly=True)
        self.log.setMaximumBlockCount(500)
        cl.addWidget(QtWidgets.QLabel("Log"))
        self.log.setMinimumHeight(220)
        cl.addWidget(self.log, 1)

        # ================= Settings tab =================
        f = group("Camera", sl)
        f.addRow("Backend", self.backend_combo)
        f.addRow(hrow(prev_btn, diag_btn))
        f.addRow("Resolution", self.res_combo)
        f.addRow("FPS", self.fps_spin)
        f.addRow("Video", self._with_browse(self.video_edit, "Video (*.mp4 *.mov *.avi *.mkv)"))
        f.addRow("Calib .npz", self._with_browse(self.calib_edit, "Calibration (*.npz)"))
        f.addRow("Config .json", self._with_browse(self.config_edit, "Config (*.json)"))
        f.addRow("Focal (no calib)", self.focal_spin)

        f = group("Floor calibration", sl)
        f.addRow(self.ar_w_label, self.ar_w)
        f.addRow(self.ar_h_label, self.ar_h)
        f.addRow(self.ar_apply_btn)
        f.addRow(self.ar_floor_chk)
        f.addRow(self.ar_adaptive)
        f.addRow(self.lock_chk)

        f = group("Robot", sl)
        f.addRow("Robot tag", self.ar_robot)
        f.addRow("Robot tag height", self.ar_robot_h)
        f.addRow("Robot radius", self.ar_radius)
        f.addRow("UDP out", self.ar_udp)

        f = group("Map and A*", sl)
        f.addRow("Map resolution", self.ar_res)
        f.addRow("Map range", self.ar_range)
        f.addRow("A* cell", self.ar_plan)
        f.addRow("Obstacle thresh.", self.ar_thr)
        f.addRow(self.ar_auto)
        f.addRow(self.ar_unknown)
        copy_wp = QtWidgets.QPushButton("Copy waypoints")
        copy_wp.setToolTip("Copy the current path (arena meters) as JSON")
        copy_wp.clicked.connect(self.copy_waypoints)
        f.addRow(hrow(reset_map, save_map, copy_wp))

        f = group("Tag detection", sl)
        f.addRow("Engine", self.engine_combo)
        f.addRow("Detect scale", self.scale_spin)
        f.addRow("Decimate (pupil)", self.decim_spin)
        f.addRow("Smoothing α", sm)
        f.addRow(self.grid_chk)

        f = group("Object detection", sl)
        self.det_model = QtWidgets.QComboBox(editable=True)
        for name, desc in od.MODEL_CHOICES:
            self.det_model.addItem(name)
            self.det_model.setItemData(self.det_model.count() - 1, desc, Qt.ItemDataRole.ToolTipRole)
        self.det_model.setToolTip("Prompt-free models find any object without a class list. "
                                  "You can also type the path of your own .pt (its labels are ignored).")
        mb = QtWidgets.QToolButton(text="…")
        mb.clicked.connect(lambda: self.det_model.setEditText(
            QtWidgets.QFileDialog.getOpenFileName(self, "Model", "", "Models (*.pt *.onnx)")[0]
            or self.det_model.currentText()))
        mrow = QtWidgets.QWidget()
        ml = QtWidgets.QHBoxLayout(mrow)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.addWidget(self.det_model, 1)
        ml.addWidget(mb)
        self.det_conf = QtWidgets.QDoubleSpinBox(minimum=0.05, maximum=0.95, singleStep=0.05, value=0.15)
        self.det_conf.setToolTip("Proposal confidence. Prompt-free scores are low; false proposals are "
                                 "filtered by floor geometry and must persist for 2 detector cycles.")
        self.det_conf.valueChanged.connect(self._det_reconfig)
        self.det_size = QtWidgets.QComboBox()
        self.det_size.addItems(["640", "960", "1280"])
        self.det_size.setToolTip("Inference size: larger finds smaller/farther objects but is slower")
        self.det_size.currentTextChanged.connect(self._det_reconfig)
        self.det_obst = QtWidgets.QCheckBox("Detected objects are A* obstacles", checked=True)
        self.det_obst.toggled.connect(lambda v: self.worker and setattr(self.worker, "det_obstacles", v))
        self.det_app_z = QtWidgets.QDoubleSpinBox(minimum=2.0, maximum=15.0, singleStep=0.5, value=5.0)
        self.det_app_z.setToolTip("How unlike the floor something must look to be an object (robust std devs). "
                                  "Lower = more sensitive (finds faint objects, more false hits).")
        self.det_app_z.valueChanged.connect(self._det_reconfig)
        self.det_yolo = QtWidgets.QCheckBox("Also use YOLO proposals (slow on CPU)")
        self.det_yolo.setToolTip("Adds a prompt-free YOLOE model's boxes (class-agnostic) on top of the "
                                 "appearance detector. Needs: pip install ultralytics.")
        self.det_yolo.toggled.connect(lambda _: self.det_chk.isChecked() and self.toggle_detection(True))
        f.addRow("Floor contrast", self.det_app_z)
        f.addRow(self.det_yolo)
        f.addRow("Model", mrow)
        f.addRow("Confidence", self.det_conf)
        f.addRow("Image size", self.det_size)
        f.addRow(self.det_obst)

        f = group("Testing", sl)
        self.window_spin = QtWidgets.QDoubleSpinBox(minimum=0.5, maximum=30, value=2.0, suffix=" s")
        self.meas_spin = QtWidgets.QDoubleSpinBox(minimum=0.5, maximum=60, value=3.0, suffix=" s")
        self.truth_edit = QtWidgets.QLineEdit(placeholderText="true x, y, z (m), optional")
        self.meas_btn = QtWidgets.QPushButton("Measure selected tag")
        self.meas_btn.clicked.connect(self.start_measure)
        self.rec_btn = QtWidgets.QPushButton("Start CSV recording", checkable=True)
        self.rec_btn.toggled.connect(self.toggle_record)
        snap = QtWidgets.QPushButton("Snapshot")
        snap.clicked.connect(self.snapshot)
        reset = QtWidgets.QPushButton("Reset")
        reset.setToolTip("Clear jitter stats and map trails")
        reset.clicked.connect(self.reset_stats)
        fit = QtWidgets.QPushButton("Fit map")
        fit.clicked.connect(lambda: self.map.autoRange())
        f.addRow("Jitter window", self.window_spin)
        f.addRow("Measure for", self.meas_spin)
        f.addRow(self.truth_edit)
        f.addRow(self.meas_btn)
        f.addRow(self.rec_btn)
        f.addRow(hrow(snap, reset, fit))
        sl.addStretch(1)

        for cb in run_page.findChildren(QtWidgets.QComboBox) + set_page.findChildren(QtWidgets.QComboBox):
            # long entries (camera names, model paths) must not widen the panel past its edge
            cb.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            cb.setMinimumContentsLength(8)
        self.ctrl_tabs = QtWidgets.QTabWidget()
        for page, title in ((run_page, "Run"), (set_page, "Settings")):
            scroll = QtWidgets.QScrollArea()
            scroll.setWidget(page)
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self.ctrl_tabs.addTab(scroll, title)
        self.ctrl_tabs.setFixedWidth(340)

        # Video
        self.video = VideoView()
        self.video.hovered.connect(self._hover)
        self.video.clicked.connect(self._click)

        # Map
        pg.setConfigOptions(antialias=True)
        self.map = pg.PlotWidget(title="Top-down map (world frame)")
        self.map.setAspectLocked(True)
        self.map.showGrid(x=True, y=True, alpha=0.25)
        self.map.setLabel("bottom", "x (m)")
        self.map.setLabel("left", "y (m)")
        for ax in ("left", "bottom"):
            self.map.getAxis(ax).enableAutoSIPrefix(False)
        self.map.addItem(pg.PlotDataItem([0, 0.2], [0, 0], pen=pg.mkPen("r", width=3)))
        self.map.addItem(pg.PlotDataItem([0, 0], [0, 0.2], pen=pg.mkPen("g", width=3)))
        self.floor_items = []
        self.cam_scatter = pg.ScatterPlotItem(size=12, brush=pg.mkBrush("w"))
        self.cam_heading = pg.PlotDataItem(pen=pg.mkPen("w", width=1))
        self.obj_scatter = pg.ScatterPlotItem(size=11, brush=pg.mkBrush(0, 220, 0), pen=pg.mkPen("k"))
        self.sel_scatter = pg.ScatterPlotItem(size=20, brush=None, pen=pg.mkPen("y", width=2))
        self.truth_scatter = pg.ScatterPlotItem(size=14, brush=None, pen=pg.mkPen(255, 140, 0, width=2))
        self.pin_scatter = pg.ScatterPlotItem(size=14, symbol="+", pen=pg.mkPen(255, 140, 0, width=2))
        self.hover_scatter = pg.ScatterPlotItem(size=8, symbol="x", pen=pg.mkPen("c", width=2))
        for it in (self.cam_heading, self.cam_scatter, self.truth_scatter, self.obj_scatter,
                   self.sel_scatter, self.pin_scatter, self.hover_scatter):
            self.map.addItem(it)
        self.trails, self.labels = {}, {}

        # Time plot
        self.plot = pg.PlotWidget(title="Selected tag position")
        self.plot.addLegend(offset=(5, 5))
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("bottom", "t (s)")
        self.plot.setLabel("left", "m")
        self.plot.getAxis("left").enableAutoSIPrefix(False)
        self.curves = {k: self.plot.plot(pen=pg.mkPen(c, width=2), name=k)
                       for k, c in (("x", "r"), ("y", "g"), ("z", (80, 140, 255)))}
        self.truth_curves = {k: self.plot.plot(pen=pg.mkPen(c, width=1, style=Qt.PenStyle.DashLine))
                             for k, c in (("x", "r"), ("y", "g"), ("z", (80, 140, 255)))}

        # Table
        self.table = QtWidgets.QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.cellClicked.connect(self._row_clicked)

        self.arena_view = ArenaView()
        self.arena_view.clicked.connect(self._arena_click)
        self.arena_view.hovered.connect(self._arena_hover)
        self.arena_status = QtWidgets.QLabel("")
        self.arena_status.setWordWrap(True)
        arena_tab = QtWidgets.QWidget()
        al = QtWidgets.QVBoxLayout(arena_tab)
        al.setContentsMargins(0, 0, 0, 0)
        al.addWidget(self.arena_view, 1)
        al.addWidget(self.arena_status)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.map, "Map")
        self.tabs.addTab(arena_tab, "Arena / A*")
        right = QtWidgets.QSplitter(Qt.Orientation.Vertical)
        right.addWidget(self.tabs)
        right.addWidget(self.plot)
        right.setSizes([500, 300])
        top = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        top.addWidget(self.video)
        top.addWidget(right)
        top.setSizes([900, 520])
        center = QtWidgets.QSplitter(Qt.Orientation.Vertical)
        center.addWidget(top)
        center.addWidget(self.table)
        center.setSizes([700, 220])

        root = QtWidgets.QWidget()
        hl = QtWidgets.QHBoxLayout(root)
        hl.setContentsMargins(4, 4, 4, 4)
        hl.addWidget(self.ctrl_tabs)
        hl.addWidget(center, 1)
        self.setCentralWidget(root)

        self.status = QtWidgets.QLabel("Stopped")
        self.hover_label = QtWidgets.QLabel("")
        self.statusBar().addWidget(self.status, 1)
        self.statusBar().addPermanentWidget(self.hover_label)
        self.refresh_cameras(prefer=self.args.camera)

    def _with_browse(self, edit, filt):
        w = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(edit)
        b = QtWidgets.QToolButton(text="…")
        b.clicked.connect(lambda: edit.setText(
            QtWidgets.QFileDialog.getOpenFileName(self, "Choose file", "", filt)[0] or edit.text()))
        h.addWidget(b)
        return w

    def refresh_cameras(self, prefer=None, force_auto=False):
        """Fill the camera list. Preselect, in order: a camera named on the command line,
        the camera used last time, then auto (Sony over USB, else the HDMI capture card).
        Built-in webcams and virtual cameras are never auto-selected."""
        prefer = prefer if isinstance(prefer, str) else "auto"
        connection = self.conn_combo.currentData()
        settings = QtCore.QSettings("apriltag_floor_mapper", "tracker_gui")
        last = "" if force_auto else settings.value(f"camera_name_{connection}", "", type=str)
        self.cam_combo.clear()
        try:
            devices = m.enumerate_cameras()
        except Exception as e:
            devices = None
            self.log_msg(f"Camera scan failed: {e}")
        if devices is None:
            self.cam_combo.setEditText(prefer if prefer not in ("auto", "sony") else "1")
            if sys.platform.startswith("win"):
                self.log_msg("Install pygrabber (pip install pygrabber) to list cameras by name. "
                             "Otherwise type the capture card's index.")
            return
        labels = {"sony": "Sony", "capture": "capture card", "skip": "webcam/virtual", "other": "camera"}
        for idx, name in devices:
            self.cam_combo.addItem(f"{idx}: {name}  ({labels[m.classify_camera(name)]})", idx)

        choice, reason = None, ""
        if prefer.lower() not in ("auto", "sony"):
            for i, (idx, name) in enumerate(devices):
                if (prefer.isdigit() and idx == int(prefer)) or prefer.lower() in name.lower():
                    choice, reason = i, "from --camera"
                    break
        if choice is None and last:
            names = [n for _, n in devices]
            if names.count(last) == 1:
                choice, reason = names.index(last), "used last time"
        if choice is None:
            pick = m.auto_pick_camera(devices, connection)
            if pick:
                choice = [idx for idx, _ in devices].index(pick[0])
                reason = pick[2]
        if choice is None:
            self.cam_combo.setCurrentIndex(-1)
            self.cam_combo.setEditText("")
            want = {"usb": "the A7 IV over USB-C", "hdmi": "an HDMI capture card",
                    "auto": "the A7 IV or a capture card"}[connection]
            self.log_msg(f"Did not find {want}. Windows sees: "
                         + (", ".join(f"{i}: {n}" for i, n in devices) or "no cameras")
                         + ". Press Diagnose for the reason, or Preview all to pick by picture.")
            return
        self.cam_combo.setCurrentIndex(choice)
        self.log_msg(f"Camera: {devices[choice][1]} (index {devices[choice][0]}, {reason})")

    def _camera_names(self):
        try:
            return dict(m.enumerate_cameras() or [])
        except Exception:
            return {}

    def preview_all(self):
        if self.worker:
            self.log_msg("Stop the running source first (the camera can only be opened once).")
            return
        backend = self.backend_combo.currentData()
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            probes = m.probe_cameras(8, backend)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        if not probes:
            self.log_msg("No camera index could be opened. Press Diagnose.")
            return
        names = self._camera_names() if backend in ("auto", "dshow") else {}
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Pick the camera that shows the Sony's view")
        grid = QtWidgets.QGridLayout(dlg)
        for k, p in enumerate(probes):
            btn = QtWidgets.QToolButton()
            btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            thumb = np.zeros((225, 400, 3), np.uint8) if p["frame"] is None else \
                cv2.resize(p["frame"], (400, 225), interpolation=cv2.INTER_AREA)
            rgb = np.ascontiguousarray(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB))
            qimg = QtGui.QImage(rgb.data, 400, 225, 1200, QtGui.QImage.Format.Format_RGB888).copy()
            btn.setIcon(QtGui.QIcon(QtGui.QPixmap.fromImage(qimg)))
            btn.setIconSize(QtCore.QSize(400, 225))
            btn.setText(f"index {p['index']}: {names.get(p['index'], 'unknown name')}\n"
                        f"{p['width']}x{p['height']}" + ("   NO SIGNAL / BLACK" if p["black"] else ""))
            btn.clicked.connect(lambda _, i=p["index"]: (self._select_index(i), dlg.accept()))
            grid.addWidget(btn, k // 3, k % 3)
        dlg.exec()

    def _select_index(self, idx):
        for i in range(self.cam_combo.count()):
            if self.cam_combo.itemData(i) == idx:
                self.cam_combo.setCurrentIndex(i)
                break
        else:
            self.cam_combo.setCurrentIndex(-1)
            self.cam_combo.setEditText(str(idx))
        self.log_msg(f"Selected camera index {idx}")

    def diagnose(self):
        if self.worker:
            self.log_msg("Stop the running source first.")
            return
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            text, sheet = m.diagnose_cameras(self.backend_combo.currentData())
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.log_msg("Camera diagnosis:\n" + text)
        QtWidgets.QMessageBox.information(self, "Camera diagnosis", text)
        self.refresh_cameras()

    def _remember_camera(self):
        i = self.cam_combo.currentIndex()
        if i >= 0 and self.cam_combo.itemData(i) is not None:
            name = self.cam_combo.itemText(i).split(": ", 1)[1].rsplit("  (", 1)[0]
            QtCore.QSettings("apriltag_floor_mapper", "tracker_gui").setValue(
                f"camera_name_{self.conn_combo.currentData()}", name)

    def _camera_spec(self):
        """Index from the selected list entry, or whatever the user typed."""
        txt = self.cam_combo.currentText().strip()
        i = self.cam_combo.findText(txt)
        if i >= 0 and self.cam_combo.itemData(i) is not None:
            return str(self.cam_combo.itemData(i))
        head = txt.split(":", 1)[0].strip()
        return head if head.isdigit() else txt

    def log_msg(self, s):
        self.log.appendPlainText(time.strftime("%H:%M:%S ") + s)

    # ---------------- run control ----------------
    def toggle_run(self):
        if self.worker:
            self.stop()
        else:
            self.start()

    def _make_detector(self):
        return m.make_detector(self.cfg, engine=self.engine_combo.currentText(),
                               detect_scale=self.scale_spin.value(), nthreads=4,
                               decimate=self.decim_spin.value())

    def start(self):
        try:
            mode = self.src_combo.currentText()
            if mode == "Synthetic":
                scene = SyntheticScene(raw=getattr(self.args, "synthetic_raw", False))
                self.scene = scene
                self.cfg, K, D = scene.cfg, scene.K_app, scene.D
                self.source = PacedSource(scene.read, self.fps_spin.value() if self.fps_spin.value() <= 60 else 30)
            else:
                cfg_path = self.config_edit.text().strip() or None
                if cfg_path is None and os.path.exists(am.DEFAULT_ARENA_FILE):
                    cfg_path = os.path.abspath(am.DEFAULT_ARENA_FILE)
                    self.config_edit.setText(cfg_path)
                    self.log_msg(f"Using saved arena {cfg_path}")
                self.cfg = m.load_config(cfg_path)
                if not self.cfg["sheet_loaded"]:
                    self.log_msg(f"WARNING: {m.DEFAULT_SHEET} not found. Tag sizes default to "
                                 f"{self.cfg['tag_size']} m, distances will be wrong.")
                if mode == "Camera":
                    w, h = (int(v) for v in self.res_combo.currentText().split("x"))
                    spec = self._camera_spec()
                    if not spec:
                        raise RuntimeError("Choose a camera in the Camera list first.")
                    idx, name = m.resolve_camera(spec, self.conn_combo.currentData())
                    backend = self.backend_combo.currentData()
                    self.source = m.CameraStream(idx, w, h, self.fps_spin.value(), m.backend_id(backend))
                    if backend == "msmf":
                        name = None   # names follow DirectShow order; under MSMF trust the index only
                    self.log_msg(f"Opened camera {idx}" + (f": {name}" if name else "")
                                 + f" at {self.source.width}x{self.source.height}")
                    if (self.source.width, self.source.height) != (w, h):
                        self.log_msg(f"Note: asked for {w}x{h}, the device gives "
                                     f"{self.source.width}x{self.source.height}.")
                        modes = m.device_formats(name) if name else None
                        if modes:
                            self.log_msg("   The camera offers only: " + ", ".join(
                                f"{a_}x{b_}@{c_}" for a_, b_, c_ in modes) + ". The resolution is set ON the camera."
                                + (" A7 IV: MENU > Setup > USB > USB Streaming > Output Res/Frame Rate > 1080p "
                                   "(or 4K); tags far away need the pixels." if m.classify_camera(name) == "sony" else ""))
                    if self.conn_combo.currentData() == "usb" and name and m.classify_camera(name) != "sony":
                        self.log_msg(f"Note: '{name}' is not the Sony. Check the Camera list.")
                    if name and m.classify_camera(name) == "capture":
                        self.log_msg("HDMI tip: set the A7 IV's Setup > External Output > HDMI Info. "
                                     "Display to Off so menus are not drawn into the image. "
                                     "Recalibrate for this input: HDMI framing differs from USB streaming.")
                    self._remember_camera()
                else:
                    read, fps = video_file_reader(self.video_edit.text().strip())
                    self.source = PacedSource(read, fps)
                ns = SimpleNamespace(calib=self.calib_edit.text().strip() or None,
                                     focal_mm=self.focal_spin.value(), sensor_width_mm=35.9)
                K, D = m.load_intrinsics(ns, self.source.width, self.source.height)
                if not ns.calib:
                    self.log_msg("No calibration file: using approximate intrinsics from focal length.")
            # Only the 4 pivot tags (largest on the sheet) may define the floor plane. Saved floor
            # tags beyond them (old learned anchors, an old survey) are dropped: they may come from
            # another coordinate frame or have been moved since, and would bend the plane.
            piv = am.arena_settings(self.cfg)["corner_ids"]
            extra = sorted(set(self.cfg["floor_tags"]) - set(piv))
            if extra:
                self.log_msg(f"Ignoring saved floor tags {extra}: only the pivot tags {piv} define the "
                             "floor plane. Those tags are tracked as objects.")
            self.cfg["floor_tags"] = {k: v for k, v in self.cfg["floor_tags"].items() if k in piv}
            if len(self.cfg["floor_tags"]) < 4:
                self.log_msg(f"Pivot tags {piv}: lay them flat at the arena corners. The floor plane is "
                             "set up automatically once all 4 are in view (other tags may be in view too).")
            self.has_calib = bool((mode == "Synthetic" and not self.scene.raw) or self.calib_edit.text().strip())
            a_ = (self.cfg or {}).get("arena") or {}
            if not self.has_calib and a_.get("focal_px") and mode != "Synthetic":
                fw, fh = a_.get("focal_at_px", [self.source.width, self.source.height])
                K = K.copy()
                K[0, 0] = K[1, 1] = float(a_["focal_px"]) * self.source.width / fw
                D = np.zeros(5)
                D[0] = float(a_.get("k1") or 0.0)
                self.log_msg(f"Starting from the lens model of the last arena setup: f {K[0, 0]:.0f} px, "
                             f"k1 {D[0]:+.3f} (refitted as soon as the 4 tags are seen)")
            self.K0 = np.array(K, np.float64)
            self.D0 = np.asarray(D, np.float64).ravel().copy() if np.size(D) else np.zeros(5)
            det = self._make_detector()
            self.mapper = m.Mapper(self.cfg, K, D, det, smoothing=self.smooth_slider.value() / 100)
        except Exception as e:
            if self.source:
                self.source.release()
                self.source = None
            QtWidgets.QMessageBox.critical(self, "Could not start", str(e))
            return

        had_arena = "arena" in self.cfg
        self._auto_arena_tried = False
        self._corner_streak = 0
        self.cfg.setdefault("tags_on_floor", self.ar_floor_chk.isChecked())
        self.cfg["arena"] = am.arena_settings(self.cfg)
        self._arena_settings_to_ui(self.cfg["arena"])
        amap = self._make_amap() if had_arena else None

        self.reset_stats()
        self._build_floor_items()
        self.fitted = False
        self.worker = Worker(self.source, self.mapper)
        self.worker.show_grid = self.grid_chk.isChecked()
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.result_ready.connect(self._on_result)
        self.worker.failed.connect(self._on_failed)
        self.worker.arena_ready.connect(self._on_arena)
        self._plane_mode_changed()
        self.mapper.floor.auto_lock = self.ar_adaptive.isChecked()
        self.mapper.floor.max_reproj_px = self._fit_limit()
        self.worker.auto_replan = self.ar_auto.isChecked()
        if amap is not None:
            self.worker.amap = amap
            self.log_msg("Arena loaded: press Set up arena to re-measure and lock, or Capture empty floor.")
        self._udp_changed()
        self._sync_stops()
        if self.det_chk.isChecked():          # on by default; kept across Stop / Start
            self.toggle_detection(True)
        self.worker.start()
        self.start_btn.setText("Stop")
        self.src_combo.setEnabled(False)
        for b in self.cam_tool_btns:
            b.setEnabled(False)
        self.log_msg(f"Started {mode} {self.source.width}x{self.source.height}, detector {det.description}, "
                     f"floor tags {sorted(self.cfg['floor_tags'])}")

    def stop(self):
        if self.rec_btn.isChecked():
            self.rec_btn.setChecked(False)
        if self.worker:
            self.worker.requestInterruption()
            self.worker.wait(2000)
            self.worker = None
        if self.source:
            self.source.release()
            self.source = None
        if self.mapper:
            _DETECTOR_KEEPALIVE.append(self.mapper.detector)
        self.lock_chk.setChecked(False)
        self.det_status.setText("on (starts with the camera)" if self.det_chk.isChecked() else "off")
        self.start_btn.setText("Start")
        self.src_combo.setEnabled(True)
        for b in self.cam_tool_btns:
            b.setEnabled(True)
        self.status.setText("Stopped")
        self.log_msg("Stopped")

    def _on_failed(self, tb):
        self.log_msg("Worker error:\n" + tb)
        self.stop()

    def closeEvent(self, e):
        self.stop()
        super().closeEvent(e)

    # ---------------- settings ----------------
    def _detector_changed(self):
        if self.worker and self.cfg:
            try:
                new_det = self._make_detector()
                _DETECTOR_KEEPALIVE.append(new_det)
                self.worker.pending_detector = new_det
                sc, dc = self.scale_spin.value(), self.decim_spin.value()
                self.log_msg(f"Detector: {self.engine_combo.currentText()} scale {sc:.2f} decimate {dc:.1f}")
                if sc < 1 and dc > 1 and new_det.engine == "pupil":
                    self.log_msg(f"Note: scale and decimate stack (about {dc / sc:.0f}x downsampling). "
                                 "Small or distant tags may drop out; set decimate to 1.")
            except Exception as e:
                self.log_msg(f"Detector error: {e}")

    def _smoothing_changed(self, v):
        self.smooth_label.setText(f"{v / 100:.2f}")
        if self.mapper:
            self.mapper.smoother.alpha = v / 100

    def _lock_toggled(self, on):
        if not self.mapper:
            return
        if on and not self.mapper.floor.valid:
            self.log_msg("Cannot lock: floor not found yet.")
            self.lock_chk.setChecked(False)
            return
        self.mapper.floor.locked = on
        self.log_msg("Floor pose " + ("locked" if on else "unlocked"))

    def toggle_record(self, on):
        if not self.worker:
            self.rec_btn.setChecked(False)
            return
        if on:
            path = os.path.abspath(time.strftime("tracking_%Y%m%d_%H%M%S.csv"))
            self.worker.start_recording(path)
            self.rec_btn.setText("Stop recording")
            self.log_msg(f"Recording to {path}")
        else:
            self.worker.stop_recording()
            self.rec_btn.setText("Start CSV recording")
            self.log_msg("Recording stopped")

    def snapshot(self):
        if self.worker and self.worker.last_frame is not None:
            path = os.path.abspath(time.strftime("snapshot_%Y%m%d_%H%M%S.png"))
            cv2.imwrite(path, self.worker.last_frame)
            self.log_msg(f"Saved {path}")

    def reset_stats(self):
        self.stats.clear()
        for it in list(self.trails.values()) + list(self.labels.values()):
            self.map.removeItem(it)
        self.trails.clear()
        self.labels.clear()

    # ---------------- arena / A* ----------------
    def _arena_ui_to_settings(self, base=None):
        a = am.arena_settings({"arena": dict(base or {}), "tag_sizes": (self.cfg or {}).get("tag_sizes", {})})
        tape = {}
        if self.ar_layout:
            if self.ar_w.value():
                tape[str(self.ar_layout["x"])] = self.ar_w.value()
            if self.ar_h.value():
                tape[str(self.ar_layout["y"])] = self.ar_h.value()
            a["tape_m"] = tape or None
            a["width_m"] = a["height_m"] = None
        else:
            a["width_m"] = self.ar_w.value() or None
            a["height_m"] = self.ar_h.value() or None
        rid = self.ar_robot.text().strip()
        a["robot_id"] = int(rid) if rid.isdigit() else None
        a["robot_radius_m"] = self.ar_radius.value()
        a["map_resolution_m"] = (self.ar_res.value() / 1000.0) or None
        a["max_range_m"] = self.ar_range.value()
        a["unknown_blocked"] = self.ar_unknown.isChecked()
        a["plan_resolution_m"] = self.ar_plan.value() / 100.0
        a["obstacle_threshold"] = self.ar_thr.value()
        return a

    def _arena_settings_to_ui(self, a):
        widgets = (self.ar_w, self.ar_h, self.ar_robot, self.ar_radius,
                   self.ar_res, self.ar_plan, self.ar_thr, self.ar_range, self.ar_unknown)
        for w_ in widgets:
            w_.blockSignals(True)
        self.pivot_label.setText(f"Pivots {', '.join(str(i) for i in sorted(a['corner_ids']))} "
                                 "(largest tags on the sheet) define the floor plane.")
        self.ar_layout = a.get("layout")
        tape = {int(k): v for k, v in (a.get("tape_m") or {}).items()}
        if self.ar_layout:
            self.ar_w.setValue(tape.get(int(self.ar_layout["x"]), 0) or a.get("width_m") or 0)
            self.ar_h.setValue(tape.get(int(self.ar_layout["y"]), 0) or a.get("height_m") or 0)
        else:
            self.ar_w.setValue(a.get("width_m") or 0)
            self.ar_h.setValue(a.get("height_m") or 0)
        self._update_tape_labels()
        self.ar_robot.setText("" if a.get("robot_id") is None else str(a["robot_id"]))
        self.ar_radius.setValue(a["robot_radius_m"])
        self.ar_res.setValue((a.get("map_resolution_m") or 0) * 1000)
        self.ar_range.setValue(float(a.get("max_range_m", 4.0)))
        self.ar_unknown.setChecked(bool(a.get("unknown_blocked", True)))
        self.ar_plan.setValue(a["plan_resolution_m"] * 100)
        self.ar_thr.setValue(int(a["obstacle_threshold"]))
        for w_ in widgets:
            w_.blockSignals(False)

    def _update_tape_labels(self, cam=None):
        L = self.ar_layout
        if L:
            cam = cam or {}
            self.ar_w_label.setText(f"Tape {L['origin']}→{L['x']} (x)" +
                                    (f"\ncam {cam.get(str(L['x'])):.3f}" if cam.get(str(L['x'])) else ""))
            self.ar_h_label.setText(f"Tape {L['origin']}→{L['y']} (y)" +
                                    (f"\ncam {cam.get(str(L['y'])):.3f}" if cam.get(str(L['y'])) else ""))
        else:
            ids = ((self.cfg or {}).get("arena") or {}).get("corner_ids") or []
            o = str(ids[0]) if ids else "origin"
            self.ar_w_label.setText(f"Tape {o}→ +x tag")
            self.ar_h_label.setText(f"Tape {o}→ +y tag")

    def _plane_mode_changed(self, *_):
        if not self.cfg:
            return
        self.cfg["tags_on_floor"] = self.ar_floor_chk.isChecked()
        rid = (self.cfg.get("arena") or {}).get("robot_id")
        if rid is not None:
            spec = self.cfg.setdefault("objects", {}).setdefault(rid, {"name": "robot"})
            if self.ar_floor_chk.isChecked() and self.ar_robot_h.value() > 0:
                spec["height_m"] = self.ar_robot_h.value()
            else:
                # height unknown: a robot's tag is usually raised, so use its 3D pose instead of
                # snapping it to the floor (enter the height for millimeter plane precision)
                spec["height_m"] = None

    def apply_tape(self):
        if not getattr(self, "_arena_samples", None):
            self.log_msg("Set up the arena first (it collects the frames used here).")
            return
        if not (self.ar_w.value() and self.ar_h.value()):
            self.log_msg("Enter both tape distances first.")
            return
        self.cfg["arena"] = self._arena_ui_to_settings(self.cfg.get("arena"))
        self._compute_arena(self._arena_samples)

    def _fit_limit(self):
        """Allowed floor fit error. Without a calibration file the lens distortion is not
        modelled, so a real lens fits a few pixels worse: be more tolerant."""
        return 4.0 if self.has_calib else 12.0

    def _make_amap(self, floor=None):
        """Global floor map from the current config if all 4 corner tags have floor positions."""
        a = self.cfg.get("arena")
        if not a or not all(i in self.cfg["floor_tags"] for i in a["corner_ids"]):
            return None
        floor = floor or (self.mapper.floor if self.mapper else None)
        gm = am.GlobalMap(a, self.cfg["floor_tags"], res=a.get("map_resolution_m"),
                          floor=floor, K=self.mapper.K if self.mapper else None,
                          D=self.mapper.D if self.mapper else None)
        self.log_msg(f"Floor map at {gm.res * 1000:.0f} mm per cell"
                     + (" (auto)" if not a.get("map_resolution_m") else "")
                     + f", mapping out to {a['max_range_m']:.1f} m from the camera; it grows as more "
                     "floor comes into view.")
        return gm

    def reset_global_map(self):
        if self.worker and self.worker.amap is not None:
            self.worker.amap.reset()
            self.log_msg("Map cleared. Press Capture empty floor to restart obstacle detection.")

    def _adaptive_changed(self, on):
        if self.mapper:
            self.mapper.floor.auto_lock = on
        if self.worker and self.worker.pending_floor is not None:
            self.worker.pending_floor.auto_lock = on

    def _arena_params_changed(self, *_):
        if not (self.worker and self.cfg and self.cfg.get("arena")):
            return
        new = self._arena_ui_to_settings(self.cfg["arena"])
        rebuild = (new.get("map_resolution_m") or 0) != (self.cfg["arena"].get("map_resolution_m") or 0)
        self.cfg["arena"].update(new)       # shared with the live map
        if rebuild:
            amap = self._make_amap()
            if amap:
                self.worker.pending_amap = amap
                self.log_msg("Map resolution changed: capture the empty floor again.")

    def start_arena_setup(self, auto=False):
        if not self.worker:
            self.log_msg("Start a source first.")
            return
        self._auto_arena_tried = True          # one rebuild per session is enough
        if auto:
            self.log_msg("All 4 corner tags in view: setting up the floor plane automatically.")
        self.cfg["arena"] = self._arena_ui_to_settings(self.cfg.get("arena"))
        a = self.cfg["arena"]
        self.arena_setup = {"end": time.perf_counter() + 2.0, "samples": []}
        self.ar_setup_btn.setEnabled(False)
        self.log_msg(f"Arena setup: collecting 2 s with corner tags {a['corner_ids']}"
                     + (", with your tape distances" if (a.get("tape_m") or (a["width_m"] and a["height_m"]))
                        else ", scale from the printed tag sizes (enter tape distances for best precision)")
                     + ". Keep the camera still.")

    def _finish_arena_setup(self):
        st, self.arena_setup = self.arena_setup, None
        self.ar_setup_btn.setEnabled(True)
        self._arena_samples = st["samples"]
        self._compute_arena(st["samples"])

    def _compute_arena(self, samples):
        """Build the floor plane from the 4 starter tags (all at z = 0) in one joint fit."""
        a = self.cfg["arena"]
        estimate = not self.has_calib
        try:
            floor_tags, rep_, K, D = am.build_arena(samples, self.K0.copy(), self.D0.copy(),
                                                    self.mapper.tag_size, a, estimate_f=estimate)
        except Exception as e:
            self.log_msg(f"Arena setup failed: {e}")
            return
        a["corner_ids"] = rep_["corner_ids"]
        a["layout"] = rep_["layout"]
        self.ar_layout = rep_["layout"]
        self._update_tape_labels(rep_.get("camera_distance_m"))
        focal = None
        if estimate:
            self.mapper.K, self.mapper.D = K, D
            a["focal_px"], a["k1"] = rep_["focal_px"], rep_["k1"]
            a["focal_at_px"] = [self.source.width, self.source.height]
            focal = (rep_["focal_px"], rep_["fit_px"], "tape" if rep_.get("tape_m") else "tags")
            rep_["focal_estimate"] = {"f_px": rep_["focal_px"],
                                      "f_mm_full_frame": round(rep_["focal_px"] / self.source.width * 35.9, 2)}
        # The plane is defined by the 4 pivots ONLY. (Merging in older floor tags kept tags from
        # another coordinate frame, e.g. a floor survey, as fixed anchors: that bent the floor
        # pose and put the whole map and the A* path off.)
        self.cfg["floor_tags"] = {int(k): v for k, v in floor_tags.items()}
        floor = m.FloorModel(self.cfg["floor_tags"], self.mapper.tag_size, self._fit_limit())
        floor.auto_lock = self.ar_adaptive.isChecked()
        locked = am.averaged_lock(floor, samples, self.mapper.K, self.mapper.D)
        old = self.worker.amap
        new_res = a.get("map_resolution_m")
        new_corners = np.array([[floor_tags[i]["x"], floor_tags[i]["y"]] for i in a["corner_ids"]])
        same_frame = (old is not None and old.corners.shape == new_corners.shape
                      and float(np.abs(old.corners - new_corners).max()) < 0.02)
        if same_frame and (not new_res or abs(new_res - old.res) < 1e-9):
            # Same frame and resolution: keep the map built so far (floor colors, empty-floor
            # reference, obstacle evidence); only the corner outline changes.
            old.corners = new_corners
            old.arena = a
            old._cache = None
            amap = old
        else:
            amap = self._make_amap(floor)
        self.worker.pending_floor = floor
        self.worker.pending_amap = amap
        self.lock_chk.blockSignals(True)
        self.lock_chk.setChecked(locked)
        self.lock_chk.blockSignals(False)
        self._build_floor_items(floor)
        self._plane_mode_changed()
        self.fitted = False
        L = rep_["layout"]
        lines = [f"Arena: origin tag {L['origin']}, +x toward {L['x']}, +y toward {L['y']}, diagonal {L['opposite']}.",
                 f"   source: {rep_['source']}",
                 f"   all 4 tags fitted on one plane (z = 0): {rep_['fit_px']:.2f} px RMS over 16 corners" +
                 (f", pose locked ({floor.reproj:.2f} px)" if locked else ", POSE NOT LOCKED"),
                 "   tag rotations: " + ", ".join(f"{t} {v['yaw_deg']:+.1f}°" for t, v in rep_["tags"].items())]
        if focal:
            how = ("together with lens distortion k1 = " + f"{rep_['k1']:+.3f}, using the tape distances"
                   if focal[2] == "tape" else "from the 4 tags on one plane (enter the tape distances to also "
                   "fix the scale and lens distortion)")
            lines.append(f"   focal length {focal[0]:.0f} px (about {rep_['focal_estimate']['f_mm_full_frame']:.1f} mm), "
                         f"fitted {how}")
        if locked and amap:
            mmpx = am.precision_report(floor, amap, self.mapper.K, self.mapper.D)
            rep_["mm_per_camera_pixel"] = mmpx
            lines.append(f"   floor resolution {min(mmpx.values()):.1f} to {max(mmpx.values()):.1f} mm per camera "
                         "pixel (near to far corner)")
        if "scale_check_pct" in rep_:
            lines.append(f"   camera vs tape after fitting: {rep_['scale_check_pct'][0]:+.2f}% / "
                         f"{rep_['scale_check_pct'][1]:+.2f}%")
        cam = rep_.get("camera_distance_m", {})
        if not rep_.get("tape_m"):
            lines.append(f"   NEXT: tape-measure {L['origin']}→{L['x']} (camera says {cam.get(str(L['x']))} m) and "
                         f"{L['origin']}→{L['y']} (camera says {cam.get(str(L['y']))} m) center to center, "
                         "enter them, press Apply tape measurements.")
        lines += ["   WARNING: " + w_ for w_ in rep_["warnings"]]
        if rep_.get("tape_m"):
            lines.append("Next: clear the arena, press Capture empty floor, then click a goal on the map.")
        self.log_msg("\n".join(lines))
        if self.src_combo.currentText() != "Synthetic":
            path = os.path.abspath(am.DEFAULT_ARENA_FILE)
            am.save_arena(path, self.cfg, self.cfg["floor_tags"], rep_)
            self.config_edit.setText(path)
            self.log_msg(f"Saved {path}. It loads automatically next time.")

    def capture_background(self):
        if not (self.worker and (self.worker.amap is not None or self.worker.pending_amap is not None)):
            self.log_msg("Set up the arena first.")
            return
        if self.src_combo.currentText() == "Synthetic" and hasattr(self, "scene"):
            self.scene.empty_frames = 6
            self.worker.capture_bg = 4
        else:
            self.worker.capture_bg = 2
        self.log_msg("Capturing the empty floor reference…")

    def clear_goal(self):
        if self.worker:
            self.worker.goal = None
            if self.pair_btn.isChecked():
                self.pair_btn.setChecked(False)

    def _arena_click(self, u, v, button):
        if not (self.worker and self.worker.amap is not None):
            return
        x, y = self.worker.amap.p2w(u, v)
        if button == 3:
            self._add_stop_item({"kind": "point", "xy": (round(x, 4), round(y, 4)),
                                 "label": f"({x:+.2f}, {y:+.2f})"})
            return
        if button == 1:
            if self.pair_btn.isChecked():
                self.pair_btn.setChecked(False)
            self.worker.goal = (round(x, 4), round(y, 4))
            self.log_msg(f"Goal ({x:+.3f}, {y:+.3f}) m")
        else:
            self.worker.manual_start = (round(x, 4), round(y, 4))
            self.log_msg(f"Manual start ({x:+.3f}, {y:+.3f}) m (used while the robot tag is not seen)")

    def _arena_hover(self, u, v):
        if self.worker and self.worker.amap is not None:
            x, y = self.worker.amap.p2w(u, v)
            self.hover_label.setText(f"arena x {x:+.4f}  y {y:+.4f} m")

    def _on_arena(self, qimg, scale, info):
        for ev in info.get("events", []):
            self.log_msg("Floor: " + ev)
        if qimg.isNull():
            return
        self.arena_view.set_frame(qimg, scale)
        if self.worker:
            self.worker.arena_free = True
        if info.get("background_captured"):
            self.log_msg("Empty floor captured. Obstacles are now detected.")
        txt = []
        if not info.get("has_background"):
            txt.append("No empty-floor reference (optional): obstacles come from tags and detected objects.")
        mp_ = info.get("map")
        if mp_:
            txt.append(f"map {mp_['known_m2']:.2f} m² known at {mp_['res_mm']:.0f} mm, "
                       f"{len(mp_['anchors'])} pivots, pose {'locked' if mp_['locked'] else 'live'}"
                       + (" (map paused while the camera moves)" if info.get("map_paused") else ""))
        rb = info.get("robot")
        txt.append("robot " + (f"({rb['x']:+.3f}, {rb['y']:+.3f}) m, heading {rb['yaw_deg']:+.1f}°"
                               if rb else "not seen"))
        pl = info.get("plan")
        if pl:
            if pl["status"] == "ok":
                txt.append(f"path {pl['length_m']:.3f} m, {len(pl['waypoints'])} waypoints, A* {pl.get('ms')} ms")
            else:
                txt.append(f"path: {pl['status']}")
            if pl.get("notes"):
                txt.append("; ".join(pl["notes"]))
        elif info.get("goal") is None:
            txt.append("left click = goal, right click = start (when no robot tag)")
        sts = info.get("stops") or []
        if sts:
            done = sum(1 for s_ in sts if s_["done"])
            legs = (info.get("plan") or {}).get("leg_lengths_m") or []
            txt.append(f"route: {len(sts)} stop(s), {done} reached"
                       + (" | legs " + ", ".join(f"{v:.2f}" for v in legs) + " m" if legs else ""))
        pi = info.get("pair")
        if pi:
            def st(seen, age):
                return "seen" if seen else ("lost" if age is None else f"last seen {age:.1f} s ago")
            pl2 = info.get("plan") or {}
            route = ""
            if sts:
                route = "".join(f"stop {s_['n']} {s_['label']}: " + ("reached" if s_["done"] else
                                ("ok" if s_["xy"] is not None else "not seen yet")) + "\n" for s_ in sts)
            self.pair_info.setText(
                f"start tag {pi['start_id']}: {st(pi['start_seen'], pi['start_age_s'])}\n" + route +
                f"goal tag {pi['goal_id']}: {st(pi['goal_seen'], pi['goal_age_s'])}\n"
                + (f"path {pl2['length_m']:.3f} m, {len(pl2['waypoints'])} waypoints"
                   if pl2.get("status") == "ok" else f"path: {pl2.get('status', 'waiting for both tags')}"))
        cmd = info.get("command")
        if cmd:
            txt.append(f"steer {cmd['heading_error_deg']:+.1f}°, {cmd['distance_to_goal_m']:.3f} m to go")
        self.arena_status.setText("  |  ".join(txt))

    def toggle_detection(self, on):
        if not self.worker:
            if on:
                self.log_msg("Start a source first, then enable detection.")
                self.det_chk.setChecked(False)
            return
        # optional YOLO: stop it first, (re)start below if wanted
        if self.worker.objdet is not None:
            self.worker.objdet.stop()
        self.worker.objdet = None
        self.worker._det_objs = []
        self.worker.appearance = on
        if not on:
            self.det_status.setText("off")
            return
        self.cfg.setdefault("arena", {})["appearance_z"] = self.det_app_z.value()
        self.worker.app_tracker = od.FloorTracker(gate_m=0.15, ttl=1.0, min_hits=3)
        self.log_msg("Object detection on: anything on the floor that does not look like the floor becomes "
                     "an object and an obstacle (compared with the rest of the floor in each frame).")
        if not self.worker.amap:
            self.log_msg("Note: objects are found once the arena (floor plane) is set up.")
        if not self.det_yolo.isChecked():
            return
        try:
            import importlib.util
            if importlib.util.find_spec("ultralytics") is None:
                raise ImportError("ultralytics")
        except ImportError:
            self.det_yolo.setChecked(False)
            self.log_msg("YOLO proposals need Ultralytics: pip install ultralytics")
            QtWidgets.QMessageBox.information(self, "Install needed", "pip install ultralytics")
            return
        name = self.det_model.currentText().strip()
        self.worker.det_obstacles = self.det_obst.isChecked()
        self.worker.tracker = od.FloorTracker()
        self.worker.objdet = od.ObjectDetector(name, self.det_conf.value(), int(self.det_size.currentText()))
        self.log_msg(f"Also loading {name} for YOLO proposals (class-agnostic)…")

    def _det_reconfig(self, *_):
        if self.cfg and self.cfg.get("arena") is not None:
            self.cfg["arena"]["appearance_z"] = self.det_app_z.value()   # shared with the live map
        if self.worker and self.worker.objdet is not None:
            self.worker.objdet.configure(self.det_conf.value(), int(self.det_size.currentText()))

    # ---------------- route stops ----------------
    def _add_stop_item(self, spec):
        it = QtWidgets.QListWidgetItem(f"{self.stops_list.count() + 1}. {spec['label']}")
        it.setData(Qt.ItemDataRole.UserRole, spec)
        self.stops_list.addItem(it)
        self._sync_stops()
        self.log_msg(f"Stop {self.stops_list.count()} added: {spec['label']}")

    def add_stop(self):
        txt = self.stop_combo.currentText().strip()
        tid = txt.split(":")[0].strip()
        if not tid.lstrip("-").isdigit():
            self.log_msg("Choose a tag / object id for the stop (or shift+click a point on the Arena view).")
            return
        self._add_stop_item({"kind": "tag", "id": int(tid), "label": txt})

    def remove_stop(self):
        row = self.stops_list.currentRow()
        if self.stops_list.count():
            self.stops_list.takeItem(row if row >= 0 else self.stops_list.count() - 1)
            self._sync_stops()

    def clear_stops(self):
        self.stops_list.clear()
        self._sync_stops()

    def _sync_stops(self):
        """List widget -> worker (order matters). Renumbers the items; the route restarts."""
        stops = []
        for i in range(self.stops_list.count()):
            it = self.stops_list.item(i)
            spec = it.data(Qt.ItemDataRole.UserRole)
            it.setText(f"{i + 1}. {spec['label']}")
            stops.append(spec)
        if self.worker:
            self.worker.stops = stops
            self.worker._next_stop = 0
            self.worker._plan = None
            self.worker._pair_key = None

    def _pair_ids(self):
        out = []
        for cb in (self.pair_start, self.pair_goal):
            txt = cb.currentText().split(":")[0].strip()
            if not txt.isdigit():
                return None
            out.append(int(txt))
        return tuple(out) if out[0] != out[1] else None

    def toggle_pair(self, on):
        if not self.worker:
            self.pair_btn.setChecked(False)
            self.log_msg("Start a source first.")
            return
        if not on:
            self.worker.pair = None
            self.worker._plan = None
            self.pair_btn.setText("Plan path")
            self.pair_info.setText("")
            self.log_msg("Tag-to-tag path stopped.")
            return
        ids = self._pair_ids()
        if ids is None:
            self.pair_btn.setChecked(False)
            self.log_msg("Choose two different tag ids for start and goal.")
            return
        if self.worker.amap is None:
            self.pair_btn.setChecked(False)
            self.log_msg("Set up the arena first (the path is planned on the floor map).")
            return
        self.worker.goal = None
        self.worker._pair_last = {}
        self.worker._plan = None
        for t in ids:
            if t in (self.cfg.get("arena") or {}).get("corner_ids", []):
                self.log_msg(f"Note: tag {t} is one of the floor's pivot tags.")
        self.worker.pair = ids
        self.pair_btn.setText(f"Stop path {ids[0]} → {ids[1]}")
        self.tabs.setCurrentIndex(1)
        self.log_msg(f"Planning A* from tag {ids[0]} (robot) to tag {ids[1]}, replanned live as they move.")

    def _swap_pair(self):
        a_, b_ = self.pair_start.currentText(), self.pair_goal.currentText()
        self.pair_start.setEditText(b_)
        self.pair_goal.setEditText(a_)
        if self.pair_btn.isChecked():
            ids = self._pair_ids()
            if ids and self.worker:
                self.worker.pair, self.worker._plan, self.worker._pair_last = ids, None, {}
                self.pair_btn.setText(f"Stop path {ids[0]} → {ids[1]}")

    def copy_waypoints(self):
        plan = self.worker._plan if self.worker else None
        if not plan or plan.get("status") != "ok":
            self.log_msg("No path to copy yet.")
            return
        data = {"frame": "arena (m)", "waypoints": plan["waypoints"], "length_m": plan["length_m"]}
        if self.worker.pair:
            data["start_tag"], data["goal_tag"] = self.worker.pair
        QtWidgets.QApplication.clipboard().setText(json.dumps(data))
        self.log_msg(f"Copied {len(plan['waypoints'])} waypoints ({plan['length_m']:.3f} m): "
                     + json.dumps(plan["waypoints"]))

    def _update_pair_choices(self):
        ids = sorted(self.stats)
        if ids == getattr(self, "_pair_choice_ids", None):
            return
        self._pair_choice_ids = ids
        names = {t: (self.stats[t].last[1]["name"] if self.stats[t].last else "") for t in ids}
        for cb in (self.pair_start, self.pair_goal, self.stop_combo):
            cur = cb.currentText()
            cb.blockSignals(True)
            cb.clear()
            for t in ids:
                cb.addItem(f"{t}: {names[t]}")
            cb.setEditText(cur)
            cb.blockSignals(False)
        if not self.pair_start.currentText() and ids:
            rid = (self.cfg.get("arena") or {}).get("robot_id")
            self.pair_start.setEditText(f"{rid}" if rid in ids else f"{ids[0]}: {names[ids[0]]}")
        if not self.pair_goal.currentText() and len(ids) > 1:
            other = [t for t in ids if str(t) != self.pair_start.currentText().split(":")[0]]
            if other:
                self.pair_goal.setEditText(f"{other[0]}: {names[other[0]]}")

    def save_arena_map(self):
        if not (self.worker and getattr(self.worker, "arena_data", None)):
            self.log_msg("No floor map yet.")
            return
        gm, occ, blocked, cell = self.worker.arena_data
        folder = os.path.abspath(time.strftime("floor_map_%Y%m%d_%H%M%S"))
        extra = {"floor_tags": {str(k): v for k, v in self.cfg["floor_tags"].items()},
                 "robot_radius_m": self.cfg["arena"]["robot_radius_m"],
                 "last_state": self.worker.arena_last}
        am.save_global_map(folder, gm, occ, blocked, cell, extra)
        self.log_msg(f"Saved map to {folder} (floor_mosaic.png, occupancy.npy, known.npy, "
                     "planning_blocked.npy, mm_per_camera_px.npy, map.json)")

    def _udp_changed(self):
        if not self.worker:
            return
        txt = self.ar_udp.text().strip()
        if not txt:
            self.worker.udp = None
            return
        try:
            host, port = txt.rsplit(":", 1)
            import socket
            self.worker.udp = (socket.socket(socket.AF_INET, socket.SOCK_DGRAM), (host, int(port)))
            self.log_msg(f"Sending arena state + path + steering to {host}:{port} every frame (JSON)")
        except ValueError:
            self.log_msg("UDP out must look like 127.0.0.1:5005")

    def start_measure(self):
        if self.selected is None:
            self.log_msg("Click a row in the table to select a tag first.")
            return
        truth = None
        txt = self.truth_edit.text().strip()
        if txt:
            try:
                truth = np.array([float(v) for v in txt.replace(",", " ").split()][:3])
                assert truth.size == 3
            except Exception:
                self.log_msg("Truth must be three numbers: x, y, z")
                return
        self.measure = {"id": self.selected, "end": time.perf_counter() + self.meas_spin.value(),
                        "pts": [], "truth": truth, "auto_truth": []}
        self.meas_btn.setEnabled(False)
        self.log_msg(f"Measuring tag {self.selected} for {self.meas_spin.value():.1f} s, hold still…")

    def _finish_measure(self):
        ms, self.measure = self.measure, None
        self.meas_btn.setEnabled(True)
        if len(ms["pts"]) < 3:
            self.log_msg(f"Measure failed: tag {ms['id']} seen in only {len(ms['pts'])} frames.")
            return
        p = np.asarray(ms["pts"])
        mean, sd = p.mean(0), p.std(0) * 1e3
        msg = (f"Tag {ms['id']}: {len(p)} frames  mean ({mean[0]:+.4f}, {mean[1]:+.4f}, {mean[2]:+.4f}) m  "
               f"σ ({sd[0]:.1f}, {sd[1]:.1f}, {sd[2]:.1f}) mm")
        truth = ms["truth"]
        if truth is None and ms["auto_truth"]:
            truth = np.asarray(ms["auto_truth"]).mean(0)
        if truth is not None:
            err = (mean - truth) * 1e3
            msg += (f"\n   vs truth ({truth[0]:+.4f}, {truth[1]:+.4f}, {truth[2]:+.4f}): error "
                    f"({err[0]:+.1f}, {err[1]:+.1f}, {err[2]:+.1f}) mm, |e| = {np.linalg.norm(err):.1f} mm")
        self.log_msg(msg)

    # ---------------- data in ----------------
    def _on_frame(self, qimg, scale):
        self.video.set_frame(qimg, scale)
        if self.worker:
            self.worker.display_free = True

    def _on_result(self, r):
        self.latest = r
        # Saved arena present but its floor fit keeps getting rejected (different lens, zoom,
        # resolution, or a camera-only arena): explain once and rebuild it from the live view.
        if (self.worker and self.worker.amap is not None and not r["floor_valid"]
                and r.get("floor_reject_px") is not None and self.arena_setup is None):
            self._reject_streak = getattr(self, "_reject_streak", 0) + 1
            if self._reject_streak == 15 and time.perf_counter() - getattr(self, "_last_auto_fix", 0) > 30:
                self._last_auto_fix = time.perf_counter()
                self.log_msg(f"The saved floor does not fit this view (fit error {r['floor_reject_px']} px, "
                             f"limit {r['floor_limit_px']:.0f} px). Likely causes: different zoom/focus/"
                             "resolution than when it was set up, or it was set up without tape and with the "
                             "28 mm focal guess. Rebuilding the arena from the live view now; your tape "
                             "distances are kept.")
                self.cfg["arena"] = self._arena_ui_to_settings(self.cfg.get("arena"))
                self.start_arena_setup()
        else:
            self._reject_streak = 0
        if (self.worker and self.arena_setup is None and not getattr(self, "_auto_arena_tried", False)):
            ids = {g for g, _ in r.get("_dets", [])}
            need = set((self.cfg.get("arena") or {}).get("corner_ids", []))
            self._corner_streak = getattr(self, "_corner_streak", 0) + 1 if need and need <= ids else 0
            if self._corner_streak >= 10:
                self._auto_arena_tried = True
                if self.worker.amap is not None:
                    self.log_msg("4 starter tags in view: rebuilding the floor plane from the live view "
                                 "(saved settings and tape distances are kept).")
                self.start_arena_setup(auto=True)
        if self.arena_setup is not None:
            self.arena_setup["samples"].append(r.get("_dets", []))
            if time.perf_counter() > self.arena_setup["end"]:
                self._finish_arena_setup()
        truth = r.get("truth") or {}
        for o in r["objects"]:
            if o["frame"] != "world":
                continue
            st = self.stats.setdefault(o["id"], TagStats())
            st.add(r["wall"], o, truth.get(o["id"]))
            if self.selected is None:
                self.selected = o["id"]
            if self.measure and o["id"] == self.measure["id"]:
                self.measure["pts"].append((o["x"], o["y"], o["z"]))
                if o["id"] in truth:
                    self.measure["auto_truth"].append(truth[o["id"]])
        if self.measure and time.perf_counter() > self.measure["end"]:
            self._finish_measure()
        if r["floor_valid"] and not self.fitted:
            self.fitted = True
            QtCore.QTimer.singleShot(300, self.map.autoRange)

    def _hover(self, u, v):
        if not (self.latest and self.mapper):
            return
        p = pixel_to_floor(self.latest.get("_floor"), self.mapper.K, self.mapper.D, u, v)
        if p is None:
            self.hover_label.setText(f"px ({u:.0f}, {v:.0f})  floor: --")
            self.hover_scatter.setData([], [])
        else:
            self.hover_label.setText(f"px ({u:.0f}, {v:.0f})  floor x {p[0]:+.3f}  y {p[1]:+.3f} m")
            self.hover_scatter.setData([p[0]], [p[1]])

    def _click(self, u, v, button):
        if button == 2:
            self.pins = []
        elif self.latest and self.mapper:
            p = pixel_to_floor(self.latest.get("_floor"), self.mapper.K, self.mapper.D, u, v)
            if p is not None:
                self.pins = self.pins + [p]
                self.log_msg(f"Pin at floor x {p[0]:+.4f}  y {p[1]:+.4f}")
        if self.worker:
            self.worker.pins = list(self.pins)
        self.pin_scatter.setData([p[0] for p in self.pins], [p[1] for p in self.pins])

    def _row_clicked(self, row, _col):
        item = self.table.item(row, 0)
        if item:
            self.selected = int(item.text())

    # ---------------- drawing ----------------
    def _build_floor_items(self, floor=None):
        floor = floor or self.mapper.floor
        for it in self.floor_items:
            self.map.removeItem(it)
        self.floor_items = []
        a = (self.cfg or {}).get("arena") or {}
        ids = a.get("corner_ids", [])
        if ids and all(i in floor.world_corners for i in ids):
            cen = np.array([floor.world_corners[i][:, :2].mean(0) for i in ids])
            cen = np.vstack([cen, cen[:1]])
            poly = pg.PlotDataItem(cen[:, 0], cen[:, 1], pen=pg.mkPen(0, 220, 255, width=2))
            self.map.addItem(poly)
            self.floor_items.append(poly)
        for tid, c in floor.world_corners.items():
            xy = np.vstack([c[:, :2], c[:1, :2]])
            line = pg.PlotDataItem(xy[:, 0], xy[:, 1], pen=pg.mkPen(255, 160, 0, width=2))
            txt = pg.TextItem(f"pivot {tid}", color=(255, 160, 0), anchor=(0.5, 1.2))
            txt.setPos(*c[:, :2].mean(0))
            for it in (line, txt):
                self.map.addItem(it)
                self.floor_items.append(it)

    def _refresh_panels(self):
        r = self.latest
        if r is None or not self.worker:
            return
        self._update_pair_choices()
        dt = self.worker.objdet
        if self.worker.appearance and dt is None:
            n_ = len(self.worker._app_objs)
            self.det_status.setText(f"{n_} object(s) on the floor" if self.worker.amap is not None
                                    else "waiting for the arena (floor plane)")
        if dt is not None:
            if dt.error and dt.status == "failed to load":
                self.det_status.setText(f"FAILED: {dt.error[:200]}")
            else:
                n_ = sum(1 for o in self.worker._det_objs if o.get("frame") == "world")
                self.det_status.setText(f"{dt.model_name}: {dt.status}" + (f", {n_} on the floor" if n_ else "")
                                        + (f" (last error: {dt.error[:80]})" if dt.error else ""))
        if getattr(self.args, "detect", None) and not getattr(self, "_detect_arg_done", False):
            self._detect_arg_done = True
            self.det_model.setEditText(self.args.detect)
            self.det_chk.setChecked(True)
        want = getattr(self.args, "path", None)
        if want and not self.pair_btn.isChecked() and not getattr(self, "_path_arg_done", False) \
                and self.worker.amap is not None and r["floor_valid"]:
            self._path_arg_done = True
            self.pair_start.setEditText(str(want[0]))
            self.pair_goal.setEditText(str(want[1]))
            self.pair_btn.setChecked(True)
        now = time.perf_counter()

        # pivot tags: how many are in view, plane state
        piv = sorted((self.cfg.get("arena") or {}).get("corner_ids") or [])
        if piv:
            seen = {g for g, _ in r.get("_dets", [])}
            n_in = sum(1 for t in piv if t in seen)
            state = (("plane locked" if r["floor_locked"] else "plane ok") if r["floor_valid"]
                     else "no plane yet")
            missing = [t for t in piv if t not in seen]
            self.pivot_label.setText(f"Pivots {', '.join(map(str, piv))}: {n_in}/4 in view, {state}."
                                     + (f" Not seen: {', '.join(map(str, missing))}." if missing else "")
                                     + " Other tags are objects.")

        # status bar
        if self.lock_chk.isChecked() != bool(r["floor_locked"]):
            self.lock_chk.blockSignals(True)
            self.lock_chk.setChecked(bool(r["floor_locked"]))
            self.lock_chk.blockSignals(False)
        if r["floor_valid"]:
            fl = f"floor {'LOCKED' if r['floor_locked'] else 'ok'} {r['floor_tags_seen']} {r['floor_reproj_px']} px"
        elif r.get("floor_reject_px") is not None:
            fl = (f"floor REJECTED: fit {r['floor_reject_px']} px > limit {r['floor_limit_px']:.0f} px "
                  f"(saved arena does not match this camera/lens)")
        else:
            fl = "floor NOT FOUND: no floor tag in view"
        cam = r["camera"]
        cam_s = f"camera ({cam['x']:+.2f}, {cam['y']:+.2f}, {cam['z']:+.2f}) m" if cam else ""
        rec = "  ● REC" if self.rec_btn.isChecked() else ""
        self.status.setText(f"proc {r['proc_fps']:4.1f} fps | source {r['grab_fps']:4.1f} fps | "
                            f"detect {r['timing_ms']['detect']:5.1f} ms | latency {r['latency_ms']:5.1f} ms | "
                            f"tags {r['n_tags']} | {fl} | {cam_s}{rec}")

        # camera on map
        if cam and r.get("_floor"):
            R, _ = r["_floor"]
            self.cam_scatter.setData([cam["x"]], [cam["y"]])
            fwd = R.T @ np.array([0, 0, 1.0])
            n = np.linalg.norm(fwd[:2])
            if n > 1e-6:
                f = fwd[:2] / n * 0.3
                self.cam_heading.setData([cam["x"], cam["x"] + f[0]], [cam["y"], cam["y"] + f[1]])

        # objects, trails, labels
        win = self.window_spin.value()
        live_x, live_y, sel_xy, truth_xy, rows = [], [], None, [], []
        for tid in sorted(self.stats):
            st = self.stats[tid]
            t_last, o = st.last
            age = now - t_last
            a = st.window(now, max(win, 5.0))
            if tid not in self.trails:
                self.trails[tid] = pg.PlotDataItem(pen=pg.mkPen(0, 200, 0, 120, width=1))
                self.labels[tid] = pg.TextItem(color=(0, 230, 0), anchor=(0, 1))
                self.map.addItem(self.trails[tid])
                self.map.addItem(self.labels[tid])
            self.trails[tid].setData(a[:, 1], a[:, 2]) if len(a) else self.trails[tid].setData([], [])
            stale = age > 0.5
            self.labels[tid].setText(f"{o['name']} z={o['z']:.3f}" + (" (lost)" if stale else ""))
            self.labels[tid].setPos(o["x"], o["y"])
            if not stale:
                live_x.append(o["x"])
                live_y.append(o["y"])
                if tid == self.selected:
                    sel_xy = (o["x"], o["y"])
                if len(a) and np.isfinite(a[-1, 4]):
                    truth_xy.append(a[-1, 4:6])

            w = a[a[:, 0] >= now - win] if len(a) else a
            sd = w[:, 1:4].std(0) * 1e3 if len(w) > 2 else [np.nan] * 3
            hz = np.sum(w[:, 0] >= now - 1.0) if len(w) else 0
            err = np.nan
            if len(a) and np.isfinite(a[-1, 4]):
                err = np.linalg.norm(a[-1, 1:4] - a[-1, 4:7]) * 1e3
            rows.append((tid, o, sd, hz, err, stale))

        self.obj_scatter.setData(live_x, live_y)
        self.sel_scatter.setData(*([[sel_xy[0]], [sel_xy[1]]] if sel_xy else [[], []]))
        if truth_xy:
            t = np.asarray(truth_xy)
            self.truth_scatter.setData(t[:, 0], t[:, 1])
        else:
            self.truth_scatter.setData([], [])

        # table
        self.table.setRowCount(len(rows))
        fmt = lambda v, p: "" if v is None or not np.isfinite(v) else f"{v:.{p}f}"
        for i, (tid, o, sd, hz, err, stale) in enumerate(rows):
            meth = o.get("method", "")
            name = o["name"] + (" [plane]" if meth == "plane" else " [3D]" if meth.startswith("pose")
                                else " [vision]" if meth == "detect" else "")
            vals = [str(tid), name, fmt(o["x"], 4), fmt(o["y"], 4), fmt(o["z"], 4),
                    fmt(o.get("yaw_deg"), 1), fmt(o.get("tilt_deg"), 1), fmt(sd[0], 1), fmt(sd[1], 1),
                    fmt(sd[2], 1), str(int(hz)), fmt(o["reproj_px"], 2), fmt(err, 1)]
            for j, v in enumerate(vals):
                it = self.table.item(i, j)
                if it is None:
                    it = QtWidgets.QTableWidgetItem()
                    self.table.setItem(i, j, it)
                it.setText(v)
                it.setForeground(QtGui.QColor("#888" if stale else "#000"))
                it.setBackground(QtGui.QColor("#fff6c0" if tid == self.selected else "#ffffff"))

        # time plot
        st = self.stats.get(self.selected)
        if st is not None:
            a = st.window(now, 15.0)
            if len(a):
                t = a[:, 0] - now
                for k, col in (("x", 1), ("y", 2), ("z", 3)):
                    self.curves[k].setData(t, a[:, col])
                    tr = a[:, col + 3]
                    if np.isfinite(tr).any():
                        self.truth_curves[k].setData(t, tr)
                    else:
                        self.truth_curves[k].setData([], [])
            self.plot.setTitle(f"Tag {self.selected} position (solid = estimate, dashed = truth)")


def main():
    ap = argparse.ArgumentParser(description="Real time AprilTag tracking test GUI")
    ap.add_argument("--connection", choices=["auto", "usb", "hdmi"], default="usb",
                    help="usb = A7 IV USB-C to C (USB Streaming), hdmi = HDMI to C capture card")
    ap.add_argument("--backend", choices=["auto", "dshow", "msmf"], default="auto")
    ap.add_argument("--camera", default="auto",
                    help="'auto' (Sony over USB, else HDMI capture card), a name substring, or an index")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--calib")
    ap.add_argument("--config")
    ap.add_argument("--video", help="play a recorded video instead of the live camera")
    ap.add_argument("--focal-mm", type=float, default=28.0)
    ap.add_argument("--synthetic", action="store_true", help="test scene with known ground truth")
    ap.add_argument("--synthetic-raw", action="store_true",
                    help="synthetic scene with nothing preset and a wrong focal length (practice the setup)")
    ap.add_argument("--autostart", action="store_true")
    ap.add_argument("--detect", metavar="MODEL", nargs="?", const=od.MODEL_CHOICES[0][0],
                    help=f"start class-agnostic object detection (default model {od.MODEL_CHOICES[0][0]})")
    ap.add_argument("--path", nargs=2, type=int, metavar=("START_TAG", "GOAL_TAG"),
                    help="plan the A* path from START_TAG to GOAL_TAG as soon as the floor is ready, "
                         "e.g. --path 28 50")
    args = ap.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOption("background", "#1e1e1e")
    win = MainWindow(args)
    win.show()
    if args.autostart:
        QtCore.QTimer.singleShot(100, win.start)
    code = app.exec()
    # Skip interpreter teardown so pupil_apriltags detectors are never destructed.
    sys.stdout.flush()
    os._exit(code)


if __name__ == "__main__":
    main()
