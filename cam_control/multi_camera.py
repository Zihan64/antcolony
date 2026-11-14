#!/usr/bin/env python3
"""
multi_camera.py

Extra cameras for the tracker, fused into one pose per tag. The goal is robustness: if one camera
drops out (unplugged, stalled USB, someone walks in front of it) the robots keep a position.

Why this works without any camera-to-camera calibration
  Every camera solves its OWN pose against the 4 pivot tags, whose floor positions are saved in
  arena.json by the arena setup. So every camera that sees the pivots reports tag positions in the
  same floor frame already. A camera needs at least one pivot in view (two or more is much better;
  all four is best) and stays fixed (adaptive lock re-locks it if it is bumped).

What each extra camera runs (its own thread, never blocks the main camera)
  capture thread (newest frame only) -> own tag detector -> own Mapper / FloorModel -> latest result

Fusion (fuse_tags), per tag id, every frame of the main camera
  * observations older than `max_age_s`, or from a camera whose floor pose is not valid, are left out;
  * each observation gets an uncertainty from how it was located: "plane" (ray hits a plane at a
    known height: error ~ distance * pixel noise / focal) or "pose" (size-based depth, error grows
    with distance^2), times the reprojection error, inflated by its age (a moving robot);
  * the largest group of observations that agree with each other (within 3 sigma, at least 3 cm)
    wins; the others are outliers (a mis-detected tag, a camera whose pose drifted) and dropped;
  * the winners are averaged with inverse-variance weights (heading: circular mean).
  The fused object keeps the fields of its best observation plus "cams" (names), "n_cams" and
  "sigma_m". Untagged objects (ids 1000+) are not fused: they come from the main camera's map.

Extra cameras are given as specs (command line --extra-camera, the GUI's Settings tab, or
cameras.json next to arena.json):

    name=go3s,camera=Insta360,width=1920,height=1080,fps=30,calib=go3s_calib.npz

    camera      index or part of the device name (as in the main camera list), or wireless:
                insta-wifi[:host[:port]]  Insta360 over its own Wi-Fi (insta360_wifi.py)
                rtsp://... http://...     a network stream, e.g. a Raspberry Pi serving a USB camera
    calib       chessboard calibration .npz (apriltag_floor_mapper.py calibrate). Strongly
                recommended for wide lenses (the GO 3S): distortion is otherwise only modelled by k1.
    focal_px,k1,k2,k3  lens model if there is no calib (printed in the log after self-calibration)
    hfov_deg    starting guess for the focal length without calib or focal_px (default 90)
    backend     auto | dshow | msmf          decimate  tag detector decimation (default 1.0)
    crosscal    1 (default) learn lens / pose from the main camera, 0 off
    max_fps     frames per second this camera PROCESSES (default 10). Tag detection is CPU-bound
                (measured: two cameras share the CPU, threads or processes alike), so extra cameras
                are capped to leave the main camera its full rate. Raise it on a fast machine.

Without calib or focal_px the camera calibrates itself from the pivots: once 2+ pivots are seen
steadily it fits the focal length (and k1 with 3+ pivots) against their known floor positions.

Cross-calibration from the main camera (CrossCalibration): wide or fisheye lenses (the GO 3S) and
cameras that see fewer than 3 pivots cannot be fitted from the pivots alone. So every extra camera
also LEARNS FROM THE MAIN CAMERA: whenever a flat tag (a robot, any tag on the floor) stands still
where both cameras see it, the main camera's floor position of its 4 corners is paired with the
pixels this camera sees. Spread over the view (robot pauses, or a tag walked around the floor),
these pairs fit the full lens (focal, k1, k2, k3) AND the camera's pose. The pose then does not
need a pivot in view at all. A learned lens is saved in camera_lens.json (per camera name and
resolution), so the next start only re-learns the pose (2 spots, or the pivots it sees).
"""
import json
import math
import os
import threading
import time
from collections import deque

import cv2
import numpy as np

import apriltag_floor_mapper as m

DEFAULT_CAMERAS_FILE = "cameras.json"
LENS_CACHE_FILE = "camera_lens.json"
SPEC_KEYS = {"name": str, "camera": str, "width": int, "height": int, "fps": int, "calib": str,
             "focal_px": float, "k1": float, "k2": float, "k3": float, "hfov_deg": float,
             "backend": str, "decimate": float, "max_fps": float, "crosscal": int}


# --------------------------------------------------------------------------
# Camera specs
# --------------------------------------------------------------------------

def parse_spec(text):
    """'name=go3s,camera=Insta360,fps=30' (or just 'Insta360' / '2') -> spec dict."""
    text = text.strip()
    if not text:
        return None
    spec = {}
    if "=" not in text:
        spec["camera"] = text
    else:
        for part in text.split(","):
            if not part.strip():
                continue
            k, sep, v = part.partition("=")
            k, v = k.strip().lower(), v.strip()
            if not sep or k not in SPEC_KEYS:
                raise ValueError(f"camera spec '{text}': unknown item '{part.strip()}' "
                                 f"(known: {', '.join(SPEC_KEYS)})")
            spec[k] = SPEC_KEYS[k](v)
    return normalize_spec(spec)


def normalize_spec(spec):
    spec = {k: SPEC_KEYS[k](v) for k, v in spec.items() if k in SPEC_KEYS and v is not None}
    if not spec.get("camera"):
        raise ValueError(f"camera spec {spec} has no 'camera' (an index or part of the device name)")
    spec.setdefault("name", spec["camera"])
    spec.setdefault("width", 1920)
    spec.setdefault("height", 1080)
    spec.setdefault("fps", 30)
    spec.setdefault("backend", "auto")
    # full resolution: extra cameras are often wide (small tags); at 2.0 the GO 3S found 1 of 3 tags
    spec.setdefault("decimate", 1.0)
    spec.setdefault("max_fps", 10.0)
    return spec


def parse_specs(text):
    """Several specs separated by ';'."""
    return [s for s in (parse_spec(p) for p in (text or "").split(";")) if s]


def load_specs(path=DEFAULT_CAMERAS_FILE):
    """cameras.json: {"cameras": [{"name": "go3s", "camera": "Insta360", ...}, ...]} (or a bare list)."""
    if not path or not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    items = data.get("cameras", []) if isinstance(data, dict) else data
    return [normalize_spec(dict(d)) for d in items if not d.get("disabled")]


def spec_text(spec):
    return ",".join(f"{k}={spec[k]}" for k in SPEC_KEYS if k in spec)


# --------------------------------------------------------------------------
# Intrinsics for a camera without a calibration file
# --------------------------------------------------------------------------

def guess_K(width, height, focal_px=None, hfov_deg=90.0):
    f = float(focal_px) if focal_px else width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    return np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1]], np.float64)


def intrinsics_for(spec, width, height):
    """(K, D, source text) for a spec at the resolution the device actually delivers."""
    if spec.get("calib"):
        ns = type("A", (), {"calib": spec["calib"], "focal_mm": 0, "sensor_width_mm": 1})()
        K, D = m.load_intrinsics(ns, width, height)
        return K, np.asarray(D, np.float64).ravel(), f"calibration {os.path.basename(spec['calib'])}"
    K = guess_K(width, height, spec.get("focal_px"), spec.get("hfov_deg", 90.0))
    D = np.zeros(5)
    D[0] = float(spec.get("k1") or 0.0)
    D[1] = float(spec.get("k2") or 0.0)
    D[4] = float(spec.get("k3") or 0.0)
    src = (f"focal_px {K[0, 0]:.0f}, k1 {D[0]:+.3f} from the spec" if spec.get("focal_px")
           else f"guess from hfov {spec.get('hfov_deg', 90.0):.0f} deg (self-calibrates from the pivots)")
    return K, D, src


def estimate_intrinsics(samples, world_corners, image_size, K0, min_frames=5):
    """Focal length (and k1 with 3+ pivots) of a FIXED camera from pivots with known floor positions.

    samples: list of per-frame detections [(tag id, corners)], world_corners: {id: (4, 3)} on z = 0.
    Pixel corners are averaged over the frames first. Returns (K, D, rms_px, n_tags).
    One oblique view of a known plane fixes the focal length; principal point stays at the center."""
    acc = {}
    for dets in samples:
        for tid, c in dets:
            if tid in world_corners:
                acc.setdefault(tid, []).append(np.asarray(c, np.float64))
    use = sorted(t for t, v in acc.items() if len(v) >= min_frames)
    if len(use) < 2:
        raise RuntimeError(f"need 2+ pivots seen steadily, have {use}")
    obj = np.concatenate([world_corners[t] for t in use]).astype(np.float32)
    img = np.concatenate([np.mean(acc[t], 0) for t in use]).astype(np.float32)
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO |
             cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
    if len(use) < 3:
        flags |= cv2.CALIB_FIX_K1        # 8 points: focal only
    rms, K, D, _, _ = cv2.calibrateCamera([obj], [img], tuple(int(v) for v in image_size),
                                          np.asarray(K0, np.float64).copy(), np.zeros(5), flags=flags)
    f = float(K[0, 0])
    if not (0.25 * K0[0, 0] < f < 4.0 * K0[0, 0]) or abs(float(D.ravel()[0])) > 1.0:
        raise RuntimeError(f"implausible lens fit (f {f:.0f} px, k1 {float(D.ravel()[0]):+.2f}): "
                           "the view may be too straight-on; give calib= or focal_px=")
    return K, D.ravel().astype(np.float64), float(rms), len(use)


# --------------------------------------------------------------------------
# Cross-calibration: an extra camera learns its lens and pose from the main camera
# --------------------------------------------------------------------------

def flat_tag_world_corners(x, y, z, yaw_deg, size):
    """World corners of a tag lying flat (face up) at (x, y, z) with heading yaw (FloorModel's order)."""
    return (m.rot_z(yaw_deg) @ m.tag_corners_local(size).T).T + np.array([x, y, z], np.float64)


def load_lens_cache(path=LENS_CACHE_FILE):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def lens_key(name, width, height):
    return f"{name}@{int(width)}x{int(height)}"


def save_lens(name, width, height, K, D, rms, spots, path=LENS_CACHE_FILE):
    data = load_lens_cache(path)
    data[lens_key(name, width, height)] = {
        "fx": round(float(K[0, 0]), 3), "fy": round(float(K[1, 1]), 3),
        "cx": round(float(K[0, 2]), 3), "cy": round(float(K[1, 2]), 3),
        "D": [round(float(v), 6) for v in np.asarray(D).ravel()[:5]],
        "rms_px": round(float(rms), 3), "spots": int(spots), "saved": time.strftime("%Y-%m-%d %H:%M")}
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=1)
    except OSError:
        pass


def cached_lens(name, width, height, path=LENS_CACHE_FILE):
    """(K, D, entry) saved by an earlier cross-calibration, or None."""
    e = load_lens_cache(path).get(lens_key(name, width, height))
    if not e:
        return None
    K = np.array([[e["fx"], 0, e["cx"]], [0, e["fy"], e["cy"]], [0, 0, 1]], np.float64)
    D = np.zeros(5)
    D[:len(e["D"])] = e["D"]
    return K, D, e


def _camera_above_floor(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    return float((-R.T @ np.asarray(tvec).reshape(3))[2]) > 0


def fit_camera(obj, img, image_size, K=None, D=None, group=4):
    """Pose (and, if K is None, the lens) of a fixed camera from world <-> pixel correspondences.

    obj (N, 3) world points (any shape: floor and raised tags mixed), img (N, 2) pixels, in groups
    of `group` (one tag's corners). The lens model grows with the data: focal only (< 4 tags),
    + k1 (< 8), + k2 (< 16), + k3. Principal point at the image center, square pixels, no
    tangential distortion (fine for webcams and action cameras). Several focal starting guesses
    (60..150 deg field of view) are tried; tags that do not fit (> max(3 px, 4x median)) are
    dropped once. Returns (K, D, rvec, tvec, rms_px, n_groups_used); raises if nothing fits."""
    obj = np.asarray(obj, np.float64).reshape(-1, 3)
    img = np.asarray(img, np.float64).reshape(-1, 2)
    W, H = int(image_size[0]), int(image_size[1])

    def errors(K_, D_, rv, tv, o, i):
        p, _ = cv2.projectPoints(o, rv, tv, K_, D_)
        return np.linalg.norm(p.reshape(-1, 2) - i, axis=1)

    def solve(o, i):
        if K is not None:
            Kf, Df = np.asarray(K, np.float64), np.asarray(D, np.float64).ravel()
            ok, rv, tv = cv2.solvePnP(o, i, Kf, Df, flags=cv2.SOLVEPNP_SQPNP)
            if not ok:
                raise RuntimeError("pose solve failed")
            rv, tv = cv2.solvePnPRefineLM(o, i, Kf, Df, rv, tv)
            e = errors(Kf, Df, rv, tv, o, i)
            return Kf, Df, rv, tv, float(np.sqrt(np.mean(e ** 2)))
        ng = len(o) // group
        flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO |
                 cv2.CALIB_ZERO_TANGENT_DIST)
        if ng < 4:
            flags |= cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
        elif ng < 8:
            flags |= cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
        elif ng < 16:
            flags |= cv2.CALIB_FIX_K3
        crit = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 200, 1e-10)
        best = None
        for hfov in (60, 80, 100, 120, 150):
            K0 = guess_K(W, H, None, hfov)
            try:
                rms, K1, D1, rvs, tvs = cv2.calibrateCamera([o.astype(np.float32)], [i.astype(np.float32)],
                                                            (W, H), K0, np.zeros(5), flags=flags, criteria=crit)
            except cv2.error:
                continue
            f = float(K1[0, 0])
            if not (0.1 * W < f < 4.0 * W) or not _camera_above_floor(rvs[0], tvs[0]):
                continue
            if best is None or rms < best[4]:
                best = (K1, D1.ravel()[:5].astype(np.float64), rvs[0], tvs[0], float(rms))
        if best is None:
            raise RuntimeError("no plausible lens fit")
        return best

    sol = solve(obj, img)
    ng = len(obj) // group
    e = errors(sol[0], sol[1], sol[2], sol[3], obj, img).reshape(-1, group).mean(1)
    keep = e <= max(3.0, 4.0 * float(np.median(e)))
    if not keep.all() and keep.sum() >= max(2, ng // 2):
        m_ = np.repeat(keep, group)
        sol = solve(obj[m_], img[m_])
        ng = int(keep.sum())
    return sol + (ng,)


class RefHistory:
    """Recent world positions of flat tags as the MAIN camera saw them (for cross-calibration)."""

    def __init__(self, maxlen=240):
        self.lock = threading.Lock()
        self.hist = deque(maxlen=maxlen)       # (frame stamp, {tag id: (x, y, z, yaw_deg)})

    def add(self, stamp, objects):
        tags = {o["id"]: (o["x"], o["y"], o["z"], o.get("yaw_deg", 0.0)) for o in objects
                if o.get("frame") == "world" and o.get("method") == "plane" and o["id"] < 1000}
        with self.lock:
            self.hist.append((stamp, tags))

    def window(self, t0, t1):
        with self.lock:
            return [h for h in self.hist if t0 <= h[0] <= t1]


class CrossCalibration:
    """Collects (world corners, pixel corners) pairs of tags that stand still, one per floor 'spot',
    and says when there is enough for a (better) fit. Lives in the extra camera's thread."""

    SPOT_M = 0.06            # one sample per 6 cm cell and tag (a parked robot is one spot)
    STILL_M = 0.006          # main-camera position spread allowed over the window
    STILL_PX = 1.5           # this camera's corner motion allowed between frames
    MIN_SPOTS_LENS = 6       # spots before the first lens fit ...
    MIN_CELLS_LENS = 4       # ... spread over at least this many of the 4 x 3 image cells
    ENOUGH_SPOTS = 40        # the lens stops refining after this many spots

    def __init__(self, image_size, lens_fixed=False):
        self.W, self.H = image_size
        self.lens_fixed = lens_fixed       # True: calib file or a saved lens -> pose only
        self.spots = {}                    # key -> [world sum (4,3), pixel sum (4,2), n]
        self.last_px = {}                  # tag id -> (corners, stamp)
        self.solved_spots = 0
        self.result = None                 # last accepted (K, D, rvec, tvec, rms, n)
        self.last_error = None

    def add(self, key, world, px):
        s = self.spots.get(key)
        if s is None:
            self.spots[key] = [np.array(world, np.float64), np.array(px, np.float64), 1]
        else:
            s[0] += world
            s[1] += px
            s[2] += 1

    def feed(self, dets, stamp, ref, pivots, tag_size, skip=()):
        """dets: this camera's [(id, corners)] of one frame; ref: RefHistory of the main camera;
        pivots: {id: world corners}. A pivot is always a valid pair; another tag only when both
        cameras saw it standing still around this frame."""
        win = ref.window(stamp - 0.5, stamp + 0.15) if ref is not None else []
        for tid, c in dets:
            c = np.asarray(c, np.float64)
            prev = self.last_px.get(tid)
            self.last_px[tid] = (c, stamp)
            if tid in pivots:
                self.add(("pivot", tid), pivots[tid], c)
                continue
            if tid in skip or prev is None or stamp - prev[1] > 0.6 or np.abs(c - prev[0]).max() > self.STILL_PX:
                continue
            obs = [h[1][tid] for h in win if tid in h[1]]
            if len(obs) < 3:
                continue
            P = np.array([o[:3] for o in obs])
            if float(np.ptp(P[:, :2], axis=0).max()) > self.STILL_M:
                continue
            yaw = _circ_mean([o[3] for o in obs], np.ones(len(obs)))
            if max(abs((o[3] - yaw + 180) % 360 - 180) for o in obs) > 1.5:
                continue
            x, y, z = P.mean(0)
            key = (tid, round(x / self.SPOT_M), round(y / self.SPOT_M))
            self.add(key, flat_tag_world_corners(x, y, z, yaw, tag_size(tid)), c)

    def cells(self):
        occ = set()
        for _, px, n in self.spots.values():
            u, v = (px / n).mean(0)
            occ.add((min(3, int(4 * u / self.W)), min(2, int(3 * v / self.H))))
        return len(occ)

    def due(self, refine_lens):
        n = len(self.spots)
        if n <= self.solved_spots:
            return False
        if not refine_lens:                          # lens known: pose only
            return n >= 2 and (self.result is None or n >= self.solved_spots + 2)
        if n < self.MIN_SPOTS_LENS or self.cells() < self.MIN_CELLS_LENS:
            return False
        return self.result is None or n >= int(self.solved_spots * 1.3) + 1

    def solve(self, K=None, D=None):
        """K None: fit the lens too. Returns (K, D, rvec, tvec, rms, n spots used)."""
        self.solved_spots = len(self.spots)
        obj = np.concatenate([w / n for w, _, n in self.spots.values()])
        img = np.concatenate([p / n for _, p, n in self.spots.values()])
        return fit_camera(obj, img, (self.W, self.H), K, D)

    def progress(self):
        return {"spots": len(self.spots), "cells": self.cells()}


# --------------------------------------------------------------------------
# One extra camera
# --------------------------------------------------------------------------

class ExtraCamera:
    """A secondary camera with its own capture, detector and floor pose, running on its own thread.

    cfg is SHARED with the main camera (tag sizes, objects / heights, tags_on_floor): changes in
    the GUI apply here too. Only floor_tags (the pivots) are copied, see set_floor_tags()."""

    MAX_FIT_CALIB = 4.0       # allowed floor fit error with a calibration file (px), as the GUI
    MAX_FIT_GUESS = 12.0      # ... with only a focal length / k1

    def __init__(self, spec, cfg, smoothing=0.5, auto_lock=True, log=print, source=None, thumb_w=320):
        self.spec, self.cfg, self.log = spec, cfg, log
        self.name = spec["name"]
        self.smoothing, self.auto_lock = smoothing, auto_lock
        self.thumb_w = thumb_w
        self._source_override = source        # tests: anything with latest()/release()/width/height
        self.source = self.mapper = None
        self.index = None
        self.lock = threading.Lock()
        self.last = None                      # (result, frame stamp, processed at)
        self.thumb = None                     # small annotated BGR image for the GUI
        self.running = False
        self.thread = None
        self.pending_floor_tags = None
        self.error = None
        self.proc_fps = 0.0
        self.calibrated = bool(spec.get("calib"))
        self.selfcal = None                   # {"samples": [...]} while collecting
        self._selfcal_done = bool(spec.get("calib") or spec.get("focal_px"))
        self._selfcal_tries = 0
        self.K_source = ""
        self.ref = None                       # RefHistory of the main camera (set by CameraRig)
        self.crosscal = None                  # CrossCalibration (created in open())
        self.lens_spots = 0                   # spots behind the current lens (0 = not learned)
        self.lens_file = os.path.abspath(LENS_CACHE_FILE)

    # ---------------- lifecycle ----------------
    def open(self, exclude_index=None):
        """Open the device (raises with a readable message). Call before start()."""
        net = None
        if self._source_override is None:
            import insta360_wifi                       # wireless: camera=insta-wifi[:host] or a stream URL
            if insta360_wifi.is_wifi_spec(self.spec["camera"]) or insta360_wifi.is_url(self.spec["camera"]):
                net = insta360_wifi.open_network_source(self.spec["camera"], log=self.log)
        if self._source_override is not None:
            self.source = self._source_override
        elif net is not None:
            self.source = net
            self.log(f"Extra camera '{self.name}': wireless {self.spec['camera']} at "
                     f"{self.source.width}x{self.source.height}")
        else:
            idx, dev = m.resolve_camera(self.spec["camera"], "auto")
            if exclude_index is not None and idx == exclude_index:
                raise RuntimeError(f"extra camera '{self.name}' resolves to index {idx}, which is the main "
                                   "camera. Give its index or a more specific name.")
            self.index = idx
            self.source = m.CameraStream(idx, self.spec["width"], self.spec["height"], self.spec["fps"],
                                         m.backend_id(self.spec.get("backend", "auto")))
            self.log(f"Extra camera '{self.name}': opened {idx}" + (f" ({dev})" if dev else "")
                     + f" at {self.source.width}x{self.source.height}")
            if (self.source.width, self.source.height) != (self.spec["width"], self.spec["height"]):
                self.log(f"   asked for {self.spec['width']}x{self.spec['height']}, device gives "
                         f"{self.source.width}x{self.source.height}")
        K, D, self.K_source = intrinsics_for(self.spec, self.source.width, self.source.height)
        saved = None if (self.spec.get("calib") or self.spec.get("focal_px")) else \
            cached_lens(self.name, self.source.width, self.source.height, self.lens_file)
        if saved is not None:
            K, D, e = saved
            self.calibrated, self._selfcal_done = True, True
            self.lens_spots = int(e.get("spots", 0))
            self.K_source = (f"saved cross-calibration ({e.get('saved', '?')}): focal_px {K[0, 0]:.1f}, "
                             f"k1 {D[0]:+.3f} k2 {D[1]:+.3f}, {e.get('rms_px', 0):.2f} px over {self.lens_spots} spots")
        if int(self.spec.get("crosscal", 1)):
            self.crosscal = CrossCalibration((self.source.width, self.source.height),
                                             lens_fixed=bool(self.spec.get("calib")))
        # one detector thread: the main camera keeps most of the CPU (see max_fps)
        det = m.make_detector(self.cfg, engine="auto", detect_scale=1.0, nthreads=1,
                              decimate=float(self.spec.get("decimate", 2.0)))
        self.mapper = m.Mapper(self.cfg, K, D, det, smoothing=self.smoothing)
        self.mapper.floor = self._new_floor(self.cfg.get("floor_tags", {}))
        self.log(f"   '{self.name}' lens: {self.K_source}")
        return self

    def _fit_limit(self):
        return self.MAX_FIT_CALIB if self.calibrated else self.MAX_FIT_GUESS

    def _new_floor(self, floor_tags):
        pivots = {k: v for k, v in floor_tags.items() if not v.get("learned")}
        fl = m.FloorModel(pivots, self.mapper.tag_size, self._fit_limit())
        fl.auto_lock = self.auto_lock
        return fl

    def set_floor_tags(self, floor_tags):
        """The arena was set up again (new pivot positions): applied on this camera's thread."""
        self.pending_floor_tags = {int(k): dict(v) for k, v in floor_tags.items()}

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, name=f"extra-cam-{self.name}", daemon=True)
        self.thread.start()
        return self

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        if self.mapper is not None and hasattr(self.mapper.detector, "close"):
            self.mapper.detector.close()
        if self.source is not None and self._source_override is None:
            self.source.release()

    # ---------------- per frame ----------------
    def _loop(self):
        fid, n, t_fps = 0, 0, time.perf_counter()
        period = 1.0 / max(float(self.spec.get("max_fps") or 10.0), 0.5)
        t_next = time.perf_counter()
        while self.running:
            try:
                wait = t_next - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)                 # processing cap: CPU for the main camera
                frame, fid, stamp = self.source.latest(fid, timeout=0.5)
                if frame is None:
                    continue
                t_next = max(t_next + period, time.perf_counter() - period)
                if self.pending_floor_tags is not None:
                    ft, self.pending_floor_tags = self.pending_floor_tags, None
                    self.mapper.floor = self._new_floor(ft)
                    self.selfcal, self._selfcal_tries = None, 0
                    if self.crosscal is not None:     # new pivot positions: old pairs are void
                        self.crosscal = CrossCalibration((self.source.width, self.source.height),
                                                         self.crosscal.lens_fixed)
                r = self.mapper.process(frame, stamp, draw=self.thumb_w > 0)
                self._self_calibrate(r)
                self._cross_calibrate(r, stamp)
                now = time.perf_counter()
                with self.lock:
                    self.last = (r, stamp, now)
                n += 1
                if now - t_fps >= 1.0:
                    self.proc_fps, n, t_fps = n / (now - t_fps), 0, now
                if self.thumb_w > 0:
                    s = self.thumb_w / frame.shape[1]
                    self.thumb = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                self.error = None
            except Exception as e:          # noqa: BLE001 - one bad frame must not end this camera
                if repr(e) != self.error:
                    self.error = repr(e)
                    self.log(f"Extra camera '{self.name}': {e!r}")
                time.sleep(0.05)

    def _self_calibrate(self, r):
        """No calibration file and no focal_px: fit the lens to the pivots once (fixed camera)."""
        fl = self.mapper.floor
        if self._selfcal_done or self._selfcal_tries >= 3 or not fl.world_corners:
            return
        pivots = [(t, c) for t, c in r.get("_dets", []) if t in fl.world_corners]
        if len(pivots) < 2:
            self.selfcal = None
            return
        if self.selfcal is None:
            self.selfcal = {"samples": [], "t0": time.perf_counter()}
        self.selfcal["samples"].append(pivots)
        if len(self.selfcal["samples"]) < 20:
            return
        samples, self.selfcal = self.selfcal["samples"], None
        self._selfcal_tries += 1
        try:
            K, D, rms, n_tags = estimate_intrinsics(samples, fl.world_corners,
                                                    (self.source.width, self.source.height), self.mapper.K)
        except Exception as e:              # noqa: BLE001
            self.log(f"Extra camera '{self.name}': lens self-calibration failed ({e}); will retry")
            return
        if rms > self.MAX_FIT_GUESS:
            self.log(f"Extra camera '{self.name}': lens fit to the pivots {rms:.1f} px is too poor (wide / "
                     "fisheye lens?). It will learn the lens from the main camera instead: let a robot pause, "
                     "or walk a tag around the floor, at 6+ spots this camera and the main camera both see.")
            self._selfcal_tries = 3
            return
        if self.lens_spots:                   # the cross-calibration got there first (better model)
            self._selfcal_done = True
            return
        self.mapper.K, self.mapper.D = K, D
        self.mapper.floor = self._new_floor(self.cfg.get("floor_tags", {}))
        self._selfcal_done = True
        self.K_source = f"self-calibrated from {n_tags} pivots: focal_px {K[0, 0]:.1f}, k1 {D[0]:+.4f}, {rms:.2f} px"
        self.log(f"Extra camera '{self.name}': {self.K_source}. To skip this next time add "
                 f"focal_px={K[0, 0]:.1f},k1={D[0]:.4f} to its spec (at {self.source.width}x{self.source.height}).")

    def _cross_calibrate(self, r, stamp):
        """Learn the lens / pose from tags the main camera locates (see CrossCalibration)."""
        cc = self.crosscal
        if cc is None or not self.mapper.floor.world_corners:
            return
        fl = self.mapper.floor
        pivots = {t: c for t, c in fl.world_corners.items() if t not in fl.learned}
        offset = {t for t, s in self.cfg.get("objects", {}).items() if np.any(np.asarray(s.get("offset", 0)) != 0)}
        cc.feed(r.get("_dets", []), stamp, self.ref, pivots, self.mapper.tag_size, skip=offset)
        # refit the lens only with MORE spots than the current lens rests on (a saved 30-spot lens is
        # not replaced by a 7-spot fit after a restart); otherwise only the pose is solved
        refine = (not cc.lens_fixed and self.lens_spots < cc.ENOUGH_SPOTS
                  and len(cc.spots) > self.lens_spots)
        if not cc.due(refine):
            return
        try:
            K, D, rvec, tvec, rms, n = cc.solve(None if refine else self.mapper.K,
                                                None if refine else self.mapper.D)
        except Exception as e:              # noqa: BLE001
            cc.last_error = str(e)
            return
        limit = self.MAX_FIT_CALIB if refine or self.calibrated else self.MAX_FIT_GUESS
        if rms > limit:
            cc.last_error = f"fit {rms:.1f} px over {n} spots (> {limit:g})"
            return
        if refine and cc.result is not None and self.lens_spots and rms > 1.5 * cc.result[4] + 0.3:
            cc.last_error = f"refit {rms:.2f} px is worse than {cc.result[4]:.2f} px: kept the old lens"
            return
        cc.result, cc.last_error = (K, D, rvec, tvec, rms, n), None
        now = time.perf_counter()
        if refine:
            self.mapper.K, self.mapper.D = K, D
            self.calibrated, self._selfcal_done, self.selfcal = True, True, None
            self.lens_spots = n
            self.K_source = (f"learned from the main camera: focal_px {K[0, 0]:.1f}, k1 {D[0]:+.3f} "
                             f"k2 {D[1]:+.3f} k3 {D[4]:+.3f}, {rms:.2f} px over {n} spots")
            save_lens(self.name, self.source.width, self.source.height, K, D, rms, n, self.lens_file)
        floor = self._new_floor(self.cfg.get("floor_tags", {}))
        floor._set(rvec, tvec, rms, [], now)
        floor.locked, floor.lock_rms = True, max(rms, 0.5)
        self.mapper.floor = floor
        C = floor.camera_position
        self.log(f"Extra camera '{self.name}': pose from the main camera's tags ({n} spots, {rms:.2f} px), "
                 f"camera at ({C[0]:+.2f}, {C[1]:+.2f}, {C[2]:.2f}) m"
                 + (f"; lens {self.K_source}" if refine else ""))

    # ---------------- for the fusion ----------------
    def snapshot(self):
        """(result, frame stamp) of the newest processed frame, or (None, None)."""
        with self.lock:
            return (self.last[0], self.last[1]) if self.last else (None, None)

    @property
    def floor(self):
        return self.mapper.floor if self.mapper is not None else None

    def status(self, now=None):
        now = time.perf_counter() if now is None else now
        r, stamp = self.snapshot()
        fl = self.floor
        st = {"name": self.name, "fps": round(self.proc_fps, 1),
              "grab_fps": round(getattr(self.source, "grab_fps", 0.0), 1),
              "age_ms": None if stamp is None else round((now - stamp) * 1e3),
              "floor": "none" if fl is None or not fl.world_corners else
                       ("locked" if fl.locked else "ok") if fl.valid else
                       (f"rejected {fl.last_reject[0]:.1f} px" if fl.last_reject else "no pivot in view"),
              "pivots": [] if r is None else list(r.get("floor_tags_seen", [])),
              "tags": [] if r is None else sorted(o["id"] for o in r["objects"] if o.get("frame") == "world"),
              "lens": self.K_source, "error": self.error,
              "selfcal": None if self.selfcal is None else len(self.selfcal["samples"]),
              "crosscal": None}
        cc = self.crosscal
        if cc is not None and not cc.lens_fixed and self.lens_spots < cc.ENOUGH_SPOTS:
            p = cc.progress()
            st["crosscal"] = dict(p, lens=bool(self.lens_spots), error=cc.last_error)
        return st


# --------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------

PIXEL_NOISE = 0.3       # px, corner noise of a sharp tag


def observation_sigma(o, fx, tag_size):
    """1-sigma position uncertainty (m) of one camera's tag observation."""
    d = float(o.get("distance_m") or 1.0)
    px = max(float(o.get("reproj_px") or 0.0), PIXEL_NOISE)
    if o.get("method") == "plane":
        # ray-plane intersection: lateral error ~ d * pixel error / f, worse at grazing angles
        return max(1e-4, 2.0 * d * px / fx)
    # size-based depth from a single tag
    return max(1e-3, d * d * px / (fx * max(tag_size, 0.01)))


def _circ_mean(deg, w):
    a = np.radians(np.asarray(deg, float))
    return math.degrees(math.atan2(float(np.sum(w * np.sin(a))), float(np.sum(w * np.cos(a)))))


def fuse_tags(views, now, max_age_s=0.25, age_speed=0.5, min_gate=0.03):
    """Fuse world-frame tag observations of several cameras.

    views: [{"name", "result", "stamp", "fx", "tag_size": f(id), "floor_valid"}]
    age_speed: m/s a tag may have moved since an older frame (turns age into uncertainty).
    Returns {tag id: fused object dict}. Untagged objects (ids >= 1000) are ignored here."""
    obs = {}
    for v in views:
        r = v.get("result")
        if r is None or not v.get("floor_valid"):
            continue
        age = max(0.0, now - v["stamp"])
        if age > max_age_s:
            continue
        for o in r.get("objects", []):
            if o.get("frame") != "world" or o.get("method") in ("detect", "appearance") or o["id"] >= 1000:
                continue
            if not (math.isfinite(o["x"]) and math.isfinite(o["y"])):
                continue
            s = observation_sigma(o, v["fx"], v["tag_size"](o["id"]))
            # how well this camera's floor pose fits (px) adds lateral error ~ distance * px / f: a
            # camera on a poor lens model (12 px fit) counts far less than one at 0.5 px
            fpx = float(v.get("floor_px") or 0.0)
            s = math.hypot(s, float(o.get("distance_m") or 1.0) * fpx / v["fx"], age_speed * age)
            obs.setdefault(o["id"], []).append((o, s, v["name"], age))
    fused = {}
    for tid, lst in obs.items():
        if len(lst) == 1:
            o, s, name, _ = lst[0]
            fused[tid] = dict(o, cams=[name], n_cams=1, sigma_m=round(s, 4))
            continue
        xy = np.array([[o["x"], o["y"]] for o, *_ in lst])
        sig = np.array([s for _, s, *_ in lst])
        # largest set of observations that agree (ties: smaller total variance)
        best = None
        for i in range(len(lst)):
            gate = np.maximum(min_gate, 3.0 * np.hypot(sig, sig[i]))
            inl = np.linalg.norm(xy - xy[i], axis=1) <= gate
            key = (int(inl.sum()), -float(np.sum(sig[inl] ** 2)))
            if best is None or key > best[0]:
                best = (key, inl)
        inl = best[1]
        w = 1.0 / sig[inl] ** 2
        w /= w.sum()
        sel = [lst[k] for k in np.flatnonzero(inl)]
        ref = min(sel, key=lambda t: t[1])[0]          # best observation: keeps its other fields
        p = (xy[inl] * w[:, None]).sum(0)
        out = dict(ref)
        out.update(x=float(p[0]), y=float(p[1]),
                   z=float(sum(wk * t[0]["z"] for wk, t in zip(w, sel))),
                   yaw_deg=_circ_mean([t[0].get("yaw_deg", 0.0) for t in sel], w),
                   cams=[t[2] for t in sel], n_cams=len(sel),
                   sigma_m=round(float(1.0 / math.sqrt(np.sum(1.0 / sig[inl] ** 2))), 4))
        rejected = [lst[k][2] for k in np.flatnonzero(~inl)]
        if rejected:
            out["rejected_cams"] = rejected
        fused[tid] = out
    return fused


class CameraRig:
    """The extra cameras plus fusion with the main camera's per-frame result."""

    def __init__(self, extras, main_name="main", max_age_s=0.25):
        self.extras = list(extras)
        self.main_name = main_name
        self.max_age_s = max_age_s
        self.history = deque(maxlen=300)        # (time, {tid: n_cams}) for the status line
        self.ref = RefHistory()                 # main camera's flat tags -> extras' cross-calibration
        for e in self.extras:
            e.ref = self.ref

    def __bool__(self):
        return bool(self.extras)

    def set_floor_tags(self, floor_tags):
        for e in self.extras:
            e.set_floor_tags(floor_tags)

    def stop(self):
        for e in self.extras:
            e.stop()

    def views(self, main_result, main_stamp, main_floor, main_K, tag_size):
        vs = []
        if main_result is not None:
            vs.append({"name": self.main_name, "result": main_result, "stamp": main_stamp,
                       "fx": float(main_K[0, 0]), "tag_size": tag_size,
                       "floor_valid": bool(main_floor is not None and main_floor.valid),
                       "floor_px": getattr(main_floor, "reproj", None)})
        for e in self.extras:
            r, st = e.snapshot()
            if r is None:
                continue
            vs.append({"name": e.name, "result": r, "stamp": st, "fx": float(e.mapper.K[0, 0]),
                       "tag_size": e.mapper.tag_size, "floor_valid": bool(e.floor is not None and e.floor.valid),
                       "floor_px": getattr(e.floor, "reproj", None)})
        return vs

    def fuse_into(self, r, stamp, main_floor, main_K, tag_size, now=None):
        """Replace the tag objects in the main result `r` with the fused ones (in place).
        r may be None (main camera silent): returns a new minimal result from the extras only.
        Pixel data (_dets) always stays the main camera's own."""
        now = time.perf_counter() if now is None else now
        if r is not None and main_floor is not None and main_floor.valid:
            self.ref.add(stamp, r.get("objects", []))     # the main camera's OWN view, before fusion
        fused = fuse_tags(self.views(r, stamp, main_floor, main_K, tag_size), now, self.max_age_s)
        if r is None:
            r = {"t": now, "objects": [], "_dets": [], "n_tags": 0, "camera": None,
                 "floor_valid": False, "floor_locked": False, "floor_tags_seen": [],
                 "floor_reproj_px": None, "floor_reject_px": None, "floor_limit_px": 0.0,
                 "timing_ms": {"detect": 0.0, "total": 0.0}, "main_camera_silent": True}
        # untagged objects and camera-frame tags (no floor yet) stay as the main camera made them
        keep = [o for o in r["objects"]
                if not (o.get("frame") == "world" and o.get("method") not in ("detect", "appearance")
                        and o["id"] < 1000)]
        r["objects"] = list(fused.values()) + keep
        cams = {}
        for e in self.extras:
            fl = e.floor
            if fl is not None and fl.valid:
                C = fl.camera_position
                cams[e.name] = [round(float(C[0]), 3), round(float(C[1]), 3), round(float(C[2]), 3)]
        if r.get("camera"):
            c = r["camera"]
            cams = {self.main_name: [round(c["x"], 3), round(c["y"], 3), round(c["z"], 3)], **cams}
        r["cameras"] = cams
        self.history.append((now, {t: o["n_cams"] for t, o in fused.items()}))
        return r

    def pub_floor(self, main_floor):
        """A valid floor for the publisher (pivot positions are the same for all cameras)."""
        if main_floor is not None and main_floor.valid:
            return main_floor
        for e in self.extras:
            if e.floor is not None and e.floor.valid:
                return e.floor
        return main_floor

    def status_text(self, now=None):
        now = time.perf_counter() if now is None else now
        parts = []
        for e in self.extras:
            s = e.status(now)
            line = f"{s['name']}: {s['fps']:.0f} fps, floor {s['floor']}"
            if s["age_ms"] is not None and s["age_ms"] > 500:
                line += f", SILENT {s['age_ms'] / 1000:.1f} s"
            if s["tags"]:
                line += f", tags {s['tags']}"
            if s["selfcal"] is not None:
                line += f", fitting lens {s['selfcal']}/20"
            cc = s.get("crosscal")
            if cc and not cc["lens"] and not s["floor"] in ("ok", "locked"):
                line += (f", learning from main cam: {cc['spots']} spots in {cc['cells']}/12 areas "
                         f"(need {CrossCalibration.MIN_SPOTS_LENS} in {CrossCalibration.MIN_CELLS_LENS}; "
                         "pause a robot / move a tag around)")
            elif cc and cc["lens"]:
                line += f", lens refining ({cc['spots']} spots)"
            if s["error"]:
                line += f", error {s['error'][:60]}"
            parts.append(line)
        return "\n".join(parts)


def open_extras(specs, cfg, smoothing=0.5, auto_lock=True, log=print, exclude_index=None):
    """Open every spec that works; a camera that fails to open is reported and skipped."""
    out, used = [], {exclude_index} - {None}
    for spec in specs:
        e = ExtraCamera(spec, cfg, smoothing, auto_lock, log)
        try:
            e.open(exclude_index)
            if e.index is not None and e.index in used:
                raise RuntimeError(f"index {e.index} is already in use by another camera")
            used.add(e.index)
            out.append(e.start())
        except Exception as ex:             # noqa: BLE001
            try:
                e.stop()                    # release a device that opened before the failure
            except Exception:               # noqa: BLE001
                pass
            log(f"Extra camera '{spec.get('name')}' not used: {ex}")
    return out
