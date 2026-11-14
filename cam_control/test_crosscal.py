#!/usr/bin/env python3
"""
Cross-calibration test: a second camera with a strongly distorted wide lens (like the GO 3S),
mounted upside down and seeing only 2 of the 4 pivots, learns its lens and pose from the main
camera while the robot stops at spots around the arena.

    python test_crosscal.py

1. fit_camera on exact geometry (no images): lens + pose recovered from tag corners.
2. End to end on rendered frames, simulated time (no threads): main Mapper -> RefHistory ->
   ExtraCamera._cross_calibrate. Checks: lens learned, camera position, robot error in the extra
   camera alone, and fusion of both cameras; then a restart that uses the saved lens.
"""
import math
import os
import sys
import tempfile
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import apriltag_floor_mapper as m  # noqa: E402
import multi_camera as mc  # noqa: E402
from apriltag_tracker_gui import SyntheticScene  # noqa: E402

FPS = 15.0
D_WIDE = np.array([-0.32, 0.12, 0.0, 0.0, -0.02])     # strong barrel distortion
K_WIDE = np.array([[640.0, 0, 640], [0, 640.0, 360], [0, 0, 1]])
WIDE_C, WIDE_TARGET = (0.6, 0.95, 0.6), (0.6, 0.0, 0.0)   # sees only pivots 29 and 30


class StopAndGoScene(SyntheticScene):
    """The robot drives to a grid of spots and pauses 1.5 s at each one (like a robot at work)."""
    SPOTS = [(x, y) for y in (0.18, 0.40, 0.62) for x in (0.20, 0.47, 0.74, 1.00)]
    DWELL, MOVE = 1.5, 0.5

    def robot_pose(self, t):
        per = self.DWELL + self.MOVE
        k, f = int(t // per), (t % per)
        a, b = self.SPOTS[k % len(self.SPOTS)], self.SPOTS[(k + 1) % len(self.SPOTS)]
        s = 0.0 if f < self.DWELL else (f - self.DWELL) / self.MOVE
        hd = 30.0 * (k % 5)
        return a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1]), hd


def wide_view(scene, C, target):
    scene.view_from(C, target)
    scene.R = np.diag([-1.0, -1.0, 1.0]) @ scene.R          # upside down (as the GO 3S hangs)
    scene.rvec, _ = cv2.Rodrigues(scene.R)
    scene.tvec = -scene.R @ scene.C
    scene.K, scene.D = K_WIDE.copy(), D_WIDE.copy()
    return scene


def test_fit_camera_geometry():
    rng = np.random.default_rng(1)
    C = np.array([0.3, 1.6, 1.4])
    sc = wide_view(SyntheticScene(), C, (0.7, 0.3, 0.0))
    obj, img = [], []
    for _ in range(30):
        x, y, z, yaw = rng.uniform(0, 1.2), rng.uniform(0, 0.8), rng.choice([0.0, 0.05]), rng.uniform(0, 360)
        w = mc.flat_tag_world_corners(x, y, z, yaw, 0.05)
        p = sc._proj(w)
        if (p < 0).any() or (p[:, 0] > 1280).any() or (p[:, 1] > 720).any():
            continue
        obj.append(w)
        img.append(p + rng.normal(0, 0.3, p.shape))
    K, D, rvec, tvec, rms, n = mc.fit_camera(np.concatenate(obj), np.concatenate(img), (1280, 720))
    R, _ = cv2.Rodrigues(rvec)
    Cf = -R.T @ tvec.ravel()
    print(f"  {n} tags: focal {K[0, 0]:.1f} (true 640), k1 {D[0]:+.3f} k2 {D[1]:+.3f} (true -0.32, +0.12), "
          f"{rms:.2f} px, camera off by {1e3 * np.linalg.norm(Cf - C):.1f} mm")
    assert abs(K[0, 0] - 640) < 15 and np.linalg.norm(Cf - C) < 0.02 and rms < 0.6


class NullSource:
    width, height = 1280, 720
    grab_fps = FPS

    def release(self):
        pass


def robot_err(r, truth):
    o = next((o for o in r["objects"] if o["id"] == 90 and o.get("frame") == "world"), None)
    return None if o is None else 1e3 * math.hypot(o["x"] - truth[0], o["y"] - truth[1])


def run_end_to_end(lens_file, seconds, label):
    main = StopAndGoScene()
    main.cfg["tags_on_floor"] = True
    cfg = main.cfg
    # the wide camera sees the arena from the other side; only 2 pivots are in its view
    wide = wide_view(StopAndGoScene(), WIDE_C, WIDE_TARGET)
    mp = m.Mapper(cfg, main.K, main.D, m.make_detector(cfg, nthreads=4, decimate=1.0, realtime=False))
    mp.floor = m.FloorModel(cfg["floor_tags"], mp.tag_size, 4.0)
    logs = []
    extra = mc.ExtraCamera(mc.parse_spec("name=wide,camera=x,width=1280,height=720"), cfg,
                           log=logs.append, source=NullSource(), thumb_w=0)
    extra.lens_file = lens_file
    extra.open()
    rig = mc.CameraRig([extra])
    err_extra, err_main, err_fused, first_ok = [], [], [], None
    pivots_seen = set()
    for k in range(int(seconds * FPS)):
        t = k / FPS
        main.t0 = wide.t0 = time.perf_counter() - t     # both cameras render the same instant
        f_main, truth = main.read()
        f_wide, _ = wide.read()
        r_main = mp.process(f_main, t, draw=False)
        r_e = extra.mapper.process(f_wide, t, draw=False)
        pivots_seen |= {tid for tid, _ in r_e["_dets"] if tid in (29, 30, 31, 32)}
        extra._cross_calibrate(r_e, t)
        with extra.lock:
            extra.last = (r_e, t, t)
        rig.ref.add(t, r_main["objects"]) if mp.floor.valid else None
        if extra.crosscal.result is None:          # measure once the main camera's fit took over
            continue
        first_ok = first_ok if first_ok is not None else t
        e1 = robot_err(r_e, truth[90])
        e0 = robot_err(r_main, truth[90])
        fused = mc.fuse_tags(rig.views(r_main, t, mp.floor, mp.K, mp.tag_size), t)
        if e1 is not None:
            err_extra.append(e1)
        if e0 is not None:
            err_main.append(e0)
        if 90 in fused:
            err_fused.append(1e3 * math.hypot(fused[90]["x"] - truth[90][0], fused[90]["y"] - truth[90][1]))
    C = extra.floor.camera_position if extra.floor.valid else None
    print(f"  [{label}] wide camera sees pivots {sorted(pivots_seen)}; lens: {extra.K_source}")
    print(f"  [{label}] pose from the main camera after {first_ok if first_ok is not None else float('nan'):.1f} s, "
          f"camera off by {1e3 * np.linalg.norm(C - wide.C):.0f} mm" if C is not None else f"  [{label}] NO POSE")
    for name, e in (("wide camera alone", err_extra), ("main camera alone", err_main), ("fused", err_fused)):
        if e:
            print(f"  [{label}] robot error, {name}: median {np.median(e):.1f} mm, 95% {np.percentile(e, 95):.1f} mm"
                  f" ({len(e)} frames)")
    return extra, err_extra, err_fused, first_ok, logs


def test_end_to_end():
    lens_file = os.path.join(tempfile.mkdtemp(), "camera_lens.json")
    extra, e_x, e_f, t_ok, logs = run_end_to_end(lens_file, 30.0, "first run")
    assert extra.floor.valid and t_ok is not None, logs
    assert extra.lens_spots >= 6 and abs(extra.mapper.K[0, 0] - 640) < 40, extra.K_source
    assert np.median(e_x) < 15, np.median(e_x)
    assert np.median(e_f) < 6, np.median(e_f)
    assert os.path.exists(lens_file)
    # restart: lens comes from camera_lens.json, only the pose is learned (2 spots: the 2 pivots)
    spots = extra.lens_spots
    extra2, e_x2, _, t_ok2, _ = run_end_to_end(lens_file, 8.0, "restart")
    assert extra2.floor.valid and t_ok2 is not None and t_ok2 < t_ok, (t_ok2, t_ok)
    assert extra2.lens_spots == spots, "the saved lens was replaced by a fit on fewer spots"
    assert np.median(e_x2) < 15


def test_baseline_pivots_only():
    """Without cross-calibration the same camera has no usable floor (only 2 pivots, wide lens)."""
    main = StopAndGoScene()
    main.cfg["tags_on_floor"] = True
    wide = wide_view(StopAndGoScene(), WIDE_C, WIDE_TARGET)
    extra = mc.ExtraCamera(mc.parse_spec("name=wide,camera=x,width=1280,height=720,crosscal=0"), main.cfg,
                           log=lambda s: None, source=NullSource(), thumb_w=0)
    extra.lens_file = os.path.join(tempfile.mkdtemp(), "none.json")
    extra.open()
    errs = []
    for k in range(int(6 * FPS)):
        t = k / FPS
        wide.t0 = time.perf_counter() - t
        f, truth = wide.read()
        r = extra.mapper.process(f, t, draw=False)
        extra._self_calibrate(r)
        e = robot_err(r, truth[90]) if extra.floor.valid else None
        if e is not None:
            errs.append(e)
    print(f"  pivots only (old behaviour): floor {'valid' if extra.floor.valid else 'NOT valid'}"
          + (f", robot error median {np.median(errs):.0f} mm" if errs else ", no robot position"))


if __name__ == "__main__":
    for name, fn in (("fit_camera on exact geometry", test_fit_camera_geometry),
                     ("baseline: pivots only", test_baseline_pivots_only),
                     ("end to end: learn from the main camera", test_end_to_end)):
        print(f"[{name}]")
        fn()
        print("  ok")
    print("all cross-calibration tests passed")
