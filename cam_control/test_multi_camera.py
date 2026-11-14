#!/usr/bin/env python3
"""
Multi-camera tests without hardware: the GUI's synthetic arena rendered from two or three
viewpoints (known ground truth).

    python test_multi_camera.py

1. fusion accuracy: main camera alone vs extra camera alone vs fused, against ground truth
2. outlier rejection: one camera reporting a tag 20 cm off is voted out
3. dropout: main camera silent -> the extra camera alone still gives the robot pose
4. self-calibration: an extra camera with a wrong focal guess fits its lens to the pivots
5. spec parsing
"""
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cv2  # noqa: E402
import apriltag_floor_mapper as m  # noqa: E402
import multi_camera as mc  # noqa: E402
from apriltag_tracker_gui import PacedSource, SyntheticScene  # noqa: E402

ROBOT = 90


def place_camera(scene, C, target=(0.6, 0.4, 0.0)):
    """Move a SyntheticScene's camera (same arena, other viewpoint)."""
    scene.C = np.asarray(C, float)
    z = (np.asarray(target) - scene.C) / np.linalg.norm(np.asarray(target) - scene.C)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    scene.R = np.vstack([x, np.cross(z, x), z])
    scene.rvec, _ = cv2.Rodrigues(scene.R)
    scene.tvec = -scene.R @ scene.C
    return scene


def make_scenes():
    a = SyntheticScene()
    b = place_camera(SyntheticScene(), (0.6, 1.55, 1.3))
    c = place_camera(SyntheticScene(), (1.9, 0.4, 1.2))
    for s in (b, c):
        s.t0 = a.t0                      # same robot motion in every view
    cfg = a.cfg
    cfg["tags_on_floor"] = True
    return a, b, c, cfg


def mapper_for(cfg, K):
    return m.Mapper(cfg, K, np.zeros(5), m.make_detector(cfg, nthreads=2, decimate=1.0))


def err_mm(o, truth):
    return 1e3 * math.hypot(o["x"] - truth[0], o["y"] - truth[1])


def test_fusion_and_outliers():
    a, b, c, cfg = make_scenes()
    ma, mb, mcam = (mapper_for(cfg, s.K) for s in (a, b, c))
    errs = {"main": [], "extra": [], "fused": []}
    n_two, outlier_ok = 0, 0
    for k in range(60):
        # frozen simulated time: all three views show the robot at the same instant (rendering a
        # view takes ~10-30 ms of wall clock, during which the robot would otherwise move)
        now = 10.0 + 0.05 * k
        frames = []
        for s in (a, b, c):
            s.t0 = time.perf_counter() - now
            frames.append(s.read())
        (fa, ta), (fb, _), (fc, _) = frames
        ra, rb, rc = ma.process(fa, now, draw=False), mb.process(fb, now, draw=False), mcam.process(fc, now, draw=False)
        if k < 20 or ROBOT not in ta:
            continue
        views = [{"name": n, "result": r, "stamp": now, "fx": 1000.0, "tag_size": mp.tag_size,
                  "floor_valid": mp.floor.valid} for n, r, mp in (("main", ra, ma), ("extra", rb, mb))]
        fused = mc.fuse_tags(views, now)
        truth = ta[ROBOT]
        for name, r in (("main", ra), ("extra", rb)):
            o = next((o for o in r["objects"] if o["id"] == ROBOT and o.get("frame") == "world"), None)
            if o:
                errs[name].append(err_mm(o, truth))
        if ROBOT in fused:
            errs["fused"].append(err_mm(fused[ROBOT], truth))
            n_two += fused[ROBOT]["n_cams"] == 2

        # outlier: a third camera reports the robot 20 cm away -> must be rejected
        bad = [dict(o, x=o["x"] + 0.20) if o["id"] == ROBOT else o for o in rc["objects"]]
        views3 = views + [{"name": "bad", "result": dict(rc, objects=bad), "stamp": now, "fx": 1000.0,
                           "tag_size": mcam.tag_size, "floor_valid": mcam.floor.valid}]
        f3 = mc.fuse_tags(views3, now)
        if ROBOT in f3 and "bad" in f3[ROBOT].get("rejected_cams", []) and err_mm(f3[ROBOT], truth) < 10:
            outlier_ok += 1
    rep = {k: (len(v), round(float(np.median(v)), 2) if v else None, round(float(np.max(v)), 2) if v else None)
           for k, v in errs.items()}
    print("  robot position error, (frames, median mm, max mm):", rep)
    assert errs["fused"] and np.median(errs["fused"]) < 5.0, rep
    assert n_two >= 0.8 * len(errs["fused"]), f"both cameras fused in only {n_two} frames"
    assert outlier_ok >= 0.8 * len(errs["fused"]), f"outlier rejected in only {outlier_ok} frames"
    print(f"  both cameras fused in {n_two}/{len(errs['fused'])} frames; "
          f"20 cm outlier rejected in {outlier_ok}/{len(errs['fused'])}")


def test_dropout_and_selfcal():
    a, b, _, cfg = make_scenes()
    logs = []
    src = PacedSource(b.read, 20)
    # no calib, no focal_px: starts from a 90 deg guess (f 640 px, true 1000 px) and must self-calibrate
    spec = mc.parse_spec("name=side,camera=synthetic,width=1280,height=720")
    extra = mc.ExtraCamera(spec, cfg, log=logs.append, source=src, thumb_w=0).open().start()
    rig = mc.CameraRig([extra])
    try:
        t_end = time.perf_counter() + 12.0
        errs = []
        while time.perf_counter() < t_end:
            time.sleep(0.05)
            r = rig.fuse_into(None, None, None, np.eye(3) * 1000, lambda t: 0.05)  # main camera silent
            robot = next((o for o in r["objects"] if o["id"] == ROBOT), None)
            _, stamp = extra.snapshot()
            if robot is not None and extra._selfcal_done and stamp is not None:
                truth = a.robot_pose(stamp - a.t0)        # where the robot was in that frame
                errs.append(err_mm(robot, truth))
        print("  log:", *logs[-2:], sep="\n    ")
        assert extra._selfcal_done, f"lens not self-calibrated: {logs}"
        f = extra.mapper.K[0, 0]
        assert abs(f - 1000.0) < 30, f"focal {f:.1f} px, true 1000"
        assert errs, "no robot pose from the extra camera while the main camera was silent"
        med = float(np.median(errs))
        print(f"  main camera silent: robot from the extra camera alone in {len(errs)} polls, "
              f"median error {med:.1f} mm vs truth at the frame time; focal {f:.1f} px")
        assert med < 15.0, med
        assert rig.pub_floor(None) is extra.floor and extra.floor.valid
        assert "side" in r["cameras"]
    finally:
        rig.stop()
        src.release()


def test_specs():
    s = mc.parse_spec("name=go3s,camera=Insta360,fps=30,focal_px=1100.5,k1=-0.21")
    assert s == {"name": "go3s", "camera": "Insta360", "fps": 30, "focal_px": 1100.5, "k1": -0.21,
                 "width": 1920, "height": 1080, "backend": "auto", "decimate": 1.0, "max_fps": 10.0}, s
    assert mc.parse_spec("2")["camera"] == "2"
    assert len(mc.parse_specs("Insta360; name=b,camera=3")) == 2
    try:
        mc.parse_spec("camera=1,colour=red")
        raise AssertionError("unknown key accepted")
    except ValueError:
        pass
    assert mc.parse_spec(mc.spec_text(s)) == s


if __name__ == "__main__":
    for name, fn in (("specs", test_specs), ("fusion + outlier rejection", test_fusion_and_outliers),
                     ("dropout + lens self-calibration", test_dropout_and_selfcal)):
        print(f"[{name}]")
        fn()
        print("  ok")
    print("all multi-camera tests passed")
    sys.stdout.flush()
    os._exit(0)          # pupil_apriltags detectors must not be destructed (see the GUI)
