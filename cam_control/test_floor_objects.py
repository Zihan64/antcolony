#!/usr/bin/env python3
"""
Untagged-object detection on a streaked carpet (the real floor has long blue streaks).

    python test_floor_objects.py

The synthetic arena's floor gets long thin streaks of another color. The appearance detector
("anything that does not look like the floor") flagged such streaks as objects. Checks:
  1. no reference: the streaks are not objects, the untagged boxes still are;
  2. with an empty-floor reference (Capture empty floor) taken WITH the streaks: still no streak
     objects, and a box that appears afterwards is found.
"""
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import apriltag_floor_mapper as m  # noqa: E402
import arena_map as am  # noqa: E402
from apriltag_tracker_gui import SyntheticScene  # noqa: E402

BOXES = {"red box": (0.57, 0.40), "green box (on the floor)": (1.41, 0.24), "crate": (0.98, 0.62)}


def streaked_scene():
    sc = SyntheticScene()
    rng = np.random.default_rng(3)
    for _ in range(14):                     # long thin blue streaks, like the carpet
        x0, y0 = int(rng.integers(0, sc.W)), int(rng.integers(0, sc.H))
        ln, ang = int(rng.integers(150, 400)), np.radians(rng.uniform(-12, 12))
        x1, y1 = int(x0 + ln * np.cos(ang)), int(y0 + ln * np.sin(ang))
        cv2.line(sc.floor_bgr, (x0, y0), (x1, y1), (150, 120, 40), int(rng.integers(2, 4)), cv2.LINE_AA)
    return sc


def run(sc, frames, gm=None, capture_at=None, empty_until=0):
    mp = m.Mapper(sc.cfg, sc.K, sc.D, m.make_detector(sc.cfg, nthreads=4, decimate=1.0, realtime=False))
    mp.floor = m.FloorModel(sc.cfg["floor_tags"], mp.tag_size, 4.0)
    a = dict(am.ARENA_DEFAULTS, **sc.cfg["arena"])
    a["map_resolution_m"] = a.get("map_resolution_m") or 0.004
    objs = []
    for k in range(frames):
        if k < empty_until:
            sc.empty_frames = 1
        f, _ = sc.read()
        raw = f.copy()
        mp.process(f, float(k), draw=False)
        if not mp.floor.valid:
            continue
        if gm is None:
            gm = am.GlobalMap(a, sc.cfg["floor_tags"], res=a["map_resolution_m"], floor=mp.floor, K=sc.K, D=sc.D)
            gm.detect_appearance = True
        gm.update(raw, mp.floor, sc.K, sc.D, float(k), max_range=4.0, min_px_per_cell=0.35,
                  capture_bg=(k == capture_at))
        objs = gm.objects_in_view()
    return gm, objs


def classify(objs):
    found, other = set(), []
    for o in objs:
        name = next((n for n, (x, y) in BOXES.items() if np.hypot(o["x"] - x, o["y"] - y) < 0.1), None)
        if name:
            found.add(name)
        else:
            other.append((round(o["x"], 2), round(o["y"], 2), round(o["width_m"], 3), round(o["depth_m"], 3)))
    return found, other


class NullSource:
    width, height = SyntheticScene.W, SyntheticScene.H
    grab_fps = 15.0

    def release(self):
        pass


def test_and_gate():
    """Objects: AND gate (main + extra camera). Tags: OR gate (either camera)."""
    import multi_camera as mc
    main = SyntheticScene()
    side = SyntheticScene().view_from((1.9, 0.4, 1.2))
    # a "phantom" only the main camera sees (glare / its own view of the carpet): a dark patch in
    # the main camera's image only, at a floor spot both cameras look at
    cv2.rectangle(main.floor_bgr, (420, 440), (490, 495), (40, 60, 200), -1)
    cfg = main.cfg
    mp = m.Mapper(cfg, main.K, main.D, m.make_detector(cfg, nthreads=4, decimate=1.0, realtime=False))
    mp.floor = m.FloorModel(cfg["floor_tags"], mp.tag_size, 4.0)
    extra = mc.ExtraCamera(mc.parse_spec(f"name=side,camera=x,width={main.W},height={main.H},"
                                         f"focal_px={side.K[0, 0]},crosscal=0"), cfg,
                           log=lambda s: None, source=NullSource(), thumb_w=0)
    extra.open()
    extra.OBJ_PERIOD = 0.0
    rig = mc.CameraRig([extra])
    a = dict(am.ARENA_DEFAULTS, **cfg["arena"])
    a["map_resolution_m"] = 0.004
    gms = {}
    for k in range(14):
        main.t0 = side.t0 = time.perf_counter() - 0.0      # robot frozen at t = 0
        fm, _ = main.read()
        fs, _ = side.read()
        raw_m = fm.copy()
        mp.process(fm, float(k), draw=False)
        extra.mapper.process(fs, float(k), draw=False)
        rig.set_objects(a, cfg["floor_tags"], 0.004)
        extra._update_objects(fs)
        for name, gate in (("main alone", None), ("AND gate", rig.confirm_cells)):
            if name not in gms:
                gms[name] = am.GlobalMap(a, cfg["floor_tags"], res=0.004, floor=mp.floor, K=main.K, D=main.D)
                gms[name].detect_appearance = True
            gms[name].update(raw_m, mp.floor, main.K, main.D, float(k), max_range=4.0, min_px_per_cell=0.35,
                             confirm=gate)
    # where is the phantom on the floor?
    u, v = 455, 467
    ph = mp.floor.pixel_to_floor(u, v, main.K, main.D)[:2]
    res = {}
    for name, gm in gms.items():
        objs = gm.objects_in_view()
        found, _ = classify(objs)
        phantom = any(np.hypot(o["x"] - ph[0], o["y"] - ph[1]) < 0.06 for o in objs)
        res[name] = (found, phantom)
        print(f"  {name}: boxes {sorted(found)}, phantom (only main sees it) "
              f"{'DETECTED' if phantom else 'rejected'}; {rig.gate_state if name != 'main alone' else ''}")
    assert res["main alone"][1], "test setup: the phantom should fool one camera"
    assert not res["AND gate"][1], "phantom passed the AND gate"
    assert "red box" in res["AND gate"][0], res["AND gate"]
    # known (detector / YOLO) objects: OR gate -- pooled from every camera, duplicates merged
    import object_detector as od
    def det(x, y, conf, cam):
        fp = np.array([(x - 0.05, y - 0.05), (x + 0.05, y - 0.05), (x + 0.05, y + 0.05), (x - 0.05, y + 0.05)])
        return {"name": "box", "conf": conf, "x": x, "y": y, "width_m": 0.1, "depth_m": 0.1, "footprint": fp, "cam": cam}
    extra.det_objs = (time.perf_counter(), [det(0.30, 0.20, 0.8, "side"), det(1.00, 0.50, 0.7, "side")])
    main_only = [det(0.70, 0.60, 0.9, "main"), det(1.01, 0.51, 0.6, "main")]
    pooled = od.dedupe(main_only + rig.detector_objects())
    where = sorted((round(o["x"], 2), round(o["y"], 2), o["cam"]) for o in pooled)
    print(f"  known objects (OR gate): main sees 2, side sees 2, one shared -> {len(pooled)}: {where}")
    assert len(pooled) == 3 and any(o["cam"] == "side" and o["x"] == 0.30 for o in pooled)
    # extra camera gone silent: gate opens, main decides alone
    extra.obj_snapshot["t"] -= 10
    assert rig.confirm_cells(gms["AND gate"], 0, 2, 0, 2) is None
    print(f"  extra camera silent -> {rig.gate_state}")
    # tags: OR gate, a tag seen by one camera only is kept
    now = 1.0
    v_main = {"name": "main", "result": {"objects": [{"id": 90, "frame": "world", "x": 0.5, "y": 0.4, "z": 0.05,
                                                      "method": "plane", "distance_m": 1.5, "reproj_px": 0.3}]},
              "stamp": now, "fx": 1000.0, "tag_size": lambda t: 0.05, "floor_valid": True}
    v_side = {"name": "side", "result": {"objects": [{"id": 47, "frame": "world", "x": 0.98, "y": 0.62, "z": 0.08,
                                                      "method": "plane", "distance_m": 1.2, "reproj_px": 0.3}]},
              "stamp": now, "fx": 1000.0, "tag_size": lambda t: 0.05, "floor_valid": True}
    fused = mc.fuse_tags([v_main, v_side], now)
    print(f"  tags (OR gate): 90 seen by {fused[90]['cams']}, 47 seen by {fused[47]['cams']} -> both kept")
    assert set(fused) == {90, 47}


if __name__ == "__main__":
    print("[objects: AND gate across cameras, tags: OR gate]")
    test_and_gate()
    print("  ok")
    print("[no empty-floor reference]")
    gm, objs = run(streaked_scene(), 12)
    found, other = classify(objs)
    print(f"  boxes found: {sorted(found)}; other objects (streaks?): {other}")
    assert {"red box", "green box (on the floor)"} <= found, found
    assert len(other) <= 1, other
    print("  ok")
    print("[empty-floor reference taken with the streaks, boxes appear afterwards]")
    gm, objs = run(streaked_scene(), 16, capture_at=3, empty_until=5)
    found, other = classify(objs)
    print(f"  boxes found: {sorted(found)}; other objects: {other}")
    assert {"red box", "green box (on the floor)"} <= found, found
    assert len(other) <= 1, other
    print("  ok")
    print("all floor-object tests passed")
    sys.stdout.flush()
    os._exit(0)
