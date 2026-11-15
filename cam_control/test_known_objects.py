#!/usr/bin/env python3
"""
Known parts vs unknown things, and a less noisy map.

    python test_known_objects.py

Only the classes the warehouse model was fine-tuned on are objects; everything else is an
unknown thing (faded, obstacle only, never a pick-up target). Checks:
  1. the class filter: other classes and low-confidence boxes are dropped, the rest marked known;
  2. the size check: a "known" part the size of a chair is dropped, unknown things are untouched;
  3. the known flag survives the floor tracker and reaches the result objects;
  4. the publisher sends known parts as objects and unknown things as obstacles (none on top of a part);
  5. the coordinator never allocates an unknown thing, keeps clear of it, but not of its own target;
  6. the map: flickering noise blobs (glare, shadows, carpet speckle) no longer become obstacles,
     while the real boxes still do.
"""
import os
import sys
from types import SimpleNamespace

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import antcolony_pub as ap  # noqa: E402
import arena_map as am  # noqa: E402
import object_detector as od  # noqa: E402

bad = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        bad.append(msg)


def sq(x, y, s=0.05):
    return np.array([[x - s, y - s], [x + s, y - s], [x + s, y + s], [x - s, y + s]], np.float64)


# 1. class filter ------------------------------------------------------------------------------
dets = [{"name": "cup", "conf": 0.9, "box": (0, 0, 10, 10)},
        {"name": "person", "conf": 0.95, "box": (0, 0, 10, 10)},       # some other model's class
        {"name": "block", "conf": 0.55, "box": (0, 0, 10, 10)}]
keep, dropped = od.filter_known(dets, od.KNOWN_CLASSES, {"block": 0.6})
check([d["name"] for d in keep] == ["cup"] and dropped == 2, "class filter keeps only known classes above their floor")
check(all(d["known"] for d in keep), "kept detections are marked known")
check(set(od.KNOWN_CLASSES) == {"bracket", "knob", "slotted_block", "cup", "holder", "stand", "block",
                                 "contour_block", "white_box"},
      "known classes = the 9 fine-tuned warehouse classes")
check(len(od.plausible([{"known": True, "name": "white_box", "width_m": 0.5, "depth_m": 0.35, "height_m": 0.1},
                        {"known": True, "name": "cup", "width_m": 0.5, "depth_m": 0.35, "height_m": 0.1}])) == 1,
      "the white tray may be bigger than the small parts (per-class size limit)")

# 2. size check --------------------------------------------------------------------------------
objs = [{"known": True, "width_m": 0.06, "depth_m": 0.06, "height_m": 0.08},
        {"known": True, "width_m": 0.9, "depth_m": 0.5, "height_m": 0.1},      # a wall projected onto the floor
        {"known": True, "width_m": 0.1, "depth_m": 0.1, "height_m": 0.9},      # a leg
        {"known": False, "width_m": 0.9, "depth_m": 0.9, "height_m": 0.0}]     # unknown: left alone
check(len(od.plausible(objs)) == 2, "size check drops implausible known parts only")

# 3. tracker keeps the flag --------------------------------------------------------------------
tr = od.FloorTracker(min_hits=2)
for k in range(3):
    tracks = tr.update([{"name": "cup", "conf": 0.9, "known": True, "footprint": sq(0.5, 0.5, 0.03),
                         "height_m": 0.08, "contact": None, "box": None}], float(k))
res = od.as_result_objects(tracks, 2.0)
check(len(res) == 1 and res[0]["known"] is True, "known flag survives tracking")

# 4. publisher split ---------------------------------------------------------------------------
floor = SimpleNamespace(valid=True, world_corners={29: np.column_stack([sq(0, 0), np.zeros(4)])})
r = {"camera": {"x": 0.8, "y": -1.0, "z": 1.8}, "objects": [
    {"id": 1000, "name": "cup", "frame": "world", "method": "detect", "known": True,
     "x": 0.5, "y": 0.5, "width_m": 0.06, "depth_m": 0.06},
    {"id": 1001, "name": "object", "frame": "world", "method": "appearance",          # the cup again
     "x": 0.51, "y": 0.5, "width_m": 0.07, "depth_m": 0.07},
    {"id": 1002, "name": "object", "frame": "world", "method": "appearance",          # a shoe
     "x": 1.2, "y": 0.8, "width_m": 0.10, "depth_m": 0.25},
    {"id": 1003, "name": "object", "frame": "world", "method": "detect", "known": False,   # YOLOE box
     "x": 0.2, "y": 1.5, "width_m": 0.05, "depth_m": 0.05}]}
msg = ap.world_message(r, floor)
check([o["id"] for o in msg["objects"]] == [1000], "only the known part is sent as an object")
check(sorted(tuple(u[:2]) for u in msg["obstacles"]) == [(0.2, 1.5), (1.2, 0.8)],
      "unknown things are sent as obstacles; the duplicate on top of the cup is not")

# 5. coordinator -------------------------------------------------------------------------------
from antcolony.common import load_config  # noqa: E402
from antcolony.coordinator import Coordinator  # noqa: E402

cfg = load_config(os.path.join(os.path.dirname(HERE), "config.json"))
bus = SimpleNamespace(world=lambda: None, telem=lambda n: None, send=lambda n, m: None)
co = Coordinator(cfg, bus, quiet=True)
for k in range(5):
    co.world.update(msg, k * 0.1)
check(len(co.world.objects) == 1 and co.world.objects[0].label == "cup", "coordinator only tracks the known part")
check(len(co.world.unknown) == 2, "coordinator keeps the unknown things as obstacles")
b = co.brains[0]
discs = co.obstacles_for(b)
check(any(abs(x - 1.2) < 1e-6 and abs(y - 0.8) < 1e-6 for x, y, _ in discs), "robots keep clear of unknown things")
co.world.unknown.append((0.5, 0.5, 0.04))                       # an unknown blob on b's own target
b.obj = co.world.objects[0]
check(not any(abs(x - 0.5) < 1e-6 and abs(y - 0.5) < 1e-6 and r_ < 0.1 for x, y, r_ in co.obstacles_for(b)[-3:]),
      "an unknown blob on the robot's own target does not wall it off")
b.obj = None
co.world.update(dict(msg, objects_ok=False, obstacles=[]), 1.0)
check(len(co.world.unknown) == 3, "objects_ok=false leaves the unknown obstacles as they are")

# 6. map noise ---------------------------------------------------------------------------------
import apriltag_floor_mapper as m  # noqa: E402
from apriltag_tracker_gui import SyntheticScene  # noqa: E402


def map_run(extra):
    """Synthetic arena, mapped twice in lockstep from the same frames: once clean, once with
    flickering noise blobs (new random places every frame: glare, shadows, carpet speckle).
    Returns (mean obstacle area m^2 the noise ADDS to the map, real boxes found in the noisy map)."""
    sc = SyntheticScene()
    rng = np.random.default_rng(7)
    mp = m.Mapper(sc.cfg, sc.K, sc.D, m.make_detector(sc.cfg, nthreads=4, decimate=1.0, realtime=False))
    mp.floor = m.FloorModel(sc.cfg["floor_tags"], mp.tag_size, 4.0)
    a = dict(am.ARENA_DEFAULTS, **sc.cfg["arena"], **extra)
    a["map_resolution_m"] = a.get("map_resolution_m") or 0.004
    clean = noisy = None
    added, found = [], 0
    for k in range(30):
        f, _ = sc.read()
        raw = f.copy()
        fn = f.copy()
        for _ in range(25):                    # 1-4 cm blobs, one frame each
            u, v = int(rng.integers(0, f.shape[1])), int(rng.integers(0, f.shape[0]))
            col = tuple(int(c) for c in rng.integers(0, 255, 3))
            cv2.circle(fn, (u, v), int(rng.integers(4, 14)), col, -1)
        mp.process(f, float(k), draw=False)
        if not mp.floor.valid:
            continue
        if clean is None:
            mk = lambda: am.GlobalMap(a, sc.cfg["floor_tags"], res=a["map_resolution_m"], floor=mp.floor, K=sc.K, D=sc.D)
            clean, noisy = mk(), mk()
            clean.detect_appearance = noisy.detect_appearance = True
        for gm, img in ((clean, raw), (noisy, fn)):
            gm.update(img, mp.floor, sc.K, sc.D, float(k), max_range=4.0, min_px_per_cell=0.35)
        objs = noisy.objects_in_view()
        if k < 10:
            continue
        extra_cells = noisy.obstacles() & ~clean.obstacles() & noisy.quad_mask
        added.append(float(extra_cells.sum()) * noisy.res ** 2)
        found = sum(any(np.hypot(o["x"] - bx, o["y"] - by) < 0.1 for o in objs) for bx, by in BOXES.values())
    return float(np.mean(added)), found


BOXES = {"red box": (0.57, 0.40), "green box": (1.41, 0.24), "crate": (0.98, 0.62)}
old_noise, old_found = map_run({"unknown_min_area_m2": 0.0, "unknown_evidence": 0.0})
new_noise, new_found = map_run({})
print(f"     map noise (obstacle area added by noise, mean per frame): before {old_noise * 1e4:.1f} cm^2, "
      f"now {new_noise * 1e4:.1f} cm^2;  real boxes found: before {old_found}, now {new_found}")
check(new_noise < 0.25 * old_noise or new_noise == 0.0, "flickering noise leaves at most a quarter of the false obstacles")
check(new_found >= old_found, "the real boxes are still mapped")

print("all known-object tests passed" if not bad else f"{len(bad)} FAILED")
sys.exit(1 if bad else 0)
