#!/usr/bin/env python3
"""
antcolony_pub.py

Sends what the tracker sees to the AntColony coordinator (python -m antcolony.coordinator),
one small JSON datagram per processed frame. Start the tracker with

    python apriltag_tracker_gui.py --antcolony 127.0.0.1          # coordinator on this computer
    python apriltag_tracker_gui.py --antcolony 192.168.8.10       # coordinator on another one
    (or set ANTCOLONY_WORLD=host[:port]; the port defaults to 9100)

What is sent (arena frame, metres, radians):

    {"t": 1763100000.0,
     "tags":    {"10": [x, y, yaw], ...}      every AprilTag that is NOT a floor tag: the robots
     "objects": [{"id": 1003, "label": "object", "x": .., "y": .., "w": .., "l": .., "ang": ..}]
                                              untagged things on the floor (YOLOE and / or the
                                              appearance detector), w = short side, l = long side
     "floor":   {"29": [x, y], ...}           paper tags on the floor (the robots keep off them)
     "corners": [29, 32, 31, 30],             the arena corner tags (the coordinator can take its
                                              arena rectangle from them: --arena-from-tags)
     "camera":  [x, y, z],                    where the camera is: a tall robot seen at a slant
                                              covers the floor behind it; the coordinator ignores
                                              detections in that shadow
     "tag_info": {"10": {"method": "plane", "z": 0.17, "pose_z": 0.171}}}
                                              how each tag was located (plane = at a known height,
                                              precise; pose = from its apparent size, cm worse)
Objects also carry "h" (estimated height, m) and "conf".

Why not the GUI's own "UDP out": that message is for ONE robot following the GUI's A* path, and
json.dumps(..., default=float) raises TypeError as soon as a detected object is in it
(its footprint_poly is a numpy array) - which stops the worker thread.

For the robots' tags to be precise, give each one its height in arena.json, e.g.
    "objects": {"10": {"name": "car A", "height_m": 0.17}, "11": {"name": "car B", "height_m": 0.17}}
(tag face above the floor, metres). Without it a raised tag is located from its apparent size,
which is several cm worse.

Standard library + numpy only.
"""
import json
import math
import os
import socket
import time

import numpy as np

DEFAULT_PORT = 9100
ELONGATED = 2.5          # as object_detector.ELONGATED: only then is the box direction the object's


def parse_target(text):
    """'host' or 'host:port' -> (host, port)."""
    text = text.strip()
    host, _, port = text.partition(":")
    return host or "127.0.0.1", int(port) if port else DEFAULT_PORT


def world_message(r, floor, corner_ids=None, now=None):
    """The tracker's per-frame result `r` (Mapper.process + merged detections) -> coordinator dict,
    or None while there is no floor yet."""
    if floor is None or not getattr(floor, "valid", False):
        return None
    tags, objs, info = {}, [], {}
    for o in r.get("objects", []):
        if o.get("frame") != "world":
            continue
        x, y = o.get("x"), o.get("y")
        if x is None or y is None or not (math.isfinite(x) and math.isfinite(y)):
            continue
        if o.get("method") in ("detect", "appearance"):
            a, b = float(o.get("width_m") or 0.0), float(o.get("depth_m") or 0.0)
            d = {"id": int(o["id"]), "label": str(o.get("name", "object")), "x": float(x), "y": float(y)}
            if o.get("height_m") is not None:
                d["h"] = round(float(o["height_m"]), 3)
            if o.get("conf") is not None:
                d["conf"] = round(float(o["conf"]), 2)
            if a > 0 and b > 0:
                d["w"], d["l"] = min(a, b), max(a, b)
                # a compact object's box is drawn along the camera's line of sight, not along the
                # object, so its angle says nothing; only a clearly long object has a direction
                if d["l"] >= ELONGATED * d["w"] and o.get("yaw_box_deg") is not None:
                    d["ang"] = math.radians(float(o["yaw_box_deg"]))
            objs.append(d)
        else:
            tags[str(int(o["id"]))] = [float(x), float(y), math.radians(float(o.get("yaw_deg") or 0.0))]
            info[str(int(o["id"]))] = {"method": str(o.get("method", "?")).split(" ")[0],
                                       "z": round(float(o.get("z") or 0.0), 3),
                                       "pose_z": None if o.get("pose_z_m") is None else round(float(o["pose_z_m"]), 3)}
    fl = {}
    for tid, c in getattr(floor, "world_corners", {}).items():
        if str(int(tid)) in tags:        # a tag that is tracked as an object (a robot) is not floor paper
            continue
        c = np.asarray(c, np.float64)
        fl[str(int(tid))] = [float(c[:, 0].mean()), float(c[:, 1].mean())]
    msg = {"t": time.time() if now is None else now, "tags": tags, "objects": objs, "floor": fl}
    if corner_ids:
        msg["corners"] = [int(i) for i in corner_ids]
    if info:
        msg["tag_info"] = info
    cam = r.get("camera") or getattr(floor, "camera_position", None)
    if isinstance(cam, dict):
        msg["camera"] = [round(float(cam["x"]), 3), round(float(cam["y"]), 3), round(float(cam["z"]), 3)]
    elif cam is not None and len(cam) >= 3:
        msg["camera"] = [round(float(cam[0]), 3), round(float(cam[1]), 3), round(float(cam[2]), 3)]
    return msg


class WorldPub:
    def __init__(self, target):
        self.addr = parse_target(target)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sent, self.err, self._beat = 0, None, 0.0

    def publish(self, r, floor, corner_ids=None):
        """Never raises: a tracker frame must not die because of the network."""
        try:
            msg = world_message(r, floor, corner_ids)
            if msg is None:
                # no floor yet: say so twice a second, so the other side can tell "tracker is
                # connected but its arena is not set up" from "nothing arrives at all"
                if time.time() - self._beat < 0.5:
                    return
                self._beat = time.time()
                msg = {"t": time.time(), "tags": {}, "objects": [], "floor_valid": False,
                       "n_tags": int(r.get("n_tags", 0)) if isinstance(r, dict) else 0}
            self.sock.sendto(json.dumps(msg, separators=(",", ":")).encode(), self.addr)
            self.sent += 1
        except Exception as e:          # noqa: BLE001 - report once, keep tracking
            if repr(e) != self.err:
                self.err = repr(e)
                print(f"[antcolony_pub] {e!r}")


def from_env():
    t = os.environ.get("ANTCOLONY_WORLD")
    return WorldPub(t) if t else None


if __name__ == "__main__":
    # self-test without a camera:  python antcolony_pub.py [host[:port]]
    import sys
    from types import SimpleNamespace
    sq = lambda x, y: np.array([[x - .05, y - .05, 0], [x + .05, y - .05, 0], [x + .05, y + .05, 0], [x - .05, y + .05, 0]])
    floor = SimpleNamespace(valid=True, world_corners={29: sq(0, 0), 32: sq(1.55, 0), 31: sq(1.55, 1.19), 30: sq(0, 1.19)})
    r = {"camera": {"x": 0.78, "y": -1.2, "z": 1.8}, "objects": [
        {"id": 10, "frame": "world", "method": "plane", "x": 0.40, "y": 0.30, "yaw_deg": 90.0},
        {"id": 11, "frame": "world", "method": "pose", "x": 1.10, "y": 0.30, "yaw_deg": -45.0},
        {"id": 1000, "name": "object", "frame": "world", "method": "detect", "x": 0.70, "y": 0.60,
         "width_m": 0.04, "depth_m": 0.04, "yaw_box_deg": 30.0, "footprint_poly": np.zeros((4, 2))},
        {"id": 1001, "name": "object", "frame": "world", "method": "appearance", "x": 0.95, "y": 0.75,
         "width_m": 0.15, "depth_m": 0.04, "yaw_box_deg": 10.0, "footprint_poly": np.zeros((4, 2))},
        {"id": -1, "name": "x", "frame": "image", "x": float("nan"), "y": float("nan")}]}
    msg = world_message(r, floor, [29, 32, 31, 30])
    print(json.dumps(msg, indent=1))
    if len(sys.argv) > 1:
        pub = WorldPub(sys.argv[1])
        for _ in range(50):
            pub.publish(r, floor, [29, 32, 31, 30])
            time.sleep(0.1)
        print(f"sent {pub.sent} datagrams to {pub.addr}")
