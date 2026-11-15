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
     "objects": [{"id": 1003, "label": "cup", "x": .., "y": .., "w": .., "l": .., "ang": ..}]
                                              KNOWN parts only (the fine-tuned warehouse classes,
                                              object_detector.KNOWN_CLASSES): what the robots may
                                              pick up. w = short side, l = long side
     "obstacles": [[x, y, r], ...]            UNKNOWN things (appearance blobs, prompt-free boxes):
                                              never targets, only discs to keep clear of
     "floor":   {"29": [x, y], ...}           paper tags on the floor (the robots keep off them)
     "corners": [29, 32, 31, 30],             the arena corner tags (the coordinator can take its
                                              arena rectangle from them: --arena-from-tags)
     "camera":  [x, y, z],                    where the camera is: a tall robot seen at a slant
                                              covers the floor behind it; the coordinator ignores
                                              detections in that shadow
     "tag_info": {"10": {"method": "plane", "z": 0.17, "pose_z": 0.171}}}
                                              how each tag was located (plane = at a known height,
                                              precise; pose = from its apparent size, cm worse);
                                              with extra cameras also "cams": ["main", "go3s"]
                                              and "sigma" (m, fused 1-sigma)
     "cameras": {"main": [x, y, z], "go3s": [x, y, z]}   with extra cameras (multi_camera.py)
     "objects_ok": false                      only when the objects were NOT looked for in this
                                              message (main camera silent, no floor yet): the
                                              coordinator then keeps the objects it already has
Objects also carry "h" (estimated height, m) and "conf".
A tag without a finite position AND heading is left out (the car is "not seen" for that frame,
and the coordinator stops it after control.pose_timeout) - never sent with a made-up heading.
Nothing non-finite is ever sent (NaN is not JSON and must not reach a car's motors).

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


def _num(v):
    """float(v) if v is a finite number, else None (None, NaN, inf, junk)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _xyz(cam):
    """Camera position (dict x/y/z or a sequence) -> [x, y, z] rounded, or None if not all finite."""
    try:
        v = [cam["x"], cam["y"], cam["z"]] if isinstance(cam, dict) else list(cam)[:3]
    except (TypeError, KeyError):
        return None
    v = [_num(c) for c in v]
    return [round(c, 3) for c in v] if len(v) == 3 and None not in v else None


def _tag_entry(o):
    """Robot / bin tag -> (id string, [x, y, heading rad], info) or None if it cannot be steered by."""
    x, y, yaw = _num(o.get("x")), _num(o.get("y")), _num(o.get("yaw_deg"))
    if x is None or y is None or yaw is None:
        # no heading = no way to steer the car: better "not seen" (the coordinator stops it after
        # control.pose_timeout) than a made-up heading of 0 that drives it the wrong way
        return None
    tid = str(int(o["id"]))
    info = {"method": str(o.get("method", "?")).split(" ")[0], "z": round(_num(o.get("z")) or 0.0, 3),
            "pose_z": None if _num(o.get("pose_z_m")) is None else round(_num(o["pose_z_m"]), 3)}
    if o.get("cams"):                    # multi-camera: which cameras the pose is fused from
        info["cams"] = [str(c) for c in o["cams"]]
        if _num(o.get("sigma_m")) is not None:
            info["sigma"] = _num(o["sigma_m"])
    return tid, [x, y, math.radians(yaw)], info


def _object_entry(o):
    """Untagged floor object -> coordinator dict, or None if its position is unusable."""
    x, y = _num(o.get("x")), _num(o.get("y"))
    if x is None or y is None:
        return None
    a, b = _num(o.get("width_m")) or 0.0, _num(o.get("depth_m")) or 0.0
    d = {"id": int(o["id"]), "label": str(o.get("name", "object")), "x": x, "y": y}
    if _num(o.get("height_m")) is not None:
        d["h"] = round(_num(o["height_m"]), 3)
    if _num(o.get("conf")) is not None:
        d["conf"] = round(_num(o["conf"]), 2)
    if a > 0 and b > 0:
        d["w"], d["l"] = min(a, b), max(a, b)
        # a compact object's box is drawn along the camera's line of sight, not along the
        # object, so its angle says nothing; only a clearly long object has a direction
        if d["l"] >= ELONGATED * d["w"] and _num(o.get("yaw_box_deg")) is not None:
            d["ang"] = math.radians(_num(o["yaw_box_deg"]))
    return d


def world_message(r, floor, corner_ids=None, now=None, skipped=None):
    """The tracker's per-frame result `r` (Mapper.process + merged detections) -> coordinator dict,
    or None while there is no floor yet.

    One bad entry never costs the whole frame: a malformed tag or object is left out (and counted
    in `skipped`, a list that gets the reasons, if given) and every other robot pose still goes out."""
    if floor is None or not getattr(floor, "valid", False):
        return None
    tags, objs, unknown, info = {}, [], [], {}
    for o in r.get("objects", []):
        if not isinstance(o, dict) or o.get("frame") != "world":
            continue
        try:
            if o.get("method") in ("detect", "appearance"):
                d = _object_entry(o)
                if d is None:
                    continue
                # only the fine-tuned parts are things to pick up; anything else is only kept clear of
                if o.get("known", o.get("method") == "detect"):
                    objs.append(d)
                else:
                    unknown.append([d["x"], d["y"], round(0.5 * max(d.get("l") or 0.04, 0.04), 3)])
            else:
                e = _tag_entry(o)
                if e is not None:
                    tags[e[0]], info[e[0]] = e[1], e[2]
                elif skipped is not None:
                    skipped.append(f"tag {o.get('id')}: no finite x/y/heading")
        except (TypeError, ValueError, KeyError) as ex:
            if skipped is not None:
                skipped.append(f"{o.get('method')} {o.get('id')!r}: {ex!r}")
    # an unknown blob on top of a known part is that part seen by the appearance detector: as an
    # obstacle it would wall off the very thing a robot is sent to pick up
    unknown = [u for u in unknown
               if all(math.hypot(u[0] - d["x"], u[1] - d["y"]) > u[2] + 0.5 * (d.get("l") or 0.04) + 0.03
                      for d in objs)]
    fl = {}
    for tid, c in getattr(floor, "world_corners", {}).items():
        if str(int(tid)) in tags:        # a tag that is tracked as an object (a robot) is not floor paper
            continue
        c = np.asarray(c, np.float64)
        p = [_num(c[:, 0].mean()), _num(c[:, 1].mean())]
        if None not in p:
            fl[str(int(tid))] = p
    msg = {"t": time.time() if now is None else now, "tags": tags, "objects": objs, "floor": fl,
           "obstacles": unknown}
    if r.get("main_camera_silent"):
        # the extra cameras alone: robot poses are good, but the untagged objects come from the main
        # camera's detector, so "objects: []" here means "not looked", not "all gone". The
        # coordinator keeps its objects as they are instead of timing them out under the cars.
        msg["objects_ok"] = False
    if corner_ids:
        msg["corners"] = [int(i) for i in corner_ids]
    if info:
        msg["tag_info"] = info
    cam = r.get("camera")
    cam = _xyz(cam) if cam is not None else None
    if cam is None and getattr(floor, "camera_position", None) is not None:
        cam = _xyz(floor.camera_position)
    if cam is not None:
        msg["camera"] = cam
    if r.get("cameras"):
        # every camera with a floor pose ("camera" above stays the one the untagged objects come
        # from: the coordinator masks robot shadows as seen from it)
        cams = {str(k): _xyz(xyz) for k, xyz in r["cameras"].items()}
        msg["cameras"] = {k: v for k, v in cams.items() if v is not None}
    return msg


class WorldPub:
    def __init__(self, target):
        self.addr = parse_target(target)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sent, self.err, self._beat = 0, None, 0.0
        self.skipped, self._skip_said = 0, set()

    def publish(self, r, floor, corner_ids=None):
        """Never raises: a tracker frame must not die because of the network."""
        try:
            skipped = []
            msg = world_message(r, floor, corner_ids, skipped=skipped)
            if skipped:
                self.skipped += len(skipped)
                for s in skipped:
                    key = s.split(":", 1)[0]
                    if key not in self._skip_said:      # each kind of problem once, not every frame
                        self._skip_said.add(key)
                        print(f"[antcolony_pub] left out {s}")
            if msg is None:
                # no floor yet: say so twice a second, so the other side can tell "tracker is
                # connected but its arena is not set up" from "nothing arrives at all"
                if time.time() - self._beat < 0.5:
                    return
                self._beat = time.time()
                msg = {"t": time.time(), "tags": {}, "objects": [], "objects_ok": False, "floor_valid": False,
                       "n_tags": int(r.get("n_tags", 0)) if isinstance(r, dict) else 0}
            # allow_nan=False: NaN is not JSON, and a NaN pose would reach the cars' motors. world_message
            # already leaves non-finite values out; this is the last guard (the frame is reported, not sent)
            self.sock.sendto(json.dumps(msg, separators=(",", ":"), allow_nan=False).encode(), self.addr)
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
