#!/usr/bin/env python3
"""AprilTags seen by the CAR's own camera: landmarks such as a tag on the bin.

For each tag (family tag36h11, the same printed sheet as the overhead tracker) it gives the
distance and the direction from the camera, so a car can turn to the bin and drive up to it
without the overhead tracker. OpenCV only (cv2.aruco; the cars have OpenCV 4.10).

    python3 landmarks.py watch                    print every tag in view, twice a second (nothing moves)
    python3 landmarks.py watch --ids 60 --size 60=0.10
    python3 landmarks.py photo look.jpg           the same for a saved photo
    python3 landmarks.py calib-fx --id 60 --size 0.10 --distance 0.50
         one-time: hold the tag square to the camera, 50 cm in front of the lens; saves the
         focal length to camera.json (until then it is guessed from a 66 deg field of view)

Tested on synthetic photos (640x480, known poses): distance within 2 %, direction within 0.2 deg
from 0.4 to 2.5 m. A 10 cm tag is about 24 px wide at 2 m and was missed once at that size: on the
bin use a bigger tag (15-20 cm) so that it is found from across the arena.

Directions are relative to where the camera points: bearing + = to the left. The MasterPi
camera sits on the arm, so use it with the arm in a pose that looks ahead (vlm_grab's drive
pose), and add the arm's pan angle if it is panned. "dist" is along the floor (the tag's height
above the camera does not count), "range" the straight line.

Import:  from landmarks import Landmarks;  lm = Landmarks();  lm.find(img, ids={60})
"""
import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAMERA_FILE = os.path.join(HERE, "camera.json")
DEFAULT_SIZE = 0.10           # m, black square edge of a tag (from the tag sheet: set per id with --size ID=M)
GUESS_HFOV_DEG = 66.0         # until calib-fx has been run


def load_camera(path=CAMERA_FILE, w=640, h=480):
    c = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            c = json.load(f)
    fx = c.get("fx") or (w / 2) / math.tan(math.radians(GUESS_HFOV_DEG / 2))
    cam = {"fx": float(fx), "fy": float(c.get("fy") or fx), "cx": float(c.get("cx") or w / 2),
           "cy": float(c.get("cy") or h / 2), "w": int(c.get("w") or w), "h": int(c.get("h") or h),
           "calibrated": "fx" in c}
    return cam


class Landmarks:
    def __init__(self, sizes=None, camera=None):
        self.sizes = {int(k): float(v) for k, v in (sizes or {}).items()}
        self.cam = camera or load_camera()
        d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        p = cv2.aruco.DetectorParameters()
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        self.det = cv2.aruco.ArucoDetector(d, p)

    def size_of(self, tid):
        return self.sizes.get(int(tid), DEFAULT_SIZE)

    def K(self, img):
        h, w = img.shape[:2]
        c = self.cam
        sx, sy = w / c["w"], h / c["h"]                 # a photo of another size: scale the intrinsics
        return np.array([[c["fx"] * sx, 0, c["cx"] * sx], [0, c["fy"] * sy, c["cy"] * sy], [0, 0, 1]], np.float64)

    def find(self, img, ids=None):
        """-> [{"id", "dist", "bearing", "range", "turn", "px", "center", "corners"}], nearest first.
        dist / range in m, bearing / turn in rad (bearing + = left of the camera axis; turn = how far
        the tag is turned away from facing the camera - only a rough hint: below ~40 px a small tag's
        pose has two near-equal solutions and the turn can come out wrong; dist and bearing do not)."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        corners, tids, _ = self.det.detectMarkers(gray)
        if tids is None:
            return []
        K, out = self.K(img), []
        for c, tid in zip(corners, tids.reshape(-1)):
            tid = int(tid)
            if ids is not None and tid not in ids:
                continue
            c = c.reshape(4, 2).astype(np.float64)
            s = self.size_of(tid)
            obj = np.array([[-s / 2, s / 2, 0], [s / 2, s / 2, 0], [s / 2, -s / 2, 0], [-s / 2, -s / 2, 0]], np.float64)
            ok, rvec, tvec = cv2.solvePnP(obj, c, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                continue
            x, y, z = tvec.reshape(3)                    # camera frame: x right, y down, z forward
            R, _ = cv2.Rodrigues(rvec)
            n = R[:, 2]                                  # tag normal in the camera frame
            bearing = math.atan2(-x, z)
            facing = math.atan2(n[0], -n[2])             # normal vs the camera axis
            turn = math.atan2(math.sin(facing - bearing), math.cos(facing - bearing))
            out.append({"id": tid, "dist": float(math.hypot(x, z)), "bearing": float(bearing),
                        "range": float(np.linalg.norm(tvec)), "turn": float(turn),
                        "px": float(np.mean([np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)])),
                        "center": c.mean(0).tolist(), "corners": c.tolist()})
        out.sort(key=lambda t: t["dist"])
        return out


def snap(cap):
    img = None
    for _ in range(3):                                   # drop frames waiting in the driver queue
        ok, f = cap.read()
        if ok:
            img = f
    return img


def open_camera():
    import glob
    devs = sorted(glob.glob("/dev/v4l/by-id/*-video-index0"))
    cap = cv2.VideoCapture(os.path.realpath(devs[0]) if devs else 0, cv2.CAP_V4L2 if devs else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    for _ in range(20):                                  # exposure / white balance
        cap.read()
    return cap


def show(tags, cam):
    if not tags:
        return "no tag in view"
    return "  ".join(f"tag {t['id']}: {t['dist'] * 100:.0f} cm, {math.degrees(t['bearing']):+.0f} deg "
                     f"({'left' if t['bearing'] > 0 else 'right'}), {t['px']:.0f} px" for t in tags) \
        + ("" if cam["calibrated"] else "   [focal length guessed: run calib-fx]")


def parse_sizes(items):
    out = {}
    for it in items or []:
        k, v = it.split("=")
        out[int(k)] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("watch", help="live, from the car's camera")
    w.add_argument("--secs", type=float, default=20)
    p = sub.add_parser("photo", help="one saved photo")
    p.add_argument("image")
    c = sub.add_parser("calib-fx", help="measure the focal length with a tag at a known distance")
    c.add_argument("--id", type=int, required=True)
    c.add_argument("--distance", type=float, required=True, help="m, lens to tag, tag facing the camera")
    for s in (w, p, c):
        s.add_argument("--ids", help="only these tag ids, comma-separated")
        s.add_argument("--size", nargs="*", help="tag edge length in m: ID=M ... (default 0.10), or one number for --id")
    a = ap.parse_args()
    sizes = {}
    for it in a.size or []:
        if "=" in it:
            sizes.update(parse_sizes([it]))
        elif a.cmd == "calib-fx":
            sizes[a.id] = float(it)
    ids = {int(i) for i in a.ids.split(",")} if a.ids else None
    lm = Landmarks(sizes)
    if a.cmd == "photo":
        img = cv2.imread(a.image)
        if img is None:
            sys.exit(f"cannot read {a.image}")
        print(show(lm.find(img, ids), lm.cam))
        return
    cap = open_camera()
    try:
        if a.cmd == "watch":
            t_end = time.time() + a.secs
            while time.time() < t_end:
                img = snap(cap)
                if img is not None:
                    print(show(lm.find(img, ids), lm.cam), flush=True)
                time.sleep(0.5)
            return
        # calib-fx: f = apparent edge (px) * distance / real edge, over 10 photos
        fs = []
        for _ in range(10):
            img = snap(cap)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners, tids, _ = lm.det.detectMarkers(gray)
            for cc, tid in zip(corners or [], (tids if tids is not None else np.empty((0, 1))).reshape(-1)):
                if int(tid) == a.id:
                    cc = cc.reshape(4, 2)
                    px = float(np.mean([np.linalg.norm(cc[i] - cc[(i + 1) % 4]) for i in range(4)]))
                    fs.append(px * a.distance / lm.size_of(a.id))
            time.sleep(0.1)
        if len(fs) < 5:
            sys.exit(f"tag {a.id} seen in only {len(fs)} of 10 photos - closer, better light, facing the camera?")
        fx = float(np.median(fs))
        h, wdt = img.shape[:2]
        with open(CAMERA_FILE, "w", encoding="utf-8") as f:
            json.dump({"fx": round(fx, 1), "fy": round(fx, 1), "cx": wdt / 2, "cy": h / 2, "w": wdt, "h": h,
                       "_how": f"calib-fx: tag {a.id} ({lm.size_of(a.id)} m) at {a.distance} m, {len(fs)} photos"}, f, indent=1)
        print(f"fx = {fx:.0f} px (field of view {math.degrees(2 * math.atan(wdt / 2 / fx)):.0f} deg) -> saved {CAMERA_FILE}")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
