#!/usr/bin/env python3
"""
apriltag_floor_mapper.py

Real time AprilTag detection and floor mapping for a Sony A7 IV (ILCE-7M4) in
USB streaming mode, or any other UVC / HDMI capture device.

How it works
  1. A capture thread always holds the newest camera frame, so processing never
     works on stale, buffered frames.
  2. AprilTags are detected every frame (pupil_apriltags if installed, otherwise
     OpenCV's built in AprilTag dictionaries).
  3. "Floor tags" lie flat on the floor at known positions. All of their corners
     go into one PnP solve, which gives the camera pose in a floor based world
     frame: x, y on the floor, z pointing up, z = 0 is the floor.
  4. Every other tag is an "object tag". Its pose is solved on its own and moved
     into the world frame, which gives x, y, z (z = height above the floor).
  5. Any pixel can be projected onto the floor plane (hover or click the image).

Subcommands
  list       list cameras (add --preview for thumbnails of every index)
  diagnose   explain why the A7 IV is not found (USB-C or HDMI)
  calibrate  chessboard calibration, writes intrinsics to an .npz
  config     write an example floor/object config JSON
  survey     measure the floor plane and floor tag positions (writes floor_survey.json)
  run        live detection and mapping

Markers: AprilTag 36h11 and ArUco DICT_4X4_50 are detected together in every frame and share
one ID space (global id = family offset + printed id):
  AprilTag 0-86  -> 0-86        ArUco 0-14 -> 87-101
Use global ids everywhere (floor_tags, objects, --ids). Either family can be a floor tag or an
object. Sizes, families and offsets come from apriltag_sheet_a4.json next to this script.

Connection: USB-C to C by default (camera: Menu > Setup > USB > USB Connection Mode =
USB Streaming). For an HDMI capture card add --connection hdmi.

Quick start
  pip install pygrabber        (Windows: lets the script find the Sony by name)
  python apriltag_floor_mapper.py list
  python apriltag_floor_mapper.py calibrate --board 9x6 --square 0.025
  python apriltag_floor_mapper.py survey --calib calib.npz --ids 27,28,0,1,2
  python apriltag_floor_mapper.py run --calib calib.npz

Keys in run mode
  q / ESC  quit            l  lock / unlock floor pose (for a fixed tripod)
  m        toggle map      s  save snapshot PNG
  c        clear pins      left click: drop a floor pin, right click: clear pins
"""

import argparse
from collections import deque
import glob
import json
import math
import os
import re
import socket
import sys
import threading
import time

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

ARUCO_APRILTAG_DICTS = {
    "tag16h5": "DICT_APRILTAG_16h5",
    "tag25h9": "DICT_APRILTAG_25h9",
    "tag36h10": "DICT_APRILTAG_36h10",
    "tag36h11": "DICT_APRILTAG_36h11",
}

EXAMPLE_CONFIG = {
    "family": "tag36h11",
    "tag_size": 0.10,
    "tag_sizes": {},
    "floor_tags": {
        "0": {"x": 0.0, "y": 0.0, "yaw_deg": 0.0},
        "1": {"x": 1.0, "y": 0.0, "yaw_deg": 0.0},
        "2": {"x": 0.0, "y": 1.0, "yaw_deg": 0.0},
        "3": {"x": 1.0, "y": 1.0, "yaw_deg": 0.0},
    },
    "objects": {
        "10": {"name": "cube", "offset": [0.0, 0.0, -0.025], "orientation": "flat"},
    },
    "_notes": [
        "All units are meters. Sizes come from apriltag_sheet_a4.json automatically (black square edge).",
        "tag_size is only the fallback for IDs not on the sheet; tag_sizes overrides single IDs.",
        "Easiest: lay tags on the floor and run `survey`; it measures floor_tags for you.",
        "floor_tags: printed face up on the floor. x, y = tag center. yaw_deg rotates the tag about +z.",
        "With yaw_deg = 0 the top edge of the printed tag faces world +y and its right edge faces +x.",
        "objects: optional names and an offset (in the tag's own frame) from tag center to object center.",
        "objects.orientation: flat | vertical | auto (default) | any. Resolves pose flips on small tags.",
        "Any tag id not listed under floor_tags is treated as an object.",
    ],
}

DEFAULT_CONFIG = {
    "family": "tag36h11",
    "tag_size": 0.10,
    "tag_sizes": {},
    "floor_tags": {},     # set by the arena setup (4 corner tags) or `survey`
    "objects": {},
}


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SHEET = os.path.join(SCRIPT_DIR, "apriltag_sheet_a4.json")
DEFAULT_SURVEY = "floor_survey.json"


def load_tag_sheet(path=DEFAULT_SHEET, with_families=False):
    """Tag sizes from a print sheet JSON: {global id: black square edge in meters}, family
    (and the marker families with their ID offsets when with_families=True)."""
    if not path or not os.path.exists(path):
        return ({}, None, None) if with_families else ({}, None)
    with open(path) as f:
        sheet = json.load(f)
    default_scale = float(sheet.get("print_scale", 1.0))
    sheets = sheet.get("sheets", {})

    def scale_of(tag):
        return float(sheets.get(tag.get("sheet"), {}).get("print_scale", default_scale))
    sizes = {int(k): float(v["black_mm"]) * scale_of(v) / 1000.0 for k, v in sheet["tags"].items()}
    if with_families:
        return sizes, sheet.get("family"), sheet.get("families")
    return sizes, sheet.get("family")


def corner_tag_ids(tag_sizes, n=4):
    """The arena pivots: the n largest printed tags on the sheet (29-32, 112 mm, on the A4
    sheets). They are the ONLY tags that define the floor plane; every other tag is an object."""
    if len(tag_sizes) < n:
        return None
    return sorted(sorted(tag_sizes, key=lambda t: (-tag_sizes[t], t))[:n])


def default_config_path():
    """Use the saved arena (arena.json) or the last floor survey in the working directory."""
    for name in ("arena.json", DEFAULT_SURVEY):
        if os.path.exists(name):
            return name
    return None


def load_config(path):
    """Defaults <- tag sheet sizes <- user config. Config may set "sheet" to another sheet JSON."""
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    user = {}
    if path:
        with open(path) as f:
            user = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    sheet_path = user.get("sheet", DEFAULT_SHEET)
    if sheet_path and not os.path.isabs(sheet_path) and not os.path.exists(sheet_path):
        sheet_path = os.path.join(SCRIPT_DIR, sheet_path)
    sheet_sizes, sheet_family, sheet_families = load_tag_sheet(sheet_path, with_families=True)
    if sheet_family:
        cfg["family"] = sheet_family
    if sheet_families:
        cfg["families"] = sheet_families
    cfg.update({k: v for k, v in user.items() if k != "tag_sizes"})
    sizes = dict(sheet_sizes)
    sizes.update({int(k): float(v) for k, v in user.get("tag_sizes", {}).items()})
    cfg["tag_sizes"] = sizes
    cfg["sheet_loaded"] = bool(sheet_sizes)
    cfg["families"] = normalize_families(cfg.get("families"), cfg.get("family", "tag36h11"))
    cfg["floor_tags"] = {int(k): v for k, v in cfg.get("floor_tags", {}).items()}
    cfg["objects"] = {int(k): v for k, v in cfg.get("objects", {}).items()}
    return cfg


# --------------------------------------------------------------------------
# Camera capture
# --------------------------------------------------------------------------

BACKENDS = {"dshow": "CAP_DSHOW", "msmf": "CAP_MSMF", "v4l2": "CAP_V4L2", "avfoundation": "CAP_AVFOUNDATION"}


def backend_id(name):
    """'auto' / None -> platform default, else 'dshow', 'msmf', ..."""
    if not name or name == "auto":
        return default_backend()
    return getattr(cv2, BACKENDS[name])


def default_backend():
    if sys.platform.startswith("win"):
        return cv2.CAP_DSHOW
    if sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_V4L2


class CameraStream:
    """Reads frames on a background thread and keeps only the newest one."""

    def __init__(self, index, width=1920, height=1080, fps=60, backend=None):
        backend = default_backend() if backend is None else backend
        # No fallback to another backend: on Windows, DirectShow and Media Foundation number
        # cameras differently, so "index 1" in one can be the laptop webcam in the other.
        self.cap = cv2.VideoCapture(index, backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {index}. Try `list` to find the right index.")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Camera opened but returned no frames. USB: is the A7 IV in USB Streaming "
                               "mode? HDMI: is the camera on and outputting HDMI to the capture card?")
        self.height, self.width = frame.shape[:2]

        self.cond = threading.Condition()
        self.frame, self.frame_id, self.stamp = frame, 1, time.perf_counter()
        self.grab_fps = 0.0
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        n, t_last = 0, time.perf_counter()
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            now = time.perf_counter()
            with self.cond:
                self.frame, self.stamp = frame, now
                self.frame_id += 1
                self.cond.notify_all()
            n += 1
            if now - t_last >= 1.0:
                self.grab_fps, n, t_last = n / (now - t_last), 0, now

    def latest(self, after_id=0, timeout=1.0):
        """Block until a frame newer than after_id exists, then return a copy of it."""
        with self.cond:
            self.cond.wait_for(lambda: self.frame_id > after_id or not self.running, timeout=timeout)
            if self.frame_id <= after_id:
                return None, after_id, 0.0
            return self.frame.copy(), self.frame_id, self.stamp

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


# How the A7 IV shows up on the computer:
#   USB-C streaming     named after the model, e.g. "ILCE-7M4"
#   HDMI capture card   named after the CARD, e.g. "USB Video", "Cam Link 4K", "HDMI Capture"
SONY_PATTERNS = ("ilce", "sony", "7m4", "imaging edge")
CAPTURE_PATTERNS = ("capture", "hdmi", "cam link", "camlink", "elgato", "avermedia", "live gamer",
                    "ugreen", "magewell", "ezcap", "usb video", "usb3 video", "usb3.0 video",
                    "usb 3.0 video", "video grabber", "uvc video")
# Never auto-picked: built-in webcams, IR cameras, and virtual cameras.
EXCLUDE_PATTERNS = ("integrated", "webcam", "web cam", "facetime", "ir camera", "infrared", "hello",
                    "hp ", "5mp", "hd camera", "hd user", "fhd camera", "laptop", "lenovo", "dell",
                    "asus", "acer", "surface camera", "chicony", "realtek", "bison", "azurewave",
                    "virtual", "obs", "droidcam", "broadcast", "snap camera", "iriun", "camo",
                    "front", "rear", "user facing", "world facing")


def device_formats(name):
    """[(width, height, max fps)] a DirectShow camera offers (Windows + pygrabber), largest
    first, or None. A UVC camera only delivers these: requesting anything else falls back."""
    try:
        from pygrabber.dshow_graph import FilterGraph
        g = FilterGraph()
        names = g.get_input_devices()
        idx = next(i for i, n in enumerate(names) if n == name or name.lower() in n.lower())
        g.add_video_input_device(idx)
        fm = g.get_input_device().get_formats()
        return sorted({(int(f["width"]), int(f["height"]), int(round(f.get("max_framerate", 0)))) for f in fm},
                      key=lambda t: (-t[0] * t[1], -t[2]))
    except Exception:
        return None


def enumerate_cameras():
    """Return [(index, name)] in the same index order OpenCV's default backend uses,
    or None if names are not available on this platform."""
    if sys.platform.startswith("win"):
        try:
            from pygrabber.dshow_graph import FilterGraph
        except ImportError:
            return None
        # DirectShow enumeration order == cv2.CAP_DSHOW index order (the backend we open with).
        return list(enumerate(FilterGraph().get_input_devices()))
    if sys.platform.startswith("linux"):
        out = []
        for p in sorted(glob.glob("/sys/class/video4linux/video*"),
                        key=lambda q: int(re.sub(r"\D", "", os.path.basename(q)) or 0)):
            try:
                with open(os.path.join(p, "name")) as f:
                    out.append((int(re.sub(r"\D", "", os.path.basename(p))), f.read().strip()))
            except OSError:
                pass
        return out
    return None  # macOS: OpenCV exposes no device names


def classify_camera(name):
    n = name.lower()
    if any(p in n for p in SONY_PATTERNS):
        return "sony"
    if any(p in n for p in EXCLUDE_PATTERNS):
        return "skip"
    if any(p in n for p in CAPTURE_PATTERNS):
        return "capture"
    return "other"


CONNECTIONS = ("auto", "usb", "hdmi")


def auto_pick_camera(devices, connection="auto"):
    """connection: "usb"  = A7 IV over USB-C in USB Streaming mode (device named ILCE-7M4 / Sony)
                   "hdmi" = A7 IV HDMI out into a capture card (device named after the card)
                   "auto" = usb first, then hdmi
    Returns (index, name, reason) or None when nothing or more than one plausible device."""
    by_kind = {}
    for idx, name in devices:
        by_kind.setdefault(classify_camera(name), []).append((idx, name))
    if connection in ("auto", "usb") and by_kind.get("sony"):
        return (*by_kind["sony"][0], "Sony over USB-C")
    if connection in ("auto", "hdmi"):
        if len(by_kind.get("capture", [])) == 1:
            return (*by_kind["capture"][0], "HDMI capture card")
        if not by_kind.get("capture") and len(by_kind.get("other", [])) == 1:
            return (*by_kind["other"][0], "only non-webcam camera")
    return None


def resolve_camera(spec, connection="auto"):
    """spec: an index ("1"), a name substring ("Cam Link"), or "auto"/"sony".
    Returns (index, name). Raises with a device list instead of silently using the webcam."""
    spec = str(spec).strip()
    if spec.isdigit():
        return int(spec), None
    devices = enumerate_cameras()
    if devices is None:
        hint = ("pip install pygrabber" if sys.platform.startswith("win")
                else "camera names are not available on this OS")
        raise RuntimeError(f"Cannot look up cameras by name ({hint}). Pass an index, e.g. --camera 1. "
                           "Run `list --preview` to see which index shows the Sony.")
    found = "\n".join(f"  {i}: {n}   [{classify_camera(n)}]" for i, n in devices) or "  (none)"
    if spec.lower() in ("auto", "sony"):
        pick = auto_pick_camera(devices, connection)
        if pick:
            return pick[0], pick[1]
        raise RuntimeError(
            f"Could not find the A7 IV ({connection} connection). Windows camera list:\n{found}\n"
            "Run `python apriltag_floor_mapper.py diagnose` to see why, or "
            "`list --preview` and pass the right index with --camera <index>.")
    for idx, name in devices:
        if spec.lower() in name.lower():
            return idx, name
    raise RuntimeError(f"No camera matching '{spec}'. Cameras found:\n{found}")


# --------------------------------------------------------------------------
# Probing and diagnosis (for "it only sees my laptop camera")
# --------------------------------------------------------------------------

def _quiet_opencv():
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception:
        pass


def probe_cameras(max_index=8, backend="auto", width=1920, height=1080, tries=15):
    """Open every index, grab one frame. Returns [{index, width, height, frame, black}].
    When device names are known (DirectShow), only the indices that exist are opened."""
    _quiet_opencv()
    names = enumerate_cameras() if backend in ("auto", "dshow") else None
    if names:
        max_index = len(names)
    out = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, backend_id(backend))
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        frame = None
        for _ in range(tries):          # capture cards often need a few frames to start
            ok, f = cap.read()
            if ok and f is not None and f.size:
                frame = f
                if f.mean() > 8:
                    break
        info = {"index": i, "frame": frame, "width": 0, "height": 0, "black": True}
        if frame is not None:
            info.update(height=frame.shape[0], width=frame.shape[1], black=bool(frame.mean() < 8))
        out.append(info)
        cap.release()
    return out


def contact_sheet(probes, names=None, path="camera_probe.png", tile=(400, 225)):
    """One image with a thumbnail per camera index, labelled, so you can see which is the Sony."""
    names = dict(names or [])
    tiles = []
    for p in probes:
        img = np.zeros((tile[1], tile[0], 3), np.uint8)
        if p["frame"] is not None:
            img = cv2.resize(p["frame"], tile, interpolation=cv2.INTER_AREA)
        label = f"index {p['index']}: {names.get(p['index'], '?')}"
        info = f"{p['width']}x{p['height']}" + ("  NO SIGNAL / BLACK" if p["black"] else "")
        for k, (txt, y) in enumerate(((label, 22), (info, tile[1] - 10))):
            _put(img, txt[:48], (6, y), (0, 255, 255) if k == 0 else (255, 255, 255), 0.55)
        tiles.append(img)
    if not tiles:
        return None
    cols = min(3, len(tiles))
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)]
    cv2.imwrite(path, np.vstack(rows))
    return os.path.abspath(path)


def windows_pnp_devices():
    """Present camera / imaging / portable devices from Windows Plug and Play, or None."""
    if not sys.platform.startswith("win"):
        return None
    import subprocess
    ps = ("Get-PnpDevice -PresentOnly | Where-Object { $_.Class -in "
          "@('Camera','Image','WPD','Media','MEDIA') -or $_.FriendlyName -match "
          "'ILCE|Sony|Capture|HDMI|Video|Cam Link|Elgato|Unknown USB' -or "
          "($_.Class -eq 'USB' -and $_.Status -ne 'OK') } | "
          "Select-Object FriendlyName, Class, Status | ConvertTo-Json -Compress")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True,
                             text=True, timeout=15).stdout.strip()
        if not out:
            return []
        data = json.loads(out)
        return data if isinstance(data, list) else [data]
    except Exception:
        return None


def diagnose_cameras(backend="auto", max_index=8, sheet_path="camera_probe.png"):
    """Plain-language report of what the computer can see. Returns (text, contact sheet path)."""
    _quiet_opencv()
    lines = [f"OpenCV {cv2.__version__}, platform {sys.platform}, backend {backend}"]
    names = enumerate_cameras()
    if names is None:
        lines.append("Camera names: unavailable" + (" (pip install pygrabber)" if sys.platform.startswith("win") else ""))
        names = []
    else:
        lines.append("Cameras Windows offers to apps (DirectShow order = index):")
        lines += [f"  {i}: {n}   [{classify_camera(n)}]" for i, n in names] or ["  none"]

    pnp = windows_pnp_devices()
    sony_wpd = False
    problem_usb = []
    capture_pnp = False
    if pnp is not None:
        lines.append("Related Plug and Play devices:")
        for d in pnp:
            fn, cl, st = d.get("FriendlyName") or "?", d.get("Class") or "?", d.get("Status") or "?"
            lines.append(f"  {fn}  [class {cl}, {st}]")
            if re.search("ilce|sony", fn, re.I) and cl in ("WPD", "Image"):
                sony_wpd = True
            if st != "OK" or re.search("unknown usb", fn, re.I):
                problem_usb.append(fn)
            if cl in ("Camera", "Image") and classify_camera(fn) in ("capture", "other", "sony"):
                capture_pnp = True
        if not pnp:
            lines.append("  none")

    probes = probe_cameras(max_index, backend)
    lines.append(f"Probed indices with backend {backend}:")
    nm = dict(names)
    for p in probes:
        lines.append(f"  index {p['index']}: {nm.get(p['index'], '?')}  {p['width']}x{p['height']}"
                     + ("  BLACK / NO SIGNAL" if p["black"] else "  live image"))
    sheet = contact_sheet(probes, names, sheet_path)
    if sheet:
        lines.append(f"Thumbnails saved to {sheet}: open it to see which index is the Sony.")

    kinds = [classify_camera(n) for _, n in names]
    lines.append("Diagnosis:")
    if "sony" in kinds:
        lines.append("  USB-C: the A7 IV is visible as a camera. Use --connection usb.")
    elif sony_wpd:
        lines.append("  USB-C: the A7 IV is connected but as a photo/storage device, not a camera. "
                     "On the camera: Menu > Setup > USB > USB Connection Mode = USB Streaming "
                     "(or Select When Connect, then choose USB Streaming on the screen). Replug.")
    if "capture" in kinds:
        caps = [p for p in probes if classify_camera(nm.get(p["index"], "")) == "capture"]
        live = [p["index"] for p in caps if not p["black"]]
        if live:
            lines.append(f"  HDMI: capture card at index {live} shows a live picture. "
                         f"Use --connection hdmi" + (f" --camera {live[0]}" if len(caps) > 1 else "") + ".")
        else:
            lines.append("  HDMI: capture card found but it shows no picture. Check the camera is on and in "
                         "shooting mode, the HDMI cable is firmly in, and Setup > External Output > "
                         "HDMI Resolution is Auto or 1080p/2160p. If it is still black, try --backend msmf.")
    if problem_usb:
        lines.append(f"  Windows has USB device(s) that failed to start: {problem_usb}. Usually a bad or "
                     "charge-only cable, a hub, or too little power. Try another cable / port.")
    if "sony" not in kinds and "capture" not in kinds and not sony_wpd:
        others = [n for n, k in zip([n for _, n in names], kinds) if k == "other"]
        if not others and not capture_pnp and pnp is not None:
            lines.append(
                "  Windows does not see the Sony or any capture device AT ALL (only built-in cameras). "
                "This is a hardware/connection problem, not a script setting:\n"
                "   HDMI to C: a plain 'HDMI to USB-C cable' is a display cable (laptop -> monitor) and "
                "cannot capture video INTO a computer. You need a capture card (a small box/dongle with a "
                "chip, e.g. Elgato Cam Link, UGREEN or MS2130-based). With one plugged in, Windows lists "
                "a camera such as 'USB Video' plus an audio device.\n"
                "   USB-C to C: camera on, not in playback, Menu > Setup > USB > USB Connection Mode = "
                "USB Streaming, a USB-C cable that carries data, into the camera's USB-C port and a data "
                "port on the PC. When it works, the Windows Camera app can switch to it.")
        elif others:
            lines.append(f"  Unrecognized camera(s) {others}: maybe your capture card under an unusual name. "
                         "Check the thumbnails and pass its index with --camera.")
        else:
            lines.append("  Neither the Sony nor a capture card is visible to Windows. USB-C: use a data cable "
                         "(many C-to-C cables are charge only) into the camera's USB-C port, camera on, USB "
                         "Streaming mode. HDMI: plug the capture card straight into the PC (not a hub), "
                         "then press rescan.")
    if backend in ("auto", "dshow") and sys.platform.startswith("win"):
        lines.append("  If the device is listed but will not open or stays black, try --backend msmf "
                     "(indices can differ there: use the thumbnails).")
    return "\n".join(lines), sheet


def list_cameras(max_index=10, connection="auto", preview=False, backend="auto"):
    if preview:
        names = enumerate_cameras() or []
        sheet = contact_sheet(probe_cameras(max_index, backend), names)
        print(f"Thumbnails: {sheet}" if sheet else "No camera could be opened.")
    devices = enumerate_cameras()
    if devices is not None:
        print("Cameras (index: name):")
        pick = auto_pick_camera(devices, connection)
        for i, n in devices:
            kind = classify_camera(n)
            mark = {"sony": "Sony", "capture": "capture card", "skip": "webcam/virtual, never auto-picked",
                    "other": "unknown"}[kind]
            arrow = "   <- auto picks this" if pick and pick[0] == i else ""
            print(f"  {i}: {n}   [{mark}]{arrow}")
        if not devices:
            print("  none found")
        elif not pick:
            print("Auto pick is ambiguous: pass --camera <index> or a name substring.")
        return
    if sys.platform.startswith("win"):
        print("Tip: pip install pygrabber to see camera names.")
    print("Probing camera indices:")
    for i in range(max_index):
        cap = cv2.VideoCapture(i, default_backend())
        if cap.isOpened():
            ok, frame = cap.read()
            if ok:
                h, w = frame.shape[:2]
                print(f"  index {i}: {w}x{h}")
        cap.release()


def open_camera(args):
    idx, name = resolve_camera(args.camera, getattr(args, "connection", "auto"))
    print(f"Using camera {idx}" + (f": {name}" if name else ""))
    cam = CameraStream(idx, args.width, args.height, args.fps, backend_id(getattr(args, "backend", "auto")))
    if (cam.width, cam.height) != (args.width, args.height):
        print(f"NOTE: asked for {args.width}x{args.height}, device gives {cam.width}x{cam.height}.")
    if name and classify_camera(name) == "capture":
        print("HDMI tip: on the A7 IV set Setup > External Output > HDMI Info. Display = Off, "
              "or the menu overlay ends up in the image.")
    return cam


# --------------------------------------------------------------------------
# Intrinsics
# --------------------------------------------------------------------------

def load_intrinsics(args, width, height):
    if args.calib:
        data = np.load(args.calib)
        K, D = data["K"].astype(np.float64), data["D"].astype(np.float64)
        cw, ch = int(data["width"]), int(data["height"])
        if (cw, ch) != (width, height):
            sx, sy = width / cw, height / ch
            if abs(sx - sy) > 1e-3:
                print(f"WARNING: calibrated at {cw}x{ch}, streaming {width}x{height} (different aspect). "
                      "Recalibrate at the streaming resolution.")
            K = K.copy()
            K[0, :] *= sx
            K[1, :] *= sy
            print(f"Scaled intrinsics from {cw}x{ch} to {width}x{height}.")
        return K, D

    # Rough fallback from lens focal length. Good enough to get started, not for accuracy.
    fx = args.focal_mm / args.sensor_width_mm * width
    print(f"WARNING: no --calib given. Approximating fx = fy = {fx:.0f}px from a {args.focal_mm}mm lens "
          f"and {args.sensor_width_mm}mm sensor width. Expect centimeter level errors.")
    K = np.array([[fx, 0, width / 2], [0, fx, height / 2], [0, 0, 1]], np.float64)
    return K, np.zeros(5)


# --------------------------------------------------------------------------
# Tag detection
# --------------------------------------------------------------------------

# pupil_apriltags can crash (segfault / heap corruption) when a Detector is freed,
# including at interpreter exit. Its destructor is disabled; detectors are only created
# a handful of times per run, so the leak is negligible.
_PUPIL_KEEPALIVE = []


def _disable_pupil_destructor(detector_cls):
    if not getattr(detector_cls, "_del_disabled", False):
        detector_cls.__del__ = lambda self: None
        detector_cls._del_disabled = True


def is_apriltag_family(name):
    return name in ARUCO_APRILTAG_DICTS


def normalize_families(families, family="tag36h11"):
    """List of {"name", "offset", "ids"} specs. Global ID = offset + native ID.
    "ids" = [first, last] native IDs that are accepted (None = all)."""
    if not families:
        return [{"name": family, "offset": 0, "ids": None}]
    if isinstance(families, dict):
        families = [dict(v, name=k) for k, v in families.items()]
    out = []
    for f in families:
        f = {"name": f["name"], "offset": int(f.get("offset", 0)), "ids": f.get("ids")}
        if f["name"] not in ARUCO_APRILTAG_DICTS and not hasattr(cv2.aruco, f["name"]):
            raise ValueError(f"Unknown marker family {f['name']} (use tag36h11, DICT_4X4_50, ...)")
        out.append(f)
    if sum(is_apriltag_family(f["name"]) for f in out) > 1:
        raise ValueError("Only one AprilTag family at a time is supported (plus any ArUco dictionaries).")
    return out


def _make_aruco_detector(dict_name, error_correction=None):
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    if hasattr(cv2.aruco, "ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
    else:
        params = cv2.aruco.DetectorParameters_create()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    if error_correction is not None:
        params.errorCorrectionRate = error_correction
    if hasattr(cv2.aruco, "ArucoDetector"):
        det = cv2.aruco.ArucoDetector(dictionary, params)
        return lambda img: det.detectMarkers(img)
    return lambda img: cv2.aruco.detectMarkers(img, dictionary, parameters=params)


def refine_quad_edges(gray, quad, search=2.5, step=0.2):
    """Sub-pixel corners from straight-line fits to the four black/white edges.

    OpenCV's ArUco corner refinement pulls corners about 0.3 to 0.7 px inward on small
    markers, which reads as a few % too far away (centimeters of depth error). Here each
    edge is sampled along its length, the strongest gradient across it is located with a
    parabolic sub-pixel fit, a line is fitted, and adjacent lines are intersected.
    quad: (4,2) TL, TR, BR, BL. Returns a refined (4,2) float32, or the input if unsure."""
    q = np.asarray(quad, np.float64)
    ctr = q.mean(0)
    img = gray.astype(np.float32)
    lines = []
    offs = np.arange(-search, search + 1e-9, step)
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        L = np.linalg.norm(b - a)
        if L < 8:
            return quad
        d = (b - a) / L
        n = np.array([-d[1], d[0]])
        if n @ ((a + b) / 2 - ctr) < 0:
            n = -n                                         # outward normal
        ts = np.linspace(0.15, 0.85, max(6, int(L / 2)))
        base = a[None, :] + ts[:, None] * (b - a)[None, :]
        pts = base[:, None, :] + offs[None, :, None] * n[None, None, :]
        prof = cv2.remap(img, pts[..., 0].astype(np.float32), pts[..., 1].astype(np.float32),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        g = np.diff(prof, axis=1)                          # dark inside -> bright outside
        k = np.argmax(g, axis=1)
        ok = (k > 0) & (k < g.shape[1] - 1) & (g[np.arange(len(k)), k] > 5)
        if ok.sum() < 4:
            return quad
        k, rows = k[ok], np.nonzero(ok)[0]
        gm, g0, gp = g[rows, k - 1], g[rows, k], g[rows, k + 1]
        den = gm - 2 * g0 + gp
        sub = np.where(np.abs(den) > 1e-6, 0.5 * (gm - gp) / den, 0.0).clip(-0.5, 0.5)
        s_edge = offs[0] + (k + 0.5 + sub) * step          # diff sits between samples
        e_pts = base[rows] + s_edge[:, None] * n[None, :]
        vx, vy, x0, y0 = cv2.fitLine(e_pts.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0], np.float64), np.array([vx, vy], np.float64)))
    out = np.empty((4, 2), np.float64)
    for i in range(4):                                     # corner i = edge(i-1) x edge(i)
        (p1, v1), (p2, v2) = lines[i - 1], lines[i]
        A = np.array([v1, -v2]).T
        if abs(np.linalg.det(A)) < 1e-6:
            return quad
        t1 = np.linalg.solve(A, p2 - p1)[0]
        out[i] = p1 + t1 * v1
    if np.max(np.linalg.norm(out - q, axis=1)) > search + 1.0:
        return quad
    return out.astype(np.float32)


def _inside(pt, quad):
    return cv2.pointPolygonTest(quad.astype(np.float32), (float(pt[0]), float(pt[1])), False) >= 0


class TagDetector:
    """Detects AprilTags and ArUco markers in one call.

    Returns [(global_id, corners)] with corners (4,2) float32 ordered TL, TR, BR, BL (OpenCV
    aruco order, which SOLVEPNP_IPPE_SQUARE expects; pupil_apriltags output is remapped).
    Each family has an ID offset so all markers share one ID space, e.g. AprilTag 36h11
    native 0..86 -> 0..86 and ArUco DICT_4X4_50 native 0..14 -> 87..101.

    Small and far tags (AprilTag family). A camera looking across the floor sees distant tags
    at a grazing angle: an 88 mm tag 3 m away is only ~20 px tall. The fast pass decimates the
    image (quad_decimate) for speed, which shrinks such a tag below what the quad detector can
    segment on a textured floor, so it is missed. Two extra stages recover them cheaply:
      * discovery: a background thread runs a full-resolution pass (no decimation) on the
        newest frame about once per second (one CPU thread) and reports every tag it finds;
      * tracking: every frame, each known tag the fast pass missed is re-detected in a small
        crop around its last position, upscaled so the tag is ~60 px, at full resolution.
    A tag is discovered once and then followed by the cheap crop tracker every frame.
    """

    TRACK_S = 1.5           # keep re-detecting a lost tag in its crop for this long
    ROI_TARGET_PX = 60.0    # upscale crops so a tag's short side is about this many pixels

    def __init__(self, family="tag36h11", engine="auto", detect_scale=1.0,
                 nthreads=4, decimate=1.5, min_margin=20.0, families=None, discover_hz=1.0):
        self.scale = float(detect_scale)
        self.min_margin = min_margin
        self.families = normalize_families(families, family)
        self.info = {}          # global id -> (family, native id), filled as markers are seen
        self.engine = None
        self._april = None
        self._arucos = []
        self._pupil = self._pupil_fine = None
        self._known = {}        # global id -> (corners, time) of the last sighting (any stage)
        self._known_lock = threading.Lock()
        self._disc = None       # background discovery state
        self.stats = {"fast": 0, "roi": 0, "discovered": 0}
        for f in self.families:
            if is_apriltag_family(f["name"]):
                self._april = f
            else:
                # Small ArUco dictionaries correct many bits; be stricter to avoid false hits.
                self._arucos.append((f, _make_aruco_detector(f["name"], error_correction=0.3)))

        parts = []
        if self._april:
            if engine in ("auto", "pupil"):
                try:
                    from pupil_apriltags import Detector
                    _disable_pupil_destructor(Detector)
                    self._pupil = Detector(families=self._april["name"], nthreads=nthreads,
                                           quad_decimate=decimate, refine_edges=1)
                    # full resolution, for tag crops (tracking) and background discovery
                    self._pupil_fine = Detector(families=self._april["name"], nthreads=1,
                                                quad_decimate=1.0, refine_edges=1)
                    self._pupil_disc = Detector(families=self._april["name"], nthreads=1,
                                                quad_decimate=1.0, refine_edges=1)
                    _PUPIL_KEEPALIVE.extend([self._pupil, self._pupil_fine, self._pupil_disc])
                    self.engine = "pupil"
                except ImportError:
                    if engine == "pupil":
                        raise
            if self.engine is None:
                self._april_cv = _make_aruco_detector(ARUCO_APRILTAG_DICTS[self._april["name"]])
                self.engine = "opencv-aruco"
            parts.append(f"{self._april['name']} ({self.engine})")
        parts += [f["name"] for f, _ in self._arucos]
        self.description = " + ".join(parts)

        # Global ID ranges owned by each ArUco family, so a stray AprilTag ID can't collide.
        self._aruco_ranges = []
        for f, _ in self._arucos:
            lo, hi = f["ids"] if f["ids"] else (0, 999)
            self._aruco_ranges.append((f["offset"] + lo, f["offset"] + hi))
        if self._april and self.engine == "pupil" and discover_hz > 0:
            self._disc = {"frame": None, "event": threading.Event(), "period": 1.0 / discover_hz,
                          "running": True}
            threading.Thread(target=self._discover_loop, daemon=True).start()

    @staticmethod
    def _accept(f, native):
        return f["ids"] is None or f["ids"][0] <= native <= f["ids"][1]

    def _april_raw(self, img, det=None):
        """[(global id, corners TL,TR,BR,BL)] of the AprilTag family in img."""
        f = self._april
        raw = []
        if self.engine == "pupil":
            for r in (det or self._pupil).detect(img):
                if r.decision_margin >= self.min_margin:
                    # pupil_apriltags order is TR, TL, BL, BR. Remap to TL, TR, BR, BL.
                    raw.append((int(r.tag_id), np.asarray(r.corners, np.float32)[[1, 0, 3, 2]]))
        else:
            corners, ids, _ = self._april_cv(img)
            if ids is not None:
                raw = [(int(i), c.reshape(4, 2).astype(np.float32)) for c, i in zip(corners, ids.ravel())]
        out = []
        for native, c in raw:
            gid = f["offset"] + native
            if not self._accept(f, native) or any(lo <= gid <= hi for lo, hi in self._aruco_ranges):
                continue
            self.info[gid] = (f["name"], native)
            out.append((gid, c))
        return out

    # ---- small / far tags: background discovery + per-frame crop tracking
    def _discover_loop(self):
        st = self._disc
        while st["running"]:
            st["event"].wait(0.5)
            st["event"].clear()
            item, st["frame"] = st["frame"], None
            if item is None or not st["running"]:
                continue
            t0 = time.perf_counter()
            try:
                found = self._april_raw(item[0], self._pupil_disc)
            except Exception:
                continue
            with self._known_lock:
                for gid, c in found:
                    prev = self._known.get(gid)
                    if prev is None or prev[1] < item[1]:
                        self._known[gid] = (c, item[1])
                self.stats["discovered"] = len(found)
            # keep the CPU for the main loop: run at most every `period` seconds
            time.sleep(max(0.0, st["period"] - (time.perf_counter() - t0)))

    def close(self):
        if self._disc:
            self._disc["running"] = False
            self._disc["event"].set()

    def _track_lost(self, gray, have, now, hints=None):
        """Re-detect, in small upscaled crops, known tags that the fast pass did not find.
        hints {id: expected image corners}: tags whose position is known anyway (the pivot tags,
        projected with the floor pose) are searched for even if never seen in this session."""
        H, W = gray.shape[:2]
        with self._known_lock:
            known = [(g, c) for g, (c, t) in self._known.items() if g not in have and now - t <= self.TRACK_S]
            for g in [g for g, (c, t) in self._known.items() if now - t > self.TRACK_S]:
                del self._known[g]
        listed = {g for g, _ in known}
        for g, c in (hints or {}).items():
            c = np.asarray(c, np.float32)
            if g not in have and g not in listed and np.isfinite(c).all()                     and (c[:, 0].max() > 0) and (c[:, 1].max() > 0) and (c[:, 0].min() < W) and (c[:, 1].min() < H):
                known.append((g, c))
        out = []
        for gid, c in known:
            lo, hi = c.min(0), c.max(0)
            pad = max(0.75 * float(np.max(hi - lo)), 20.0)        # room for motion since last seen
            x0, y0 = (int(v) for v in np.maximum(np.floor(lo - pad), 0))
            x1, y1 = (int(v) for v in np.minimum(np.ceil(hi + pad), [W, H]))
            if x1 - x0 < 12 or y1 - y0 < 12:
                continue
            sides = np.linalg.norm(c - np.roll(c, -1, 0), axis=1)
            up = float(np.clip(self.ROI_TARGET_PX / max(sides.min(), 1.0), 1.0, 4.0))
            crop = gray[y0:y1, x0:x1]
            if up > 1.0:
                crop = cv2.resize(crop, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
            for g, q in self._april_raw(np.ascontiguousarray(crop), self._pupil_fine):
                if g in have:
                    continue
                # crop pixel -> image pixel (pixel centers: x_in = (x_out + 0.5) / up - 0.5)
                out.append((g, ((q + 0.5) / up - 0.5 + [x0, y0]).astype(np.float32)))
                have.add(g)
        return out

    def detect(self, gray, hints=None):
        now = time.perf_counter()
        img = gray
        if self.scale != 1.0:
            img = cv2.resize(gray, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)

        out = []
        if self._april:
            out = self._april_raw(img)

        april_quads = [c for _, c in out]
        if self.engine != "pupil":
            # OpenCV-detected AprilTags need the same edge refinement as ArUco (pupil_apriltags
            # already refines its own edges, and its corners tested better left as they are).
            out = [(gid, refine_quad_edges(img, c)) for gid, c in out]
        for f, det in self._arucos:
            corners, ids, _ = det(img)
            if ids is None:
                continue
            for c, native in zip(corners, ids.ravel()):
                native = int(native)
                c = c.reshape(4, 2).astype(np.float32)
                if not self._accept(f, native):
                    continue
                # An ArUco "marker" found inside an AprilTag is a false hit on its pattern.
                if any(_inside(c.mean(0), q) for q in april_quads):
                    continue
                gid = f["offset"] + native
                self.info[gid] = (f["name"], native)
                out.append((gid, refine_quad_edges(img, c)))

        if out and self.scale != 1.0:
            # Detect at low resolution for speed, then refine corners at full resolution.
            pts = np.concatenate([c for _, c in out]).reshape(-1, 1, 2) / self.scale
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
            win = max(3, int(round(2 / self.scale)))
            pts = cv2.cornerSubPix(gray, pts.astype(np.float32), (win, win), (-1, -1), crit)
            pts = pts.reshape(-1, 4, 2)
            out = [(tid, pts[k]) for k, (tid, _) in enumerate(out)]

        if self._april and self.engine == "pupil":
            self.stats["fast"] = len(out)
            extra = self._track_lost(gray, {g for g, _ in out}, now, hints)
            self.stats["roi"] = len(extra)
            out += extra
            fam = self._april["name"]
            with self._known_lock:
                for gid, c in out:
                    if self.info.get(gid, (None,))[0] == fam:     # crops only re-find AprilTags
                        self._known[gid] = (c, now)
            st = self._disc
            if st is not None and st["frame"] is None:
                st["frame"] = (np.ascontiguousarray(gray), now)
                st["event"].set()
        return out

    def label(self, gid):
        fam, native = self.info.get(gid, (None, gid))
        if fam is None or is_apriltag_family(fam):
            return f"tag{gid}"
        return f"aruco{native}"


def make_detector(cfg, **kw):
    return TagDetector(cfg.get("family", "tag36h11"), families=cfg.get("families"), **kw)


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

def tag_corners_local(size):
    """Tag frame: origin at center, +x right, +y toward the tag's top edge, +z out of the face."""
    h = size / 2.0
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float64)


def rot_z(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float64)


def solve_tag_pose_candidates(corners, size, K, D):
    """Both IPPE solutions for one tag, sorted by reprojection error: [(R_cam_tag, t_cam_tag, err)]."""
    n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
        tag_corners_local(size), corners.astype(np.float64), K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    out = []
    for i in range(n):
        R, _ = cv2.Rodrigues(rvecs[i])
        out.append((R, tvecs[i].reshape(3), float(np.asarray(errs).ravel()[i])))
    return sorted(out, key=lambda c: c[2])


def faces_camera(R_cam_tag, t_cam_tag):
    """A visible tag's +z (out of the printed face) must point back toward the camera."""
    return float(R_cam_tag[:, 2] @ -t_cam_tag) > 0


def solve_tag_pose(corners, size, K, D):
    """Lowest error pose of one tag in the camera frame, or None."""
    cands = solve_tag_pose_candidates(corners, size, K, D)
    return cands[0] if cands else None


def rotation_angle(Ra, Rb):
    return math.acos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))


class FloorModel:
    """Camera pose relative to the floor, estimated from all visible floor tags (anchors) at once.

    Adaptive behaviour (auto_lock=True):
      * stable view for `stable_frames` frames -> the anchor corners of those frames are
        averaged and the pose is locked (removes jitter for a fixed camera);
      * while locked, every frame checks the locked pose against the live detections; if it no
        longer fits (the camera was bumped or moved) for `moved_frames` frames it unlocks,
        tracks live, and re-locks once steady again;
      * learned anchors (tags added at runtime) that stop agreeing with the rest are dropped,
        so a marker that gets moved cannot drag the floor with it.
    """

    def __init__(self, floor_tags, tag_size_of, max_reproj_px=4.0):
        self.world_corners = {}
        for tid, spec in floor_tags.items():
            R = rot_z(spec.get("yaw_deg", 0.0))
            t = np.array([spec.get("x", 0.0), spec.get("y", 0.0), spec.get("z", 0.0)], np.float64)
            self.world_corners[tid] = (R @ tag_corners_local(tag_size_of(tid)).T).T + t
        self.learned = {tid for tid, spec in floor_tags.items() if spec.get("learned")}
        self._update_planar()
        self.max_reproj_px = max_reproj_px
        self.R = self.t = None          # world -> camera
        self.rvec = self.tvec = None
        self.reproj = None
        self.used = []
        self.locked = False
        self.last_ok = 0.0
        self.last_reject = None          # (fit error px, time) of the last rejected pose
        self.auto_lock = False
        self.stable_frames, self.moved_frames = 15, 5
        self.lock_rms = None
        self._buf = deque(maxlen=30)
        self._stable = self._bad = 0
        self._tag_bad = {}
        self.events = []                # human-readable notes for the UI ("re-locked", ...)

    def _update_planar(self):
        zs = np.concatenate([c[:, 2] for c in self.world_corners.values()]) if self.world_corners else [0]
        self.planar = np.ptp(zs) < 1e-9

    @property
    def valid(self):
        return self.R is not None

    def add_anchor(self, tid, corners_world):
        self.world_corners[tid] = np.asarray(corners_world, np.float64)
        self.learned.add(tid)
        self._update_planar()

    def _gather(self, detections):
        obj, img, used = [], [], []
        for tid, c in detections:
            if tid in self.world_corners:
                obj.append(self.world_corners[tid])
                img.append(np.asarray(c, np.float64))
                used.append(tid)
        if not used:
            return None, None, []
        return np.concatenate(obj), np.concatenate(img), used

    @staticmethod
    def _rms(obj, img, rvec, tvec, K, D):
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
        return float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))

    def _solve(self, obj, img, K, D):
        flags = cv2.SOLVEPNP_IPPE if self.planar else cv2.SOLVEPNP_SQPNP
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, img, K, D, flags=flags)
        if n == 0:
            return None
        # Of the (up to two) planar solutions, keep only ones with the camera ABOVE the floor.
        # A flipped solution puts the camera under the floor, which turns every
        # tilt reading upside down (a flat tag reads ~170 deg instead of ~0).
        order = np.argsort(np.asarray(errs).ravel())
        valid = []
        for i in order:
            R_i, _ = cv2.Rodrigues(rvecs[i])
            if (-R_i.T @ tvecs[i].reshape(3))[2] > 0:
                valid.append(i)
        if not valid:
            return None
        i = valid[0]
        e = np.asarray(errs).ravel()
        if (len(valid) > 1 and self.R is not None and e[valid[1]] < 3.0 * max(e[i], 0.05)):
            # Ambiguous (typically one small floor tag): stay with the pose closest to last frame.
            i = min(valid[:2], key=lambda k: rotation_angle(cv2.Rodrigues(rvecs[k])[0], self.R))
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, D, rvecs[i].copy(), tvecs[i].copy())
        return rvec, tvec, self._rms(obj, img, rvec, tvec, K, D)

    def _set(self, rvec, tvec, rms, used, now):
        self.rvec, self.tvec = rvec, tvec
        self.R, _ = cv2.Rodrigues(rvec)
        self.t = tvec.reshape(3)
        self.reproj, self.used, self.last_ok = rms, used, now

    def _drop_bad_learned(self, obj, img, used, rvec, tvec, K, D):
        if not self.learned or len(used) < 3:
            return False
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
        err = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1).reshape(-1, 4).mean(1)
        med = float(np.median(err))
        dropped = False
        for tid, e in zip(used, err):
            if tid in self.learned and e > max(2.5, 4 * med):
                del self.world_corners[tid]
                self.learned.discard(tid)
                self.events.append(f"anchor {tid} no longer fits ({e:.1f} px): it moved, dropped")
                dropped = True
        if dropped:
            self._update_planar()
        return dropped

    def update(self, detections, K, D, now):
        obj, img, used = self._gather(detections)
        if not used:
            self.used = []
            return
        if self.locked:
            if not self.auto_lock:
                return
            limit = max(2.5, 4 * (self.lock_rms or 0.5))
            proj, _ = cv2.projectPoints(obj, self.rvec, self.tvec, K, D)
            per_tag = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1).reshape(-1, 4).mean(1)
            bad = {t for t, e in zip(used, per_tag) if e > limit}
            good = [i for i, t in enumerate(used) if t not in bad]
            if bad and bad <= self.learned and good:
                # Only learned anchors disagree while the others still fit: those markers moved.
                for t in bad:
                    self._tag_bad[t] = self._tag_bad.get(t, 0) + 1
                    if self._tag_bad[t] >= self.moved_frames:
                        del self.world_corners[t]
                        self.learned.discard(t)
                        self._tag_bad.pop(t)
                        self.events.append(f"anchor {t} moved (off by {per_tag[used.index(t)]:.1f} px), dropped")
                        self._update_planar()
                self._bad = 0
                self.used = used
                return
            self._tag_bad = {}
            rms_locked = self._rms(obj, img, self.rvec, self.tvec, K, D)
            self._bad = self._bad + 1 if rms_locked > limit else 0
            if self._bad < self.moved_frames:
                self.used = used
                return
            self.locked = False
            self._bad = self._stable = 0
            self._buf.clear()
            self.events.append(f"camera moved (locked pose off by {rms_locked:.1f} px): tracking live")
        sol = self._solve(obj, img, K, D)
        if sol is None:
            return
        rvec, tvec, rms = sol
        if self._drop_bad_learned(obj, img, used, rvec, tvec, K, D):
            obj, img, used = self._gather(detections)
            sol = self._solve(obj, img, K, D)
            if sol is None:
                return
            rvec, tvec, rms = sol
        if rms > self.max_reproj_px:
            self.last_reject = (rms, now)
            return
        self.last_reject = None
        prev = (self.rvec, self.tvec) if self.valid else None
        self._set(rvec, tvec, rms, used, now)
        if not self.auto_lock:
            return
        # Steady if last frame's pose still explains this frame's corners to within ~1 px.
        steady = prev is not None and self._rms(obj, img, prev[0], prev[1], K, D) < max(1.0, 1.5 * rms)
        self._stable = self._stable + 1 if steady else 0
        self._buf.append([(t, c) for t, c in detections if t in self.world_corners])
        if self._stable >= self.stable_frames:
            acc = {}
            for dets in list(self._buf)[-self.stable_frames:]:
                for t, c in dets:
                    acc.setdefault(t, []).append(np.asarray(c, np.float64))
            avg = [(t, np.mean(v, 0)) for t, v in acc.items() if len(v) >= self.stable_frames // 2]
            o2, i2, u2 = self._gather(avg)
            sol = self._solve(o2, i2, K, D) if u2 else None
            if sol is not None and sol[2] <= self.max_reproj_px:
                self._set(sol[0], sol[1], sol[2], u2, now)
                self.locked = True
                self.lock_rms = sol[2]
                self.events.append(f"pose locked (averaged {self.stable_frames} frames, {sol[2]:.2f} px)")
            self._stable = 0

    def pop_events(self):
        ev, self.events = self.events, []
        return ev

    def cam_to_world(self, p_cam):
        return self.R.T @ (np.asarray(p_cam, np.float64) - self.t)

    def rot_cam_to_world(self, R_cam):
        return self.R.T @ R_cam

    @property
    def camera_position(self):
        return -self.R.T @ self.t

    def pixel_to_floor(self, u, v, K, D, height=0.0):
        """Intersect the camera ray through pixel (u, v) with the plane z = height."""
        if not self.valid:
            return None
        n = cv2.undistortPoints(np.array([[[u, v]]], np.float64), K, D).reshape(2)
        d = self.R.T @ np.array([n[0], n[1], 1.0])
        C = self.camera_position
        if abs(d[2]) < 1e-9:
            return None
        s = (height - C[2]) / d[2]
        return None if s <= 0 else C + s * d

    def world_to_pixel(self, pts, K, D):
        pts = np.asarray(pts, np.float64).reshape(-1, 3)
        proj, _ = cv2.projectPoints(pts, self.rvec, self.tvec, K, D)
        return proj.reshape(-1, 2)


class Smoother:
    """Exponential smoothing of world positions per tag id, reset on large jumps or dropouts."""

    def __init__(self, alpha=0.5, reset_dist=0.15, timeout=0.5):
        self.alpha, self.reset_dist, self.timeout = alpha, reset_dist, timeout
        self.state = {}

    def __call__(self, tid, p, now):
        prev = self.state.get(tid)
        if prev is None or now - prev[1] > self.timeout or np.linalg.norm(p - prev[0]) > self.reset_dist:
            s = p
        else:
            s = self.alpha * p + (1 - self.alpha) * prev[0]
        self.state[tid] = (s, now)
        return s


# --------------------------------------------------------------------------
# Floor survey: measure where the floor tags are, so you don't have to
# --------------------------------------------------------------------------

def survey_floor(samples, K, D, tag_size_of, floor_ids=None, origin_id=None, min_frames=5):
    """Estimate the floor plane and each floor tag's position from a static camera.

    samples   list of per-frame detections [(tag_id, corners)], camera not moving
    floor_ids only these IDs are treated as lying on the floor (None = every tag seen)
    origin_id tag at world (0, 0) whose right edge defines +x (None = largest tag seen)

    Image corners are averaged over all frames first (removes pixel noise), then one plane
    is fitted through the corners of ALL tags together. One small tag on its own gives a
    poorly defined plane; several tags spread across the floor pin it down.
    Returns (floor_tags dict for the config, report dict).
    """
    acc = {}
    for dets in samples:
        for tid, c in dets:
            if floor_ids is None or tid in floor_ids:
                acc.setdefault(tid, []).append(np.asarray(c, np.float64))
    corners = {tid: np.mean(cs, 0) for tid, cs in acc.items() if len(cs) >= min_frames}
    if not corners:
        raise RuntimeError("No floor tag was seen steadily during the survey.")

    cands = {}
    for tid, c in corners.items():
        cs = solve_tag_pose_candidates(c, tag_size_of(tid), K, D)
        cs = [x for x in cs if faces_camera(x[0], x[1])] or cs
        if cs:
            cands[tid] = cs
    if not cands:
        raise RuntimeError("Could not solve any floor tag pose.")

    # Reference normal: plane through tag centers if there are 3+ spread out tags,
    # otherwise the best solution of the tag that looks biggest in the image.
    centers = np.array([v[0][1] for v in cands.values()])
    n_ref = None
    if len(centers) >= 3:
        _, S, Vt = np.linalg.svd(centers - centers.mean(0))
        if S[1] > 0.05 * S[0]:   # not all in a line
            n_ref = Vt[2]
    if n_ref is None:
        big = max(cands, key=lambda t: cv2.contourArea(corners[t].astype(np.float32)))
        n_ref = cands[big][0][0][:, 2]
    if n_ref @ -centers.mean(0) < 0:      # normal points toward the camera
        n_ref = -n_ref
    chosen = {tid: max(v[:2], key=lambda c: c[0][:, 2] @ n_ref) for tid, v in cands.items()}

    # Plane fit through every corner of every tag (camera frame).
    pts = np.concatenate([(R @ tag_corners_local(tag_size_of(t)).T).T + tt
                          for t, (R, tt, _) in chosen.items()])
    centroid = pts.mean(0)
    _, _, Vt = np.linalg.svd(pts - centroid)
    n = Vt[2]
    if n @ -centroid < 0:
        n = -n
    rms_mm = float(np.sqrt(np.mean(((pts - centroid) @ n) ** 2)) * 1e3)

    if origin_id not in chosen:
        origin_id = max(chosen, key=lambda t: (tag_size_of(t), -t))
    R0, t0, _ = chosen[origin_id]
    origin = t0 - ((t0 - centroid) @ n) * n
    x = R0[:, 0] - (R0[:, 0] @ n) * n
    x /= np.linalg.norm(x)
    R_wc = np.vstack([x, np.cross(n, x), n])   # camera vectors -> world

    floor_tags, per_tag = {}, {}
    for tid, (R, t, err) in sorted(chosen.items()):
        p = R_wc @ (t - origin)
        xa = R_wc @ R[:, 0]
        floor_tags[tid] = {"x": round(float(p[0]), 4), "y": round(float(p[1]), 4),
                           "yaw_deg": round(math.degrees(math.atan2(xa[1], xa[0])), 2)}
        per_tag[tid] = {
            "frames": len(acc[tid]),
            "size_mm": round(tag_size_of(tid) * 1e3, 1),
            "height_off_plane_mm": round(float(p[2]) * 1e3, 1),
            "tilt_vs_plane_deg": round(math.degrees(math.acos(np.clip(R[:, 2] @ n, -1, 1))), 1),
            "reproj_px": round(err, 3),
        }
    report = {
        "origin_tag": origin_id,
        "n_tags": len(chosen),
        "plane_rms_mm": round(rms_mm, 2),
        "camera_height_m": round(float(-origin @ n), 4),
        "camera_tilt_from_vertical_deg": round(math.degrees(math.acos(np.clip(-n[2], -1, 1))), 1),
        "tags": per_tag,
        "warnings": [],
    }
    if len(chosen) == 1:
        report["warnings"].append("Only one floor tag: the plane comes from a single tag and will be "
                                  "noticeably tilted at the edges of the view. Use 3 or more, spread out.")
    for tid, r in per_tag.items():
        if r["tilt_vs_plane_deg"] > 4:
            report["warnings"].append(f"Tag {tid} is {r['tilt_vs_plane_deg']} deg off the fitted plane: "
                                      "curled paper, not flat, or wrong size in the sheet.")
    return floor_tags, report


def print_survey_report(report):
    print(f"Floor plane from {report['n_tags']} tag(s), fit RMS {report['plane_rms_mm']} mm, "
          f"origin tag {report['origin_tag']}")
    print(f"Camera height {report['camera_height_m']} m, "
          f"tilted {report['camera_tilt_from_vertical_deg']} deg from straight down")
    for tid, r in report["tags"].items():
        print(f"  tag {tid}: {r}")
    for w in report["warnings"]:
        print("  WARNING:", w)


def save_survey(path, cfg, floor_tags, report):
    out = {k: v for k, v in cfg.items() if k not in ("tag_sizes", "sheet_loaded", "families")}
    out["tag_sizes"] = {}   # sizes come from the sheet; add per-ID overrides here if needed
    out["floor_tags"] = {str(k): v for k, v in floor_tags.items()}
    out["objects"] = {str(k): v for k, v in cfg.get("objects", {}).items()}
    out["_survey_report"] = report
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)


# --------------------------------------------------------------------------
# Mapper: one call per frame
# --------------------------------------------------------------------------

class Mapper:
    def __init__(self, cfg, K, D, detector, smoothing=0.5, max_reproj_px=4.0):
        self.cfg, self.K, self.D, self.detector = cfg, K, D, detector
        self.floor = FloorModel(cfg["floor_tags"], self.tag_size, max_reproj_px)
        self.smoother = Smoother(alpha=smoothing)
        self.prev_rot = {}   # tid -> (R_cam_tag, time), for resolving pose flips

    def tag_size(self, tid):
        return self.cfg["tag_sizes"].get(tid, self.cfg["tag_size"])

    def plane_height(self, tid, spec):
        """Height of the tag's face above the floor if the tag is known to lie flat at a known
        height (objects.<id>.height_m, or cfg["tags_on_floor"] for tags without one), else None."""
        if "height_m" in spec:
            return None if spec["height_m"] is None else float(spec["height_m"])
        return 0.0 if self.cfg.get("tags_on_floor") else None

    def locate_on_plane(self, corners, height, offset=(0, 0, 0)):
        """Precise position of a flat tag at a known height: intersect the camera rays through
        its corners and its exact center (crossing of the diagonals, which is preserved by
        perspective) with the plane z = height. Accuracy then depends on pixels, not on how
        well the tag's distance can be judged from its size. Returns ((x, y, z), heading_deg)."""
        c = np.asarray(corners, np.float64)
        d1, d2 = c[2] - c[0], c[3] - c[1]
        A = np.array([d1, -d2]).T
        if abs(np.linalg.det(A)) < 1e-9:
            return None
        s_ = np.linalg.solve(A, c[1] - c[0])[0]
        center = c[0] + s_ * d1
        pts = [self.floor.pixel_to_floor(u, v, self.K, self.D, height) for u, v in np.vstack([c, center])]
        if any(p is None for p in pts):
            return None
        pts = np.array(pts)
        xdir = (pts[1] - pts[0]) + (pts[2] - pts[3])          # tag +x: TL->TR and BL->BR
        heading = math.atan2(xdir[1], xdir[0])
        off = np.asarray(offset, np.float64)
        ch, sh = math.cos(heading), math.sin(heading)
        p = pts[4] + np.array([ch * off[0] - sh * off[1], sh * off[0] + ch * off[1], off[2]])
        return p, math.degrees(heading)

    def choose_pose(self, tid, cands, now, ambiguity_ratio=3.0):
        """A single small tag has two poses that fit the pixels almost equally well (the
        planar flip): same position, orientation mirrored, e.g. a flat tag reading 120 deg
        tilt. When the two reprojection errors are close, break the tie with physics:

          objects.<id>.orientation = "flat"     tag faces up (top of a box, on the floor)
                                     "vertical" tag on a wall / side face
                                     "auto"     (default) whichever is nearer flat or vertical
                                     "any"      no assumption, keep closest to last frame
        """
        if not cands:
            return None
        cands = [c for c in cands if faces_camera(c[0], c[1])] or cands
        best = cands[0]
        ambiguous = len(cands) > 1 and cands[1][2] < ambiguity_ratio * max(cands[0][2], 0.05)
        if ambiguous:
            spec = self.cfg["objects"].get(tid, {})
            orient = spec.get("orientation", "flat" if (spec.get("flat") or self.cfg.get("tags_on_floor")) else "auto")
            if orient != "any" and self.floor.valid:
                def tilt(c):
                    return math.degrees(math.acos(np.clip((self.floor.R.T @ c[0])[2, 2], -1, 1)))
                if orient == "flat":
                    score = tilt
                elif orient == "vertical":
                    score = lambda c: abs(tilt(c) - 90)
                else:
                    score = lambda c: min(tilt(c), abs(tilt(c) - 90))
                best = min(cands[:2], key=score)
            elif tid in self.prev_rot and now - self.prev_rot[tid][1] < 0.5:
                Rp = self.prev_rot[tid][0]
                best = min(cands[:2], key=lambda c: rotation_angle(c[0], Rp))
        self.prev_rot[tid] = (best[0], now)
        return best

    def process(self, frame, now=None, draw=True):
        now = time.perf_counter() if now is None else now
        t0 = time.perf_counter()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        hints = None
        if self.floor.valid:
            # pivot tags sit at known floor positions: if the fast pass misses one (small, blurred,
            # partly shaded), look for it in an upscaled crop where the floor pose says it is
            hints = {tid: self.floor.world_to_pixel(c, self.K, self.D)
                     for tid, c in self.floor.world_corners.items() if tid not in self.floor.learned}
        try:
            dets = self.detector.detect(gray, hints)
        except TypeError:          # a detector without hint support
            dets = self.detector.detect(gray)
        t_det = time.perf_counter()

        self.floor.update(dets, self.K, self.D, now)
        floor_ok = self.floor.valid

        objects = []
        for tid, corners in dets:
            size = self.tag_size(tid)
            # learned anchors still show up as objects (you placed them, you may want their position)
            is_floor = tid in self.floor.world_corners and tid not in self.floor.learned
            cands = solve_tag_pose_candidates(corners, size, self.K, self.D)
            pose = cands[0] if (is_floor or not cands) else self.choose_pose(tid, cands, now)
            if draw:
                color = (255, 160, 0) if is_floor else (0, 220, 0)
                cv2.polylines(frame, [corners.astype(np.int32)], True, color, 2)
                if pose is not None:
                    rvec, _ = cv2.Rodrigues(pose[0])
                    cv2.drawFrameAxes(frame, self.K, self.D, rvec, pose[1], size * 0.6, 2)
            if is_floor or pose is None:
                if draw and is_floor:
                    _put(frame, f"floor {tid}" + (f" ({self.detector.label(tid)})"
                                                  if hasattr(self.detector, "label") and
                                                  self.detector.label(tid) != f"tag{tid}" else ""),
                         corners[0] + (0, -8), (255, 160, 0))
                continue

            R_ct, t_ct, err = pose
            spec = self.cfg["objects"].get(tid, {})
            offset = np.asarray(spec.get("offset", [0, 0, 0]), np.float64)
            p_cam = t_ct + R_ct @ offset
            fam, native = getattr(self.detector, "info", {}).get(tid, (None, tid))
            obj = {"id": tid, "name": spec.get("name", self.detector.label(tid)
                                               if hasattr(self.detector, "label") else f"tag{tid}"),
                   "family": fam, "native_id": native, "reproj_px": round(err, 2),
                   "distance_m": round(float(np.linalg.norm(t_ct)), 4)}
            plane_h = self.plane_height(tid, spec)
            above = None
            if floor_ok and plane_h is not None and "height_m" not in spec:
                # Global "tags on floor" only applies to tags that really are on the floor:
                # a tag whose 3D pose sits clearly above it keeps the 3D method.
                z_pose = float(self.floor.cam_to_world(t_ct)[2])
                # A single tag's 3D height is uncertain by ~ d^2 * 0.3 px / (f * size): allow 3 sigma,
                # so small far tags on the floor are not mistaken for raised ones.
                d_ = float(np.linalg.norm(t_ct))
                tol = max(self.cfg.get("floor_tolerance_m", 0.03), 3 * d_ * d_ * 0.3 / (self.K[0, 0] * size))
                # Only a tag clearly ABOVE the plane is raised (on a box, on the robot). A tag below
                # the floor is physically impossible: that is single-tag pose error (depth and the
                # planar flip ambiguity are poorly defined for a small far tag), so stay with the
                # plane, which is far more precise. (Before, -0.13 m read as "not on the floor"
                # and the robot / goal tags fell back to the 3D pose, several cm off.)
                if z_pose - plane_h > tol:
                    plane_h, above = None, round(z_pose, 3)
            plane = self.locate_on_plane(corners, plane_h, offset) if floor_ok and plane_h is not None else None
            if plane is not None:
                p_w = self.smoother(tid, plane[0], now)
                obj.update(frame="world", x=float(p_w[0]), y=float(p_w[1]), z=float(p_w[2]),
                           yaw_deg=plane[1], tilt_deg=0.0, method="plane",
                           pose_z_m=round(float(self.floor.cam_to_world(t_ct)[2]), 4))
                label = f"{obj['name']} x{p_w[0]:+.4f} y{p_w[1]:+.4f} (plane z={plane_h:g})"
            elif floor_ok:
                p_w = self.smoother(tid, self.floor.cam_to_world(p_cam), now)
                R_w = self.floor.rot_cam_to_world(R_ct)
                obj.update(frame="world", x=float(p_w[0]), y=float(p_w[1]), z=float(p_w[2]),
                           yaw_deg=math.degrees(math.atan2(R_w[1, 0], R_w[0, 0])),
                           tilt_deg=math.degrees(math.acos(np.clip(R_w[2, 2], -1, 1))),
                           method="pose" if above is None else "pose (above floor: set objects.<id>.height_m "
                                                                "for plane precision)")
                label = f"{obj['name']} x{p_w[0]:+.3f} y{p_w[1]:+.3f} z{p_w[2]:+.3f}"
            else:
                obj.update(frame="camera", x=float(p_cam[0]), y=float(p_cam[1]), z=float(p_cam[2]))
                label = f"{obj['name']} cam {p_cam[0]:+.2f},{p_cam[1]:+.2f},{p_cam[2]:+.2f}"
            objects.append(obj)
            if draw:
                _put(frame, label, corners[0] + (0, -8), (0, 220, 0))

        t_end = time.perf_counter()
        result = {
            "t": now,
            "floor_valid": floor_ok,
            "floor_locked": self.floor.locked,
            "floor_tags_seen": list(self.floor.used),
            "floor_reproj_px": None if self.floor.reproj is None else round(self.floor.reproj, 3),
            "floor_reject_px": None if self.floor.last_reject is None else round(self.floor.last_reject[0], 2),
            "floor_limit_px": self.floor.max_reproj_px,
            "camera": None,
            "objects": objects,
            "timing_ms": {"detect": (t_det - t0) * 1e3, "total": (t_end - t0) * 1e3},
            "n_tags": len(dets),
            "_dets": [(tid, c.copy()) for tid, c in dets],
        }
        if floor_ok:
            C = self.floor.camera_position
            result["camera"] = {"x": float(C[0]), "y": float(C[1]), "z": float(C[2])}
        return result


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------

def _put(img, text, org, color, scale=0.55):
    org = (int(org[0]), int(org[1]))
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.rectangle(img, (org[0] - 2, org[1] - h - 3), (org[0] + w + 2, org[1] + base), (0, 0, 0), -1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_floor_grid(frame, floor, K, D, extent=2.0, step=0.25):
    """Project a floor grid into the image so you can see if the floor estimate is right."""
    if not floor.valid:
        return
    center = np.concatenate(list(floor.world_corners.values())).mean(0)
    cx, cy = round(center[0] / step) * step, round(center[1] / step) * step
    vals = np.arange(-extent, extent + 1e-9, step)
    for v in vals:
        for a, b in ((np.array([cx + v, cy - extent, 0]), np.array([cx + v, cy + extent, 0])),
                     (np.array([cx - extent, cy + v, 0]), np.array([cx + extent, cy + v, 0]))):
            # Sample along the line and keep only points in front of the camera.
            pts = np.linspace(a, b, 40)
            pc = (floor.R @ pts.T).T + floor.t
            pts = pts[pc[:, 2] > 0.05]
            if len(pts) < 2:
                continue
            px = floor.world_to_pixel(pts, K, D).astype(np.int32)
            cv2.polylines(frame, [px], False, (90, 90, 90), 1, cv2.LINE_AA)
    o = floor.world_to_pixel([[0, 0, 0], [0.3, 0, 0], [0, 0.3, 0], [0, 0, 0.3]], K, D).astype(int)
    for k, col in ((1, (0, 0, 255)), (2, (0, 255, 0)), (3, (255, 0, 0))):
        cv2.arrowedLine(frame, tuple(o[0]), tuple(o[k]), col, 3, cv2.LINE_AA, tipLength=0.15)


class TopDownMap:
    def __init__(self, size=600, margin=40):
        self.size, self.margin = size, margin
        self.bounds = None

    def _fit(self, pts):
        pts = np.asarray(pts)
        lo, hi = pts.min(0) - 0.25, pts.max(0) + 0.25
        if self.bounds is None:
            self.bounds = [lo, hi]
        else:
            self.bounds = [np.minimum(self.bounds[0], lo), np.maximum(self.bounds[1], hi)]
        span = max((self.bounds[1] - self.bounds[0]).max(), 0.5)
        self.scale = (self.size - 2 * self.margin) / span

    def _px(self, x, y):
        u = self.margin + (x - self.bounds[0][0]) * self.scale
        v = self.size - self.margin - (y - self.bounds[0][1]) * self.scale
        return int(round(u)), int(round(v))

    def render(self, floor, result, pins):
        img = np.full((self.size, self.size, 3), 30, np.uint8)
        pts = [c[:, :2].mean(0) for c in floor.world_corners.values()]
        pts += [(o["x"], o["y"]) for o in result["objects"] if o["frame"] == "world"]
        pts += [p[:2] for p in pins]
        if result["camera"]:
            pts.append((result["camera"]["x"], result["camera"]["y"]))
        if not pts:
            return img
        self._fit(pts)

        step = next(s for s in (0.1, 0.25, 0.5, 1.0, 2.0, 5.0) if s * self.scale >= 40)
        x0 = math.floor(self.bounds[0][0] / step) * step
        y0 = math.floor(self.bounds[0][1] / step) * step
        for x in np.arange(x0, self.bounds[1][0] + step, step):
            cv2.line(img, self._px(x, -1e3), self._px(x, 1e3), (55, 55, 55), 1)
        for y in np.arange(y0, self.bounds[1][1] + step, step):
            cv2.line(img, self._px(-1e3, y), self._px(1e3, y), (55, 55, 55), 1)
        cv2.arrowedLine(img, self._px(0, 0), self._px(step, 0), (0, 0, 255), 2, tipLength=0.2)
        cv2.arrowedLine(img, self._px(0, 0), self._px(0, step), (0, 255, 0), 2, tipLength=0.2)

        for tid, c in floor.world_corners.items():
            poly = np.array([self._px(x, y) for x, y in c[:, :2]], np.int32)
            seen = tid in result["floor_tags_seen"]
            cv2.polylines(img, [poly], True, (255, 160, 0) if seen else (110, 80, 30), 2)
            _put(img, str(tid), poly[0], (255, 160, 0), 0.45)

        if result["camera"]:
            c = result["camera"]
            p = self._px(c["x"], c["y"])
            fwd = floor.R.T @ np.array([0, 0, 1.0])
            if np.linalg.norm(fwd[:2]) > 1e-6:
                f = fwd[:2] / np.linalg.norm(fwd[:2]) * 0.3
                cv2.arrowedLine(img, p, self._px(c["x"] + f[0], c["y"] + f[1]), (200, 200, 200), 1)
            cv2.circle(img, p, 6, (255, 255, 255), -1)
            _put(img, f"cam h={c['z']:.2f}m", (p[0] + 8, p[1]), (255, 255, 255), 0.45)

        for o in result["objects"]:
            if o["frame"] != "world":
                continue
            p = self._px(o["x"], o["y"])
            cv2.circle(img, p, 7, (0, 220, 0), -1)
            h = math.radians(o["yaw_deg"])
            cv2.line(img, p, (int(p[0] + 14 * math.cos(h)), int(p[1] - 14 * math.sin(h))), (0, 255, 255), 2)
            _put(img, f"{o['name']} z={o['z']:.3f}", (p[0] + 9, p[1] - 6), (0, 220, 0), 0.45)

        for x, y, _ in pins:
            p = self._px(x, y)
            cv2.drawMarker(img, p, (0, 140, 255), cv2.MARKER_CROSS, 14, 2)
            _put(img, f"{x:+.2f},{y:+.2f}", (p[0] + 8, p[1] + 14), (0, 140, 255), 0.4)

        _put(img, f"grid {step:g} m   (+x red, +y green)", (10, self.size - 12), (200, 200, 200), 0.45)
        return img


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------

def public(result):
    return {k: v for k, v in result.items() if not k.startswith("_")}


def run(args):
    if not args.config and default_config_path():
        args.config = default_config_path()
        print(f"Using floor survey {args.config}")
    cfg = load_config(args.config)
    if not cfg["sheet_loaded"]:
        print(f"WARNING: tag sheet {DEFAULT_SHEET} not found; all tags assumed {cfg['tag_size']} m.")
    cam = open_camera(args)
    print(f"Streaming {cam.width}x{cam.height}")
    K, D = load_intrinsics(args, cam.width, cam.height)
    detector = make_detector(cfg, engine=args.engine, detect_scale=args.detect_scale,
                             nthreads=args.threads, decimate=args.decimate)
    print(f"Detector: {detector.description}, floor tags {sorted(cfg['floor_tags'])}")
    mapper = Mapper(cfg, K, D, detector, smoothing=args.smoothing, max_reproj_px=args.max_reproj)
    floor = mapper.floor

    udp = None
    if args.udp:
        host, port = args.udp.rsplit(":", 1)
        udp = (socket.socket(socket.AF_INET, socket.SOCK_DGRAM), (host, int(port)))

    topdown = TopDownMap()
    show_map = not args.no_map
    mouse = {"hover": None, "pins": []}

    def on_mouse(event, x, y, flags, _):
        mouse["hover"] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            p = floor.pixel_to_floor(x, y, K, D)
            if p is not None:
                mouse["pins"].append(p)
                print(f"floor pin: x={p[0]:+.4f} y={p[1]:+.4f}")
        elif event == cv2.EVENT_RBUTTONDOWN:
            mouse["pins"].clear()

    if not args.headless:
        cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("camera", min(cam.width, 1280), int(min(cam.width, 1280) * cam.height / cam.width))
        cv2.setMouseCallback("camera", on_mouse)

    frame_id, n_proc, t_fps, proc_fps, last_print = 0, 0, time.perf_counter(), 0.0, 0.0
    try:
        while True:
            frame, frame_id, stamp = cam.latest(frame_id)
            if frame is None:
                continue
            result = mapper.process(frame, stamp, draw=not args.headless)
            now = time.perf_counter()
            result["latency_ms"] = (now - stamp) * 1e3
            result["pins"] = [{"x": float(p[0]), "y": float(p[1])} for p in mouse["pins"]]

            n_proc += 1
            if now - t_fps >= 1.0:
                proc_fps, n_proc, t_fps = n_proc / (now - t_fps), 0, now

            if udp:
                udp[0].sendto(json.dumps(public(result)).encode(), udp[1])
            if args.print_hz > 0 and now - last_print >= 1.0 / args.print_hz:
                last_print = now
                objs = "  ".join(f"{o['name']}=({o['x']:+.3f},{o['y']:+.3f},{o['z']:+.3f})"
                                 for o in result["objects"])
                print(f"[{proc_fps:4.1f} fps] floor={'ok' if result['floor_valid'] else '--'} {objs}")

            if args.headless:
                continue

            draw_floor_grid(frame, floor, K, D)
            for p in mouse["pins"]:
                if floor.valid:
                    u, v = floor.world_to_pixel([p], K, D)[0]
                    cv2.drawMarker(frame, (int(u), int(v)), (0, 140, 255), cv2.MARKER_CROSS, 22, 2)
            if mouse["hover"] and floor.valid:
                p = floor.pixel_to_floor(*mouse["hover"], K, D)
                if p is not None:
                    _put(frame, f"floor {p[0]:+.3f},{p[1]:+.3f}",
                         (mouse["hover"][0] + 12, mouse["hover"][1] - 12), (0, 140, 255))

            status = (f"proc {proc_fps:4.1f} fps  cam {cam.grab_fps:4.1f} fps  "
                      f"detect {result['timing_ms']['detect']:5.1f} ms  latency {result['latency_ms']:5.1f} ms  "
                      f"tags {result['n_tags']}")
            if result["floor_valid"]:
                c = result["camera"]
                fl = (f"floor {'LOCKED' if floor.locked else 'ok'} tags {result['floor_tags_seen']} "
                      f"reproj {result['floor_reproj_px']}px  camera ({c['x']:+.2f},{c['y']:+.2f},{c['z']:+.2f})")
                col = (0, 200, 255) if floor.locked else (0, 255, 0)
            else:
                fl, col = "floor NOT FOUND: show a floor tag", (0, 0, 255)
            _put(frame, status, (10, 25), (255, 255, 255))
            _put(frame, fl, (10, 50), col)

            cv2.imshow("camera", frame)
            if show_map:
                cv2.imshow("map", topdown.render(floor, result, mouse["pins"]))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("l") and floor.valid:
                floor.locked = not floor.locked
                print("floor pose", "locked" if floor.locked else "unlocked")
            elif key == ord("m"):
                show_map = not show_map
                if not show_map:
                    cv2.destroyWindow("map")
            elif key == ord("c"):
                mouse["pins"].clear()
            elif key == ord("s"):
                name = time.strftime("snapshot_%Y%m%d_%H%M%S.png")
                cv2.imwrite(name, frame)
                print("saved", name)
    finally:
        cam.release()
        cv2.destroyAllWindows()


# --------------------------------------------------------------------------
# survey (CLI)
# --------------------------------------------------------------------------

def survey(args):
    cfg = load_config(args.config)
    cam = open_camera(args)
    K, D = load_intrinsics(args, cam.width, cam.height)
    det = make_detector(cfg)
    size_of = lambda t: cfg["tag_sizes"].get(t, cfg["tag_size"])
    ids = {int(v) for v in args.ids.split(",")} if args.ids else None
    print(f"Surveying for {args.seconds:.1f} s. Keep the camera still.")
    samples, fid, t_end = [], 0, time.perf_counter() + args.seconds
    try:
        while time.perf_counter() < t_end:
            frame, fid, _ = cam.latest(fid)
            if frame is not None:
                samples.append(det.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
    finally:
        cam.release()
    floor_tags, report = survey_floor(samples, K, D, size_of, ids, args.origin)
    print_survey_report(report)
    save_survey(args.out, cfg, floor_tags, report)
    print(f"Saved {args.out}. `run` loads it automatically from this folder.")


# --------------------------------------------------------------------------
# calibrate
# --------------------------------------------------------------------------

def calibrate(args):
    cols, rows = (int(v) for v in args.board.lower().split("x"))
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * args.square

    cam = open_camera(args)
    W, H = cam.width, cam.height
    print(f"Calibrating at {W}x{H} with a {cols}x{rows} inner corner board, {args.square} m squares.")
    print("Move the board around the whole frame and tilt it. Frames are captured automatically.")
    print("SPACE capture now, c compute and save, q quit.")

    obj_pts, img_pts, last_cap, last_c = [], [], 0.0, None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    fid = 0
    cv2.namedWindow("calibrate", cv2.WINDOW_NORMAL)
    try:
        while True:
            frame, fid, _ = cam.latest(fid)
            if frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            s = 960 / W if W > 960 else 1.0
            small = cv2.resize(gray, None, fx=s, fy=s) if s < 1 else gray
            found, corners = cv2.findChessboardCorners(
                small, (cols, rows), cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK)
            key = cv2.waitKey(1) & 0xFF
            if found:
                corners = cv2.cornerSubPix(gray, (corners / s).astype(np.float32), (11, 11), (-1, -1), crit)
                cv2.drawChessboardCorners(frame, (cols, rows), corners, True)
                moved = last_c is None or np.mean(np.linalg.norm(corners - last_c, axis=2)) > 0.05 * W
                now = time.perf_counter()
                if key == ord(" ") or (args.auto and moved and now - last_cap > 0.8):
                    obj_pts.append(objp)
                    img_pts.append(corners)
                    last_cap, last_c = now, corners
                    print(f"captured {len(img_pts)}")
            _put(frame, f"captured {len(img_pts)} (need >= 15)  SPACE capture  c calibrate  q quit",
                 (10, 30), (0, 255, 0) if found else (0, 0, 255), 0.8)
            cv2.imshow("calibrate", frame)
            if key in (ord("q"), 27):
                return
            if key == ord("c"):
                if len(img_pts) < 10:
                    print("Need at least 10 captures.")
                    continue
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()

    rms, K, D, _, _ = cv2.calibrateCamera(obj_pts, img_pts, (W, H), None, None)
    np.savez(args.out, K=K, D=D, width=W, height=H, rms=rms)
    print(f"RMS reprojection error {rms:.3f} px (under ~0.5 px is good). Saved {args.out}")
    print("K =\n", K, "\nD =", D.ravel())


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def cam_args(p):
        p.add_argument("--connection", choices=CONNECTIONS, default="usb",
                       help="usb (default) = A7 IV over USB-C to C in USB Streaming mode, "
                            "hdmi = through an HDMI capture card, auto = usb then hdmi")
        p.add_argument("--backend", choices=["auto"] + list(BACKENDS), default="auto",
                       help="capture backend. auto = DirectShow on Windows; try msmf if a device stays black")
        p.add_argument("--camera", default="auto",
                       help="'auto' (default: Sony over USB, else HDMI capture card), "
                            "a name substring, or an index (see `list`)")
        p.add_argument("--width", type=int, default=1920)
        p.add_argument("--height", type=int, default=1080)
        p.add_argument("--fps", type=int, default=60)

    pl = sub.add_parser("list", help="list cameras and which one auto picks")
    pl.add_argument("--connection", choices=CONNECTIONS, default="usb")
    pl.add_argument("--backend", choices=["auto"] + list(BACKENDS), default="auto")
    pl.add_argument("--preview", action="store_true", help="save thumbnails of every index")
    pd = sub.add_parser("diagnose", help="explain why the Sony / capture card is not found")
    pd.add_argument("--backend", choices=["auto"] + list(BACKENDS), default="auto")

    pc = sub.add_parser("calibrate", help="chessboard calibration")
    cam_args(pc)
    pc.add_argument("--board", default="9x6", help="inner corners, cols x rows")
    pc.add_argument("--square", type=float, default=0.025, help="square size in meters")
    pc.add_argument("--out", default="calib.npz")
    pc.add_argument("--no-auto", dest="auto", action="store_false", help="disable auto capture")

    pg = sub.add_parser("config", help="write an example config JSON")
    pg.add_argument("path", nargs="?", default="floor.json")

    ps = sub.add_parser("survey", help="measure the floor plane and floor tag positions")
    cam_args(ps)
    ps.add_argument("--config", help="base config (objects, sheet). Default: none")
    ps.add_argument("--calib", help="intrinsics .npz from `calibrate`")
    ps.add_argument("--focal-mm", type=float, default=28.0)
    ps.add_argument("--sensor-width-mm", type=float, default=35.9)
    ps.add_argument("--seconds", type=float, default=3.0)
    ps.add_argument("--ids", help="floor tag IDs, e.g. 27,28,0,1,2 (default: every tag visible)")
    ps.add_argument("--origin", type=int, help="tag at world (0,0) (default: largest)")
    ps.add_argument("--out", default=DEFAULT_SURVEY)

    pr = sub.add_parser("run", help="live detection and mapping")
    cam_args(pr)
    pr.add_argument("--config", help="floor/object config JSON (see `config`)")
    pr.add_argument("--calib", help="intrinsics .npz from `calibrate`")
    pr.add_argument("--focal-mm", type=float, default=28.0, help="fallback if no --calib")
    pr.add_argument("--sensor-width-mm", type=float, default=35.9, help="A7 IV full frame width")
    pr.add_argument("--engine", choices=["auto", "pupil", "aruco"], default="auto")
    pr.add_argument("--detect-scale", type=float, default=1.0,
                    help="downscale for detection, e.g. 0.5 at 4K. Corners are refined at full res.")
    pr.add_argument("--decimate", type=float, default=1.5,
                    help="pupil_apriltags quad_decimate. 2 is faster but misses small sheet tags, 1 is slowest")
    pr.add_argument("--threads", type=int, default=4)
    pr.add_argument("--smoothing", type=float, default=0.5,
                    help="EMA weight of the newest position (1 = no smoothing)")
    pr.add_argument("--max-reproj", type=float, default=4.0, help="reject floor poses above this RMS px")
    pr.add_argument("--udp", help="send JSON results each frame to host:port")
    pr.add_argument("--print-hz", type=float, default=2.0, help="console print rate, 0 to disable")
    pr.add_argument("--no-map", action="store_true")
    pr.add_argument("--headless", action="store_true", help="no windows, just print / UDP")

    args = ap.parse_args()
    if args.cmd == "list":
        list_cameras(connection=args.connection, preview=args.preview, backend=args.backend)
    elif args.cmd == "diagnose":
        print(diagnose_cameras(args.backend)[0])
    elif args.cmd == "calibrate":
        calibrate(args)
    elif args.cmd == "config":
        with open(args.path, "w") as f:
            json.dump(EXAMPLE_CONFIG, f, indent=2)
        print(f"Wrote {args.path}. Edit tag sizes and floor tag positions to match your setup.")
    elif args.cmd == "run":
        run(args)
    elif args.cmd == "survey":
        survey(args)


if __name__ == "__main__":
    main()
