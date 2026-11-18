#!/usr/bin/env python3
"""
insta360_monitor.py

Quick live monitor for the Insta360 GO 3S (or any USB camera): picture, delivered resolution and
frame rate, stalls, and optionally the AprilTags it can see (with their size in pixels, which
tells whether the camera can track the robots from where it is mounted).

    python insta360_monitor.py                 # dropdown of every camera; picks the Insta360 by itself
    python insta360_monitor.py --camera 1      # any other camera (index or part of the name)

The GO 3S must be in USB webcam mode: connect the Action Pod with a data cable and choose "Webcam"
on the prompt (unplug / replug the cable to get the prompt again). Until then Windows lists it as
a USB device with only a vendor/debug interface (no video interface), so no program can read video
from it; the camera must be docked in the Pod for webcam mode. The monitor says so and keeps waiting.
It reconnects by itself if the stream stalls or the cable is pulled.
"""
import argparse
import os
import subprocess
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import apriltag_floor_mapper as m  # noqa: E402
import insta360_wifi  # noqa: E402

WIFI_ITEM = "insta-wifi"            # dropdown data of the "Wi-Fi: Insta360" entry
URL_ITEM = "__url__"                # dropdown data of the "Network stream URL..." entry

STALL_S = 2.0          # no frame for this long -> reopen the camera
# device names the GO 3S may show up under in webcam mode (it is not always "Insta360 ...")
LIKELY = ("insta", "go 3", "go3", "360", "action pod")


def list_cameras():
    """[(index, name)] of every camera Windows offers (DirectShow order = OpenCV index), or
    probed indices without names if names are unavailable."""
    try:
        devs = m.enumerate_cameras()
    except Exception:          # noqa: BLE001
        devs = None
    if devs is not None:
        return devs
    out = []
    for i in range(6):                         # no names on this platform: probe indices
        cap = cv2.VideoCapture(i, m.default_backend())
        if cap.isOpened():
            out.append((i, f"camera {i}"))
        cap.release()
    return out


def likely_index(devs):
    """The entry that looks like the Insta360, else the first that is not a built-in webcam."""
    for idx, name in devs:
        if any(p in name.lower() for p in LIKELY):
            return idx
    for idx, name in devs:
        if m.classify_camera(name) != "skip":
            return idx
    return None            # only built-in webcams: never switched on without being asked


INSTA_VID = "VID_4255"   # Insta360's USB vendor id (the device name can change between USB modes)


def usb_insta_state():
    """Windows Plug and Play: 'camera' if the Insta360 shows up as a camera, 'usb' if only as a USB
    device (not in webcam mode), None if not connected / unknown.

    Measured on a GO 3S + Action Pod that is NOT in webcam mode: one device "Insta360 GO 3S",
    USB\\VID_4255&PID_D002, interface Class FF / SubClass 42 / Prot 01 (vendor-specific, the
    Android-debug-style interface) on the WinUSB driver. There is no video-class (Class 0E)
    interface at all, so no program and no driver can read video from it in that state."""
    if not sys.platform.startswith("win"):
        return None
    ps = (f"Get-PnpDevice -PresentOnly | Where-Object {{ $_.FriendlyName -match 'insta' -or "
          f"$_.InstanceId -match '{INSTA_VID}' }} | ForEach-Object {{ $_.Class }}")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True,
                             timeout=10, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
    except Exception:          # noqa: BLE001
        return None
    classes = {c.strip().lower() for c in out.splitlines() if c.strip()}
    if classes & {"camera", "image"}:
        return "camera"
    return "usb" if classes else None


class Grabber(QtCore.QThread):
    """Opens the camera, keeps the newest frame, reopens on stalls. All stats live here."""
    status = QtCore.pyqtSignal(str)

    def __init__(self, spec, width, height, fps, backend):
        super().__init__()
        self.spec, self.req = spec, (width, height, fps)
        self.backend = backend
        self.lock = threading.Lock()
        self.frame = None
        self.stamp = 0.0
        self.stamps = deque(maxlen=120)
        self.size = None
        self.name = ""
        self.reconnects = 0
        self.tags = []
        self.detect_tags = False
        self._det = None

    def _open(self):
        if insta360_wifi.is_wifi_spec(self.spec):          # wireless: the camera's own Wi-Fi
            src = insta360_wifi.InstaWifiStream(*insta360_wifi.parse_host(self.spec), log=self.status.emit)
            self.name = f"Wi-Fi {':'.join(map(str, insta360_wifi.parse_host(self.spec)))} ({src.codec or '?'})"
            return src
        if insta360_wifi.is_url(self.spec):                 # wireless: a network stream (e.g. Pi bridge)
            self.name = self.spec
            return insta360_wifi.open_url(self.spec)
        idx, name = m.resolve_camera(self.spec, "auto")
        w, h, f = self.req
        cap = cv2.VideoCapture(idx, m.backend_id(self.backend))
        if not cap.isOpened():
            raise RuntimeError(f"could not open camera {idx}")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, f)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.name = f"{idx}: {name}" if name else str(idx)
        return cap

    def _tag_loop(self):
        """Tag detection on its own thread (newest frame), so it never slows the capture or the fps reading."""
        while not self.isInterruptionRequested():
            with self.lock:
                frame = self.frame
            if not self.detect_tags or frame is None:
                self.msleep(100)
                continue
            try:
                self._find_tags(frame)
            except Exception:                       # noqa: BLE001
                self.msleep(500)
            self.msleep(30)

    def run(self):
        threading.Thread(target=self._tag_loop, daemon=True).start()
        cap = None
        while not self.isInterruptionRequested():
            if cap is None:
                try:
                    cap = self._open()
                    self.status.emit(f"opened {self.name}, waiting for frames ...")
                    last_ok = time.perf_counter()
                except Exception as e:          # noqa: BLE001
                    self.status.emit(f"waiting for the camera: {str(e).splitlines()[0]}")
                    self.msleep(1500)
                    continue
            ok, frame = cap.read()
            now = time.perf_counter()
            if not ok or frame is None:
                if now - last_ok > STALL_S:
                    self.status.emit(f"no frames for {STALL_S:.0f} s: reconnecting")
                    cap.release()
                    cap = None
                    self.reconnects += 1
                else:
                    self.msleep(10)
                continue
            last_ok = now
            with self.lock:
                self.frame, self.stamp = frame, now
                self.stamps.append(now)
                self.size = (frame.shape[1], frame.shape[0])
        if cap is not None:
            cap.release()

    def _find_tags(self, frame):
        if self._det is None:
            self._det = m.make_detector(m.load_config(None), nthreads=2, decimate=1.0, realtime=False)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.tags = [(tid, c) for tid, c in self._det.detect(gray)]

    def stats(self):
        with self.lock:
            st = list(self.stamps)
            frame, stamp, size = self.frame, self.stamp, self.size
        fps = gap = None
        recent = [t for t in st if t > time.perf_counter() - 3.0]
        if len(recent) > 2:
            fps = (len(recent) - 1) / (recent[-1] - recent[0])
            gap = max(np.diff(recent)) * 1e3
        return frame, stamp, size, fps, gap


class Monitor(QtWidgets.QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.setWindowTitle("Insta360 monitor")
        self.resize(1100, 720)
        self.grabber = None

        self.view = QtWidgets.QLabel("starting ...")
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view.setStyleSheet("background:#111; color:#aaa; font-size:15px;")
        self.view.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Ignored)
        self.view.setWordWrap(True)

        self.cam_combo = QtWidgets.QComboBox()
        self.cam_combo.setMinimumWidth(280)
        self.cam_combo.setToolTip("Every camera connected. The list refreshes by itself when a device is "
                                  "plugged in or switched to webcam mode.")
        self.cam_combo.activated.connect(lambda _: self.restart())    # user picked a camera
        rescan = QtWidgets.QToolButton(text="⟳")
        rescan.setToolTip("Rescan cameras")
        rescan.clicked.connect(lambda: self.refresh_cameras(force=True))
        self._devs = None
        self.urls = []                  # network stream URLs entered this session
        self._user_picked = args.camera not in ("", "auto")
        self.res_combo = QtWidgets.QComboBox()
        self.res_combo.addItems(["1920x1080", "1280x720", "2560x1440", "3840x2160"])
        self.res_combo.setCurrentText(f"{args.width}x{args.height}")
        self.fps_spin = QtWidgets.QSpinBox(minimum=1, maximum=60, value=args.fps)
        connect = QtWidgets.QPushButton("Connect")
        connect.clicked.connect(self.restart)
        self.tag_chk = QtWidgets.QCheckBox("Show AprilTags")
        self.tag_chk.toggled.connect(lambda v: self.grabber and setattr(self.grabber, "detect_tags", v))
        snap = QtWidgets.QPushButton("Snapshot")
        snap.clicked.connect(self.snapshot)

        bar = QtWidgets.QHBoxLayout()
        for w in (QtWidgets.QLabel("Camera"), self.cam_combo, rescan, QtWidgets.QLabel("Resolution"), self.res_combo,
                  QtWidgets.QLabel("FPS"), self.fps_spin, connect, self.tag_chk, snap):
            bar.addWidget(w)
        bar.addStretch(1)
        self.info = QtWidgets.QLabel("")
        self.info.setStyleSheet("font-family: Consolas, monospace;")
        root = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(root)
        lay.addLayout(bar)
        lay.addWidget(self.view, 1)
        lay.addWidget(self.info)
        self.setCentralWidget(root)
        self.status_msg = ""
        self.usb_state = None
        self._usb_busy = False

        self.timer = QtCore.QTimer(self, interval=33, timeout=self.refresh)
        self.timer.start()
        self.usb_timer = QtCore.QTimer(self, interval=4000, timeout=self._periodic)
        self.usb_timer.start()
        self.check_usb()
        self.refresh_cameras(initial=args.camera)

    def _periodic(self):
        self.check_usb()
        self.refresh_cameras()

    def refresh_cameras(self, force=False, initial=None):
        """Fill the dropdown with every connected camera; keep the current choice if it is still there.
        A newly appearing camera that looks like the Insta360 is switched to (unless the user picked one)."""
        devs = list_cameras()
        if devs == self._devs and not force:
            return
        old = self._devs or []
        self._devs = devs
        cur = self.cam_combo.currentData()
        self.cam_combo.blockSignals(True)
        self.cam_combo.clear()
        for idx, name in devs:
            tag = "   (looks like the Insta360)" if any(p in name.lower() for p in LIKELY) else ""
            self.cam_combo.addItem(f"{idx}: {name}{tag}", idx)
        # wireless sources: always listed (nothing to detect until this PC joins the camera's Wi-Fi)
        self.cam_combo.insertSeparator(self.cam_combo.count())
        self.cam_combo.addItem(f"Wi-Fi: Insta360 camera network ({insta360_wifi.DEFAULT_HOST})", WIFI_ITEM)
        for u in self.urls:
            self.cam_combo.addItem(f"Network: {u}", u)
        self.cam_combo.addItem("Network stream URL...  (rtsp:// or http://, e.g. a Raspberry Pi bridge)", URL_ITEM)
        self.cam_combo.blockSignals(False)
        pick = None
        if initial not in (None, "", "auto"):          # --camera: index, part of a name, insta-wifi or a URL
            if insta360_wifi.is_wifi_spec(initial) or insta360_wifi.is_url(initial):
                if insta360_wifi.is_url(initial) and initial not in self.urls:
                    self.urls.append(initial)
                    self.cam_combo.insertItem(self.cam_combo.count() - 1, f"Network: {initial}", initial)
                self.cam_combo.setCurrentIndex(max(0, self.cam_combo.findData(
                    WIFI_ITEM if insta360_wifi.is_wifi_spec(initial) else initial)))
                if insta360_wifi.is_wifi_spec(initial):
                    self.cam_combo.setItemData(self.cam_combo.currentIndex(), initial)
                self.restart(auto=True)
                return
            pick = next((i for i, n in devs if initial == str(i) or initial.lower() in n.lower()), None)
        if isinstance(cur, str) and self.cam_combo.findData(cur) >= 0:
            # a wireless source is running: never switched away from automatically
            self.cam_combo.setCurrentIndex(self.cam_combo.findData(cur))
            return
        new = [d for d in devs if d not in old]
        if pick is None and cur is not None and any(i == cur for i, _ in devs):
            pick = cur
            likely_new = likely_index(new) if new and not self._user_picked else None
            if likely_new is not None and any(p in dict(devs)[likely_new].lower() for p in LIKELY):
                pick = likely_new                       # the Insta360 just came up as a camera
        if pick is None:
            pick = likely_index(devs)
        if pick is None:                               # only the laptop's own camera(s)
            self.cam_combo.setCurrentIndex(-1)
            self.cam_combo.setPlaceholderText("pick a camera")
            if self.grabber is None:
                self.status_msg = (("Only built-in cameras are connected: " + ", ".join(n for _, n in devs) + ". ")
                                   if devs else "No USB camera connected. ") + \
                    ("Wireless: pick \"Wi-Fi: Insta360\" after joining the camera's Wi-Fi network, or a network "
                     "stream URL.")
            return
        self.cam_combo.setCurrentIndex(max(0, self.cam_combo.findData(pick)))
        if pick != cur or force or self.grabber is None:
            self.restart(auto=True)

    def restart(self, auto=False):
        if not auto:
            self._user_picked = True
        if self.grabber:
            self.grabber.requestInterruption()
            self.grabber.wait(3000)
            self.grabber = None
        idx = self.cam_combo.currentData()
        if idx == URL_ITEM:
            url, ok = QtWidgets.QInputDialog.getText(
                self, "Network stream", "Stream URL (e.g. rtsp://192.168.1.50:8554/cam or "
                "http://192.168.1.50:8080/stream.mjpg):")
            url = url.strip()
            if not ok or not insta360_wifi.is_url(url):
                self.cam_combo.setCurrentIndex(-1)
                self.status_msg = "no URL given" if not url else f"not a stream URL: {url}"
                return
            if url not in self.urls:
                self.urls.append(url)
                self.cam_combo.insertItem(self.cam_combo.count() - 1, f"Network: {url}", url)
            self.cam_combo.setCurrentIndex(self.cam_combo.findData(url))
            idx = url
        if idx is None:
            self.status_msg = ("no camera chosen: pick one from the list" if self._devs else
                               "no camera found: plug one in (the list refreshes by itself)")
            return
        if idx == WIFI_ITEM or (isinstance(idx, str) and insta360_wifi.is_wifi_spec(idx)):
            self.status_msg = ("connecting to the Insta360 over Wi-Fi ... (this PC must be on the camera's "
                               "Wi-Fi network; the phone app must be closed)")
        w, h = (int(v) for v in self.res_combo.currentText().split("x"))
        self.grabber = Grabber(str(idx), w, h, self.fps_spin.value(), self.args.backend)
        self.grabber.detect_tags = self.tag_chk.isChecked()
        self.grabber.status.connect(self._on_status)
        self.grabber.start()

    def _on_status(self, s):
        self.status_msg = s

    def check_usb(self):
        """Background PnP query: explains a GO 3S that is plugged in but not in webcam mode."""
        if self._usb_busy:
            return
        self._usb_busy = True

        def work():
            self.usb_state = usb_insta_state()
            self._usb_busy = False
        threading.Thread(target=work, daemon=True).start()

    def _usb_hint(self):
        if self.usb_state != "usb" or isinstance(self.cam_combo.currentData(), str):
            return ""                   # not plugged in, or a wireless source is selected
        return ("\n\nThe Insta360 is connected, but NOT as a camera. Its USB port currently offers only a "
                "vendor/debug interface and no video interface, so no program on this PC can read its picture. "
                "(The preview on the Pod's screen comes over the wireless link from the camera; it does not go "
                "out of the USB port.)\n\n"
                "To fix it:\n"
                "1. Put the camera back INTO the Action Pod (webcam video goes through the Pod's connector).\n"
                "2. Unplug and replug the USB-C cable at the Pod.\n"
                "3. On the screen prompt that appears, choose \"Webcam\" (not file transfer / USB drive).\n"
                "The camera then appears in the list above by itself (a new device, class Camera).")

    def refresh(self):
        g = self.grabber
        if g is None:
            self.view.setText(self.status_msg + self._usb_hint())
            return
        frame, stamp, size, fps, gap = g.stats()
        now = time.perf_counter()
        live = frame is not None and now - stamp < STALL_S
        if not live:
            self.view.setText(self.status_msg + self._usb_hint())
            self.info.setText(f"reconnects {g.reconnects}")
            return
        disp = frame.copy()
        tags = list(g.tags) if g.detect_tags else []
        for tid, c in tags:
            c = np.asarray(c)
            edge = float(np.min(np.linalg.norm(np.roll(c, -1, 0) - c, axis=1)))
            col = (0, 220, 0) if edge >= 15 else (0, 140, 255)
            cv2.polylines(disp, [c.astype(np.int32)], True, col, 3)
            m._put(disp, f"tag {tid}  {edge:.0f}px", c[0] + (0, -10), col, 0.8)
        pm = QtGui.QPixmap.fromImage(self._qimage(disp)).scaled(
            self.view.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.view.setPixmap(pm)
        sharp = float(cv2.Laplacian(cv2.cvtColor(cv2.resize(frame, (640, 360)), cv2.COLOR_BGR2GRAY),
                                    cv2.CV_64F).var())
        w, h, f = g.req
        res = f"{size[0]}x{size[1]}" + ("" if size == (w, h) else f" (asked {w}x{h})")
        self.info.setText(
            f"{self.cam_combo.currentText().split('   (')[0]} | {res} | {fps or 0:5.1f} fps (asked {f}) | worst frame gap {gap or 0:4.0f} ms | "
            f"frame age {1e3 * (now - stamp):3.0f} ms | sharpness {sharp:5.0f} | reconnects {g.reconnects}"
            + (f" | tags {len(tags)} (green >= 15 px edge: trackable)" if g.detect_tags else ""))

    @staticmethod
    def _qimage(bgr):
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        h, w = rgb.shape[:2]
        return QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888).copy()

    def snapshot(self):
        frame = self.grabber.stats()[0] if self.grabber else None
        if frame is None:
            return
        path = os.path.abspath(time.strftime("insta360_%Y%m%d_%H%M%S.png"))
        cv2.imwrite(path, frame)
        self.status_msg = f"saved {path}"
        self.statusBar().showMessage(f"saved {path}", 5000)

    def closeEvent(self, e):
        if self.grabber:
            self.grabber.requestInterruption()
            self.grabber.wait(3000)
        super().closeEvent(e)


def main():
    ap = argparse.ArgumentParser(description="Live monitor for the Insta360 GO 3S (or any USB camera)")
    ap.add_argument("--camera", default="auto", help="auto (the camera that looks like the Insta360), an index, or part of a name")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--backend", default="auto", choices=["auto", "dshow", "msmf"])
    args = ap.parse_args()
    app = QtWidgets.QApplication(sys.argv)
    win = Monitor(args)
    win.show()
    code = app.exec()
    sys.stdout.flush()
    os._exit(code)       # pupil_apriltags detectors must not be destructed


if __name__ == "__main__":
    main()
