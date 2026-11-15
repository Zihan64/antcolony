#!/usr/bin/env python3
"""
screen_recorder.py

Screen recording for the tracker GUI: every experiment run gets its own folder with an mp4 of
what was on screen, so a run can be watched again (and shown) exactly as it looked.

    experiments/run_20261115_142301_two-cars/
        screen.mp4      the GUI window (or the whole screen), real-time speed
        run.json        when it started / stopped, source, fps, size, frames, dropped frames

How it works
  * Capture runs on the Qt GUI thread (a QTimer at `fps`): QWidget.grab() for the GUI window
    (works even when other windows cover it) or QScreen.grabWindow(0) for the whole screen the
    GUI is on. Only the copy into a numpy array happens there (a few ms).
  * Encoding runs on its own thread (cv2.VideoWriter, H.264 "avc1", else "mp4v"), so the tracker
    and the GUI never wait for the encoder. If the encoder falls behind, the newest frame wins and
    older ones are dropped (counted in run.json), never queued without bound.
  * Real-time playback: frames are written by wall clock. When the GUI is busy and a capture comes
    late, the previous frame is repeated, so 60 s of experiment is 60 s of video.
  * The output size is fixed by the first frame (rounded to even pixels for the codec, scaled to at
    most `max_width`); later frames of another size (window resized) are letterboxed into it.

Console noise: on Windows without openh264-*.dll, the first recording of a session prints
"Failed to load OpenH264 library" and two [ERROR] lines from OpenCV's FFmpeg plugin. They are
harmless: FFmpeg then uses another H.264 encoder (run.json "codec": "avc1" = real H.264).

Standard library + numpy + OpenCV + PyQt6.
"""
import json
import os
import queue
import re
import shutil
import threading
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.join(HERE, "experiments")
CODECS = ("avc1", "mp4v")          # H.264 first (plays everywhere), MPEG-4 part 2 as the fallback


def run_folder(root=DEFAULT_ROOT, name="", now=None):
    """New, unique folder for one experiment run: run_YYYYmmdd_HHMMSS[_name]."""
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", name.strip()).strip("-.")[:60]
    base = os.path.join(root, f"run_{stamp}" + (f"_{safe}" if safe else ""))
    path, k = base, 2
    while os.path.exists(path):
        path, k = f"{base}_{k}", k + 1
    os.makedirs(path)
    return path


def qimage_to_bgr(img):
    """QImage -> HxWx3 uint8 BGR (copied, safe to hand to another thread)."""
    from PyQt6.QtGui import QImage
    img = img.convertToFormat(QImage.Format.Format_RGB32)        # bytes B, G, R, A on little endian
    w, h, bpl = img.width(), img.height(), img.bytesPerLine()
    if w <= 0 or h <= 0:
        return None
    buf = np.frombuffer(img.constBits().asarray(h * bpl), np.uint8).reshape(h, bpl // 4, 4)
    return np.ascontiguousarray(buf[:, :w, :3])


def output_size(w, h, max_width):
    """Video size for a first frame of w x h: scaled to at most max_width, even numbers."""
    s = min(1.0, max_width / float(w)) if max_width else 1.0
    return max(2, int(w * s) // 2 * 2), max(2, int(h * s) // 2 * 2)


def fit_frame(frame, size):
    """Frame scaled to fit `size` (w, h), centred on black (aspect kept)."""
    W, H = size
    h, w = frame.shape[:2]
    if (w, h) == (W, H):
        return frame
    s = min(W / w, H / h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    small = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    if (nw, nh) == (W, H):
        return small
    out = np.zeros((H, W, 3), np.uint8)
    x0, y0 = (W - nw) // 2, (H - nh) // 2
    out[y0:y0 + nh, x0:x0 + nw] = small
    return out


class VideoSink:
    """Background mp4 writer. put(frame, t) never blocks; frames are written at a constant rate
    by their wall-clock time t (late -> previous frame repeated; too many waiting -> oldest dropped)."""

    def __init__(self, path, fps, max_width=1920, log=print):
        self.path, self.fps, self.max_width, self.log = path, float(fps), max_width, log
        self.q = queue.Queue(maxsize=8)
        self.size = self.codec = self.error = None
        self.written = self.dropped = 0
        self.t0 = None
        self._writer = None
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def put(self, frame, t):
        try:
            self.q.put_nowait((frame, t))
        except queue.Full:
            try:
                self.q.get_nowait()                  # the encoder is behind: drop the oldest
                self.dropped += 1
            except queue.Empty:
                pass
            try:
                self.q.put_nowait((frame, t))
            except queue.Full:
                self.dropped += 1

    def close(self, timeout=10.0):
        self.q.put((None, None))                     # blocking: the end marker must arrive
        self._thread.join(timeout)
        return self.error is None

    def prepare(self, frame):
        """Open the writer now, sized for `frame`, on the caller's thread. Opening H.264 takes up to
        ~0.7 s (FFmpeg probes OpenH264 first); done lazily on the encoder thread, the first ~0.5 s of
        frames piled up and were dropped. Raises if no codec works."""
        self._open(frame)

    def _open(self, frame):
        h, w = frame.shape[:2]
        self.size = output_size(w, h, self.max_width)
        try:                                         # codec probing logs scary [ERROR] lines: keep them quiet
            old = cv2.utils.logging.getLogLevel()
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
        except AttributeError:
            old = None
        try:
            for cc in CODECS:
                wr = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*cc), self.fps, self.size)
                if wr.isOpened():
                    self._writer, self.codec = wr, cc
                    return
                wr.release()
        finally:
            if old is not None:
                cv2.utils.logging.setLogLevel(old)
        raise RuntimeError(f"no mp4 codec available ({', '.join(CODECS)})")

    def _loop(self):
        last = None
        try:
            while True:
                frame, t = self.q.get()
                if frame is None:
                    break
                if self._writer is None:
                    self._open(frame)
                if self.t0 is None:
                    self.t0 = t
                frame = fit_frame(frame, self.size)
                # constant frame rate by wall clock: fill the gap since the last frame with copies of it
                due = int((t - self.t0) * self.fps) + 1      # frames that should exist up to t
                while last is not None and self.written < due - 1:
                    self._writer.write(last)
                    self.written += 1
                if self.written < due:
                    self._writer.write(frame)
                    self.written += 1
                last = frame
        except Exception as e:                       # noqa: BLE001 - report, never take the GUI down
            self.error = f"{type(e).__name__}: {e}"
            self.log(f"[screen rec] stopped: {self.error}")
            while True:                              # keep draining so put() never fills up
                try:
                    if self.q.get(timeout=0.5)[0] is None:
                        break
                except queue.Empty:
                    break
        finally:
            if self._writer is not None:
                self._writer.release()


class ScreenRecorder:
    """Records a Qt widget (source="window") or the screen it is on (source="screen") to
    <run folder>/screen.mp4. Call start()/stop() on the GUI thread."""

    SOURCES = ("window", "screen")

    def __init__(self, widget, root=DEFAULT_ROOT, log=print):
        self.widget, self.root, self.log = widget, root, log
        self.sink = self.timer = None
        self.folder = self.meta = None
        self.source, self.fps = "window", 15.0

    @property
    def active(self):
        return self.sink is not None

    def elapsed(self):
        return 0.0 if not self.active else time.time() - self.meta["started_unix"]

    def start(self, name="", source="window", fps=15.0, max_width=1920, extra=None):
        """Begin a new run folder and record into it. Returns the folder."""
        from PyQt6 import QtCore
        if self.active:
            self.stop()
        if source not in self.SOURCES:
            raise ValueError(f"source must be one of {self.SOURCES}")
        self.source, self.fps = source, float(fps)
        first = self._capture()                      # sizes the video; fails here, not silently later
        if first is None:
            raise RuntimeError(f"nothing to capture ({source})")
        self.folder = run_folder(self.root, name)
        sink = VideoSink(os.path.join(self.folder, "screen.mp4"), self.fps, max_width, self.log)
        try:
            sink.prepare(first)
        except Exception:
            sink.close()
            shutil.rmtree(self.folder, ignore_errors=True)   # no empty run folders left behind
            self.folder = None
            raise
        self.sink = sink
        self.meta = {"name": name, "source": source, "fps": self.fps, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "started_unix": time.time(), **(extra or {})}
        self._write_meta()
        self.sink.put(first, time.time())
        self.timer = QtCore.QTimer(self.widget)
        self.timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self._grab)
        self.timer.start(max(10, int(round(1000.0 / self.fps))))
        return self.folder

    def _capture(self):
        """The window / screen now, as BGR (None if it cannot be grabbed)."""
        if self.source == "screen":
            scr = self.widget.screen() or self.widget.window().screen()
            img = scr.grabWindow(0).toImage()
        else:
            img = self.widget.window().grab().toImage()
        return qimage_to_bgr(img)

    def _grab(self):
        if self.sink is None:
            return
        try:
            frame = self._capture()
            if frame is not None:
                self.sink.put(frame, time.time())
        except Exception as e:                       # noqa: BLE001 - a missed frame is not worth a crash
            self.log(f"[screen rec] capture failed: {e!r}")

    def stop(self):
        """Finish the video. Returns (folder, path of the mp4 or None if it failed)."""
        if not self.active:
            return None, None
        self.timer.stop()
        self.timer.deleteLater()
        sink, self.sink, self.timer = self.sink, None, None
        ok = sink.close()
        dur = time.time() - self.meta["started_unix"]
        self.meta.update(stopped=time.strftime("%Y-%m-%d %H:%M:%S"), duration_s=round(dur, 2),
                         frames=sink.written, dropped_frames=sink.dropped, codec=sink.codec,
                         size=list(sink.size) if sink.size else None, error=sink.error)
        self._write_meta()
        path = os.path.join(self.folder, "screen.mp4")
        return self.folder, (path if ok and sink.written else None)

    def note(self, **kw):
        """Add fields to this run's run.json (e.g. the CSV file recorded alongside)."""
        if self.meta is not None:
            self.meta.update(kw)
            self._write_meta()

    def _write_meta(self):
        with open(os.path.join(self.folder, "run.json"), "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=1)
