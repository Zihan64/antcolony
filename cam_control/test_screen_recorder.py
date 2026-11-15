#!/usr/bin/env python3
"""
Screen recording for experiment runs (screen_recorder.py).

    python test_screen_recorder.py

Checks:
  1. every run gets its own folder (name sanitised, never overwrites an earlier run);
  2. frames of another size are letterboxed into the video size, which is even and capped;
  3. the video plays in real time: captures that come late are filled with the previous frame,
     so N seconds of experiment = N seconds of video;
  4. a real Qt widget (offscreen) is recorded to a playable mp4 + run.json, and its content
     (a colour that changes during the run) is in the video.
"""
import json
import os
import shutil
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import screen_recorder as sr  # noqa: E402

bad = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        bad.append(msg)


def video_info(path):
    cap = cv2.VideoCapture(path)
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return n, fps, (w, h), frames


tmp = tempfile.mkdtemp(prefix="screenrec_test_")
try:
    # 1. run folders ---------------------------------------------------------------------------
    t = time.mktime((2026, 11, 15, 14, 23, 1, 0, 0, -1))
    a = sr.run_folder(tmp, "two cars / trial #3", now=t)
    b = sr.run_folder(tmp, "two cars / trial #3", now=t)
    check(os.path.basename(a) == "run_20261115_142301_two-cars-trial-3", f"folder name is safe: {os.path.basename(a)}")
    check(a != b and os.path.isdir(a) and os.path.isdir(b), "a second run in the same second gets its own folder")

    # 2. sizes ---------------------------------------------------------------------------------
    check(sr.output_size(2561, 1441, 1920) == (1920, 1080), "video size capped at max width, even pixels")
    big = np.full((100, 300, 3), 200, np.uint8)
    fit = sr.fit_frame(big, (200, 200))
    check(fit.shape == (200, 200, 3) and fit[0, 100].max() == 0 and fit[100, 100].min() > 150,
          "a wider frame is letterboxed (black bars), not stretched")

    # 3. real-time duration with late captures -------------------------------------------------
    path = os.path.join(tmp, "rt.mp4")
    sink = sr.VideoSink(path, fps=10, log=lambda *_: None)
    t0 = 1000.0
    for k, dt in enumerate([0.0, 0.1, 0.2, 0.9, 1.0, 1.95]):     # capture stalls 0.2 -> 0.9 s and 1.0 -> 1.95 s
        sink.put(np.full((120, 160, 3), 40 * k, np.uint8), t0 + dt)
    check(sink.close(), "sink closes cleanly")
    n, fps, size, frames = video_info(path)
    check(abs(fps - 10) < 0.01 and n == 20, f"2 s at 10 fps = 20 frames (got {n} at {fps:.1f} fps)")
    check(abs(int(frames[5].mean()) - 80) < 6, "a stall repeats the last frame seen before it")

    # 4. a real widget -------------------------------------------------------------------------
    from PyQt6 import QtCore, QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = QtWidgets.QWidget()
    win.resize(321, 241)                                          # odd size: must still encode
    lab = QtWidgets.QLabel("experiment", win)
    lab.resize(321, 241)
    win.setStyleSheet("background: rgb(0, 0, 255);")              # blue first ...
    win.show()
    rec = sr.ScreenRecorder(win, root=tmp, log=lambda *_: None)
    folder = rec.start("widget", source="window", fps=15, extra={"note": "test"})
    t_end = time.time() + 2.0
    switched = False
    while time.time() < t_end:
        if not switched and time.time() > t_end - 1.0:
            win.setStyleSheet("background: rgb(255, 0, 0);")      # ... then red
            switched = True
        app.processEvents()
        time.sleep(0.005)
    folder2, mp4 = rec.stop()
    meta = json.load(open(os.path.join(folder, "run.json"), encoding="utf-8"))
    n, fps, size, frames = video_info(mp4 or "")
    check(mp4 is not None and os.path.getsize(mp4) > 0, f"mp4 written ({meta.get('codec')})")
    check(abs(n / 15.0 - meta["duration_s"]) < 0.3, f"video length matches the run ({n / 15.0:.2f} s vs {meta['duration_s']} s)")
    check(size == (320, 240), f"odd window size encoded as even {size}")
    check(frames[0][..., 0].mean() > 150 and frames[-1][..., 2].mean() > 150,
          "the widget's content is in the video (blue at the start, red at the end)")
    check(meta["note"] == "test" and meta["frames"] == n and "stopped" in meta, "run.json describes the run")
    check(not rec.active and rec.stop() == (None, None), "stop twice is harmless")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("all screen recorder tests passed" if not bad else f"{len(bad)} FAILED")
sys.exit(1 if bad else 0)
