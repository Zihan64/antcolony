#!/usr/bin/env python3
"""Runs the trained warehouse detector exactly as object_detector.ObjectDetector does (same
confidence floor, same box filters) on the real val/test photos, prints every detection plus
anything the model sees at conf >= 0.1, and saves an annotated contact sheet.

  python training/check_real_photos.py [out.jpg]
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import object_detector as od  # noqa: E402
from ultralytics import YOLO  # noqa: E402

out = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "runs" / "real_photo_check.jpg")
m = YOLO(od.WAREHOUSE_MODEL)
src = ROOT / "datasets" / "yolo26_warehouse" / "images"
files = sorted((src / "test").glob("*.jpeg")) + sorted((src / "val").glob("*.jpeg"))
tiles = []
for f in files:
    im = cv2.imread(str(f))
    t = time.perf_counter()
    r = m.predict(im, conf=od.CLOSED_SET_MIN_CONF, imgsz=640, verbose=False, agnostic_nms=True)[0]
    ms = (time.perf_counter() - t) * 1e3
    low = m.predict(im, conf=0.1, imgsz=640, verbose=False)[0]
    dets = od.ObjectDetector._parse(r, im.shape, m.names)
    seen = [(m.names[int(c)], round(float(p), 2)) for c, p in zip(low.boxes.cls, low.boxes.conf)]
    print(f"{f.parent.name}/{f.name}: kept {[(d['name'], round(d['conf'], 2)) for d in dets]}"
          f"  | all>=0.1 {seen}  {ms:.0f} ms")
    for d in dets:
        x1, y1, x2, y2 = map(int, d["box"])
        cv2.rectangle(im, (x1, y1), (x2, y2), (255, 0, 255), 4)
        cv2.putText(im, f"{d['name']} {d['conf']:.2f}", (x1, max(30, y1 - 8)), 0, 1.3, (255, 0, 255), 3)
    tiles.append(cv2.resize(im, (300, 400)))
while len(tiles) % 5:
    tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(out, np.vstack([np.hstack(tiles[i:i + 5]) for i in range(0, len(tiles), 5)]))
print("saved", out)
