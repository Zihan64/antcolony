#!/usr/bin/env python3
"""
run_on_folder.py

Apply the trained warehouse detector to a folder of images (YOLO layout: <dir>/images,
optional <dir>/labels) exactly as object_detector.ObjectDetector does (same confidence floor and
box filters), save one annotated image per input plus a contact sheet, and, if labels exist,
score it: a prediction is correct when it has the right class and IoU >= 0.5 with a label.

  python training/run_on_folder.py "../tests/custum yolo test"  [--conf 0.5] [--imgsz 640]

Drawing: predicted boxes in the class colour with "class conf"; ground-truth boxes thin white.
Output: <dir>/predictions/*.jpg, <dir>/predictions/contact_sheet.jpg, <dir>/predictions/report.json
        (or --out <folder>, e.g. one folder per model to compare them)
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import object_detector as od  # noqa: E402

COLOURS = {"bracket": (0, 165, 255), "knob": (255, 0, 255), "slotted_block": (255, 128, 0),
           "cup": (0, 200, 0), "holder": (0, 0, 255), "stand": (200, 200, 0), "block": (0, 255, 255)}


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--model", default=od.WAREHOUSE_MODEL)
    ap.add_argument("--conf", type=float, default=od.CLOSED_SET_MIN_CONF)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", help="output folder (default <folder>/predictions): keep one per model to compare")
    a = ap.parse_args()

    from ultralytics import YOLO
    m = YOLO(a.model)
    names = m.names
    d = Path(a.folder)
    out = Path(a.out) if a.out else d / "predictions"
    out.mkdir(exist_ok=True)
    report, tiles = {"model": str(a.model), "conf": a.conf, "imgsz": a.imgsz, "images": {}}, []
    tot = dict(tp=0, fp=0, fn=0, wrong_class=0)
    for ip in sorted((d / "images").iterdir()):
        im = cv2.imread(str(ip))
        if im is None:
            continue
        H, W = im.shape[:2]
        r = m.predict(im, conf=a.conf, imgsz=a.imgsz, verbose=False, agnostic_nms=True)[0]
        dets = od.ObjectDetector._parse(r, im.shape, names)
        gts = []
        lp = d / "labels" / (ip.stem + ".txt")
        if lp.exists():
            for line in lp.read_text().split("\n"):
                if line.strip():
                    c, x, y, w, h = line.split()
                    x, y, w, h = float(x) * W, float(y) * H, float(w) * W, float(h) * H
                    gts.append((names[int(c)], (x - w / 2, y - h / 2, x + w / 2, y + h / 2)))
        # greedy matching, most confident prediction first
        used, rows = set(), []
        for dt in sorted(dets, key=lambda q: -q["conf"]):
            best, bi = None, 0.0
            for k, (gn, gb) in enumerate(gts):
                v = iou(dt["box"], gb)
                if k not in used and v > bi:
                    best, bi = k, v
            if best is not None and bi >= 0.5 and gts[best][0] == dt["name"]:
                used.add(best)
                verdict = "correct"
                tot["tp"] += 1
            elif best is not None and bi >= 0.5:
                used.add(best)
                verdict = f"wrong class (is {gts[best][0]})"
                tot["fp"] += 1
                tot["wrong_class"] += 1
                tot["fn"] += 1
            else:
                verdict = "false detection"
                tot["fp"] += 1
            rows.append(dict(name=dt["name"], conf=round(dt["conf"], 3), iou=round(bi, 2), verdict=verdict,
                             box=[round(v) for v in dt["box"]]))
        missed = [gts[k][0] for k in range(len(gts)) if k not in used]
        tot["fn"] += len([k for k in range(len(gts)) if k not in used])
        report["images"][ip.name] = dict(detections=rows, missed=missed, n_labels=len(gts))
        # draw: ground truth thin white, predictions thick in class colour
        vis = im.copy()
        t = max(2, int(round(min(H, W) / 300)))
        for gn, gb in gts:
            cv2.rectangle(vis, tuple(map(int, gb[:2])), tuple(map(int, gb[2:])), (255, 255, 255), max(1, t // 2))
        for row in rows:
            x1, y1, x2, y2 = row["box"]
            col = COLOURS.get(row["name"], (255, 0, 255)) if row["verdict"] == "correct" else (0, 0, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, t)
            lab = f"{row['name']} {row['conf']:.2f}" + ("" if row["verdict"] == "correct" else " X")
            fs = 0.45 * t
            (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, fs, t)
            y0 = max(th + 4, y1 - 4)
            cv2.rectangle(vis, (x1, y0 - th - 4), (x1 + tw + 4, y0 + 2), col, -1)
            cv2.putText(vis, lab, (x1 + 2, y0 - 2), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), t)
        status = f"{ip.stem}: {sum(r_['verdict'] == 'correct' for r_ in rows)}/{len(gts)} found"
        if missed:
            status += "  missed: " + ", ".join(missed)
        cv2.putText(vis, status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * t, (0, 0, 0), t + 3)
        cv2.putText(vis, status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * t, (255, 255, 255), t)
        cv2.imwrite(str(out / f"{ip.stem}_pred.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
        tiles.append(cv2.resize(vis, (450, int(450 * H / W))))
        print(f"{ip.name}: " + "; ".join(f"{q['name']} {q['conf']:.2f} [{q['verdict']}]" for q in rows)
              + (f"  | MISSED: {missed}" if missed else ""))
    if tiles:
        h = max(t_.shape[0] for t_ in tiles)
        tiles = [cv2.copyMakeBorder(t_, 0, h - t_.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t_ in tiles]
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(out / "contact_sheet.jpg"),
                    np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))
    p = tot["tp"] / max(1, tot["tp"] + tot["fp"])
    rc = tot["tp"] / max(1, tot["tp"] + tot["fn"])
    report["summary"] = dict(tot, precision=round(p, 3), recall=round(rc, 3))
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print("summary:", report["summary"])
    print("saved", out)


if __name__ == "__main__":
    main()
