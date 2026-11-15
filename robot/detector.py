#!/usr/bin/env python3
"""Pluggable "find the objects in this photo" for vlm_grab.py: the VLM (default, unchanged) or a
local YOLO network running on the car itself.

Which one is chosen in detector.json next to vlm_grab.py (missing file = VLM, as before):

    {"backend": "vlm"}                                    the laptop VLM via masterpi_lite/vlm.py
    {"backend": "yolo", "model": "models/blocks.onnx",    a YOLOv8 / YOLO11 detection model
     "conf": 0.35, "iou": 0.5, "imgsz": 416,
     "targets": ["cube", "cylinder"],                     classes to pick up (missing = all)
     "classes": ["cube", "cylinder", ...]}                class names, only needed when the model
                                                          file does not carry them (cv2.dnn path)

Nothing in vlm_grab.py changes: grab_runner.py swaps masterpi_lite.vlm.detect_all for
Detector.detect_all before it runs vlm_grab.main(), and the result has the same shape
(a list of {"box": (x1, y1, x2, y2) pixels, "label": str, "raw": str}).

YOLO runs with what is installed, in this order:
  ultralytics   any Ultralytics model (.pt, .onnx, NCNN folder); heavy to install on a Pi
  onnxruntime   .onnx exported with  yolo export model=best.pt format=onnx imgsz=416
  cv2.dnn       the same .onnx with OpenCV only (already on the cars; slower)

Test it on a photo or on the car's camera (nothing moves):
    python3 detector.py look.jpg                          what the configured backend finds
    python3 detector.py look.jpg --backend yolo --model models/yolo11n.onnx --targets all
    python3 detector.py --camera --repeat 10              timing on the car's own camera
"""
import argparse
import ast
import json
import os
import re
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "detector.json")
ANY = "any"                                     # as masterpi_lite.vlm.ANY: "whatever lies there"
ANY_MAX_FRAC = 0.35                             # as vlm.py: a box bigger than this is not a thing to pick up
DEFAULTS = {"backend": "vlm", "engine": "auto", "conf": 0.35, "iou": 0.5, "imgsz": 416, "targets": None,
            "classes": None, "min_px": 6, "max_frac": ANY_MAX_FRAC}


def load_config(path=CONFIG):
    cfg = dict(DEFAULTS)
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            cfg.update({k: v for k, v in json.load(f).items() if not k.startswith("_")})
    if cfg.get("model") and not os.path.isabs(cfg["model"]):
        cfg["model"] = os.path.join(os.path.dirname(os.path.abspath(path)), cfg["model"])
    return cfg


def onnx_metadata(path):
    """metadata_props of an .onnx file ({"names": "{0: 'cube', ...}", "imgsz": "[416, 416]", ...}),
    read straight from the protobuf so that no onnx package is needed (cv2.dnn cannot see them)."""
    with open(path, "rb") as f:
        b = f.read()

    def varint(i):
        v, sh = 0, 0
        while True:
            c = b[i]
            v |= (c & 0x7F) << sh
            i += 1
            if c < 0x80:
                return v, i
            sh += 7

    def fields(i, end):
        while i < end:
            key, i = varint(i)
            num, wt = key >> 3, key & 7
            if wt == 0:
                _, i = varint(i)
            elif wt == 1:
                i += 8
            elif wt == 5:
                i += 4
            elif wt == 2:
                n, i = varint(i)
                yield num, i, i + n
                i += n
            else:
                return

    meta = {}
    for num, a, z in fields(0, len(b)):
        if num == 14:                                   # ModelProto.metadata_props
            kv = {k: b[x:y].decode("utf-8", "replace") for k, x, y in fields(a, z)}
            if 1 in kv:
                meta[kv[1]] = kv.get(2, "")
    return meta


def letterbox(img, size):
    """Resize keeping the aspect ratio, pad to size x size. -> (image, scale, pad_x, pad_y)"""
    h, w = img.shape[:2]
    s = min(size / w, size / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    px, py = (size - nw) // 2, (size - nh) // 2
    out = np.full((size, size, 3), 114, np.uint8)
    out[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, s, px, py


def decode(pred, conf, iou, scale, px, py, w, h):
    """YOLOv8/11 detection head output (1, 4 + nc, N) -> [(x1, y1, x2, y2, score, class)] in image px."""
    p = np.squeeze(pred)
    if p.ndim != 2:
        raise ValueError(f"unexpected YOLO output shape {np.shape(pred)}")
    if p.shape[0] < p.shape[1]:                  # (4 + nc, N) -> (N, 4 + nc)
        p = p.T
    scores = p[:, 4:]
    cls = scores.argmax(1)
    best = scores[np.arange(len(p)), cls]
    keep = best >= conf
    p, cls, best = p[keep], cls[keep], best[keep]
    if not len(p):
        return []
    cx, cy, bw, bh = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    x1 = (cx - bw / 2 - px) / scale
    y1 = (cy - bh / 2 - py) / scale
    x2 = (cx + bw / 2 - px) / scale
    y2 = (cy + bh / 2 - py) / scale
    rects = [[float(a), float(b), float(c - a), float(d - b)] for a, b, c, d in zip(x1, y1, x2, y2)]
    idx = cv2.dnn.NMSBoxesBatched(rects, best.tolist(), cls.tolist(), conf, iou) if hasattr(cv2.dnn, "NMSBoxesBatched") \
        else cv2.dnn.NMSBoxes(rects, best.tolist(), conf, iou)
    out = []
    for i in np.array(idx).reshape(-1):
        out.append((max(0.0, float(x1[i])), max(0.0, float(y1[i])), min(w - 1.0, float(x2[i])), min(h - 1.0, float(y2[i])),
                    float(best[i]), int(cls[i])))
    return out


class Yolo:
    def __init__(self, cfg):
        self.cfg, self.model_path = cfg, cfg.get("model")
        if not self.model_path or not os.path.exists(self.model_path):
            raise FileNotFoundError(f"YOLO model not found: {self.model_path!r} (set \"model\" in detector.json)")
        self.names, self.engine = None, None
        self.size = int(cfg.get("imgsz", 416))
        want = cfg.get("engine", "auto")             # auto | ultralytics | onnxruntime | cv2
        try:
            if want not in ("auto", "ultralytics"):
                raise ImportError
            from ultralytics import YOLO
            self.net, self.engine = YOLO(self.model_path, task="detect"), "ultralytics"
            self.names = self.net.names
        except ImportError:
            if want == "ultralytics":
                raise RuntimeError("engine 'ultralytics' asked for, but the ultralytics package is not installed")
            if not self.model_path.endswith(".onnx"):
                raise RuntimeError("without the ultralytics package only .onnx models can be used")
            try:
                if want not in ("auto", "onnxruntime"):
                    raise ImportError
                import onnxruntime as ort
                self.sess = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
                self.inp = self.sess.get_inputs()[0]
                meta = self.sess.get_modelmeta().custom_metadata_map
                if meta.get("names"):
                    self.names = ast.literal_eval(meta["names"])
                if meta.get("imgsz"):
                    self.size = int(ast.literal_eval(meta["imgsz"])[0])
                shape = self.inp.shape
                if isinstance(shape[2], int):
                    self.size = shape[2]                 # a fixed-size export must get exactly that size
                self.engine = "onnxruntime"
            except ImportError:
                if want == "onnxruntime":
                    raise RuntimeError("engine 'onnxruntime' asked for, but onnxruntime is not installed")
                self.net, self.engine = cv2.dnn.readNetFromONNX(self.model_path), "cv2.dnn"
                try:
                    meta = onnx_metadata(self.model_path)
                    if meta.get("names"):
                        self.names = ast.literal_eval(meta["names"])
                    if meta.get("imgsz"):
                        self.size = int(ast.literal_eval(meta["imgsz"])[0])
                except (OSError, ValueError, SyntaxError, IndexError):
                    pass
        if cfg.get("classes"):
            self.names = dict(enumerate(cfg["classes"]))
        if self.names is None:
            self.names = {}
        if isinstance(self.names, list):
            self.names = dict(enumerate(self.names))
        t = cfg.get("targets")
        self.targets = None if not t or t == "all" else {str(x).lower() for x in t}

    def name(self, c):
        return str(self.names.get(c, f"class{c}"))

    def infer(self, img):
        """-> [(x1, y1, x2, y2, score, class name)] in image px, every class."""
        h, w = img.shape[:2]
        conf, iou = float(self.cfg.get("conf", 0.35)), float(self.cfg.get("iou", 0.5))
        if self.engine == "ultralytics":
            r = self.net.predict(img, imgsz=self.size, conf=conf, iou=iou, verbose=False)[0]
            b = r.boxes
            return [(*map(float, xyxy), float(s), self.name(int(c)))
                    for xyxy, s, c in zip(b.xyxy.tolist(), b.conf.tolist(), b.cls.tolist())]
        lb, scale, px, py = letterbox(img, self.size)
        blob = cv2.dnn.blobFromImage(lb, 1 / 255.0, (self.size, self.size), swapRB=True, crop=False)
        if self.engine == "onnxruntime":
            pred = self.sess.run(None, {self.inp.name: blob})[0]
        else:
            self.net.setInput(blob)
            pred = self.net.forward()
        return [(x1, y1, x2, y2, s, self.name(c)) for x1, y1, x2, y2, s, c in decode(pred, conf, iou, scale, px, py, w, h)]


class Detector:
    """detect_all(img_bgr, obj) with the same result as masterpi_lite.vlm.detect_all."""

    def __init__(self, cfg=None):
        self.cfg = cfg or load_config()
        self.backend = self.cfg.get("backend", "vlm")
        self.yolo = Yolo(self.cfg) if self.backend == "yolo" else None
        self.last_ms = None

    def detect_all(self, img_bgr, obj, verbose=True):
        if self.yolo is None:                    # the VLM, exactly as before
            sys.path.insert(0, HERE)
            from masterpi_lite import vlm
            return vlm.detect_all(img_bgr, obj, verbose)
        h, w = img_bgr.shape[:2]
        t0 = time.time()
        raw = self.yolo.infer(img_bgr)
        self.last_ms = (time.time() - t0) * 1000
        want = None
        if obj and obj != ANY:                   # a named object: classes whose name is in it (or it in them)
            words = set(re.findall(r"[a-z0-9]+", obj.lower()))
            want = {n for n in {r[5].lower() for r in raw} if n in obj.lower() or words & set(re.findall(r"[a-z0-9]+", n))}
        out, dropped = [], []
        for x1, y1, x2, y2, s, n in raw:
            bw, bh = x2 - x1, y2 - y1
            ok = bw >= self.cfg["min_px"] and bh >= self.cfg["min_px"]
            if obj == ANY and bw * bh > self.cfg["max_frac"] * w * h:     # as vlm.py: only for "anything"
                ok = False
            if self.yolo.targets is not None and n.lower() not in self.yolo.targets:
                ok = False
            if want and n.lower() not in want:
                ok = False
            d = {"box": (x1, y1, x2, y2), "label": n, "raw": f"{n} {s:.2f}", "conf": s}
            (out if ok else dropped).append(d)
        if verbose:
            print(f"  YOLO {self.last_ms:.0f} ms ({self.yolo.engine}): "
                  + (", ".join(f"{d['label']} {d['conf']:.2f} {[int(v) for v in d['box']]}" for d in out) or "nothing")
                  + (f"  (ignored: {', '.join(d['label'] for d in dropped)})" if dropped else ""))
        return out


def install(cfg=None, log=print):
    """Make masterpi_lite.vlm.detect_all use the configured backend (no-op for the VLM)."""
    det = Detector(cfg)
    if det.backend == "vlm":
        return det
    sys.path.insert(0, HERE)
    from masterpi_lite import vlm
    vlm.detect_all = det.detect_all
    log(f"[detector] {det.backend}: {os.path.basename(det.yolo.model_path)} via {det.yolo.engine}, "
        f"targets {sorted(det.yolo.targets) if det.yolo.targets else 'all classes'}")
    return det


def _snap():
    import glob
    devs = sorted(glob.glob("/dev/v4l/by-id/*-video-index0"))
    cap = cv2.VideoCapture(os.path.realpath(devs[0]) if devs else 0, cv2.CAP_V4L2 if devs else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    img = None
    for _ in range(30):                          # let exposure / white balance settle
        ok, f = cap.read()
        if ok:
            img = f
    cap.release()
    if img is None:
        sys.exit("camera: no picture")
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?", help="a photo (else --camera)")
    ap.add_argument("--camera", action="store_true", help="take the picture with the car's camera")
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--backend", choices=["vlm", "yolo"])
    ap.add_argument("--model")
    ap.add_argument("--targets", help="comma-separated class names, or 'all'")
    ap.add_argument("--object", default=ANY, help="what to look for, as vlm_grab.py is told (default: any)")
    ap.add_argument("--conf", type=float)
    ap.add_argument("--engine", choices=["auto", "ultralytics", "onnxruntime", "cv2"])
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--repeat", type=int, default=1, help="run N times and report the time per photo")
    ap.add_argument("--out", help="save an annotated copy here")
    a = ap.parse_args()
    cfg = load_config(a.config)
    for k in ("backend", "model", "conf", "imgsz", "engine"):
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)
    if a.targets:
        cfg["targets"] = "all" if a.targets == "all" else a.targets.split(",")
    img = _snap() if a.camera or not a.image else cv2.imread(a.image)
    if img is None:
        sys.exit(f"cannot read {a.image}")
    det = Detector(cfg)
    times = []
    for _ in range(max(1, a.repeat)):
        t0 = time.time()
        res = det.detect_all(img, a.object, verbose=(not times))
        times.append((time.time() - t0) * 1000)
    if a.repeat > 1:
        print(f"{det.backend}: median {sorted(times)[len(times) // 2]:.0f} ms per photo over {len(times)} runs")
    for d in res:
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, d["label"], (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    if a.out:
        cv2.imwrite(a.out, img)
        print("saved", a.out)


if __name__ == "__main__":
    main()
