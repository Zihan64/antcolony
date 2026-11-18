#!/usr/bin/env python3
"""YOLO on the car's own Raspberry Pi (onnxruntime, no laptop, no Wi-Fi in the loop).

    ~/yolo_env/bin/python yolo_pi.py serve --model models/dozer.onnx --port 11447
    ~/yolo_env/bin/python yolo_pi.py bench --model models/dozer.onnx            # speed on this Pi
    ~/yolo_env/bin/python yolo_pi.py photo look.jpg --model models/dozer.onnx   # boxes in one picture

`serve` answers the same POST /detect as the laptop's yolo_server.py, so cleaner.py / dozer_auto.py
only change their URL:  {"image": <base64 jpeg>, "target": "cardboard box" | "cargo" | "all" | ...}
  -> {"objects": [{"bbox_2d": [x1,y1,x2,y2] (0-1000), "label": "trash", "conf": 0.8}], "ms": 60}

Models: export on the laptop with
    yolo export model=runs/.../best.pt format=onnx imgsz=416 opset=17 simplify=True
(output [1, 4 + classes, N]: box centre/size in input pixels, then one score per class; NMS here).
Class names are read from the ONNX metadata Ultralytics writes. Needs only onnxruntime, numpy,
Pillow:  python3 -m venv --system-site-packages ~/yolo_env && ~/yolo_env/bin/pip install onnxruntime
"""
import argparse
import ast
import base64
import io
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import onnxruntime as ort
from PIL import Image

BIN_WORDS = ("cardboard", "bin", "carton", "basket", "crate")
CARGO_WORDS = ("cargo", "cube")
CARGO_CLASSES = ("cargo", "block")       # regular boxes / cubes: never pushed (the team's 7-class model calls them "block")
NOT_TRASH = CARGO_CLASSES + ("bin", "stand")   # "stand": framed the fridge and a grey board on car C


class Yolo:
    def __init__(self, path, conf=0.35, iou=0.5, threads=4):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.name, self.size = inp.name, inp.shape[2]
        meta = self.sess.get_modelmeta().custom_metadata_map
        self.names = ast.literal_eval(meta["names"]) if "names" in meta else {}
        self.conf, self.iou = conf, iou

    def __call__(self, img, classes=None, conf=None):
        """[(class name, score, (x1, y1, x2, y2) fractions)], biggest first."""
        conf = self.conf if conf is None else conf
        W, H = img.size
        s = self.size / max(W, H)
        nw, nh = round(W * s), round(H * s)
        canvas = Image.new("RGB", (self.size, self.size), (114, 114, 114))
        px, py = (self.size - nw) // 2, (self.size - nh) // 2
        canvas.paste(img.resize((nw, nh), Image.BILINEAR), (px, py))
        x = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
        out = self.sess.run(None, {self.name: x})[0][0]           # (4 + nc, N)
        boxes, scores = out[:4].T, out[4:].T
        cls = scores.argmax(1)
        sc = scores[np.arange(len(cls)), cls]
        keep = sc >= conf
        if classes is not None:
            keep &= np.isin(cls, classes)
        boxes, sc, cls = boxes[keep], sc[keep], cls[keep]
        if not len(sc):
            return []
        xy = np.empty_like(boxes)
        xy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2 - px) / s
        xy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2 - py) / s
        xy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2 - px) / s
        xy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2 - py) / s
        out = []
        for c in np.unique(cls):                                  # per-class NMS
            idx = np.where(cls == c)[0]
            idx = idx[np.argsort(-sc[idx])]
            while len(idx):
                i, idx = idx[0], idx[1:]
                out.append(i)
                if not len(idx):
                    break
                a, b = xy[i], xy[idx]
                iw = np.clip(np.minimum(a[2], b[:, 2]) - np.maximum(a[0], b[:, 0]), 0, None)
                ih = np.clip(np.minimum(a[3], b[:, 3]) - np.maximum(a[1], b[:, 1]), 0, None)
                inter = iw * ih
                union = (a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter
                idx = idx[inter / np.maximum(union, 1e-9) < self.iou]
        res = []
        for i in out:
            x1, y1, x2, y2 = np.clip(xy[i], 0, [W, H, W, H])
            res.append((self.names.get(int(cls[i]), str(int(cls[i]))), float(sc[i]), (x1 / W, y1 / H, x2 / W, y2 / H)))
        res.sort(key=lambda r: -(r[2][2] - r[2][0]) * (r[2][3] - r[2][1]))
        return res

    def classes_for(self, target):
        """Same target words as the laptop's yolo_server.py."""
        t = target.lower()
        ids = {v: k for k, v in self.names.items()}
        if t == "all":
            return None
        if any(w in t for w in BIN_WORDS) and "bin" in ids:
            return [ids["bin"]]
        if any(w in t for w in CARGO_WORDS):
            return [ids[n] for n in CARGO_CLASSES if n in ids] or [0]
        if "trash" in ids or "debris" in ids:
            return [ids.get("trash", ids.get("debris"))]
        return [i for n, i in ids.items() if n not in NOT_TRASH]   # every other class is trash


def serve(model, port):
    lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            t0 = time.time()
            try:
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                img = Image.open(io.BytesIO(base64.b64decode(req["image"]))).convert("RGB")
                with lock:
                    res = model(img, model.classes_for(req["target"]))
                body = {"objects": [{"bbox_2d": [v * 1000 for v in b], "label": n, "conf": round(c, 2)} for n, c, b in res],
                        "ms": round((time.time() - t0) * 1000)}
                code = 200
            except Exception as e:
                body, code = {"error": str(e)}, 500
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    model(Image.new("RGB", (640, 360)))                           # warm-up
    print(f"YOLO on the Pi: {model.names} on 127.0.0.1:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["serve", "bench", "photo"])
    ap.add_argument("image", nargs="?")
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=11447)
    ap.add_argument("--conf", type=float, default=0.35)
    a = ap.parse_args()
    m = Yolo(a.model, a.conf)
    if a.cmd == "serve":
        serve(m, a.port)
    elif a.cmd == "photo":
        for n, c, b in m(Image.open(a.image).convert("RGB")):
            print(f"{n:6} {c:.2f}  {tuple(round(v, 3) for v in b)}")
    else:
        img = Image.new("RGB", (768, 432), (90, 100, 110))
        m(img)
        t0 = time.time()
        for _ in range(10):
            m(img)
        print(f"{(time.time() - t0) * 100:.0f} ms per picture ({m.size}x{m.size}, {len(m.names)} classes)")


if __name__ == "__main__":
    sys.exit(main())
