#!/usr/bin/env python3
"""Run ON A CAR: time each step of one VLM call as vlm_grab.py makes it, against a real server.

  ssh <car> 'python3 - --url http://<mac-ip>:8090 --photo ~/Desktop/"Master Pi5"/masterpi_lite/look.jpg' < bench/car_timing.py

Prints: camera open / warm-up / one frame (as vlm_grab.snap does), then for each of --n calls:
  prep     = resize to <=400 image tokens + JPEG + base64 + JSON (on the car's CPU)
  server   = the server's own timings (prompt eval incl. image encoding, + token generation)
  network  = the rest of the request: Wi-Fi upload/download + HTTP overhead
Each call gets 8x8 random pixels so the server's cache cannot answer. Needs python3 + OpenCV.
"""
import argparse
import base64
import json
import math
import os
import time
import urllib.request

import cv2
import numpy as np

P = ('Detect every separate object lying on the floor or table surface in the image (toys, blocks, '
     'trash, tools, anything that is not the surface itself). Output ONLY JSON like '
     '[{"bbox_2d": [x1, y1, x2, y2], "label": "short name"}] with pixel coordinates. '
     'If the surface is empty, output [].')

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--photo", required=True)
ap.add_argument("--n", type=int, default=5)
ap.add_argument("--tokens", type=int, default=400)
ap.add_argument("--no-camera", action="store_true")
a = ap.parse_args()

if not a.no_camera:
    t = time.time(); cap = cv2.VideoCapture(0); t_open = time.time() - t
    t = time.time()
    for _ in range(10):
        cap.read()
    t_warm = time.time() - t
    t = time.time(); ok, _ = cap.read(); t_one = time.time() - t
    cap.release()
    print(f"camera: open {t_open*1000:.0f} ms | 10 warm-up frames {t_warm*1000:.0f} ms | 1 frame {t_one*1000:.0f} ms | ok={ok}")

img = cv2.imread(os.path.expanduser(a.photo))
rows = []
for i in range(a.n):
    im = img.copy()
    im[:8, :8] = np.random.randint(0, 256, (8, 8, 3), np.uint8)
    t0 = time.time()
    h, w = im.shape[:2]
    k = math.sqrt(a.tokens * 784 / (w * h))
    sw, sh = int(w * k / 28) * 28, int(h * k / 28) * 28
    small = cv2.resize(im, (sw, sh), interpolation=cv2.INTER_AREA)
    ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 92])
    b64 = base64.b64encode(jpg.tobytes()).decode()
    body = json.dumps({"temperature": 0, "max_tokens": 200, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
        {"type": "text", "text": P}]}]}).encode()
    t1 = time.time()
    req = urllib.request.Request(a.url.rstrip("/") + "/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=120).read())
    t2 = time.time()
    tm = r["timings"]
    srv = tm["prompt_ms"] + tm["predicted_ms"]
    row = dict(prep=(t1 - t0) * 1000, total=(t2 - t1) * 1000, prompt=tm["prompt_ms"], gen=tm["predicted_ms"],
               gen_n=tm["predicted_n"], net=(t2 - t1) * 1000 - srv, cache=tm["cache_n"],
               jpg_kb=len(jpg) / 1024, body_kb=len(body) / 1024, size=f"{sw}x{sh}")
    rows.append(row)
    print("prep {prep:.0f} ms ({size}, jpeg {jpg_kb:.0f} KB, body {body_kb:.0f} KB) | request {total:.0f} ms = "
          "prompt {prompt:.0f} + gen {gen:.0f} ({gen_n} tok) + network {net:.0f} | cache_n {cache}".format(**row))
med = lambda k: sorted(r[k] for r in rows)[len(rows) // 2]
print("median: prep {:.0f} | prompt {:.0f} | gen {:.0f} | network {:.0f} | request {:.0f} ms".format(
    med("prep"), med("prompt"), med("gen"), med("net"), med("total")))
