#!/usr/bin/env python3
"""Compare VLMs on the same robot-camera photos: latency + what they detect.

Runs every photo in bench/photos/ (see manifest.json) through one or more Ollama models with
the SAME prompt the cars use (masterpi_lite/vlm.py on the robots), and reports per model:
  * load:      first call, model cold (seconds)
  * latency:   median / p90 per NEW photo once the model is loaded (seconds; repeats of the
               same photo hit Ollama's cache and are not counted)
  * hit rate:  photos with an object where the model reported >= 1 box
  * false pos: empty photos where the model still reported a box
Annotated copies go to bench/results/<model>/ so you can check by eye that the boxes sit on
the objects (there are no ground-truth boxes yet).

Needs only Python 3 + Pillow (pip install pillow) and a running Ollama.

  python3 bench/vlm_bench.py --models qwen2.5vl:3b qwen3-vl:2b-instruct
  python3 bench/vlm_bench.py --models qwen2.5vl:3b --target "blue block"   # named object
  python3 bench/vlm_bench.py --models some-model --coords norm   # model answers in 0..1
  python3 bench/vlm_bench.py --url http://172.20.1.112:11434 ...  # Ollama on another machine

--coords: how the model's box numbers are meant. auto = qwen2.5vl -> pixel, qwen3-vl -> rel1000,
anything else -> guessed per answer (all <= 1 -> norm, any > image size -> rel1000, else pixel).
"""
import argparse
import base64
import io
import json
import math
import os
import random
import re
import statistics
import time
import urllib.request

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))

# --- identical to the robots' masterpi_lite/vlm.py (2026-11-13) ---------------------------
PROMPT = ('Locate the {obj} in the image. Output ONLY JSON like '
          '[{{"bbox_2d": [x1, y1, x2, y2], "label": "{obj}"}}] with pixel coordinates. '
          'If there is no {obj}, output [].')
PROMPT_ANY = ('Detect every separate object lying on the floor or table surface in the image (toys, blocks, '
              'trash, tools, anything that is not the surface itself). Output ONLY JSON like '
              '[{"bbox_2d": [x1, y1, x2, y2], "label": "short name"}] with pixel coordinates. '
              'If the surface is empty, output [].')
ANY_MAX_FRAC = 0.35
ANY_IGNORE = ("bin", "basket", "bucket", "trash can", "container", "robot", "car", "vehicle",
              "wall", "table", "chair", "leg", "cable", "wire", "floor", "carpet", "shadow")


def model_size(w, h, tokens):
    k = math.sqrt(tokens * 784 / (w * h))
    return max(1, int(w * k / 28)) * 28, max(1, int(h * k / 28)) * 28


def parse_boxes(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except ValueError:
        nums = re.findall(r"-?\d+(?:\.\d+)?", m.group(0))
        return [{"bbox_2d": [float(n) for n in nums[:4]]}] if len(nums) >= 4 else []
    if isinstance(items, dict):
        items = [items]
    if len(items) == 4 and all(isinstance(n, (int, float)) for n in items):
        items = [{"bbox_2d": items}]
    items = [{"bbox_2d": i} if isinstance(i, list) else i for i in items]
    return [i for i in items if isinstance(i, dict) and len(i.get("bbox_2d", [])) == 4]
# -----------------------------------------------------------------------------------------


def smart_resize(w, h, min_pixels, max_pixels, factor=28):
    """Qwen2.5-VL's resize (as in its HF processor): the size the model really sees.
    Ollama uses min_pixels = 28*28*1024 = 802816, so a 640x480 photo is UPSCALED to
    1036x784 and pixel boxes refer to that (measured 2026-11-14, Ollama 0.40.13)."""
    hb, wb = max(factor, round(h / factor) * factor), max(factor, round(w / factor) * factor)
    if hb * wb > max_pixels:
        beta = math.sqrt(h * w / max_pixels)
        hb, wb = math.floor(h / beta / factor) * factor, math.floor(w / beta / factor) * factor
    elif hb * wb < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        hb, wb = math.ceil(h * beta / factor) * factor, math.ceil(w * beta / factor) * factor
    return wb, hb


def coord_mode(model, boxes, sw, sh, forced):
    if forced != "auto":
        return forced
    if model.startswith("qwen2.5vl") or model.startswith("qwen2.5-vl"):
        return "pixel"
    if model.startswith("qwen3-vl") or model.startswith("qwen3vl"):
        return "rel1000"
    vals = [v for b in boxes for v in b["bbox_2d"]]
    if vals and max(vals) <= 1.0:
        return "norm"
    if vals and (max(vals) > max(sw, sh) * 1.05):
        return "rel1000"
    return "pixel"


def to_image_px(box, mode, w, h, sw, sh):
    x1, y1, x2, y2 = box
    if mode == "pixel":            # pixels of the image the model was sent (sw x sh)
        sx, sy = w / sw, h / sh
    elif mode == "rel1000":
        sx, sy = w / 1000.0, h / 1000.0
    else:                          # norm 0..1
        sx, sy = w, h
    return (min(x1, x2) * sx, min(y1, y2) * sy, max(x1, x2) * sx, max(y1, y2) * sy)


def chat(url, model, jpg_bytes, prompt, max_tokens):
    body = {"model": model, "stream": False, "keep_alive": -1,
            "messages": [{"role": "user", "content": prompt,
                          "images": [base64.b64encode(jpg_bytes).decode()]}],
            "options": {"temperature": 0, "num_predict": max_tokens}}
    req = urllib.request.Request(url.rstrip("/") + "/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())
    return out["message"]["content"], time.time() - t0


def run_model(a, model, photos, manifest):
    out_dir = os.path.join(HERE, "results", re.sub(r"[^\w.-]+", "_", model))
    os.makedirs(out_dir, exist_ok=True)
    prompt = PROMPT_ANY if a.target == "any" else PROMPT.format(obj=a.target)
    rows, lat = [], []
    load_s = None
    for i, name in enumerate(photos):
        img = Image.open(os.path.join(a.photos, name)).convert("RGB")
        w, h = img.size
        sw, sh = model_size(w, h, a.img_tokens)
        buf = io.BytesIO()
        small = img.resize((sw, sh), Image.LANCZOS)
        # one random corner pixel: a picture Ollama has never seen, so its cache cannot answer
        small.putpixel((0, 0), tuple(random.randrange(256) for _ in range(3)))
        small.save(buf, "JPEG", quality=92)
        if i == 0:                               # model load: a throw-away call on a blank picture
            blank = io.BytesIO()
            Image.new("RGB", (sw, sh), "gray").save(blank, "JPEG")
            load_s = chat(a.url, model, blank.getvalue(), prompt, a.max_tokens)[1]
        # 1st call = a NEW picture (what a car sends every time); later ones hit Ollama's cache
        times = []
        for k in range(a.repeat):
            text, dt = chat(a.url, model, buf.getvalue(), prompt, a.max_tokens)
            times.append(dt)
        lat.append(times[0])
        raw = parse_boxes(text)
        mode = coord_mode(model, raw, sw, sh, a.coords)
        if mode == "pixel":                      # pixels of the picture the model really saw
            sw, sh = smart_resize(sw, sh, a.min_pixels, a.max_pixels)
        kept, dropped = [], []
        for it in raw:
            box = to_image_px(it["bbox_2d"], mode, w, h, sw, sh)
            label = str(it.get("label", a.target))
            bw, bh = box[2] - box[0], box[3] - box[1]
            bad = bw < 6 or bh < 6 or (a.target == "any" and (
                bw * bh > ANY_MAX_FRAC * w * h or
                any(re.search(r"\b%s\b" % k_, label.lower()) for k_ in ANY_IGNORE)))
            (dropped if bad else kept).append((box, label))
        expected = manifest.get(name, {}).get("objects")
        if a.target != "any" and expected is not None:
            expected = [o for o in expected if a.target.split()[-1] in o]
        verdict = "?" if expected is None else (
            ("hit" if kept else "MISS") if expected else ("ok" if not kept else "FALSE+"))
        rows.append(dict(photo=name, s=times[0], cached_s=times[1:], n=len(kept), verdict=verdict,
                         mode=mode, text=" ".join(text.split())[:160]))
        draw = ImageDraw.Draw(img)
        for box, label in kept:
            draw.rectangle(box, outline=(0, 255, 0), width=3)
            draw.text((box[0] + 3, max(0, box[1] - 12)), label, fill=(0, 255, 0))
        for box, label in dropped:
            draw.rectangle(box, outline=(255, 140, 0), width=1)
        draw.text((5, 5), f"{model}  {times[0]:.2f}s  {verdict}", fill=(255, 255, 0))
        img.save(os.path.join(out_dir, name))
        print(f"  {name:24s} {times[0]:6.2f}s  {verdict:6s} boxes={len(kept)}  {rows[-1]['text'][:70]}")
    pos = [r for r in rows if r["verdict"] in ("hit", "MISS")]
    neg = [r for r in rows if r["verdict"] in ("ok", "FALSE+")]
    lat.sort()
    summ = dict(model=model, load_s=round(load_s or 0, 2),
                median_s=round(statistics.median(lat), 2),
                p90_s=round(lat[min(len(lat) - 1, int(len(lat) * 0.9))], 2),
                hits=f"{sum(r['verdict'] == 'hit' for r in pos)}/{len(pos)}",
                false_pos=f"{sum(r['verdict'] == 'FALSE+' for r in neg)}/{len(neg)}",
                coords=sorted({r["mode"] for r in rows}), out=out_dir)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"summary": summ, "settings": vars(a), "rows": rows}, f, indent=1)
    return summ


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:11434")
    ap.add_argument("--photos", default=os.path.join(HERE, "photos"))
    ap.add_argument("--target", default="any", help='"any" (what the cars use) or an object name')
    ap.add_argument("--img-tokens", type=int, default=400, help="same as the cars' vlm_server.json")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--repeat", type=int, default=1, help="calls per photo; only the 1st (uncached) is timed")
    ap.add_argument("--min-pixels", type=int, default=28 * 28 * 1024,
                    help="server-side resize floor for pixel-coord models (Ollama qwen2.5vl: 802816; llama-server: set to its --image-min-tokens * 784)")
    ap.add_argument("--max-pixels", type=int, default=28 * 28 * 16384)
    ap.add_argument("--coords", default="auto", choices=["auto", "pixel", "rel1000", "norm"])
    ap.add_argument("--only", nargs="*", help="only these photo names")
    a = ap.parse_args()
    with open(os.path.join(a.photos, "manifest.json")) as f:
        manifest = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    photos = a.only or sorted(manifest)
    summaries = []
    for m in a.models:
        print(f"\n=== {m}  ({len(photos)} photos, target={a.target!r}, img_tokens={a.img_tokens})")
        summaries.append(run_model(a, m, photos, manifest))
    print("\nmodel                         load   median    p90   hits   false+  coords")
    for s in summaries:
        print(f"{s['model']:28s} {s['load_s']:6.1f}s {s['median_s']:6.2f}s {s['p90_s']:6.2f}s "
              f"{s['hits']:>6s} {s['false_pos']:>7s}  {','.join(s['coords'])}")
    print("annotated photos:", ", ".join(s["out"] for s in summaries))


if __name__ == "__main__":
    main()
