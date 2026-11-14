#!/usr/bin/env python3
"""Simulate several cars asking ONE VLM server at the same time.

Each simulated car is a thread that sends --rounds new pictures (bench photos with 8x8 random
pixels, so no cache hits), with the cars' exact prompt and picture size (<= 400 image tokens).

  --mode sync     every round all cars send at the same instant (worst case)
  --mode stagger  each car waits a random 0..--gap s between its requests (closer to real life)

Reports per request: wall time seen by the car, and for llama-server also how it splits into
queue wait (request accepted but not started), prompt (image encoding + prefill) and generation.

  python3 bench/concurrency.py --api openai --url http://127.0.0.1:8090 --cars 1 2
  python3 bench/concurrency.py --api ollama --url http://127.0.0.1:11434 --model qwen3-vl:2b-instruct --cars 1 2
"""
import argparse
import base64
import io
import json
import math
import os
import random
import statistics
import threading
import time
import urllib.request

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT_ANY = ('Detect every separate object lying on the floor or table surface in the image (toys, blocks, '
              'trash, tools, anything that is not the surface itself). Output ONLY JSON like '
              '[{"bbox_2d": [x1, y1, x2, y2], "label": "short name"}] with pixel coordinates. '
              'If the surface is empty, output [].')


def load_photos(folder, tokens):
    out = []
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".jpg"):
            continue
        im = Image.open(os.path.join(folder, name)).convert("RGB")
        w, h = im.size
        k = math.sqrt(tokens * 784 / (w * h))
        out.append(im.resize((max(1, int(w * k / 28)) * 28, max(1, int(h * k / 28)) * 28), Image.LANCZOS))
    return out


def fresh_jpeg(im):
    im = im.copy()
    for x in range(8):
        for y in range(8):
            im.putpixel((x, y), tuple(random.randrange(256) for _ in range(3)))
    b = io.BytesIO()
    im.save(b, "JPEG", quality=92)
    return base64.b64encode(b.getvalue()).decode()


def ask(a, b64):
    if a.api == "openai":
        body = {"model": a.model, "temperature": 0, "max_tokens": 200, "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
            {"type": "text", "text": PROMPT_ANY}]}]}
        path = "/v1/chat/completions"
    else:
        body = {"model": a.model, "stream": False, "keep_alive": -1,
                "messages": [{"role": "user", "content": PROMPT_ANY, "images": [b64]}],
                "options": {"temperature": 0, "num_predict": 200}}
        path = "/api/chat"
    req = urllib.request.Request(a.url.rstrip("/") + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=600).read())
    wall = (time.time() - t0) * 1000
    if a.api == "openai":
        tm = r["timings"]
        return dict(wall=wall, prompt=tm["prompt_ms"], gen=tm["predicted_ms"], gen_n=tm["predicted_n"],
                    cache=tm["cache_n"], wait=wall - tm["prompt_ms"] - tm["predicted_ms"])
    # Ollama: load / prompt eval / eval durations in ns
    p = r.get("prompt_eval_duration", 0) / 1e6
    g = r.get("eval_duration", 0) / 1e6
    return dict(wall=wall, prompt=p, gen=g, gen_n=r.get("eval_count", 0), cache=None, wait=wall - p - g)


def run(a, cars, photos):
    results = [[] for _ in range(cars)]
    barrier = threading.Barrier(cars)

    def car(i):
        rnd = random.Random(i)
        for k in range(a.rounds):
            b64 = fresh_jpeg(photos[(i * 7 + k) % len(photos)])
            if a.mode == "sync":
                barrier.wait()
            else:
                time.sleep(rnd.uniform(0, a.gap))
            t_start = time.time()
            r = ask(a, b64)
            r["start"], r["round"] = t_start, k
            results[i].append(r)

    ask(a, fresh_jpeg(photos[0]))           # warm up (model loaded, kernels compiled)
    ts = [threading.Thread(target=car, args=(i,)) for i in range(cars)]
    t0 = time.time()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    span = time.time() - t0
    allr = [r for rs in results for r in rs]
    return allr, span


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * q))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", default="openai", choices=["openai", "ollama"])
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--model", default="qwen2.5-vl-3b")
    ap.add_argument("--cars", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--rounds", type=int, default=8, help="requests per car")
    ap.add_argument("--mode", default="sync", choices=["sync", "stagger"])
    ap.add_argument("--gap", type=float, default=3.0, help="stagger: max random pause between requests (s)")
    ap.add_argument("--tokens", type=int, default=400)
    ap.add_argument("--label", default="")
    ap.add_argument("--json", help="append the summary to this JSON-lines file")
    a = ap.parse_args()
    photos = load_photos(os.path.join(HERE, "photos"), a.tokens)
    for n in a.cars:
        rs, span = run(a, n, photos)
        walls = [r["wall"] for r in rs]
        s = dict(label=a.label, api=a.api, model=a.model, cars=n, mode=a.mode, requests=len(rs),
                 wall_median=round(statistics.median(walls)), wall_p90=round(pct(walls, 0.9)),
                 wall_max=round(max(walls)),
                 prompt_median=round(statistics.median(r["prompt"] for r in rs)),
                 gen_median=round(statistics.median(r["gen"] for r in rs)),
                 wait_median=round(statistics.median(r["wait"] for r in rs)),
                 throughput_per_min=round(len(rs) / span * 60, 1),
                 cache_hits=sum(1 for r in rs if (r["cache"] or 0) > 100))   # >100: the picture itself, not the 15-token chat prefix
        print(f"[{a.label or a.model}] cars={n} mode={a.mode}: wall median {s['wall_median']} ms, p90 {s['wall_p90']}, "
              f"max {s['wall_max']} | prompt {s['prompt_median']} + gen {s['gen_median']} + wait/other {s['wait_median']} ms"
              f" | {s['throughput_per_min']} answers/min | cache hits {s['cache_hits']}")
        if a.mode == "sync" and n > 1:
            by_round = {}
            for r in rs:
                by_round.setdefault(r["round"], []).append(r["wall"])
            s["first_car_median"] = round(statistics.median(min(v) for v in by_round.values()))
            s["last_car_median"] = round(statistics.median(max(v) for v in by_round.values()))
            print(f"    per round: first answer {s['first_car_median']} ms, last answer {s['last_car_median']} ms (medians)")
        if a.json:
            with open(a.json, "a") as f:
                f.write(json.dumps(s) + "\n")


if __name__ == "__main__":
    main()
