#!/usr/bin/env python3
"""
build_warehouse_dataset.py

Builds the training set for the 5-class warehouse detector (bracket, knob, slotted_block, cup,
holder) from datasets/yolo26_warehouse (the photos + labels from yolo_dataset.csv), and adds
what those 30 countertop photos lack: carpet.

The photos were all taken on a white marble counter. A detector trained on them alone has never
seen carpet, so speckle, fibers and woven patterns on the arena floor are "unknown" and can
fire as objects. This script adds:
  * carpet negatives      procedural carpet / rug textures with EMPTY label files, so the
                          model learns that carpet texture is background;
  * objects on carpet     each labeled object is cut out of its photo (GrabCut inside its box)
                          and pasted onto carpet at overhead-camera scales and any rotation;
                          the label is the tight box of the pasted mask (no padding);
  * countertop negatives  crops of the original photos away from every box (veins, shadows).
Splits never mix: val composites use only val objects, test composites only test objects.

  python training/build_warehouse_dataset.py            -> datasets/warehouse_carpet/
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / "datasets" / "yolo26_warehouse"
OUT = ROOT / "datasets" / "warehouse_carpet"
NAMES = {0: "bracket", 1: "knob", 2: "slotted_block", 3: "cup", 4: "holder"}

# carpet base colours (BGR): beige, cream, greys, taupe, brown, blue-grey, navy, green-grey, red rug
CARPET_COLOURS = [(150, 170, 185), (190, 205, 215), (120, 120, 120), (90, 92, 95), (160, 160, 158),
                  (110, 125, 140), (70, 85, 105), (130, 115, 95), (85, 65, 50), (110, 120, 110),
                  (55, 50, 120), (200, 200, 200), (60, 60, 62)]


# ----------------------------------------------------------------------------- carpet textures
def _noise(rng, h, w, sigma):
    n = rng.standard_normal((h, w)).astype(np.float32)
    if sigma > 0:
        n = cv2.GaussianBlur(n, (0, 0), sigma)
        n /= n.std() + 1e-6
    return n


def carpet(rng, h, w):
    """One random carpet / rug texture, uint8 BGR (h, w)."""
    base = np.array(CARPET_COLOURS[rng.integers(len(CARPET_COLOURS))], np.float32)
    base = np.clip(base * rng.uniform(0.8, 1.2) + rng.normal(0, 8, 3), 10, 245)
    img = np.ones((h, w, 3), np.float32) * base
    lum = np.zeros((h, w), np.float32)
    # multi-scale speckle: single fibres, tufts, clumps
    lum += rng.uniform(6, 22) * _noise(rng, h, w, 0)
    lum += rng.uniform(5, 18) * _noise(rng, h, w, rng.uniform(0.8, 1.6))
    lum += rng.uniform(3, 12) * _noise(rng, h, w, rng.uniform(3, 8))
    kind = rng.choice(["plain", "fibres", "berber", "stripes", "diamond", "flecks", "blotch"],
                      p=[0.15, 0.2, 0.2, 0.12, 0.12, 0.11, 0.1])
    if kind == "fibres":                  # short directional streaks
        streak = np.zeros((h, w), np.float32)
        ang0 = rng.uniform(0, np.pi)
        for _ in range(int(h * w / rng.uniform(60, 200))):
            x, y = rng.uniform(0, w), rng.uniform(0, h)
            a = ang0 + rng.normal(0, 0.5)
            L = rng.uniform(3, 12)
            cv2.line(streak, (int(x), int(y)), (int(x + L * np.cos(a)), int(y + L * np.sin(a))),
                     float(rng.choice([-1, 1])), 1)
        lum += rng.uniform(10, 25) * cv2.GaussianBlur(streak, (0, 0), 0.6)
    elif kind == "berber":                # loop pile: a grid of small bumps
        p = rng.uniform(4, 10)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        a = rng.uniform(0, np.pi)
        u, v = xx * np.cos(a) + yy * np.sin(a), -xx * np.sin(a) + yy * np.cos(a)
        loops = np.sin(2 * np.pi * u / p) * np.sin(2 * np.pi * v / (p * rng.uniform(0.7, 1.4)))
        lum += rng.uniform(10, 30) * loops * (0.6 + 0.4 * _noise(rng, h, w, 4))
    elif kind == "stripes":
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        a = rng.uniform(0, np.pi)
        s = np.sin(2 * np.pi * (xx * np.cos(a) + yy * np.sin(a)) / rng.uniform(15, 80))
        lum += rng.uniform(10, 35) * np.sign(s) * np.abs(s) ** 0.3
    elif kind == "diamond":               # woven rug motif
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        p = rng.uniform(25, 90)
        d = np.abs(((xx + yy) % p) - p / 2) + np.abs(((xx - yy) % p) - p / 2)
        lum += rng.uniform(15, 40) * (d < p * rng.uniform(0.2, 0.45)).astype(np.float32)
        if rng.random() < 0.5:            # second colour in the motif
            col = np.array(CARPET_COLOURS[rng.integers(len(CARPET_COLOURS))], np.float32)
            m = (d < p * 0.15)[..., None]
            img = np.where(m, 0.5 * img + 0.5 * col, img)
    elif kind == "flecks":                # multi-coloured berber flecks
        for _ in range(int(h * w / rng.uniform(150, 600))):
            col = np.array(CARPET_COLOURS[rng.integers(len(CARPET_COLOURS))], np.float32)
            cv2.circle(img, (int(rng.uniform(0, w)), int(rng.uniform(0, h))),
                       int(rng.integers(1, 3)), col.tolist(), -1)
    elif kind == "blotch":                # stains / worn patches
        lum += rng.uniform(10, 30) * np.tanh(2 * _noise(rng, h, w, rng.uniform(15, 40)))
    img += lum[..., None] * rng.uniform(0.7, 1.0, 3)[None, None, :]
    # uneven lighting: gradient + soft vignette
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g = 1 + rng.uniform(-0.25, 0.25) * (xx / w - 0.5) + rng.uniform(-0.25, 0.25) * (yy / h - 0.5)
    g *= 1 - rng.uniform(0, 0.3) * (((xx / w - 0.5) ** 2 + (yy / h - 0.5) ** 2) * 2)
    img *= g[..., None]
    img = np.clip(img, 0, 255).astype(np.uint8)
    return perspective(rng, img) if rng.random() < 0.6 else img


def perspective(rng, img):
    """Tilt the texture as an overhead camera at an angle would see it (far side denser)."""
    h, w = img.shape[:2]
    s = rng.uniform(0.05, 0.3)
    big = cv2.resize(img, (int(w * (1 + 2 * s)), h))
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [big.shape[1], 0], [big.shape[1], h], [0, h]]),
                                    np.float32([[s * w, 0], [w - s * w, 0], [w + s * w, h], [-s * w, h]]))
    return cv2.warpPerspective(big, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def camera_effects(rng, img):
    """Blur, sensor noise and JPEG artefacts of a cheap webcam."""
    if rng.random() < 0.5:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.2))
    if rng.random() < 0.5:
        img = np.clip(img + rng.normal(0, rng.uniform(1, 5), img.shape), 0, 255).astype(np.uint8)
    q = int(rng.uniform(55, 95))
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)


# ----------------------------------------------------------------------------- object cut-outs
def read_labels(p):
    if not p.exists():
        return []
    return [(int(a[0]), *map(float, a[1:5])) for a in (l.split() for l in p.read_text().splitlines()) if a]


def cutouts(split):
    """GrabCut every labeled object of a split out of its photo -> list of (cls, BGR, mask)."""
    out = []
    for ip in sorted((SRC / "images" / split).glob("*.jpeg")):
        img = cv2.imread(str(ip))
        H, W = img.shape[:2]
        for c, xc, yc, bw, bh in read_labels(SRC / "labels" / split / (ip.stem + ".txt")):
            x1, y1 = int((xc - bw / 2) * W), int((yc - bh / 2) * H)
            x2, y2 = int((xc + bw / 2) * W), int((yc + bh / 2) * H)
            if (x2 - x1) < 40 or (y2 - y1) < 40 or y1 <= 2:      # truncated slivers at the edge
                continue
            pad = int(0.08 * max(x2 - x1, y2 - y1)) + 4
            X1, Y1, X2, Y2 = max(0, x1 - pad), max(0, y1 - pad), min(W, x2 + pad), min(H, y2 + pad)
            crop = img[Y1:Y2, X1:X2].copy()
            mask = np.zeros(crop.shape[:2], np.uint8)
            rect = (x1 - X1, y1 - Y1, x2 - x1, y2 - y1)
            bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
            try:
                cv2.grabCut(crop, mask, rect, bgd, fgd, 6, cv2.GC_INIT_WITH_RECT)
                m = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
            except cv2.error:
                m = np.zeros(crop.shape[:2], np.uint8)
            # keep the largest blob; GrabCut failing (tiny mask) -> fall back to the labeled box
            n, lab, st, _ = cv2.connectedComponentsWithStats(m)
            if n > 1:
                k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
                m = np.where(lab == k, 255, 0).astype(np.uint8)
            if m.sum() / 255 < 0.25 * rect[2] * rect[3]:
                m = np.zeros_like(m)
                m[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = 255
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
            out.append((c, crop, m, ip.stem))
    return out


def paste(rng, canvas, obj, boxes, scale_px):
    """Paste one cut-out (random rotation / scale / colour jitter) with a soft edge and a contact
    shadow. Returns the tight box (x1, y1, x2, y2) of the pasted mask, or None if it overlaps."""
    c, crop, m, _ = obj
    H, W = canvas.shape[:2]
    s = scale_px / max(crop.shape[:2])
    crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    m = cv2.resize(m, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LINEAR)
    # rotate about the centre on an enlarged canvas so nothing is clipped
    h, w = crop.shape[:2]
    a = rng.uniform(0, 360)
    R = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
    cos, sin = abs(R[0, 0]), abs(R[0, 1])
    nw, nh = int(h * sin + w * cos) + 2, int(h * cos + w * sin) + 2
    R[0, 2] += nw / 2 - w / 2
    R[1, 2] += nh / 2 - h / 2
    crop = cv2.warpAffine(crop, R, (nw, nh), flags=cv2.INTER_LINEAR)
    m = cv2.warpAffine(m, R, (nw, nh), flags=cv2.INTER_LINEAR)
    if rng.random() < 0.5:
        crop, m = crop[:, ::-1], m[:, ::-1]
    ys, xs = np.nonzero(m > 127)
    if len(xs) < 20 or nw >= W or nh >= H:
        return None
    # colour / brightness jitter of the object (lighting differs between rooms)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-4, 4)) % 180
    hsv[..., 1] *= rng.uniform(0.8, 1.15)
    hsv[..., 2] *= rng.uniform(0.7, 1.2)
    crop = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    for _ in range(30):
        ox, oy = int(rng.uniform(0, W - nw)), int(rng.uniform(0, H - nh))
        b = (ox + xs.min(), oy + ys.min(), ox + xs.max() + 1, oy + ys.max() + 1)
        if all(b[2] < q[0] - 4 or b[0] > q[2] + 4 or b[3] < q[1] - 4 or b[1] > q[3] + 4 for q in boxes):
            break
    else:
        return None
    roi = canvas[oy:oy + nh, ox:ox + nw].astype(np.float32)
    # soft contact shadow below / around the object
    sh = cv2.GaussianBlur(m, (0, 0), max(1.5, 0.06 * scale_px)).astype(np.float32) / 255
    sh = np.roll(sh, (int(0.04 * scale_px), int(0.02 * scale_px)), (0, 1))
    roi *= 1 - rng.uniform(0.15, 0.4) * sh[..., None]
    alpha = cv2.GaussianBlur(m, (0, 0), 0.8).astype(np.float32)[..., None] / 255
    roi = roi * (1 - alpha) + crop.astype(np.float32) * alpha
    canvas[oy:oy + nh, ox:ox + nw] = np.clip(roi, 0, 255).astype(np.uint8)
    return b


def composite(rng, objs, size):
    W, H = size
    img = carpet(rng, H, W)
    boxes, labels = [], []
    k = int(rng.choice([1, 2, 3, 4, 5], p=[0.25, 0.3, 0.25, 0.12, 0.08]))
    for i in rng.permutation(len(objs))[:k]:
        # overhead camera: objects span ~5-35 % of the short side
        b = paste(rng, img, objs[i], boxes, rng.uniform(0.07, 0.4) * min(W, H))
        if b is not None:
            boxes.append(b)
            labels.append(objs[i][0])
    return camera_effects(rng, img), [(c, *b) for c, b in zip(labels, boxes)]


def counter_negative(rng, split):
    """A crop of an original photo that contains no labeled object (marble, veins, shadow)."""
    ims = sorted((SRC / "images" / split).glob("*.jpeg"))
    for _ in range(50):
        ip = ims[rng.integers(len(ims))]
        img = cv2.imread(str(ip))
        H, W = img.shape[:2]
        bx = [((xc - bw / 2) * W, (yc - bh / 2) * H, (xc + bw / 2) * W, (yc + bh / 2) * H)
              for _, xc, yc, bw, bh in read_labels(SRC / "labels" / split / (ip.stem + ".txt"))]
        s = int(rng.uniform(0.3, 0.55) * min(W, H))
        x, y = int(rng.uniform(0, W - s)), int(rng.uniform(0, H - s))
        if all(x + s < b[0] or x > b[2] or y + s < b[1] or y > b[3] for b in bx):
            return cv2.resize(img[y:y + s, x:x + s], (640, 640))
    return None


# ----------------------------------------------------------------------------- writing
def write(split, name, img, boxes):
    H, W = img.shape[:2]
    cv2.imwrite(str(OUT / "images" / split / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    lines = [f"{c} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}"
             for c, x1, y1, x2, y2 in boxes]
    (OUT / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-composites", type=int, default=260)
    ap.add_argument("--train-carpet-neg", type=int, default=140)
    ap.add_argument("--train-counter-neg", type=int, default=30)
    ap.add_argument("--orig-repeat", type=int, default=4, help="copies of each real train photo")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if OUT.exists():
        shutil.rmtree(OUT)
    for s in ("train", "val", "test", "test_carpet_neg"):
        (OUT / "images" / s).mkdir(parents=True)
        (OUT / "labels" / s).mkdir(parents=True)

    plan = {  # split: (composites, carpet negatives, counter negatives, seed offset)
        "train": (a.train_composites, a.train_carpet_neg, a.train_counter_neg, 0),
        "val": (60, 40, 5, 1000),
        "test": (60, 0, 0, 2000),
    }
    for split, (n_comp, n_neg, n_cneg, off) in plan.items():
        rng = np.random.default_rng(a.seed + off)
        # 1. the real photos (repeated in train so the composites do not drown them out)
        for ip in sorted((SRC / "images" / split).glob("*.jpeg")):
            img = cv2.imread(str(ip))
            H, W = img.shape[:2]
            bx = [(c, (xc - bw / 2) * W, (yc - bh / 2) * H, (xc + bw / 2) * W, (yc + bh / 2) * H)
                  for c, xc, yc, bw, bh in read_labels(SRC / "labels" / split / (ip.stem + ".txt"))]
            for r in range(a.orig_repeat if split == "train" else 1):
                write(split, f"{ip.stem}_r{r}", img, bx)
        # 2. objects of THIS split pasted on carpet
        objs = cutouts(split)
        print(f"{split}: {len(objs)} object cut-outs")
        for i in range(n_comp):
            size = (640, 640) if rng.random() < 0.5 else ((800, 600) if rng.random() < 0.5 else (600, 800))
            img, bx = composite(rng, objs, size)
            write(split, f"carpet_obj_{i:04d}", img, bx)
        # 3. pure carpet and countertop negatives (empty label files)
        for i in range(n_neg):
            write(split, f"carpet_neg_{i:04d}", camera_effects(rng, carpet(rng, 640, 640)), [])
        for i in range(n_cneg):
            im = counter_negative(rng, split)
            if im is not None:
                write(split, f"counter_neg_{i:04d}", im, [])
    # held-out carpet-only images, never trained on: measures false detections on the floor
    rng = np.random.default_rng(a.seed + 3000)
    for i in range(200):
        sz = [(640, 640), (800, 600), (1280, 720)][i % 3]
        write("test_carpet_neg", f"carpet_hold_{i:04d}", camera_effects(rng, carpet(rng, sz[1], sz[0])), [])

    yaml.safe_dump({"path": str(OUT), "train": "images/train", "val": "images/val", "test": "images/test",
                    "names": NAMES}, open(OUT / "data.yaml", "w"), sort_keys=False)
    for s in ("train", "val", "test", "test_carpet_neg"):
        print(s, len(list((OUT / "images" / s).iterdir())), "images")
    print("wrote", OUT / "data.yaml")


if __name__ == "__main__":
    main()
