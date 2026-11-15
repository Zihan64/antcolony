#!/usr/bin/env python3
"""
train_warehouse_yolo.py

Fine-tunes a COCO-pretrained YOLO26 detector on the warehouse objects
(datasets/warehouse_carpet, built by build_warehouse_dataset.py), evaluates it, and copies the
best weights to  cam_control/models/warehouse_yolo26.pt.

  python training/build_warehouse_dataset.py
  python training/train_warehouse_yolo.py --model yolo26s.pt --epochs 80

Evaluation (written to models/warehouse_yolo26_eval.json):
  * mAP50 / mAP50-95 / precision / recall on the held-out test split (real photos + test
    objects pasted on carpet; none of these objects were trained on);
  * the same on the 5 real test photos alone;
  * false detections on 200 held-out carpet-only images at several confidence thresholds:
    the number to watch for "does it fire on the carpet".
"""

import argparse
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "datasets" / "warehouse_carpet"
MODELS = ROOT / "models"


def evaluate(weights, imgsz):
    from ultralytics import YOLO
    import yaml
    m = YOLO(str(weights))
    out = {}
    r = m.val(data=str(DATA / "data.yaml"), split="test", imgsz=imgsz, device="cpu", plots=False,
              verbose=False, project=str(ROOT / "runs"), name="eval_test", exist_ok=True)
    out["test_all"] = dict(mAP50=r.box.map50, mAP50_95=r.box.map, precision=r.box.mp, recall=r.box.mr,
                           per_class_mAP50_95={m.names[i]: float(v) for i, v in zip(r.box.ap_class_index, r.box.maps[r.box.ap_class_index])})
    # the 5 real test photos alone (a tiny set: a sanity check, not a statistic)
    real = DATA / "_real_test"
    if real.exists():
        shutil.rmtree(real)
    for s in ("images", "labels"):
        (real / s / "test").mkdir(parents=True)
        for f in (DATA / s / "test").glob("IMG_*"):
            shutil.copy(f, real / s / "test" / f.name)
    cfg = yaml.safe_load(open(DATA / "data.yaml"))
    cfg.update(path=str(real), train="images/test", val="images/test", test="images/test")
    yaml.safe_dump(cfg, open(real / "data.yaml", "w"))
    r = m.val(data=str(real / "data.yaml"), split="test", imgsz=imgsz, device="cpu", plots=False,
              verbose=False, project=str(ROOT / "runs"), name="eval_real", exist_ok=True)
    out["test_real_photos"] = dict(mAP50=r.box.map50, mAP50_95=r.box.map, precision=r.box.mp, recall=r.box.mr)
    shutil.rmtree(real)
    # false positives on carpet that was never trained on
    negs = sorted((DATA / "images" / "test_carpet_neg").glob("*.jpg"))
    confs = []
    for i in range(0, len(negs), 16):
        for res in m.predict([str(p) for p in negs[i:i + 16]], imgsz=imgsz, conf=0.05, device="cpu", verbose=False):
            confs.append([float(c) for c in res.boxes.conf.cpu().numpy()])
    out["carpet_false_positives"] = {
        f"conf>={t}": dict(images_with_any=sum(any(c >= t for c in cs) for cs in confs),
                           total_boxes=sum(sum(c >= t for c in cs) for cs in confs), images=len(confs))
        for t in (0.1, 0.25, 0.4, 0.5, 0.6)}
    out["carpet_max_conf"] = max((max(cs) for cs in confs if cs), default=0.0)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(HERE / "yolo26n.pt"), help="pretrained start point (yolo26s.pt: ~2.5x slower on CPU)")
    # Fine-tuning, not training from scratch: the COCO weights are loaded (708/708 tensors) and
    # the backbone is frozen, so only the neck + detection head learn the 5 classes. On CPU this
    # cuts the time per epoch and the model converges in ~10 epochs (mAP50 0.99 by epoch 4).
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--freeze", type=int, default=10, help="freeze the first N layers (10 = backbone)")
    ap.add_argument("--imgsz", type=int, default=512, help="training size")
    ap.add_argument("--eval-imgsz", type=int, default=640, help="evaluation size (the GUI's default)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="warehouse_yolo26")
    ap.add_argument("--eval-only", metavar="WEIGHTS")
    ap.add_argument("--resume", action="store_true",
                    help="continue runs/<name>/weights/last.pt after an interruption (sleep, closed "
                         "terminal); no RAM cache, to spare memory. Then copy + evaluate as usual.")
    a = ap.parse_args()

    MODELS.mkdir(exist_ok=True)
    if a.eval_only:
        print(json.dumps(evaluate(a.eval_only, a.eval_imgsz), indent=2))
        return

    from ultralytics import YOLO
    if a.resume:
        YOLO(str(ROOT / "runs" / a.name / "weights" / "last.pt")).train(resume=True, cache=False, workers=2)
    else:
        train_new(YOLO(a.model), a)
    best = ROOT / "runs" / a.name / "weights" / "best.pt"
    dst = MODELS / "warehouse_yolo26.pt"
    shutil.copy(best, dst)
    print("best weights ->", dst)
    res = evaluate(dst, a.eval_imgsz)
    res.update(base_model=a.model, epochs=a.epochs, freeze=a.freeze, train_imgsz=a.imgsz, eval_imgsz=a.eval_imgsz, run_dir=str(ROOT / "runs" / a.name))
    (MODELS / "warehouse_yolo26_eval.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def train_new(model, a):
    model.train(
        data=str(DATA / "data.yaml"), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device="cpu",
        workers=4, project=str(ROOT / "runs"), name=a.name, exist_ok=True, seed=0, deterministic=False,
        freeze=a.freeze, cache="ram", patience=5, cos_lr=True, warmup_epochs=1,
        optimizer="AdamW", lr0=0.002,
        # overhead camera: objects lie either way up in the image. Large-angle rotation is NOT
        # done here: rotating a box's corners inflates it (loose boxes); the carpet composites
        # already hold every object at every angle with tight mask-derived boxes.
        degrees=10, flipud=0.5, fliplr=0.5, translate=0.15, scale=0.6,
        hsv_h=0.02, hsv_s=0.6, hsv_v=0.45, mosaic=1.0, close_mosaic=3, mixup=0.05,
        plots=False, verbose=True)


if __name__ == "__main__":
    main()
