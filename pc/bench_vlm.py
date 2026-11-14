#!/usr/bin/env python3
"""
在你自己的电脑上实测各个 VLM 的延迟和定位效果，选出最快且够准的那个。

  python bench_vlm.py --target "red cup"                       # 用 Mac 自带摄像头
  python bench_vlm.py --target "red cup" --pi 192.168.1.50     # 用小车摄像头
  python bench_vlm.py --target "red cup" --image test.jpg
  python bench_vlm.py --models qwen3-vl:2b-instruct qwen3-vl:4b-instruct
"""
import argparse
import statistics

import cv2

import config as C
from vlm import VLM

DEFAULT_MODELS = ["qwen3-vl:2b-instruct", "qwen3-vl:4b-instruct", "qwen3-vl:8b-instruct", "qwen2.5vl:3b"]


def get_frame(a):
    if a.image:
        return cv2.imread(a.image)
    if a.pi:
        from robot_link import RobotLink
        f, _ = RobotLink(a.pi).frame(3000)
        return f
    cap = cv2.VideoCapture(0)
    for _ in range(10):  # 丢掉前几帧，等自动曝光稳定
        ok, f = cap.read()
    cap.release()
    return f if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="red cup")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--image")
    ap.add_argument("--pi")
    ap.add_argument("--runs", type=int, default=5)
    a = ap.parse_args()

    frame = get_frame(a)
    if frame is None:
        raise SystemExit("拿不到图像")
    frame = cv2.resize(frame, (640, 480))

    rows = []
    for m in a.models:
        vlm = VLM(m, C.OLLAMA_URL, C.VLM_IMG_W, C.VLM_IMG_H, C.VLM_MAX_TOKENS)
        try:
            load = vlm.warmup(frame)
        except Exception as e:
            print(f"{m:26s} 跳过（{e}）—— 先运行: ollama pull {m}")
            continue
        times, box, raw = [], None, ""
        for _ in range(a.runs):
            box, dt, raw = vlm.locate(frame, a.target)
            times.append(dt * 1000)
        med = statistics.median(times)
        rows.append((m, med, box))
        print(f"{m:26s} 加载 {load:5.1f}s | 定位中位延迟 {med:6.0f} ms | box={box} | 原始输出: {raw[:80]!r}")

        vis = frame.copy()
        if box:
            cv2.rectangle(vis, box[:2], box[2:], (0, 255, 0), 2)
        cv2.putText(vis, f"{m} {med:.0f}ms", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(m, vis)
        cv2.waitKey(1)

    if rows:
        print("\n按速度排序：")
        for m, med, box in sorted(rows, key=lambda r: r[1]):
            print(f"  {m:26s} {med:6.0f} ms  {'找到' if box else '没找到'}")
        print("\n看弹出的窗口确认框是否框准，选『框得准里最快的』填进 config.py 的 VLM_MODEL。任意键退出。")
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
