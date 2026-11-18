"""
本地 VLM 封装（通过 Ollama 调用）。
主要功能：locate(frame, "红色杯子") -> 物体在原图中的像素框 (x1, y1, x2, y2) 或 None

坐标格式：
  qwen3-vl  输出 0~1000 的相对坐标
  qwen2.5vl 输出相对于输入图像的绝对像素坐标
"""
import base64
import json
import re
import time

import cv2
import requests


class VLM:
    def __init__(self, model, url="http://127.0.0.1:11434", img_w=448, img_h=336, max_tokens=96):
        self.model = model
        self.url = url.rstrip("/") + "/api/chat"
        self.img_w, self.img_h = img_w, img_h
        self.max_tokens = max_tokens
        self.coord_mode = "abs" if model.startswith("qwen2.5vl") else "rel1000"
        if self.coord_mode == "abs":
            # Qwen2.5-VL 的 patch 是 28 像素，把尺寸对齐到 28 的倍数，避免内部再缩放导致坐标偏移
            self.img_w = max(28, round(img_w / 28) * 28)
            self.img_h = max(28, round(img_h / 28) * 28)
        self.http = requests.Session()  # 复用 TCP 连接

    # ---------- 底层 ----------
    def _encode(self, frame):
        small = cv2.resize(frame, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf.tobytes()).decode()

    def chat(self, frame, prompt, max_tokens=None):
        body = {
            "model": self.model,
            "stream": False,
            "keep_alive": -1,  # 模型常驻显存，不会每次重新加载（否则首帧要好几秒）
            "messages": [{"role": "user", "content": prompt, "images": [self._encode(frame)]}],
            "options": {"temperature": 0, "num_predict": max_tokens or self.max_tokens},
        }
        r = self.http.post(self.url, json=body, timeout=60)
        r.raise_for_status()
        text = r.json()["message"]["content"]
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()

    def warmup(self, frame):
        t0 = time.time()
        self.chat(frame, "Describe the image in 3 words.", max_tokens=8)
        return time.time() - t0

    # ---------- 目标定位 ----------
    def locate(self, frame, target):
        """返回 (bbox 或 None, 耗时秒, 原始输出)。bbox 是原图像素坐标 (x1,y1,x2,y2)。"""
        prompt = (
            f'Locate the "{target}" in the image. '
            f'Output ONLY JSON: [{{"bbox_2d": [x1, y1, x2, y2], "label": "{target}"}}]. '
            "If it is not in the image, output []."
        )
        t0 = time.time()
        text = self.chat(frame, prompt)
        dt = time.time() - t0
        box = self._parse_box(text)
        if box is None:
            return None, dt, text

        x1, y1, x2, y2 = box
        H, W = frame.shape[:2]
        if self.coord_mode == "rel1000":
            sx, sy = W / 1000.0, H / 1000.0
        else:
            sx, sy = W / self.img_w, H / self.img_h
        x1, x2 = sorted((x1 * sx, x2 * sx))
        y1, y2 = sorted((y1 * sy, y2 * sy))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W - 1, x2), min(H - 1, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None, dt, text
        return (int(x1), int(y1), int(x2), int(y2)), dt, text

    def yes_no(self, frame, question):
        text = self.chat(frame, question + " Answer only yes or no.", max_tokens=4)
        return text.lower().startswith("yes")

    @staticmethod
    def _parse_box(text):
        # 先尝试标准 JSON
        m = re.search(r"\[.*\]|\{.*\}", text, re.S)
        if m:
            try:
                data = json.loads(m.group(0))
                if isinstance(data, dict):
                    data = [data]
                if isinstance(data, list):
                    if not data:
                        return None
                    item = data[0]
                    if isinstance(item, dict):
                        b = item.get("bbox_2d") or item.get("bbox") or item.get("box")
                        if b and len(b) == 4:
                            return [float(v) for v in b]
                    elif len(data) == 4 and all(isinstance(v, (int, float)) for v in data):
                        return [float(v) for v in data]
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        # 兜底：找 4 个连续数字
        nums = re.findall(r"-?\d+(?:\.\d+)?", text)
        if len(nums) >= 4:
            return [float(v) for v in nums[:4]]
        return None
