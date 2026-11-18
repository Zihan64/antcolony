"""电脑 <-> 树莓派 通信：收视频帧、发控制指令。"""
import json
import time

import cv2
import numpy as np
import zmq


class RobotLink:
    def __init__(self, ip, video_port=5555, cmd_port=5556):
        self.ctx = zmq.Context.instance()

        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.setsockopt(zmq.CONFLATE, 1)     # 只保留最新一帧 —— 低延迟的关键
        self.sub.setsockopt(zmq.RCVHWM, 1)
        self.sub.setsockopt(zmq.SUBSCRIBE, b"")
        self.sub.connect(f"tcp://{ip}:{video_port}")

        self.req = self.ctx.socket(zmq.REQ)
        self.req.setsockopt(zmq.REQ_RELAXED, 1)  # 超时后可以继续发，不会卡死
        self.req.setsockopt(zmq.REQ_CORRELATE, 1)
        self.req.setsockopt(zmq.RCVTIMEO, 800)
        self.req.setsockopt(zmq.LINGER, 0)
        self.req.connect(f"tcp://{ip}:{cmd_port}")

    def frame(self, timeout_ms=1000):
        """返回 (BGR 图像, 接收时间)；超时返回 (None, None)。"""
        if self.sub.poll(timeout_ms) == 0:
            return None, None
        buf = self.sub.recv()
        img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        return img, time.time()

    def send(self, **cmd):
        try:
            self.req.send_string(json.dumps(cmd))
            return json.loads(self.req.recv())
        except zmq.Again:
            return {"ok": False, "error": "timeout"}

    def drive(self, left, right, ms=150):
        return self.send(cmd="drive", left=round(left, 3), right=round(right, 3), ms=ms)

    def stop(self):
        return self.send(cmd="stop")

    def arm(self, action):
        return self.send(cmd="arm", action=action)

    def wait_arm(self, timeout=15):
        t0 = time.time()
        time.sleep(0.2)
        while time.time() - t0 < timeout:
            st = self.send(cmd="status")
            if st.get("ok") and not st.get("arm_busy"):
                return True
            time.sleep(0.1)
        return False

    def ping(self):
        t0 = time.time()
        r = self.send(cmd="ping")
        return r.get("ok", False), (time.time() - t0) * 1000
