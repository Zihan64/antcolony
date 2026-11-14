#!/usr/bin/env python3
"""
树莓派端（小车上运行）
  1. 摄像头 -> JPEG -> ZeroMQ PUB 推流给电脑（端口 5555，只保留最新一帧，低延迟）
  2. ZeroMQ REP 接收电脑的 JSON 指令（端口 5556），控制底盘电机和机械臂
  3. 看门狗：一段时间收不到行驶指令自动停车（网络断了也不会乱跑）

机械臂的抓取动作序列在树莓派本地执行，电脑只发一个 "grasp"，避免多次网络往返。
没装 gpiozero / adafruit-servokit 时会进入「模拟模式」，只打印不动作，方便先调通网络。
"""
import json
import threading
import time

import cv2
import zmq

# ===================== 按你的硬件修改这里 =====================
CAM_W, CAM_H, CAM_FPS = 640, 480, 30
JPEG_QUALITY = 70                # 60~80 之间；越低越快、画质越差
VIDEO_PORT, CMD_PORT = 5555, 5556
DRIVE_WATCHDOG_S = 0.6           # 超过这个时间没有新的行驶指令 -> 自动停车

# 电机驱动板（L298N / TB6612 等）GPIO 引脚，BCM 编号
LEFT_MOTOR = dict(forward=17, backward=27, enable=12)
RIGHT_MOTOR = dict(forward=23, backward=24, enable=13)

# 机械臂舵机：PCA9685 舵机板的通道号
ARM_CHANNELS = {"base": 0, "shoulder": 1, "elbow": 2, "wrist": 3, "gripper": 4}
GRIPPER_OPEN, GRIPPER_CLOSE = 30, 110   # 夹爪张开 / 闭合角度，务必实测

# 预设姿态（角度，0~180）。第一次用一定要逐个关节小心标定！
ARM_POSES = {
    "home":  {"base": 90, "shoulder": 150, "elbow": 30, "wrist": 90},   # 收起，不挡摄像头
    "reach": {"base": 90, "shoulder": 70,  "elbow": 110, "wrist": 60},  # 伸到车前方地面抓取点
    "lift":  {"base": 90, "shoulder": 130, "elbow": 60, "wrist": 90},   # 抓起后抬高
}
SERVO_STEP_DEG, SERVO_STEP_S = 3, 0.015  # 舵机平滑运动：每步角度 / 每步间隔
# =============================================================


class Camera:
    """优先用 CSI 摄像头 (picamera2)，失败则用 USB 摄像头 (OpenCV)。"""

    def __init__(self):
        self.picam = None
        self.cap = None
        try:
            from picamera2 import Picamera2
            self.picam = Picamera2()
            cfg = self.picam.create_video_configuration(
                main={"size": (CAM_W, CAM_H), "format": "RGB888"},  # RGB888 实际是 BGR 排列，正好给 OpenCV
                controls={"FrameRate": CAM_FPS},
                buffer_count=2,
            )
            self.picam.configure(cfg)
            self.picam.start()
            print("[camera] 使用 CSI 摄像头 (picamera2)")
        except Exception as e:
            print(f"[camera] picamera2 不可用 ({e})，改用 USB 摄像头")
            self.picam = None
            self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
            self.cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 不攒旧帧
            if not self.cap.isOpened():
                raise RuntimeError("打不开摄像头，检查连线 / 是否启用摄像头")

    def read(self):
        if self.picam is not None:
            return self.picam.capture_array()
        ok, frame = self.cap.read()
        return frame if ok else None


class Drive:
    """差速底盘。left/right 速度范围 -1.0 ~ 1.0。"""

    def __init__(self):
        self.robot = None
        try:
            from gpiozero import Motor, Robot
            self.robot = Robot(left=Motor(**LEFT_MOTOR), right=Motor(**RIGHT_MOTOR))
            print("[drive] gpiozero 电机已就绪")
        except Exception as e:
            print(f"[drive] 模拟模式（{e}）")
        self.stop_at = 0.0
        self.lock = threading.Lock()
        threading.Thread(target=self._watchdog, daemon=True).start()

    def set(self, left, right, ms):
        left = max(-1.0, min(1.0, float(left)))
        right = max(-1.0, min(1.0, float(right)))
        ms = max(0, min(int(ms), 1000))  # 单条指令最多跑 1 秒
        with self.lock:
            if self.robot:
                self.robot.value = (left, right)
            else:
                print(f"[drive] L={left:+.2f} R={right:+.2f} {ms}ms")
            self.stop_at = time.time() + ms / 1000.0

    def stop(self):
        with self.lock:
            if self.robot:
                self.robot.stop()
            self.stop_at = 0.0

    def _watchdog(self):
        while True:
            time.sleep(0.02)
            if self.stop_at and time.time() > self.stop_at:
                self.stop()


class Arm:
    def __init__(self):
        self.kit = None
        try:
            from adafruit_servokit import ServoKit
            self.kit = ServoKit(channels=16)
            print("[arm] PCA9685 舵机板已就绪")
        except Exception as e:
            print(f"[arm] 模拟模式（{e}）")
        self.angles = {}
        self.busy = False
        self.lock = threading.Lock()

    def _write(self, joint, angle):
        angle = max(0, min(180, angle))
        if self.kit:
            self.kit.servo[ARM_CHANNELS[joint]].angle = angle
        self.angles[joint] = angle

    def move_joint(self, joint, target):
        cur = self.angles.get(joint)
        if cur is None:  # 第一次没有当前角度，直接写
            self._write(joint, target)
            return
        step = SERVO_STEP_DEG if target > cur else -SERVO_STEP_DEG
        for a in range(int(cur), int(target), step):
            self._write(joint, a)
            time.sleep(SERVO_STEP_S)
        self._write(joint, target)

    def move_pose(self, name):
        pose = ARM_POSES[name]
        # 各关节并行运动，更快
        ts = [threading.Thread(target=self.move_joint, args=(j, a)) for j, a in pose.items()]
        for t in ts:
            t.start()
        for t in ts:
            t.join()

    def gripper(self, closed):
        self.move_joint("gripper", GRIPPER_CLOSE if closed else GRIPPER_OPEN)

    def run_action(self, action):
        """在后台线程执行，命令立即返回，电脑用 status 轮询 busy。"""
        if self.busy:
            return False

        def job():
            with self.lock:
                self.busy = True
                try:
                    print(f"[arm] 执行 {action}")
                    if action == "grasp":
                        self.gripper(False)
                        self.move_pose("reach")
                        time.sleep(0.2)
                        self.gripper(True)
                        time.sleep(0.3)
                        self.move_pose("lift")
                    elif action == "release":
                        self.move_pose("reach")
                        self.gripper(False)
                        self.move_pose("home")
                    elif action == "open":
                        self.gripper(False)
                    elif action == "close":
                        self.gripper(True)
                    elif action in ARM_POSES:
                        self.move_pose(action)
                finally:
                    self.busy = False

        threading.Thread(target=job, daemon=True).start()
        return True


def stream_loop(cam, ctx):
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.SNDHWM, 1)     # 队列只留 1 帧，网络慢时丢旧帧而不是积压
    pub.setsockopt(zmq.LINGER, 0)
    pub.bind(f"tcp://*:{VIDEO_PORT}")
    params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    n, t0 = 0, time.time()
    while True:
        frame = cam.read()
        if frame is None:
            time.sleep(0.01)
            continue
        ok, buf = cv2.imencode(".jpg", frame, params)
        if ok:
            pub.send(buf.tobytes(), zmq.NOBLOCK, copy=False)
        n += 1
        if time.time() - t0 > 5:
            print(f"[camera] {n / (time.time() - t0):.1f} fps")
            n, t0 = 0, time.time()


def main():
    ctx = zmq.Context()
    cam = Camera()
    drive = Drive()
    arm = Arm()
    arm.run_action("home")
    arm.run_action("open")

    threading.Thread(target=stream_loop, args=(cam, ctx), daemon=True).start()

    rep = ctx.socket(zmq.REP)
    rep.bind(f"tcp://*:{CMD_PORT}")
    print(f"[server] 视频端口 {VIDEO_PORT}，指令端口 {CMD_PORT}，等待电脑连接…")

    while True:
        try:
            msg = json.loads(rep.recv())
            cmd = msg.get("cmd")
            if cmd == "drive":
                drive.set(msg.get("left", 0), msg.get("right", 0), msg.get("ms", 150))
                reply = {"ok": True}
            elif cmd == "stop":
                drive.stop()
                reply = {"ok": True}
            elif cmd == "arm":
                reply = {"ok": arm.run_action(msg["action"])}
            elif cmd == "servo":  # 标定用：直接设置某个关节角度
                arm._write(msg["joint"], int(msg["angle"]))
                reply = {"ok": True}
            elif cmd == "status":
                reply = {"ok": True, "arm_busy": arm.busy, "angles": arm.angles}
            elif cmd == "ping":
                reply = {"ok": True, "t": time.time()}
            else:
                reply = {"ok": False, "error": f"unknown cmd {cmd}"}
        except Exception as e:
            reply = {"ok": False, "error": str(e)}
        rep.send_string(json.dumps(reply))


if __name__ == "__main__":
    main()
