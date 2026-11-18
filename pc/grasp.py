#!/usr/bin/env python3
"""
电脑端主程序：VLM 找目标 -> 视觉伺服开到抓取点 -> 树莓派执行抓取 -> VLM 确认

低延迟设计：
  * VLM 在后台线程异步运行（约 0.3~1.5 秒一次），只负责"认出并框出目标"和纠偏
  * 控制循环每帧用 OpenCV 跟踪器（几毫秒）更新目标位置，15Hz 给小车发指令
  * 视频只取最新帧（ZeroMQ CONFLATE），不会积压延迟

用法：
  python grasp.py --target "red cup"          # 自动抓取
  python grasp.py --target "red cup" --calib  # 标定：小车不动，只显示目标位置
  python grasp.py --view                      # 只看画面，测网络延迟
按键：q 退出   空格 暂停/继续（暂停时立刻停车）
"""
import argparse
import threading
import time

import cv2

import config as C
from robot_link import RobotLink
from vlm import VLM


# ---------------------------------------------------------------- 工具函数
def make_tracker():
    """优先 KCF（最快），其次 CSRT（更准），最后 MIL（OpenCV 自带）。"""
    for name in ("TrackerKCF_create", "TrackerCSRT_create"):
        if hasattr(cv2, name):
            return getattr(cv2, name)()
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, name):
            return getattr(cv2.legacy, name)()
    return cv2.TrackerMIL_create()


def clip(v, lo, hi):
    return max(lo, min(hi, v))


def deadband(v):
    if abs(v) < 0.05:
        return 0.0
    return v if abs(v) >= C.MIN_SPEED else (C.MIN_SPEED if v > 0 else -C.MIN_SPEED)


# ---------------------------------------------------------------- VLM 后台线程
class Detector(threading.Thread):
    """不停地拿"最新一帧"跑 VLM，结果带上该帧的拍摄时间，主循环自己判断新旧。"""

    def __init__(self, vlm, target):
        super().__init__(daemon=True)
        self.vlm, self.target = vlm, target
        self.cv = threading.Condition()
        self.pending = None   # (frame, t)
        self.result = None    # dict(box, frame, t, latency, seq)
        self.seq = 0
        self.enabled = True

    def submit(self, frame, t):
        with self.cv:
            self.pending = (frame, t)
            self.cv.notify()

    def run(self):
        while True:
            with self.cv:
                while self.pending is None or not self.enabled:
                    self.cv.wait(0.1)
                frame, t = self.pending
                self.pending = None
            try:
                box, dt, raw = self.vlm.locate(frame, self.target)
            except Exception as e:
                print(f"[vlm] 出错: {e}")
                time.sleep(0.5)
                continue
            self.seq += 1
            self.result = dict(box=box, frame=frame, t=t, latency=dt, seq=self.seq, raw=raw)


# ---------------------------------------------------------------- 主程序
def view_only(link):
    last = time.time()
    while True:
        frame, _ = link.frame()
        if frame is None:
            print("没收到画面，检查树莓派程序 / IP")
            continue
        ok, ms = link.ping()
        fps = 1 / max(1e-3, time.time() - last)
        last = time.time()
        cv2.putText(frame, f"{fps:.0f} fps  cmd RTT {ms:.0f} ms", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("robot", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def run(link, vlm, target, calib=False):
    frame, _ = link.frame(3000)
    if frame is None:
        raise SystemExit("收不到树莓派画面：先确认 robot_server.py 在运行、IP 正确、同一 Wi-Fi")
    H, W = frame.shape[:2]
    print(f"[vlm] 预热 {vlm.model} …")
    print(f"[vlm] 预热完成 {vlm.warmup(frame):.1f}s")

    det = Detector(vlm, target)
    det.start()

    state = "SEARCH"
    tracker, box = None, None
    last_seq, misses, stable = 0, 0, 0
    last_move_t = 0.0      # 最后一次让小车动的时间
    paused = False
    period = 1.0 / C.CONTROL_HZ
    vlm_ms = 0

    while True:
        loop_t = time.time()
        frame, ft = link.frame()
        if frame is None:
            link.stop()
            continue

        # 静止时才把画面交给 VLM（运动模糊 + 位置变化会让结果过时）；跟踪时持续交
        if state == "TRACK" or ft - last_move_t > C.SETTLE_S:
            det.submit(frame, ft)

        # ---------- 1. 处理新的 VLM 结果 ----------
        r = det.result
        if r and r["seq"] != last_seq:
            last_seq = r["seq"]
            vlm_ms = r["latency"] * 1000
            fresh = r["t"] > last_move_t + C.SETTLE_S or state == "TRACK"
            if r["box"] and fresh:
                misses = 0
                x1, y1, x2, y2 = r["box"]
                tracker = make_tracker()
                tracker.init(r["frame"], (x1, y1, x2 - x1, y2 - y1))  # 在 VLM 看的那帧上初始化
                ok, b = tracker.update(frame)                          # 再追到当前帧
                if ok:
                    box = tuple(int(v) for v in b)
                    if state == "SEARCH":
                        print(f"[vlm] 找到目标 ({vlm_ms:.0f} ms)")
                    state = "TRACK" if state in ("SEARCH", "TRACK") else state
            elif not r["box"] and fresh:
                misses += 1
                if state == "TRACK" and misses >= C.LOST_LIMIT:
                    print("[vlm] 目标丢失，重新搜索")
                    state, tracker, box = "SEARCH", None, None
                if state == "SEARCH" and not calib and not paused:
                    # 原地转一下再看
                    link.drive(C.SEARCH_TURN_SPEED, -C.SEARCH_TURN_SPEED, C.SEARCH_TURN_MS)
                    last_move_t = time.time() + C.SEARCH_TURN_MS / 1000

        # ---------- 2. 跟踪器每帧更新 ----------
        if state == "TRACK" and tracker is not None:
            ok, b = tracker.update(frame)
            if ok:
                box = tuple(int(v) for v in b)
            else:
                state, tracker, box = "SEARCH", None, None

        # ---------- 3. 控制 ----------
        info = ""
        if state == "TRACK" and box:
            x, y, w, h = box
            cx = (x + w / 2) / W
            bottom = (y + h) / H
            ex = cx - C.GRASP_X              # >0 目标偏右
            ey = C.GRASP_BOTTOM_Y - bottom   # >0 还没开到位
            info = f"cx={cx:.2f} bottom={bottom:.2f}"

            if abs(ex) < C.ALIGN_TOL_X and abs(ey) < C.ALIGN_TOL_Y:
                stable += 1
            else:
                stable = 0

            if calib or paused:
                pass
            elif stable >= C.STABLE_FRAMES:
                state = "GRASP"
            else:
                turn = clip(C.KP_TURN * ex, -C.MAX_TURN, C.MAX_TURN)
                fwd = clip(C.KP_FWD * ey, -C.MAX_FWD, C.MAX_FWD)
                fwd *= max(0.0, 1 - abs(ex) * 4)   # 偏得多就先转正再前进
                left, right = deadband(fwd + turn), deadband(fwd - turn)
                if left or right:
                    link.drive(left, right, C.PULSE_MS)
                    last_move_t = time.time()

        if state == "GRASP":
            link.stop()
            det.enabled = False
            print("[arm] 对准完成，开始抓取")
            link.arm("grasp")
            link.wait_arm()
            if C.VERIFY_AFTER_GRASP:
                f2, _ = link.frame()
                if f2 is not None:
                    f2, _ = link.frame()  # 再取一帧确保是最新的
                got = f2 is not None and vlm.yes_no(f2, f"Is the robot gripper holding the {target}?")
                print(f"[vlm] 抓取{'成功' if got else '失败'}")
                if not got:
                    link.arm("open")
                    link.wait_arm()
                    link.arm("home")
                    link.wait_arm()
                    state, tracker, box, stable = "SEARCH", None, None, 0
                    det.enabled = True
                    continue
            state = "DONE"
            print("完成！按 q 退出")

        # ---------- 4. 显示 ----------
        vis = frame.copy()
        gx, gy = int(C.GRASP_X * W), int(C.GRASP_BOTTOM_Y * H)
        cv2.drawMarker(vis, (gx, gy), (255, 0, 255), cv2.MARKER_CROSS, 30, 2)
        if box:
            x, y, w, h = box
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        loop_ms = (time.time() - loop_t) * 1000
        txt = f"{state}{' PAUSED' if paused else ''} | VLM {vlm_ms:.0f}ms | loop {loop_ms:.0f}ms | {info}"
        cv2.putText(vis, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.imshow("robot-vlm", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused
            link.stop()

        time.sleep(max(0, period - (time.time() - loop_t)))

    link.stop()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="red cup", help='要抓的物体，例如 "red cup" / "blue block"')
    ap.add_argument("--pi", default=C.PI_IP)
    ap.add_argument("--model", default=C.VLM_MODEL)
    ap.add_argument("--calib", action="store_true", help="标定模式：小车不动")
    ap.add_argument("--view", action="store_true", help="只看画面和延迟")
    a = ap.parse_args()

    link = RobotLink(a.pi, C.VIDEO_PORT, C.CMD_PORT)
    ok, ms = link.ping()
    print(f"[link] 树莓派 {'在线' if ok else '无响应'}，指令往返 {ms:.0f} ms")
    if a.view:
        view_only(link)
        return
    vlm = VLM(a.model, C.OLLAMA_URL, C.VLM_IMG_W, C.VLM_IMG_H, C.VLM_MAX_TOKENS)
    try:
        run(link, vlm, a.target, a.calib)
    finally:
        link.stop()


if __name__ == "__main__":
    main()
