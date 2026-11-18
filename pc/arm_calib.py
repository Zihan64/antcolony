#!/usr/bin/env python3
"""机械臂标定：交互式设置每个关节角度，找到 home / reach / lift 姿态后填进 pi/robot_server.py 的 ARM_POSES。
用法: python arm_calib.py --pi 192.168.1.50
输入示例:  shoulder 90    |  grasp    |  home    |  q
"""
import argparse
import config as C
from robot_link import RobotLink

ap = argparse.ArgumentParser()
ap.add_argument("--pi", default=C.PI_IP)
link = RobotLink(ap.parse_args().pi)
print("关节: base shoulder elbow wrist gripper；动作: home reach lift grasp release open close")
while True:
    s = input("> ").strip().split()
    if not s:
        continue
    if s[0] == "q":
        break
    if len(s) == 2:
        print(link.send(cmd="servo", joint=s[0], angle=int(s[1])))
    else:
        print(link.arm(s[0]))
        link.wait_arm()
    print("当前角度:", link.send(cmd="status").get("angles"))
