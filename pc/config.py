"""电脑端配置。改这里就行，也可以用命令行参数覆盖。"""

PI_IP = "192.168.50.93"          # 树莓派的 IP（在树莓派上运行 hostname -I 查看）
VIDEO_PORT, CMD_PORT = 5555, 5556

OLLAMA_URL = "http://127.0.0.1:11434"
# 推荐顺序（M2 16GB）：
#   qwen3-vl:4b-instruct  —— 默认，速度和定位精度最平衡
#   qwen3-vl:2b-instruct  —— 最快，简单物体够用
#   qwen3-vl:8b-instruct  —— 最准，延迟约翻倍
#   qwen2.5vl:3b          —— 备选，Ollama 老版本也能跑
VLM_MODEL = "qwen3-vl:2b-instruct"   # 实测 M2: 2b ≈ 0.75s, 4b ≈ 1.5s；框不准时换回 4b

# 送进 VLM 的图像尺寸：越小越快。448 宽对抓取定位一般够用
VLM_IMG_W, VLM_IMG_H = 448, 336
VLM_MAX_TOKENS = 96

# ---------------- 控制参数（需要在你的小车上微调） ----------------
CONTROL_HZ = 15                 # 控制循环频率
PULSE_MS = 150                  # 每条行驶指令持续时间（> 1/CONTROL_HZ，保证连续）
MIN_SPEED = 0.35                # 电机死区：小于这个速度轮子转不动
MAX_TURN = 0.6
MAX_FWD = 0.6
KP_TURN = 1.2                   # 转向比例系数
KP_FWD = 2.5                    # 前进比例系数

# 抓取点在画面中的位置（归一化 0~1）。机械臂 reach 姿态能够到的那个点：
#   把物体放在夹爪正下方，运行 grasp.py --calib 查看物体中心 x、底边 y，填到这里
GRASP_X = 0.50                  # 物体中心应该对准的横坐标
GRASP_BOTTOM_Y = 0.88           # 物体框底边到达这个纵坐标 = 距离刚好
ALIGN_TOL_X = 0.05              # 横向对准容差
ALIGN_TOL_Y = 0.03              # 纵向（距离）容差
STABLE_FRAMES = 4               # 连续多少帧对准才执行抓取

SEARCH_TURN_SPEED = 0.45        # 找不到目标时原地转的速度
SEARCH_TURN_MS = 180
SETTLE_S = 0.15                 # 小车停下后等多久再相信新画面（避免运动模糊）
LOST_LIMIT = 3                  # VLM 连续几次说"看不到"就判定丢失
VERIFY_AFTER_GRASP = True       # 抓完后让 VLM 确认是否抓到
