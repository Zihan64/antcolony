# 电脑跑 VLM + 树莓派小车抓取物体

```
 ┌──────────── Mac (M2 16GB) ─────────────┐        Wi-Fi         ┌────── 树莓派小车 ──────┐
 │ Ollama: qwen3-vl:4b  (常驻内存)         │                      │ robot_server.py        │
 │   ↑ 后台线程 0.3~1.5s 一次: 认出+框出目标 │ ◄── JPEG 视频 5555 ── │  摄像头 640x480        │
 │ grasp.py  15Hz 控制循环                  │                      │  电机 (gpiozero)       │
 │   OpenCV 跟踪器每帧更新目标位置 (几 ms)   │ ── JSON 指令 5556 ──► │  机械臂 (PCA9685)      │
 └─────────────────────────────────────────┘                      │  看门狗自动停车        │
                                                                   └────────────────────────┘
```

**为什么快**：VLM 不放在控制环里。VLM 只负责"认出目标、给出框"，每帧的跟随由 OpenCV 跟踪器完成（几毫秒），所以小车的反应速度取决于跟踪器（约 15Hz），而不是 VLM。视频只取最新一帧，抓取动作序列在树莓派本地执行。

---

## 推荐的 VLM（针对你的 M2 16GB Mac）

| 模型 | 大小 | 预计定位延迟* | 适合 |
|---|---|---|---|
| **qwen3-vl:4b-instruct** ⭐ 默认 | ~3 GB | 约 0.5–1.2 s | 速度和框的准确度最平衡，能听懂"左边那个蓝色积木"这种描述 |
| qwen3-vl:2b-instruct | ~2 GB | 约 0.3–0.7 s | 最快，桌面上单个、颜色明显的物体够用 |
| qwen3-vl:8b-instruct | ~6 GB | 约 1.5–3 s | 最准，物体多、描述复杂时用 |
| qwen2.5vl:3b | ~3 GB | 约 0.6–1.3 s | 备选，旧版 Ollama 也能用 |

\* 延迟是估计值，**以第 4 步 `bench_vlm.py` 在你电脑上实测为准**。

选 Qwen-VL 系列是因为它们专门训练过**输出物体框坐标**（grounding），这是抓取最需要的能力；很多其它小模型只会描述图片，不会给准确坐标。

---

## 一步步操作

### 第 1 步：Mac 上安装 Ollama 并下载模型

```bash
brew install ollama
```

```bash
ollama serve
```

上面这个窗口保持开着，另开一个终端：

```bash
ollama pull qwen3-vl:4b-instruct
```

```bash
ollama pull qwen3-vl:2b-instruct
```

> 如果提示找不到 `-instruct` 标签，改用 `qwen3-vl:4b`，并同步修改 `pc/config.py` 里的 `VLM_MODEL`。
> 如果 Ollama 版本太老不支持 qwen3-vl，先 `brew upgrade ollama`。

### 第 2 步：Mac 上安装 Python 依赖

```bash
cd robot_vlm/pc && python3 -m pip install -r requirements.txt
```

### 第 3 步：先只测 VLM（不用小车）

拿一个红色杯子放在 Mac 摄像头前：

```bash
python3 bench_vlm.py --target "red cup" --models qwen3-vl:2b-instruct qwen3-vl:4b-instruct
```

会打印每个模型的延迟，并弹窗显示框的位置。**选框得准的里面最快的那个**，写进 `config.py` 的 `VLM_MODEL`。

### 第 4 步：树莓派准备

1. 树莓派和 Mac 连**同一个 Wi-Fi**（最好 5GHz，或者直接网线）。
2. 树莓派上查 IP：`hostname -I`，填到 Mac 上 `pc/config.py` 的 `PI_IP`。建议在路由器里给树莓派固定 IP。
3. 把 `pi/` 文件夹拷到树莓派（在 Mac 上执行，`pi@` 换成你的用户名）：

```bash
scp -r robot_vlm/pi pi@192.168.1.50:~/
```

4. 在树莓派上安装依赖（CSI 摄像头用系统自带的 picamera2，所以用 `--system-site-packages`）：

```bash
sudo apt install -y python3-picamera2 python3-opencv python3-zmq python3-gpiozero
```

```bash
python3 -m venv --system-site-packages ~/venv && ~/venv/bin/pip install adafruit-circuitpython-servokit
```

5. 用了 PCA9685 舵机板的话，打开 I2C：`sudo raspi-config` → Interface Options → I2C → Enable。
6. **按你的接线修改 `pi/robot_server.py` 顶部**：电机引脚 `LEFT_MOTOR / RIGHT_MOTOR`、舵机通道 `ARM_CHANNELS`、夹爪角度。
   - 如果你的小车是厂商套件（自带驱动板 / 串口舵机），只需改 `Drive.set()` 和 `Arm._write()` 两个函数，换成厂商提供的控制代码，其它都不用动。

7. 启动：

```bash
~/venv/bin/python ~/pi/robot_server.py
```

> 没有接电机/舵机库时会显示"模拟模式"，只打印指令不动作 —— 可以先用这个调通网络。

### 第 5 步：测试连接和延迟

Mac 上：

```bash
python3 grasp.py --view
```

窗口里看到小车画面，左上角显示帧率和指令往返时间（局域网正常 < 20 ms）。按 `q` 退出。

### 第 6 步：标定机械臂（第一次必须做，**先把车架空，轮子离地**）

```bash
python3 arm_calib.py
```

输入 `shoulder 90`、`elbow 100` 这样一点一点调，找到三个姿态：
- `home`：手臂收起，**不挡摄像头**
- `reach`：夹爪伸到车前方地面的抓取点
- `lift`：抓起后抬高

把角度填回 `pi/robot_server.py` 的 `ARM_POSES`，重启 `robot_server.py`。输入 `grasp` 测试完整抓取动作。

### 第 7 步：标定抓取点

把物体手动放到夹爪 `reach` 姿态正好能抓到的位置，然后：

```bash
python3 grasp.py --target "red cup" --calib
```

小车不会动。窗口上方显示 `cx=… bottom=…`，把这两个数字填进 `config.py` 的 `GRASP_X` 和 `GRASP_BOTTOM_Y`。

### 第 8 步：自动抓取 🎉

把物体放到车前 0.5–1 米处：

```bash
python3 grasp.py --target "red cup"
```

流程：`SEARCH`（原地转着找）→ `TRACK`（边跟踪边开过去对准紫色十字）→ `GRASP`（树莓派执行抓取）→ VLM 确认是否抓到，失败自动重来。
**空格键 = 急停/继续，`q` = 退出。** 目标描述用英文效果最好，如 `"blue block"`、`"the bottle on the left"`。

---

## 调参与提速

| 现象 | 调整（`pc/config.py`） |
|---|---|
| 轮子不转 / 抖动 | 调大 `MIN_SPEED`（电机死区） |
| 左右来回摆 | 调小 `KP_TURN` |
| 冲过头 | 调小 `KP_FWD`、`MAX_FWD` |
| 总差一点抓不到 | 重新做第 7 步，或收紧 `ALIGN_TOL_X/Y` |
| VLM 太慢 | 换 2b 模型；或把 `VLM_IMG_W/H` 改成 `384, 288` |
| 框不准 | 换 8b 模型；或把 `VLM_IMG_W/H` 改成 `640, 480`；目标描述写具体些 |
| 抓完确认总说失败 | 物体抬起后可能不在摄像头视野里，把 `VERIFY_AFTER_GRASP` 设为 `False` |
| 画面卡顿 | 树莓派上把 `JPEG_QUALITY` 降到 60；用 5GHz Wi-Fi |

**再进一步提速**（目标是固定几种东西时）：用 VLM 自动标注几百张图，训练一个 YOLO 小模型代替 VLM 做检测，延迟能到 10–20 ms；VLM 只用来理解指令和确认抓取结果。

## 文件说明

```
pi/robot_server.py   树莓派：摄像头推流 + 电机 + 机械臂 + 看门狗
pc/config.py         所有参数
pc/grasp.py          主程序（搜索 → 跟踪 → 抓取 → 确认）
pc/vlm.py            Ollama VLM 封装，输出物体框
pc/robot_link.py     网络通信
pc/bench_vlm.py      模型测速对比
pc/arm_calib.py      机械臂标定
```
