# 演示操作手册：两辆车把场上所有物体放进白盒子

离线可用（局域网没有互联网也行）。所有命令都在 **你的 Mac** 的终端里运行，除了第 3 步（俯视摄像头那台电脑）。
每条命令前都先进入项目文件夹：

```bash
cd ~/antcolony-git
```

---

## 0. 演示前的准备

**电池**
- A、B 两辆车都**充满电**。低于 7.0 V 时夹取经常失败，程序会拒绝启动（见第 9 节）。

**物体（5 个：2 个方块 + 3 个杂物）**
- 选**颜色鲜艳**的。白、黑、灰色的东西车只在非常确定时才拿，还容易把场地角上的白纸杯当成杂物。
- **5 cm 以内、大致方形、放得稳**。夹爪最多张开 7.5 cm；细长的东西会被跳过；小圆球容易滚走。
- **5 个颜色尽量各不相同**。两辆车靠颜色区分"你拿这个、我拿那个"，同色的会排队一个一个拿。

**摆放**
- 物体之间、物体和墙、物体和白盒子之间，都隔开 **30 cm 以上**。
- 白盒子放在开阔处，周围不要有别的白色东西。顶上的 **54 号 tag** 要朝上、露在外面，俯视摄像头要看得到。
- 两辆车**并排**放在起点，间隔约 30 cm，**车头朝场地里面**，前面至少留 **15 cm 空地**（出发时每辆车会往前开约 10 cm 校准方向）。
- 场地里和车的视野里**不要有人的手脚**。

---

## 0.5 换到 AntColony 网络（每次换 Wi-Fi 后都要做一次）

**你的 Mac、两辆车、俯视摄像头那台电脑，必须连在同一个 Wi-Fi 上。** 换网之后每台设备的 IP 都会变，下面这一条命令会把所有地址更新好。

1. 你的 Mac 和俯视摄像头那台电脑都连上 **AntColony** Wi-Fi。
2. 打开 A、B 两辆车，**等 1–2 分钟**。
3. 先启动 VLM（第 1 步的命令），再在你的 Mac 上运行：

```bash
python3 scripts/net.py
```

它会：
- 找到你 Mac 的新 IP；
- 扫一遍网络，按网卡硬件地址找到 A、B、C（不管路由器给它们分了什么地址）；
- 把地址写进 `config.json`，后面所有命令都自动用新地址；
- **把每辆车里的 VLM 地址改成你 Mac 的新 IP**（不改的话车连不上 VLM，抓取和找盒子都会失败），并重启车上的程序；
- 最后打印一行**俯视摄像头那台电脑要用的命令**，类似：

```
python cam_control/run_tracker.py --antcolony 10.0.0.12
```

**把这一行原样抄下来，第 3 步要用。**

- 显示 `car A: NOT FOUND`：那辆车没开机，或者连到了别的 Wi-Fi。等一会儿再运行一次 `python3 scripts/net.py`。
- 想先看看它会改什么、不真的改：`python3 scripts/net.py --dry-run`

**备用办法：`net.py` 出错时手动改。** AntColony 上 A 是 **192.168.0.63**，B 是 **192.168.0.64**，配置里已经写好了。只需要手动把车里的 VLM 地址改成你 Mac 的新 IP：

1. 查你 Mac 的 IP：

```bash
ipconfig getifaddr en0
```

2. 把下面两条命令里的 `MAC_IP` 换成上一步显示的地址，然后运行：

```bash
ssh zxz5500@192.168.0.63 'echo "{\"url\": \"http://MAC_IP:8090\", \"img_tokens\": 256}" > "/home/zxz5500/Desktop/Master Pi5/masterpi_lite/masterpi_lite/vlm_server.json"'
```

```bash
ssh mitra@192.168.0.64 'echo "{\"url\": \"http://MAC_IP:8090\", \"img_tokens\": 256}" > "/home/mitra/Desktop/Master Pi5/masterpi_lite/masterpi_lite/vlm_server.json"'
```

3. 重启车上的程序：`bash scripts/cars.sh start`
4. tracker 用 `python cam_control/run_tracker.py --antcolony MAC_IP` 启动。

---

## 1. 启动 VLM（你的 Mac，开一个窗口一直放着）

```bash
bash scripts/vlm.sh
```

- 如果显示 `VLM server is already running` 就不用管了。
- 否则大约 6 秒后会出现很多行日志，**这个窗口不要关**。

---

## 2. 打开两辆车，启动车上的程序

1. 打开 A、B 的电源，**等 1–2 分钟**让树莓派连上 Wi-Fi。
2. 在你的 Mac 上（新开一个窗口）：

```bash
bash scripts/cars.sh start
```

正常的结果：

```
car A: program running
car B: program running
[A] battery 7.4x V, sonar ..., holding: no
[B] battery 7.4x V, sonar ..., holding: no
```

- `NOT ON THE NETWORK`：那辆车还没连上网，再等一会儿重试；还不行就重启那辆车。
- 电池低于 **7.0 V**：先充电。

随时查看两辆车的状态（车不会动）：

```bash
bash scripts/cars.sh status
```

---

## 3. 俯视摄像头（Junbo 的电脑）

1. 启动 tracker，**用第 0.5 步最后打印出来的那条命令**（里面是你 Mac 在 AntColony 网络上的 IP）。如果 tracker 已经在运行、但用的是旧 IP，先关掉再用新命令重新启动：

```bash
python cam_control/run_tracker.py --antcolony <你 Mac 的新 IP>
```

2. 在界面里按 **Start**。
3. **重新设置场地**（摄像头挪动过就必须做）：
   - 四个角的 tag（29–32 号，最大的四张）平放在场地四角，**四张都在画面里**、没被挡住。
   - 摄像头别动，按 **"Set up arena (2 s) / redo"**。
   - 按日志提示，用卷尺量两段距离（原点 tag → +x 方向那张，原点 → +y 方向那张），填进去，按 **"Apply tape measurements"**。
4. 保持 **"Adaptive pose (auto lock / re-lock)"** 勾选。

---

## 4. 检查（车不会动）

在你的 Mac 上：

```bash
python3 -m antcolony.check
```

要看到：
- `== tracker`：每秒 10 帧以上，四个角都看到。
- A、B 两行：`tag seen in ...%`、`agent ok`、电池 7 V 以上。
- 没有 `robot A/B stands outside the arena`。没有这条警告，第 6 步就可以打开边界保护。
- C 车那几行 PROBLEM 可以忽略（这次不用 C）。

想看两辆车的实时位置（Ctrl-C 结束）：

```bash
python3 -m antcolony.tagwatch
```

**`check` 和 `tagwatch` 看完一定要关掉（Ctrl-C）再进行第 6 步**，它们和演示程序用同一个端口。

---

## 5. 摆好场地

按第 0 节摆放物体、白盒子和两辆车。

---

## 6. 开始演示

**边界保护关闭**（场地没重新设置、或 `check` 说车在场外时用这个）：

```bash
bash scripts/show.sh
```

**边界保护打开**（第 3 步做过、第 4 步没有 "outside the arena" 警告时用这个）：

```bash
bash scripts/show.sh fence
```

屏幕上每一行是 `秒数 [车名] 正在做什么`。你会看到：
1. 两辆车各往前开约 10 cm（`heading check`）。
2. A 先选离自己最近的物体（`find`），B 等 A 选完再选，避开 A 选的颜色。
3. 夹取（`grab, try 1 of 3`），失败会后退、重新对准、再试。
4. 夹住后去白盒子（`to the tray`），俯视摄像头先把车转向盒子。
5. 放进去（`place: ok`），后退 35 cm，再找下一个。
6. 两辆车都找不到物体时结束，最后一行：`=== finished: N object(s) in the tray`。

最长运行 25 分钟，到时自动停。

---

## 7. 演示中途要停

- 在演示窗口按 **Ctrl-C**：两辆车都会停下。
- 来不及就**直接把车拿起来**。
- 从另一个窗口紧急停车：

```bash
bash scripts/cars.sh stop
```

**停了之后接着演示**：直接再运行一次 `bash scripts/show.sh`。车会记得自己是不是夹着东西，夹着的话会先送进盒子。

---

## 8. 演示结束后

让两辆车放下手里的东西、机械臂收回：

```bash
bash scripts/cars.sh home
```

记录和照片自动保存在 `runs/show_<时间>/` 文件夹里。

---

## 9. 常见问题

| 看到什么 | 怎么办 |
|---|---|
| `no answer - run: bash scripts/cars.sh start` | 车上的程序没运行：`bash scripts/cars.sh start` |
| `NOT ON THE NETWORK` | 车没开机或没连上 Wi-Fi：检查电源，等 1–2 分钟。刚换过 Wi-Fi 的话运行 `python3 scripts/net.py` |
| `check` 里 `== tracker` 没有数据 | tracker 发数据的地址不对：用 `python3 scripts/net.py` 最后打印的命令重新启动 tracker |
| 抓取时报 VLM 连不上 / 一直 "not found" | 车里的 VLM 地址是旧的：运行 `python3 scripts/net.py`；VLM 本身没开就运行 `bash scripts/vlm.sh` |
| `battery below 7.0 V ... Charge, or --min-volt 0` | 充电。实在要跑：`python3 -m antcolony.clear --fence-m 0 --min-volt 6.5` |
| `The VLM server is not running` | 另开窗口运行 `bash scripts/vlm.sh` |
| `cannot listen on UDP 9201` | 另一个控制程序还开着（check、tagwatch、twin、show）：先把它关掉 |
| 车一直转圈找盒子 | 看 54 号 tag 是否露在外面、俯视摄像头是否在运行；盒子周围别有其他白色东西 |
| 车一直夹不住 | 多半是电池低；或物体太大、太圆、颜色不鲜艳 |
| 车朝人或墙开过去 | 把人挪出视野；拿起车，Ctrl-C 后重新开始 |
| `OUTSIDE THE ARENA ... STOPPED` | 边界保护叫停了这辆车。如果车其实在场内，说明场地没设置好：用不带 `fence` 的 `bash scripts/show.sh` |
| 车停下时手里还夹着东西 | `bash scripts/cars.sh home` 放下；或直接再运行 `bash scripts/show.sh`，它会先送进盒子 |

**同一时间只运行一个控制程序**：`check`、`tagwatch`、`twin`、`show.sh` 不能同时开。
