# AntColony 蚁群：多车协同清场

俯视相机 + AprilTag 负责定位，一台主机统一调度三台车：铲车 C 把远处的杂物拢到筐边，机械臂车 A、B 把杂物夹进筐。

这个仓库是**调度层**：它接在你们已有的两部分之间。

```
俯视跟踪程序（已有：AprilTag + YOLOE）
        │  UDP 9100  世界快照（标签位姿 + 地面物体）
        ▼
   coordinator（本仓库，跑在主机上）──── http://主机:8080 实时看板
        │  UDP 9200  速度指令 / 技能指令          ▲ UDP 9201 遥测
        ▼                                         │
   robot/agent.py（每台车一个）── 调用 masterpi_lite 和 vlm_grab.py（已有）
```

## 状态（2026-11-13）

| 部分 | 状态 |
| --- | --- |
| 调度、路径规划、三车状态机、交通规则、标定工具 | 在自带模拟器里跑通：默认模式 8 种随机布局全部清场，车与车零接触 |
| UDP 通信、看板 | 模拟器进程 + 调度进程分开跑通过 |
| `robot/agent.py` | 用假硬件（`--hw mock`）跑通 |
| `robot/hw_masterpi.py` | 用假串口对着你们的 `chassis.py` / `arm.py` 核对过轮子方向和可达性，**没上过真车** |
| `robot/hw_freenove.py`（铲车） | 用假 I2C 总线对着 Freenove 官方 `motor.py` 的通道逻辑核对过，**没上过真车** |
| `tracker_bridge/world_pub.py` | 只测过辅助函数，需要你们接进跟踪程序 |
| VLM 分诊（`triage.py`） | 没对着真的 VLM 服务器测过，默认关闭 |

模拟器是玩具级的，它验证的是调度逻辑不死锁、不撞车，不能说明真车的抓取成功率和速度。

## 五分钟上手（不需要硬件）

```bash
pip install numpy
python tests/test_sim.py                 # 8 种布局，约 4 分钟
# 或者看着跑：
python -m antcolony.sim --objects 6      # 终端 1：假相机 + 假小车
python -m antcolony.coordinator --sim    # 终端 2：调度；浏览器打开 http://localhost:8080
```

## 接真车的步骤

1. **主机**：`pip install numpy`，把 `config.json` 里的场地范围、筐、标签编号、三台车的 IP 改成实际值。
2. **跟踪程序**：按 `tracker_bridge/world_pub.py` 开头的说明加 5 行，把标签和物体发到 UDP 9100。要求见下一节。
3. **机械臂车 A、B**：把 `robot/agent.py`、`robot/hw_masterpi.py` 拷到 `masterpi_lite/`（和 `vlm_grab.py` 同一层），运行
   `python3 agent.py --name A --min-duty 30`（地毯 30，光滑地面 18）。**铲车 C** 见下面单独一节。
4. **标定**（每台车一分钟，一次只放一台车在场地中间，四周留 50 cm）：
   `python -m antcolony.calibrate --robot A`，结果存到 `calib/A.json`。它自动算出标签朝向偏差、标签到转向中心的偏移、左右转和横移的正负号，不用量尺寸。
5. **开跑**：`python -m antcolony.coordinator`，看板在 `http://主机IP:8080`。Ctrl+C 会让所有车停下。

## 铲车（Freenove FNK0043 四驱车 + Pi 5）上车步骤

不需要安装 Freenove 自己的那套服务端代码，`hw_freenove.py` 直接通过 I2C 控制扩展板。

1. 用 Raspberry Pi Imager 烧 64 位 Raspberry Pi OS，在 Imager 里预先填好你们路由器的 WiFi、打开 SSH、设主机名（比如 `catguard-c`）。
2. 开 I2C 并装依赖，然后重启：
   ```bash
   sudo raspi-config nonint do_i2c 0
   sudo apt install -y python3-smbus2 i2c-tools
   sudo reboot
   ```
3. `i2cdetect -y 1` 应该看到 `40`（电机驱动 PCA9685）和 `48`（电池电压 ADC）。看不到就检查扩展板是否插紧、电池开关是否打开。
4. 把 `robot/agent.py` 和 `robot/hw_freenove.py` 拷到 Pi 上同一个目录。
5. 把车架起来让轮子悬空：`python3 hw_freenove.py wheels`，逐个确认四个轮子的方向（要在你自己的 SSH 终端里跑，它会问 y/n）。然后落地跑 `python3 hw_freenove.py drive`，应该依次前进、后退、左转、右转。
6. 启动车端服务：`python3 agent.py --name C --hw freenove --min-duty 35 --rot-min 55`。
   - `--min-duty`：直行刚好能动的占空比（百分比）。
   - `--rot-min`：原地转向刚好能转起来的占空比。四个普通轮子原地转是靠打滑，地毯上需要的值会明显更高，带着铲板推东西时还要再留余量。
7. 车顶贴一个标签（放平，别被摄像头云台挡住），在 `config.json` 里填 C 的标签编号和 IP，然后量两个尺寸填进去：
   - `blade_fwd`：从车体中心（前后轮轴中点）到铲板推面的距离，米；
   - `blade_w`：铲板宽度，米；`radius` 取车体中心到铲板最远角的距离。
8. `python -m antcolony.calibrate --robot C`。这台车不能横移，标定结果里 `strafe_ok` 会是 false，属于正常。

两件要留意的事：
- Pi 5 由小车扩展板供电。推东西或原地转向时电机电流大，如果 Pi 重启或报低电压，就给 Pi 单独接一个充电宝。这一点我没法替你们验证。
- 这台车没有看门狗硬件：`agent.py` 正常退出或被 `kill` 时会刹车，但如果 Pi 死机，电机会保持最后的输出。调试时手边留着电池开关。

## 跟踪程序要提供什么

- 所有车顶标签和筐标签在**同一个地面坐标系**里的位置（米）和朝向，每帧都发，哪怕检测网络很慢。
- 每个地面物体的位置；有分割轮廓的话再给短边 `w`、长边 `l`、长轴方向 `ang`（`floor_rect()` 帮你算）。有了长轴方向，机械臂车会沿长轴接近，让夹爪夹短边（这台机械臂没有手腕旋转）。
- 不要把场地角标签当成物体发出来；做地面差分检测之前先把车身和筐的区域抠掉。

## 场地怎么摆

- 杂物区四周、筐的四周都要留 **30–50 cm 的车道**，车必须能绕到任何物体和筐的后面。贴着边的物体，铲车没法绕到它身后。
- 筐沿高度不超过约 6 cm（机械臂在前方 18 cm、高 10 cm 处松手），筐要固定在地上。
- 所有杂物都要是夹爪夹得起来的：短边 ≤ 5.5 cm（`triage.grasp_max_short`），长边 ≤ 16 cm，够轻，不是薄片。超出的物体会在看板上标红，没有车会去动它。

## 三台车各自做什么

- **机械臂车**：俯视相机把车引到物体前 30 cm 并摆正 → 车上的 `vlm_grab.py approach` 完成最后一段（找、靠近、对齐、夹、抬起、电压检查）→ 俯视相机引到筐边 → 伸过筐沿松手 → 倒车离开。筐边同一时间只进一台车，另一台在外圈等。
- **铲车**：挑离筐最远的物体，绕到它身后，推到筐前的停放点（弧形排开，互相隔 30 cm 以上）。机械臂车从外向里沿"车道"夹起后直走就到筐。
- 空闲的车回自己的停车位；挡了别人的路会自动让开。

可选模式：`config.json` 里把 `bin.floor_entry` 设为 `true`，表示筐有一面贴地敞口，铲车直接把物体推进筐，夹爪夹不了的大件也交给它。

## 目录

```
antcolony/common.py       几何、报文、配置
antcolony/planner.py      栅格 A*
antcolony/control.py      路径跟随：远处连续控制，原地转向的最后几度用定时脉冲
antcolony/world.py        世界模型：标签→车体位姿，物体关联、遮挡、消失判定
antcolony/brains.py       机械臂车 / 铲车的任务状态机
antcolony/coordinator.py  任务分配、筐的互斥锁、交通规则、主循环、UDP
antcolony/triage.py       按尺寸判断夹不夹得了；可选再问一次 VLM
antcolony/dashboard.py    看板
antcolony/calibrate.py    单车行驶标定
antcolony/sim.py          模拟器（同时扮演相机和小车）
robot/agent.py            车端服务（看门狗、技能、遥测）
robot/hw_masterpi.py      机械臂车：适配 masterpi_lite + vlm_grab.py
robot/hw_freenove.py      铲车：Freenove FNK0043，直接走 I2C；自带轮子方向测试
tracker_bridge/world_pub.py  跟踪程序 → 调度 的发布器
tests/test_sim.py         端到端测试
```

## 报文（都是 UDP 上的一条 JSON）

```
世界快照 → 主机:9100   {"t":..,"tags":{"10":[x,y,朝向弧度],..},"objects":[{"id","label","x","y","w","l","ang"},..]}
速度     → 车:9200     {"type":"vel","vx":前,"vy":左,"w":逆时针,"dur":秒,"guard":true}   各量 -1..1，最长 0.4 秒后自动停车
技能     → 车:9200     {"type":"skill","id":17,"name":"grab|drop|home|open|stop","args":{..}}   同一个 id 重发无害
遥测     → 主机:9201   {"type":"telem","name":"A","batt_mv":..,"sonar_cm":..,"skill":{"id","state","detail"},"holding":..}
```

铲车只用到 `vel` 这一条报文；`vy` 会被忽略，因为它不能横移。

## 上真车前要验证的三件事

1. `vlm_grab.py approach` 结束后重新打开串口时，夹爪是否仍然夹着（`hw_masterpi.py` 抓取前后各开关一次串口）。
2. 携带姿态 `CARRY` 和松手位置（`drop.y_cm`、`drop.z_cm`）是否越得过你们的筐沿。
3. `--min-duty`、`--rate-min` 是否是比赛地面上刚好能动的最小值；`control.rot_rate0` 会在运行中自己修正。
