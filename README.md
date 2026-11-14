# AntColony: Multi-Robot Floor Clearing

An overhead camera with AprilTags provides localization, and a single host coordinates three robots: dozer C sweeps far-away debris toward the bin, and arm robots A and B pick the debris up and drop it into the bin.

## HackPrinceton Fall 2026

- **Main track:** Sustainability (smart waste management: a robot team that clears and collects floor litter without a human doing the picking)
- **Optional track:** Best Hardware Hack
- **Special tracks:** Best Physical AI Hack; Best Practical AI Innovation (Amazon)

This repository is the **coordination layer**. It sits between the two parts you already have:

```
Overhead tracker (existing: AprilTag + YOLOE)
        │  UDP 9100  world snapshot (tag poses + floor objects)
        ▼
   coordinator (this repo, runs on the host) ──── http://host:8080 live dashboard
        │  UDP 9200  velocity / skill commands       ▲ UDP 9201 telemetry
        ▼                                            │
   robot/agent.py (one per robot) ── calls masterpi_lite and vlm_grab.py (existing)
```

## Status (2026-11-13)

| Component | Status |
| --- | --- |
| Coordination, path planning, three-robot state machines, traffic rules, calibration tool | Working in the built-in simulator: in the default mode all 8 random layouts are fully cleared with zero robot-robot contact |
| UDP messaging, dashboard | Verified with the simulator process and coordinator process running separately |
| `robot/agent.py` | Working with fake hardware (`--hw mock`) |
| `robot/hw_masterpi.py` | Wheel directions and reachability checked against your `chassis.py` / `arm.py` using a fake serial port; **never run on a real robot** |
| `robot/hw_freenove.py` (dozer) | Channel logic checked against Freenove's official `motor.py` using a fake I2C bus; **never run on a real robot** |
| `tracker_bridge/world_pub.py` | Only the helper functions are tested; you need to wire it into the tracker |
| VLM triage (`triage.py`) | Not tested against a real VLM server; off by default |

The simulator is toy-grade. It verifies that the coordination logic does not deadlock or collide; it says nothing about real grasp success rates or speeds.

## Five-minute quick start (no hardware needed)

```bash
pip install numpy
python tests/test_sim.py                 # 8 layouts, about 4 minutes
# or watch it run:
python -m antcolony.sim --objects 6      # terminal 1: fake camera + fake robots
python -m antcolony.coordinator --sim    # terminal 2: coordinator; open http://localhost:8080
```

## Bringing up the real robots

1. **Host**: `pip install numpy`, then set the arena bounds, bin, tag IDs, and the robots' IPs in `config.json` to the real values.
   Robot IPs change with the network, so they live in named profiles under `networks` in `config.json`. `"network"` picks the active one (currently `hyatt`); you can also override it per run with `ANTCOLONY_NET=school python -m antcolony.coordinator`. A `null` IP falls back to that robot's own `addr`. The coordinator prints the active profile at startup.

   | Profile | A | B | C |
   | --- | --- | --- | --- |
   | `hyatt` | 172.20.1.113 | 172.20.1.120 | 172.20.1.136 |
   | `school` | not set | not set | not set |
2. **Tracker**: add the 5 lines described at the top of `tracker_bridge/world_pub.py` to publish tags and objects to UDP 9100. Requirements are in the next section.
3. **Arm robots A, B**: copy `robot/agent.py` and `robot/hw_masterpi.py` into `~/Desktop/"Master Pi5"/masterpi_lite/` (same level as `vlm_grab.py`) and run
   `python3 agent.py --name A --min-duty 30` (30 on carpet, 18 on smooth floors). For **dozer C**, see the separate section below.
4. **Calibration** (one minute per robot; put only one robot in the middle of the arena at a time, with 50 cm clear on all sides):
   `python -m antcolony.calibrate --robot A`; the result is saved to `calib/A.json`. It automatically computes the tag heading offset, the tag-to-turning-center offset, and the signs for left/right turning and strafing, so no measuring is needed.
5. **Run**: `python -m antcolony.coordinator`; the dashboard is at `http://<host IP>:8080`. Ctrl+C stops all robots.

## Current robot deployment (2026-11-14)

| Robot | Host | User | Code folder | Bench test |
| --- | --- | --- | --- | --- |
| A (arm) | catguard-a | zxz5500 | `~/Desktop/Master Pi5/masterpi_lite/` | real hardware, wheels off the ground: all drive directions correct |
| B (arm) | catguard-b | mitra | `~/Desktop/Master Pi5/masterpi_lite/` | real hardware, wheels off the ground: all drive directions correct |
| C (dozer) | — | — | — | not set up yet |

Both arm robots now run the same `vlm_grab.py` + `masterpi_lite` code (copied from B to A). Per-robot files are **not** shared and must stay on each robot: `masterpi_lite/motor_directions.json`, `masterpi_lite/arm_directions.json`, `Deviation.yaml`. A's wiring needs two options that B's code did not have, so A's copy carries two small patches ported from A's old code (`~/MasterPi/MasterPi/`, kept as a fallback):
- `chassis.py`: `mirror_lr` in `motor_directions.json` (left/right motor wiring swapped);
- `arm.py`: `ports` in `arm_directions.json` (servo 3 is plugged into port 2).

The originals are saved next to them as `chassis.py.orig` / `arm.py.orig`. A currently uses B's `vision_calib.json`; recalibrate A's camera if its grasps are off.

Bench-testing a robot's drive directions from a laptop (wheels off the ground, agent running without `--hw mock`):

```bash
jog(){ python3 - "$@" <<'PY'
import socket,json,time,sys
ip,vx,vy,w=sys.argv[1],*map(float,sys.argv[2:5])
s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
for _ in range(10):
    s.sendto(json.dumps({"type":"vel","vx":vx,"vy":vy,"w":w,"dur":0.4,"guard":False}).encode(),(ip,9200)); time.sleep(0.2)
PY
}
jog <robot IP> 0.3 0 0     # forward; then -0.3 0 0 back, 0 0 0.5 turn left, 0 0 -0.5 turn right, 0 0.3 0 strafe left
```

## Arm robot hardware (A, B: Hiwonder MasterPi, Raspberry Pi 5 version)

From the vendor's spec sheet:

| Item | Value |
| --- | --- |
| Size (L × W × H) | 185 × 162 × 347 mm (arm column ≈ 220 mm) |
| Weight | 1100 g |
| Drive | 4 TT motors with mecanum wheels (can strafe) |
| Arm | 4 DOF + gripper, no wrist rotation; LD-1501MG digital servos + LFD-01M micro servo |
| Camera | 480p, on the arm |
| Compute | Raspberry Pi 5 + MasterPi expansion board |
| Link | Wi-Fi / Ethernet |
| Battery | 2 × 18650 3.7 V cells in series (stock: Hiwonder 1800 mAh), about 60 min of continuous use |

What this means for `config.json`:
- `radius` 0.15 m covers the body: half the footprint diagonal is ≈ 0.123 m, so there is ≈ 2.7 cm of margin.
- `base_fwd` (body center → arm rotation axis, default 0.055 m) is **not** on the spec sheet and should be measured on each robot.
- Battery: one charge is roughly one hour of driving. Charge between judging rounds and keep spare packs if you have them.

## Dozer setup (Freenove FNK0043 4WD car + Pi 5)

You do not need Freenove's own server code; `hw_freenove.py` drives the expansion board directly over I2C.

1. Flash 64-bit Raspberry Pi OS with Raspberry Pi Imager. In Imager, pre-fill your router's WiFi, enable SSH, and set a hostname (e.g. `catguard-c`).
2. Enable I2C, install dependencies, then reboot:
   ```bash
   sudo raspi-config nonint do_i2c 0
   sudo apt install -y python3-smbus2 i2c-tools
   sudo reboot
   ```
3. `i2cdetect -y 1` should show `40` (PCA9685 motor driver) and `48` (battery voltage ADC). If not, check that the expansion board is seated firmly and the battery switch is on.
4. Copy `robot/agent.py` and `robot/hw_freenove.py` into the same directory on the Pi.
5. Prop the car up so the wheels are off the ground and run `python3 hw_freenove.py wheels` to confirm each wheel's direction one at a time (run it in your own SSH terminal; it asks y/n). Then put it on the floor and run `python3 hw_freenove.py drive`; it should go forward, backward, turn left, and turn right in that order.
6. Start the robot service: `python3 agent.py --name C --hw freenove --min-duty 35 --rot-min 55`.
   - `--min-duty`: the duty cycle (percent) at which the car just starts moving straight.
   - `--rot-min`: the duty cycle at which the car just starts turning in place. Four ordinary wheels turn in place by skidding, so on carpet this value will be noticeably higher, and you need extra margin when pushing things with the blade.
7. Put a tag on top of the car (flat, not blocked by the camera gimbal), fill in C's tag ID and IP in `config.json`, then measure two dimensions and enter them:
   - `blade_fwd`: distance from the body center (midpoint between front and rear axles) to the pushing face of the blade, in meters;
   - `blade_w`: blade width, in meters; set `radius` to the distance from the body center to the farthest corner of the blade.
8. `python -m antcolony.calibrate --robot C`. This car cannot strafe, so `strafe_ok` will be false in the calibration result; that is expected.

Two things to watch out for:
- The Pi 5 is powered by the car's expansion board. Motor current is high when pushing or turning in place; if the Pi reboots or reports undervoltage, give the Pi its own power bank. I could not verify this for you.
- This car has no hardware watchdog: `agent.py` brakes when it exits normally or is `kill`ed, but if the Pi hangs, the motors keep their last output. Keep the battery switch within reach while debugging.

## What the tracker must provide

- Position (meters) and heading of every robot tag and the bin tag in **one shared floor coordinate frame**, sent every frame, even if the detection network is slow.
- The position of every floor object; if you have segmentation contours, also give the short side `w`, long side `l`, and long-axis direction `ang` (`floor_rect()` computes these for you). With the long-axis direction, the arm robots approach along the long axis so the gripper grabs the short side (this arm has no wrist rotation).
- Do not publish arena corner tags as objects; mask out the robot bodies and the bin before doing floor-difference detection.

## Arena layout

- Leave a **30–50 cm lane** around the debris area and around the bin; robots must be able to get behind any object and behind the bin. The dozer cannot get behind an object that is pressed against an edge.
- The bin rim must be no higher than about 6 cm (the arm releases 18 cm in front and 10 cm high), and the bin must be fixed to the floor.
- Every piece of debris must be grippable: short side ≤ 5.5 cm (`triage.grasp_max_short`), long side ≤ 16 cm, light enough, and not a thin sheet. Oversized objects are marked red on the dashboard and no robot will touch them.

## What each robot does

- **Arm robots**: the overhead camera guides the robot to 30 cm in front of the object and aligns it → the on-board `vlm_grab.py approach` handles the last stretch (find, approach, align, grab, lift, voltage check) → the overhead camera guides it to the bin → it reaches over the rim and releases → it backs away. Only one robot may be at the bin at a time; the other waits in the outer ring.
- **Dozer**: picks the object farthest from the bin, gets behind it, and pushes it to a staging spot in front of the bin (arranged in an arc, at least 30 cm apart). The arm robots pick up staged objects from the outside in along the "lanes" and then drive straight to the bin.
- Idle robots return to their own parking spot and automatically move aside if they block someone else.

Optional mode: set `bin.floor_entry` to `true` in `config.json` if one side of the bin is open at floor level; the dozer then pushes objects straight into the bin and also handles large items the gripper cannot pick up.

## Layout

```
antcolony/common.py       geometry, messages, config
antcolony/planner.py      grid A*
antcolony/control.py      path following: continuous control at a distance, timed pulses for the last few degrees of in-place turns
antcolony/world.py        world model: tag → body pose, object association, occlusion, disappearance
antcolony/brains.py       task state machines for arm robots / dozer
antcolony/coordinator.py  task allocation, bin mutex, traffic rules, main loop, UDP
antcolony/triage.py       decides grippability by size; optionally asks a VLM as well
antcolony/dashboard.py    dashboard
antcolony/calibrate.py    single-robot drive calibration
antcolony/sim.py          simulator (plays both camera and robots)
robot/agent.py            on-robot service (watchdog, skills, telemetry)
robot/hw_masterpi.py      arm robot: adapter for masterpi_lite + vlm_grab.py
robot/hw_freenove.py      dozer: Freenove FNK0043, direct I2C; built-in wheel direction test
tracker_bridge/world_pub.py  tracker → coordinator publisher
tests/test_sim.py         end-to-end test
```

## Messages (each is one JSON object over UDP)

```
World snapshot → host:9100   {"t":..,"tags":{"10":[x,y,heading_rad],..},"objects":[{"id","label","x","y","w","l","ang"},..]}
Velocity       → robot:9200  {"type":"vel","vx":fwd,"vy":left,"w":ccw,"dur":sec,"guard":true}   each in -1..1, auto-stops after at most 0.4 s
Skill          → robot:9200  {"type":"skill","id":17,"name":"grab|drop|home|open|stop","args":{..}}   resending the same id is harmless
Telemetry      → host:9201   {"type":"telem","name":"A","batt_mv":..,"sonar_cm":..,"skill":{"id","state","detail"},"holding":..}
```

The dozer only uses the `vel` message; `vy` is ignored because it cannot strafe.

## Three things to verify before running on real robots

1. Whether the gripper is still holding after the serial port is reopened at the end of `vlm_grab.py approach` (`hw_masterpi.py` opens and closes the serial port once before and once after grabbing).
2. Whether the carry pose `CARRY` and the release position (`drop.y_cm`, `drop.z_cm`) clear your bin rim.
3. Whether `--min-duty` and `--rate-min` (arm robots; `--rot-min` on the dozer) are the minimum values that just get the robot moving on the competition floor; `control.rot_rate0` corrects itself at runtime.
