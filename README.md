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

## Status (2026-11-14)

| Component | Status |
| --- | --- |
| Coordination, path planning, three-robot state machines, traffic rules, calibration tool | Working in the built-in simulator: in the default mode all 8 random layouts are fully cleared with zero robot-robot contact |
| UDP messaging, dashboard | Verified with the simulator process and coordinator process running separately |
| `robot/agent.py` | Working with fake hardware (`--hw mock`); on car C with `--hw freenove` (sonar stop tested on the floor) |
| `robot/hw_masterpi.py` | Wheel directions and reachability checked against your `chassis.py` / `arm.py` using a fake serial port; **never run on a real robot** |
| `robot/hw_freenove.py` (dozer) | Run on car C (2026-11-14): wheel directions, drive, battery, front sonar stop (35 cm) tested on the floor |
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
   Robot IPs change with the network, so they live in named profiles under `networks` in `config.json`. `"network"` picks the active one (currently `hyatt`, the hotel; **at the competition switch to `lan`**); you can also override it per run with `ANTCOLONY_NET=school python -m antcolony.coordinator`. A `null` IP falls back to that robot's own `addr`. The coordinator prints the active profile at startup.
   When the network changes, the VLM computer's IP changes too: update `url` in `masterpi_lite/masterpi_lite/vlm_server.json` on both arm cars and the tracker's `--antcolony <coordinator IP>`.

   | Profile | A | B | C |
   | --- | --- | --- | --- |
   | `hyatt` | 192.168.13.106 | 192.168.13.49 | 192.168.13.110 |
   | `school` | not set | not set | not set |
   | `lan` (**use this at the competition**: the team's own AntColony router) | 192.168.0.63 | 192.168.0.64 | not set |
2. **Tracker**: add the 5 lines described at the top of `tracker_bridge/world_pub.py` to publish tags and objects to UDP 9100. Requirements are in the next section.
3. **Arm robots A, B**: copy `robot/agent.py` and `robot/hw_masterpi.py` into `~/Desktop/"Master Pi5"/masterpi_lite/` (same level as `vlm_grab.py`) and run
   `python3 agent.py --name A --min-duty 30` (30 on carpet, 18 on smooth floors). For **dozer C**, see the separate section below.
4. **Calibration** (one minute per robot; put only one robot in the middle of the arena at a time, with 50 cm clear on all sides):
   `python -m antcolony.calibrate --robot A`; the result is saved to `calib/A.json`. It automatically computes the tag heading offset, the tag-to-turning-center offset, and the signs for left/right turning and strafing, so no measuring is needed.
5. **Run**: `python -m antcolony.coordinator`; the dashboard is at `http://<host IP>:8080`. Ctrl+C stops all robots.

## Two-car edge test on two computers (2026-11-14)

Task: a few blocks of different colours on the floor; arm cars **A and B each bring one block to the arena edge**. A missed grab is tried again; after **5 misses in a row** that car gives up and drives home. No bin, no dozer. Everything specific to this test is in `profiles/edge2.json` (an overlay on `config.json`).

```
Computer 1 (VLM)                    Computer 2 (overhead camera)                 cars A, B
llama-server :8090  ◄── HTTP ───────────────────────────────────────────────── vlm_grab.py (last 30 cm + grab)
                                    run_tracker.py --antcolony 127.0.0.1   (starts apriltag_tracker_gui.py)
                                         │ UDP 9100 (tags, objects, floor tags)
                                         ▼
                                    python -m antcolony.coordinator ── UDP 9200 ──► agent.py
                                         ▲────────────────────── UDP 9201 telemetry ──┘
```

The coordinator can run on either computer; on the camera computer the tag data never crosses Wi-Fi. If it runs elsewhere, start the tracker with `--antcolony <that computer's IP>`.

**Status: the coordination logic for this task passes in the simulator (`python tests/test_edge.py`: 18/18 runs, arena 1.55 × 1.19 m). It has not been run on the real cars or with the real tracker yet.**

### Before the first run

1. **Network.** All four devices on the same LAN. Put the cars' IPs into the `lan` profile under `networks` in `config.json` and set `"network": "lan"` (or run with `ANTCOLONY_NET=lan`).
2. **VLM (computer 1).** `llama-server` with `--host 0.0.0.0 --port 8090 -np 1` (see "Arm robots' VLM"). On **each car** set `masterpi_lite/masterpi_lite/vlm_server.json` to `{"url": "http://<computer 1's new IP>:8090", "img_tokens": 256}` — the old hotel address no longer works on the new LAN.
3. **Tags.** Put the tag id that is on each car into `robots.A.tag` / `robots.B.tag` in `config.json` (defaults 10 and 11). In the tracker's `arena.json` give each car's tag its height so it is located precisely: `"objects": {"10": {"name": "car A", "height_m": 0.17}, "11": {"name": "car B", "height_m": 0.17}}` (tag face above the floor, measure it).
4. **Tracker (computer 2).** Once per machine: `python -m pip install -r cam_control/requirements.txt` (add `ultralytics` for object detection). Then `python cam_control/run_tracker.py --antcolony 127.0.0.1` (plus your usual camera options; `--antcolony <IP>` = the computer that runs the coordinator), press Start, set up the arena with its four corner tags, switch object detection on. The blocks must show up as objects. `run_tracker.py` attaches the publisher from outside, so it works with any version of the GUI and never conflicts with edits to it; the console prints `AntColony: attached ...` and a status line every 10 s.
5. **Cars.** Copy the current `robot/agent.py` and `robot/hw_masterpi.py` next to `vlm_grab.py`, then on each car: `python3 agent.py --name A --min-duty 30` (`--name B` on B; 30 on carpet).
6. **Drive calibration**, one car at a time with ~50 cm free around it: `python -m antcolony.calibrate --robot A --u 0.3 --fwd-secs 0.8 --strafe-secs 0.8` (then B). The arm cars cover about 30 cm per second on carpet, so the default 1.5 s moves are too long for a 1 m wide arena. The car is stopped at once if the tracker loses its tag. Saves `calib/A.json`. Done for A and B on 2026-11-14 (tag 8.5 cm behind the turning centre on both).
   If a car does something odd, look at one short move at a time first: `python -m antcolony.jog --robot A --w 0.3 --secs 0.4` (also `--vx`, `--vy`) prints how far the tag moved and turned.
7. **Check.** `python -m antcolony.check --overlay profiles/edge2.json --vlm http://<computer 1>:8090` — nothing moves; it lists what is missing (tracker data, a car's tag, calibration, agent, VLM, objects too big or already at the edge).

### Run

```bash
python -m antcolony.coordinator --overlay profiles/edge2.json --arena-from-tags
# dashboard: http://<coordinator computer>:8080      Ctrl+C stops both cars
```

- `--arena-from-tags`: the arena is the rectangle of the tracker's corner tags (otherwise `arena` in `config.json` must match the tracker's frame by hand).
- Put the blocks **more than 20 cm inside the boundary** (`deliver.edge_zone`): anything nearer counts as already delivered and is left alone. Keep them 30 cm or more apart if both cars should work at the same time.
- Keep feet, cables and bags out of the arena: whatever the tracker reports inside it is a target. If a car's own arm shows up as an object next to it (the check lists such cases), raise that robot's `mask_r` in `config.json` above the default 0.22 m.
- Each car parks 30 cm in front of its block, `vlm_grab.py approach` does the rest, the car carries the block to the nearest free place at the boundary, puts it down 8 cm inside it (`deliver.inset`), backs off and returns to where it started. The cars stay off the paper floor tags.
- The log shows `missed #3: ... (try 2 of 5) - trying again`, `GIVES UP on #3 after 5 missed grabs`, and at the end `FINISHED in 94 s - A: 1 delivered, B: gave up`.
- If the tracker does not detect the small blocks reliably, type their positions (tracker frame, metres; hover over them in the tracker GUI): `--objects 0.50,0.40 1.00,0.70`. The tracker's own detections are then ignored; the cars' cameras still do the final approach.

### What the first hours with the real tracker and cars showed (2026-11-14)

- **The camera looks at a slant** (1.84 m up, beside the arena). Parts of a 35 cm tall car land on the floor up to 40–60 cm behind it as seen from the camera, and the tracker reported them as objects (18 "objects" for two cars and two blocks). The tracker now sends the camera position and the coordinator ignores detections in each car's shadow (`World.hidden`). Consequences: a block in a car's shadow is invisible until the car moves, so **park the cars on the side away from the camera** and do not put blocks right behind them.
- **The arm must rest stretched forward.** `arm.home()` folds it back over the chassis, above the tag; the tag was then visible only while the camera was behind the car. `hw_masterpi.py` now uses the carry pose as its rest pose.
- Tags: A = 33, B = 36 (in `config.json`). Both are located with the plane method at their height (11–13 cm).
- Speeds: about 0.30 m/s at command 0.3; turning 70–125°/s at the smallest turn command. `profiles/edge2.json` caps the commands (`control.u_max` 0.35, `control.w_max` 0.5).

### Known gaps (found in the review, not fixable in software alone)

- **Car A's grasp is off**: A still uses car B's camera calibration (`vision_calib.json`); on 2026-11-14 its jaws came down beside the block. Run `python3 vlm_grab.py calib --object "<small flat thing>"` on A, or expect A to use up its 5 tries.
- `approach` + grab has only been run on carpet once (car A, missed) and never on car B on carpet; `agent.py` → `hw_masterpi.py` → `vlm_grab.py` as one chain has never run on a real car.
- The driving controller's gains (`control` in `config.json`) have only ever run in the simulator.
- The tracker's own "UDP out" (single-robot path following) crashes its worker thread once an object is detected (`json.dumps(..., default=float)` on a numpy outline). `--antcolony` does not use it; leave "UDP out" empty.

## Onboard vision, with AprilTag as a helper (2026-11-14)

Goal: the cars find and grab objects with their own cameras; the overhead tracker (AprilTags) helps when it can see them, and the system no longer stops when it cannot.

**1. Switchable object finder on the car** (`robot/detector.py`, `robot/grab_runner.py`). `hw_masterpi.py` now starts `grab_runner.py`, which is `vlm_grab.py` unchanged except that "find the objects in this photo" comes from `detector.json`: the laptop VLM (default, also when there is no `detector.json`) or a local YOLO model on the car (see `robot/detector.example.json`). The YOLO runs with ultralytics, onnxruntime or OpenCV's `cv2.dnn` (already on the cars, nothing to install). Tested: the real `vlm_grab.py find --dry-run` on a photo with both backends; the three engines give identical boxes; the module runs on the cars' OpenCV 4.10. Not tested yet: a trained model, speed on the Pi.

To use your own YOLO: train, `yolo export model=best.pt format=onnx imgsz=416 opset=12`, copy the `.onnx` to `models/` next to `vlm_grab.py`, write `detector.json`, check with `python3 detector.py --camera` (nothing moves). Photos for training are best taken with the cars' own camera, on the arena carpet, from the car's height.

**2. Tags seen by the car itself** (`robot/landmarks.py`): finds tag36h11 tags in the car's camera image and gives distance and direction, e.g. a tag on the bin, so a car can find the bin without the tracker. `python3 landmarks.py watch` (nothing moves). On synthetic photos: distance within 2 %, direction within 0.2 deg from 0.4 to 2.5 m. A 10 cm tag is ~24 px at 2 m and was missed once there: **use a 15–20 cm tag on the bin**. Run `calib-fx` once per car (focal length; until then it is guessed). Not yet used by any driving code.

**3. Coordinator: a hidden tag no longer stops a car** (`lost` and `pick.blind_grab_m` in `profiles/edge2.json`). Cause found on the real cars: the tag on the rear deck is hidden by the arm column whenever the car faces the slanted camera. Now
- the car lines up to grab and puts things down facing away from the camera where it can, and drives towards the camera **in reverse** (`lost.cone_deg` 40);
- a car that is unseen (also one put down facing the camera at the start) turns on the spot in short pulses until its tag shows again (at most one turn);
- a car that loses its tag on the last 60 cm to its object hands over to its own camera (`vlm_grab approach`).

In the simulator with tags hidden within ±35 deg of facing the camera: without this 1 of 8 layouts finished, with it 8 of 8, as fast as with no hiding (`tests/test_edge.py` check 4). Raising the tags above the arm (12–15 cm) is still the better fix; this keeps the cars going until then.

## One car alone: grab, carry round obstacles, put down (2026-11-14)

No tracker and no coordinator: `python -m antcolony.solo --robot B` sends one arm car three skills in a row — `grab` (what lies 25–40 cm in front of it; a miss is tried again, at most `--tries` 5), `carry` (20 cm forward, `--dist-cm`), `drop`. `--steps carry,drop` if it already holds something, `--steps status` only reads battery and sonar (nothing moves). Needs `agent.py` on the car and, for the grab, the VLM server.

`carry` is a skill in `robot/hw_masterpi.py`. The car drives in short pulses (0.33 s ≈ 5.7 cm on carpet at duty 30) and reads its front sonar while standing between them. Something nearer than the rest of the way + 15 cm (`margin_cm`: the gripper and the object stick out in front of the sonar) counts as in the way: the car slides sideways until the sonar is clear, about 10 cm more for its own half width, then drives on. If one side does not clear within 30 cm of the starting line it comes back and tries the other; if both fail it returns to the line and reports failure, still holding the object.

Limits, all from the hardware: the sonar only sees things about 10 cm tall or more, straight ahead (a second block lying on the floor is not seen); there is no sensor to the sides, so the side the car slides to must be free; the 20 cm are counted from drive pulses, not measured (expect ±20 %; `cm_s` / `side_cm_s` in the skill arguments set the assumed speeds, `--carry '{"cm_s": 15}'`).

`tests/test_carry.py` runs the skill on a fake floor (16 cases: boxes ahead, a box reaching far to one side, a wall right across, real speeds 30 % off, narrow / wide sonar beam, no sonar, cancel): the car never touches an obstacle. Not yet run on a real car.

## Current robot deployment (2026-11-14)

| Robot | Host | User | Code folder | Bench test |
| --- | --- | --- | --- | --- |
| A (arm) | catguard-a (172.20.1.113) | zxz5500 | `~/Desktop/Master Pi5/masterpi_lite/` | real hardware, wheels off the ground: all drive directions correct; VLM on the Mac: 2.37 s per new photo |
| B (arm) | catguard-b (172.20.1.120) | mitra | `~/Desktop/Master Pi5/masterpi_lite/` | real hardware, wheels off the ground: all drive directions correct; VLM on the Mac: 2.57 s per new photo |
| C (dozer) | WhiteSky (172.20.1.136) | szk13579 | `~/antcolony/` | wheels, drive, sonar stop (35 cm) tested on the floor; camera (Camera Module 3 Wide) and pan/tilt servos (PCA9685 ch 8/9) work; blade not mounted yet |

Both arm robots now run the same `vlm_grab.py` + `masterpi_lite` code (copied from B to A). Per-robot files are **not** shared and must stay on each robot: `masterpi_lite/motor_directions.json`, `masterpi_lite/arm_directions.json`, `Deviation.yaml`. A's wiring needs two options that B's code did not have, so A's copy carries two small patches ported from A's old code (`~/MasterPi/MasterPi/`, kept as a fallback):
- `chassis.py`: `wheels` in `motor_directions.json` = which board port drives which wheel: `{"FL": 3, "FR": 1, "RL": 4, "RR": 2}` (2026-11-14, backups `*.before_wheelmap`). On A, ports 1 and 3 are the right and left wheel of ONE axle, not the two wheels of one side. The earlier `mirror_lr` patch did not cover that: "rotate" drove the front axle forward and the rear axle back, so the car crept forward or back instead of turning (measured with the overhead tracker: 0.7 s "turn right" = 8 cm backwards, 2° of turn; after the fix 0.4 s = 37–50° on the spot). Driving straight never showed it because all four wheels turn the same way;
- `arm.py`: `ports` in `arm_directions.json` (servo 3 is plugged into port 2).

The originals are saved next to them as `chassis.py.orig` / `arm.py.orig`. A currently uses B's `vision_calib.json`; recalibrate A's camera if its grasps are off.
Checked 2026-11-14 by comparing checksums: A's code equals B's (19 identical files), apart from the per-robot files above and these two patches (A's `.orig` files equal B's current `chassis.py` / `arm.py`).

### Arm robots' VLM (2026-11-14)

The on-Pi VLM took 20–28 s per photo, so both arm robots now ask a VLM server on a laptop. `vlm_grab.py` and `vlm.py` are unchanged; only `masterpi_lite/masterpi_lite/vlm_server.json` was changed on each car (the old file is kept as `vlm_server.json.before_hyatt`):

```json
{"url": "http://172.20.1.112:8090", "img_tokens": 256}
```

- Server: `llama-server` (llama.cpp) with `ggml-org/Qwen2.5-VL-3B-Instruct-GGUF` Q4_K_M + its Q8_0 mmproj on the MacBook Air M4 at `172.20.1.112:8090` (hotel network). Use llama-server, not Ollama: Ollama upscales small photos to ~1000 image tokens (7 s per photo) and its pixel boxes then refer to the upscaled picture.
- `img_tokens 256` instead of 400: 0.6 s faster per photo, same detections, boxes within 5 px. Measured from the cars: ~2.4–2.6 s per new photo (image encoding + prefill ~1.4 s, ~41 answer tokens ~0.9 s, Wi-Fi ~0.07 s).
- With two cars on one server, start it with `-np 1` (queue): two simultaneous requests are answered at 2.5 s and 4.1 s; with the default 4 slots both take 5.4–6.4 s.
- When the network changes (own router), update `url` on both cars to the laptop's new IP. The laptop must stay awake and keep that IP.
- Full measurements, the photo test set and the benchmark scripts: branch `bench`, `bench/TIMING.md`.

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
5. Prop the car up so the wheels are off the ground and run `python3 hw_freenove.py wheels` to confirm each wheel's direction one at a time (run it in your own SSH terminal; it asks y/n). **Front = the end with the ultrasonic sensor and the camera, where the blade goes.** Then put it on the floor and run `python3 hw_freenove.py drive`; it should go forward, backward, turn left, and turn right in that order.
   If the whole car drives the wrong way (all wheels agreed, but "forward" moves away from the sensor), do not reverse all four wheels — that also mirrors turning. Set `"flip_front": true` in `freenove_directions.json` instead; it flips only forward/back. Car C has this set (2026-11-14).
6. Start the robot service: `python3 agent.py --name C --hw freenove --min-duty 35 --rot-min 55 --stop-cm 35`.
   - `--stop-cm 35`: the front ultrasonic (GPIO 27 trigger / 22 echo) blocks forward motion below this distance, except while pushing (`guard: false`). Measured on car C at min-duty 35 (~90 cm/s): about 14 cm of coasting after the stop, so 35 cm left it 10 cm from a wall; 20 cm let it touch the wall. Check `python3 hw_freenove.py sonar` after mounting the blade: a constant few cm means the sensor sees the blade, then add `--no-sonar`.
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
antcolony/check.py        pre-flight check: what is missing before a real run (nothing moves)
antcolony/jog.py          one short move of one robot, measured by the tracker
antcolony/solo.py         one arm car alone: grab -> carry round obstacles -> drop (no tracker)
robot/detector.py         car: object finder, laptop VLM or local YOLO (detector.json)
robot/grab_runner.py      car: vlm_grab.py with that object finder
robot/landmarks.py        car: AprilTags in the car's own camera (distance, direction)
profiles/edge2.json       overlay for the two-car "one block each to the arena edge" test
cam_control/              overhead tracker (AprilTag floor frame, YOLOE objects); antcolony_pub.py sends it to the coordinator
antcolony/sim.py          simulator (plays both camera and robots)
robot/agent.py            on-robot service (watchdog, skills, telemetry)
robot/hw_masterpi.py      arm robot: adapter for masterpi_lite + vlm_grab.py
robot/hw_freenove.py      dozer: Freenove FNK0043, direct I2C; built-in wheel direction test
tracker_bridge/world_pub.py  tracker → coordinator publisher
tests/test_sim.py         end-to-end test
tests/test_edge.py        end-to-end test of the edge task (one block each, 5 tries, give up)
tests/test_carry.py       the carry skill (sonar obstacle avoidance) on a fake floor
```

## Messages (each is one JSON object over UDP)

```
World snapshot → host:9100   {"t":..,"tags":{"10":[x,y,heading_rad],..},"objects":[{"id","label","x","y","w","l","ang"},..],
                              "floor":{"29":[x,y],..},"corners":[29,32,31,30]}     floor / corners are optional
Velocity       → robot:9200  {"type":"vel","vx":fwd,"vy":left,"w":ccw,"dur":sec,"guard":true}   each in -1..1, auto-stops after at most 0.4 s
Skill          → robot:9200  {"type":"skill","id":17,"name":"grab|drop|home|open|stop","args":{..}}   resending the same id is harmless
Telemetry      → host:9201   {"type":"telem","name":"A","batt_mv":..,"sonar_cm":..,"skill":{"id","state","detail"},"holding":..}
```

The dozer only uses the `vel` message; `vy` is ignored because it cannot strafe.

## Three things to verify before running on real robots

1. Whether the gripper is still holding after the serial port is reopened at the end of `vlm_grab.py approach` (`hw_masterpi.py` opens and closes the serial port once before and once after grabbing).
2. Whether the carry pose `CARRY` and the release position (`drop.y_cm`, `drop.z_cm`) clear your bin rim.
3. Whether `--min-duty` and `--rate-min` (arm robots; `--rot-min` on the dozer) are the minimum values that just get the robot moving on the competition floor; `control.rot_rate0` corrects itself at runtime.
