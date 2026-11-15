#!/usr/bin/env python3
"""On-robot agent: the only program the coordinator talks to on each car.

    python3 agent.py --name A                      # MasterPi arm car (run it inside masterpi_lite/)
    python3 agent.py --name C --hw freenove        # bulldozer: Freenove FNK0043 4WD car
    python3 agent.py --name A --hw mock            # no hardware: prints what it would do

Arm cars: copy agent.py and hw_masterpi.py next to vlm_grab.py (~/Desktop/"Master Pi5"/masterpi_lite/).
Bulldozer: copy agent.py and hw_freenove.py into any folder on its Pi.

Protocol (JSON datagrams, coordinator -> UDP 9200, telemetry -> coordinator UDP 9201):
  {"type":"vel","vx":..,"vy":..,"w":..,"dur":0.2,"guard":true}
        normalised -1..1 (forward, left, counter-clockwise). Runs for `dur` seconds (default and
        maximum = the 0.4 s watchdog), then the car stops by itself. Ignored while a skill runs.
  {"type":"skill","id":17,"name":"grab","args":{...}}     names: grab, drop, carry, seek, home, open, stop
        carry (arm cars): {"dist_cm": 20} forward with the object, round tall things (front sonar)
        seek (arm cars): {"target": "white box", "stop_cm": 22} find it with the camera + VLM, drive up to it
        arm (arm cars): {"x": 0, "y": 15, "z": 10, "pitch": -45, "ms": 250, "grip": "open"} one step of live arm control
        clean (dozer C): {"minutes": 15} push every piece of irregular trash into the open cardboard box
              (robot/cleaner.py); detail at the end: "<n> pieces in the box"
  {"type":"peers","peers":[{"name":"A","dist":0.8,"bearing_deg":30}, ..]}
        where the other cars are, relative to this one (from the overhead camera, antcolony.clear --dozer).
        Written to PEERS_FILE for cleaner.py: it keeps away from them and never takes them for trash.
        Re-sending the same id is harmless. A new id cancels a running skill.
  telemetry, 10 Hz: {"type":"telem","name":"A","batt_mv":7900,"sonar_cm":35.0,
                     "skill":{"id":17,"state":"running|ok|fail","detail":".."},"holding":false}
"""
import argparse
import json
import signal
import socket
import sys
import threading
import time

PORT_CMD, PORT_TELEM, WATCHDOG = 9200, 9201, 0.4
PEERS_FILE = "/tmp/antcolony_peers.json"     # the other cars' positions, for a skill running in another process


class MockHW:
    """Stand-in hardware for bench tests: prints instead of moving."""
    holding = False

    def set_velocity(self, vx, vy, w):
        print(f"  vel {vx:+.2f} {vy:+.2f} {w:+.2f}")

    def stop(self):
        pass

    def battery_mv(self):
        return 7900

    def sonar_cm(self):
        return None

    def skill(self, name, args, cancel):
        print(f"  skill {name} {args}")
        for _ in range(20):
            if cancel.is_set():
                return False, "cancelled"
            time.sleep(0.1)
        if name == "grab":
            self.holding = True
        if name in ("drop", "open"):
            self.holding = False
        return True, "mock"

    def close(self):
        pass


class Agent:
    def __init__(self, name, hw, port, stop_cm):
        self.name, self.hw, self.stop_cm = name, hw, stop_cm
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", port))
        self.sock.settimeout(0.02)
        self.peer = None                       # the coordinator = whoever sent the last command
        self.vel, self.vel_until, self.moving = (0.0, 0.0, 0.0, True), 0.0, False
        self.skill = {"id": 0, "state": "idle", "detail": ""}
        self.worker, self.cancel = None, threading.Event()
        self.applied, self.applied_t = None, 0.0

    def busy(self):
        return self.worker is not None and self.worker.is_alive()

    def on_peers(self, m):
        """Store where the other cars are (atomic write: the clean skill reads it at any time)."""
        try:
            import os
            tmp = PEERS_FILE + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"t": time.time(), "peers": m.get("peers", [])}, f)
            os.replace(tmp, PEERS_FILE)
        except OSError:
            pass

    def on_msg(self, m, now):
        if m.get("type") == "peers":
            self.on_peers(m)
            return
        if m.get("type") == "vel":
            if not self.busy():
                dur = min(float(m.get("dur", WATCHDOG)), WATCHDOG)
                self.vel = (float(m.get("vx", 0)), float(m.get("vy", 0)), float(m.get("w", 0)), m.get("guard", True))
                self.vel_until = now + dur
        elif m.get("type") == "skill" and m.get("id") != self.skill["id"]:
            if self.busy():
                self.cancel.set()
                self.worker.join(timeout=5)
            self.hw.stop()
            self.moving, self.vel_until = False, 0.0
            self.skill = {"id": m["id"], "state": "running", "detail": ""}
            self.cancel = threading.Event()
            self.worker = threading.Thread(target=self.run_skill, args=(dict(self.skill), m, self.cancel), daemon=True)
            self.worker.start()

    def run_skill(self, mine, m, cancel):
        try:
            ok, detail = (True, "stopped") if m["name"] == "stop" else self.hw.skill(m["name"], m.get("args", {}), cancel)
        except Exception as e:                 # never let a skill kill the agent
            ok, detail = False, f"{type(e).__name__}: {e}"
        if self.skill["id"] == mine["id"]:
            self.skill = {"id": mine["id"], "state": "ok" if ok else "fail", "detail": str(detail)[:120]}
        print(f"[{self.name}] skill {m['name']} -> {'ok' if ok else 'FAIL'} {detail}")

    def drive(self, now):
        if self.busy():
            return
        if now < self.vel_until:
            vx, vy, w, guard = self.vel
            if guard and vx > 0 and self.stop_cm:
                d = self.hw.sonar_cm()
                if d is not None and d < self.stop_cm:
                    vx = 0.0                   # something tall right ahead: no forward motion
            if (vx, vy, w) != self.applied or now - self.applied_t > 0.1:   # do not flood the serial port
                self.hw.set_velocity(vx, vy, w)
                self.applied, self.applied_t = (vx, vy, w), now
            self.moving = True
        elif self.moving:
            self.hw.stop()                     # watchdog / end of a timed pulse
            self.moving, self.applied = False, None

    def loop(self):
        next_telem = 0.0
        print(f"[{self.name}] agent listening on UDP {self.sock.getsockname()[1]}")
        while True:
            try:
                data, addr = self.sock.recvfrom(65535)
                m = json.loads(data.decode())
                self.peer = addr[0]
                self.on_msg(m, time.monotonic())
            except socket.timeout:
                pass
            except (ValueError, KeyError, TypeError):
                pass
            now = time.monotonic()
            self.drive(now)
            if self.peer and now >= next_telem:
                next_telem = now + 0.1
                t = {"type": "telem", "name": self.name, "skill": self.skill, "holding": bool(self.hw.holding),
                     "batt_mv": None if self.busy() else self.hw.battery_mv(),
                     "sonar_cm": None if self.busy() else self.hw.sonar_cm()}
                try:
                    self.sock.sendto(json.dumps(t).encode(), (self.peer, PORT_TELEM))
                except OSError:
                    pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="robot name as in config.json (A, B, C)")
    ap.add_argument("--hw", default="masterpi", choices=["masterpi", "freenove", "mock"])
    ap.add_argument("--port", type=int, default=PORT_CMD)
    ap.add_argument("--no-arm", action="store_true", help="masterpi used without its arm")
    ap.add_argument("--min-duty", type=float, default=None,
                    help="duty (percent) that just moves the car straight on this floor. "
                         "Default: masterpi 30 (carpet; 18 on smooth floors), freenove 35")
    ap.add_argument("--max-duty", type=float, default=None, help="default: masterpi 45, freenove 65")
    ap.add_argument("--rate-min", type=float, default=0.20, help="masterpi: smallest in-place turn rate that moves the car")
    ap.add_argument("--rate-max", type=float, default=0.36, help="masterpi")
    ap.add_argument("--rot-min", type=float, default=55, help="freenove: smallest duty (percent) that turns the car on the spot")
    ap.add_argument("--rot-max", type=float, default=80, help="freenove")
    ap.add_argument("--pcb", type=int, default=2, choices=[1, 2], help="freenove: shield PCB version (battery reading only)")
    ap.add_argument("--stop-cm", type=float, default=8, help="sonar emergency stop distance, 0 = off")
    ap.add_argument("--no-sonar", action="store_true",
                    help="freenove: do not use the front ultrasonic (e.g. when the blade blocks it)")
    a = ap.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))          # `kill` also stops the motors (see finally)
    if a.hw == "mock":
        hw = MockHW()
    elif a.hw == "freenove":
        from hw_freenove import Freenove
        hw = Freenove(a.min_duty or 35, a.max_duty or 65, a.rot_min, a.rot_max, pcb=a.pcb, sonar=not a.no_sonar)
    else:
        from hw_masterpi import MasterPi
        hw = MasterPi(a.min_duty or 30, a.max_duty or 45, a.rate_min, a.rate_max, arm=not a.no_arm)
    agent = Agent(a.name, hw, a.port, a.stop_cm)
    try:
        agent.loop()
    except KeyboardInterrupt:
        pass
    finally:
        hw.stop()
        hw.close()
        print("\nstopped")


if __name__ == "__main__":
    main()
