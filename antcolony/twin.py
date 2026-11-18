"""Live keyboard control of the ARMS of several cars at once - every car makes the same move.

    python -m antcolony.twin                    # cars A and B (addresses from config.json)
    python -m antcolony.twin --robots B         # one car
    python -m antcolony.twin --addr A=192.168.0.63 --addr B=192.168.0.64

Run it in your own terminal (it reads single key presses). Only the arms move, the wheels do not.

    left / right    turn the arm left / right          up / down    reach out / in
    w / s           tip up / down                      q / e        tilt the gripper up / down
    space           open / close the jaws              1 / 2        small / big steps
    h               driving pose (jaws closed)         x, Ctrl-C    driving pose and quit

The same numbers go to every car: the gripper tip's place (cm from the arm base) and its tilt. Each car
turns them into its own servo values with its own calibration (A and B are calibrated differently, so
the same raw servo values would NOT give the same pose), and adds its own reach offset (car A reaches
1.8 cm short, robot/grab_defaults.A.json). If any car cannot reach a place, every car goes back to the
last place all of them reached - the arms never part ways. Steps go out at most 12 times a second.

Needs agent.py with the "arm" skill on every car (robot/hw_masterpi.py, 2026-11-14) and UDP 9201 on this
computer: no coordinator / solo / duo for these cars at the same time - a new command also cancels
whatever a car is doing.
"""
import argparse
import json
import math
import os
import select
import socket
import sys
import threading
import time

from .common import load_config

PORT_CMD, PORT_TELEM = 9200, 9201
HELP = """
    left / right    turn the arm left / right          up / down    reach out / in
    w / s           tip up / down                      q / e        tilt the gripper up / down
    space           open / close the jaws              1 / 2        small / big steps
    h               driving pose (jaws closed)         x, Ctrl-C    driving pose and quit
"""
START = {"turn": 0.0, "reach": 13.0, "z": 12.0, "pitch": -60.0}       # near the cars' driving pose
LIMITS = {"turn": (-90.0, 90.0), "reach": (5.0, 22.0), "z": (0.0, 24.0), "pitch": (-90.0, 20.0)}
STEPS = {"fine": {"turn": 3.0, "reach": 1.0, "z": 1.0, "pitch": 5.0},
         "big": {"turn": 10.0, "reach": 3.0, "z": 3.0, "pitch": 15.0}}
KEYS = {"left": ("turn", +1), "right": ("turn", -1), "up": ("reach", +1), "down": ("reach", -1),
        "w": ("z", +1), "s": ("z", -1), "q": ("pitch", +1), "e": ("pitch", -1)}


class Twin:
    """The arm pose all cars share, and the rule that keeps them together. No keyboard, no sockets:
    send(name, msg) and telem results come in from outside (tested in tests/test_twin.py)."""

    def __init__(self, names, send, every=0.08, ms=250):
        self.names, self.send, self.every, self.ms = list(names), send, every, ms
        self.pose, self.good = dict(START), dict(START)          # wanted now / last one every car reached
        self.jaws, self.step = "close", "fine"
        self.sid, self.sent_at, self.dirty = int(time.time()) % 1_000_000 * 10, None, True
        self.pending = {}                                        # skill id -> (pose, {name: state})
        self.status = {n: "-" for n in self.names}
        self.note = ""

    def target(self):
        t = math.radians(self.pose["turn"])
        r = self.pose["reach"]
        return {"x": round(-r * math.sin(t), 2), "y": round(r * math.cos(t), 2), "z": round(self.pose["z"], 2),
                "pitch": round(self.pose["pitch"], 1)}

    def key(self, k):
        """One key press. -> False to quit."""
        if k in KEYS:
            what, sign = KEYS[k]
            lo, hi = LIMITS[what]
            self.pose[what] = max(lo, min(hi, self.pose[what] + sign * STEPS[self.step][what]))
            self.dirty = True
        elif k == " ":
            self.jaws = "open" if self.jaws == "close" else "close"
            self.dirty = True
        elif k in ("1", "2"):
            self.step = "fine" if k == "1" else "big"
        elif k == "h":
            self.home()
        elif k in ("x", "ctrl-c"):
            return False
        return True

    def home(self):
        self.sid += 1
        for n in self.names:
            self.send(n, {"type": "skill", "id": self.sid, "name": "home", "args": {}})
        self.pose, self.good, self.jaws, self.dirty = dict(START), dict(START), "close", False
        self.pending.clear()
        self.note = "driving pose"

    def tick(self, now):
        """Send the newest pose if it changed and the last send is long enough ago."""
        if not self.dirty or (self.sent_at is not None and now - self.sent_at < self.every):
            return
        self.sid += 1
        args = dict(self.target(), ms=self.ms if self.sent_at is not None else 900, grip=self.jaws)   # the first move: slowly
        for n in self.names:
            self.send(n, {"type": "skill", "id": self.sid, "name": "arm", "args": args})
        self.pending[self.sid] = (dict(self.pose), {})
        for old in [i for i in self.pending if i < self.sid - 50]:
            del self.pending[old]
        self.sent_at, self.dirty = now, False

    def result(self, name, sid, state, detail):
        """A car's answer for skill `sid` (from its telemetry)."""
        if sid not in self.pending or state not in ("ok", "fail"):
            return
        pose, got = self.pending[sid]
        if name in got:
            return
        got[name] = state
        self.status[name] = "ok" if state == "ok" else "out of reach"
        if state == "fail":                                      # this car cannot: everyone back to the last good pose
            self.pose, self.dirty = dict(self.good), True
            self.note = f"{name}: {detail[:40]} - all back"
            for i in [i for i in self.pending if i <= sid]:      # answers to older poses no longer count
                del self.pending[i]
        elif len(got) == len(self.names) and all(v == "ok" for v in got.values()):
            self.good = dict(pose)
            if not self.note.startswith("driving"):
                self.note = ""

    def line(self):
        p = self.pose
        cars = "  ".join(f"{n}:{self.status[n]}" for n in self.names)
        return (f"{cars} | turn {p['turn']:+4.0f}  reach {p['reach']:4.1f}  height {p['z']:4.1f}  tilt {p['pitch']:+4.0f}"
                f"  jaws {self.jaws:5s} | {self.step:4s} {self.note}")


class Net:
    """Commands to the cars' agents, their telemetry back (one socket on UDP 9201 for all cars)."""

    def __init__(self, addrs):
        self.addrs = addrs
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.rx.bind(("0.0.0.0", PORT_TELEM))
        self.rx.settimeout(0.2)
        self.ips = {socket.gethostbyname(a[0]): n for n, a in addrs.items()}
        self.latest, self.on_result, self.alive = {}, None, True
        threading.Thread(target=self.loop, daemon=True).start()

    def send(self, name, msg):
        try:
            self.tx.sendto(json.dumps(msg).encode(), self.addrs[name])
        except OSError:
            pass

    def loop(self):
        while self.alive:
            try:
                data, src = self.rx.recvfrom(65535)
                m = json.loads(data.decode())
            except (socket.timeout, ValueError, OSError):
                continue
            name = self.ips.get(src[0])
            if not isinstance(m, dict) or name is None or m.get("name") != name:
                continue
            self.latest[name] = m
            sk = m.get("skill") or {}
            if self.on_result and sk.get("id") is not None:
                self.on_result(name, sk["id"], sk.get("state"), sk.get("detail", ""))

    def wait_for_all(self, secs=3.0):
        t_end = time.monotonic() + secs
        while time.monotonic() < t_end and len(self.latest) < len(self.addrs):
            for n in self.addrs:
                self.send(n, {"type": "vel", "vx": 0, "vy": 0, "w": 0, "dur": 0.1})
            time.sleep(0.2)
        return {n: self.latest.get(n) for n in self.addrs}


def read_keys(fd, timeout):
    """Key names pressed within `timeout` s: arrows as left/right/up/down, others as typed."""
    if not select.select([fd], [], [], timeout)[0]:
        return []
    data = os.read(fd, 64).decode(errors="ignore")
    keys, i = [], 0
    arrows = {"A": "up", "B": "down", "C": "right", "D": "left"}
    while i < len(data):
        if data[i] == "\x1b" and i + 2 < len(data) and data[i + 1] in "[O":
            keys.append(arrows.get(data[i + 2], ""))
            i += 3
        elif data[i] == "\x03":
            keys.append("ctrl-c")
            i += 1
        else:
            keys.append(data[i].lower())
            i += 1
    return [k for k in keys if k]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--robots", default="A,B", help="cars to move together, comma-separated")
    ap.add_argument("--addr", action="append", default=[], help="NAME=IP, if not the address in config.json")
    ap.add_argument("--ms", type=int, default=250, help="time for one step, ms (smaller = snappier, jerkier)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    names = [n.strip() for n in a.robots.split(",") if n.strip()]
    addrs = {n: tuple(cfg["robots"][n]["addr"]) for n in names}
    for it in a.addr:
        n, ip = it.split("=")
        addrs[n] = (ip, PORT_CMD)
    if not sys.stdin.isatty():
        sys.exit("run this in a terminal: it reads single key presses")
    net = Net(addrs)
    seen = net.wait_for_all()
    for n, t in seen.items():
        if t is None:
            sys.exit(f"car {n} does not answer at {addrs[n][0]} - is agent.py running there?")
        b = t.get("batt_mv")
        print(f"  car {n}: battery {b / 1000:.2f} V" if b else f"  car {n}: battery ?")
    twin = Twin(names, net.send, ms=a.ms)
    net.on_result = twin.result
    print(HELP)
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        twin.dirty, last = True, ""
        while True:
            for k in read_keys(fd, 0.03):
                if not twin.key(k):
                    raise KeyboardInterrupt
            twin.tick(time.monotonic())
            line = twin.line()
            if line != last:
                sys.stdout.write("\r" + line.ljust(118)[:118])
                sys.stdout.flush()
                last = line
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        twin.home()
        time.sleep(0.3)
        net.alive = False
        print("\nboth arms back to the driving pose - bye")


if __name__ == "__main__":
    main()
