"""Live arm control of several cars (antcolony/twin.py) and the car side of it (robot/hw_masterpi.py
arm_step), without cars, keyboard or network.

    python tests/test_twin.py
"""
import math
import os
import sys
import threading

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "robot"))
from antcolony import twin                               # noqa: E402
import hw_masterpi                                       # noqa: E402

bad = 0


def check(name, cond, info=""):
    global bad
    bad += not cond
    print(f"{'ok  ' if cond else 'FAIL'} {name}" + (f"  [{info}]" if info and not cond else ""))


# ---------------------------------------------------------------- Mac side
sent = []
t = twin.Twin(["A", "B"], lambda n, m: sent.append((n, m)), every=0.08)
t.tick(0.0)
first = [m for n, m in sent]
check("the first pose goes to both cars, the same numbers, slowly", len(first) == 2 and first[0] == first[1]
      and first[0]["name"] == "arm" and first[0]["args"]["ms"] == 900, first)
sent.clear()
for k in ["up", "up", "left", "w"]:
    t.key(k)
t.tick(0.05)
check("no second send within 80 ms", not sent)
t.tick(0.10)
a, b = sent[0][1]["args"], sent[1][1]["args"]
check("after keys: one send, identical for A and B", len(sent) == 2 and a == b, sent)
check("up up = reach +2 cm, left = turn +3 deg (tip goes to the left: x < 0), w = 1 cm higher",
      abs(t.pose["reach"] - 15) < 1e-9 and t.pose["turn"] == 3 and t.pose["z"] == 13 and a["x"] < 0,
      (t.pose, a))
check("reach and turn give the right tip place", abs(math.hypot(a["x"], a["y"]) - 15) < 0.02)
sid = sent[-1][1]["id"]
t.result("A", sid, "ok", "pitch -60")
t.result("B", sid, "ok", "pitch -60")
check("both ok: this pose becomes the fallback", t.good == t.pose)
good = dict(t.good)
sent.clear()
for _ in range(12):
    t.key("up")
check("reach stops at its limit (22 cm)", t.pose["reach"] == 22.0, t.pose)
t.tick(0.3)
sid = sent[-1][1]["id"]
t.result("A", sid, "ok", "")
t.result("B", sid, "fail", "out of reach: (0.0, 22.0, 13.0)")
check("one car cannot reach: everyone goes back to the last pose all reached", t.pose == good and t.dirty, t.pose)
sent.clear()
t.tick(0.5)
check("... and that pose is sent to both", len(sent) == 2 and sent[0][1]["args"] == sent[1][1]["args"]
      and abs(math.hypot(sent[0][1]["args"]["x"], sent[0][1]["args"]["y"]) - 15) < 0.02)
t.result("A", sid, "ok", "")
check("a late answer to an abandoned pose changes nothing", t.pose == good)
sent.clear()
t.key(" ")
t.tick(1.0)
check("space opens the jaws on both", [m["args"]["grip"] for n, m in sent] == ["open", "open"])
t.key("2")
t.key("s")
check("2 = big steps (3 cm down)", t.pose["z"] == 13 - 3)
sent.clear()
t.key("h")
check("h: driving pose on both, the shared pose starts over", [m["name"] for n, m in sent] == ["home", "home"]
      and t.pose == twin.START and t.jaws == "close")
check("x quits", t.key("x") is False and t.key("ctrl-c") is False and t.key("w") is True)

r, w = os.pipe()
os.write(w, b"\x1b[A\x1b[Dwq \x03")
check("keys: arrows, letters, space, Ctrl-C", twin.read_keys(r, 0.1) == ["up", "left", "w", "q", " ", "ctrl-c"],
      twin.read_keys(r, 0.1))


# ---------------------------------------------------------------- car side
class FakeArm:
    def __init__(self, reach=30.0):
        self.calls, self.reach = [], reach

    def move_to(self, x, y, z, pitch=0, pitch_min=-90, pitch_max=90, ms=None):
        self.calls.append((x, y, z, pitch, pitch_min, pitch_max, ms))
        return None if math.hypot(x, y) > self.reach else ({}, max(pitch_min, min(pitch_max, pitch)), ms)

    def gripper(self, closed, ms=500):
        self.calls.append(("grip", closed, ms))


class Car(hw_masterpi.MasterPi):
    def __init__(self, reach=30.0):
        self.use_arm, self.holding, self.lock, self.arm = True, True, threading.RLock(), FakeArm(reach)


real = hw_masterpi.reach_offset
try:
    hw_masterpi.reach_offset = lambda: 0.0
    c = Car()
    ok, d = c.skill("arm", {"x": -3.0, "y": 14.0, "z": 10, "pitch": -45, "ms": 200, "grip": "open"}, threading.Event())
    x, y, z, p, p0, p1, ms = c.arm.calls[0]
    check("car B (no reach offset): the tip goes exactly where it is sent", ok and (x, y, z, p, ms) == (-3.0, 14.0, 10.0, -45.0, 200), c.arm.calls)
    check("... jaws opened, 'holding' cleared", c.arm.calls[1] == ("grip", False, 200) and not c.holding and "jaws open" in d, d)
    hw_masterpi.reach_offset = lambda: 1.8
    c = Car()
    ok, d = c.skill("arm", {"x": -3.0, "y": 14.0, "z": 10}, threading.Event())
    x, y = c.arm.calls[0][:2]
    check("car A (reach offset 1.8 cm): sent 1.8 cm farther along the same line",
          ok and abs(math.hypot(x, y) - math.hypot(-3, 14) - 1.8) < 1e-6 and abs(math.atan2(x, y) - math.atan2(-3, 14)) < 1e-9, (x, y))
    c = Car(reach=10.0)
    ok, d = c.skill("arm", {"x": 0.0, "y": 20.0, "z": 10}, threading.Event())
    check("out of reach: fails and says so (nothing else is sent)", not ok and "out of reach" in d and len(c.arm.calls) == 1, d)
finally:
    hw_masterpi.reach_offset = real

print("all ok" if not bad else f"{bad} FAILED")
sys.exit(1 if bad else 0)
