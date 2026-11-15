"""antcolony.duo with two fake cars: who takes what, and one car at a time at the tray.

    python tests/test_duo.py
"""
import os
import sys
import threading
import time
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from antcolony import duo

LOG, LOCK = [], threading.Lock()


class FakeCar:
    """Sees coloured things at distances (cm); 'find' takes the nearest allowed one."""

    def __init__(self, name, sees, find_s=0.05, grab_fail=0, seek_ok=True):
        self.name, self.sees, self.find_s, self.grab_fail, self.seek_ok = name, sees, find_s, grab_fail, seek_ok
        self.at_tray = False

    def skill(self, skill, args, timeout):
        with LOCK:
            LOG.append((self.name, skill, dict(args)))
        if skill == "seek" and args.get("detector") == "colour":
            time.sleep(self.find_s)
            skip = [w for w in ("red", "blue", "green") if "not " + w in args["target"]]
            c = sorted((d, k) for k, d in self.sees.items() if k not in skip)
            return (True, f"close to the object; colour={c[0][1]}", 1) if c else (False, "'object' not found in a full turn (16 looks)", 1)
        if skill == "grab":
            if self.grab_fail > 0:
                self.grab_fail -= 1
                return False, "missed", 1
            return True, "done - holding", 1
        if skill == "seek":
            if not self.seek_ok:
                return False, "'white tray' not found", 1
            self.at_tray = True
            others = [c for c in CARS if c is not self and c.at_tray]
            assert not others, f"{self.name} went to the tray while {others[0].name} was there"
            time.sleep(0.05)
            return True, "in front of the white tray", 1
        if skill == "dock":
            time.sleep(0.05)
            return True, "in the tray", 1
        return True, "ok", 1

    def back_up(self, cm):
        with LOCK:
            LOG.append((self.name, "back", cm))
        if self.at_tray and cm > 30:
            self.at_tray = False


ARGS = types.SimpleNamespace(noun="block", obj_height=3.0, tries=3, back_cm=18, find_bottom=0.4, target="white tray",
                             seek="{}", place_z=10, away_cm=35, choose_s=5)
CARS = []


def play(a_car, b_car):
    global CARS
    LOG.clear()
    CARS = [a_car, b_car]
    team, res = duo.Team(), {}
    ths = [threading.Thread(target=lambda: res.__setitem__("A", duo.run_car("A", a_car, team, ARGS, lambda m: None))),
           threading.Thread(target=lambda: res.__setitem__("B", duo.run_car("B", b_car, team, ARGS, lambda m: None, "A")))]
    [t.start() for t in ths]
    [t.join(10) for t in ths]
    grabbed = {n: next((x[2]["label"] for x in LOG if x[0] == n and x[1] == "grab"), None) for n in "AB"}
    return res, grabbed, team


bad = 0


def check(name, cond, info=""):
    global bad
    bad += not cond
    print(f"{'ok  ' if cond else 'FAIL'} {name} {info}")


res, g, team = play(FakeCar("A", {"red": 50, "blue": 90}), FakeCar("B", {"red": 40, "blue": 80}))
check("both nearest to red: A takes red, B takes blue", res == {"A": True, "B": True} and g == {"A": "red block", "B": "blue block"}, g)
res, g, team = play(FakeCar("A", {"red": 90, "blue": 50}), FakeCar("B", {"red": 40, "blue": 80}))
check("A nearest to blue, B nearest to red: no conflict", g == {"A": "blue block", "B": "red block"} and all(res.values()), g)
res, g, team = play(FakeCar("A", {"red": 50, "blue": 90}, find_s=0.4), FakeCar("B", {"red": 40, "blue": 80}))
check("B waits for A's (slow) choice", g == {"A": "red block", "B": "blue block"}, g)
res, g, team = play(FakeCar("A", {}), FakeCar("B", {"red": 40}))
check("A finds nothing: B is free to choose", res == {"A": False, "B": True} and g["B"] == "red block", (res, g))
res, g, team = play(FakeCar("A", {"red": 50}), FakeCar("B", {"red": 40}))
check("only one object: B finds nothing, does not take A's", res == {"A": True, "B": False} and g["B"] is None, (res, g))
res, g, team = play(FakeCar("A", {"red": 50, "blue": 90}, grab_fail=5), FakeCar("B", {"red": 40, "blue": 80}))
check("A never holds its object: B still delivers, the tray is not blocked", res == {"A": False, "B": True}, res)
res, g, team = play(FakeCar("A", {"red": 50, "blue": 90}, grab_fail=1), FakeCar("B", {"red": 40, "blue": 80}))
check("A misses once, backs up, holds on try 2", res == {"A": True, "B": True} and ("A", "back", 18) in LOG, res)
res, g, team = play(FakeCar("A", {"red": 50, "blue": 90}, seek_ok=False), FakeCar("B", {"red": 40, "blue": 80}))
check("A cannot find the tray: gives the turn back, B delivers", res == {"A": False, "B": True}, res)
print("all ok" if not bad else f"{bad} failed")
sys.exit(1 if bad else 0)
