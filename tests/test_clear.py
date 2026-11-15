"""antcolony.clear with two fake cars: the floor gets cleared, claims, tray turns, giving up, the fence.

    python tests/test_clear.py
"""
import os
import sys
import threading
import time
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from antcolony import clear

LOCK = threading.Lock()
COLOURS = ("red", "blue", "green", "yellow")


class World:
    def __init__(self, objects):
        self.objects, self.tray, self.at_tray, self.both_at_tray, self.same_time, self.seeks = list(objects), [], set(), False, False, []
        self.working = {}


class FakeCar:
    """sees: colour -> distance for this car. find takes the nearest allowed colour still on the floor."""

    def __init__(self, name, world, dist, cannot_grab=(), leave_after=None):
        self.name, self.w, self.dist, self.cannot, self.leave_after = name, world, dist, set(cannot_grab), leave_after
        self.out, self.hold, self.n = threading.Event(), None, 0

    def skill(self, skill, args, timeout):
        if self.out.is_set():
            return False, "outside the arena - stopped", 0
        self.n += 1
        if self.leave_after and self.n > self.leave_after:
            self.out.set()
            return False, "outside the arena - stopped", 0
        time.sleep(0.01)
        with LOCK:
            if skill == "seek" and args.get("detector") == "colour":
                skip = [c for c in COLOURS if "not " + c in args["target"]]
                want = next((c for c in COLOURS if c in args["target"] and c not in skip), None)
                c = sorted((self.dist.get(o, 99), o) for o in set(self.w.objects) if o not in skip and (want is None or o == want))
                if not c:
                    return False, "'object' not found in a full turn (16 looks)", 1
                other = [v for k, v in self.w.working.items() if k != self.name]
                if want is None:
                    if c[0][1] in other:
                        self.w.same_time = True
                    self.w.working[self.name] = c[0][1]
                return True, f"close to the object; colour={c[0][1]}", 1
            if skill == "grab":
                col = args["label"].split()[0]
                if col in self.cannot or col not in self.w.objects:
                    return False, "missed", 1
                self.w.objects.remove(col)
                self.hold = col
                return True, "done - holding", 1
            if skill == "seek":
                self.w.seeks.append((self.name, args.get("arrive_cm")))
                if args.get("arrive_cm", 40) > 60:           # told to stop short: not at the tray yet
                    return True, "in front of the white tray: about 80 cm", 1
                self.near = True
                return True, "in front of the white tray", 1
            if skill == "dock":
                if self.w.at_tray - {self.name}:
                    self.w.both_at_tray = True
                self.w.at_tray.add(self.name)
                time.sleep(0.05)
                self.w.tray.append(self.hold)
                self.hold = None
                self.w.working.pop(self.name, None)
                return True, "in the tray", 1
        return True, "ok", 1

    def back_up(self, cm, *a):
        with LOCK:
            if cm > 30:
                self.w.at_tray.discard(self.name)
            else:
                self.w.working.pop(self.name, None)


ARGS = types.SimpleNamespace(noun="block", obj_height=3.0, tries=2, back_cm=18, find_bottom=0.4, target="white tray",
                             seek="{}", place_z=10, away_cm=35, choose_s=5, queue_cm=80)


def play(objects, a_kw=None, b_kw=None, da=None, db=None):
    w = World(objects)
    A = FakeCar("A", w, da or {}, **(a_kw or {}))
    B = FakeCar("B", w, db or {}, **(b_kw or {}))
    team, res = clear.Team(), {}
    ths = [threading.Thread(target=lambda: res.__setitem__("A", clear.run_car("A", A, team, ARGS, lambda m: None, "B", True))),
           threading.Thread(target=lambda: res.__setitem__("B", clear.run_car("B", B, team, ARGS, lambda m: None, "A", False)))]
    [t.start() for t in ths]
    [t.join(20) for t in ths]
    return w, res


bad = 0


def check(name, cond, info=""):
    global bad
    bad += not cond
    print(f"{'ok  ' if cond else 'FAIL'} {name} {info}")


w, r = play(["red", "blue", "green"])
check("three objects: all in the tray, both cars worked", sorted(w.tray) == ["blue", "green", "red"] and not w.objects and r["A"] + r["B"] == 3, r)
check("  never both cars at the rim, never the same object at the same time", not w.both_at_tray and not w.same_time)
w, r = play(["red", "red", "blue", "blue", "green"])
check("five objects, colours repeated: floor cleared", not w.objects and len(w.tray) == 5 and not w.same_time, r)
w, r = play(["red", "blue"], da={"blue": 1, "red": 5}, db={"blue": 1, "red": 5})
check("both nearest to blue: A takes blue first, B red", not w.objects and w.tray and not w.same_time, w.tray)
w, r = play(["red", "blue"], a_kw={"cannot_grab": ["red", "blue"]})
check("A cannot grab anything: B clears the floor alone", not w.objects and r == {"A": 0, "B": 2}, r)
w, r = play(["red", "blue", "green"], a_kw={"cannot_grab": ["green"]}, b_kw={"cannot_grab": ["green"]})
check("nobody can grab the green one: the rest is cleared, both cars stop", w.objects == ["green"] and len(w.tray) == 2, (w.objects, r))
w, r = play(["red", "blue", "green"], a_kw={"leave_after": 1})
check("A leaves the arena: it stops, B clears the rest", r["A"] == 0 and "A" not in w.at_tray and len(w.tray) >= 2, (w.objects, r))
w, r = play([])
check("empty floor: both cars done at once", r == {"A": 0, "B": 0})

sq = [(0, 0), (1.5, 0), (1.3, 2.0), (0, 1.8)]
check("fence geometry", clear.outside_m((0.7, 1.0), sq) == 0 and abs(clear.outside_m((0.7, -0.2), sq) - 0.2) < 1e-6
      and abs(clear.outside_m((-0.3, 0.5), sq) - 0.3) < 1e-6 and clear.outside_m((2.0, 3.0), sq) > 0.8)
links = {"A": types.SimpleNamespace(out=threading.Event(), halt=lambda: None), "B": types.SimpleNamespace(out=threading.Event(), halt=lambda: None)}
f = clear.Fence.__new__(clear.Fence)
f.links, f.tags, f.margin, f.seen, f.where, f.poly = links, {"A": 33, "B": 36}, 0.3, set(), {}, None
snap = {"corners": [29, 32, 31, 30], "floor": {"29": [0, 0], "32": [1.5, 0], "31": [1.3, 2.0], "30": [0, 1.8]}}
check("fence: start 20 cm outside is tolerated", f.check(dict(snap, tags={"33": [0.5, -0.2, 0], "36": [0.8, 0.5, 0]})) == [])
check("fence: 45 cm outside stops that car only", f.check(dict(snap, tags={"33": [0.5, -0.45, 0], "36": [0.8, 0.5, 0]})) == ["A"]
      and links["A"].out.is_set() and not links["B"].out.is_set())
check("fence: tag not seen = no action", f.check(dict(snap, tags={})) == [])
print("all ok" if not bad else f"{bad} failed")
sys.exit(1 if bad else 0)
