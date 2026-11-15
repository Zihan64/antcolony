"""Two arm cars clear the floor: every object into the tray, until none is left. The overhead
camera is only a backup that keeps the cars inside the arena.

    python -m antcolony.clear                          # cars A and B, tray = "white tray"
    python -m antcolony.clear --fence-m 0.3 --max-min 20

Each car, again and again (the steps of antcolony.duo, same car skills):
  choose   the object NEAREST to itself, found by its colour with its own camera. One car chooses
           at a time (the first car first); a colour the other car is working on is left alone,
           and so is a colour this car has given up on. Two objects of one colour are taken one
           after the other, not at the same time (the cars cannot tell them apart).
  grab     up to --tries; after a miss: back up, line up on it again by its colour.
           None held: this car leaves that colour to the other car and chooses again.
  tray     one car at a time: seek the tray, place, back away.
A car is done when it finds nothing more to take (a whole look-around with nothing in it).

Boundary (backup): the tracker's snapshots (UDP 9100) give the arena corner tags and each car's
tag. A car whose tag is more than --fence-m outside the corner quadrilateral is stopped at once
and takes no further part (it is not driven back: the tag does not say which way is "in"
reliably enough). No tracker data, or a car's tag not seen: no fence for it - the walls and the
car's own sonar (it never drives up to something tall while looking for objects) are the first line.
"""
import argparse
import json
import math
import re
import socket
import sys
import threading
import time

from .common import PORT_WORLD, load_config
from .duo import Hub, HubLink
from .solo import PORT_TELEM, show


def outside_m(p, poly):
    """How far point p lies OUTSIDE the convex polygon `poly` (m); 0 if inside."""
    n, inside, best = len(poly), True, float("inf")
    cx, cy = sum(q[0] for q in poly) / n, sum(q[1] for q in poly) / n
    for i in range(n):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
        ex, ey = x2 - x1, y2 - y1
        if ((ex * (p[1] - y1) - ey * (p[0] - x1)) > 0) != ((ex * (cy - y1) - ey * (cx - x1)) > 0):
            inside = False                                    # p and the centre on different sides of this edge
        t = max(0.0, min(1.0, ((p[0] - x1) * ex + (p[1] - y1) * ey) / max(1e-9, ex * ex + ey * ey)))
        best = min(best, math.hypot(p[0] - x1 - t * ex, p[1] - y1 - t * ey))
    return 0.0 if inside else best


class FenceLink(HubLink):
    """A HubLink whose skills end at once when the fence says this car has left the arena."""

    def __init__(self, name, addr, hub):
        super().__init__(name, addr, hub)
        self.out = threading.Event()

    def halt(self):
        self.send({"type": "skill", "id": int(time.time() * 10) % 2_000_000_000 + 3, "name": "stop", "args": {}})
        self.send({"type": "vel", "vx": 0, "vy": 0, "w": 0, "dur": 0.1})

    def skill(self, name, args, timeout):
        if self.out.is_set():
            return False, "outside the arena - stopped", 0.0
        sid = int(time.time() * 10) % 2_000_000_000
        msg, t0, last = {"type": "skill", "id": sid, "name": name, "args": args}, time.monotonic(), 0.0
        while time.monotonic() - t0 < timeout:
            if self.out.is_set():
                self.halt()
                return False, "outside the arena - stopped", time.monotonic() - t0
            if time.monotonic() - last > 0.5:
                self.send(msg)
                last = time.monotonic()
            s = (self.pump(0.1) or {}).get("skill") or {}
            if s.get("id") == sid and s.get("state") in ("ok", "fail"):
                return s["state"] == "ok", s.get("detail", ""), time.monotonic() - t0
        self.send({"type": "skill", "id": sid + 1, "name": "stop", "args": {}})
        return False, f"no answer within {timeout:.0f} s", time.monotonic() - t0

    def back_up(self, cm, cm_per_pulse=5.7, secs=0.33):
        for _ in range(max(1, round(cm / cm_per_pulse))):
            if self.out.is_set():
                return self.halt()
            self.send({"type": "vel", "vx": -0.05, "vy": 0, "w": 0, "dur": secs})
            self.pump(secs + 0.3)


class Fence(threading.Thread):
    """Watches the tracker: a car outside the corner tags' quadrilateral by more than `margin` is stopped."""

    def __init__(self, links, tags, margin, say, port=PORT_WORLD):
        super().__init__(daemon=True)
        self.links, self.tags, self.margin, self.say = links, tags, margin, say
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.bind(("0.0.0.0", port))
        self.rx.settimeout(0.5)
        self.seen, self.where, self.poly, self.t_last = set(), {}, None, 0.0

    def check(self, m):
        """One tracker snapshot. -> names of the cars that just left."""
        corners, floor = m.get("corners") or [], m.get("floor") or {}
        pts = [floor.get(str(c)) for c in corners]
        if len(pts) >= 3 and all(pts):
            self.poly = [(float(p[0]), float(p[1])) for p in pts]
        left = []
        if not self.poly:
            return left
        for name, tag in self.tags.items():
            p = (m.get("tags") or {}).get(str(tag))
            if not p:
                continue
            d = outside_m((float(p[0]), float(p[1])), self.poly)
            self.where[name] = d
            self.seen.add(name)
            if d > self.margin and not self.links[name].out.is_set():
                self.links[name].out.set()
                self.links[name].halt()
                left.append(name)
        return left

    def run(self):
        while True:
            try:
                data, _ = self.rx.recvfrom(65535)
                m = json.loads(data.decode())
            except (socket.timeout, ValueError, OSError):
                continue
            self.t_last = time.monotonic()
            for name in self.check(m):
                self.say(name, f"OUTSIDE THE ARENA by {100 * self.where[name]:.0f} cm (overhead camera) - STOPPED, out of the run")


class Team:
    def __init__(self):
        self.claims, self.choosing, self.tray = {}, threading.Lock(), threading.Lock()
        self.started = threading.Event()                     # the first car has made its first choice


def run_car(name, link, team, a, say, other=None, first=True):
    """One car until nothing is left for it. -> number of objects it put in the tray."""
    skip, delivered, misses = set(), 0, 0
    try:
        if not first:
            say("waiting for the first car to choose ...")
            team.started.wait(a.choose_s)
        while True:
            if link.out.is_set():
                return delivered
            with team.choosing:                              # one car chooses at a time
                avoid = set(skip)
                if other and team.claims.get(other):
                    avoid.add(team.claims[other])
                target = "object" + "".join(f" not {c}" for c in sorted(avoid))
                say("find: the nearest object" + (f" (not {', '.join(sorted(avoid))})" if avoid else "") + " ...")
                ok, detail, secs = link.skill("seek", {"target": target, "detector": "colour",
                                                       "bottom_stop": a.find_bottom, "stop_cm": 12}, 390)
                m = re.search(r"colour=(\w+)", detail)
                colour = m[1] if ok and m and m[1] not in ("None", "coloured") else None
                say(f"  {'ok' if ok else 'nothing'} - {detail}  ({secs:.0f} s)")
                team.claims[name] = colour if ok else None
                team.started.set()
            if link.out.is_set():
                return delivered
            if not ok:
                misses += 1
                if "not found" in detail or misses >= 2:
                    say(f"nothing more for me - done ({delivered} in the tray)")
                    return delivered
                continue
            misses = 0
            label = f"{colour} {a.noun}" if colour else "any"
            say(f"my object: the {label}")
            held = False
            for i in range(1, a.tries + 1):
                say(f"grab, try {i} of {a.tries} ...")
                gargs = {"label": label}
                if a.obj_height:
                    gargs["obj_height"] = a.obj_height
                ok, detail, secs = link.skill("grab", gargs, 300)
                say(f"  {'ok' if ok else 'missed'} - {detail}  ({secs:.0f} s)")
                if ok:
                    held = True
                    break
                if link.out.is_set():
                    return delivered
                if i < a.tries:
                    link.back_up(a.back_cm)
                    if colour:
                        ok2, d2, s2 = link.skill("seek", {"target": f"{colour} object", "detector": "colour",
                                                          "bottom_stop": a.find_bottom, "stop_cm": 12}, 120)
                        say(f"  lined up on it again: {'ok' if ok2 else 'FAILED'} - {d2}  ({s2:.0f} s)")
            if not held:
                say(f"gave up on the {label} ({a.tries} grabs) - leaving it to the other car, choosing again")
                if colour:
                    skip.add(colour)
                team.claims[name] = None
                link.back_up(a.back_cm)
                if not colour:
                    return delivered
                continue
            if team.tray.locked():
                say("holding it - waiting for my turn at the tray")
            with team.tray:
                say(f"my turn at the tray - seek: '{a.target}' ...")
                ok, detail, secs = link.skill("seek", dict(json.loads(a.seek), target=a.target), 390)
                say(f"  {'ok' if ok else 'FAILED'} - {detail}  ({secs:.0f} s)")
                if ok:
                    ok, detail, secs = link.skill("dock", {"drop": True, "z_cm": a.place_z}, 150)
                    say(f"  place: {'ok' if ok else 'FAILED'} - {detail}  ({secs:.0f} s)")
                if not ok:
                    say(f"STOPPING: could not deliver - I still hold the {label}")
                    return delivered
                delivered += 1
                say(f"in the tray: {label}  (my {delivered}.)  backing away {a.away_cm:.0f} cm")
                link.back_up(a.away_cm)
            team.claims[name] = None
    finally:
        team.claims[name] = None
        team.started.set()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--first", default="A", help="the car that chooses first")
    ap.add_argument("--second", default="B")
    ap.add_argument("--noun", default="block")
    ap.add_argument("--obj-height", type=float, default=3.0, help="cm; 0 = let the grab guess")
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--back-cm", type=float, default=18)
    ap.add_argument("--find-bottom", type=float, default=0.40)
    ap.add_argument("--target", default="white tray")
    ap.add_argument("--seek", default='{"img_tokens": 576, "view": "far"}')
    ap.add_argument("--place-z", type=float, default=10)
    ap.add_argument("--away-cm", type=float, default=35)
    ap.add_argument("--choose-s", type=float, default=120)
    ap.add_argument("--fence-m", type=float, default=0.30, help="stop a car whose tag is this far outside the corner tags (0 = no fence)")
    ap.add_argument("--max-min", type=float, default=20, help="stop everything after this many minutes")
    ap.add_argument("--min-volt", type=float, default=7.0, help="do not start below this battery voltage (0 = start anyway)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    try:
        hub = Hub()
    except OSError as e:
        sys.exit(f"cannot listen on UDP {PORT_TELEM} ({e}) - is the coordinator, check, duo or solo still running?")
    t0, team, lock = time.monotonic(), Team(), threading.Lock()

    def say_to(name, msg):
        with lock:
            print(f"{time.monotonic() - t0:5.0f}s [{name}] {msg}", flush=True)

    names = (a.first, a.second)
    links = {n: FenceLink(n, tuple(cfg["robots"][n]["addr"]), hub) for n in names}
    low = []
    for n, l in links.items():
        t = l.status() or l.status(3.0) or l.status(3.0)
        say_to(n, show(t))
        if t is None:
            sys.exit(1)
        if a.min_volt and (t.get("batt_mv") or 0) < 1000 * a.min_volt:
            low.append(f"{n} {(t.get('batt_mv') or 0) / 1000:.2f} V")
    if low:
        sys.exit(f"battery below {a.min_volt} V: {', '.join(low)} - grabs miss at that voltage. Charge, or --min-volt 0 to start anyway.")
    fence = None
    if a.fence_m > 0:
        try:
            fence = Fence(links, {n: cfg["robots"][n]["tag"] for n in names}, a.fence_m, say_to)
            fence.start()
            time.sleep(1.5)
            if not fence.poly:
                print("  fence: no arena from the overhead camera yet - the fence starts when its data arrives")
            for n in names:
                print(f"  fence: {n} " + (f"is {100 * fence.where[n]:.0f} cm outside the corner tags (limit {100 * a.fence_m:.0f})"
                                          if n in fence.where else "- tag not seen by the overhead camera (no fence for it until it is)"))
        except OSError as e:
            print(f"  fence OFF: cannot listen on UDP {PORT_WORLD} ({e})")
    result = {}

    def job(n, other, first):
        try:
            result[n] = run_car(n, links[n], team, a, lambda m: say_to(n, m), other, first)
        except Exception as e:
            say_to(n, f"stopped: {e}")
            result.setdefault(n, 0)

    ths = [threading.Thread(target=job, args=(a.first, a.second, True), daemon=True),
           threading.Thread(target=job, args=(a.second, a.first, False), daemon=True)]
    why = "finished"
    try:
        for th in ths:
            th.start()
        while any(th.is_alive() for th in ths):
            time.sleep(0.2)
            if time.monotonic() - t0 > 60 * a.max_min:
                why = f"time limit ({a.max_min:.0f} min)"
                raise KeyboardInterrupt
    except KeyboardInterrupt:
        for l in links.values():
            l.halt()
        if why == "finished":
            why = "stopped by hand"
    total = sum(result.get(n, 0) for n in names)
    print(f"{time.monotonic() - t0:5.0f}s === {why}: {total} object(s) in the tray  (" +
          ", ".join(f"{n}: {result.get(n, 0)}" + (" - LEFT THE ARENA" if links[n].out.is_set() else "") for n in names) + ")")
    sys.exit(0 if why == "finished" and not any(l.out.is_set() for l in links.values()) else 2)


if __name__ == "__main__":
    main()
