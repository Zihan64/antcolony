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
  tray     both cars may search for the tray and drive to it at once; only the last stretch (up to
           the rim, place, back away) is one car at a time - a car arriving meanwhile waits --queue-cm short.
A car is done when it finds nothing more to take (a whole look-around with nothing in it).

Boundary (backup): the tracker's snapshots (UDP 9100) give the arena corner tags and each car's
tag. A car whose tag is more than --fence-m outside the corner quadrilateral is stopped at once
and takes no further part (it is not driven back: the tag does not say which way is "in"
reliably enough). No tracker data, or a car's tag not seen: no fence for it - the walls and the
car's own sonar (it never drives up to something tall while looking for objects) are the first line.

Tray (shared knowledge): an AprilTag on the tray (--tray-tag, default 54 on top and 53 on the front) that only the
overhead camera reads. A car holding an object is first turned to face the tray by it (class
Guide), then finds, approaches and docks at the tray with its own camera as before. Tag not in
view: the car looks around for the tray by itself.
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
        self.seen, self.where, self.poly, self.t_last, self.snap = set(), {}, None, 0.0, None

    def newest(self, max_age=1.0):
        """The newest tracker snapshot, or None if it is older than max_age seconds."""
        return self.snap if self.snap is not None and time.monotonic() - self.t_last <= max_age else None

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
            self.t_last, self.snap = time.monotonic(), m
            for name in self.check(m):
                self.say(name, f"OUTSIDE THE ARENA by {100 * self.where[name]:.0f} cm (overhead camera) - STOPPED, out of the run")


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Guide:
    """The overhead camera as the cars' common map for ONE thing: where the tray is. The tray carries
    an AprilTag that only the overhead camera reads (user, 2026-11-14: the cars recognise the tray
    by sight). Before a car searches for the tray it is turned to FACE it, from its own tag's pose
    and the tray tag's position - then its camera has the tray in the first picture instead of
    after a look-around of a minute or two. Everything after that is the car's own eyes, as before.

    snapshot() -> newest tracker message or None. calib[name] = that car's calib/<name>.json
    (th_sign, th_off: tag yaw -> the car's heading; tag_off: where the tag sits on the car)."""

    def __init__(self, snapshot, tags, tray_tag, calib, sleep=time.sleep, tol_deg=12.0):
        self.snapshot, self.tags, self.calib, self.sleep = snapshot, tags, calib, sleep
        self.tray_tags = [str(i) for i in (tray_tag if isinstance(tray_tag, (list, tuple)) else [tray_tag])]   # first one in view counts
        self.tol = math.radians(tol_deg)

    def where(self, name):
        """-> (angle the car must turn, + = left, rad; distance to the tray, m) or None if unknown."""
        m = self.snapshot()
        tags = (m or {}).get("tags") or {}
        car = tags.get(str(self.tags[name]))
        tray = next((tags[i] for i in self.tray_tags if i in tags), None)
        if not car or not tray:
            return None
        c = self.calib.get(name) or {}
        th = wrap(c.get("th_sign", 1) * car[2] + c.get("th_off", 0.0))
        ox, oy = c.get("tag_off", (0.0, 0.0))
        x = car[0] - (math.cos(th) * ox - math.sin(th) * oy)
        y = car[1] - (math.sin(th) * ox + math.cos(th) * oy)
        return wrap(math.atan2(tray[1] - y, tray[0] - x) - th), math.hypot(tray[0] - x, tray[1] - y)

    def face(self, name, link, say, w=0.3, rate_deg_s=250.0, dead=0.12, max_pulses=10):
        """Turn the car until it faces the tray. -> True if it does, False if the camera cannot say."""
        c = self.calib.get(name) or {}
        sign, rate = float(c.get("w_sign", 1)), math.radians(rate_deg_s)
        first = self.where(name)
        if first is None:
            say("  overhead camera: tray tag or my tag not in view - searching with my own camera")
            return False
        say(f"  overhead camera: the tray is {abs(math.degrees(first[0])):.0f} deg to my "
            f"{'left' if first[0] > 0 else 'right'}, {100 * first[1]:.0f} cm away - turning to it")
        err = first[0]
        for _ in range(max_pulses):
            if abs(err) <= self.tol:
                return True
            if getattr(link, "out", None) is not None and link.out.is_set():
                return False
            secs = min(0.6, dead + abs(err) / rate)
            t_end, left = time.monotonic() + secs, secs
            link.send({"type": "vel", "vx": 0, "vy": 0, "w": math.copysign(w, err) * sign, "dur": min(0.4, secs), "guard": True})
            self.sleep(min(0.4, secs))
            if secs > 0.4:
                link.send({"type": "vel", "vx": 0, "vy": 0, "w": math.copysign(w, err) * sign, "dur": secs - 0.4, "guard": True})
                self.sleep(secs - 0.4)
            link.send({"type": "vel", "vx": 0, "vy": 0, "w": 0, "dur": 0.1})
            self.sleep(0.9)                                  # stand still, a fresh snapshot
            now = self.where(name)
            if now is None:
                say("  overhead camera lost my tag while turning - searching with my own camera")
                return False
            did = wrap(err - now[0])                         # how far it really turned towards the tray
            if abs(did) > math.radians(4):
                if did * err < 0:
                    sign = -sign                             # turned the wrong way round: this car's w is reversed
                else:
                    rate = max(math.radians(60), min(math.radians(500), 0.5 * rate + 0.5 * abs(did) / max(0.03, secs - dead)))
            err = now[0]
        return abs(err) <= 2 * self.tol


class Team:
    def __init__(self):
        self.claims, self.choosing, self.tray = {}, threading.Lock(), threading.Lock()
        self.started = threading.Event()                     # the first car has made its first choice


def run_car(name, link, team, a, say, other=None, first=True, guide=None):
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
            # Both cars may look for the tray and drive towards it at the same time (user, 2026-11-14: the
            # tray is far away, a car holding its object must not stand idle while the other one searches).
            # Only the last stretch is one at a time: a car that arrives while the other is placing stops
            # --queue-cm short of the tray (room for the other to back away), then goes on when it is free.
            sargs = dict(json.loads(a.seek), target=a.target)
            if guide is not None:
                guide.face(name, link, say)
            queued = team.tray.locked()
            say(f"to the tray - seek: '{a.target}'" + (f" (the other car is there: stopping {a.queue_cm:.0f} cm short)" if queued else "") + " ...")
            ok, detail, secs = link.skill("seek", dict(sargs, arrive_cm=a.queue_cm) if queued else sargs, 390)
            say(f"  {'ok' if ok else 'FAILED'} - {detail}  ({secs:.0f} s)")
            if not ok:
                say(f"STOPPING: tray not found - I still hold the {label}")
                return delivered
            if team.tray.locked():
                say("near the tray - waiting for the other car to finish there")
            with team.tray:
                if queued:                                   # the rest of the way, now that the tray is free
                    ok, detail, secs = link.skill("seek", sargs, 200)
                    say(f"  my turn - up to the tray: {'ok' if ok else 'FAILED'} - {detail}  ({secs:.0f} s)")
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
    ap.add_argument("--queue-cm", type=float, default=80, help="a car that arrives while the other is at the tray stops this far short of it")
    ap.add_argument("--choose-s", type=float, default=120)
    ap.add_argument("--fence-m", type=float, default=0.30, help="stop a car whose tag is this far outside the corner tags (0 = no fence)")
    ap.add_argument("--tray-tag", type=int, nargs="+", default=[54, 53],
                    help="AprilTag(s) on the tray, read by the overhead camera; the first one in view counts (54 = on top, 53 = on the front)")
    ap.add_argument("--no-guide", action="store_true", help="do not turn the cars to the tray by the overhead camera")
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
    guide = None
    if fence is not None and not a.no_guide:
        import os
        cdir, calib = os.path.join(os.path.dirname(os.path.abspath(a.config)), "calib"), {}
        for n in names:
            try:
                with open(os.path.join(cdir, n + ".json")) as f:
                    calib[n] = json.load(f)
            except (OSError, ValueError):
                print(f"  guide: no calib/{n}.json - {n} searches for the tray with its own camera only")
        tray_tag = a.tray_tag
        guide = Guide(fence.newest, {n: cfg["robots"][n]["tag"] for n in names if n in calib}, tray_tag, calib)
        seen = any(str(i) in ((fence.newest() or {}).get("tags") or {}) for i in tray_tag)
        print(f"  guide: tray tag {tray_tag} " + ("is in the overhead camera's view - cars are turned to face the tray before they search"
                                                 if seen else "NOT in the overhead camera's view right now - cars search with their own cameras"))
        for n in list(guide.tags):
            w_ = guide.where(n)
            if w_:
                print(f"  guide: from {n} the tray is {abs(math.degrees(w_[0])):.0f} deg {'left' if w_[0] > 0 else 'right'}, {100 * w_[1]:.0f} cm")
    result = {}

    def job(n, other, first):
        try:
            g = guide if guide is not None and n in guide.tags else None
            result[n] = run_car(n, links[n], team, a, lambda m: say_to(n, m), other, first, g)
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
