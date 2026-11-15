"""Look around - and if need be patrol the floor - until a coloured object is close enough to grab.

    python -m antcolony.patrol --robot A --label "red block"            # find it, end up close to it
    python -m antcolony.patrol --robot A --label "red block" --grab     # ... and grab it
    (on the car itself, while another computer drives another car: --addr 127.0.0.1)

The car's agent.py must be running. Only skills the car already has are used (seek look_only,
carry, grab) plus short timed turns, so nothing on the car needs updating:

  1. look around on the spot: one photo (arm raised, the object found by its colour), turn
     about --scan-deg, photo again ... until a full circle is done
  2. seen: turn towards it, drive closer in steps (carry: sonar, slides round tall things) and
     look again after each step, until it is --close wide in the picture (or near the bottom)
  3. not seen in a full circle: drive --hop-cm to a new spot (carry, round obstacles; if that
     side is blocked, turn 90 deg and try there) and look around again, at most --spots spots

Turns are timed pulses; the turn rate is learned from how far the object moved in the picture.
Until then it is a guess (--turn-deg-s): the look-around covers a full circle if the car turns at
least 2/3 as fast as guessed (car B turns faster than the guess). New spots lie straight ahead of
where the car was put down - point it into the arena: the sonar sees tall things, not the tape.
"""
import argparse
import math
import re
import sys
import time

from .common import load_config
from .solo import Link, show

SEEN = re.compile(r"(\d+) deg (left|right), (\d+) % of the width")


class Patrol:
    def __init__(self, link, a, say):
        self.link, self.a, self.say = link, a, say
        self.rate, self.dead = a.turn_deg_s, a.turn_dead_s       # deg/s while turning, seconds lost per pulse
        self.looks, self.learned = 0, False

    # ---------------------------------------------------------------- car moves
    def look(self):
        """One photo. -> {"bearing": deg (+ = left), "wide": 0..1} or None."""
        self.looks += 1
        ok, detail, _ = self.link.skill("seek", {"target": self.a.label, "detector": "colour", "look_only": True}, 45)
        m = SEEN.search(detail) if ok else None
        if not m:
            self.say(f"  look {self.looks}: not in view")
            return None
        s = {"bearing": float(m[1]) * (1 if m[2] == "left" else -1), "wide": float(m[3]) / 100.0}
        self.say(f"  look {self.looks}: {self.a.label} {abs(s['bearing']):.0f} deg {m[2]}, {m[3]} % wide")
        return s

    def turn(self, deg):
        """Turn about `deg` (+ = left) with one timed pulse. -> seconds."""
        secs = min(1.2, self.dead + abs(deg) / self.rate)
        self.drive(0.0, math.copysign(self.a.w, deg), secs)
        return secs

    def drive(self, vx, w, secs):
        t_end = time.monotonic() + secs
        while True:
            left = t_end - time.monotonic()
            if left <= 0:
                break
            self.link.send({"type": "vel", "vx": vx, "vy": 0, "w": w, "dur": min(0.3, left), "guard": True})
            self.link.pump(min(0.15, left))
        self.link.send({"type": "vel", "vx": 0, "vy": 0, "w": 0, "dur": 0.1})
        self.link.pump(0.6)                                 # let the car (and the arm on it) settle

    def forward(self, cm):
        """Forward `cm` round anything tall (the car's carry skill). -> ok, detail."""
        ok, detail, _ = self.link.skill("carry", {"dist_cm": cm}, 30 + 2 * cm)
        self.say(f"  forward {cm:.0f} cm: {'ok' if ok else 'FAILED'} - {detail}")
        return ok, detail

    # ---------------------------------------------------------------- the search
    def scan(self):
        """Full circle on the spot. -> sighting or None. The turn rate is only a guess until the
        object has been seen once, so the photo count allows for turning a third slower than
        guessed (a faster car just looks round a bit more than once)."""
        n = math.ceil(360.0 / self.a.scan_deg / (1.0 if self.learned else 0.66))
        for _ in range(n):
            s = self.look()
            if s:
                return s
            self.turn(self.a.scan_deg)
        # Nothing seen: face the way the car faced before the circle, by the same pulses the other
        # way round (no turn rate needed), so that the patrol keeps going the way the car was put down.
        self.say("  nothing seen - turning back to the heading it started with")
        for _ in range(n):
            self.turn(-self.a.scan_deg)
        return None

    def approach(self, s):
        """Turn to it and drive closer until it is close enough. -> True when close."""
        lost = 0
        for _ in range(self.a.max_steps):
            b = s["bearing"]
            if abs(b) > self.a.aim_deg:
                secs = self.turn(b)
                s2 = self.look()
                if s2:
                    did = b - s2["bearing"]
                    if did * b > 0 and abs(did) > 3:        # turned the right way: learn the rate
                        r = abs(did) / max(0.03, secs - self.dead)
                        self.rate = max(40.0, min(600.0, r if not self.learned else 0.5 * self.rate + 0.5 * r))
                        self.learned = True
                        self.say(f"  (turn rate now {self.rate:.0f} deg/s)")
                    elif did * b < 0 and abs(did) > 3:
                        self.say("  (it moved the wrong way in the picture - check the turn direction)")
                    s = s2
                    continue
                lost += 1
                if lost >= 2:
                    return False
                s = {"bearing": b / 2, "wide": s["wide"]}     # overshot it: look back half way
                self.turn(-b / 2)
                s2 = self.look()
                if not s2:
                    return False
                s = s2
                continue
            if s["wide"] >= self.a.close:
                self.say(f"close enough: {100 * s['wide']:.0f} % wide, {abs(s['bearing']):.0f} deg off")
                return True
            step = self.a.far_step if s["wide"] < self.a.close / 2 else self.a.near_step
            ok, _ = self.forward(step)
            if not ok:
                return False
            s2 = self.look()
            if not s2:
                lost += 1
                if lost >= 2:
                    return False
                continue
            s = s2
        return False

    def run(self):
        for spot in range(1, self.a.spots + 1):
            self.say(f"spot {spot}: looking around")
            s = self.scan()
            if s and self.approach(s):
                return True
            if s:
                self.say("lost it on the way - looking around again from here")
                s = self.scan()
                if s and self.approach(s):
                    return True
            if spot == self.a.spots:
                break
            self.say(f"nothing here - moving {self.a.hop_cm:.0f} cm to a new spot")
            ok, _ = self.forward(self.a.hop_cm)
            if not ok:
                self.turn(90)
                self.forward(self.a.hop_cm)
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--robot", required=True)
    ap.add_argument("--addr", help="the car's IP, if not the one in config.json (127.0.0.1 = run on the car)")
    ap.add_argument("--label", default="red block", help="what to look for; found by its colour word")
    ap.add_argument("--grab", action="store_true", help="grab it when close (up to --tries tries)")
    ap.add_argument("--obj-height", type=float, default=3.0, help="cm, for the grab")
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--spots", type=int, default=4, help="places to look around from")
    ap.add_argument("--hop-cm", type=float, default=50, help="distance between spots")
    ap.add_argument("--scan-deg", type=float, default=40, help="turn between photos when looking around")
    ap.add_argument("--close", type=float, default=0.09, help="close enough when the object is this wide in the picture")
    ap.add_argument("--aim-deg", type=float, default=8, help="drive towards it when it is within this of straight ahead")
    ap.add_argument("--far-step", type=float, default=35, help="cm per step while it is small in the picture")
    ap.add_argument("--near-step", type=float, default=15)
    ap.add_argument("--max-steps", type=int, default=14)
    ap.add_argument("--w", type=float, default=0.3, help="turn command (normalised)")
    ap.add_argument("--turn-deg-s", type=float, default=250, help="first guess of the turn rate")
    ap.add_argument("--turn-dead-s", type=float, default=0.12)
    a = ap.parse_args()
    cfg = load_config(a.config)
    addr = (a.addr, 9200) if a.addr else tuple(cfg["robots"][a.robot]["addr"])
    link = Link(a.robot, addr)
    say = lambda msg: print(f"[{a.robot}] {msg}", flush=True)
    t = link.status()
    say(show(t))
    if t is None:
        sys.exit(1)
    t0 = time.monotonic()
    p = Patrol(link, a, say)
    if not p.run():
        say(f"'{a.label}' not found / not reached ({p.looks} looks, {time.monotonic() - t0:.0f} s)")
        sys.exit(2)
    say(f"next to '{a.label}' after {p.looks} looks, {time.monotonic() - t0:.0f} s")
    if not a.grab:
        return
    for i in range(1, a.tries + 1):
        say(f"grab, try {i} of {a.tries} ...")
        ok, detail, secs = link.skill("grab", {"label": a.label, "obj_height": a.obj_height}, 300)
        say(f"  {'ok' if ok else 'missed'} - {detail}  ({secs:.0f} s)")
        if ok:
            say(show(link.status()))
            return
        if i < a.tries:
            p.drive(-0.05, 0.0, 1.0)                        # back off so the camera sees it again
    sys.exit(3)


if __name__ == "__main__":
    main()
