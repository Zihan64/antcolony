#!/usr/bin/env python3
"""Standalone bulldozer (car C, Freenove FNK0043): finds debris with its own camera + the laptop
VLM and pushes it out of the way, avoiding tall obstacles with the ultrasonic. No overhead
tracker, no coordinator (that is agent.py's job in the team mode).

    python3 dozer_auto.py look                     # nothing moves: head sweep, photos + what the VLM sees
    python3 dozer_auto.py calib-turn               # car turns a few short pulses: measures deg/s on this floor
    python3 dozer_auto.py run [--minutes 5] [--pushes 3] [--target "small object on the floor"]
    python3 dozer_auto.py bin --target "yellow object on the floor" --bin "red box"
                                                   # push the debris INTO a box lying on the floor
    python3 dozer_auto.py sonar-check              # nothing moves: does the blade block the sonar?
    python3 dozer_auto.py wander [--minutes 2]     # drive around, sonar obstacle avoidance (no VLM)
    python3 dozer_auto.py route "F0.5 L90 F0.3 R90 B0.2"   # metres / degrees, timed (no feedback)
    python3 dozer_auto.py line [--minutes 2]       # follow black tape with the 3 IR sensors underneath

Loop:  SEARCH (head sweep 45/90/135 deg, then turn the car ~150 deg, up to a full circle)
    -> ALIGN (turn the car until the object is straight ahead)
    -> APPROACH (short guarded hops, re-checking with the VLM, until the object is at the blade)
    -> PUSH (drive on for --push-s seconds, sonar still guarding)
    -> RETREAT (back off, turn away so the next search does not find the same thing) -> SEARCH

Sonar (rides on the pan servo, ~waist of the head): the debris is lower than the beam, so it does
not trigger it; walls, feet and furniture do. A sonar hit while approaching is checked against the
photo: if the target fills the bottom of the picture it is tall debris and is pushed anyway,
otherwise the car backs off and turns towards the side where the sonar sees more room.

The VLM: vlm_server.py on the laptop, reached through an SSH reverse tunnel started there
(  PI=szk13579@<car C> ~/vlm_robot/start_vlm.sh  ), so the URL on the car is 127.0.0.1:11436.
"what the VLM is asked for" (--target): "small object on the floor" finds the blocks reliably;
"trash ..." made it box lamps and bags on the test photos (2026-11-14).
Turn rate: calib-turn saves dozer_measured.json next to this file (photo shift between pulses).

bin: the car keeps a rough heading (sum of its own turns) and the bearings of the debris and the
box. It faces the debris; if the box is not straight behind it (more than BIN_ALIGN deg off), it
"orbits": turns 90 deg away from the box side, drives a short hop, turns back and looks again -
each hop swings the line car -> debris towards the box. Lined up, it pushes with the head
watching both (sonar guard off: the box is tall and is the goal) until the box fills the view or
the sonar reads < --bin-cm, gives one last shove, backs off and checks whether the debris is gone.
"""
import argparse
import base64
import io
import json
import math
import os
import signal
import sys
import time
import urllib.error
import urllib.request

import numpy as np

from hw_freenove import Freenove

HERE = os.path.dirname(os.path.abspath(__file__))
MEASURED = os.path.join(HERE, "dozer_measured.json")
LOG_DIR = os.path.join(HERE, "dozer_log")
W, H, HFOV = 768, 432, 102.0            # Camera Module 3 Wide, full sensor width (2304x1296 mode)
PANS = (90, 45, 135)                    # head sweep: straight, left, right
TILT_SEARCH, TILT_NEAR = 80, 68         # degrees; 90 = level, lower = looking at the floor
NEAR_V = 0.82                           # box bottom this low in the picture (head at TILT_SEARCH) = at the blade
ALIGN_DEG = 9                           # good enough to drive at it
FAST_BURST_S = 0.15                     # --fast push: drive this long between looks
SONAR_RECOVER_S = 1.3                   # sonar deaf after the motors ran (see sonar_settled)
GUARD_HOP_S = 0.5                       # guarded forward drive: hop length between sonar checks (~18 cm)
STILL_DIFF = 4.0                        # wander: mean grey change (0-255) below this after a hop = did not move
BIN_ALIGN = 10                          # box this close to the line car -> debris: push


def say(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# PIL HSV hue (0..255) per colour word: a box for "yellow ..." must really contain that colour.
# 2026-11-14: asked for "yellow object", the VLM also boxed the orange arm car A and a lamp base.
HUES = {"red": ((0, 10), (240, 255)), "orange": ((10, 25),), "yellow": ((28, 50),), "green": ((55, 115),),
        "blue": ((125, 175),), "purple": ((180, 225),), "cardboard": ((10, 30),), "brown": ((10, 30),)}
SAT_MIN = {"cardboard": 50, "brown": 50}  # cardboard is a dull colour; the rest need sat > 90
SAT_MAX = {"cardboard": 180, "brown": 180}  # ...and never vivid: an orange box has the same hue at sat ~226
# (cardboard 117-127 median on car C's photos, 2026-11-14), and YOLO took it for the bin
MIN_COLOUR = 0.12                        # share of the box's pixels that must have the colour
# 2026-11-14: asked for "cardboard box", the VLM boxed something in every photo, also with no box in
# the room; on 72 car photos the real box had 13-39 % cardboard-coloured pixels, the wrong ones 0-7 %.


def colour_share(img, box, hues, sat_min=90, sat_max=255):
    x1, y1, x2, y2 = (int(v * n) for v, n in zip(box, (img.width, img.height, img.width, img.height)))
    hsv = np.asarray(img.crop((x1, y1, max(x2, x1 + 1), max(y2, y1 + 1))).convert("HSV")).reshape(-1, 3)
    h, sat, val = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    ok = np.zeros(len(h), bool)
    for lo, hi in hues:
        ok |= (h >= lo) & (h <= hi)
    return float(np.mean(ok & (sat > sat_min) & (sat <= sat_max) & (val > 60))) if len(h) else 0.0


class VLM:
    def __init__(self, url, target, side=480, small=True):
        """small: only boxes that can be debris (not cushions, furniture); False for the bin."""
        self.url, self.target, self.side, self.ms, self.small = url, target, side, None, small
        words = target.lower().split()
        word = next((c for c in HUES if c in words), None)
        self.hues, self.sat_min, self.sat_max = ((HUES[word], SAT_MIN.get(word, 90), SAT_MAX.get(word, 255))
                                                 if word else (None, 90, 255))

    def find(self, img):
        """Boxes of the target in a PIL image, as fractions (x1, y1, x2, y2), biggest first."""
        small = img.copy()
        small.thumbnail((self.side, self.side))
        buf = io.BytesIO()
        small.save(buf, "JPEG", quality=85)
        body = json.dumps({"image": base64.b64encode(buf.getvalue()).decode(), "target": self.target}).encode()
        t0 = time.time()
        req = urllib.request.Request(self.url, body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as r:
            reply = json.loads(r.read())
        self.ms = round((time.time() - t0) * 1000)
        out = []
        for o in reply.get("objects", []):
            x1, y1, x2, y2 = (v / 1000.0 for v in o["bbox_2d"])
            x1, x2, y1, y2 = sorted((x1, x2)) + sorted((y1, y2))
            area = (x2 - x1) * (y2 - y1)
            if area <= 0 or (self.small and not (area < 0.15 and x2 - x1 < 0.4)):
                continue                             # bigger = cushions, furniture: not something to push
            if self.hues:
                share = colour_share(img, (x1, y1, x2, y2), self.hues, self.sat_min, self.sat_max)
                if share < MIN_COLOUR:
                    continue                         # wrong colour: another robot, a lamp, ...
                if self.sat_max < 255 and colour_share(img, (x1, y1, x2, y2), self.hues, self.sat_max) > share:
                    continue                         # mostly the vivid version of the hue: an orange box
            out.append((x1, y1, x2, y2))
        return sorted(out, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))


class Eye:
    """Camera on the pan/tilt head."""

    def __init__(self):
        from picamera2 import Picamera2
        self.cam = Picamera2()
        cfg = self.cam.create_video_configuration(main={"size": (W, H), "format": "RGB888"},
                                                  sensor={"output_size": (2304, 1296)})
        self.cam.configure(cfg)
        self.cam.start()
        time.sleep(1.0)                              # auto exposure

    def photo(self):
        return self.cam.capture_image("main").convert("RGB")

    def gray(self):
        return np.asarray(self.photo().convert("L").resize((256, 144)), dtype=np.float32)

    def close(self):
        self.cam.stop()
        self.cam.close()


def shift_deg(a, b):
    """Horizontal image shift between two grey photos, in degrees of view (phase correlation)."""
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    fa, fb = np.fft.fft2((a - a.mean()) * win), np.fft.fft2((b - b.mean()) * win)
    r = fa * np.conj(fb)
    r = np.fft.ifft2(r / (np.abs(r) + 1e-9)).real
    _, dx = np.unravel_index(np.argmax(r), r.shape)
    if dx > a.shape[1] // 2:
        dx -= a.shape[1]
    return dx / a.shape[1] * HFOV


def wrap(deg):
    return (deg + 180.0) % 360.0 - 180.0


def load_measured():
    try:
        with open(MEASURED) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


class Dozer:
    def __init__(self, a):
        self.a = a
        self.car = Freenove(a.min_duty, a.max_duty, a.rot_min, a.rot_max, sonar=True)
        self.eye = Eye()
        self.vlm = VLM(a.url, a.target)
        m = load_measured()
        self.turn_dps, self.turn_dead = m.get("turn_dps", 120.0), m.get("turn_dead", 0.05)
        self.pan, self.tilt = 90, TILT_SEARCH
        self.shots, self.pushes, self.side, self.tall = 0, 0, 1, False
        self.heading = 0.0                           # deg, + = left; sum of commanded turns (drifts)
        self.bin_vlm = VLM(a.url, a.bin, small=False)
        os.makedirs(LOG_DIR, exist_ok=True)
        self.head(90, TILT_SEARCH)

    # ---------------------------------------------------------------- senses
    def head(self, pan, tilt, settle=0.5):
        if (pan, tilt) != (self.pan, self.tilt):
            self.car.look(pan, tilt)
            self.pan, self.tilt = pan, tilt
            time.sleep(settle)

    def see(self, tag="", vlm=None):
        """[(bearing_deg (+ = left of the car's nose), bottom_v 0..1, box)] for the target, biggest first."""
        vlm = vlm or self.vlm
        img = self.eye.photo()
        try:
            boxes = vlm.find(img)
        except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
            raise SystemExit(f"VLM not reachable at {self.a.url} ({e}). Start the tunnel on the laptop.")
        self.shots += 1
        img.save(os.path.join(LOG_DIR, f"{self.shots % 50:02d}_{tag}_p{self.pan}_t{self.tilt}_{len(boxes)}.jpg"))
        out = []
        for b in boxes:
            u = (b[0] + b[2]) / 2
            out.append(((90 - self.pan) + (0.5 - u) * HFOV, b[3], b))
        say(f"  see[{tag}] pan {self.pan} tilt {self.tilt}: "
            + (", ".join(f"{d[0]:+.0f}deg v={d[1]:.2f}" for d in out) or "nothing") + f"  ({vlm.ms} ms)")
        return out

    def sonar(self):
        d = self.car.sonar_cm()
        return 999.0 if d is None else d

    def battery_ok(self):
        mv = self.car.battery_mv()
        if mv is not None and mv < self.a.min_mv:
            time.sleep(1.0)
            mv = self.car.battery_mv()
            if mv is not None and mv < self.a.min_mv:
                say(f"battery {mv} mV < {self.a.min_mv}: stopping")
                return False
        return True

    # ---------------------------------------------------------------- motion
    def turn(self, deg):
        """Turn on the spot by about `deg` (+ = left), timed from the measured rate."""
        if abs(deg) < 2:
            return
        self.heading += deg
        n = math.ceil(abs(deg) / 20.0)               # pulses of <= 20 deg: those were measured accurate
        for _ in range(n):
            self.car.set_velocity(0, 0, math.copysign(self.a.w, deg))
            time.sleep(self.turn_dead + abs(deg) / n / self.turn_dps)
            self.car.stop()
            time.sleep(0.25)

    def sonar_settled(self):
        """Sonar reading with the motors stopped. Car C's sonar hears no echo while the motors run
        and for ~1 s after they stop (measured 2026-11-14: 200 cm for 0.9 s, then the real 37 cm),
        so wait until a reading comes back, at most SONAR_RECOVER_S."""
        self.car.stop()
        t_end = time.time() + SONAR_RECOVER_S
        while True:
            d = self.sonar()
            if d < 199 or time.time() > t_end:
                return d
            time.sleep(0.05)

    def drive(self, secs, vx=None, guard=True, w=0.0):
        """Forward (vx > 0) or back for `secs`. A guarded forward drive goes in hops of GUARD_HOP_S
        and checks the sonar while stopped (it is deaf while the motors run); it ends early if
        something is closer than --stop-cm (head straight ahead). Returns False if it stopped."""
        vx = self.a.vx if vx is None else vx
        guarded = guard and vx > 0 and self.pan == 90 and self.a.stop_cm > 0
        left, ok = secs, True
        while left > 0:
            if guarded and self.sonar_settled() < self.a.stop_cm:
                ok = False
                break
            hop = min(left, GUARD_HOP_S) if guarded else left
            t_end = time.time() + hop
            while time.time() < t_end:
                self.car.set_velocity(vx, 0, w)
                time.sleep(0.03)
            left -= hop
        self.car.stop()
        time.sleep(0.2)
        return ok

    # ---------------------------------------------------------------- behaviours
    def search(self):
        """Head sweep, then car turns, up to a full circle. Returns (bearing, v) or None."""
        turned = 0.0
        while turned < 360:
            for pan in PANS:
                self.head(pan, TILT_SEARCH)
                hits = self.see("search")
                if hits:
                    self.head(90, TILT_SEARCH)
                    return hits[0][0], hits[0][1]
            self.head(90, TILT_SEARCH)
            step = 150 * self.side
            say(f"nothing in this direction: turning {step:+.0f} deg")
            self.turn(step)
            turned += abs(step)
        return None

    def avoid(self):
        """Something tall ahead: back off, look left and right with the sonar, turn to the freer side."""
        say(f"obstacle at {self.sonar():.0f} cm: backing off")
        self.drive(0.4, vx=-self.a.vx, guard=False)
        room = {}
        for pan in (45, 135):
            self.head(pan, TILT_SEARCH, settle=0.4)
            room[pan] = np.median([self.sonar() for _ in range(5)])
        self.head(90, TILT_SEARCH)
        left = room[45] >= room[135]
        say(f"  room left {room[45]:.0f} cm, right {room[135]:.0f} cm -> turning {'left' if left else 'right'}")
        self.turn(70 if left else -70)

    def approach(self, bearing, v, near_v=NEAR_V):
        """Align and close in until the object is at the blade. True = ready to push.
        The head stays at TILT_SEARCH (the blade is out of view there); once the object drops out
        of the bottom of the picture, one look with the head down tells whether it is right ahead."""
        self.tall, lost = False, 0
        for _ in range(25):
            expect = bearing
            if abs(bearing) > ALIGN_DEG:
                step = max(-35.0, min(35.0, bearing * 0.8))
                self.turn(step)
                expect = bearing - step
            elif v >= near_v:
                return True
            else:
                hop = 0.25 if v < 0.6 else 0.12
                if not self.drive(hop, w=max(-0.4, min(0.4, bearing / 30))):
                    big = self.see("blocked")
                    if big and big[0][1] > 0.65 and abs(big[0][0]) < 20:
                        say("  sonar hit, but it is the target (tall debris): pushing it")
                        self.tall = True
                        return True
                    self.avoid()
                    return False
            self.head(90, TILT_SEARCH)
            hits = self.see("approach")
            if not hits and v > 0.55:                # probably dropped out of the bottom of the picture
                self.head(90, TILT_NEAR)
                low = self.see("near")
                self.head(90, TILT_SEARCH)
                if low:
                    say("  right in front of the blade")
                    if abs(low[0][0]) > ALIGN_DEG:
                        self.turn(low[0][0] * 0.9)
                    return True
            if not hits:
                lost += 1
                for pan in (45, 135):
                    self.head(pan, TILT_SEARCH)
                    hits = self.see("relook")
                    if hits:
                        break
                self.head(90, TILT_SEARCH)
                if not hits:
                    if lost >= 2:
                        return False
                    continue
            best = min(hits, key=lambda h: abs(h[0] - expect))   # stay on the same object
            bearing, v = best[0], best[1]
        return False

    def push(self):
        say(f"PUSH {self.a.push_s:.1f} s" + (" (tall debris: sonar guard off)" if self.tall else ""))
        self.head(90, TILT_SEARCH, settle=0.2)
        done = self.drive(self.a.push_s, guard=not self.tall)
        if not done:
            say("  push stopped by the sonar (wall or obstacle ahead)")
        self.pushes += 1
        say(f"RETREAT, pushes so far: {self.pushes}")
        self.drive(0.5, vx=-self.a.vx, guard=False)
        self.turn(110 * self.side)                   # face away from what was just pushed
        self.side = -self.side

    def run(self):
        t_end = time.time() + self.a.minutes * 60
        while time.time() < t_end and self.pushes < self.a.pushes:
            if not self.battery_ok():
                break
            say("SEARCH")
            found = self.search()
            if found is None:
                say("nothing found in a full circle: moving to a new spot")
                if not self.drive(0.6):
                    self.avoid()
                continue
            say(f"FOUND at {found[0]:+.0f} deg (v={found[1]:.2f}): APPROACH")
            if self.approach(*found):
                self.push()
            else:
                say("  approach gave up")
        say(f"done: {self.pushes} pushes")

    # ---------------------------------------------------------------- debris into a box
    def locate(self, vlm, tag, turn_ok=True):
        """Absolute bearing (deg, same frame as self.heading) and bottom v of the biggest `vlm`
        target: straight ahead, then the head sweep, then (turn_ok) the car turns. None if not found."""
        for pan in PANS:
            self.head(pan, TILT_SEARCH)
            hits = self.see(tag, vlm)
            if hits:
                self.head(90, TILT_SEARCH)
                return self.heading + hits[0][0], hits[0][1]
        self.head(90, TILT_SEARCH)
        if not turn_ok:
            return None
        for _ in range(2):                           # 2 x 150 deg + the +-96 deg the sweep sees = all round
            self.turn(150)
            for pan in PANS:
                self.head(pan, TILT_SEARCH)
                hits = self.see(tag, vlm)
                if hits:
                    self.head(90, TILT_SEARCH)
                    return self.heading + hits[0][0], hits[0][1]
        self.head(90, TILT_SEARCH)
        return None

    @staticmethod
    def where(t, b):
        """Debris box t vs bin box b (fractions) seen from the front: 'inside' (its bottom edge is
        above the box's front edge), 'mouth' (level with it) or 'outside'. Car C, 2026-11-14: the
        debris stuck half in the mouth looked lined up but closer than the box, and was left there."""
        if not t or not b:
            return "outside"
        tb, bb = t[0][2], b[0][2]
        cx = (tb[0] + tb[2]) / 2
        if not (bb[0] - 0.02 < cx < bb[2] + 0.02):
            return "outside"
        if bb[3] > 0.97:                             # box's front edge below the picture (car right at it):
            between = tb[0] >= bb[0] - 0.01 and tb[2] <= bb[2] + 0.01 and tb[2] - tb[0] < 0.8 * (bb[2] - bb[0])
            return "inside" if between and tb[1] > bb[1] + 0.1 * (bb[3] - bb[1]) else "mouth"   # walls decide
        depth = (bb[3] - tb[3]) / max(bb[3] - bb[1], 0.05)   # in box heights; > 0: debris further away
        return "inside" if depth > 0.12 else "mouth" if depth > -0.12 else "outside"

    def face(self, absolute):
        self.turn(wrap(absolute - self.heading))

    def into_bin_fast(self):
        """--fast (YOLO server, ~0.1 s per look): drive continuously, steering on every look, and
        go straight into one fast shove the moment the debris reaches the box's mouth."""
        self.head(90, TILT_SEARCH, settle=0.3)
        t_end, aim, lost = time.time() + 15, 0.0, 0
        while time.time() < t_end:
            t, b = self.see("fast", self.vlm), self.see("fast", self.bin_vlm)
            state = self.where(t, b)
            near_box = b and (b[0][2][2] - b[0][2][0] > 0.55 or b[0][1] > 0.97)
            if t and b and abs(b[0][0] - t[0][0]) < 25:
                aim = max(t[0][0] - 10, min(t[0][0] + 10, b[0][0]))
            elif t or b:
                aim = (t or b)[0][0]
            if state != "outside" or near_box:
                say(f"  at the box (debris {state}): one shove {self.a.shove_s:.1f} s at vx {self.a.shove_vx}")
                self.drive(self.a.shove_s, vx=self.a.shove_vx, guard=False, w=max(-0.4, min(0.4, aim / 30)))
                return True
            lost = 0 if (t or b) else lost + 1
            if lost > 8:
                self.car.stop()
                say("  lost both the debris and the box")
                return False
            # a short burst, then the next look with the car standing: photos taken while rolling were
            # too blurred for YOLO (car C, 2026-11-14: most frames "nothing"); still ~3-4 looks a second
            t_burst = time.time() + FAST_BURST_S
            while time.time() < t_burst:
                self.car.set_velocity(self.a.vx, 0, max(-0.5, min(0.5, aim / 25)))
                time.sleep(0.03)
            self.car.stop()
            time.sleep(0.08)
        self.car.stop()
        return False

    def into_bin(self):
        """Push straight at the box. True once the box is reached and the debris shoved in."""
        if self.a.fast:
            return self.into_bin_fast()
        trash_lost = 0
        for _ in range(20):
            self.head(90, TILT_SEARCH, settle=0.3)
            t = self.see("bin-push", self.vlm)
            b = self.see("bin-push", self.bin_vlm)
            near_box = b and (b[0][2][2] - b[0][2][0] > 0.55 or b[0][1] > 0.97)
            sonar = self.sonar() if self.a.bin_cm else 999.0
            state = self.where(t, b)
            if state != "outside" or near_box or sonar < self.a.bin_cm:
                say(f"  at the box (debris {state}, sonar {sonar:.0f} cm)")
                return True
            if t:
                trash_lost = 0
                aim = t[0][0]
                if b and t[0][1] >= 0.75:            # debris on the blade: steer the pair at the box,
                    aim = max(t[0][0] - 12, min(t[0][0] + 12, b[0][0]))   # keeping the debris on the blade
                elif b and abs(b[0][0] - t[0][0]) > BIN_ALIGN * 2:
                    say(f"  box drifted {b[0][0] - t[0][0]:+.0f} deg off the debris line: realign")
                    return False
            else:
                trash_lost += 1                      # in front of the blade, below the picture
                aim = b[0][0] if b else 0.0
                if trash_lost > 6 and not b:
                    say("  lost both the debris and the box")
                    return False
            if abs(aim) > ALIGN_DEG:
                self.turn(max(-20.0, min(20.0, aim * 0.7)))
            self.drive(0.25, guard=False, w=max(-0.4, min(0.4, aim / 30)))
        return False

    def both(self, tag):
        """One photo pose (head straight), both targets: (debris hits, box hits)."""
        self.head(90, TILT_SEARCH)
        return self.see(tag, self.vlm), self.see(tag, self.bin_vlm)

    def bin_task(self):
        """Debris -> box. Bearings come from the head wherever possible; the car body turns only to
        face the debris, and to orbit when the box is not behind it."""
        for attempt in range(1, self.a.tries + 1):
            if not self.battery_ok():
                return False
            say(f"BIN attempt {attempt}")
            t, b = self.both("both")
            if t and b:                              # one photo: relative bearing without any car turn
                tr, bx = (self.heading + t[0][0], t[0][1]), (self.heading + b[0][0], b[0][1])
            else:
                tr = (self.heading + t[0][0], t[0][1]) if t else self.locate(self.vlm, "debris")
                if tr is None:
                    # the debris is usually near the box, too small to see from afar: drive towards the box
                    bx = (self.heading + b[0][0], b[0][1]) if b else self.locate(self.bin_vlm, "box", turn_ok=False)
                    if bx is None:
                        say("  no debris anywhere")
                        return False
                    say("  no debris in sight, but the box is: driving closer to it to look again")
                    self.face(bx[0])
                    if not self.drive(1.2):
                        self.avoid()
                    continue
                bx = (self.heading + b[0][0], b[0][1]) if b else self.locate(self.bin_vlm, "box", turn_ok=False)
                if bx is None:                       # the head cannot see it: the body has to look round
                    bx = self.locate(self.bin_vlm, "box")
                if bx is None:
                    say("  no box anywhere")
                    return False
            alpha = wrap(bx[0] - tr[0])
            say(f"  debris {wrap(tr[0] - self.heading):+.0f} deg, box {wrap(bx[0] - self.heading):+.0f} deg: "
                f"box is {alpha:+.0f} deg off the line")
            if abs(wrap(tr[0] - self.heading)) > ALIGN_DEG:
                self.face(tr[0])
                t, b = self.both("faced")
            state = self.where(t, b)
            if state == "inside":
                say("DONE: the debris is in the box")
                return True
            if state == "mouth" and abs(alpha) < BIN_ALIGN * 1.5:
                say("debris is in the mouth of the box: shoving it in")
                if self.shove_until_in():
                    return True
                continue
            if abs(alpha) <= BIN_ALIGN:
                say("LINED UP: pushing into the box")
                if self.into_bin() and (self.a.fast and self.back_and_check() or self.shove_until_in(check_first=True)):
                    return True
                continue
            # orbit: move to the side away from the box so the line car -> debris swings onto it
            side = -1 if alpha > 0 else 1            # box left of the debris -> step to the right
            if abs(alpha) > 150:                     # box (nearly) behind us: go round the debris
                side = 1 if self.side > 0 else -1
            hop = max(0.3, min(0.7, 0.25 + abs(alpha) / 60 * 0.4))   # 0.21 s hops barely moved on carpet
            say(f"ORBIT: step {'right' if side < 0 else 'left'} {hop:.2f} s")
            if tr[1] < 0.6:                          # far away: close in first, the orbit is shorter then
                self.approach(0.0, tr[1], near_v=0.65)
            elif tr[1] > 0.85:                       # debris on the blade: it would travel along with
                say("  debris is on the blade: backing off before the side step")   # the car (car C, 2026-11-14)
                self.drive(0.6, vx=-self.a.vx, guard=False)
            self.turn(90 * side)
            if not self.drive(hop):
                self.avoid()
            self.turn(-90 * side)
        say("gave up")
        return False

    def back_and_check(self):
        self.drive(0.8, vx=-self.a.vx, guard=False)
        return self.check_in()

    def shove_until_in(self, check_first=False, shoves=3):
        """Debris at the mouth: short fast shoves (momentum gets it over the box's floor lip),
        backing off after each to look. True once it is inside."""
        if check_first and self.check_in():
            return True
        for k in range(shoves):
            t, b = self.both("shove")
            if t and b and abs(b[0][0] - t[0][0]) < 25:
                aim = max(t[0][0] - 10, min(t[0][0] + 10, b[0][0]))
            else:
                aim = t[0][0] if t else (b[0][0] if b else 0.0)
            if abs(aim) > ALIGN_DEG:
                self.turn(max(-20.0, min(20.0, aim * 0.7)))
            say(f"  shove {k + 1}/{shoves}: {self.a.shove_s:.1f} s at vx {self.a.shove_vx}")
            self.drive(self.a.shove_s, vx=self.a.shove_vx, guard=False)
            self.drive(0.8, vx=-self.a.vx, guard=False)     # far enough back to see the box's front edge
            if self.check_in():
                return True
        return False

    def check_in(self):
        self.head(90, TILT_SEARCH)
        t, b = self.see("check", self.vlm), self.see("check", self.bin_vlm)
        state = self.where(t, b)
        if state == "inside" or (not t and b):
            say("DONE: the debris is in the box" + (" (seen inside it)" if t else " (no longer visible)"))
            return True
        say(f"  debris {state}: again")
        return False

    # ---------------------------------------------------------------- other features
    def sonar_check(self):
        """Is the blade in the sonar beam? Readings at a few head positions (nothing moves but the head)."""
        meds = []
        for pan, tilt in ((90, 90), (90, 80), (90, 70), (60, 90), (120, 90)):
            self.head(pan, tilt, settle=0.6)
            r = [self.sonar() for _ in range(9)]
            meds.append(float(np.median(r)))
            say(f"  pan {pan} tilt {tilt}: median {meds[-1]:.1f} cm  (min {min(r):.1f}, max {max(r):.1f})")
        self.head(90, TILT_SEARCH)
        if max(meds) < 20:
            say("BLOCKED: the sonar reads a few cm everywhere - it sees the blade. Raise the sensor or use --stop-cm 0")
        else:
            say("OK: the sonar sees the room, not the blade")

    def wander(self):
        """Drive around; anything tall closer than --stop-cm -> back off and turn to the freer side.
        Stuck check by the camera, not the sonar: something below the beam (a sofa base, a box) stops
        the car without the sonar noticing, and the sonar reads 200 cm for any open space - so a
        hop after which the picture has hardly changed counts as blocked."""
        t_end = time.time() + self.a.minutes * 60
        self.head(90, 90)
        still, next_say = 0, 0.0
        while time.time() < t_end and self.battery_ok():
            before = self.eye.gray()
            ok = self.drive(GUARD_HOP_S)
            change = float(np.mean(np.abs(self.eye.gray() - before)))
            if time.time() >= next_say:
                say(f"  driving, sonar {self.sonar():.0f} cm, picture change {change:.1f}, battery {self.car.battery_mv()} mV")
                next_say = time.time() + 2
            still = still + 1 if ok and change < STILL_DIFF else 0
            if not ok:
                self.avoid()
            elif still >= 2:
                say(f"  STUCK (picture unchanged over {still} hops): something low in front")
                self.drive(0.6, vx=-self.a.vx, guard=False)
                self.turn(120 * self.side)
                self.side, still = -self.side, 0
            else:
                continue
            self.head(90, 90)

    def route(self, spec):
        """'F0.5 L90 F0.3 R90 B0.2': metres forward/back, degrees left/right; timed (no feedback)."""
        self.head(90, 90)
        for tok in spec.split():
            k, val = tok[0].upper(), float(tok[1:])
            say(f"  {tok}")
            if k == "F":
                if not self.drive(val / self.a.speed):
                    say(f"  obstacle at {self.sonar():.0f} cm: route stopped")
                    return
            elif k == "B":
                self.drive(val / self.a.speed, vx=-self.a.vx, guard=False)
            elif k in "LR":
                self.turn(val if k == "L" else -val)
            else:
                raise SystemExit(f"bad route step {tok!r} (use F/B metres, L/R degrees)")

    def line(self):
        """Follow a dark line with the three IR sensors under the front (left, middle, right)."""
        t_end, last_w, lost_t = time.time() + self.a.minutes * 60, 0.0, None
        self.head(90, 90)
        while time.time() < t_end:
            if self.sonar() < self.a.stop_cm:
                self.car.stop()
                time.sleep(0.1)
                continue
            l, m, r = self.car.line()
            w = {(0, 1, 0): 0.0, (1, 1, 0): 0.5, (1, 0, 0): 1.0, (0, 1, 1): -0.5, (0, 0, 1): -1.0}.get((l, m, r))
            if (l, m, r) == (1, 1, 1):
                say("all three sensors on black: end of the line / crossing - stop")
                break
            if w is None:                            # off the line: keep the last correction a moment
                lost_t = lost_t or time.time()
                if time.time() - lost_t > 0.6:
                    say("line lost")
                    break
                w = last_w
            else:
                lost_t, last_w = None, w
            self.car.set_velocity(self.a.vx, 0, w)
            time.sleep(0.02)
        self.car.stop()

    def close(self):
        try:
            self.car.stop()
            self.car.look(90, 90)
        finally:
            self.eye.close()
            self.car.close()


def calib_turn(dz):
    """Short turn pulses, measured by the photo shift; saves turn_dps / turn_dead."""
    dz.head(90, 90)
    pts = []
    for t in (0.08, 0.16, 0.08, 0.16):
        for sgn in (1, -1):
            a = dz.eye.gray()
            dz.car.set_velocity(0, 0, sgn * dz.a.w)
            time.sleep(t)
            dz.car.stop()
            time.sleep(0.6)
            d = abs(shift_deg(a, dz.eye.gray()))
            say(f"  pulse {t:.2f} s {'left ' if sgn > 0 else 'right'}: {d:.1f} deg")
            if 2.0 < d < HFOV * 0.45:                # beyond that the shift measure wraps around
                pts.append((t, d))
    if len({p[0] for p in pts}) < 2:
        say("not enough clean measurements (turns too big or a blank wall in view); nothing saved")
        return
    ts, ds = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    rate, off = np.polyfit(ts, ds, 1)
    if rate <= 0:
        say(f"measurements do not grow with the pulse length ({pts}); nothing saved")
        return
    dead = max(0.0, -off / rate) if rate > 0 else 0.05
    m = load_measured()
    m.update(turn_dps=round(float(rate), 1), turn_dead=round(float(dead), 3), w=dz.a.w)
    with open(MEASURED, "w") as f:
        json.dump(m, f)
    say(f"saved {MEASURED}: {rate:.0f} deg/s, dead time {dead:.2f} s (at w={dz.a.w})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["look", "calib-turn", "run", "bin", "sonar-check", "wander", "route", "line"])
    ap.add_argument("spec", nargs="?", default="", help="route: e.g. \"F0.5 L90 F0.3\"")
    ap.add_argument("--bin", default="red box", help="bin: what the VLM is asked for to find the box")
    ap.add_argument("--bin-cm", type=float, default=12, help="bin: sonar distance = at the box (0 = camera only)")
    ap.add_argument("--shove-s", type=float, default=0.5, help="bin: one shove into the box")
    ap.add_argument("--shove-vx", type=float, default=0.4, help="bin: shove speed 0..1 (momentum over the box lip)")
    ap.add_argument("--tries", type=int, default=6, help="bin: line-up attempts")
    ap.add_argument("--fast", action="store_true",
                    help="bin: detector is fast (yolo_server.py, ~0.1 s): push without stopping, one shove")
    ap.add_argument("--speed", type=float, default=0.9, help="route: m/s at min duty (measure it)")
    ap.add_argument("--target", default="small object on the floor", help="what the VLM is asked to find")
    ap.add_argument("--url", default=os.environ.get("DOZER_VLM_URL", "http://127.0.0.1:11436/detect"))
    ap.add_argument("--minutes", type=float, default=5)
    ap.add_argument("--pushes", type=int, default=3, help="stop after this many pushes")
    ap.add_argument("--push-s", type=float, default=0.6, help="how long one push drives on (~0.9 m/s at min duty)")
    ap.add_argument("--vx", type=float, default=0.0, help="forward command 0..1 (0 = min duty)")
    ap.add_argument("--w", type=float, default=0.05, help="turn command 0..1 (0.05 = just above --rot-min)")
    ap.add_argument("--min-duty", type=float, default=35)
    ap.add_argument("--max-duty", type=float, default=65)
    ap.add_argument("--rot-min", type=float, default=55)
    ap.add_argument("--rot-max", type=float, default=80)
    ap.add_argument("--stop-cm", type=float, default=35, help="sonar stop distance while driving forward")
    ap.add_argument("--min-mv", type=int, default=6400, help="stop below this battery voltage")
    a = ap.parse_args()
    a.vx = a.vx or 0.03                              # just above the dead band = min duty
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    dz = Dozer(a)
    try:
        if a.cmd == "look":
            for pan in PANS:
                for tilt in (TILT_SEARCH, TILT_NEAR):
                    dz.head(pan, tilt)
                    dz.see("look")
                    say(f"    sonar {dz.sonar():.0f} cm")
            say(f"photos in {LOG_DIR}")
        elif a.cmd == "calib-turn":
            calib_turn(dz)
        elif a.cmd == "bin":
            dz.bin_task()
        elif a.cmd == "sonar-check":
            dz.sonar_check()
        elif a.cmd == "wander":
            dz.wander()
        elif a.cmd == "route":
            dz.route(a.spec)
        elif a.cmd == "line":
            dz.line()
        else:
            dz.run()
    except KeyboardInterrupt:
        pass
    finally:
        dz.close()


if __name__ == "__main__":
    main()
