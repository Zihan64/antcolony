#!/usr/bin/env python3
"""Car C cleans up: every piece of TRASH on the floor goes into the open cardboard box; CARGO
(regular cubes / rectangular boxes) is never pushed. Trash is told apart by shape, not colour.

    python3 cleaner.py survey                 # nothing drives: head sweep, what it would push / leave
    python3 cleaner.py run [--minutes 10]     # clean until a full search finds no more trash

Three models on the laptop, each doing what it is good at (all reached through SSH tunnels):
  4B VLM  :11438  /ask  - lists every object in a photo (Qwen JSON boxes, ~5 s) and judges each crop:
                         "simple cube or rectangular box?" yes = cargo, no = trash (9/10 right on
                         car C's photos, 2026-11-14; the 2B model got 4/10). It also names the target
                         ("purple comb-shaped object") for tracking.
  2B VLM  :11436  /detect - follows the locked target by that name (~1 s per look).
  YOLO26n :11437  /detect - the cardboard box (trained on car C's photos, ~0.05 s).

Search, as asked: the head first (wide pan sweep, then tilted down to the carpet), then the wheels
(back off, then turns of 90 deg) with a head sweep at every stop - a full circle before giving up.
The push itself is dozer_auto's bin task: line up debris and box, push, check, back off and push
again if it is not in.
"""
import argparse
import base64
import io
import json
import math
import os
import re
import signal
import sys
import time
import urllib.request

import dozer_auto as dz
from dozer_auto import Dozer, VLM, say, wrap, TILT_SEARCH, TILT_NEAR

SWEEP = (90, 60, 30, 120, 150)          # head pans: straight, left, right - small steps (the servo is slow)
SAME_DEG = 10                           # survey: detections this close in bearing = the same object
BOX_RE = re.compile(r'"bbox_2d"\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]')
LIST_Q = ('Detect every separate object lying on the floor (not furniture, people, walls, cables or the floor '
          'itself). Output a JSON list like [{"bbox_2d": [x1, y1, x2, y2], "label": "short name"}].')
OTHER_Q = ('Is the object in the middle of the picture a robot, a machine, furniture, a person, a shoe, a bag '
           'or part of a wall? Answer yes or no.')
CARGO_Q = ('Look at the object in the middle of the picture. Is it a simple cube or rectangular box (six flat '
           'faces, straight edges, right angles)? Answer yes or no.')
NAME_Q = 'Name the object in the middle of the picture by its colour and shape in 2 to 4 words.'


def b64(img, side=None):
    if side:
        img = img.copy()
        img.thumbnail((side, side))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


class Brain:
    """The 4B VLM: list the floor objects in a photo, judge one crop."""

    def __init__(self, url):
        self.url = url

    def ask(self, img, question, max_tokens=8, side=None):
        body = json.dumps({"image": b64(img, side), "question": question, "max_tokens": max_tokens}).encode()
        req = urllib.request.Request(self.url, body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())["answer"]

    def objects(self, img):
        """Boxes (fractions) of the separate objects on the floor."""
        ans = self.ask(img, LIST_Q, max_tokens=300, side=640)
        out = []
        for m in BOX_RE.findall(ans):
            x1, y1, x2, y2 = (float(v) / 1000.0 for v in m)
            x1, x2, y1, y2 = sorted((x1, x2)) + sorted((y1, y2))
            if x2 - x1 > 0.01 and y2 - y1 > 0.01:
                out.append((x1, y1, x2, y2))
        return out

    @staticmethod
    def crop(img, box):
        x1, y1, x2, y2 = box
        mx, my = (x2 - x1) * 0.3 + 0.01, (y2 - y1) * 0.3 + 0.01
        W, H = img.size
        return img.crop((int(max(0, x1 - mx) * W), int(max(0, y1 - my) * H),
                         int(min(1, x2 + mx) * W), int(min(1, y2 + my) * H)))

    def is_other(self, img, box):
        return self.ask(self.crop(img, box), OTHER_Q).strip().lower().startswith("yes")

    def is_cargo(self, img, box):
        return self.ask(self.crop(img, box), CARGO_Q).strip().lower().startswith("yes")

    def name(self, img, box):
        return re.sub(r"[^a-z -]", "", self.ask(self.crop(img, box), NAME_Q, max_tokens=12).lower()).strip()


def yolo_all(url, img):
    """Every box of the 3-class YOLO: [(label, (x1, y1, x2, y2) fractions)]."""
    body = json.dumps({"image": b64(img, 480), "target": "all"}).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        objs = json.loads(r.read()).get("objects", [])
    return [(o.get("label", "trash"), tuple(v / 1000.0 for v in o["bbox_2d"])) for o in objs]


class Cleaner(Dozer):
    def __init__(self, a):
        super().__init__(a)
        self.brain = Brain(a.brain_url)
        self.bin_vlm = VLM(a.yolo_url, a.bin, small=False)            # YOLO: the cardboard box
        self.cargo = []                                                 # absolute bearings of cargo seen
        self.cleaned = 0
        self.track_abs = None                                           # locked trash: last absolute bearing
        if a.fast and a.verify:                                         # first 4B call takes ~10 s: do it now
            try:
                self.brain.ask(self.eye.photo(), CARGO_Q)
            except OSError as e:
                say(f"  (4B VLM not reachable: {e}) - running without the cargo check")
                a.verify = False
        if a.fast:
            self.vlm = VLM(a.yolo3_url, "trash")                        # YOLO follows the trash class

    def see(self, tag="", vlm=None):
        """Dozer.see, but for the locked trash the hit nearest its last bearing comes first, so the
        car stays on ONE piece when several are in view."""
        hits = super().see(tag, vlm)
        if (vlm is None or vlm is self.vlm) and hits and self.track_abs is not None:
            hits.sort(key=lambda h: abs(wrap(self.heading + h[0] - self.track_abs)))
            if abs(wrap(self.heading + hits[0][0] - self.track_abs)) < 35:
                self.track_abs = self.heading + hits[0][0]
        return hits

    def head(self, pan, tilt, settle=None):
        if self.a.fixed_head:                                           # servos dead: the camera stays where it is
            return
        if settle is None:                                              # big pan moves take longer
            settle = 0.4 + abs(pan - self.pan) / 50.0 + abs(tilt - self.tilt) / 80.0   # a 120 deg jump was
            # still moving after 1.4 s (car C, 2026-11-14: a red cube came out at -59 instead of +55 deg)
        super().head(pan, tilt, settle)

    # ---------------------------------------------------------------- perception
    def survey_view(self):
        """One head pose: trash and cargo in view. -> (trash, cargo, bin): trash/cargo are lists of
        (abs bearing, bottom v, box), bin is (abs bearing, v, box) or None."""
        img = self.eye.photo()
        self.shots += 1
        img.save(os.path.join(dz.LOG_DIR, f"{self.shots % 50:02d}_survey_p{self.pan}_t{self.tilt}.jpg"))
        binb = [b for b in self.bin_vlm.find(img)
                if not self.near_peer((90 - self.pan) + (0.5 - (b[0] + b[2]) / 2) * dz.HFOV)]
        if self.a.fast:                                                 # YOLO: all classes in ~0.1 s
            labelled = [(lab, b) for lab, b in yolo_all(self.a.yolo3_url, img) if lab != "bin"]
            labelled = [("cargo" if lab in ("cargo", "block") else "trash", b) for lab, b in labelled]
        else:                                                           # 4B: list, then judge each crop
            labelled = [(None, b) for b in self.brain.objects(img)]
        trash, cargo = [], []
        for lab, b in labelled:
            area = (b[2] - b[0]) * (b[3] - b[1])
            if area > 0.2 or b[2] - b[0] > 0.6:                         # walls, sofa, the whole floor
                continue
            if binb and (iou(b, binb[0]) > 0.3 or (binb[0][0] < (b[0] + b[2]) / 2 < binb[0][2]
                                                   and b[3] < binb[0][3] + 0.03 and b[1] > binb[0][1] - 0.03)):
                continue                                                # the box itself / already inside it
            rel = (90 - self.pan) + (0.5 - (b[0] + b[2]) / 2) * dz.HFOV
            if self.near_peer(rel):
                continue                                                # car A / B, or behind it
            bearing = self.heading + rel
            is_cargo = (lab == "cargo") if lab else self.brain.is_cargo(img, b)
            (cargo if is_cargo else trash).append((bearing, b[3], b, img))
        bn = None
        if binb:
            bb = binb[0]
            bn = (self.heading + (90 - self.pan) + (0.5 - (bb[0] + bb[2]) / 2) * dz.HFOV, bb[3], bb)
        say(f"  survey pan {self.pan} tilt {self.tilt}: {len(trash)} trash, {len(cargo)} cargo"
            + (", box in view" if bn else ""))
        return trash, cargo, bn

    def survey_fixed(self, turn_ok=True):
        """--fixed-head: the body turns instead of the head, 60 deg at a time (the camera sees 102)."""
        for k in range(6 if turn_ok else 1):
            if k:
                self.turn(60)
            trash, cargo, bn = self.survey_view()
            if bn:
                self.bin_seen = bn
            self.cargo += [c[0] for c in cargo]
            if trash:
                return max(trash, key=lambda t: t[1])
        return None

    def survey(self, turn_ok=True):
        """Head first (wide, then low), then the wheels: back off, then 90 deg turns with a head
        sweep at each stop. Every view votes trash / cargo per object (grouped by bearing); the majority
        decides and a tie is cargo (one stray vote made the purple comb cargo with "any vote"). Returns the nearest
        trash (abs bearing, v, box, img) or None."""
        def sweep(tilt):
            seen = []                                                   # [bearing, cargo votes, trash votes, best view]
            for pan in SWEEP:
                self.head(pan, tilt)
                trash, cargo, bn = self.survey_view()
                if bn:
                    self.bin_seen = bn
                for kind, items in (("cargo", cargo), ("trash", trash)):
                    for t in items:
                        g = next((g for g in seen if abs(wrap(g[0] - t[0])) < SAME_DEG), None)
                        if g is None:
                            g = [t[0], 0, 0, t]
                            seen.append(g)
                        g[1 if kind == "cargo" else 2] += 1
                        if kind == "trash" and (g[3] is None or t[1] > g[3][1]):
                            g[3] = t
            is_cargo = lambda g: g[1] >= g[2]                          # majority; a tie counts as cargo
            self.cargo += [g[0] for g in seen if is_cargo(g)]
            trash = [g for g in seen if not is_cargo(g)]
            for g in seen:
                say(f"    object at {wrap(g[0] - self.heading):+.0f} deg: {g[1]} cargo / {g[2]} trash votes"
                    f" -> {'CARGO, left alone' if is_cargo(g) else 'trash'}")
            if not trash:
                return None
            if self.bin_seen:                                           # nearest first, but stay near the box
                return max(trash, key=lambda g: g[3][1] - abs(wrap(g[0] - self.bin_seen[0])) / 180.0)[3]
            return max(trash, key=lambda g: g[3][1])[3]

        if self.a.fixed_head:
            return self.survey_fixed(turn_ok)
        for stage in range(6 if turn_ok else 1):
            if stage == 1:
                say("  nothing: backing off for a wider view")
                self.head(90, TILT_SEARCH)
                self.drive(0.5, vx=-self.a.vx, guard=False)
            elif stage >= 2:
                say("  nothing: turning 90 deg to look further round")
                self.head(90, TILT_SEARCH)
                self.turn(90)
            for tilt in (TILT_SEARCH, TILT_NEAR):
                best = sweep(tilt)
                if best:
                    self.head(90, TILT_SEARCH)
                    return best
        self.head(90, TILT_SEARCH)
        return None

    # ---------------------------------------------------------------- tracking a locked target
    def locate(self, vlm, tag, turn_ok=True):
        """dozer_auto calls this to find the debris / box again. Wider than Dozer.locate for the
        debris: the full head sweep, then low, then the wheels (see survey)."""
        if self.a.fixed_head:                                          # no head: turn the body round
            for k in range(6 if turn_ok else 1):
                if k:
                    self.turn(60)
                hits = self.see(tag, vlm)
                if hits:
                    return self.heading + hits[0][0], hits[0][1]
            return None
        if vlm is self.bin_vlm:
            found = super().locate(vlm, tag, turn_ok=False)              # head sweep first
            for _ in range(3 if turn_ok and self.bin_seen else 0):      # then towards where it was last seen
                if found:
                    break
                say(f"  box not in sight: heading for where it was last seen ({wrap(self.bin_seen[0] - self.heading):+.0f} deg)")
                self.face(self.bin_seen[0])
                if not self.drive(1.0):
                    self.avoid()
                found = super().locate(vlm, tag, turn_ok=False)
            if not found and turn_ok:
                found = super().locate(vlm, tag, turn_ok=True)
            if found:
                self.bin_seen = (found[0], found[1], None)
            return found
        for stage in range(4 if turn_ok else 1):
            if stage == 1:
                self.head(90, TILT_SEARCH)
                self.drive(0.5, vx=-self.a.vx, guard=False)
            elif stage >= 2:
                self.head(90, TILT_SEARCH)
                self.turn(90 if stage == 2 else 180)
            for tilt in (TILT_SEARCH, TILT_NEAR):
                for pan in SWEEP:
                    self.head(pan, tilt)
                    hits = self.see(tag, vlm)
                    if hits:
                        self.head(90, TILT_SEARCH)
                        return self.heading + hits[0][0], hits[0][1]
        self.head(90, TILT_SEARCH)
        return None

    def lock(self, target):
        """Face the chosen trash and look again close up: still trash (not cargo)? -> its name for
        the 2B tracker. False if the close look says cargo."""
        bearing, v, box, img = target
        self.face(bearing)
        self.head(90, TILT_SEARCH)
        trash, cargo, _ = self.survey_view()
        near = lambda items: [t for t in items if abs(wrap(t[0] - bearing)) < SAME_DEG * 1.5]
        if near(cargo) and not near(trash):
            say("  close look: it is CARGO after all - leaving it")
            self.cargo.append(bearing)
            return False
        if near(trash):
            bearing, v, box, img = min(near(trash), key=lambda t: abs(wrap(t[0] - bearing)))
            self.face(bearing)
        if self.a.fast:
            t0 = time.time()
            if self.a.verify and self.brain.is_cargo(img, box):         # one 4B look (~1.5 s): never push cargo
                say(f"  4B check: CARGO, not trash - leaving it ({time.time() - t0:.1f} s)")
                self.cargo.append(bearing)
                return False
            if self.a.verify and self.brain.is_other(img, box):         # ...nor a robot, furniture, a bag
                say(f"  4B check: a robot / furniture / bag, not trash - leaving it ({time.time() - t0:.1f} s)")
                self.cargo.append(bearing)
                return False
            self.track_abs = bearing
            say(f"LOCKED on trash at {wrap(bearing - self.heading):+.0f} deg (4B check {time.time() - t0:.1f} s)")
            return True
        name = self.brain.name(img, box) or "object"
        self.vlm = VLM(self.a.url, f"{name} on the floor")               # the 2B VLM follows it by name
        say(f"LOCKED on '{name}' at {wrap(bearing - self.heading):+.0f} deg")
        return True

    # ---------------------------------------------------------------- carry the trash to the box
    def burst(self, secs, w=0.0, vx=None):
        """Drive forward a moment and stop (looks happen standing: rolling photos are blurred).
        Not at all while another car is in the way."""
        if self.peer_in_the_way():
            self.give_way()
            return
        t_end = time.time() + secs
        while time.time() < t_end and not self.peer_in_the_way():
            self.car.set_velocity(self.a.vx if vx is None else vx, 0, max(-0.6, min(0.6, w)))
            time.sleep(0.03)
        self.car.stop()
        time.sleep(0.1)

    def box_bearing(self, b=None):
        """Bearing of the box (deg, + = left of the nose): this view, else a quick head look."""
        if b:
            return b[0][0]
        for pan in (55, 125):
            self.head(pan, TILT_SEARCH)
            hits = self.see("box?", self.bin_vlm)
            if hits:
                self.head(90, TILT_SEARCH)
                self.bin_seen = (self.heading + hits[0][0], hits[0][1], None)
                return hits[0][0]
        self.head(90, TILT_SEARCH)
        return wrap(self.bin_seen[0] - self.heading) if self.bin_seen else None

    def to_blade(self):
        """Close in on the locked trash until it sits at the blade. True when it is there."""
        misses = 0
        for _ in range(40):
            t = self.see("approach", self.vlm)
            if not t:
                misses += 1
                if misses == 2:                                         # just below the picture?
                    self.head(90, TILT_NEAR)
                    t = self.see("approach-low", self.vlm)
                    self.head(90, TILT_SEARCH)
                    if t:
                        return True
                if misses > 4:
                    return False
                continue
            misses = 0
            bearing, v = t[0][0], t[0][1]
            if v >= 0.88 and abs(bearing) < 12:
                return True
            if abs(bearing) > 15:
                self.turn(max(-20.0, min(20.0, bearing * 0.8)))
                continue
            self.burst(0.25 if v < 0.7 else 0.15, w=bearing / 25)
        return False

    def carry(self):
        """Trash at the blade: push it along, steering gently towards the box, until it reaches the
        box's mouth; then shove. True once the check says it is inside."""
        lost = 0
        for step in range(60):
            t, b = self.both("carry")
            state = self.where(t, b)
            if state != "outside" or (b and b[0][2][3] > 0.97 and not t):
                say(f"  at the box (debris {state}): shove")
                return self.shove_until_in(shoves=3)
            if not t:
                self.head(90, TILT_NEAR)
                t = self.see("carry-low", self.vlm)
                self.head(90, TILT_SEARCH)
                if not t:
                    lost += 1
                    if lost > 2:
                        say("  the trash slipped off the blade")
                        return False
                    continue
                t = [(t[0][0], 1.0, t[0][2])]                           # seen with the head down: at the blade
            lost = 0
            tb = t[0][0]
            if t[0][1] < 0.75:                                          # it got away from the blade: catch up
                if not self.to_blade():
                    return False
                continue
            bb = self.box_bearing(b)
            if bb is None:
                say("  no idea where the box is")
                return False
            # keep the trash on the blade first, then turn the pair towards the box
            w = tb / 20 if abs(tb) > 10 else max(-0.35, min(0.35, bb / 40))
            if step % 4 == 0:
                say(f"  carrying: trash {tb:+.0f} deg, box {bb:+.0f} deg -> steer {w:+.2f}")
            self.burst(0.3, w=w)
        return False

    def carry_task(self):
        """--fast: lock -> to the blade -> carry to the box -> shove -> check; again after a miss."""
        for attempt in range(1, self.a.tries + 1):
            if not self.battery_ok():
                return False
            say(f"CARRY attempt {attempt}")
            if attempt > 1:
                tr = self.locate(self.vlm, "trash")
                if tr is None:
                    say("  lost the trash")
                    return False
                self.face(tr[0])
            if not self.to_blade():
                say("  could not get the trash onto the blade")
                continue
            if self.carry():
                return True
            say("  not in yet: backing off to try again")
            self.drive(0.5, vx=-self.a.vx, guard=False)
        return False

    # ---------------------------------------------------------------- the job
    def run(self):
        t_end = time.time() + self.a.minutes * 60
        empty = 0
        while time.time() < t_end and self.battery_ok():
            say(f"SURVEY (cleaned so far: {self.cleaned})")
            target = self.survey()
            if target is None:
                empty += 1
                say("no trash found all round" + (": done" if empty >= 2 else ": once more"))
                if empty >= 2:
                    break
                continue
            empty = 0
            if not self.lock(target):
                continue
            if self.carry_task() if self.a.fast else self.bin_task():
                self.cleaned += 1
                say(f"*** trash #{self.cleaned} is in the box ***")
                self.drive(0.6, vx=-self.a.vx, guard=False)
                self.turn(120 * self.side)                             # look elsewhere for the next one
                self.side = -self.side
            else:
                say("  could not get it in; surveying again")
        say(f"FINISHED: {self.cleaned} pieces of trash in the box")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["survey", "run", "collect"])
    ap.add_argument("--steps", type=int, default=6, help="collect: car headings (360/steps deg apart)")
    ap.add_argument("--url", default="http://127.0.0.1:11436/detect", help="2B VLM: tracking by name")
    ap.add_argument("--brain-url", default="http://127.0.0.1:11438/ask", help="4B VLM: list + judge")
    ap.add_argument("--yolo-url", default="http://127.0.0.1:11437/detect", help="YOLO26n 2-class: the cardboard box")
    ap.add_argument("--yolo3-url", default="http://127.0.0.1:11439/detect",
                    help="YOLO26n 3-class (trash / bin / cargo, yolo/autolabel3.py): finds and tracks trash")
    ap.add_argument("--bin", default="cardboard box")
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--tries", type=int, default=10)
    ap.add_argument("--bin-cm", type=float, default=0)
    ap.add_argument("--shove-s", type=float, default=0.5)
    ap.add_argument("--shove-vx", type=float, default=0.4)
    ap.add_argument("--vx", type=float, default=0.03)
    ap.add_argument("--w", type=float, default=0.05)
    ap.add_argument("--min-duty", type=float, default=35)
    ap.add_argument("--max-duty", type=float, default=65)
    ap.add_argument("--rot-min", type=float, default=55)
    ap.add_argument("--rot-max", type=float, default=80)
    ap.add_argument("--stop-cm", type=float, default=25)
    ap.add_argument("--min-mv", type=int, default=5800)
    ap.add_argument("--fast", action="store_true", help="3-class YOLO finds/judges/tracks; push without stopping")
    ap.add_argument("--peers", default=None,
                    help="file with the other cars' positions (agent.py writes /tmp/antcolony_peers.json in the team run)")
    ap.add_argument("--fixed-head", action="store_true", help="the pan/tilt servos do not work: turn the body to look round")
    ap.add_argument("--no-verify", dest="verify", action="store_false", help="--fast: skip the 4B cargo check")
    a = ap.parse_args()
    a.target, a.push_s, a.pushes, a.speed = "object on the floor", 0.6, 99, 0.37
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    c = Cleaner(a)
    c.bin_seen = None
    try:
        if a.cmd == "collect":
            # training photos for YOLO: a head sweep (5 pans x 2 tilts) at every car heading
            out = os.path.join(dz.HERE, "dataset")
            os.makedirs(out, exist_ok=True)
            stamp = time.strftime("%H%M%S")
            for k in range(a.steps):
                for tilt in (TILT_SEARCH, TILT_NEAR):
                    for pan in (SWEEP if tilt == TILT_SEARCH else tuple(reversed(SWEEP))):
                        c.head(pan, tilt)
                        c.eye.photo().save(os.path.join(out, f"{stamp}_h{k}_p{pan}_t{tilt}.jpg"), quality=92)
                say(f"  heading {k + 1}/{a.steps}: 10 photos")
                c.head(90, TILT_SEARCH)
                c.turn(360.0 / a.steps)
            say(f"photos in {out}")
        elif a.cmd == "survey":
            best = c.survey(turn_ok=False)
            if best:
                say(f"would push first: '{c.brain.name(best[3], best[2])}' at {wrap(best[0] - c.heading):+.0f} deg")
        else:
            c.run()
    except KeyboardInterrupt:
        pass
    finally:
        c.close()


if __name__ == "__main__":
    main()
