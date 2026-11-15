"""The arm car's `seek` skill (robot/seek.py) on a fake floor: find a box with the camera, turn to
it, drive up, stop by sonar (or by the picture when the sonar does not see it). No hardware, no
VLM, no waiting (the clock is faked).

    python tests/test_seek.py          # all cases
    python tests/test_seek.py -v       # with the car's own log lines

The fake: the car is a point with a heading; the box is a 25 cm wide disc-like target at (x, y)
cm. The camera sees it within +-33 deg (bearing with a little error, width from the distance);
the sonar sees it within +-15 deg. The car's real turn rate differs from the skill's first guess.
A case passes if the result is the expected one and the car never drove into the box.
"""
import math
import os
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "robot"))
import hw_masterpi                                   # noqa: E402
import seek                                          # noqa: E402

BOX_W = 25.0


class Clock:
    def __init__(self):
        self.t, self.car = 0.0, None

    def monotonic(self):
        return self.t

    def sleep(self, dt):
        while dt > 1e-9:
            h = min(dt, 0.01)
            self.t += h
            self.car.move(h)
            dt -= h


class FakeCar(hw_masterpi.MasterPi):
    def __init__(self, box, heading=0.0, turn_deg_s=250.0, turn_sign=1, sonar_sees=True, miss_every=0, cancel_at=None):
        self.box, self.th = box, math.radians(heading)            # heading 0 = facing +y
        self.x = self.y = 0.0
        self.turn_rate, self.turn_sign, self.sonar_sees, self.miss_every = math.radians(turn_deg_s), turn_sign, sonar_sees, miss_every
        self.v, self.w, self.vs, self.hit, self.looks, self.posed, self.spin = 0.0, 0.0, 0.0, False, 0, 0, 0.0
        self.cancel, self.cancel_at = threading.Event(), cancel_at
        self.sonar_sees_set = True
        self.gives_dist, self.dist_err, self.closest = True, 1.0, 1e9
        self.pan_deg, self.glances, self.can_follow, self.steer, self.pans = 0, 0, True, 0.0, 0

    # geometry: bearing of the box from the car's nose line, + = left; distance to its face
    def rel(self):
        if self.box is None:
            return None, None
        dx, dy = self.box[0] - self.x, self.box[1] - self.y
        ang = math.atan2(-dx, dy) - self.th                       # + = left of straight ahead
        return math.degrees(math.atan2(math.sin(ang), math.cos(ang))), math.hypot(dx, dy) - BOX_W / 2

    def set_velocity(self, vx, vy, w):
        self.v = math.copysign(17.0, vx) if abs(vx) > 0.02 else 0.0
        self.vs = math.copysign(15.0, vy) if abs(vy) > 0.02 else 0.0     # sliding left (+) / right
        self.steer = w if abs(vx) > 0.02 else 0.0
        w_new = math.copysign(self.turn_rate, w) * self.turn_sign if abs(w) > 0.02 and abs(vx) <= 0.02 else 0.0
        if w_new and not self.w:
            self.spin = 0.0                                       # a turn pulse starts: nothing moves for the first 0.12 s
        self.w = w_new

    def stop(self):
        self.v = self.w = self.vs = 0.0

    def close(self):                                              # the camera being released
        pass

    def move(self, h):
        if self.w:
            self.spin += h
        if self.v:                                                # steering while rolling: gentle, proportional
            self.th += math.radians(600.0) * self.steer * self.turn_sign * h
        else:
            self.th += self.w * h if self.spin > 0.12 else 0.0
        self.x += -math.sin(self.th) * self.v * h - math.cos(self.th) * self.vs * h
        self.y += math.cos(self.th) * self.v * h - math.sin(self.th) * self.vs * h
        b, d = self.rel()
        if d is not None:
            self.closest = min(self.closest, d)
        if d is not None and self.sonar_sees_set and d < 6:      # (dock cases creep right up to the edge on purpose)
            self.hit = True
        if self.cancel_at is not None and math.hypot(self.x, self.y) >= self.cancel_at:
            self.cancel.set()

    def sonar_cm(self):
        b, d = self.rel()
        if d is None or not self.sonar_sees or abs(b) > 15:
            return 500.0
        return round(d, 1)

    # the camera + VLM
    def pan(self, deg):                                           # the arm turns, the car does not
        if deg != self.pan_deg:
            self.pan_deg = deg
            self.pans += deg != 0
            hw_masterpi.time.sleep(0.7)

    def glance(self):                                             # one camera frame while moving: time passes
        hw_masterpi.time.sleep(0.15)
        self.glances += 1
        b, d = self.rel()
        if d is None or abs(b - self.pan_deg) > 33 or not self.can_follow:
            return None
        return self.see(count=False)

    def see(self, count=True):
        if count:
            self.looks += 1
        b, d = self.rel()
        if d is None or abs(b - self.pan_deg) > 33 or (count and self.miss_every and self.looks % self.miss_every == 0):
            return None
        wide = min(1.0, math.degrees(2 * math.atan(BOX_W / 2 / max(d, 1.0))) / 66.0)
        return {"bearing": b * 0.9 + 1.5, "wide": wide, "bottom": min(1.0, 0.05 + 30.0 / max(d, 1.0)), "colour": "red",
                "dist": (d + 6.0) * self.dist_err if self.gives_dist else None}     # arm base is 6 cm behind the sonar


def run(box, args=None, dist_err=1.0, **kw):
    clock = Clock()
    car = FakeCar(box, **kw)
    car.dist_err = dist_err
    clock.car = car
    hw_masterpi.time = seek.time = clock

    def posed():
        car.posed += 1
    # eye=None and a stand-in for seek.Eye: the same path as on the car, where seek.run makes its own camera
    # (2026-11-14: passing eye=car hid that the arm pan and the rolling were decided before the camera existed)
    real_eye, seek.Eye = seek.Eye, lambda *a, **k: car
    try:
        ok, detail = seek.run(car, dict(args or {}), car.cancel, eye=None, carry_pose=posed)
    finally:
        seek.Eye = real_eye
    return car, ok, detail, clock.t


def near(c, lo=14, hi=36):
    b, d = c.rel()
    return lo <= d <= hi and abs(b) < 20


CASES = [
    ("box straight ahead, 70 cm: one smooth roll, tracked frame by frame", (0, 70), {}, {}, lambda c, ok: ok and near(c) and c.glances > 5),
    ("box 25 deg to the left, 90 cm", (-38, 82), {}, {}, lambda c, ok: ok and near(c)),
    ("box to the right, outside the picture: found by panning the arm", (60, 40), {}, {}, lambda c, ok: ok and near(c) and c.pans >= 1),
    ("box behind the car", (10, -80), {}, {}, lambda c, ok: ok and near(c)),
    ("box already right in front (20 cm)", (0, 32), {}, {}, lambda c, ok: ok and near(c) and math.hypot(c.x, c.y) < 3),
    ("the car turns half as fast as guessed", (-60, 60), {}, {"turn_deg_s": 120}, lambda c, ok: ok and near(c)),
    ("the car turns twice as fast as guessed", (-60, 60), {}, {"turn_deg_s": 450}, lambda c, ok: ok and near(c)),
    ("the car turns the other way round", (-50, 60), {}, {"turn_sign": -1}, lambda c, ok: ok and near(c)),
    ("the sonar never sees the box: stops by the picture", (0, 80), {}, {"sonar_sees": False}, lambda c, ok: ok and near(c, 8, 40)),
    ("the VLM misses the box every third look", (-30, 90), {}, {"miss_every": 3}, lambda c, ok: ok and near(c)),
    ("no box in the room: one turn, then gives up", None, {}, {}, lambda c, ok: not ok and math.hypot(c.x, c.y) < 1),
    ("far away (1.8 m)", (40, 175), {}, {}, lambda c, ok: ok and near(c)),
    ("cancelled on the way", (0, 120), {}, {"cancel_at": 30}, lambda c, ok: not ok and math.hypot(c.x, c.y) < 45),
    ("look only: nothing drives", (-20, 80), {"look_only": True}, {}, lambda c, ok: ok and math.hypot(c.x, c.y) == 0 and c.th == 0),
    ("aim only: turns to it, does not drive", (-50, 70), {"stop": "aim"}, {}, lambda c, ok: ok and math.hypot(c.x, c.y) < 1 and abs(c.rel()[0]) < 15),
    ("small block 1.5 m away to the left: turn, drive until it is low in the picture", (-70, 135),
     {"bottom_stop": 0.4, "stop_cm": 12}, {"sonar_sees": False},
     # the rule is "lower edge at 40 % of the picture" (= ~86 cm in this fake camera); rolling stops a little
     # before it (36 %) so as not to overshoot - the grab takes over from there
     lambda c, ok: ok and 50 <= c.rel()[1] + 12.5 <= 100 and abs(c.rel()[0]) < 20),
    ("tray 1.6 m away, picture distance 20 % too long: still stops well before it", (20, 160), {}, {"sonar_sees": False},
     lambda c, ok: ok and c.closest >= 14),
    ("tray 1.5 m away, cannot be followed between looks (falls back to short steps)", (-30, 150), {"follow": False}, {"sonar_sees": False},
     lambda c, ok: ok and c.closest >= 14),
    ("box behind the car, arm cannot pan (car turns 30 deg per look)", (10, -80), {"pan": False}, {}, lambda c, ok: ok and near(c)),
    ("low tray 1.2 m away that the sonar never sees", (30, 115), {}, {"sonar_sees": False}, lambda c, ok: ok and near(c, 8, 45)),
    ("low tray behind the car, sonar blind, VLM misses every 4th look", (-20, -90), {}, {"sonar_sees": False, "miss_every": 4},
     lambda c, ok: ok and near(c, 8, 45)),
]


TRAY_D = 20.0                                         # the fake tray: BOX_W (25) wide, 20 cm deep


class FakeEdge:
    """The docking camera. Lower view (blob): the tray's sideways offset and near-edge distance, 12 % off.
    Look-down view (point): what drop_point() would say - the best spot inside the tray within the
    picture (x -7..9 cm, y 9.4..24 cm) and the arm's reach, and how far inside the rim it is."""

    def __init__(self, car, noise=0.0):
        self.car, self.noise, self.n = car, noise, 0

    def view(self):
        pass

    def look(self):
        pass

    def geom(self):
        c = self.car
        return (c.box[1] - BOX_W / 2 - c.y) + 6.0, c.x - c.box[0]     # near edge ahead of the arm base; tray centre left of the car's line

    def blob(self):
        d, lat = self.geom()
        if d > 150 or abs(lat) > 0.7 * max(d, 21) + BOX_W / 2:
            return None
        return {"off": lat * 0.88, "dist": max(21.0, d * 1.12)}

    def point(self, margin=4.0):
        self.n += 1
        d, lat = self.geom()
        d += self.noise if self.n % 2 else -self.noise
        best = None
        for xi in range(-7, 10):                          # x: cm right of the car's line
            for yi in range(10, 25):
                ins = min(yi - d, d + TRAY_D - yi, BOX_W / 2 - abs(xi + lat))      # cm inside the rim (+lat: tray centre is lat LEFT)
                if ins <= 0:
                    continue
                ok = 12.5 <= math.hypot(xi, yi) <= 21.0 and ins >= margin
                key = (ok, min(ins, 8) - 0.05 * abs(xi))
                if best is None or key > best[0]:
                    best = (key, {"ok": ok, "x": float(xi), "y": float(yi), "margin": float(ins)})
        return best[1] if best else {"ok": False}


def run_dock(start_cm, noise=0.0, off=0.0, **kw):
    clock = Clock()
    car = FakeCar((off, start_cm + BOX_W / 2), **kw)
    car.sonar_sees_set = False
    clock.car = car
    hw_masterpi.time = seek.time = clock

    def posed():
        car.posed += 1
    car.dropped = None

    def drop(a):
        car.dropped = (a["x_cm"], a["y_cm"])
        return True, "dropped"
    ok, detail = seek.dock(car, {"drop": True}, car.cancel, eye=FakeEdge(car, noise), carry_pose=posed, drop=drop)
    return car, ok, detail


DOCK = [("place from 30 cm", 30, 0.0, 0), ("place from 14 cm", 14, 0.0, 0),
        ("place: tray already within reach, straight ahead - must not move at all", 8, 0.0, 0),
        ("place: within reach but 7 cm to the right - reaches over, must not move", 8, 0.0, 7),
        ("place: within reach, 6 cm to the left - reaches over, must not move", 9, 0.0, -6),
        ("place from 4 cm (too close)", 0.5, 0.0, 0), ("place with +-1 cm measuring noise", 22, 1.0, 0),
        ("place: tray 15 cm to the right of the car's line", 12, 0.0, 15),
        ("place: tray 40 cm ahead and 12 cm to the left", 40, 0.0, -12),
        ("dock: tray 90 cm away - refuses, does not move", 90, 0.0, 0)]


if __name__ == "__main__":
    verbose = "-v" in sys.argv
    if not verbose:
        seek.print = hw_masterpi.print = lambda *a, **k: None
    bad = 0
    for name, box, args, kw, check in CASES:
        if verbose:
            print(f"--- {name}")
        car, ok, detail, secs = run(box, args, **kw)
        b, d = car.rel()
        good = check(car, ok) and not car.hit and car.posed == 1 and (box is None or car.closest >= 14)   # never within 14 cm
        bad += not good
        where = "" if d is None else f", box face {d:.0f} cm away, {b:+.0f} deg"
        print(f"{'ok  ' if good else 'FAIL'} {name}: {'ok' if ok else 'fail'} - {detail}  "
              f"[{car.looks} looks{where}{', BUMPED the box' if car.hit else ''}]")
    for name, start, noise, off in DOCK:
        if verbose:
            print(f"--- {name}")
        car, ok, detail = run_dock(start, noise, off)
        edge, lat = (car.box[1] - BOX_W / 2 - car.y) + 6.0, car.x - car.box[0]
        inside = None
        if car.dropped:                                   # where the object really lands, cm inside the rim
            dx, dy = car.dropped
            inside = min(dy - edge, edge + TRAY_D - dy, BOX_W / 2 - abs(dx + lat))
        moved = math.hypot(car.x, car.y)
        good = ((ok and inside is not None and inside >= 2.5) if start < 60 else (not ok and edge > 40 and moved < 1)) \
            and car.closest + 6.0 >= min(9.5, start + 6.0) - 0.3    # chassis ends 6 cm in front of the arm base: never nearer than 3.5 cm
        bad += not good
        print(f"{'ok  ' if good else 'FAIL'} {name}: {'ok' if ok else 'fail'} - {detail}  [lands {inside if inside is None else round(inside, 1)} cm inside the rim, "
              f"car moved {moved:.0f} cm, never nearer than {car.closest + 6.0:.1f} cm]")
    print(f"\n{len(CASES) + len(DOCK) - bad}/{len(CASES) + len(DOCK)} cases ok")
    sys.exit(1 if bad else 0)
