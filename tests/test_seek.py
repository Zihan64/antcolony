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
    def __init__(self, box, heading=0.0, turn_deg_s=140.0, turn_sign=1, sonar_sees=True, miss_every=0, cancel_at=None):
        self.box, self.th = box, math.radians(heading)            # heading 0 = facing +y
        self.x = self.y = 0.0
        self.turn_rate, self.turn_sign, self.sonar_sees, self.miss_every = math.radians(turn_deg_s), turn_sign, sonar_sees, miss_every
        self.v, self.w, self.hit, self.looks, self.posed = 0.0, 0.0, False, 0, 0
        self.cancel, self.cancel_at = threading.Event(), cancel_at

    # geometry: bearing of the box from the car's nose line, + = left; distance to its face
    def rel(self):
        if self.box is None:
            return None, None
        dx, dy = self.box[0] - self.x, self.box[1] - self.y
        ang = math.atan2(-dx, dy) - self.th                       # + = left of straight ahead
        return math.degrees(math.atan2(math.sin(ang), math.cos(ang))), math.hypot(dx, dy) - BOX_W / 2

    def set_velocity(self, vx, vy, w):
        self.v = math.copysign(17.0, vx) if abs(vx) > 0.02 else 0.0
        self.w = math.copysign(self.turn_rate, w) * self.turn_sign if abs(w) > 0.02 else 0.0

    def stop(self):
        self.v = self.w = 0.0

    def move(self, h):
        self.th += self.w * h
        self.x += -math.sin(self.th) * self.v * h
        self.y += math.cos(self.th) * self.v * h
        b, d = self.rel()
        if d is not None and d < 6:                               # the gripper sticks out ~9 cm: this is a bump
            self.hit = True
        if self.cancel_at is not None and math.hypot(self.x, self.y) >= self.cancel_at:
            self.cancel.set()

    def sonar_cm(self):
        b, d = self.rel()
        if d is None or not self.sonar_sees or abs(b) > 15:
            return 500.0
        return round(d, 1)

    # the camera + VLM
    def see(self):
        self.looks += 1
        b, d = self.rel()
        if d is None or abs(b) > 33 or (self.miss_every and self.looks % self.miss_every == 0):
            return None
        wide = min(1.0, math.degrees(2 * math.atan(BOX_W / 2 / max(d, 1.0))) / 66.0)
        return {"bearing": b * 0.9 + 1.5, "wide": wide, "bottom": min(1.0, 0.45 + 18.0 / max(d, 1.0))}


def run(box, args=None, **kw):
    clock = Clock()
    car = FakeCar(box, **kw)
    clock.car = car
    hw_masterpi.time = seek.time = clock

    def posed():
        car.posed += 1
    ok, detail = seek.run(car, dict(args or {}), car.cancel, eye=car, carry_pose=posed)
    return car, ok, detail, clock.t


def near(c, lo=14, hi=30):
    b, d = c.rel()
    return lo <= d <= hi and abs(b) < 20


CASES = [
    ("box straight ahead, 70 cm", (0, 70), {}, {}, lambda c, ok: ok and near(c)),
    ("box 25 deg to the left, 90 cm", (-38, 82), {}, {}, lambda c, ok: ok and near(c)),
    ("box to the right, outside the picture", (60, 40), {}, {}, lambda c, ok: ok and near(c)),
    ("box behind the car", (10, -80), {}, {}, lambda c, ok: ok and near(c)),
    ("box already right in front (20 cm)", (0, 32), {}, {}, lambda c, ok: ok and near(c) and math.hypot(c.x, c.y) < 3),
    ("the car turns half as fast as guessed", (-60, 60), {}, {"turn_deg_s": 55}, lambda c, ok: ok and near(c)),
    ("the car turns twice as fast as guessed", (-60, 60), {}, {"turn_deg_s": 230}, lambda c, ok: ok and near(c)),
    ("the car turns the other way round", (-50, 60), {}, {"turn_sign": -1}, lambda c, ok: ok and near(c)),
    ("the sonar never sees the box: stops by the picture", (0, 80), {}, {"sonar_sees": False}, lambda c, ok: ok and near(c, 8, 40)),
    ("the VLM misses the box every third look", (-30, 90), {}, {"miss_every": 3}, lambda c, ok: ok and near(c)),
    ("no box in the room: one turn, then gives up", None, {}, {}, lambda c, ok: not ok and math.hypot(c.x, c.y) < 1),
    ("far away (1.8 m)", (40, 175), {}, {}, lambda c, ok: ok and near(c)),
    ("cancelled on the way", (0, 120), {}, {"cancel_at": 30}, lambda c, ok: not ok and math.hypot(c.x, c.y) < 45),
    ("look only: nothing drives", (-20, 80), {"look_only": True}, {}, lambda c, ok: ok and math.hypot(c.x, c.y) == 0 and c.th == 0),
]


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
        good = check(car, ok) and not car.hit and car.posed == 1
        bad += not good
        where = "" if d is None else f", box face {d:.0f} cm away, {b:+.0f} deg"
        print(f"{'ok  ' if good else 'FAIL'} {name}: {'ok' if ok else 'fail'} - {detail}  "
              f"[{car.looks} looks{where}{', BUMPED the box' if car.hit else ''}]")
    print(f"\n{len(CASES) - bad}/{len(CASES)} cases ok")
    sys.exit(1 if bad else 0)
