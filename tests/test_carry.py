"""The arm car's `carry` skill (robot/hw_masterpi.py) on a fake floor: forward in pulses, front
sonar read at rest, slide sideways round what is in the way. No hardware, no waiting (the clock
is faked).

    python tests/test_carry.py          # all cases
    python tests/test_carry.py -v       # with the car's own log lines

The fake: the car is a 20 x 22 cm box with the sonar at the middle of its front edge and the
gripper + object sticking out 9 cm further (8 cm wide). The sonar has a +-15 deg beam and
reports the nearest obstacle face in it. Obstacles are boxes (x0, x1, y0, y1) in cm, x = left,
y = forward, the sonar starts at (0, 0). A case passes if the result is the expected one and
the car never touched anything.
"""
import math
import os
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "robot"))
import hw_masterpi                                   # noqa: E402

HALF_W, LENGTH, NOSE, NOSE_HALF_W, BEAM = 10.0, 22.0, 9.0, 4.0, 15


class Clock:
    """Stands in for the `time` module inside hw_masterpi: sleeping moves the fake car."""

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
    def __init__(self, boxes, fwd_cm_s=17.0, side_cm_s=15.0, sonar=True, creep=0.0, cancel_at=None, beam=BEAM):
        self.min_duty, self.max_duty, self.holding = 30, 45, True
        self.boxes, self.fwd, self.sideways, self.has_sonar, self.creep = boxes, fwd_cm_s, side_cm_s, sonar, creep
        self.beam = beam
        self.x = self.y = 0.0                        # the sonar: cm left of / ahead of where it started
        self.v, self.hit, self.cancel_at, self.cancel = (0.0, 0.0), False, cancel_at, threading.Event()

    def set_velocity(self, vx, vy, w):
        self.v = (math.copysign(self.fwd, vx) if abs(vx) > 0.02 else 0.0,
                  math.copysign(self.sideways, vy) if abs(vy) > 0.02 else 0.0)

    def stop(self):
        self.v = (0.0, 0.0)

    def move(self, h):
        self.y += self.v[0] * h + abs(self.v[1]) * self.creep * h      # creep: sliding also drifts forward
        self.x += self.v[1] * h
        for x0, x1, y0, y1 in self.boxes:
            body = x0 < self.x + HALF_W and x1 > self.x - HALF_W and y0 < self.y and y1 > self.y - LENGTH
            nose = x0 < self.x + NOSE_HALF_W and x1 > self.x - NOSE_HALF_W and y0 < self.y + NOSE and y1 > self.y
            self.hit |= body or nose
        if self.cancel_at is not None and self.y >= self.cancel_at:
            self.cancel.set()

    def sonar_cm(self):
        if not self.has_sonar:
            return None
        best = 500.0
        for deg in range(-self.beam, self.beam + 1):
            dx, dy = math.sin(math.radians(deg)), math.cos(math.radians(deg))
            for x0, x1, y0, y1 in self.boxes:
                for t in ([(y0 - self.y) / dy] + ([(x0 - self.x) / dx, (x1 - self.x) / dx] if abs(dx) > 1e-9 else [])):
                    px, py = self.x + t * dx, self.y + t * dy
                    if t > 0 and x0 - 1e-6 <= px <= x1 + 1e-6 and y0 - 1e-6 <= py <= y1 + 1e-6:
                        best = min(best, t)
        return round(best, 1)


def run(boxes, args=None, **kw):
    clock = Clock()
    car = FakeCar(boxes, **kw)
    clock.car, hw_masterpi.time = car, clock
    ok, detail = car.carry(dict(args or {}), car.cancel)
    return car, ok, detail, clock.t


CASES = [
    # name, obstacles, skill args, fake-car options, check(car, ok)
    ("nothing in the way", [], {}, {},
     lambda c, ok: ok and 17 <= c.y <= 23 and abs(c.x) < 0.1),
    ("10 cm box dead ahead", [(-5, 5, 24, 34)], {}, {},
     lambda c, ok: ok and c.y >= 17 and c.x > 10),
    ("same box, told to go right", [(-5, 5, 24, 34)], {"side": "right"}, {},
     lambda c, ok: ok and c.y >= 17 and c.x < -10),
    ("box that reaches far to the left: comes back, goes right", [(-5, 60, 24, 34)], {}, {},
     lambda c, ok: ok and c.y >= 17 and c.x < -10),
    ("wall right across: gives up on the starting line, no forward move", [(-80, 80, 26, 30)], {}, {},
     lambda c, ok: not ok and abs(c.x) < 3 and c.y < 1),
    ("box beyond the 20 cm: ignored", [(-5, 5, 45, 55)], {}, {},
     lambda c, ok: ok and abs(c.x) < 0.1),
    ("35 cm, box that is too far to matter at the start", [(-5, 5, 45, 55)], {"dist_cm": 35}, {},
     lambda c, ok: ok and c.y >= 30 and abs(c.x) > 10),
    ("narrow sonar beam (+-8 deg)", [(-5, 5, 24, 34)], {}, {"beam": 8},
     lambda c, ok: ok and c.y >= 17),
    ("wide sonar beam (+-25 deg)", [(-5, 5, 24, 34)], {}, {"beam": 25},
     lambda c, ok: ok and c.y >= 17),
    ("two boxes side by side with a gap too small for the car", [(-5, 5, 24, 34), (12, 22, 24, 34)], {}, {},
     lambda c, ok: (ok and c.y >= 17) or (not ok and c.y < 8)),
    ("the car is really 30 % slower sideways than assumed", [(-5, 5, 24, 34)], {}, {"side_cm_s": 10.5},
     lambda c, ok: ok and c.y >= 17),
    ("the car is really 25 % faster than assumed", [(-5, 5, 24, 34)], {}, {"fwd_cm_s": 21, "side_cm_s": 19},
     lambda c, ok: ok and c.y >= 17),
    ("sliding also creeps forward (20 %)", [(-5, 5, 24, 34)], {}, {"creep": 0.2},
     lambda c, ok: ok and c.y >= 17),
    ("no sonar: does not move", [], {}, {"sonar": False},
     lambda c, ok: not ok and c.y == 0),
    ("cancelled half way: stops", [], {}, {"cancel_at": 9},
     lambda c, ok: not ok and c.y < 13),
    ("35 cm, two boxes one after the other", [(-6, 6, 22, 30), (10, 26, 46, 54)], {"dist_cm": 35}, {},
     lambda c, ok: ok and c.y >= 30),
]


if __name__ == "__main__":
    verbose = "-v" in sys.argv
    if not verbose:
        import builtins
        real_print = builtins.print
        hw_masterpi.print = lambda *a, **k: None     # the skill's own progress lines
    bad = 0
    for name, boxes, args, kw, check in CASES:
        if verbose:
            print(f"--- {name}")
        car, ok, detail, secs = run(boxes, args, **kw)
        good = check(car, ok) and not car.hit
        bad += not good
        print(f"{'ok  ' if good else 'FAIL'} {name}: {'ok' if ok else 'fail'} - {detail}  "
              f"[ended {car.y:.0f} cm ahead, {car.x:+.0f} cm left, {secs:.0f} s{', TOUCHED an obstacle' if car.hit else ''}]")
    print(f"\n{len(CASES) - bad}/{len(CASES)} cases ok")
    sys.exit(1 if bad else 0)
