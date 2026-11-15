"""antcolony.patrol on a fake car: a point with a heading and a block somewhere; no network, a
simulated clock. Turn rates from 0.7x to 1.6x the first guess, a block behind the car, out of
sight (the car patrols), a blocked first hop, nothing there.

    python tests/test_patrol.py
"""
import math, sys, types
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), ".."))
import antcolony.patrol as P
CLOCK = [0.0]
P.time = types.SimpleNamespace(monotonic=lambda: CLOCK[0])

class FakeLink:
    def __init__(self, block, rate=180.0, blocked=False):
        self.x = self.y = self.th = 0.0          # cm, heading rad (0 = +y, + = left)
        self.block, self.rate, self.blocked, self.vw, self.moves, self.dead_left = block, rate, blocked, 0, [], 0.0
    def rel(self):
        dx, dy = self.block[0] - self.x, self.block[1] - self.y
        d = math.hypot(dx, dy)
        b = math.degrees(math.atan2(-dx, dy)) - math.degrees(self.th)
        return (b + 180) % 360 - 180, d
    def send(self, m):
        if m["type"] == "vel":
            w = (1 if m["w"] > 0 else -1 if m["w"] < 0 else 0)
            if w and not self.vw:
                self.dead_left = 0.12                     # a new turn pulse: the car needs a moment to start
            self.vw = w
    def pump(self, s=0.1):
        CLOCK[0] += s
        run = max(0.0, s - self.dead_left); self.dead_left = max(0.0, self.dead_left - s)
        self.th += math.radians(self.rate) * self.vw * run
        return None
    def skill(self, name, args, timeout):
        if name == "seek":
            b, d = self.rel()
            wide = min(1.0, 6.0 / max(d, 1.0))
            if abs(b) > 33 or d > 250:
                return True, f"{args['target']}: not in the picture; sonar 500 cm", 1
            return True, f"{args['target']}: {abs(b):.0f} deg {'left' if b > 0 else 'right'}, {round(100 * wide)} % of the width; sonar 500 cm", 1
        if name == "carry":
            if self.blocked:
                self.blocked = False
                return False, "blocked", 1
            c = args["dist_cm"]
            self.x += -math.sin(self.th) * c; self.y += math.cos(self.th) * c
            return True, "ok", 1
        return True, "ok", 1

def run(block, **kw):
    a = types.SimpleNamespace(label="red block", spots=4, hop_cm=50, scan_deg=40, close=0.09, aim_deg=8, far_step=35,
                              near_step=15, max_steps=14, w=0.3, turn_deg_s=250, turn_dead_s=0.12)
    L = FakeLink(block, **kw)
    ok = P.Patrol(L, a, lambda m: None).run()
    b, d = L.rel()
    return ok, d, b

bad = 0
for name, block, kw, want in [
    ("ahead 1.5 m", (0, 150), {}, True), ("turns 0.7x the guess", (90, -80), {"rate": 175}, True), ("turns 1.6x the guess", (-90, -80), {"rate": 400}, True), ("23 deg left, 1.8 m", (-70, 165), {}, True),
    ("behind", (30, -120), {}, True), ("right side, turns 0.7x the guess", (120, 20), {"rate": 175}, True),
    ("fast turns", (-100, -60), {"rate": 400}, True), ("out of sight at 3.5 m: patrol", (0, 350), {}, True),
    ("first hop blocked: turns left, patrols there", (-270, 0), {"blocked": True}, True), ("out of sight 3.2 m ahead, slow turns", (20, 320), {"rate": 175}, True), ("nothing there", (5000, 5000), {}, False)]:
    ok, d, b = run(block, **kw)
    good = ok == want and (not ok or (d < 80 and abs(b) < 15))
    bad += not good
    print(f"{'ok  ' if good else 'FAIL'} {name}: found={ok} dist {d:.0f} cm bearing {b:+.0f}")
print("all ok" if not bad else f"{bad} failed")
sys.exit(1 if bad else 0)
