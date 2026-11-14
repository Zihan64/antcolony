"""2-D simulator that stands in for BOTH the overhead tracker and the robot agents, so the
coordinator can be developed and tested with no hardware.

    python -m antcolony.sim --config config.json --objects 6      # then, in another terminal:
    python -m antcolony.coordinator --config config.json --sim    # and open http://localhost:8080

What it models (roughly): dead-band and lag of open-loop mecanum drive, camera latency and
noise, detection drop-outs, objects hidden under a car, a flat blade that lets objects slide
off, grasps that miss, objects too wide for the gripper, and - with --quirks - tags mounted
at an unknown angle/offset and motors wired the other way round (what calibrate.py solves).
"""
import argparse
import collections
import math
import random
import socket
import time

from .common import PORT_CMD, PORT_TELEM, PORT_WORLD, clamp, dec, dist, enc, load_config, sgn, to_robot, to_world, wrap

V_MIN, V_MAX, W_MIN, W_MAX = 0.07, 0.22, 0.9, 2.4        # m/s and rad/s at the smallest / largest command
TAU, LAT_EFF, WATCHDOG, R_BODY = 0.12, 0.6, 0.4, 0.12


class SimRobot:
    def __init__(self, name, rc, rng, quirks, start=None):
        self.name, self.rc, self.role = name, rc, rc["role"]
        home = rc["home"] if isinstance(rc.get("home"), (list, tuple)) else start   # "start": wherever it stands
        (self.x, self.y), self.th = home, rng.uniform(-math.pi, math.pi)
        self.start = tuple(home)
        self.cmd, self.cmd_until, self.v = (0.0, 0.0, 0.0), -1.0, [0.0, 0.0, 0.0]
        self.skill = {"id": 0, "state": "idle", "detail": ""}
        self.job = self.hold = self.target = None
        self.q = {"th_sign": 1, "th_off": 0.0, "tag_off": (0.0, 0.0), "vy_sign": 1, "w_sign": 1}
        if quirks:
            self.q = {"th_sign": rng.choice((1, -1)), "th_off": rng.uniform(-math.pi, math.pi),
                      "tag_off": (rng.uniform(-0.09, -0.05), rng.uniform(-0.01, 0.01)),
                      "vy_sign": rng.choice((1, -1)), "w_sign": rng.choice((1, -1))}

    @property
    def pose(self):
        return (self.x, self.y, self.th)

    @property
    def xy(self):
        return (self.x, self.y)


class SimObj:
    def __init__(self, i, x, y, w, l, ang):
        self.id, self.x, self.y, self.w, self.l, self.ang, self.state = i, x, y, w, l, ang, "free"

    @property
    def xy(self):
        return (self.x, self.y)

    @property
    def r(self):
        return self.l / 2


class SimWorld:
    def __init__(self, cfg, seed=0, n_objects=6, quirks=False, p_grasp=0.75, big_frac=0.35, latency=0.15,
                 floor_tags=False):
        self.cfg, self.rng, self.t, self.dt = cfg, random.Random(seed), 0.0, 0.02
        self.p_grasp, self.latency = p_grasp, latency
        a, n = cfg["arena"], len(cfg["robots"])
        self.edge = cfg.get("deliver", {}).get("mode", "bin") == "edge"      # no bin: put things at the boundary
        self.zone = cfg.get("deliver", {}).get("edge_zone", 0.20)
        self.robots = {name: SimRobot(name, rc, self.rng, quirks,
                                      (a["xmin"] + (i + 1) * (a["xmax"] - a["xmin"]) / (n + 1), a["ymin"] + 0.25))
                       for i, (name, rc) in enumerate(cfg["robots"].items())}
        bc = cfg["bin"]
        self.bx, self.by, self.bth, self.br = bc["pose"][0], bc["pose"][1], math.radians(bc["pose"][2]), bc["radius"]
        # paper tags at the arena corners, as the real tracker reports them (ids as in cam_control/arena.json)
        self.floor = {"29": [a["xmin"], a["ymin"]], "32": [a["xmax"], a["ymin"]],
                      "31": [a["xmax"], a["ymax"]], "30": [a["xmin"], a["ymax"]]} if floor_tags else None
        self.grabs = {name: 0 for name in self.robots}                       # grab attempts per robot
        self.objs = []
        while len(self.objs) < n_objects:
            m = cfg.get("field_margin", 0.30)            # debris only inside the taped field, never against the edge
            x, y = self.rng.uniform(a["xmin"] + m, a["xmax"] - m), self.rng.uniform(a["ymin"] + m, a["ymax"] - m)
            if (not self.edge and dist((x, y), (self.bx, self.by)) < self.br + 0.25) or \
                    any(dist((x, y), r.xy) < 0.30 for r in self.robots.values()) or \
                    any(dist((x, y), o.xy) < 0.16 for o in self.objs):
                continue
            if self.rng.random() < big_frac:
                w = self.rng.uniform(0.07, 0.10)                     # too wide for the gripper
                l = w * self.rng.uniform(1.0, 1.4)
            else:
                w = self.rng.uniform(0.02, 0.05)
                l = min(0.12, w * self.rng.uniform(1.0, 2.5))
            self.objs.append(SimObj(len(self.objs) + 1, x, y, w, l, self.rng.uniform(0, math.pi)))
        self.hist, self.next_snap = collections.deque(maxlen=50), 0.0
        self.collisions, self.bump_s, self._touch = 0, 0.0, set()
        self.solo = None            # name of the only robot on the floor (calibration), else None

    # ------------------------------------------------------------- robot agent side
    def handle(self, name, msg):
        r = self.robots[name]
        if msg.get("type") == "vel":
            if r.job is None:
                r.cmd = (clamp(msg["vx"], -1, 1), clamp(msg["vy"], -1, 1), clamp(msg["w"], -1, 1))
                r.cmd_until = self.t + min(msg.get("dur", WATCHDOG), WATCHDOG)
        elif msg.get("type") == "skill" and msg["id"] != r.skill["id"]:
            r.cmd, r.job, r.target = (0.0, 0.0, 0.0), None, None
            r.skill = {"id": msg["id"], "state": "running", "detail": ""}
            fn = {"grab": self._grab, "drop": self._drop}.get(msg["name"])
            if fn:
                r.job = fn(r, msg.get("args", {}))
            else:                                                   # stop / home / open: instant
                r.skill["state"] = "ok"

    def telem(self, name):
        r = self.robots[name]
        return {"type": "telem", "name": name, "batt_mv": 7900, "skill": dict(r.skill), "holding": r.hold is not None}

    def _wait(self, secs):
        t_end = self.t + secs
        while self.t < t_end:
            yield

    def _end(self, r, ok, detail=""):
        r.skill["state"], r.skill["detail"], r.target = ("ok" if ok else "fail"), detail, None

    def _edge_dist(self, p):
        a = self.cfg["arena"]
        return min(p[0] - a["xmin"], a["xmax"] - p[0], p[1] - a["ymin"], a["ymax"] - p[1])

    def _grab(self, r, args):
        bf = r.rc.get("base_fwd", 0.055)
        self.grabs[r.name] += 1
        yield from self._wait(2.0)                                  # camera looks ahead, VLM call
        base = to_world(r.pose, bf)
        seen = []
        for o in self.objs:
            ex, ey = to_robot((base[0], base[1], r.th), o.xy)
            if o.state == "free" and 0.04 < ex < 0.55 and abs(math.atan2(ey, ex)) < 0.6:
                seen.append((ex, o))
        if not seen:
            yield from self._wait(1.0)
            return self._end(r, False, "not seen")
        o = r.target = min(seen, key=lambda s: s[0])[1]
        t_end = self.t + 20.0
        while True:                                                 # colour tracking steers the car up to it
            base = to_world(r.pose, bf)
            ex, ey = to_robot((base[0], base[1], r.th), o.xy)
            if abs(ex - 0.115) < 0.012 and abs(ey) < 0.015:
                break
            if self.t > t_end or o.state != "free":
                return self._end(r, False, "approach did not finish")
            r.th = wrap(r.th + clamp(2.0 * math.atan2(ey, max(ex, 0.03)), -1.0, 1.0) * self.dt)
            v = clamp(2.0 * (ex - 0.115), -0.08, 0.08)
            r.x, r.y = r.x + v * math.cos(r.th) * self.dt, r.y + v * math.sin(r.th) * self.dt
            yield
        yield from self._wait(9.0)                                  # locate, align, lower, close, lift, miss check
        if o.state == "free" and o.w <= 0.058 and self.rng.random() < self.p_grasp:
            o.state, r.hold = "held", o
            return self._end(r, True, "holding")
        o.x, o.y = o.x + self.rng.uniform(-0.02, 0.02), o.y + self.rng.uniform(-0.02, 0.02)
        self._end(r, False, "miss")

    def _drop(self, r, args):
        yield from self._wait(4.0)
        o, bf = r.hold, r.rc.get("base_fwd", 0.055)
        if o and self.edge:                                         # it lands where the arm reaches to
            o.x, o.y = to_world(r.pose, bf + args.get("y_cm", 16) / 100.0)
            o.state = "parked" if self._edge_dist(o.xy) < self.zone else "free"
            r.hold = None
        elif o:
            base = to_world(r.pose, bf)
            d, dd = dist(base, (self.bx, self.by)), self.cfg["bin"]["drop_dist"]
            facing = abs(wrap(math.atan2(self.by - base[1], self.bx - base[0]) - r.th))
            if dd - 0.07 <= d <= dd + 0.05 and facing < 0.45:
                o.state = "binned"
            else:                                                   # let go in the wrong place: it lands on the floor
                o.state, (o.x, o.y) = "free", to_world(r.pose, bf + 0.18)
            r.hold = None
        self._end(r, True)

    # ------------------------------------------------------------------- physics
    def _drive(self, r):
        dt = self.dt
        vx, vy, w = r.cmd if self.t < r.cmd_until else (0.0, 0.0, 0.0)
        vy, w = vy * r.q["vy_sign"], w * r.q["w_sign"]
        if not r.rc.get("mecanum", True):
            vy = 0.0
        h = math.hypot(vx, vy)
        if h > 0.02:
            sp = V_MIN + (V_MAX - V_MIN) * min(1.0, h)
            tx, ty, tw = sp * vx / h, sp * vy / h * LAT_EFF, w * W_MAX
        else:
            tx = ty = 0.0
            tw = sgn(w) * (W_MIN + (W_MAX - W_MIN) * abs(w)) if abs(w) > 0.02 else 0.0
        k = min(1.0, dt / TAU)
        for i, tgt in enumerate((tx, ty, tw)):
            r.v[i] += (tgt - r.v[i]) * k
        n = 1 + self.rng.gauss(0, 0.05)
        c, s = math.cos(r.th), math.sin(r.th)
        r.x += (r.v[0] * c - r.v[1] * s) * n * dt
        r.y += (r.v[0] * s + r.v[1] * c) * n * dt
        r.th = wrap(r.th + r.v[2] * n * dt)

    def _mouth(self, p):
        return bool(self.cfg["bin"].get("floor_entry")) and \
            abs(wrap(math.atan2(p[1] - self.by, p[0] - self.bx) - self.bth)) < 0.9

    def _physics(self):
        a, rs = self.cfg["arena"], [r for r in self.robots.values() if self.solo in (None, r.name)]
        for r in rs:
            r.x, r.y = clamp(r.x, a["xmin"] + 0.02, a["xmax"] - 0.02), clamp(r.y, a["ymin"] + 0.02, a["ymax"] - 0.02)
            d = dist(r.xy, (self.bx, self.by))
            lim = self.br + R_BODY
            if r.role == "dozer" and self._mouth(r.xy):
                lim = r.rc.get("blade_fwd", 0.14) - 0.03             # the blade may go into the mouth
            if 1e-6 < d < lim and not self.solo and not self.edge:
                r.x, r.y = self.bx + (r.x - self.bx) * lim / d, self.by + (r.y - self.by) * lim / d
        touch = set()
        for i, p in enumerate(rs):
            for q in rs[i + 1:]:
                d = dist(p.xy, q.xy)
                if 1e-6 < d < 2 * R_BODY:
                    touch.add((p.name, q.name))
                    k = (2 * R_BODY - d) / 2 / d
                    dx, dy = (p.x - q.x) * k, (p.y - q.y) * k
                    p.x, p.y, q.x, q.y = p.x + dx, p.y + dy, q.x - dx, q.y - dy
        self.collisions += len(touch - self._touch)
        self._touch = touch
        for o in self.objs:
            if o.state not in ("free", "parked") or self.solo:
                continue
            for r in rs:
                if r.target is o:
                    continue
                ex, ey = to_robot(r.pose, o.xy)
                bf, bw = r.rc.get("blade_fwd", 0.14), r.rc.get("blade_w", 0.22)
                if r.role == "dozer" and 0 < ex < bf + o.r and abs(ey) < bw / 2 + 0.01:
                    ey = ey * (1 + 0.25 * self.dt) + self.rng.gauss(0, 0.0015)   # flat blade: it creeps sideways
                    o.x, o.y = to_world(r.pose, bf + o.r, ey)
                    continue
                d = math.hypot(ex, ey)
                if 1e-6 < d < R_BODY + o.r:
                    k = (R_BODY + o.r) / d
                    o.x, o.y = to_world(r.pose, ex * k, ey * k)
                    self.bump_s += self.dt
            o.x, o.y = clamp(o.x, a["xmin"] + 0.01, a["xmax"] - 0.01), clamp(o.y, a["ymin"] + 0.01, a["ymax"] - 0.01)
            if self.edge:                                           # lying in the edge band = delivered
                o.state = "parked" if self._edge_dist(o.xy) < self.zone else "free"
                continue
            d = dist(o.xy, (self.bx, self.by))
            if d < self.br + o.r * 0.5:
                if self._mouth(o.xy):
                    if d < self.br - 0.01:
                        o.state = "binned"
                elif d > 1e-6:
                    k = (self.br + o.r * 0.5) / d
                    o.x, o.y = self.bx + (o.x - self.bx) * k, self.by + (o.y - self.by) * k
        for r in rs:
            if r.hold:
                r.hold.x, r.hold.y = to_world(r.pose, r.rc.get("base_fwd", 0.055) + 0.12)

    def step(self, dt=0.02):
        self.dt = dt
        self.t += dt
        for r in self.robots.values():
            if r.job is not None:
                try:
                    next(r.job)
                except StopIteration:
                    r.job = None
            else:
                self._drive(r)
        self._physics()
        if self.t >= self.next_snap:
            self.next_snap = self.t + 0.1
            self.hist.append(self.snapshot())

    def advance(self, secs):
        t_end = self.t + secs
        while self.t < t_end:
            self.step()

    # ------------------------------------------------------------------ tracker side
    def snapshot(self):
        g, tags = self.rng.gauss, {}
        for r in self.robots.values():
            tx, ty = to_world(r.pose, *r.q["tag_off"])
            h = r.q["th_sign"] * wrap(r.th - r.q["th_off"])
            tags[str(r.rc["tag"])] = [tx + g(0, 0.003), ty + g(0, 0.003), wrap(h + g(0, 0.01))]
        tags[str(self.cfg["bin"]["tag"])] = [self.bx, self.by, self.bth]
        objs = []
        for o in self.objs:
            if o.state not in ("free", "parked") or any(dist(o.xy, r.xy) < 0.10 for r in self.robots.values()):
                continue
            if self.rng.random() < 0.05:
                continue                                            # detector missed it this frame
            objs.append({"id": 1000 + o.id, "label": "debris", "x": o.x + g(0, 0.003), "y": o.y + g(0, 0.003),
                         "w": o.w, "l": o.l, "ang": o.ang})
        snap = {"t": self.t, "tags": tags, "objects": objs}
        if self.floor:
            snap["floor"], snap["corners"] = self.floor, [29, 32, 31, 30]
            a = self.cfg["arena"]                          # camera on a stand beside the arena, looking in
            snap["camera"] = [(a["xmin"] + a["xmax"]) / 2, a["ymin"] - 1.2, 1.8]
        return snap

    def delayed(self):
        """Newest snapshot that is at least `latency` old (what the camera pipeline would deliver now)."""
        best = None
        for s in self.hist:
            if s["t"] <= self.t - self.latency:
                best = s
        return best

    def left(self):
        return sum(1 for o in self.objs if o.state not in ("binned", "parked"))


class SimBus:
    """In-process stand-in for UdpBus (deterministic tests, faster than real time)."""

    def __init__(self, sim):
        self.sim, self._last = sim, -1.0

    def pump(self):
        pass

    def world(self):
        s = self.sim.delayed()
        if s is None or s["t"] <= self._last:
            return None
        self._last = s["t"]
        return s

    def telem(self, name):
        return self.sim.telem(name)

    def send(self, name, msg):
        self.sim.handle(name, msg)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--objects", type=int, default=6)
    ap.add_argument("--quirks", action="store_true", help="random tag mounting / motor wiring (run calibrate.py first)")
    ap.add_argument("--overlay", help="same overlay file as the coordinator, e.g. profiles/edge2.json")
    ap.add_argument("--p-grasp", type=float, default=0.75, help="chance that a grab holds (0 = every grab misses)")
    ap.add_argument("--floor-tags", action="store_true", help="also report paper tags at the arena corners")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overlay)
    sim = SimWorld(cfg, a.seed, a.objects, a.quirks, p_grasp=a.p_grasp, floor_tags=a.floor_tags,
                   big_frac=0.0 if cfg.get("deliver", {}).get("mode") == "edge" else 0.35)
    socks = {}
    for i, n in enumerate(cfg["robots"]):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(("127.0.0.1", PORT_CMD + 10 + i))
        s.setblocking(False)
        socks[n] = s
    tx, last, next_tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM), -1.0, 0.0
    print(f"simulator: {a.objects} objects, robots {list(socks)} - now start the coordinator with --sim")
    t0 = time.monotonic()
    while True:
        for n, s in socks.items():
            while True:
                try:
                    data, _ = s.recvfrom(65535)
                except (BlockingIOError, ConnectionError):
                    break
                m = dec(data)
                if isinstance(m, dict):
                    sim.handle(n, m)
        while sim.t < time.monotonic() - t0:
            sim.step()
        if sim.t >= next_tx:
            next_tx = sim.t + 0.1
            snap = sim.delayed()
            if snap and snap["t"] > last:
                last = snap["t"]
                tx.sendto(enc(snap), ("127.0.0.1", cfg.get("world_port", PORT_WORLD)))
            for n in socks:
                tx.sendto(enc(sim.telem(n)), ("127.0.0.1", PORT_TELEM))
        time.sleep(0.005)


if __name__ == "__main__":
    main()
