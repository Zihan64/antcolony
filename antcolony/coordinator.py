"""The coordinator: world model + task allocation + per-robot brains + traffic rules.

    python -m antcolony.coordinator --config config.json            # real robots
    python -m antcolony.coordinator --config config.json --sim      # against `python -m antcolony.sim`
    python -m antcolony.coordinator --overlay profiles/edge2.json --arena-from-tags
                                    # two arm cars, each puts one block at the arena edge
    ... --objects 0.50,0.40 1.00,0.70     # block positions typed by hand (tracker detections ignored)
Dashboard: http://localhost:8080      Check everything first: python -m antcolony.check
"""
import argparse
import json
import math
import socket
import time

from .brains import make_brain
from .common import PORT_CMD, PORT_TELEM, PORT_WORLD, ZERO, dec, dist, enc, load_config
from .triage import Triage
from .world import ACTIVE, World


class UdpBus:
    """Network side of the coordinator. world(): newest tracker snapshot; telem(name): newest
    robot telemetry; send(name, msg): one datagram to that robot's agent."""

    def __init__(self, cfg, sim=False):
        self.addr = {n: (("127.0.0.1", PORT_CMD + 10 + i) if sim else tuple(rc["addr"]))
                     for i, (n, rc) in enumerate(cfg["robots"].items())}
        self.sw = self._sock(cfg.get("world_port", PORT_WORLD))
        self.st = self._sock(PORT_TELEM)
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._world, self._telem = None, {}
        self._world_t = None             # tracker stamp "t" of the newest snapshot taken
        self.n_old = 0                   # snapshots dropped because a newer one had already arrived

    @staticmethod
    def _sock(port):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("0.0.0.0", port))
        s.setblocking(False)
        return s

    def pump(self):
        for s, world in ((self.sw, True), (self.st, False)):
            while True:
                try:
                    data, _ = s.recvfrom(65535)
                except (BlockingIOError, ConnectionError):
                    break
                m = dec(data)
                if not isinstance(m, dict):
                    continue
                if world:
                    self._take_world(m)
                elif "name" in m:
                    self._telem[m["name"]] = m

    RESTART_S = 5.0                      # a stamp this far back = the tracker was restarted: accept it

    def _take_world(self, m):
        """Keep the NEWEST tracker snapshot by its own stamp "t", not the last one to arrive: UDP
        over Wi-Fi can reorder, and an older datagram would move a robot back to where it was."""
        t = m.get("t")
        t = float(t) if isinstance(t, (int, float)) and math.isfinite(t) else None
        if t is not None and self._world_t is not None and self._world_t - self.RESTART_S < t <= self._world_t:
            self.n_old += 1
            return
        self._world, self._world_t = m, (t if t is not None else self._world_t)

    def world(self):
        self.pump()
        w, self._world = self._world, None
        return w

    def telem(self, name):
        return self._telem.get(name)

    def send(self, name, msg):
        try:
            self.tx.sendto(enc(msg), self.addr[name])
        except OSError:
            pass


class Coordinator:
    def __init__(self, cfg, bus, log_path=None, quiet=False, manual=None):
        self.cfg, self.bus, self.quiet = cfg, bus, quiet
        # block positions given by hand instead of detected: [(x, y)] -> fixed "detections"
        self.manual = None if manual is None else [{"id": 9001 + i, "label": "block", "x": x, "y": y}
                                                   for i, (x, y) in enumerate(manual)]
        self.pending, self.snap_t, self.warn_t, self.first_t = False, None, -1e9, None
        self.triage = Triage(cfg)
        self.world = World(cfg, self.triage.classify)
        self.brains = [make_brain(self, r) for r in self.world.robots.values()]
        self.locks, self.events = {}, []
        self.t0 = self.t_clear = self.t_done = None
        self.now = 0.0
        self.logf = open(log_path, "a", encoding="utf-8") if log_path else None

    # ------------------------------------------------------------------ services for brains
    def log(self, msg):
        line = f"[{self.now - (self.t0 or self.now):6.1f}s] {msg}"
        self.events.append(line)
        del self.events[:-30]
        if not self.quiet:
            print(line, flush=True)
        if self.logf:
            self.logf.write(json.dumps({"t": round(self.now, 2), "msg": msg}) + "\n")
            self.logf.flush()

    def lock(self, name, who):
        """Mutex, e.g. 'bin_drop': one arm car at the bin at a time."""
        if self.locks.get(name, who) != who:
            return False
        self.locks[name] = who
        return True

    def unlock(self, who):
        for k in [k for k, v in self.locks.items() if v == who]:
            del self.locks[k]

    def deliver(self, o, by, now):
        self.world.deliver(o)
        where = "at the edge" if self.world.edge else "in the bin"
        self.log(f"{by} delivered #{o.id} ({o.label})  [{len(self.world.delivered)} {where}]")

    def grabbed(self, o):
        """An object is in a gripper: a hand-typed position for it is no longer true."""
        if self.manual:
            self.manual = [m for m in self.manual if m["id"] != o.src]

    def obstacles_for(self, brain, bin_pad=0.02, objects=True, robots=True):
        """Discs the centre of `brain`'s robot must stay out of: other robots, objects, the bin."""
        w, me, tc = self.world, brain.r, self.cfg["traffic"]
        obs = [(r.x, r.y, r.radius + me.radius + tc["robot_gap"])
               for r in w.robots.values() if robots and r is not me and r.x is not None]
        for b in self.brains:                                # a car creeping up to its object: keep out of its way
            if robots and b is not brain and b.state == "GRAB" and b.r.x is not None:
                fx, fy = b.r.front(0.22)
                obs.append((fx, fy, 0.14 + me.radius))
        if objects:                                          # clearance is set by the car's half width
            obs += [(o.x, o.y, o.r + tc["obj_gap"]) for o in w.objects
                    if o.state in ("free", "assigned", "pushing", "grabbing")]
            # unknown things (not a trained part): keep clear, unless it lies on this robot's own
            # target (the same thing seen twice) - that would wall the target off
            mine = brain.obj.xy if getattr(brain, "obj", None) is not None else None
            obs += [(x, y, r + tc["obj_gap"]) for x, y, r in w.unknown
                    if mine is None or dist((x, y), mine) > r + 0.06]
        if w.bin.on:
            obs.append((w.bin.x, w.bin.y, w.bin.r + me.radius + bin_pad))
        if w.edge:                                           # what already lies at the edge stays where it is
            lying = [(p[0], p[1]) for p in w.parked]
            lying += [o.xy for o in w.delivered if all(dist(o.xy, q) > 0.08 for q in lying)]
            obs += [(x, y, 0.02 + tc["obj_gap"]) for x, y in lying]
        obs += [(x, y, tc.get("tag_gap", 0.10) + me.radius) for x, y in w.floor_tags.values()]   # paper: keep off
        return obs

    def has_dozer(self, now):
        return any(b.role == "dozer" and b.r.fresh(now, 5.0) for b in self.brains)

    def stage_slot(self, o):
        """Nearest free parking spot beside the bin for the dozer to push `o` to (None if all taken).
        The spots sit on an arc in front of the bin, far enough apart that an arm car can drive
        in along one 'lane', pick up, and carry straight on to the bin without touching the rest."""
        b, bc = self.world.bin, self.cfg["bin"]
        sr = bc.get("stage_r", 0.45)
        slots = [(b.x + sr * math.cos(b.th + a), b.y + sr * math.sin(b.th + a))
                 for a in bc.get("stage_angles", (0.0, 0.75, -0.75, 1.5, -1.5))]
        ok = [s for s in slots if self.world._in_arena(s, -0.15)
              and all(dist(s, x.xy) > 0.20 for x in self.world.objects if x is not o)]
        return min(ok, key=lambda s: dist(s, o.xy)) if ok else None

    # ------------------------------------------------------------------ allocation
    def allocate(self, now):
        """Give each idle robot a job. Arm cars take what is cheapest to fetch (near them, near the
        bin, parked spots first); the dozer takes the farthest thing left and brings it to the bin."""
        w, ac, bc = self.world, self.cfg["alloc"], self.cfg["bin"]
        mouth = bool(bc.get("floor_entry"))
        free = [o for o in w.objects if o.state == "free"]
        busy_pts = [b.obj.xy for b in self.brains if b.obj is not None and not b.free]
        idle = [b for b in self.brains if b.free and b.r.fresh(now, self.cfg["control"]["pose_timeout"])]
        # is there still something a robot may do later (temporary bans run out, permanent ones do not)?
        self.pending = any(b.available and b.role == "arm" and o.cls != "push"
                           and o.ban.get(b.name, 0) != float("inf") for b in self.brains for o in free)
        for b in sorted(idle, key=lambda b: b.role == "dozer"):          # arm cars choose first
            if not b.available:                                          # gave up, or has done its share
                if b.state == "IDLE" and b.r.home is not None and dist(b.r.xy, b.r.home) > 0.12 \
                        and now > b.evade_until + 3.0:
                    b.set("PARK", now, "gave up" if b.retired else "done")
                continue
            best = None
            for o in free:
                if o.ban.get(b.name, 0) > now or any(dist(o.xy, p) < ac["spacing"] for p in busy_pts):
                    continue
                d_bin = dist(o.xy, w.bin.xy) if w.bin.on else 0.0
                if b.role == "arm":
                    if o.cls == "push":
                        continue                                         # too big for the gripper
                    cost = dist(b.r.xy, o.xy) + 0.5 * d_bin
                elif not w.bin.on:
                    continue                                             # edge mode: nothing for a dozer to do
                elif mouth:                                              # optional mode: push straight into the bin
                    if o.cls != "push" and not ac["dozer_helps"]:
                        continue
                    cost = (0 if o.cls == "push" else 10) - d_bin
                else:                                                    # default: bring far objects to the bin
                    if o.staged or o.cls == "push" or d_bin < bc.get("stage_r", 0.45) + 0.25 \
                            or self.stage_slot(o) is None:
                        continue
                    cost = -d_bin
                if best is None or cost < best[0]:
                    best = (cost, o)
            if best:
                b.assign(best[1], now)
                free.remove(best[1])
                busy_pts.append(best[1].xy)
            elif b.state == "IDLE" and b.r.home is not None and dist(b.r.xy, b.r.home) > 0.12 \
                    and now > b.evade_until + 20.0:
                b.set("PARK", now, "nothing to do")

    # ------------------------------------------------------------------ traffic
    def traffic(self, cmds, now):
        """Last line of defence (paths already avoid robots). Inside the stop distance nobody
        may drive towards the other robot; the lower-priority one stops a little earlier, and
        an idle one gets out of the way."""
        bs, gap = self.brains, self.cfg["traffic"]["stop_gap"]
        for b in bs:
            b._yield = False
        for i, a in enumerate(bs):
            for b in bs[i + 1:]:
                if a.r.x is None or b.r.x is None:
                    continue
                d = dist(a.r.xy, b.r.xy)
                lim = a.r.radius + b.r.radius + gap
                if d > lim + 0.05:
                    continue
                lo, hi = (a, b) if (a.priority(), b.name) < (b.priority(), a.name) else (b, a)
                if lo.free and not hi.free:
                    lo.evade(hi.r.xy, now)                   # parked in a working robot's way: step aside
                for me, other in ((lo, hi),) + (((hi, lo),) if d < lim else ()):
                    c = cmds.get(me.name)
                    if not c or (not c["vx"] and not c["vy"]):
                        continue
                    th = me.r.th
                    wx = c["vx"] * math.cos(th) - c["vy"] * math.sin(th)
                    wy = c["vx"] * math.sin(th) + c["vy"] * math.cos(th)
                    closing = (wx * (other.r.x - me.r.x) + wy * (other.r.y - me.r.y)) / (math.hypot(wx, wy) * max(d, 1e-6))
                    if closing > 0.3:
                        cmds[me.name] = dict(ZERO)
                        me._yield = True
        for b in bs:
            b.yield_since = (b.yield_since or now) if b._yield else None
            if b.yield_since and now - b.yield_since > 3.0:
                b.plan_t = -1e9                              # stuck behind someone: plan a new way round
                b.yield_since = now

    # ------------------------------------------------------------------ main step
    def tick(self, now):
        self.now = now
        snap = self.bus.world()
        if snap is not None:
            self.snap_t = now
            self.no_floor = snap.get("floor_valid") is False     # tracker heartbeat: arena not set up
            if self.manual is not None:                      # positions typed by hand replace the detector
                snap = dict(snap, objects=[dict(m) for m in self.manual], objects_ok=True)
            self.world.update(snap, now)
            if self.t0 is None and self.world.n_seen:
                self.t0 = now
                self.log(f"start: {len(self.world.active())} objects on the floor")
        for b in self.brains:
            b.r.telem = self.bus.telem(b.name)
        self.allocate(now)
        cmds = {b.name: b.tick(now) for b in self.brains}
        self.traffic(cmds, now)
        for b in self.brains:
            c = cmds[b.name]
            if c is not None:
                rc = b.r.rc
                self.bus.send(b.name, dict(c, type="vel", vy=c["vy"] * rc.get("vy_sign", 1),
                                           w=c["w"] * rc.get("w_sign", 1)))
        self.health(now)
        todo = any(o.state in ACTIVE for o in self.world.objects)
        limited = bool(self.cfg["pick"].get("max_tries") or self.cfg.get("deliver", {}).get("per_robot"))
        if limited:                                          # finished = nobody is working and nobody may start
            todo = self.pending or any(not b.free for b in self.brains)
        if self.t0 is not None and self.t_done is None:
            self.t_clear = None if todo else (self.t_clear or now)
            if self.t_clear and now - self.t_clear > 3.0 and all(b.free for b in self.brains):
                self.t_done = self.t_clear
                if limited:
                    per = ", ".join(f"{b.name}: " + (f"{b.done_n} delivered" if b.done_n else
                                                     "gave up" if b.retired else "nothing delivered")
                                    for b in self.brains if b.role == "arm")
                    self.log(f"FINISHED in {self.t_done - self.t0:.0f} s - {per}")
                else:
                    self.log(f"ARENA CLEAR: {len(self.world.delivered)} objects in {self.t_done - self.t0:.0f} s")

    def health(self, now):
        """Every 5 s, say what is missing (tracker data, a robot's tag, a robot's telemetry)."""
        self.first_t = now if self.first_t is None else self.first_t
        if self.quiet or now - self.warn_t < 5.0 or now - self.first_t < 3.0:
            return
        self.warn_t = now
        bad = []
        if self.snap_t is None or now - self.snap_t > 2.0:
            bad.append(f"no tracker data on UDP {self.cfg.get('world_port', PORT_WORLD)}")
        elif getattr(self, "no_floor", False):
            bad.append("tracker connected but it has no floor yet (set up the arena / corner tags in its GUI)")
        else:
            for b in self.brains:
                if not b.r.fresh(now, 2.0):
                    bad.append(f"{b.name}: tag {b.r.rc['tag']} not seen")
        for b in self.brains:
            if b.r.x is not None and self.bus.telem(b.name) is None:
                bad.append(f"{b.name}: no telemetry from {self.cfg['robots'][b.name]['addr'][0]} (agent.py running?)")
        if bad:
            print("  [waiting] " + "; ".join(bad), flush=True)

    def state(self):
        """Everything the dashboard draws."""
        w = self.world
        return {
            "t": round((self.t_done or self.now) - (self.t0 or self.now), 1), "done": self.t_done is not None,
            "arena": self.cfg["arena"],
            "bin": {"x": w.bin.x, "y": w.bin.y, "th": w.bin.th, "r": w.bin.r, "on": w.bin.on,
                    "mouth": bool(self.cfg["bin"].get("floor_entry"))},
            "edge": w.dc.get("edge_zone", 0.20) if w.edge else None,
            "lying": [[p[0], p[1]] for p in w.parked] + [list(o.xy) for o in w.delivered] if w.edge else [],
            "floor_tags": [list(p) for p in w.floor_tags.values()],
            "robots": [{"name": b.name, "role": b.role, "x": b.r.x, "y": b.r.y, "th": b.r.th, "r": b.r.radius,
                        "state": b.state, "note": b.note, "path": b.path[:20],
                        "batt": (b.r.telem or {}).get("batt_mv")} for b in self.brains if b.r.x is not None],
            "objects": [{"id": o.id, "x": o.x, "y": o.y, "r": o.r, "cls": o.cls, "state": o.state,
                         "owner": o.owner, "label": o.label} for o in w.objects if o.state != "new"],
            "delivered": len(w.delivered), "total": w.n_seen, "log": self.events[-12:],
        }


def arena_from_tags(bus, cfg, wait_s=15.0, log=print):
    """Set cfg["arena"] to the rectangle spanned by the tracker's arena-corner floor tags.
    True if the tracker sent them within `wait_s`."""
    t_end = time.monotonic() + wait_s
    while time.monotonic() < t_end:
        s = bus.world()
        floor = (s or {}).get("floor") or {}
        ids = [str(i) for i in (s or {}).get("corners") or []] or list(floor)
        pts = [floor[i] for i in ids if i in floor]
        if len(pts) >= 3:
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            cfg["arena"].update(xmin=round(min(xs), 3), xmax=round(max(xs), 3),
                                ymin=round(min(ys), 3), ymax=round(max(ys), 3))
            a = cfg["arena"]
            log(f"[arena] from floor tags {ids}: x {a['xmin']}..{a['xmax']} m, y {a['ymin']}..{a['ymax']} m "
                f"({a['xmax'] - a['xmin']:.2f} x {a['ymax'] - a['ymin']:.2f} m)")
            return True
        time.sleep(0.05)
    log(f"[arena] the tracker sent no floor tags in {wait_s:.0f} s - keeping the arena from the config: {cfg['arena']}")
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--overlay", help="JSON with only the keys that differ from the config, e.g. profiles/edge2.json")
    ap.add_argument("--sim", action="store_true", help="robots are simulated on localhost (python -m antcolony.sim)")
    ap.add_argument("--arena-from-tags", action="store_true",
                    help="arena = rectangle of the tracker's corner floor tags (instead of config 'arena')")
    ap.add_argument("--objects", nargs="+", metavar="X,Y",
                    help="object positions typed by hand, metres in the tracker frame; the tracker's own "
                         "object detections are then ignored (its tags are still used)")
    ap.add_argument("--no-dash", action="store_true")
    ap.add_argument("--log", default="run_log.jsonl")
    ap.add_argument("--max-secs", type=float, default=0.0, help="stop every robot and exit after this long (0 = run until Ctrl+C)")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overlay)
    bus = UdpBus(cfg, a.sim)
    if a.arena_from_tags:
        arena_from_tags(bus, cfg)
    manual = [tuple(float(v) for v in o.split(",")) for o in a.objects] if a.objects else None
    co = Coordinator(cfg, bus, a.log, manual=manual)
    dv, pk = cfg.get("deliver", {}), cfg["pick"]
    print(f"robots: {', '.join(cfg['robots'])} | deliver: {dv.get('mode', 'bin')}"
          + (f", {dv['per_robot']} per robot" if dv.get("per_robot") else "")
          + (f" | up to {pk['max_tries']} grab tries" if pk.get("max_tries") else "")
          + (f" | {len(manual)} objects typed by hand" if manual else ""))
    if not a.no_dash:
        from .dashboard import serve
        serve(co)
    dt = 1.0 / cfg["control"]["hz"]
    print(f"coordinator running at {cfg['control']['hz']} Hz - Ctrl+C stops every robot")
    t_start = time.monotonic()
    try:
        while True:
            t = time.monotonic()
            co.tick(t)
            if a.max_secs and t - t_start > a.max_secs:
                print(f"--max-secs {a.max_secs:.0f} reached - stopping")
                break
            if co.t_done is not None and a.max_secs and all(b.state == "IDLE" or b.r.home is None or
                                                              dist(b.r.xy, b.r.home) < 0.15 for b in co.brains if b.r.x is not None):
                print("finished and everyone is home - stopping")
                break
            time.sleep(max(0.0, dt - (time.monotonic() - t)))
    except KeyboardInterrupt:
        pass
    finally:
        for _ in range(3):
            for b in co.brains:
                co.bus.send(b.name, dict(ZERO, type="vel"))
            time.sleep(0.05)
        print("\nstopped")


if __name__ == "__main__":
    main()
