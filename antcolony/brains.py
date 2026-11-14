"""Host-side task state machines, one per robot.

ArmBrain  : NAV_PICK -> GRAB (on-board skill) -> NAV_BIN -> DROP (skill) -> RETREAT
DozerBrain: NAV_BEHIND -> PUSH (one or two stages) -> RETREAT
Both      : IDLE / PARK when there is nothing to do.

A brain returns one velocity command per tick (or None = send nothing, e.g. while the robot
runs a skill by itself or a timed pulse is settling).
"""
import math
import random

from .common import ZERO, clamp, dist, to_robot, unit, vel, wrap
from .control import Driver
from .planner import Grid, plan


class Brain:
    def __init__(self, co, robot):
        self.co, self.r, self.name, self.role = co, robot, robot.name, robot.role
        self.c = co.cfg["control"]
        self.drv = Driver(self.c, robot.rc.get("holonomic", False))
        self.state, self.t_state, self.obj, self.note = "IDLE", 0.0, None, ""
        self.goal, self.path, self.blocked, self.plan_t = None, [], False, -1e9
        self.ret_from = (0.0, 0.0)
        self.skill, self.sid = None, random.randrange(1000, 900000)
        self.yield_since = self.blocked_since = None
        self.evade_until, self.evade_goal = -1e9, None

    # ------------------------------------------------------------- bookkeeping
    def set(self, state, now, note=""):
        self.state, self.t_state, self.note = state, now, note
        self.goal, self.path, self.blocked, self.blocked_since = None, [], False, None
        if self.r.x is not None:
            self.ret_from = self.r.xy
        self.drv.reset()

    @property
    def free(self):
        return self.state in ("IDLE", "PARK")

    def priority(self):
        if self.state in ("GRAB", "DROP"):
            return 5                      # under on-board control: everyone else keeps clear
        if self.state == "PUSH":
            return 4
        if self.state in ("NAV_BIN", "RETREAT"):
            return 3
        return 1 if self.free else 2

    def release(self, o, now, ban=0.0):
        """Give an object back to the pool (optionally not to this robot for `ban` seconds)."""
        if o is not None and o.owner == self.name:
            o.owner = None
            if o.state in ("assigned", "pushing", "grabbing"):
                o.state = "free"
            if ban:
                o.ban[self.name] = now + ban
        self.obj = None

    def mine(self, o, *states):
        return o is not None and o.owner == self.name and o.state in states

    # ------------------------------------------------------------------ skills
    def skill_start(self, name, args, now, timeout):
        self.sid += 1
        self.skill = {"type": "skill", "id": self.sid, "name": name, "args": args}
        self.skill_t0, self.skill_sent, self.skill_to = now, now, timeout
        self.co.bus.send(self.name, self.skill)

    def skill_poll(self, now):
        """None while the skill runs, else (ok, detail). Re-sends the request until the robot echoes its id."""
        ts = (self.r.telem or {}).get("skill", {})
        if ts.get("id") == self.sid and ts.get("state") in ("ok", "fail"):
            self.skill = None
            return ts["state"] == "ok", ts.get("detail", "")
        if now - self.skill_t0 > self.skill_to:
            self.sid += 1
            self.co.bus.send(self.name, {"type": "skill", "id": self.sid, "name": "stop", "args": {}})
            self.skill = None
            return False, "timeout"
        if ts.get("id") != self.sid and now - self.skill_sent > 0.3:
            self.skill_sent = now
            self.co.bus.send(self.name, self.skill)
        return None

    # -------------------------------------------------------------- navigation
    def goto(self, goal, final_th, now, bin_pad=0.02):
        """Plan (and keep re-planning) a path to `goal`, follow it, end facing `final_th`."""
        cfg = self.co.cfg
        if self.goal is None or dist(goal, self.goal) > 0.03:
            self.goal, self.plan_t = (goal[0], goal[1]), -1e9
            self.drv.reset()
        if now - self.plan_t > self.c["replan_s"] and not self.drv.arrived and not self.drv.pulse:
            i1, i2 = {}, {}
            p = plan(self.r.xy, self.goal, self.co.obstacles_for(self, bin_pad),
                     cfg["arena"], cfg["cell"], cfg["wall_margin"], i1)
            if p is None or not i1["exact"]:               # clutter in the way: allow nudging loose objects aside
                p2 = plan(self.r.xy, self.goal, self.co.obstacles_for(self, bin_pad, objects=False),
                          cfg["arena"], cfg["cell"], cfg["wall_margin"], i2)
                if p2 is not None and (p is None or i2["exact"]):
                    p, i1 = p2, i2
            self.plan_t, self.blocked = now, p is None
            ok = p is not None and i1.get("exact")
            self.blocked_since = None if ok else (self.blocked_since or now)
            if not ok and not self.free:                    # an idle robot parked on the goal: ask it to move
                for b in self.co.brains:
                    if b is not self and b.free and b.r.x is not None and \
                            dist(b.r.xy, self.goal) < b.r.radius + self.r.radius + 0.12:
                        b.evade(self.goal, now)
            if p is not None:
                self.path = p
        if self.blocked or not self.path:
            return dict(ZERO), False
        cmd, done = self.drv.follow(self.r.pose, self.path, final_th, now)
        if done and dist(self.r.xy, self.goal) > 0.08:     # stopped short: something is standing on the goal
            self.drv.reset()
            return dict(ZERO), False
        return cmd, done

    def spot_ok(self, p, bin_pad=0.02, objects=True):
        """Could the robot stand at `p`? Other robots are ignored here: they move on."""
        cfg = self.co.cfg
        return Grid(cfg["arena"], cfg["cell"], cfg["wall_margin"],
                    self.co.obstacles_for(self, bin_pad, objects, robots=False)).free(p)

    def retreat(self, now, metres):
        """Back straight out. -> (command, done)"""
        if dist(self.r.xy, self.ret_from) > metres or now - self.t_state > 5.0:
            return dict(ZERO), True
        return vel(-0.5, 0, 0, guard=False), False

    def park(self, now):
        cmd, done = self.goto(self.r.home, None, now)
        return dict(ZERO) if done else cmd

    def evade(self, other_xy, now):
        """Idle and in someone's way: move to the free spot nearby that is farthest from them."""
        if now < self.evade_until:
            return
        spots = [(self.r.x + 0.4 * math.cos(k * math.pi / 4), self.r.y + 0.4 * math.sin(k * math.pi / 4)) for k in range(8)]
        spots = [p for p in spots if self.spot_ok(p)]
        if spots:
            self.evade_goal, self.evade_until = max(spots, key=lambda p: dist(p, other_xy)), now + 4.0

    def tick(self, now):
        if not self.r.fresh(now, self.c["pose_timeout"]) and self.state not in ("GRAB", "DROP"):
            return dict(ZERO)                              # no fresh pose: stand still
        if self.free and now < self.evade_until:
            cmd, done = self.goto(self.evade_goal, None, now)
            return dict(ZERO) if done else cmd
        if self.state == "IDLE":
            return dict(ZERO)
        if self.state == "PARK":
            return self.park(now)
        return self._tick(now)


class ArmBrain(Brain):
    def assign(self, o, now):
        self.obj, o.state, o.owner = o, "assigned", self.name
        self.beta, self.beta_at, self.bad, self.tries, self.drop_b = None, None, [], 0, None
        self.lock_t, self.no_lock_until = None, -1e9
        self.set("NAV_PICK", now, f"pick #{o.id}")

    def pick_pose(self, o):
        """Where to stand to grab `o`: (x, y, heading), or None. The arm has no wrist roll, so
        an elongated object is approached along its long axis (jaws close across the short one)."""
        r, pc = self.r, self.co.cfg["pick"]
        if self.beta is None or dist(o.xy, self.beta_at) > 0.05:
            bearing = math.atan2(o.y - r.y, o.x - r.x)
            b = self.co.world.bin
            lane = math.atan2(b.y - o.y, b.x - o.x) if o.staged else None
            if o.ang is not None:
                cands = [o.ang, o.ang + math.pi]
                if o.l / max(o.w, 1e-3) < 1.3:
                    cands += [o.ang + math.pi / 2, o.ang - math.pi / 2]
                cands += [bearing + 0.3 + k * math.pi / 4 for k in range(8)]      # last resort: any side
            else:
                cands = [bearing + k * math.pi / 4 for k in range(8)]
            cfg = self.co.cfg
            g = Grid(cfg["arena"], cfg["cell"], cfg["wall_margin"], self.co.obstacles_for(self, robots=False))
            best, d = None, pc["pregrasp_dist"] + r.rc.get("base_fwd", 0.055)
            if lane is not None:
                cands.append(lane)
            for i, b in enumerate(cands):
                if any(abs(wrap(b - x)) < 0.3 for x in self.bad):
                    continue
                goal = (o.x - d * math.cos(b), o.y - d * math.sin(b))
                if not g.free(goal):
                    continue
                cost = dist(r.xy, goal) + 0.05 * abs(wrap(b - bearing)) + (0.5 if o.ang is not None and i >= 4 else 0)
                if lane is not None:
                    cost += 0.6 * abs(wrap(b - lane))    # staged: prefer facing the bin
                if best is None or cost < best[0]:
                    best = (cost, b)
            if best is None:
                return None
            self.beta, self.beta_at = best[1], o.xy
        d = pc["pregrasp_dist"] + r.rc.get("base_fwd", 0.055)
        return o.x - d * math.cos(self.beta), o.y - d * math.sin(self.beta), self.beta

    def drop_bearing(self):
        """Side of the bin to drop from (direction bin -> robot), away from the dozer's mouth."""
        b, bc, r = self.co.world.bin, self.co.cfg["bin"], self.r
        d, best = bc["drop_dist"] + r.rc.get("base_fwd", 0.055), None
        here = math.atan2(r.y - b.y, r.x - b.x)            # straight ahead first, then every 30 deg
        for a in [here] + [k * math.pi / 6 for k in range(12)]:
            if bc.get("floor_entry") and abs(wrap(a - b.th)) < math.radians(bc["mouth_half_deg"]):
                continue
            goal = (b.x + d * math.cos(a), b.y + d * math.sin(a))
            if not self.spot_ok(goal, bin_pad=-0.04, objects=False):
                continue
            if best is None or dist(r.xy, goal) < best[0]:
                best = (dist(r.xy, goal), a)
        return best[1] if best else wrap(b.th + math.pi)

    def _tick(self, now):
        co, r, o, st = self.co, self.r, self.obj, self.state
        pc, bc = co.cfg["pick"], co.cfg["bin"]
        if st == "NAV_PICK":
            if not self.mine(o, "assigned"):
                self.obj = None
                self.set("IDLE", now)
                return dict(ZERO)
            pose = self.pick_pose(o)
            if pose is None:
                self.release(o, now, ban=8)
                self.set("IDLE", now)
                return dict(ZERO)
            cmd, done = self.goto(pose[:2], pose[2], now)
            if self.blocked_since and now - self.blocked_since > 3.0:
                self.release(o, now, ban=8)              # no way there right now: let someone else try
                self.set("IDLE", now)
                return dict(ZERO)
            if done:
                ex, ey = to_robot(r.pose, o.xy)
                if 0.12 < ex < 0.60 and abs(math.atan2(ey, ex)) < 0.45:
                    o.state = "grabbing"
                    label = o.label if pc["label_mode"] == "label" else "any"
                    self.skill_start("grab", {"label": label, "min_speed": pc["min_speed"],
                                              "fwd_cm": round(ex * 100, 1), "left_cm": round(ey * 100, 1)},
                                     now, pc["grab_timeout"])
                    self.set("GRAB", now, f"grab #{o.id}")
                    return None
                self.tries += 1                          # ended up somewhere it cannot see the object from
                self.bad.append(self.beta)
                self.beta = None
                if self.tries >= 3:
                    self.release(o, now, ban=10)
                    self.set("IDLE", now)
                return dict(ZERO)
            return cmd
        if st == "GRAB":
            res = self.skill_poll(now)
            if res is None:
                return None                              # the car drives itself for the last few cm
            ok, detail = res
            if ok:
                o.state = "held"
                co.log(f"{self.name} grabbed #{o.id} ({o.label})")
                self.set("NAV_BIN", now, f"carry #{o.id}")
            else:
                o.fails += 1
                if o.fails >= pc["max_fails"] and bc.get("floor_entry") and co.has_dozer(now):
                    o.cls = "push"                       # (floor-level bin mouth only) hand it to the bulldozer
                co.log(f"{self.name} missed #{o.id}: {detail} (fail {o.fails})")
                self.release(o, now, ban=6)
                self.next_state = "IDLE"
                self.set("RETREAT", now, "back off after a miss")
            return None
        if st == "NAV_BIN":
            b = co.world.bin
            o.x, o.y = r.front(r.rc.get("base_fwd", 0.055) + 0.12)        # it rides in the gripper
            if self.lock_t and now - self.lock_t > 25.0:    # cannot get in: give the other car a turn
                co.unlock(self.name)
                self.lock_t, self.drop_b, self.no_lock_until = None, None, now + 6.0
            near = dist(r.xy, b.xy) < bc["zone"] + 0.25 and now > self.no_lock_until
            if near and co.lock("bin_drop", self.name):
                self.lock_t = self.lock_t or now
                if self.drop_b is None:
                    self.drop_b = self.drop_bearing()
                ux, uy = math.cos(self.drop_b), math.sin(self.drop_b)
                d = bc["drop_dist"] + r.rc.get("base_fwd", 0.055)
                cmd, done = self.goto((b.x + d * ux, b.y + d * uy), self.drop_b + math.pi, now, bin_pad=-0.04)
                if done:
                    base = r.front(r.rc.get("base_fwd", 0.055))
                    off = abs(wrap(math.atan2(b.y - base[1], b.x - base[0]) - r.th))
                    if abs(dist(base, b.xy) - bc["drop_dist"]) > 0.05 or off > 0.35:
                        self.drop_b, self.goal = None, None      # could not get to that side: pick another
                        return dict(ZERO)
                    dc = co.cfg["drop"]
                    self.skill_start("drop", {"y_cm": dc["y_cm"], "z_cm": dc["z_cm"]}, now, dc["timeout"])
                    self.set("DROP", now, f"drop #{o.id}")
                    return None
                if cmd is not None:
                    cmd["guard"] = False                 # the bin itself is in front of the sonar
                return cmd
            ux, uy = unit(r.x - b.x, r.y - b.y)          # bin busy: wait on a ring around it, out of the way
            cmd, done = self.goto((b.x + bc["zone"] * ux, b.y + bc["zone"] * uy), None, now)
            return dict(ZERO) if done else cmd
        if st == "DROP":
            res = self.skill_poll(now)
            if res is None:
                return None
            if res[0]:
                co.deliver(o, self.name, now)
            else:
                co.log(f"{self.name} drop failed: {res[1]}")
                co.world.drop(o)
            self.obj, self.next_state = None, "IDLE"
            self.set("RETREAT", now, "leave the bin")
            return None
        if st == "RETREAT":
            cmd, done = self.retreat(now, 0.15)
            if done:
                co.unlock(self.name)
                self.set(self.next_state, now)
            return cmd
        return dict(ZERO)


class DozerBrain(Brain):
    def assign(self, o, now):
        self.obj, o.state, o.owner = o, "assigned", self.name
        self.stages, self.attempts, self.detours = self.plan_stages(o), 0, 0
        self.set("NAV_BEHIND", now, f"push #{o.id}")

    def detour(self, o, ideal, sd):
        """A 30 cm push in a direction the dozer CAN get behind, preferring one after which
        the real push becomes possible. None if the object is boxed in."""
        a, bn, best = self.co.cfg["arena"], self.co.world.bin, None
        final = next((g for k, g in self.stages if k != "free"), self.stages[-1][1])
        for da in (0.6, -0.6, 1.2, -1.2, 1.57, -1.57, 2.2, -2.2):
            d = ideal + da
            tgt = (o.x + 0.30 * math.cos(d), o.y + 0.30 * math.sin(d))
            if not (a["xmin"] + 0.25 < tgt[0] < a["xmax"] - 0.25 and a["ymin"] + 0.25 < tgt[1] < a["ymax"] - 0.25):
                continue
            if dist(tgt, bn.xy) < bn.r + 0.12 or not self.spot_ok((o.x - sd * math.cos(d), o.y - sd * math.sin(d))):
                continue
            ux, uy = unit(final[0] - tgt[0], final[1] - tgt[1])
            later = self.spot_ok((tgt[0] - sd * ux, tgt[1] - sd * uy), objects=False)
            score = (0 if later else 1, abs(da))
            if best is None or score < best[0]:
                best = (score, tgt)
        return best[1] if best else None

    def plan_stages(self, o):
        """[(kind, target point)]. With a floor-level bin mouth: straight in when the object
        already sits in front of the mouth, else first to a 'gate' point on the mouth axis."""
        b, bc, cfg = self.co.world.bin, self.co.cfg["bin"], self.co.cfg
        if not bc.get("floor_entry"):                      # default: park it beside the bin for the arm cars
            slot = self.co.stage_slot(o) or o.xy
            pts = plan(o.xy, slot, [(b.x, b.y, b.r + 0.22)], cfg["arena"], cfg["cell"], 0.25) or [slot]
            return [("free", p) for p in pts[:-1]] + [("stage", slot)]
        off = abs(wrap(math.atan2(o.y - b.y, o.x - b.x) - b.th))
        if off < math.radians(25) and dist(o.xy, b.xy) > b.r + 0.12:
            return [("bin", b.xy)]
        g = bc["gate_dist"] + b.r
        gate = (b.x + g * math.cos(b.th), b.y + g * math.sin(b.th))
        pts = plan(o.xy, gate, [(b.x, b.y, b.r + 0.30)], cfg["arena"], cfg["cell"], 0.25) or [gate]
        return [("free", p) for p in pts[:-1]] + [("gate", gate), ("bin", b.xy)]

    def _tick(self, now):
        co, r, o, st = self.co, self.r, self.obj, self.state
        pp = co.cfg["push"]
        bf, bw = r.rc.get("blade_fwd", 0.14), r.rc.get("blade_w", 0.22)
        if st == "RETREAT":
            cmd, done = self.retreat(now, pp["retreat"])
            if done:
                self.set("IDLE", now)
            return cmd
        if not self.mine(o, "assigned", "pushing"):
            self.obj = None
            self.set("IDLE", now)
            return dict(ZERO)
        kind, g = self.stages[0]
        dx, dy = unit(g[0] - o.x, g[1] - o.y)
        if st == "NAV_BEHIND":
            o.state = "assigned"
            sd = bf + o.r + pp["standoff"]
            start = (o.x - dx * sd, o.y - dy * sd)
            if not self.spot_ok(start):
                pre = self.detour(o, math.atan2(dy, dx), sd) if self.detours < 4 else None
                if pre is None:
                    co.log(f"{self.name} cannot get behind #{o.id} at ({o.x:.2f},{o.y:.2f}) for {kind}")
                    self.release(o, now, ban=10)         # cannot get behind it (wall / bin in the way)
                    self.set("IDLE", now)
                    return dict(ZERO)
                self.detours += 1                        # e.g. slide it along the wall / away from the bin first
                if kind == "free":
                    self.stages[0] = ("free", pre)
                else:
                    self.stages.insert(0, ("free", pre))
                return dict(ZERO)
            cmd, done = self.goto(start, math.atan2(dy, dx), now)
            if self.blocked_since and now - self.blocked_since > 4.0:
                co.log(f"{self.name} cannot reach the push start for #{o.id}")
                self.release(o, now, ban=8)
                self.set("IDLE", now)
                return dict(ZERO)
            if done:
                self.set("PUSH", now, f"push #{o.id} -> {kind}")
                self.lost_t = None
                return dict(ZERO)
            return cmd
        # ---- PUSH
        o.state = "pushing"
        ex, ey = to_robot(r.pose, o.xy)
        on_blade = ex < bf + o.r + 0.05 and abs(ey) < bw / 2
        if now - o.seen_t > 0.4 and on_blade:
            o.x, o.y = r.front(bf + o.r, clamp(ey, -bw / 2, bw / 2))       # hidden by the car: it is on the blade
            ex, ey = to_robot(r.pose, o.xy)
        b = co.world.bin
        if kind == "bin":
            unseen = now - o.seen_t > 0.4
            if dist(o.xy, b.xy) < b.r - 0.05 or (unseen and dist(r.front(bf), b.xy) < b.r - 0.07):   # well past the lip
                co.deliver(o, self.name, now)
                self.obj = None
                self.set("RETREAT", now, "leave the bin")
                return dict(ZERO)
            if dist(r.front(bf), b.xy) < b.r - 0.09:       # blade is in the mouth but the object is still outside
                self.attempts += 1
                self.set("RETREAT2", now, "missed the mouth - line up again")
                return dict(ZERO)
        elif dist(o.xy, g) < pp["tol"]:
            self.stages.pop(0)
            if kind == "free":
                self.stages = self.plan_stages(o)        # it moved: work out the route to the bin afresh
            if self.stages:
                self.set("NAV_BEHIND", now, f"line up #{o.id} with the bin")
            else:
                o.staged = True
                co.log(f"{self.name} staged #{o.id} beside the bin")
                self.release(o, now)
                self.set("RETREAT", now, "staged")
            return dict(ZERO)
        slipped = now - o.seen_t < 0.4 and (ex < bf - 0.04 or abs(ey) > bw / 2 + 0.02)
        self.lost_t = (self.lost_t or now) if slipped else None
        if (self.lost_t and now - self.lost_t > 0.8) or now - self.t_state > pp["stage_timeout"]:
            self.attempts += 1
            if self.attempts >= 4:
                co.log(f"{self.name} gave up pushing #{o.id} for now")
                self.release(o, now, ban=15)
                self.set("RETREAT", now, "gave up")
            else:
                self.set("RETREAT2", now, "lost it - line up again")
            return dict(ZERO)
        # steer onto the line through the object towards the target, then drive along it
        n = dx * (r.y - o.y) - dy * (r.x - o.x)             # + = robot is left of that line
        e = wrap(math.atan2(dy, dx) - clamp(math.atan(pp["k_n"] * n), -0.6, 0.6) - r.th)
        if abs(e) > 0.7:
            return vel(0, 0, math.copysign(0.4, e), guard=False)
        vy = clamp(-4.0 * n, -0.4, 0.4) if r.rc.get("holonomic") else 0.0
        return vel(pp["u"] * max(0.3, math.cos(e)), vy, clamp(self.c["k_w"] * e, -1, 1), guard=False)

    def tick(self, now):
        if self.state == "RETREAT2":                       # short back-off, then line up again
            if not self.r.fresh(now, self.c["pose_timeout"]):
                return dict(ZERO)
            cmd, done = self.retreat(now, 0.12)
            if done:
                self.set("NAV_BEHIND", now, f"push #{self.obj.id}" if self.obj else "")
            return cmd
        return super().tick(now)


def make_brain(co, robot):
    return (ArmBrain if robot.role == "arm" else DozerBrain)(co, robot)
