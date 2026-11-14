"""World model: turns raw tracker snapshots into robots, a bin and persistent objects.

Snapshot (from the tracker bridge, JSON over UDP):
    {"t": 12.3,
     "tags": {"10": [x_m, y_m, heading_rad], ...},
     "objects": [{"id": 1000, "label": "bag", "x": .., "y": .., "w": .., "l": .., "ang": ..}, ...]}
Everything is in the tracker's floor frame, metres. `ang` (long-axis direction) is optional.
Optional extras: "floor": {"29": [x, y], ...} = paper tags lying on the floor (robots keep off
them), "corners": [29, 32, 31, 30] = which of those mark the arena corners.

deliver.mode "edge" (instead of the bin): an object counts as delivered once it lies in the
band `deliver.edge_zone` wide along the arena boundary. Detections in that band are never
targets, only obstacles (`parked`).
"""
import math

from .common import clamp, dist, to_world, wrap

LIVE = ("new", "free", "assigned", "pushing", "grabbing")       # states matched against detections
ACTIVE = ("free", "assigned", "pushing", "grabbing", "held")    # states that still need work


class Robot:
    def __init__(self, name, rc):
        self.name, self.rc, self.role = name, rc, rc["role"]
        self.radius = rc.get("radius", 0.15)
        h = rc.get("home")                 # [x, y], or "start" = wherever the robot is first seen
        self.home = tuple(h) if isinstance(h, (list, tuple)) else None
        self.x = self.y = self.th = None
        self.seen_t = -1e9
        self.telem = None
        # detections this close to the tag-derived centre are the robot itself; for the dozer the mask
        # must end just behind the blade face, or the object it is pushing would vanish too
        self.mask_r = rc.get("mask_r", 0.22 if self.role == "arm" else rc.get("blade_fwd", 0.14) - 0.02)

    def set_tag(self, x, y, h, now):
        """Tag pose -> pose of the chassis centre, using this robot's calibration."""
        rc = self.rc
        th = wrap(rc.get("th_sign", 1) * h + rc.get("th_off", 0.0))
        ox, oy = rc.get("tag_off", (0.0, 0.0))            # tag position in the robot frame (fwd, left)
        c, s = math.cos(th), math.sin(th)
        self.x, self.y, self.th, self.seen_t = x - (c * ox - s * oy), y - (s * ox + c * oy), th, now
        if self.home is None:
            self.home = (self.x, self.y)

    @property
    def pose(self):
        return (self.x, self.y, self.th)

    @property
    def xy(self):
        return (self.x, self.y)

    def front(self, d, left=0.0):
        return to_world(self.pose, d, left)

    def fresh(self, now, timeout):
        return self.x is not None and now - self.seen_t < timeout


class Bin:
    def __init__(self, bc):
        self.bc = bc
        self.x, self.y = bc["pose"][0], bc["pose"][1]
        self.th = math.radians(bc["pose"][2])              # direction the open (mouth) side faces
        self.r = bc["radius"]
        self.on = True                     # False in edge mode: there is no bin on the floor

    @property
    def xy(self):
        return (self.x, self.y)


class Obj:
    _n = 0

    def __init__(self, d, now, default_size):
        Obj._n += 1
        self.id = Obj._n
        self.x, self.y = d["x"], d["y"]
        self.w = d.get("w") or default_size
        self.l = d.get("l") or default_size
        if self.w > self.l:
            self.w, self.l = self.l, self.w
        self.ang = d.get("ang")
        self.label = d.get("label", "object")
        self.src = d.get("id")
        self.crop = d.get("crop")
        self.hits, self.born, self.seen_t = 1, now, now
        self.state, self.owner, self.cls = "new", None, None
        self.fails, self.staged, self.ban = 0, False, {}

    @property
    def xy(self):
        return (self.x, self.y)

    @property
    def r(self):
        return clamp(self.l / 2, 0.02, 0.12)

    def see(self, d, now):
        a = 0.5
        self.x += a * (d["x"] - self.x)
        self.y += a * (d["y"] - self.y)
        if d.get("w") and d.get("l"):
            w, l = sorted((d["w"], d["l"]))
            self.w += 0.2 * (w - self.w)
            self.l += 0.2 * (l - self.l)
        if d.get("ang") is not None:
            self.ang = d["ang"]
        self.crop = d.get("crop") or self.crop
        self.hits += 1
        self.seen_t = now


class World:
    def __init__(self, cfg, classify=None):
        self.cfg, self.oc = cfg, cfg["objects"]
        self.robots = {n: Robot(n, rc) for n, rc in cfg["robots"].items()}
        self.bin = Bin(cfg["bin"])
        self.dc = cfg.get("deliver", {})
        self.edge = self.dc.get("mode", "bin") == "edge"
        self.bin.on = not self.edge
        self.floor_tags = {}             # id -> (x, y) of paper tags on the floor
        self.parked = []                 # edge mode: [x, y, last seen] of things lying in the edge band
        self.objects = []                # live + held
        self.delivered = []
        self.n_seen = 0
        self.classify = classify or (lambda o: "grasp")
        self.t = 0.0

    # ------------------------------------------------------------------ helpers
    def _in_arena(self, p, pad=0.0):
        a = self.cfg["arena"]
        return a["xmin"] - pad <= p[0] <= a["xmax"] + pad and a["ymin"] - pad <= p[1] <= a["ymax"] + pad

    def edge_dist(self, p):
        """Distance from `p` to the arena boundary (negative outside)."""
        a = self.cfg["arena"]
        return min(p[0] - a["xmin"], a["xmax"] - p[0], p[1] - a["ymin"], a["ymax"] - p[1])

    def hidden(self, p, now, pad=0.0):
        """True where the tracker cannot be trusted: under a robot or in the bin."""
        if self.bin.on and dist(p, self.bin.xy) < self.bin.r + 0.02:
            return True
        return any(r.x is not None and dist(p, r.xy) < r.mask_r + pad for r in self.robots.values())

    # ------------------------------------------------------------------ update
    def update(self, snap, now):
        self.t = now
        tags = snap.get("tags", {})
        for r in self.robots.values():
            t = tags.get(str(r.rc["tag"]))
            if t:
                r.set_tag(t[0], t[1], t[2], now)
        t = tags.get(str(self.bin.bc.get("tag")))
        if t:
            self.bin.x, self.bin.y = t[0], t[1]
            self.bin.th = wrap(t[2] + math.radians(self.bin.bc.get("th_off_deg", 0.0)))

        if snap.get("floor"):
            own = {str(r.rc["tag"]) for r in self.robots.values()} | {str(self.bin.bc.get("tag"))}
            self.floor_tags = {str(k): (v[0], v[1]) for k, v in snap["floor"].items() if str(k) not in own}

        dets = [d for d in snap.get("objects", [])
                if self._in_arena((d["x"], d["y"]), 0.05) and not self.hidden((d["x"], d["y"]), now)]
        if self.edge:                            # things in the edge band are done: obstacles, not targets
            zone = self.dc.get("edge_zone", 0.20)
            for d in dets:
                if self.edge_dist((d["x"], d["y"])) < zone:
                    near = [p for p in self.parked if dist(p, (d["x"], d["y"])) < 0.08]
                    if near:
                        near[0][:] = [d["x"], d["y"], now]
                    else:
                        self.parked.append([d["x"], d["y"], now])
            self.parked = [p for p in self.parked if now - p[2] < 5.0 or self.hidden(p, now, 0.05)]
            dets = [d for d in dets if self.edge_dist((d["x"], d["y"])) >= zone]
        live = [o for o in self.objects if o.state in LIVE]
        pairs = sorted((dist(o.xy, (d["x"], d["y"])), oi, di)
                       for oi, o in enumerate(live) for di, d in enumerate(dets))
        used_o, used_d = set(), set()
        for dd, oi, di in pairs:
            if dd > self.oc["gate"]:
                break
            if oi in used_o or di in used_d:
                continue
            used_o.add(oi)
            used_d.add(di)
            live[oi].see(dets[di], now)
        for di, d in enumerate(dets):
            if di not in used_d:
                self.objects.append(Obj(d, now, self.oc["default_size"]))

        for o in list(self.objects):
            if o.state == "new":
                if o.hits >= self.oc["min_hits"]:
                    o.state, o.cls = "free", self.classify(o)
                    self.n_seen += 1
                elif now - o.born > 1.5:
                    self.objects.remove(o)
            elif o.state in ("free", "assigned") and now - o.seen_t > self.oc["timeout"] \
                    and not self.hidden(o.xy, now, 0.05):
                o.state = "gone"                         # it really is not there any more
                self.objects.remove(o)

    def active(self):
        return [o for o in self.objects if o.state in ACTIVE]

    def deliver(self, o):
        o.state = "delivered"
        if o in self.objects:
            self.objects.remove(o)
        self.delivered.append(o)

    def drop(self, o):
        """Forget an object (lost, or let go somewhere unknown). The tracker re-finds it if it is still there."""
        o.state = "gone"
        if o in self.objects:
            self.objects.remove(o)
