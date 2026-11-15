"""World model: turns raw tracker snapshots into robots, a bin and persistent objects.

Snapshot (from the tracker bridge, JSON over UDP):
    {"t": 12.3,
     "tags": {"10": [x_m, y_m, heading_rad], ...},
     "objects": [{"id": 1000, "label": "bag", "x": .., "y": .., "w": .., "l": .., "ang": ..}, ...]}
Everything is in the tracker's floor frame, metres. `ang` (long-axis direction) is optional.
Optional extras: "floor": {"29": [x, y], ...} = paper tags lying on the floor (robots keep off
them), "corners": [29, 32, 31, 30] = which of those mark the arena corners, "camera": [x, y, z]
= where the camera is. A camera that looks at a slant sees a tall robot lying across the floor
behind it; detections in that shadow are parts of the robot, not objects (see World.hidden).
"objects_ok": false = the tracker did not look for objects this time (main camera silent, no
floor): tags are applied, the object list is left as it is.
"obstacles": [[x, y, r], ...] = unknown things (not one of the fine-tuned parts): kept clear of
(Coordinator.obstacles_for), never allocated as targets.

deliver.mode "edge" (instead of the bin): an object counts as delivered once it lies in the
band `deliver.edge_zone` wide along the arena boundary. Detections in that band are never
targets, only obstacles (`parked`).
"""
import math

from .common import clamp, dist, finite, to_world, wrap

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
        self.height = rc.get("height", 0.35 if self.role == "arm" else 0.12)   # tallest part above the floor, m
        self.shadow_r = rc.get("shadow_r", 0.12)       # half width of what that tall part covers (the arm column)
        # where the tall part is: metres ahead of the centre. The arm rests stretched forward
        # (hw_masterpi CARRY), so its shadow starts in front of the car, not at its centre
        self.arm_at = rc.get("arm_at", (0.0, 0.07, 0.14, 0.21) if self.role == "arm" else (0.0,))

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
        self.camera = None               # (x, y, z) of the overhead camera, if the tracker says
        self.parked = []                 # edge mode: [x, y, last seen] of things lying in the edge band
        self.unknown = []                # (x, y, r) of unknown things: obstacles only, never picked up
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
        for r in self.robots.values():
            if r.x is None:
                continue
            if dist(p, r.xy) < r.mask_r + pad:
                return True
            cam = self.camera
            if cam and cam[2] > r.height + 0.2:
                # a point `height` up, seen from the camera, lands on the floor this far behind where
                # it stands. Only the arm is that tall, so the strips are narrow (shadow_r), one from
                # each of a few points along the arm (centre ... gripper)
                k = r.height / (cam[2] - r.height)
                c, s = math.cos(r.th), math.sin(r.th)
                for f in r.arm_at:
                    ax, ay = r.x + c * f, r.y + s * f
                    dx, dy = (ax - cam[0]) * k, (ay - cam[1]) * k
                    t = clamp(((p[0] - ax) * dx + (p[1] - ay) * dy) / max(dx * dx + dy * dy, 1e-9), 0.0, 1.0)
                    if dist(p, (ax + t * dx, ay + t * dy)) < r.shadow_r + pad:
                        return True
        return False

    # ------------------------------------------------------------------ update
    def update(self, snap, now):
        """Apply one tracker snapshot. Malformed entries (wrong length, non-numbers, NaN) are
        skipped one by one: a bad datagram must neither crash the coordinator nor put a NaN pose
        into the controllers (which would turn into NaN motor commands)."""
        self.t = now
        tags = snap.get("tags")
        tags = tags if isinstance(tags, dict) else {}
        for r in self.robots.values():
            t = finite(tags.get(str(r.rc["tag"])), 3)
            if t:
                r.set_tag(t[0], t[1], t[2], now)
        t = finite(tags.get(str(self.bin.bc.get("tag"))), 3)
        if t:
            self.bin.x, self.bin.y = t[0], t[1]
            self.bin.th = wrap(t[2] + math.radians(self.bin.bc.get("th_off_deg", 0.0)))

        cam = finite(snap.get("camera"), 3)
        if cam:
            self.camera = cam
        if isinstance(snap.get("floor"), dict) and snap["floor"]:
            own = {str(r.rc["tag"]) for r in self.robots.values()} | {str(self.bin.bc.get("tag"))}
            fl = {str(k): finite(v, 2) for k, v in snap["floor"].items() if str(k) not in own}
            self.floor_tags = {k: v for k, v in fl.items() if v}

        if snap.get("objects_ok") is not False and isinstance(snap.get("obstacles"), list):
            # unknown things (not one of the trained parts): never targets, only kept clear of
            obs = (finite(u, 3) for u in snap["obstacles"])
            self.unknown = [u for u in obs if u and self._in_arena(u, 0.05) and not self.hidden(u, now)]
        if snap.get("objects_ok") is False:
            # the tracker did not look for objects in this snapshot (main camera silent, floor lost):
            # an empty list is "no news", not "everything is gone". Keep the objects as they are
            # rather than timing out the very ones the cars are driving to.
            return
        dets = []
        for d in snap.get("objects") or []:
            p = finite((d.get("x"), d.get("y")), 2) if isinstance(d, dict) else None
            if p is None:
                continue
            d = dict(d, x=p[0], y=p[1])
            for k in ("w", "l", "ang", "h"):             # optional fields: drop rather than trust junk
                if k in d and finite((d[k],), 1) is None:
                    del d[k]
            if self._in_arena(p, 0.05) and not self.hidden(p, now):
                dets.append(d)
        max_h = self.oc.get("max_h")             # taller than this = not something to pick up
        if max_h:
            dets = [d for d in dets if not d.get("h") or d["h"] <= max_h]
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
            if o.state == "free" and o.cls == "push":    # first sized too big (merged with a shadow, ...): look again
                o.cls = self.classify(o)
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
