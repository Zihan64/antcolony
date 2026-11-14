"""Shared helpers: geometry, wire format, config loading. Standard library only."""
import json
import math
import os

PORT_WORLD = 9100    # tracker  -> coordinator  (UDP JSON world snapshots)
PORT_CMD = 9200      # coordinator -> robot agent
PORT_TELEM = 9201    # robot agent -> coordinator
PORT_DASH = 8080     # dashboard (HTTP)
ZERO = {"vx": 0.0, "vy": 0.0, "w": 0.0}


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def unit(dx, dy):
    n = math.hypot(dx, dy)
    return (dx / n, dy / n) if n > 1e-9 else (1.0, 0.0)


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def sgn(v):
    return 1.0 if v >= 0 else -1.0


def to_robot(pose, p):
    """World point -> robot frame (forward, left)."""
    x, y, th = pose[0], pose[1], pose[2]
    dx, dy = p[0] - x, p[1] - y
    c, s = math.cos(th), math.sin(th)
    return c * dx + s * dy, -s * dx + c * dy


def to_world(pose, f, l=0.0):
    """Robot-frame point (forward, left) -> world."""
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return pose[0] + c * f - s * l, pose[1] + s * f + c * l


def vel(vx=0.0, vy=0.0, w=0.0, dur=None, guard=True):
    d = {"vx": float(vx), "vy": float(vy), "w": float(w)}
    if dur is not None:
        d["dur"] = float(dur)
    if not guard:
        d["guard"] = False
    return d


def enc(d):
    return json.dumps(d, separators=(",", ":")).encode()


def dec(b):
    try:
        return json.loads(b.decode())
    except (ValueError, UnicodeDecodeError):
        return None


def apply_network(cfg):
    """Override robot IPs from the active network profile.

    The profile is cfg["network"], or the ANTCOLONY_NET environment variable if set.
    A null IP in the profile keeps the robot's own "addr" from config.json.
    """
    net = os.environ.get("ANTCOLONY_NET") or cfg.get("network")
    if not net:
        return None
    profiles = cfg.get("networks", {})
    if net not in profiles:
        raise SystemExit(f"unknown network profile {net!r}; known: {', '.join(profiles) or 'none'}")
    for name, ip in profiles[net].items():
        if ip and name in cfg["robots"]:
            cfg["robots"][name]["addr"] = [ip, cfg["robots"][name]["addr"][1]]
    print(f"[config] network profile: {net}")
    return net


def load_config(path):
    """config.json + per-robot calibration from calib/<name>.json next to it."""
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    apply_network(cfg)
    cdir = os.path.join(os.path.dirname(os.path.abspath(path)), "calib")
    for name, rc in cfg["robots"].items():
        p = os.path.join(cdir, name + ".json")
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                rc.update(json.load(f))
    c = cfg["control"]
    for k in ("ang_tol", "turn_first", "fine_ang"):
        c[k] = math.radians(c[k + "_deg"])
    return cfg
