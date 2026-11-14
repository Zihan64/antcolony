"""One-minute drive calibration for one robot, using the overhead tracker as the ruler.

    python -m antcolony.calibrate --config config.json --robot A        (add --sim for the simulator)

Put the robot in the middle of the arena with ~50 cm of free floor around it, tracker and
robot agent running, coordinator NOT running. The robot drives forward, turns, drives forward
again and strafes. From what the tracker saw this works out, and saves to calib/<robot>.json:

  th_sign, th_off  tag heading -> direction the robot actually drives when told "forward"
  tag_off          where the tag sits relative to the point the robot turns about (m: fwd, left)
  w_sign, vy_sign  +1/-1 so that "turn left" / "strafe left" mean the same to robot and tracker
  strafe_ok        False if strafing barely moves the robot (carpet): keep "holonomic": false

Nothing has to be measured by hand, and it does not matter which way round the tag is glued
on or whether the tracker's y axis points up or down.
"""
import argparse
import json
import math
import os
import time

import numpy as np

from .common import ZERO, load_config, wrap


def _cmean(angles):
    return math.atan2(sum(math.sin(a) for a in angles), sum(math.cos(a) for a in angles))


def _tag(bus, tag, wait, n=5):
    xs, ys, hs = [], [], []
    for _ in range(200):
        wait(0.05)
        s = bus.world()
        if s and tag in s.get("tags", {}):
            x, y, h = s["tags"][tag]
            xs.append(x), ys.append(y), hs.append(h)
            if len(xs) >= n:
                return sum(xs) / n, sum(ys) / n, _cmean(hs)
    raise RuntimeError(f"tag {tag} not seen by the tracker")


def _run(bus, name, tag, cmd, secs, wait, samples=None, until_turn=None):
    t, turned, last = 0.0, 0.0, None
    while t < secs:
        bus.send(name, dict(cmd, type="vel", guard=False))
        wait(0.1)
        t += 0.1
        s = bus.world()
        if s and tag in s.get("tags", {}):
            p = s["tags"][tag]
            if samples is not None:
                samples.append(p)
            if last is not None:
                turned += wrap(p[2] - last)
            last = p[2]
        if until_turn and abs(turned) >= until_turn:
            break
    bus.send(name, dict(ZERO, type="vel"))
    wait(0.9)
    return turned


def calibrate(bus, name, cfg, wait=time.sleep, u=0.6, log=print):
    tag = str(cfg["robots"][name]["tag"])
    p0 = _tag(bus, tag, wait)
    _run(bus, name, tag, {"vx": u, "vy": 0, "w": 0}, 1.5, wait)
    p1 = _tag(bus, tag, wait)
    rot = [p1]
    _run(bus, name, tag, {"vx": 0, "vy": 0, "w": u}, 5.0, wait, rot, until_turn=1.3)
    p2 = _tag(bus, tag, wait)
    rot.append(p2)
    _run(bus, name, tag, {"vx": u, "vy": 0, "w": 0}, 1.5, wait)
    p3 = _tag(bus, tag, wait)
    d1, d2 = math.dist(p0[:2], p1[:2]), math.dist(p2[:2], p3[:2])
    if min(d1, d2) < 0.05:
        raise RuntimeError(f"{name} barely moved ({d1:.2f} / {d2:.2f} m): agent running? battery? min duty too low?")
    dh = sum(wrap(b[2] - a[2]) for a, b in zip(rot, rot[1:]))
    if abs(dh) < 0.5:
        raise RuntimeError(f"{name} did not turn enough ({math.degrees(dh):.0f} deg)")
    phi1 = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    phi2 = math.atan2(p3[1] - p2[1], p3[0] - p2[0])
    dphi = wrap(phi2 - phi1)
    s = 1 if abs(wrap(dphi - dh)) <= abs(wrap(dphi + dh)) else -1
    th_off = _cmean([wrap(phi1 - s * _cmean([p0[2], p1[2]])), wrap(phi2 - s * _cmean([p2[2], p3[2]]))])
    w_sign = 1 if s * dh > 0 else -1
    # the tag circles the turning centre: tag_i = centre + R(theta_i) * tag_off
    A, b = [], []
    for x, y, h in rot:
        th = s * h + th_off
        A += [[1, 0, math.cos(th), -math.sin(th)], [0, 1, math.sin(th), math.cos(th)]]
        b += [x, y]
    sol = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)[0]
    ox, oy = float(sol[2]), float(sol[3])

    def centre(p):
        th = s * p[2] + th_off
        return p[0] - (math.cos(th) * ox - math.sin(th) * oy), p[1] - (math.sin(th) * ox + math.cos(th) * oy), th

    c0 = centre(p3)
    _run(bus, name, tag, {"vx": 0, "vy": u, "w": 0}, 1.2, wait)
    c1 = centre(_tag(bus, tag, wait))
    dx, dy = c1[0] - c0[0], c1[1] - c0[1]
    lat = -math.sin(c0[2]) * dx + math.cos(c0[2]) * dy
    strafe_ok = abs(lat) > 0.04 and abs(lat) > 0.6 * math.hypot(dx, dy)
    res = {"th_sign": s, "th_off": round(th_off, 4), "tag_off": [round(ox, 4), round(oy, 4)],
           "w_sign": w_sign, "vy_sign": (1 if lat >= 0 else -1) if strafe_ok else 1, "strafe_ok": strafe_ok,
           "v_fwd_at_%.1f" % u: round((d1 + d2) / 3.0, 3)}
    log(f"{name}: {res}")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--robot", required=True)
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--u", type=float, default=0.6, help="command level used for the test moves (0..1)")
    a = ap.parse_args()
    from .coordinator import UdpBus
    cfg = load_config(a.config)
    res = calibrate(UdpBus(cfg, a.sim), a.robot, cfg, u=a.u)
    d = os.path.join(os.path.dirname(os.path.abspath(a.config)), "calib")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, a.robot + ".json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"saved {os.path.join(d, a.robot + '.json')}")


if __name__ == "__main__":
    main()
