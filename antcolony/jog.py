"""One short move of one robot, measured by the overhead tracker. For finding out what a robot
really does with a command before trusting it with longer ones.

    python -m antcolony.jog --robot A --w 0.3 --secs 0.3      # "turn left" for 0.3 s
    python -m antcolony.jog --robot A --vx 0.3 --secs 0.3     # "forward"
    python -m antcolony.jog --robot A --vy 0.3 --secs 0.3     # "strafe left"

Tracker and the robot's agent.py running, coordinator NOT running. The robot's tag must be in
view; if the tracker loses it for more than 0.4 s during the move the robot is stopped.
Prints how far the tag moved (cm, and in which direction relative to where the tag pointed
before) and how much it turned (degrees, + = counter-clockwise seen from above).

What to expect from a healthy arm car (mecanum):
  --w    turns on the spot: tens of degrees, the tag moves on a small arc (a few cm if the tag is
         not on the turning centre)
  --vx   moves along one straight line, hardly any turn
  --vy   moves at right angles to the --vx line (little on carpet), hardly any turn
A "turn" that moves the tag in a straight line with almost no rotation means the motor
mapping is wrong (two wheels swapped): fix motor_directions.json on that car first.
"""
import argparse
import math
import time

from .common import ZERO, load_config, wrap
from .coordinator import UdpBus


def tag_pose(bus, tag, secs=1.0):
    """Mean tag pose over `secs`, or None if it was never seen."""
    xs, ys, hs, t_end = [], [], [], time.monotonic() + secs
    while time.monotonic() < t_end:
        s = bus.world()
        if s and tag in s.get("tags", {}):
            x, y, h = s["tags"][tag]
            xs.append(x), ys.append(y), hs.append(h)
        time.sleep(0.02)
    if not xs:
        return None
    return (sum(xs) / len(xs), sum(ys) / len(ys),
            math.atan2(sum(math.sin(h) for h in hs), sum(math.cos(h) for h in hs)))


def jog(bus, name, tag, cmd, secs, log=print):
    p0 = tag_pose(bus, tag)
    if p0 is None:
        raise SystemExit(f"tag {tag} is not in view - nothing sent")
    t0, unseen, last_seen, lost = time.monotonic(), 0.0, time.monotonic(), False
    while time.monotonic() - t0 < secs:
        bus.send(name, dict(cmd, type="vel", dur=0.2, guard=False))
        t_tick = time.monotonic() + 0.05
        while time.monotonic() < t_tick:
            s = bus.world()
            if s and tag in s.get("tags", {}):
                last_seen = time.monotonic()
            time.sleep(0.01)
        if time.monotonic() - last_seen > 0.4:
            lost = True
            break
    for _ in range(3):
        bus.send(name, dict(ZERO, type="vel"))
        time.sleep(0.05)
    if lost:
        log(f"tag {tag} lost during the move - robot stopped after {time.monotonic() - t0:.2f} s")
    time.sleep(1.0)                                   # coast, and let the tracker catch up
    p1 = tag_pose(bus, tag)
    if p1 is None:
        log(f"tag {tag} is not in view after the move (hidden at this heading, or out of the picture)")
        return None
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    d, turn = math.hypot(dx, dy), math.degrees(wrap(p1[2] - p0[2]))
    rel = math.degrees(wrap(math.atan2(dy, dx) - p0[2])) if d > 0.005 else 0.0
    log(f"tag {tag}: ({p0[0]:.3f}, {p0[1]:.3f}, {math.degrees(p0[2]):.0f} deg) -> "
        f"({p1[0]:.3f}, {p1[1]:.3f}, {math.degrees(p1[2]):.0f} deg)")
    log(f"  moved {d * 100:.1f} cm" + (f" in direction {rel:+.0f} deg relative to where the tag pointed" if d > 0.005 else "")
        + f", turned {turn:+.1f} deg")
    return d, rel, turn


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--robot", required=True)
    ap.add_argument("--vx", type=float, default=0.0)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--w", type=float, default=0.0)
    ap.add_argument("--secs", type=float, default=0.3)
    a = ap.parse_args()
    if a.secs > 1.0:
        raise SystemExit("jog is for short moves: --secs at most 1.0")
    cfg = load_config(a.config)
    tag = str(cfg["robots"][a.robot]["tag"])
    jog(UdpBus(cfg), a.robot, tag, {"vx": a.vx, "vy": a.vy, "w": a.w}, a.secs)


if __name__ == "__main__":
    main()
