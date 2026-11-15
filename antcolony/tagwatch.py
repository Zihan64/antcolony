"""Live view of the robots' tags as the tracker reports them. Listen-only: it never sends anything
to the robots, so it is safe to run while a car is being tested.

    python3 -m antcolony.tagwatch --overlay profiles/edge2.json   # the robots of that test
    python3 -m antcolony.tagwatch                 # all robots in config.json
    python3 -m antcolony.tagwatch --ids 33 36     # just these tags

Prints one line per tag, refreshed twice a second:
  seen%     share of tracker snapshots in the last 2 s that contained the tag
  x y hdg   position (m) and heading (deg) in the arena frame
  method    plane = located on a plane at a known height (precise); pose = from the tag's
            apparent size (several cm off)
  z         the height the tracker uses for the tag
  pose_z    the height the tag's own 3D pose suggests (noisy). If z and pose_z differ by more
            than ~2 cm, z is probably wrong: measure the tag face and set objects.<id>.height_m
            in the tracker's arena.json
  jitter    std dev of x / y over the last 2 s (mm), standing still should be a few mm

Needs UDP 9100 to itself: stop the coordinator / antcolony.check first.
"""
import argparse
import json
import math
import socket
import statistics
import sys
import time
from collections import deque

from .common import PORT_WORLD, load_config

WINDOW = 2.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--overlay")
    ap.add_argument("--ids", nargs="+", help="tag ids to show (default: the robots' tags in the config)")
    a = ap.parse_args()

    if a.ids:
        names = {str(i): f"tag {i}" for i in a.ids}
    else:
        cfg = load_config(a.config, a.overlay)
        names = {str(rc["tag"]): f"{n} (tag {rc['tag']})" for n, rc in cfg["robots"].items()}
        port = cfg.get("world_port", PORT_WORLD)
    port = PORT_WORLD if a.ids else port

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.bind(("0.0.0.0", port))
    except OSError as e:
        sys.exit(f"cannot listen on UDP {port} ({e}): the coordinator or antcolony.check is using it")
    s.settimeout(0.1)

    snaps = deque()                  # (time, msg)
    src, next_print = None, 0.0
    print(f"listening on UDP {port} (nothing is sent to the robots) - Ctrl+C to stop\n")
    while True:
        try:
            data, addr = s.recvfrom(65535)
            snaps.append((time.time(), json.loads(data)))
            src = addr[0]
        except socket.timeout:
            pass
        except ValueError:
            continue
        now = time.time()
        while snaps and now - snaps[0][0] > WINDOW:
            snaps.popleft()
        if now < next_print:
            continue
        next_print = now + 0.5

        out = [time.strftime("%H:%M:%S") + (f"  tracker {src}, {len(snaps) / WINDOW:.0f}/s" if snaps
                                            else "  NO TRACKER DATA")]
        for tid, label in names.items():
            hits = [m for _, m in snaps if tid in m.get("tags", {})]
            if not hits:
                out.append(f"  {label:<14} not seen")
                continue
            x, y, th = hits[-1]["tags"][tid]
            info = (hits[-1].get("tag_info") or {}).get(tid, {})
            line = (f"  {label:<14} seen {100 * len(hits) // max(1, len(snaps)):3d}%  "
                    f"x {x:5.2f}  y {y:5.2f}  hdg {math.degrees(th):5.0f}  "
                    f"{info.get('method', '?'):<5}  z {info.get('z', float('nan')):.3f}")
            pz = info.get("pose_z")
            if pz is not None:
                line += f"  pose_z {pz:.3f}"
            if len(hits) >= 3:
                xs = [m["tags"][tid][0] for m in hits]
                ys = [m["tags"][tid][1] for m in hits]
                line += f"  jitter {1000 * statistics.pstdev(xs):3.0f}/{1000 * statistics.pstdev(ys):3.0f} mm"
            out.append(line)
        print("\n".join(out) + "\n", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
