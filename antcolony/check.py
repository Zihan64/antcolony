"""Pre-flight check: is everything the coordinator needs actually there? Nothing moves.

    python -m antcolony.check --overlay profiles/edge2.json --vlm http://192.168.8.10:8090

Run it on the computer that will run the coordinator, with the tracker and the robots' agent.py
running and the coordinator NOT running (they would share the UDP ports). It listens for a few
seconds, sends each robot a "stop" (zero velocity) so that its agent answers with telemetry,
and prints what it found:

  tracker   snapshots per second, the arena corner tags, the arena they span
  robots    each robot's tag (seen? where?), drive calibration file, agent telemetry (battery, sonar)
  objects   what the tracker reports on the floor: size, grasp/push, target or "already at the edge"
  VLM       the server the arm cars ask (--vlm URL): alive? how many parallel slots?

Exit code 0 = no PROBLEM lines (WARNING lines are for you to judge).
"""
import argparse
import json
import math
import os
import time
import urllib.request

from .common import ZERO, load_config
from .coordinator import UdpBus
from .triage import Triage
from .world import World


def listen(bus, cfg, secs):
    """Collect snapshots and telemetry for `secs`; poke the robots so their agents report."""
    snaps, t0, nxt = [], time.monotonic(), 0.0
    while time.monotonic() - t0 < secs:
        s = bus.world()
        if s is not None:
            snaps.append((time.monotonic() - t0, s))
        if time.monotonic() - t0 >= nxt:
            nxt += 0.5
            for name in cfg["robots"]:
                bus.send(name, dict(ZERO, type="vel", dur=0.1))
        time.sleep(0.01)
    return snaps


def vlm_status(url):
    url = url.rstrip("/")
    for path in ("/health", "/api/version"):                 # llama-server, Ollama
        try:
            with urllib.request.urlopen(url + path, timeout=3) as r:
                if r.status == 200:
                    break
        except OSError:
            continue
    else:
        return None
    try:
        with urllib.request.urlopen(url + "/props", timeout=3) as r:
            return {"slots": json.loads(r.read()).get("total_slots")}
    except (OSError, ValueError):
        return {"slots": None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--overlay")
    ap.add_argument("--secs", type=float, default=6.0)
    ap.add_argument("--vlm", help="URL of the VLM server the arm cars use, e.g. http://192.168.8.10:8090")
    ap.add_argument("--arena-from-tags", action="store_true", default=None,
                    help="judge positions against the tracker's corner tags (default when the tracker sends them)")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overlay)
    problems, warnings = [], []
    P, W = problems.append, warnings.append

    print(f"\nlistening for {a.secs:.0f} s on UDP {cfg.get('world_port', 9100)} (tracker) and 9201 (robots) ...")
    try:
        bus = UdpBus(cfg)
    except OSError as e:
        raise SystemExit(f"cannot open the UDP ports ({e}) - is the coordinator still running?")
    snaps = listen(bus, cfg, a.secs)

    # ------------------------------------------------------------------ tracker
    print("\n== tracker")
    last = snaps[-1][1] if snaps else {}
    if not snaps:
        P("no data from the tracker. Start it with  --antcolony <this computer's IP>  "
          "(cam_control/apriltag_tracker_gui.py), check the IP and this computer's firewall (UDP 9100)")
        print("  nothing received")
    elif last.get("floor_valid") is False:
        print(f"  the tracker is connected ({len(snaps)} heartbeats), but it has NO FLOOR yet "
              f"(it sees {last.get('n_tags', 0)} tag(s) in the picture)")
        P("the tracker has no floor: set up the arena in its GUI (the four corner tags must be visible and "
          "locked); until then it cannot place the robots or the objects")
    else:
        rate = len(snaps) / a.secs
        print(f"  {len(snaps)} snapshots, {rate:.0f} per second")
        if rate < 5:
            W(f"tracker sends only {rate:.1f} snapshots/s - the robots will drive in jerks (10+ is good)")
        floor = last.get("floor") or {}
        ids = [str(i) for i in last.get("corners") or []]
        pts = [floor[i] for i in ids if i in floor]
        print(f"  floor tags: {sorted(floor, key=int) or 'none'}   arena corners: {ids or 'not sent'}")
        if len(pts) >= 3 and a.arena_from_tags is not False:
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            cfg["arena"].update(xmin=min(xs), xmax=max(xs), ymin=min(ys), ymax=max(ys))
            print(f"  arena from the corner tags: x {min(xs):.2f}..{max(xs):.2f}, y {min(ys):.2f}..{max(ys):.2f} m "
                  f"({max(xs) - min(xs):.2f} x {max(ys) - min(ys):.2f} m)  -> run the coordinator with --arena-from-tags")
        else:
            ar = cfg["arena"]
            print(f"  arena from the config: x {ar['xmin']}..{ar['xmax']}, y {ar['ymin']}..{ar['ymax']} m")
            if snaps and last.get("floor_valid") is not False:
                W("the tracker sent no arena corner tags: the arena in config.json must match the tracker's frame")
    ar = cfg["arena"]
    if min(ar["xmax"] - ar["xmin"], ar["ymax"] - ar["ymin"]) < 1.0:
        W("arena narrower than 1 m: two cars will hardly pass each other")

    # ------------------------------------------------------------------- robots
    print("\n== robots")
    world = World(cfg, Triage(cfg).classify)
    for _, s in snaps:
        world.update(s, time.monotonic())
    seen_tags = sorted({t for _, s in snaps for t in s.get("tags", {})}, key=int)
    cdir = os.path.join(os.path.dirname(os.path.abspath(a.config)), "calib")
    for name, rc in cfg["robots"].items():
        r, tag = world.robots[name], str(rc["tag"])
        n = sum(1 for _, s in snaps if tag in s.get("tags", {}))
        line = f"  {name} ({rc['role']}, tag {tag}, {rc['addr'][0]}): "
        if not n:
            line += "TAG NOT SEEN"
            if snaps and last.get("floor_valid") is not False:
                P(f"robot {name}: tag {tag} not seen. Tags the tracker does see: {seen_tags or 'none'} "
                  f"- set robots.{name}.tag in config.json to the tag on that car")
        else:
            line += f"tag seen in {100 * n // max(1, len(snaps))}% of snapshots, centre ({r.x:.2f}, {r.y:.2f}), " \
                    f"heading {math.degrees(r.th):.0f} deg"
            if n < 0.8 * len(snaps):
                W(f"robot {name}: tag seen in only {100 * n // len(snaps)}% of snapshots (glare? arm over the tag?)")
            if not (ar["xmin"] <= r.x <= ar["xmax"] and ar["ymin"] <= r.y <= ar["ymax"]):
                W(f"robot {name} stands outside the arena")
        cal = os.path.isfile(os.path.join(cdir, name + ".json"))
        line += " | drive calibration: " + ("ok" if cal else "MISSING")
        if not cal:
            P(f"robot {name}: no calib/{name}.json - run  python -m antcolony.calibrate --robot {name}"
              "  (robot alone in the middle, 50 cm free)")
        t = bus.telem(name)
        if t is None:
            line += " | agent: NO ANSWER"
            P(f"robot {name}: no telemetry from {rc['addr'][0]} - is agent.py running on it, is the IP right "
              f"(network profile), does this computer's firewall let UDP 9201 in?")
        else:
            mv, so = t.get("batt_mv"), t.get("sonar_cm")
            line += f" | agent ok, battery {mv} mV, sonar {so} cm, skill {t.get('skill', {}).get('state')}"
            if mv and rc["role"] == "arm" and mv < 7000:
                W(f"robot {name}: battery {mv} mV is low (below 7.0 V the arm and motors misbehave)")
        print(line)

    # ------------------------------------------------------------------ objects
    print("\n== objects on the floor (as the coordinator will see them)")
    zone = world.dc.get("edge_zone", 0.20)
    objs = [o for o in world.objects if o.state != "new"]
    if world.edge:
        print(f"  mode: put objects down at the arena edge; anything within {zone * 100:.0f} cm of the boundary is 'already there'")
    for o in sorted(objs, key=lambda o: o.id):
        print(f"  #{o.id}: ({o.x:.2f}, {o.y:.2f})  {o.w * 100:.0f} x {o.l * 100:.0f} cm  -> "
              + ("TARGET (grasp)" if o.cls == "grasp" else "too big for the gripper: nobody will touch it"))
        if o.cls != "grasp":
            W(f"object #{o.id} measures {o.w * 100:.0f} x {o.l * 100:.0f} cm from above and is classed 'push' "
              f"(limits: triage.grasp_max_short / grasp_max_long)")
        for r in world.robots.values():
            if r.x is not None and math.dist(o.xy, r.xy) < 0.40:
                W(f"object #{o.id} is only {math.dist(o.xy, r.xy) * 100:.0f} cm from robot {r.name}'s centre: a real "
                  f"object, or a part of the robot seen as one? (if so raise robots.{r.name}.mask_r above 0.22)")
    for p in world.parked:
        print(f"  at the edge already: ({p[0]:.2f}, {p[1]:.2f})  (not a target)")
    raw = len(last.get("objects", []))
    if snaps and not objs and not world.parked and last.get("floor_valid") is not False:
        (P if not raw else W)("the tracker reports no objects in the arena" + (
            f" ({raw} outside it or under a robot)" if raw else
            ": is object detection switched on in the tracker? As a fallback give the coordinator the "
            "block positions by hand:  --objects 0.50,0.40 1.00,0.70"))
    quota = world.dc.get("per_robot", 0)
    arms = [n for n, rc in cfg["robots"].items() if rc["role"] == "arm"]
    targets = [o for o in objs if o.cls == "grasp"]
    if quota and len(targets) < quota * len(arms):
        W(f"{len(targets)} target object(s) for {len(arms)} arm cars that should each deliver {quota}")
    close = [(p, q) for i, p in enumerate(targets) for q in targets[i + 1:]
             if math.dist(p.xy, q.xy) < cfg["alloc"]["spacing"]]
    if close:
        W(f"{len(close)} pair(s) of objects closer than {cfg['alloc']['spacing'] * 100:.0f} cm: the cars take those one after the other")

    # ---------------------------------------------------------------------- VLM
    print("\n== VLM server")
    if not a.vlm:
        print("  not checked (give --vlm http://<VLM computer>:8090)")
    else:
        st = vlm_status(a.vlm)
        if st is None:
            P(f"VLM server {a.vlm} does not answer - llama-server running with --host 0.0.0.0? firewall?")
            print("  NO ANSWER")
        else:
            print(f"  {a.vlm} is up, parallel slots: {st['slots']}")
            if st["slots"] and st["slots"] > 1 and len(arms) > 1:
                W(f"VLM server has {st['slots']} slots: with two cars start it with -np 1 "
                  f"(queued requests finish at 2.5 s / 4.1 s instead of 5.4 s / 6.4 s)")
        print("  each car must point at it: masterpi_lite/masterpi_lite/vlm_server.json "
              f'{{"url": "{a.vlm or "http://<VLM computer>:8090"}", "img_tokens": 256}}  (not checked from here)')

    print("\n== result")
    for w in warnings:
        print("  WARNING  " + w)
    for p in problems:
        print("  PROBLEM  " + p)
    print(f"  {len(problems)} problem(s), {len(warnings)} warning(s)"
          + ("" if problems else " - ready to start the coordinator"))
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
