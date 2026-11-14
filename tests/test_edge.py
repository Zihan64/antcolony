"""Two arm cars, no bin: each brings ONE block to the arena edge; a missed grab is tried again,
at most 5 misses in a row, then that car gives up (profiles/edge2.json).

    python tests/test_edge.py              # the three checks below, 6 layouts each
    python tests/test_edge.py 3 -v         # layout 3 only, with the event log

Checks, in an arena the size of the real one (1.55 x 1.19 m, paper tags at the corners):
  1. normal grasping (75 % hold): both cars deliver exactly one block each, into the edge band
     (a third block may end up there too if a car bumps one it could not see - the camera looks
     at a slant and each car hides a strip of floor behind it; `bump_s` shows how long);
  2. every grab misses: each car tries exactly 5 times, then gives up; nothing is delivered;
  3. poor grasping (35 % hold): a car that succeeds needed at most 5 tries, a car that gave up
     tried exactly 5 times.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from antcolony.calibrate import calibrate          # noqa: E402
from antcolony.common import load_config           # noqa: E402
from antcolony.coordinator import Coordinator      # noqa: E402
from antcolony.sim import SimBus, SimWorld         # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ARENA = {"xmin": 0.0, "xmax": 1.55, "ymin": 0.0, "ymax": 1.19}


def run(seed, p_grasp, n_objects=4, verbose=False, max_t=900.0):
    cfg = load_config(os.path.join(ROOT, "config.json"), os.path.join(ROOT, "profiles", "edge2.json"))
    cfg["arena"].update(ARENA)
    sim = SimWorld(cfg, seed, n_objects, quirks=True, p_grasp=p_grasp, big_frac=0.0, floor_tags=True)
    bus = SimBus(sim)
    for name, r in sim.robots.items():               # one robot at a time in the middle, as by hand
        sim.solo, (r.x, r.y) = name, ((ARENA["xmin"] + ARENA["xmax"]) / 2, (ARENA["ymin"] + ARENA["ymax"]) / 2)
        sim.advance(0.5)
        cfg["robots"][name].update(calibrate(bus, name, cfg, wait=sim.advance, log=lambda *_: None))
        r.x, r.y = r.start
    sim.solo = None
    sim.advance(1.0)
    co = Coordinator(cfg, bus, quiet=not verbose)
    t_start, n = sim.t, 0
    while sim.t - t_start < max_t and co.t_done is None:
        sim.step()
        n += 1
        if n % 3 == 0:
            co.tick(sim.t)
    zone = cfg["deliver"]["edge_zone"]
    lying = [o for o in sim.objs if o.state == "parked"]
    return {"seed": seed, "done": co.t_done is not None, "time": round((co.t_done or sim.t) - (co.t0 or t_start)),
            "delivered": {b.name: b.done_n for b in co.brains}, "gave_up": {b.name: b.retired for b in co.brains},
            "grabs": dict(sim.grabs), "at_edge": len(lying), "in_field": sim.left(),
            "edge_ok": all(sim._edge_dist(o.xy) < zone for o in lying),
            "collisions": sim.collisions, "bump_s": round(sim.bump_s, 1)}


def check(name, rows, rule):
    bad = [r for r in rows if not r["done"] or not rule(r)]
    for r in rows:
        print("   ", "ok  " if r not in bad else "FAIL", r)
    print(f"{name}: {len(rows) - len(bad)}/{len(rows)} layouts ok, mean {sum(r['time'] for r in rows) / len(rows):.0f} s, "
          f"robot-robot contacts {sum(r['collisions'] for r in rows)}\n")
    return not bad


if __name__ == "__main__":
    seeds = [int(a) for a in sys.argv[1:] if a.isdigit()] or list(range(1, 7))
    v = "-v" in sys.argv
    ok = check("1. normal grasping: one block each", [run(s, 0.75, verbose=v) for s in seeds],
               lambda r: all(n == 1 for n in r["delivered"].values()) and r["at_edge"] >= 2 and r["edge_ok"]
               and all(g <= 5 for g in r["grabs"].values()))
    ok &= check("2. every grab misses: 5 tries, then give up", [run(s, 0.0, verbose=v) for s in seeds],
                lambda r: all(r["gave_up"].values()) and all(g == 5 for g in r["grabs"].values())
                and not any(r["delivered"].values()))
    ok &= check("3. poor grasping: never more than 5 tries", [run(s, 0.35, verbose=v) for s in seeds],
                lambda r: all((r["delivered"][n] == 1 and r["grabs"][n] <= 5) or
                              (r["gave_up"][n] and r["grabs"][n] == 5 and r["delivered"][n] == 0)
                              for n in r["delivered"]) and r["edge_ok"])
    sys.exit(0 if ok else 1)
