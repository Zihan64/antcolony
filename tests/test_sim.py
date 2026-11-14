"""End-to-end test in the simulator: calibrate each robot, then clear the arena.

    python tests/test_sim.py            # 8 random layouts, default mode (dozer parks things by the bin)
    python tests/test_sim.py 3 -v       # seed 3 with the event log
    python tests/test_sim.py --mouth    # optional mode: bin with a floor-level mouth, 35 % of objects too big to grip
    python tests/test_sim.py --robots AB    # what if the dozer is missing?
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from antcolony.calibrate import calibrate          # noqa: E402
from antcolony.common import load_config, wrap     # noqa: E402
from antcolony.coordinator import Coordinator      # noqa: E402
from antcolony.sim import SimBus, SimWorld         # noqa: E402

CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.json")


def run(seed, n_objects=6, verbose=False, max_t=600.0, quirks=True, robots=None, mouth=False, **simkw):
    cfg = load_config(CFG)
    if mouth:                                        # optional mode: floor-level bin mouth, dozer pushes big things in
        cfg["bin"]["floor_entry"] = True
        simkw.setdefault("big_frac", 0.35)
    simkw.setdefault("big_frac", 0.0)                # default mode: everything on the floor fits the gripper
    if robots:                                       # e.g. robots="AB": what if the dozer is missing?
        cfg["robots"] = {n: rc for n, rc in cfg["robots"].items() if n in robots}
    sim = SimWorld(cfg, seed, n_objects, quirks, **simkw)
    bus = SimBus(sim)
    cal_err = 0.0
    if quirks:
        a = cfg["arena"]
        for name, r in sim.robots.items():           # one robot at a time in the middle, as you would do by hand
            sim.solo, (r.x, r.y) = name, ((a["xmin"] + a["xmax"]) / 2, (a["ymin"] + a["ymax"]) / 2)
            sim.advance(0.5)                         # let the (delayed) camera catch up with the move
            res = calibrate(bus, name, cfg, wait=sim.advance, log=lambda *_: None)
            cfg["robots"][name].update(res)
            assert res["th_sign"] == r.q["th_sign"] and res["w_sign"] == r.q["w_sign"], (name, res, r.q)
            assert not res["strafe_ok"] or res["vy_sign"] == r.q["vy_sign"], (name, res, r.q)
            cal_err = max(cal_err, abs(wrap(res["th_off"] - r.q["th_off"])), abs(res["tag_off"][0] - r.q["tag_off"][0]) * 10)
            r.x, r.y = r.rc["home"]
        sim.solo = None
        sim.advance(1.0)
    t_start = sim.t
    co = Coordinator(cfg, bus, quiet=not verbose)
    n = 0
    while sim.t - t_start < max_t and co.t_done is None:
        sim.step()
        n += 1
        if n % 3 == 0:                               # ~16 Hz control on 10 Hz, 0.15 s-late camera data
            co.tick(sim.t)
    return {"seed": seed, "objects": n_objects, "left": sim.left(), "done": co.t_done is not None,
            "time": round((co.t_done or sim.t) - (co.t0 or t_start)), "collisions": sim.collisions,
            "bump_s": round(sim.bump_s, 1), "cal_err": round(cal_err, 3),
            "by_dozer": sum(1 for e in co.events if e.split("] ")[-1].startswith("C ") and ("staged" in e or "delivered" in e)),
            "stuck": [(round(o.x, 2), round(o.y, 2), o.cls, o.state) for o in co.world.objects]}


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    seeds = [int(a) for a in args if a.isdigit()] or list(range(1, 9))
    robots = sys.argv[sys.argv.index("--robots") + 1] if "--robots" in sys.argv else None
    rows = [run(s, verbose="-v" in sys.argv, mouth="--mouth" in sys.argv, robots=robots) for s in seeds]
    for r in rows:
        print(r)
    ok = sum(r["done"] and r["left"] == 0 for r in rows)
    print(f"\ncleared {ok}/{len(rows)} layouts, mean time {sum(r['time'] for r in rows) / len(rows):.0f} s, "
          f"robot-robot contacts {sum(r['collisions'] for r in rows)}")
    sys.exit(0 if ok == len(rows) else 1)
