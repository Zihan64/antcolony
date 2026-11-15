"""Two arm cars clear the floor together, one object each into the tray. No tracker.

    python -m antcolony.duo                         # cars A and B, tray = "white tray"
    python -m antcolony.duo --first A --second B --noun block --obj-height 3

Rules (user, 2026-11-14):
  * each car takes the object NEAREST to itself (found by its colour, as in antcolony.solo `find`)
  * the first car (A) chooses first and tells the other which one it took; the second car leaves
    that one alone. The cars have no common map, so "which one" is the object's COLOUR: the
    objects must differ in colour (two red blocks cannot be told apart this way).
  * one car at a time at the tray: whoever holds its object first goes first; it puts the object
    in, backs away, and only then may the other car drive up. (Two cars docking at once would
    meet in front of the tray.)

Each car: find -> grab (a miss: back up, again, --tries) -> [tray turn] seek tray -> place -> back away.
If one car fails it stops where it is and gives the tray turn back; the other carries on.
Both agents (agent.py) must be running, and the VLM server. Runs on the computer both cars answer
to (UDP 9201 must be free: no coordinator / check / solo for these cars at the same time).
"""
import argparse
import json
import re
import socket
import sys
import threading
import time

from .common import load_config
from .solo import Link, PORT_TELEM, show


class Hub:
    """One UDP socket for the telemetry of all cars (two sockets on one port would steal from each other)."""

    def __init__(self):
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.bind(("0.0.0.0", PORT_TELEM))
        self.rx.settimeout(0.2)
        self.latest, self.alive = {}, True
        threading.Thread(target=self.loop, daemon=True).start()

    def loop(self):
        while self.alive:
            try:
                data, src = self.rx.recvfrom(65535)
                m = json.loads(data.decode())
            except (socket.timeout, ValueError, OSError):
                continue
            if isinstance(m, dict) and m.get("name"):
                self.latest[(m["name"], src[0])] = m


class HubLink(Link):
    """solo.Link, reading its telemetry from the shared Hub."""

    def __init__(self, name, addr, hub):
        self.name, self.addr, self.ip, self.hub = name, addr, socket.gethostbyname(addr[0]), hub
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.telem = None

    def pump(self, secs=0.1):
        time.sleep(secs)
        self.telem = self.hub.latest.get((self.name, self.ip), self.telem)
        return self.telem

    def status(self, secs=1.5):
        self.hub.latest.pop((self.name, self.ip), None)
        return super().status(secs)


class Team:
    """What the cars tell each other: which colours are taken, and whose turn it is at the tray."""

    def __init__(self):
        self.claims, self.chosen, self.tray = {}, {}, threading.Lock()

    def claim(self, name, colour):
        self.claims[name] = colour                       # None = that car found nothing
        self.chosen.setdefault(name, threading.Event()).set()

    def wait_for(self, name, timeout):
        self.chosen.setdefault(name, threading.Event()).wait(timeout)
        return self.claims.get(name)


def run_car(name, link, team, a, say, after=None):
    """One car's whole job. `after` = the car that chooses before this one. -> True if its object is in the tray."""
    taken = None
    try:
        if after:
            say(f"waiting for {after} to choose ...")
            taken = team.wait_for(after, a.choose_s)
            say(f"{after} took the {taken} one - leaving it alone" if taken else f"{after} has not claimed anything - free choice")
        target = "object" + (f" not {taken}" if taken else "")
        say("find: the nearest object" + (f" that is not {taken}" if taken else "") + " ...")
        ok, detail, secs = link.skill("seek", {"target": target, "detector": "colour", "bottom_stop": a.find_bottom, "stop_cm": 12}, 390)
        m = re.search(r"colour=(\w+)", detail)
        colour = m[1] if ok and m and m[1] not in ("None", "coloured") else None
        say(f"  {'ok' if ok else 'FAILED'} - {detail}  ({secs:.0f} s)")
        team.claim(name, colour if ok else None)
        if not ok:
            say("GIVING UP: no object found - I stop here")
            return False
        label = f"{colour} {a.noun}" if colour else "any"
        say(f"my object: the {label}")
        held = False
        for i in range(1, a.tries + 1):
            say(f"grab, try {i} of {a.tries} ...")
            gargs = {"label": label}
            if a.obj_height:
                gargs["obj_height"] = a.obj_height
            ok, detail, secs = link.skill("grab", gargs, 300)
            say(f"  {'ok' if ok else 'missed'} - {detail}  ({secs:.0f} s)")
            if ok:
                held = True
                break
            if i < a.tries:
                # After a miss the car stands over the object. Backing up alone is not enough: 18 cm back
                # puts a small block ~30 cm ahead, beyond the grab's near view, and its far view often
                # misses a pale block (2026-11-14, car A: three tries "not seen" in a row). So back up,
                # then line up on it again by its colour, as before the first try.
                link.back_up(a.back_cm)
                if colour:
                    ok2, d2, s2 = link.skill("seek", {"target": f"{colour} object", "detector": "colour",
                                                      "bottom_stop": a.find_bottom, "stop_cm": 12}, 120)
                    say(f"  lined up on it again: {'ok' if ok2 else 'FAILED'} - {d2}  ({s2:.0f} s)")
        if not held:
            say(f"GIVING UP: {a.tries} grabs, none held - I stop here; the {label} stays on the floor")
            return False
        if team.tray.locked():
            say("holding it - waiting for my turn at the tray")
        with team.tray:
            say(f"my turn at the tray - seek: '{a.target}' ...")
            ok, detail, secs = link.skill("seek", dict(json.loads(a.seek), target=a.target), 390)
            say(f"  {'ok' if ok else 'FAILED'} - {detail}  ({secs:.0f} s)")
            if not ok:
                return False
            ok, detail, secs = link.skill("dock", {"drop": True, "z_cm": a.place_z}, 150)
            say(f"  place: {'ok' if ok else 'FAILED'} - {detail}  ({secs:.0f} s)")
            if not ok:
                return False
            say(f"backing away from the tray, {a.away_cm:.0f} cm")
            link.back_up(a.away_cm)
        return True
    finally:
        team.claim(name, team.claims.get(name))          # never leave the other car waiting


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--first", default="A", help="the car that chooses first")
    ap.add_argument("--second", default="B")
    ap.add_argument("--noun", default="block", help='what the objects are called for the grab ("red block")')
    ap.add_argument("--obj-height", type=float, default=3.0, help="cm; 0 = let the grab guess")
    ap.add_argument("--tries", type=int, default=4)
    ap.add_argument("--back-cm", type=float, default=18)
    ap.add_argument("--find-bottom", type=float, default=0.40)
    ap.add_argument("--target", default="white tray")
    ap.add_argument("--seek", default='{"img_tokens": 576, "view": "far"}')
    ap.add_argument("--place-z", type=float, default=10)
    ap.add_argument("--away-cm", type=float, default=35, help="back away from the tray this far after placing")
    ap.add_argument("--choose-s", type=float, default=120, help="the second car waits at most this long for the first to choose")
    a = ap.parse_args()
    cfg = load_config(a.config)
    try:
        hub = Hub()
    except OSError as e:
        sys.exit(f"cannot listen on UDP {PORT_TELEM} ({e}) - is the coordinator, check or solo still running?")
    t0, team, lock = time.monotonic(), Team(), threading.Lock()

    def sayer(name):
        def say(msg):
            with lock:
                print(f"{time.monotonic() - t0:5.0f}s [{name}] {msg}", flush=True)
        return say

    links = {n: HubLink(n, tuple(cfg["robots"][n]["addr"]), hub) for n in (a.first, a.second)}
    for n, l in links.items():
        t = l.status() or l.status(3.0) or l.status(3.0)      # the first datagram to a quiet car is often lost (ARP)
        sayer(n)(show(t))
        if t is None:
            sys.exit(1)
    result = {}

    def job(n, after):
        try:
            result[n] = run_car(n, links[n], team, a, sayer(n), after)
        except Exception as e:                            # one car's trouble must not stop the other
            sayer(n)(f"stopped: {e}")
            result[n] = False

    ths = [threading.Thread(target=job, args=(a.first, None), daemon=True),
           threading.Thread(target=job, args=(a.second, a.first), daemon=True)]
    try:
        for th in ths:
            th.start()
        while any(th.is_alive() for th in ths):
            time.sleep(0.2)
    except KeyboardInterrupt:
        for n, l in links.items():                        # stop both cars at once
            l.send({"type": "skill", "id": int(time.time() * 10) % 2_000_000_000 + 7, "name": "stop", "args": {}})
        print("\nstopped both cars")
        sys.exit(130)
    done = sum(1 for v in result.values() if v)
    print(f"{time.monotonic() - t0:5.0f}s === {done} of 2 objects in the tray: " +
          ", ".join(f"{n} {'ok' if result.get(n) else 'FAILED'} ({team.claims.get(n)})" for n in (a.first, a.second)))
    sys.exit(0 if done == 2 else 2)


if __name__ == "__main__":
    main()
