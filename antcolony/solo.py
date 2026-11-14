"""One arm car on its own - no tracker, no coordinator: grab what lies in front of it, carry it
forward round anything tall in the way, put it down.

    python -m antcolony.solo --robot B                        # grab -> carry 20 cm -> drop
    python -m antcolony.solo --robot B --steps carry,drop     # it already holds something
    python -m antcolony.solo --robot B --dist-cm 30 --side right
    python -m antcolony.solo --robot B --steps status         # battery, sonar, holding - nothing moves
    python -m antcolony.solo --robot B --steps grab,seek,drop --target "white box"
                                                              # bring it to a landmark: find the white box with the
                                                              # camera + VLM, drive up to it, put the object down there
    python -m antcolony.solo --robot B --steps look --target "white box"    # one photo: where is it? (only the arm moves)

The car's agent.py must be running, and (for grab) the VLM server. Put the object 25-40 cm in
front of the car. A missed grab is tried again, at most --tries times (default 5); before each
new try the car backs up --back-cm (default 18), because after a miss it stands right over the
object and its camera no longer sees it (2026-11-14: without this, tries 2-5 looked past the
block at the obstacle behind it). With something else in view, name the object (--label "red
block") instead of the default "any". If the carry fails (blocked on both sides) the car keeps
holding the object: put it down with `--steps drop` once there is room in front of it.

What "round anything in the way" means is in robot/hw_masterpi.py, carry(): front sonar only,
things about 10 cm tall or more, the side the car slides to must be free.
"""
import argparse
import json
import socket
import sys
import time

from .common import load_config

PORT_TELEM = 9201


class Link:
    """Skills to one robot's agent, telemetry back."""

    def __init__(self, name, addr):
        self.name, self.addr, self.ip = name, addr, socket.gethostbyname(addr[0])
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.rx.bind(("0.0.0.0", PORT_TELEM))
        self.rx.settimeout(0.1)
        self.telem = None

    def send(self, msg):
        self.tx.sendto(json.dumps(msg).encode(), self.addr)

    def pump(self, secs=0.1):
        """Read telemetry for `secs`; -> the newest message from this robot (or None)."""
        t_end = time.monotonic() + secs
        while time.monotonic() < t_end:
            try:
                data, src = self.rx.recvfrom(65535)
                m = json.loads(data.decode())
            except (socket.timeout, ValueError):
                continue
            if isinstance(m, dict) and m.get("name") == self.name and src[0] == self.ip:     # not another car's
                self.telem = m
        return self.telem

    def status(self, secs=1.5):
        """A zero velocity command makes the agent send telemetry to us. -> telemetry or None."""
        self.telem, t_end = None, time.monotonic() + secs
        while time.monotonic() < t_end and (self.telem is None or self.telem.get("sonar_cm") is None):
            self.send({"type": "vel", "vx": 0, "vy": 0, "w": 0, "dur": 0.1})
            self.pump(0.2)
        return self.telem

    def back_up(self, cm, cm_per_pulse=5.7, secs=0.33):
        """Straight back in short pulses (the way the car came is free). Counted, not measured."""
        for _ in range(max(1, round(cm / cm_per_pulse))):
            self.send({"type": "vel", "vx": -0.05, "vy": 0, "w": 0, "dur": secs})
            self.pump(secs + 0.3)

    def skill(self, name, args, timeout):
        """Run one skill to its end. -> (ok, detail, seconds)."""
        sid = int(time.time() * 10) % 2_000_000_000
        msg, t0, last = {"type": "skill", "id": sid, "name": name, "args": args}, time.monotonic(), 0.0
        while time.monotonic() - t0 < timeout:
            if time.monotonic() - last > 0.5:            # the same id again is harmless; a lost datagram is not
                self.send(msg)
                last = time.monotonic()
            t = self.pump(0.1)
            s = (t or {}).get("skill") or {}
            if s.get("id") == sid and s.get("state") in ("ok", "fail"):
                return s["state"] == "ok", s.get("detail", ""), time.monotonic() - t0
        self.send({"type": "skill", "id": sid + 1, "name": "stop", "args": {}})      # cancels it on the car
        return False, f"no answer within {timeout:.0f} s", time.monotonic() - t0


def show(t):
    if t is None:
        return "no telemetry - is agent.py running on the car, and is this the right address?"
    b, s = t.get("batt_mv"), t.get("sonar_cm")
    return (f"battery {b / 1000:.2f} V" if b else "battery ?") + ", " + \
           (f"sonar {s:.0f} cm" if s is not None else "sonar ?") + f", holding: {'yes' if t.get('holding') else 'no'}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--robot", required=True)
    ap.add_argument("--addr", help="the car's IP, if not the one in config.json")
    ap.add_argument("--steps", default="grab,carry,drop", help="any of back, grab, carry, seek, look, drop, home, status - in this order")
    ap.add_argument("--label", default="any", help="what to grab (English), default: anything")
    ap.add_argument("--tries", type=int, default=5, help="grab attempts before giving up")
    ap.add_argument("--back-cm", type=float, default=18, help="back up this far before each new grab try (0 = stay)")
    ap.add_argument("--obj-height", type=float, help="cm, if the object is a block of known height (skips the shape guess)")
    ap.add_argument("--dist-cm", type=float, default=20, help="how far to carry it forward")
    ap.add_argument("--side", choices=["left", "right"], help="which way to try first round an obstacle (default left)")
    ap.add_argument("--carry", default="{}", help='more carry settings as JSON, e.g. \'{"margin_cm": 18, "cm_s": 15}\'')
    ap.add_argument("--target", default="white box", help="seek / look: what to drive up to (English)")
    ap.add_argument("--stop-cm", type=float, default=22, help="seek: stop when the front sonar reads this")
    ap.add_argument("--seek", default="{}", help="more seek settings as JSON")
    ap.add_argument("--drop-y", type=float, default=16, help="cm ahead of the arm base")
    ap.add_argument("--drop-z", type=float, default=4, help="cm above the floor")
    a = ap.parse_args()
    cfg = load_config(a.config)
    addr = (a.addr, 9200) if a.addr else tuple(cfg["robots"][a.robot]["addr"])
    link, name = Link(a.robot, addr), a.robot
    say = lambda msg: print(f"[{name}] {msg}", flush=True)

    t = link.status()
    say(show(t))
    if t is None:
        sys.exit(1)
    for step in [s.strip() for s in a.steps.split(",") if s.strip()]:
        if step == "status":
            continue
        if step == "grab":
            gargs, backed = {"label": a.label}, 0.0
            if a.obj_height:
                gargs["obj_height"] = a.obj_height
            for i in range(1, a.tries + 1):
                say(f"grab, try {i} of {a.tries} ...")
                ok, detail, secs = link.skill("grab", gargs, 150)
                say(f"  {'ok' if ok else 'missed'} - {detail}  ({secs:.0f} s)")
                if ok:
                    break
                stood = "nothing moved" in detail or "not found" in detail      # the car did not drive up to anything
                if i < a.tries and a.back_cm > 0 and (not stood or backed < 2 * a.back_cm):
                    say(f"  backing up {a.back_cm:.0f} cm to see the object again")
                    link.back_up(a.back_cm)
                    backed = backed + a.back_cm if stood else a.back_cm
            else:
                say(f"GIVES UP after {a.tries} tries")
                sys.exit(2)
            say("with the object in the gripper: " + show(link.status()))
        elif step == "carry":
            args = dict(json.loads(a.carry), dist_cm=a.dist_cm)
            if a.side:
                args["side"] = a.side
            say(f"carry {a.dist_cm:.0f} cm forward ...")
            ok, detail, secs = link.skill("carry", args, float(args.get("timeout", 60)) + 30)
            say(f"  {'ok' if ok else 'FAILED'} - {detail}  ({secs:.0f} s)")
            if not ok:
                say("still holding the object - when there is room in front: --steps drop")
                sys.exit(3)
        elif step in ("seek", "look"):
            args = dict(json.loads(a.seek), target=a.target, stop_cm=a.stop_cm, look_only=step == "look")
            say(f"{'looking for' if step == 'look' else 'seek:'} '{a.target}' ...")
            ok, detail, secs = link.skill("seek", args, float(args.get("timeout", 240)) + 30)
            say(f"  {'ok' if ok else 'FAILED'} - {detail}  ({secs:.0f} s)")
            if not ok and step == "seek":
                say("did not get there - the car keeps what it holds (--steps drop to put it down)")
                sys.exit(5)
        elif step == "drop":
            say("drop ...")
            ok, detail, secs = link.skill("drop", {"y_cm": a.drop_y, "z_cm": a.drop_z}, 20)
            say(f"  {'ok' if ok else 'FAILED'} - {detail}  ({secs:.0f} s)")
            if not ok:
                sys.exit(4)
        elif step == "back":
            say(f"backing up {a.back_cm:.0f} cm")
            link.back_up(a.back_cm)
        elif step == "home":
            ok, detail, secs = link.skill("home", {}, 15)
            say(f"home: {'ok' if ok else 'FAILED'} - {detail}")
        else:
            sys.exit(f"unknown step {step!r}")
    say("done")


if __name__ == "__main__":
    main()
