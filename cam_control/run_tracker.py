#!/usr/bin/env python3
"""
run_tracker.py

Starts apriltag_tracker_gui.py with the AntColony publisher attached, WITHOUT changing a line
of the GUI's code (so the GUI can keep changing without merge conflicts):

    python run_tracker.py --antcolony 172.20.1.112 [any apriltag_tracker_gui.py options]

--antcolony HOST[:PORT] (or the environment variable ANTCOLONY_WORLD) is the computer that runs
`python -m antcolony.coordinator`; the port defaults to 9100. Every other option is passed on to
the GUI unchanged.

How: once per processed frame the GUI's Worker thread calls its _fast_control(result, now)
(versions before 2026-11-14 06:00 instead emit result_ready(result)). This script wraps that
call (or connects to that signal), on the tracking thread, and hands the result to
antcolony_pub.WorldPub: the coordinator gets the robots' tags, the detected objects and the
floor tags. While the arena is not set up yet it gets a heartbeat twice a second instead.

The console says what is going on:  "AntColony: attached ..." once Start is pressed, then every
10 s how many datagrams went out and whether the floor is valid.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import antcolony_pub  # noqa: E402


def pop_target(argv):
    """Take --antcolony HOST[:PORT] out of argv. -> (target or None, the remaining argv)"""
    target = os.environ.pop("ANTCOLONY_WORLD", None)     # popped: a hook inside the GUI must stay off
    out, i = [], 0
    while i < len(argv):
        a = argv[i]
        if a == "--antcolony" and i + 1 < len(argv):
            target, i = argv[i + 1], i + 2
            continue
        if a.startswith("--antcolony="):
            target, i = a.split("=", 1)[1], i + 1
            continue
        out.append(a)
        i += 1
    return target, out


def attach(gui, pub, log=print):
    """Make every Worker the GUI creates publish its results through `pub`."""
    from PyQt6.QtCore import Qt

    # GUI since the planner thread was split off: result_ready is no longer emitted; the full
    # per-frame result goes through Worker._fast_control(r, now) on the tracking thread
    per_frame_call = hasattr(gui.Worker, "_fast_control")
    orig_init = gui.Worker.__init__

    def init(self, *a, **k):
        orig_init(self, *a, **k)
        self.world_pub = None                   # a GUI version with its own hook: leave that one off
        state = {"t": time.time(), "n": 0}

        def send(r, w=self):
            try:
                amap = getattr(w, "amap", None)
                ids = (amap.arena.get("corner_ids") if amap is not None else
                       ((getattr(w.mapper, "cfg", None) or {}).get("arena") or {}).get("corner_ids"))
                floor = w.mapper.floor
                pub.publish(r, floor, ids)
                if time.time() - state["t"] > 10.0:
                    n_obj = sum(1 for o in r.get("objects", []) if o.get("method") in ("detect", "appearance"))
                    log(f"AntColony: {pub.sent - state['n']} datagrams in the last 10 s -> {pub.addr[0]}:{pub.addr[1]}, "
                        + (f"floor ok, {len(r.get('objects', [])) - n_obj} tag(s), {n_obj} object(s)"
                           if getattr(floor, "valid", False) else "NO FLOOR yet (set up the arena)"))
                    state["t"], state["n"] = time.time(), pub.sent
            except Exception as e:              # noqa: BLE001 - never let this stop the tracker
                if repr(e) != pub.err:
                    pub.err = repr(e)
                    log(f"AntColony: {e!r}")

        self._antcolony_send = send             # keep it alive as long as the worker
        if not per_frame_call:                  # old GUI: the per-frame signal
            self.result_ready.connect(send, Qt.ConnectionType.DirectConnection)
        log(f"AntColony: attached - sending tags + objects to {pub.addr[0]}:{pub.addr[1]}")

    gui.Worker.__init__ = init

    if per_frame_call:
        orig_fast = gui.Worker._fast_control

        def fast_control(self, r, now):
            out = orig_fast(self, r, now)
            send = getattr(self, "_antcolony_send", None)
            if send is not None:
                send(r)
            return out

        gui.Worker._fast_control = fast_control


def main():
    target, rest = pop_target(sys.argv[1:])
    if not target:
        sys.exit("say where the coordinator runs:  python run_tracker.py --antcolony 172.20.1.112 [GUI options]")
    import apriltag_tracker_gui as gui
    pub = antcolony_pub.WorldPub(target)
    attach(gui, pub)
    print(f"AntColony: will send to {pub.addr[0]}:{pub.addr[1]} as soon as the tracker is started (press Start)")
    sys.argv = [os.path.join(HERE, "apriltag_tracker_gui.py")] + rest
    gui.main()


if __name__ == "__main__":
    main()
