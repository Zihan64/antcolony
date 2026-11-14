#!/usr/bin/env python3
"""Runs vlm_grab.py with the object finder chosen in detector.json (VLM or a local YOLO), without
changing a line of vlm_grab.py. Same arguments, same exit codes:

    python3 grab_runner.py approach any --min-speed 30
    python3 grab_runner.py find any

Lives next to vlm_grab.py (with detector.py). hw_masterpi.py starts this instead of vlm_grab.py
when it is there. Without detector.json, or with {"backend": "vlm"}, it is exactly vlm_grab.py.

Per-car defaults: grab_defaults.json next to this file holds options that THIS car always needs,
e.g. {"--y-offset": 1.8}. Each one is added unless the command line gives that option itself.
(vlm_grab.py's arm reach table was measured on car B; car A's arm stops about 1.8 cm short of
it, 2026-11-14 - without the offset its jaws close on air between the car and the block.)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def car_defaults(argv, path=None):
    """argv plus this car's options from grab_defaults.json that argv does not set itself."""
    path = path or os.path.join(HERE, "grab_defaults.json")
    try:
        with open(path) as f:
            defaults = json.load(f)
    except FileNotFoundError:
        return list(argv)
    except (OSError, ValueError) as e:
        print(f"  grab_defaults.json ignored ({e})")
        return list(argv)
    out, given = list(argv), {a.split("=")[0] for a in argv if a.startswith("--")}
    for opt, val in defaults.items():
        if not opt.startswith("--") or opt in given:        # "_note" and the like are comments
            continue
        if val is True:
            out.append(opt)
        elif val not in (False, None):
            out += [opt, str(val)]
    if out != list(argv):
        print("  this car's defaults (grab_defaults.json):", " ".join(out[len(argv):]))
    return out


def main():
    import detector
    detector.install()                       # masterpi_lite.vlm.detect_all -> the configured backend
    import vlm_grab                          # its `from masterpi_lite import vlm` sees the swapped function
    sys.argv = [os.path.join(HERE, "vlm_grab.py")] + car_defaults(sys.argv[1:])
    vlm_grab.main()


if __name__ == "__main__":
    main()
