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
"_colour_track": true in that file switches on colour_track() below (off by default: see main()).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _defaults(path=None):
    """grab_defaults.json as a dict ({} if there is none)."""
    try:
        with open(path or os.path.join(HERE, "grab_defaults.json")) as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        print(f"  grab_defaults.json ignored ({e})")
        return {}


def car_defaults(argv, path=None):
    """argv plus this car's options from grab_defaults.json that argv does not set itself."""
    defaults = _defaults(path)
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


def _iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    return w * h / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h)


def colour_track(vlm, blob=None, log=print, size_tol=0.3):
    """Fewer waits for the VLM while the car creeps up to a coloured object. vlm_grab asks the VLM
    again after every 1-2 cm nudge (3-5 s each, the car standing still: 2026-11-14, five questions
    in one grab). Here the first answer is checked against the object's colour blob in the same
    photo; if both frame the same thing (overlap >= 0.6), the following photos are answered from the
    colour - but only while the colour box keeps the SIZE of the VLM's box (within size_tol): the
    grab opens the gripper to the box width, and a pale block's colour box can be just a part of it
    (2026-11-14, car B: a light-blue block read 1 cm wide, the gripper opened 2.6 cm and came down
    on top of it). A colour box of another size, no matching blob, or the blob gone: the VLM is
    asked as before. No colour word in the name: always the VLM.
    `blob(img, name)` -> box or None (default: seek.colour_blob)."""
    real, state = vlm.detect_all, {"name": None, "size": None}
    if blob is None:
        try:
            import cv2
            import numpy as np
            import seek
        except Exception as e:               # no seek.py on this car: plain VLM
            log(f"  (colour tracking off: {e})")
            return
        if not hasattr(seek, "colour_blob") or not hasattr(seek, "HUES"):
            return
        hues = seek.HUES
        blob = lambda img, name: seek.colour_blob(cv2, np, img, name) if any(k in name.lower() for k in hues) else None

    def size(b):
        return max(1.0, b[2] - b[0]), max(1.0, b[3] - b[1])

    def same_size(b, s):
        (w, h), (W, H) = size(b), s
        return abs(w / W - 1) <= size_tol and abs(h / H - 1) <= size_tol

    def detect_all(img, obj, verbose=True):
        b = blob(img, obj) if isinstance(obj, str) else None
        if b is not None and state["name"] == obj:
            if same_size(b, state["size"]):
                if verbose:
                    log(f"  colour: {[int(v) for v in b]}  (same thing the VLM named - not asked again)")
                return [{"box": tuple(float(v) for v in b), "label": obj, "raw": "colour"}]
            if verbose:
                log(f"  colour box {[int(v) for v in b]} is not the size the VLM gave - asking the VLM")
        dets = real(img, obj, verbose)
        match = [d for d in dets if b is not None and _iou(b, d["box"]) >= 0.6 and same_size(b, size(d["box"]))]
        state["name"], state["size"] = (obj, size(match[0]["box"])) if match else (None, None)
        return dets

    vlm.detect_all = detect_all


def main():
    import detector
    detector.install()                       # masterpi_lite.vlm.detect_all -> the configured backend
    from masterpi_lite import vlm
    # OFF unless grab_defaults.json says "_colour_track": true. On the real cars (2026-11-14, two-car
    # run) it broke three grabs in a row: vlm_grab's approach changes the camera view (far -> near),
    # and in the near view the colour finder answered with another blue patch (the car's own blue
    # light on the carpet). vlm_grab rejected it and gave up the approach. A box size or position
    # from one view says nothing about the next view, so the check in colour_track() cannot catch it.
    if _defaults().get("_colour_track", False):
        colour_track(vlm)
    import vlm_grab                          # its `from masterpi_lite import vlm` sees the swapped function
    sys.argv = [os.path.join(HERE, "vlm_grab.py")] + car_defaults(sys.argv[1:])
    vlm_grab.main()


if __name__ == "__main__":
    main()
