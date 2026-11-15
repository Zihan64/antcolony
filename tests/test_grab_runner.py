"""robot/grab_runner.py: per-car defaults and colour tracking (no car, no VLM).

    python tests/test_grab_runner.py
"""
import json
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "robot"))
import grab_runner as g  # noqa: E402

bad = 0


def check(name, cond):
    global bad
    bad += not cond
    print(f"{'ok  ' if cond else 'FAIL'} {name}")


d = tempfile.mkdtemp()
p = os.path.join(d, "grab_defaults.json")
json.dump({"--y-offset": 1.8, "_note": "car A", "--no-align": True, "--release": False}, open(p, "w"))
check("defaults added", g.car_defaults(["grab", "x"], p) == ["grab", "x", "--y-offset", "1.8", "--no-align"])
check("command line wins", g.car_defaults(["grab", "x", "--y-offset=0.5"], p) == ["grab", "x", "--y-offset=0.5", "--no-align"])
check("no file: unchanged", g.car_defaults(["grab"], os.path.join(d, "none.json")) == ["grab"])

A = (100, 100, 200, 200)


def fake():
    calls = []
    vlm = types.SimpleNamespace(detect_all=lambda img, obj, verbose=True:
                                (calls.append(obj), [{"box": img["vlm"], "label": obj}] if img["vlm"] else [])[1])
    g.colour_track(vlm, blob=lambda img, name: img["blob"], log=lambda m: None)
    return vlm, calls


vlm, calls = fake()
vlm.detect_all({"vlm": A, "blob": (105, 98, 205, 204)}, "red block")
r = vlm.detect_all({"vlm": None, "blob": (110, 110, 212, 210)}, "red block")
check("agreed once: next photo from the colour, no VLM call", len(calls) == 1 and r[0]["box"][0] == 110)
r = vlm.detect_all({"vlm": A, "blob": (100, 100, 135, 200)}, "red block")
check("colour box shrinks to part of the block (car B, 2026-11-14): asks the VLM, keeps its box",
      len(calls) == 2 and r[0]["box"] == A)
vlm.detect_all({"vlm": A, "blob": None}, "red block")
check("colour lost: VLM", len(calls) == 3)
vlm.detect_all({"vlm": A, "blob": A}, "blue block")
check("another name: VLM", len(calls) == 4)
vlm, calls = fake()
vlm.detect_all({"vlm": A, "blob": (100, 100, 200, 160)}, "red block")
vlm.detect_all({"vlm": A, "blob": (100, 100, 200, 160)}, "red block")
check("never agreed in size: always the VLM", len(calls) == 2)
print("all ok" if not bad else f"{bad} failed")
sys.exit(1 if bad else 0)
