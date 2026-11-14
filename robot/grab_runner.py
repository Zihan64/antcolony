#!/usr/bin/env python3
"""Runs vlm_grab.py with the object finder chosen in detector.json (VLM or a local YOLO), without
changing a line of vlm_grab.py. Same arguments, same exit codes:

    python3 grab_runner.py approach any --min-speed 30
    python3 grab_runner.py find any

Lives next to vlm_grab.py (with detector.py). hw_masterpi.py starts this instead of vlm_grab.py
when it is there. Without detector.json, or with {"backend": "vlm"}, it is exactly vlm_grab.py.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    import detector
    detector.install()                       # masterpi_lite.vlm.detect_all -> the configured backend
    import vlm_grab                          # its `from masterpi_lite import vlm` sees the swapped function
    sys.argv = [os.path.join(HERE, "vlm_grab.py")] + sys.argv[1:]
    vlm_grab.main()


if __name__ == "__main__":
    main()
