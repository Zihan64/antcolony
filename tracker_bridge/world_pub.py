"""Tracker -> coordinator bridge. Add ~5 lines to the AprilTag tracker GUI:

    from world_pub import WorldPublisher, floor_rect
    pub = WorldPublisher("127.0.0.1")              # IP of the machine that runs the coordinator

    # once per processed frame, with what the table in the GUI already shows:
    pub.publish(tags={28: (x_m, y_m, heading_deg), 50: (...), ...},
                objects=[{"id": 1000, "label": "bag", "x": 2.27, "y": -0.23,
                          **floor_rect(mask_polygon_on_floor_m)}, ...],
                frame=bgr_image, boxes={1000: (x1, y1, x2, y2)})        # optional: crops for the VLM triage

Standard library + numpy/OpenCV only (both already used by the tracker).
What the coordinator needs from the tracker:
  * every robot tag and the bin tag in ONE floor frame, metres, with the tag's heading;
  * every loose object's floor position; its size (w, l) and long-axis angle if you have a mask;
  * tags each frame even when the detector is slow (run the detector in its own thread).
Do NOT send the floor-corner tags as objects, and mask out the robots before blob detection.
"""
import json
import math
import os
import socket
import time


class WorldPublisher:
    def __init__(self, host="127.0.0.1", port=9100, crop_dir="crops"):
        self.addr, self.crop_dir = (host, port), crop_dir
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._cropped = set()

    def publish(self, tags, objects, frame=None, boxes=None):
        """tags: {id: (x_m, y_m, heading_deg)}; objects: [{"id","label","x","y"[,"w","l","ang"]}]"""
        objs = []
        for o in objects:
            o = dict(o)
            if frame is not None and boxes and o.get("id") in boxes:
                o["crop"] = self._crop(frame, o["id"], boxes[o["id"]])
            objs.append(o)
        msg = {"t": time.time(),
               "tags": {str(i): [float(x), float(y), math.radians(h)] for i, (x, y, h) in tags.items()},
               "objects": objs}
        self.sock.sendto(json.dumps(msg, separators=(",", ":")).encode(), self.addr)

    def _crop(self, frame, oid, box, pad=0.25):
        """Save one JPEG per object (first sighting) for the VLM triage; returns its path."""
        import cv2
        path = os.path.abspath(os.path.join(self.crop_dir, f"{oid}.jpg"))
        if oid not in self._cropped:
            os.makedirs(self.crop_dir, exist_ok=True)
            x1, y1, x2, y2 = box
            dx, dy = (x2 - x1) * pad, (y2 - y1) * pad
            h, w = frame.shape[:2]
            c = frame[max(0, int(y1 - dy)):min(h, int(y2 + dy)), max(0, int(x1 - dx)):min(w, int(x2 + dx))]
            if c.size:
                cv2.imwrite(path, c)
                self._cropped.add(oid)
        return path


def floor_rect(poly_m):
    """Object footprint from its mask outline projected onto the floor (Nx2, metres):
    {"w": short side, "l": long side, "ang": direction of the long side in radians}.
    The arm cars approach along `ang` so the jaws close across the short side."""
    import cv2
    import numpy as np
    (_, _), (a, b), deg = cv2.minAreaRect(np.asarray(poly_m, dtype=np.float32))
    if a >= b:
        return {"w": float(b), "l": float(a), "ang": math.radians(deg)}
    return {"w": float(a), "l": float(b), "ang": math.radians(deg + 90)}
