"""Arm-car skill `seek`: find a named thing with the car's own camera and drive up to it.

    {"type": "skill", "name": "seek", "args": {"target": "white box", "stop_cm": 22}}

Used to bring a held object to a landmark ("put the block next to the white box"): seek, then
drop. No tracker. The loop is look - move - look:

  look    arm raised so the camera looks out across the floor (VIEW; the gripper servo is not
          touched: what it holds stays held), one photo, the VLM on the laptop boxes the target (~3 s)
  not seen    turn left about 30 deg and look again (gives up after a full turn)
  off centre  turn towards it (short timed pulse; the turn rate - and its sign - are learned
          from how far the target moved in the picture)
  centred     drive forward in short pulses, reading the front sonar at rest before each one,
          at most `seg_cm` (30), then look again

Arrived = the sonar reads `stop_cm` or less (after aiming at the target, the tall thing straight
ahead is taken to be the target). If the sonar never sees it (slanted face) the picture stops
the car instead: the target fills most of the width or its lower edge is at the bottom.

Limits: the target must be tall enough for the sonar or big enough in the picture; nothing is
avoided on the way (unlike `carry`); distances driven are counted from pulses, not measured.
`{"look_only": true}` moves only the arm: one photo, says where the target is.
Every photo is saved next to this file: seek_01.jpg, seek_02.jpg ... (box drawn in when found),
seek.jpg = the last one.
"""
import glob
import math
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FAR = {3: 950, 4: 2271, 5: 1627, 6: 1500}     # vlm_grab.DRIVE_POSES "far": looks at the floor up to ~1 m ahead
# For a landmark the camera must look out, not down: the far driving view cuts off everything
# beyond about 1 m and the top of anything tall (2026-11-14: a box across the room was never in
# the picture during a full turn). Gripper tip (x, y, z) cm and pitch; the camera looks along the gripper.
VIEW = (0, 10, 16, -25)
HFOV_DEG = 66.0                               # guess; only sets how big the first turn is


class Eye:
    """The car's camera in the far view + the VLM. see() -> None or
    {"bearing": deg (+ = left), "wide": box width / picture width, "bottom": lower edge / height}."""

    def __init__(self, hw, target, view=None):
        import cv2
        from masterpi_lite import vlm
        self.cv2, self.vlm, self.hw, self.target = cv2, vlm, hw, target
        devs = sorted(glob.glob("/dev/v4l/by-id/*-video-index0"))
        self.cap = cv2.VideoCapture(os.path.realpath(devs[0]) if devs else 0, cv2.CAP_V4L2 if devs else cv2.CAP_ANY)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        x, y, z, pitch = view or VIEW
        with hw.lock:
            ok = hw.arm.move_to(x, y, z, pitch=pitch, pitch_min=pitch - 12, pitch_max=pitch + 12, ms=1500)
            if ok is None:                                # out of reach: the driving view is better than none
                print(f"  seek: view {(x, y, z, pitch)} out of reach - using the far driving view", flush=True)
                hw.arm.set_pulses(1500, dict(FAR))
                if hasattr(hw.arm, "pos"):
                    hw.arm.pos = None
        time.sleep(1.9)
        self.n = 0
        for f in glob.glob(os.path.join(HERE, "seek_[0-9][0-9].jpg")):
            os.remove(f)
        for _ in range(20):                               # exposure / white balance
            self.cap.read()

    def see(self):
        frame = None
        for _ in range(10):                               # frames queued while the car moved
            ok, f = self.cap.read()
            if ok:
                frame = f
        if frame is None:
            raise RuntimeError("camera: no picture (USB cable?)")
        h, w = frame.shape[:2]
        det = self.vlm.detect(frame, self.target, verbose=False)
        out = None
        if det:
            x1, y1, x2, y2 = det["box"]
            fx = (w / 2) / math.tan(math.radians(HFOV_DEG / 2))
            out = {"bearing": math.degrees(math.atan((w / 2 - (x1 + x2) / 2) / fx)),
                   "wide": (x2 - x1) / w, "bottom": y2 / h}
            self.cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        self.n += 1
        self.cv2.imwrite(os.path.join(HERE, "seek.jpg"), frame)
        self.cv2.imwrite(os.path.join(HERE, f"seek_{self.n:02d}.jpg"), frame)       # every look, for afterwards
        return out

    def close(self):
        self.cap.release()


def run(hw, args, cancel, eye=None, carry_pose=None):
    """-> (ok, detail). `eye` and `carry_pose` are replaced in tests/test_seek.py."""
    target = str(args.get("target", "white box"))
    stop_cm = float(args.get("stop_cm", 22))
    tol = float(args.get("tol_deg", 7))
    rate = float(args.get("turn_deg_s", 110))             # first guess, re-measured after every turn
    w_cmd, u = float(args.get("w", 0.3)), float(args.get("u", 0.05))
    cm_s, step = float(args.get("cm_s", 17)), float(args.get("step_s", 0.33))
    seg, max_cm = float(args.get("seg_cm", 30)), float(args.get("max_cm", 200))
    scan_deg, max_scans = float(args.get("scan_deg", 30)), int(args.get("max_scans", 13))
    timeout = float(args.get("timeout", 240))
    say = lambda msg: print(f"  seek: {msg}", flush=True)
    t0, sign, drove, looks, scans, last = time.monotonic(), 1.0, 0.0, 0, 0, None
    made = eye is None
    if made:
        eye = Eye(hw, target, args.get("view"))

    def turn(deg):
        """Turn about `deg` (+ = left) with one timed pulse. -> seconds it lasted."""
        t = min(0.6, max(0.08, abs(deg) / rate))
        hw.set_velocity(0.0, 0.0, math.copysign(w_cmd, deg) * sign)
        t_end = time.monotonic() + t
        while time.monotonic() < t_end and not cancel.is_set():
            time.sleep(0.01)
        hw.stop()
        time.sleep(0.7)                                   # the raised arm sways after a turn: a photo taken too early is blurred
        return t

    def finish(ok, detail):
        hw.stop()
        if made:
            eye.close()
        (carry_pose or hw.carry_pose)()
        return ok, detail

    try:
        while True:
            if cancel.is_set():
                return finish(False, f"cancelled after {drove:.0f} cm")
            if time.monotonic() - t0 > timeout:
                return finish(False, f"timeout after {looks} looks, {drove:.0f} cm")
            s = eye.see()
            looks += 1
            if args.get("look_only"):
                d = hw.sonar_med()
                return finish(s is not None, f"{target}: " + ("not in the picture" if s is None else
                              f"{abs(s['bearing']):.0f} deg {'left' if s['bearing'] > 0 else 'right'}, "
                              f"{100 * s['wide']:.0f} % of the width") + f"; sonar {d if d is None else round(d)} cm")
            if s is None:
                if scans >= max_scans:
                    return finish(False, f"'{target}' not found in a full turn ({looks} looks)")
                say(f"look {looks}: '{target}' not in the picture - turning left")
                turn(scan_deg)
                scans, last = scans + 1, None
                continue
            b = s["bearing"]
            if last is not None:                          # what did the last turn really do?
                asked, secs, b0 = last
                did = b0 - b                              # turning left moves the target to the right
                if abs(did) > 4 and did * asked < 0:
                    sign = -sign
                    say(f"the car turned the other way ({did:+.0f} deg for {asked:+.0f}) - swapping the turn direction")
                elif abs(did) > 2:
                    rate = max(30.0, min(300.0, 0.5 * rate + 0.5 * abs(did) / secs))
                    if b * b0 < 0 and abs(b) <= 14:       # the shortest pulse overshoots: this is as good as it gets
                        tol = max(tol, abs(b) + 1)        # (the sonar beam is +-15 deg, and we look again on the way)
                last = None
            d = hw.sonar_med()
            say(f"look {looks}: '{target}' {abs(b):.0f} deg {'left' if b > 0 else 'right'}, "
                f"{100 * s['wide']:.0f} % wide, sonar {d if d is None else round(d)} cm")
            if abs(b) > tol:
                ask = 0.8 * b                             # aim a little short: an overshoot costs a whole look
                last = (ask, turn(ask), b)
                continue
            # centred: forward, sonar first
            near_by_eye = s["wide"] >= 0.75 or (s["bottom"] >= 0.92 and s["wide"] >= 0.4)
            gone = 0.0
            while gone < seg:
                if cancel.is_set():
                    return finish(False, f"cancelled after {drove:.0f} cm")
                d = hw.sonar_med()
                if d is not None and d <= stop_cm:
                    return finish(True, f"at the {target}: sonar {d:.0f} cm, {looks} looks, drove {drove:.0f} cm")
                if near_by_eye and (d is None or d > 3 * stop_cm):
                    return finish(True, f"at the {target} by the picture (sonar {d if d is None else round(d)} cm does not see it), "
                                        f"{looks} looks, drove {drove:.0f} cm")
                if drove >= max_cm:
                    return finish(False, f"drove {drove:.0f} cm and never reached the {target}")
                t = step if d is None else min(step, max(0.12, (d - stop_cm + 2) / cm_s))
                hw.pulse(u, 0.0, t, cancel)
                gone, drove = gone + cm_s * t, drove + cm_s * t
                if near_by_eye:
                    break                                 # close: look after every step
    except Exception as e:
        return finish(False, f"{type(e).__name__}: {e}")
