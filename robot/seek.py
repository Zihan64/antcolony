"""Arm-car skill `seek`: find a named thing with the car's own camera and drive up to it.

    {"type": "skill", "name": "seek", "args": {"target": "white box", "stop_cm": 22}}

Used to bring a held object to a landmark ("put the block next to the white box"): seek, then
drop. No tracker. The loop is look - move - look:

  look    arm raised so the camera looks out across the floor (VIEW; the gripper servo is not
          touched: what it holds stays held), one photo, the VLM on the laptop boxes the target (~3 s)
  not seen    turn left about 20-30 deg and look again (gives up after a full turn per camera view)
  off centre  turn towards it (short timed pulse; the turn rate - and its sign - are learned
          from how far the target moved in the picture)
  centred     drive forward in short pulses, reading the front sonar at rest before each one,
          at most `seg_cm` (30), then look again

Arrived = the sonar reads `stop_cm` or less (after aiming at the target, the tall thing straight
ahead is taken to be the target). If the sonar never sees it (slanted face) the picture stops
the car instead: the target fills most of the width or its lower edge is at the bottom.

Limits: the target must be tall enough for the sonar or big enough in the picture; nothing is
avoided on the way (unlike `carry`); distances driven are counted from pulses, not measured.

Options: {"detector": "colour"} finds a small coloured thing ("red block") by its colour instead
of asking the VLM - far away a 3 cm block is ~30 px and the VLM does not see it, a vivid blob
is found at once; {"stop": "aim"} only turns until the target is centred; {"bottom_stop": 0.4}
drives towards a small thing on the floor until its lower edge is 40 % down the picture (then
`grab` takes over); {"img_tokens": 576} sends the VLM a bigger picture than the car's default.

Skill `dock` (same file): the last centimetres to a low white tray / basket that the sonar does
not see. Arm in vlm_grab's calibrated LOOK pose (camera looks down at the floor 8-22 cm ahead),
the nearest carpet-to-white edge straight ahead is measured in cm with the table calibration,
and the car creeps until that edge is `edge_cm` (12; the look-down picture covers 9.4-24 cm) in
front of the arm base. `drop` with y_cm 21 then lets go about 9 cm inside the rim.
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
VIEW = (0, 10, 16, -25)                       # really about -33 deg (the nearest the arm can do there): floor up to ~1.5 m
VIEW_FAR = (0, 10, 20, -12)                   # nearly level: the whole floor to the walls; nearest floor ~30 cm ahead.
                                              # {"view": "far"} - for a landmark that may be across the room (2026-11-14: a
                                              # tray 2 m away was never in the lower view during a full turn)
HFOV_DEG = 66.0                               # guess; only sets how big the first turn is


HUES = {"red": [(0, 9), (171, 180)], "orange": [(8, 22)], "yellow": [(20, 35)], "green": [(36, 85)],
        "blue": [(90, 130)], "purple": [(125, 155)], "pink": [(150, 172)]}     # OpenCV hue, 0..180


def colour_blob(cv2, np, frame, name):
    """Biggest vivid blob of the colour named in `name` ("red block"; no colour word = any vivid
    colour). -> (x1, y1, x2, y2) or None. Carpet, wood, paper and white things are not vivid."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
    vivid = (hsv[:, :, 1] > 140) & (hsv[:, :, 2] > 110)
    rng = next((v for k, v in HUES.items() if k in name.lower()), None)
    if rng:
        hue = np.zeros(vivid.shape, bool)
        for lo, hi in rng:
            hue |= (hsv[:, :, 0] >= lo) & (hsv[:, :, 0] <= hi)
        vivid &= hue
    m = cv2.morphologyEx(vivid.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cs = [c for c in cs if 40 <= cv2.contourArea(c) <= 0.25 * m.size]
    if not cs:
        return None
    x, y, w, h = cv2.boundingRect(max(cs, key=cv2.contourArea))
    return x, y, x + w, y + h


def open_cam(cv2):
    devs = sorted(glob.glob("/dev/v4l/by-id/*-video-index0"))
    cap = cv2.VideoCapture(os.path.realpath(devs[0]) if devs else 0, cv2.CAP_V4L2 if devs else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class Eye:
    """The car's camera in the far view + the VLM. see() -> None or
    {"bearing": deg (+ = left), "wide": box width / picture width, "bottom": lower edge / height}."""

    def __init__(self, hw, target, view=None, detector="vlm", img_tokens=None):
        import cv2
        import numpy as np
        from masterpi_lite import vlm
        self.cv2, self.np, self.vlm, self.hw, self.target, self.detector = cv2, np, vlm, hw, target, detector
        self.tokens0 = vlm.model_size.__defaults__
        if img_tokens:                                    # (the default is bound when vlm.py is imported)
            vlm.model_size.__defaults__ = (int(img_tokens),)
        self.cap = open_cam(cv2)
        # "far" = the level view first; if a whole turn shows nothing, the lower view (a thing right
        # in front of the car is below the level view, or hidden behind what the gripper holds)
        self.views = [(VIEW_FAR, 4), (VIEW, 12)] if view == "far" else [(tuple(view), 4)] if view else [(VIEW, 12)]
        self.pose(*self.views.pop(0))
        self.n = 0
        for f in glob.glob(os.path.join(HERE, "seek_[0-9][0-9].jpg")):
            os.remove(f)
        for _ in range(20):                               # exposure / white balance
            self.cap.read()

    def pose(self, view, slack):
        x, y, z, pitch = view
        with self.hw.lock:
            ok = self.hw.arm.move_to(x, y, z, pitch=pitch, pitch_min=pitch - slack, pitch_max=pitch + slack, ms=1500)
            if ok is None:                                # out of reach: the driving view is better than none
                print(f"  seek: view {view} out of reach - using the far driving view", flush=True)
                self.hw.arm.set_pulses(1500, dict(FAR))
                if hasattr(self.hw.arm, "pos"):
                    self.hw.arm.pos = None
        time.sleep(1.9)

    def next_view(self):
        """Switch to the next camera view, if there is one. -> switched?"""
        if not self.views:
            return False
        self.pose(*self.views.pop(0))
        return True

    def see(self):
        frame = None
        for attempt in range(4):
            for _ in range(10 if attempt == 0 else 5):    # frames queued while the car moved
                ok, f = self.cap.read()
                if ok:
                    frame = f
            if frame is None:
                raise RuntimeError("camera: no picture (USB cable?)")
            gray = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)
            if self.cv2.Laplacian(gray, self.cv2.CV_64F).var() >= 200:     # sharp (still photos here: 500-1400; smeared: < 100)
                break
            time.sleep(0.4)                               # the arm is still swaying: wait, take it again
        h, w = frame.shape[:2]
        if self.detector == "colour":
            box = colour_blob(self.cv2, self.np, frame, self.target)
        else:
            det = self.vlm.detect(frame, self.target, verbose=False)
            box = det["box"] if det else None
        out = None
        if box is not None:
            x1, y1, x2, y2 = box
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
        self.vlm.model_size.__defaults__ = self.tokens0


def run(hw, args, cancel, eye=None, carry_pose=None):
    """-> (ok, detail). `eye` and `carry_pose` are replaced in tests/test_seek.py."""
    target = str(args.get("target", "white box"))
    stop_cm = float(args.get("stop_cm", 22))
    tol = float(args.get("tol_deg", 7))
    rate = float(args.get("turn_deg_s", 110))             # first guess, re-measured after every turn
    w_cmd, u = float(args.get("w", 0.3)), float(args.get("u", 0.05))
    cm_s, step = float(args.get("cm_s", 17)), float(args.get("step_s", 0.33))
    seg, max_cm = float(args.get("seg_cm", 30)), float(args.get("max_cm", 200))
    scan_deg, max_scans = float(args.get("scan_deg", 20)), int(args.get("max_scans", 20))   # car B really turns ~1.5x the guess
    timeout = float(args.get("timeout", 360))
    say = lambda msg: print(f"  seek: {msg}", flush=True)
    t0, sign, drove, looks, scans, last, had = time.monotonic(), 1.0, 0.0, 0, 0, None, False
    made = eye is None
    if made:
        eye = Eye(hw, target, args.get("view"), args.get("detector", "vlm"), args.get("img_tokens"))

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
                if had and last is None:                  # it was there a moment ago and the car has not turned:
                    had = False                           # the VLM misses now and then - look once more first
                    say(f"look {looks}: '{target}' not found this time - looking again")
                    continue
                if scans >= max_scans:
                    if hasattr(eye, "next_view") and eye.next_view():
                        say("nothing in a whole turn with this view - once more with the other camera view")
                        scans = 0
                        continue
                    return finish(False, f"'{target}' not found in a full turn ({looks} looks)")
                say(f"look {looks}: '{target}' not in the picture - turning left")
                turn(scan_deg)
                scans, last, had = scans + 1, None, False
                continue
            b, had, scans = s["bearing"], True, 0
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
            if args.get("stop") == "aim":
                return finish(True, f"facing the {target}: {abs(b):.0f} deg off, {100 * s['wide']:.0f} % wide, "
                                    f"lower edge at {100 * s['bottom']:.0f} % of the picture, {looks} looks")
            bstop = args.get("bottom_stop")               # small target on the floor: close enough when its lower
            if bstop is not None and s["bottom"] >= float(bstop):      # edge is this far down the picture
                return finish(True, f"close to the {target}: lower edge at {100 * s['bottom']:.0f} % of the picture, "
                                    f"{looks} looks, drove {drove:.0f} cm")
            # centred: forward, sonar first
            near_by_eye = s["wide"] >= 0.75 or (s["bottom"] >= 0.92 and s["wide"] >= 0.4)
            # the closer it looks, the shorter the stretch driven before the next look (a low
            # tray is not seen by the sonar: only the picture keeps the car from driving into it)
            seg_now = seg if (s["wide"] < 0.3 and s["bottom"] < 0.6) else seg / 2 if (s["wide"] < 0.5 and s["bottom"] < 0.8) else 6.0
            if bstop is not None:
                seg_now = min(seg_now, 20.0 if s["bottom"] < float(bstop) - 0.15 else 8.0)
            gone = 0.0
            while gone < seg_now:
                if cancel.is_set():
                    return finish(False, f"cancelled after {drove:.0f} cm")
                d = hw.sonar_med()
                if d is not None and d <= stop_cm:
                    if bstop is not None:                 # the sonar does not see a small block: this is something else
                        return finish(False, f"something tall {d:.0f} cm ahead on the way to the {target}")
                    return finish(True, f"at the {target}: sonar {d:.0f} cm, {looks} looks, drove {drove:.0f} cm")
                if near_by_eye:                           # the picture alone decides: the sonar may be reading something
                    return finish(True, f"at the {target} by the picture ({100 * s['wide']:.0f} % wide, lower edge at "   # BEHIND a low
                                        f"{100 * s['bottom']:.0f} %; sonar {d if d is None else round(d)} cm), "        # tray (2026-11-14)
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


# ------------------------------------------------------------------ dock: up to a low white rim
LOOK = {3: 565, 4: 2271, 5: 1627, 6: 1500}    # vlm_grab.LOOK_PULSES: the pose the table calibration was made in


class EdgeEye:
    """Camera in the calibrated LOOK pose. edge() -> cm from the arm base to the nearest
    carpet-to-white edge straight ahead, or "none" (only carpet in view: too far) or "close"
    (white right at the bottom of the picture: too near). Saves dock.jpg with the edge drawn."""

    def __init__(self, hw):
        import json
        import cv2
        import numpy as np
        self.cv2, self.np, self.hw = cv2, np, hw
        with open(os.path.join(HERE, "masterpi_lite", "vision_calib.json"), encoding="utf-8") as f:
            self.H = np.array(json.load(f)["H"], dtype=np.float64)
        self.cap = open_cam(cv2)
        with hw.lock:
            hw.arm.set_pulses(1500, dict(LOOK))
            if hasattr(hw.arm, "pos"):
                hw.arm.pos = None
        time.sleep(2.3)
        for _ in range(20):
            self.cap.read()

    def edge(self):
        cv2, np = self.cv2, self.np
        frame = None
        for _ in range(10):
            ok, f = self.cap.read()
            if ok:
                frame = f
        if frame is None:
            raise RuntimeError("camera: no picture (USB cable?)")
        h, w = frame.shape[:2]
        out = white_edge(cv2, np, frame)
        if isinstance(out, int):
            y_cm = float(cv2.perspectiveTransform(np.array([[[w / 2.0, float(out)]]]), self.H)[0, 0, 1])
            cv2.line(frame, (0, out), (w, out), (0, 255, 0), 2)
            cv2.putText(frame, f"{y_cm:.1f} cm", (8, max(20, out - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            out = y_cm
        cv2.imwrite(os.path.join(HERE, "dock.jpg"), frame)
        return out

    def close(self):
        self.cap.release()


def white_edge(cv2, np, frame):
    """Row (px) of the nearest carpet-to-white edge in the middle of the picture, or "none" / "close".
    White here = clearly brighter than the rest of the strip, not coloured, and SMOOTH: brightness
    alone does not do it - the light stripes of a patterned carpet are as bright as a white wall in
    shadow (2026-11-14), but carpet pile is speckled and a tray wall is not."""
    h, w = frame.shape[:2]
    strip = frame[:, w // 2 - 70: w // 2 + 70]
    hsv = cv2.cvtColor(cv2.GaussianBlur(strip, (5, 5), 0), cv2.COLOR_BGR2HSV)
    val, sat = np.median(hsv[:, :, 2], axis=1), np.median(hsv[:, :, 1], axis=1)
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY).astype(np.float32)
    rough = np.abs(gray - cv2.blur(gray, (9, 9))).mean(axis=1)             # speckle per row
    smooth = rough < max(2.5, 0.45 * float(np.median(rough)))
    if smooth.sum() < 12:
        return "none"
    thr = float(np.median(val[~smooth])) + 25 if (~smooth).sum() >= 12 else 0.0   # brighter than the carpet rows
    white = smooth & (sat < 90) & (val > thr)
    if white.mean() > 0.9:
        return "close"
    for r in range(h - 1, 12, -1):                        # from the car outwards: first white stretch
        if white[r - 11: r + 1].sum() >= 10:
            return "close" if r >= h - 6 else r
    return "none"


def dock(hw, args, cancel, eye=None, carry_pose=None):
    """Creep until the near edge of the white tray is `edge_cm` in front of the arm base. -> (ok, detail)."""
    want, tol = float(args.get("edge_cm", 12)), float(args.get("tol_cm", 1.5))
    cm_s, u = float(args.get("cm_s", 17)), float(args.get("u", 0.05))
    say = lambda msg: print(f"  dock: {msg}", flush=True)
    made = eye is None
    if made:
        eye = EdgeEye(hw)

    def finish(ok, detail):
        hw.stop()
        if made:
            eye.close()
        (carry_pose or hw.carry_pose)()
        return ok, detail

    try:
        blind = 0
        for i in range(int(args.get("max_moves", 14))):
            if cancel.is_set():
                return finish(False, "cancelled")
            y = eye.edge()
            if y == "none":
                blind += 1
                if blind > int(args.get("max_blind", 5)):
                    return finish(False, "no white edge in front of the car")
                say("no white edge in the picture yet - forward 5 cm")
                hw.pulse(u, 0.0, 5.0 / cm_s, cancel)
                continue
            if y == "close":
                say("white right under the camera - back 2.5 cm")
                hw.pulse(-u, 0.0, 2.5 / cm_s, cancel)
                continue
            err = y - want
            say(f"edge {y:.1f} cm ahead (want {want:.0f})")
            if abs(err) <= tol:
                return finish(True, f"docked: edge {y:.1f} cm in front of the arm base")
            hw.pulse(math.copysign(u, err), 0.0, min(0.33, max(0.10, 0.8 * abs(err) / cm_s)), cancel)
        return finish(False, "could not settle at the edge")
    except Exception as e:
        return finish(False, f"{type(e).__name__}: {e}")
