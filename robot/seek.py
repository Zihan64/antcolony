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
and the car creeps until that edge is `edge_cm` (13; the look-down picture covers 9.4-24 cm) in
front of the arm base. `drop` with y_cm 22 then lets go about 9 cm inside the rim. See dock().
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
# Rough distance from the picture, for things standing on the floor: where the camera is in each
# view (cm ahead of the arm base, cm above the floor, deg it looks down), worked out from the
# "camera" entry of vision_calib.json (LOOK pose) and the arm geometry; focal length from the same
# calibration (10 cm on the floor = 388 px at 11 cm). Good to maybe +-20 %: enough to decide how
# far the car may still drive before it has to look again, not for the last centimetres (dock).
F_PX = 430.0
GEOM = {VIEW: (13.3, 14.9, 33.0), VIEW_FAR: (13.4, 20.2, 12.0)}


def floor_cm(geom, row, h=480):
    """cm ahead of the arm base of the floor point seen at picture row `row`; None near / above the horizon."""
    cam_y, cam_h, down = geom
    ang = math.radians(down) + math.atan((row - h / 2.0) * (480.0 / h) / F_PX)
    return None if ang < math.radians(4) else cam_y + cam_h / math.tan(ang)


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
        self.geom = GEOM.get(tuple(view))
        with self.hw.lock:
            ok = self.hw.arm.move_to(x, y, z, pitch=pitch, pitch_min=pitch - slack, pitch_max=pitch + slack, ms=1500)
            if ok is None:                                # out of reach: the driving view is better than none
                print(f"  seek: view {view} out of reach - using the far driving view", flush=True)
                self.geom = None
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
                   "wide": (x2 - x1) / w, "bottom": y2 / h,
                   "dist": floor_cm(self.geom, y2, h) if self.geom else None}     # its foot on the floor, cm ahead
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
            # The car must NEVER touch the target (user, 2026-11-14: it lunged at the tray). So this skill
            # only brings it to `arrive_cm` in front of the target, by the picture: the stretch driven
            # before the next look is at most half of what is left to that line, and `dock` does the rest.
            dist, arrive = s.get("dist"), float(args.get("arrive_cm", 40))
            if bstop is None and dist is not None and dist <= arrive:
                return finish(True, f"in front of the {target}: about {dist:.0f} cm ahead by the picture, "
                                    f"{looks} looks, drove {drove:.0f} cm")
            # centred: forward, sonar first
            near_by_eye = s["wide"] >= 0.75 or (s["bottom"] >= 0.92 and s["wide"] >= 0.4)
            seg_now = seg if (s["wide"] < 0.3 and s["bottom"] < 0.6) else seg / 2 if (s["wide"] < 0.5 and s["bottom"] < 0.8) else 6.0
            if bstop is not None:
                seg_now = min(seg_now, 20.0 if s["bottom"] < float(bstop) - 0.15 else 8.0)
            elif dist is not None:
                seg_now = max(5.0, min(seg, 0.5 * (dist - arrive) + 3.0))
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
                if seg_now - gone >= 15 and (d is None or d > stop_cm + 40) and hasattr(hw, "cruise"):
                    went = hw.cruise(u, seg_now - gone, cancel, guard_cm=stop_cm + 12)     # far from everything: one
                    gone, drove = gone + max(went, 1.0), drove + went                     # smooth move, not stop-and-go
                    continue
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
    """The camera for docking. view() + blob(): the lower search view, where the tray is roughly
    (sideways offset and distance, good to a few cm); look() + edge(): the calibrated LOOK pose,
    what measure() says about the white edge right in front of the car. Saves dock_view.jpg / dock.jpg."""

    def __init__(self, hw):
        import json
        import cv2
        import numpy as np
        self.cv2, self.np, self.hw = cv2, np, hw
        with open(os.path.join(HERE, "masterpi_lite", "vision_calib.json"), encoding="utf-8") as f:
            self.H = np.array(json.load(f)["H"], dtype=np.float64)
        self.cap = open_cam(cv2)
        for _ in range(20):
            self.cap.read()

    def frame(self):
        frame = None
        time.sleep(0.3)                                   # let the arm stop swaying
        for _ in range(10):
            ok, f = self.cap.read()
            if ok:
                frame = f
        if frame is None:
            raise RuntimeError("camera: no picture (USB cable?)")
        return frame

    def view(self):
        x, y, z, pitch = VIEW
        with self.hw.lock:
            self.hw.arm.move_to(x, y, z, pitch=pitch, pitch_min=pitch - 12, pitch_max=pitch + 12, ms=1200)
        time.sleep(1.8)

    def look(self):
        with self.hw.lock:
            self.hw.arm.set_pulses(1500, dict(LOOK))
            if hasattr(self.hw.arm, "pos"):
                self.hw.arm.pos = None
        time.sleep(2.3)

    def blob(self):
        cv2, frame = self.cv2, self.frame()
        b = white_blob(cv2, self.np, frame)
        if b is not None:
            x1, y1, x2, y2 = b["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"off {b['off']:+.1f} cm, near edge {b['dist']:.0f} cm", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(HERE, "dock_view.jpg"), frame)
        return b

    def edge(self):
        cv2, frame = self.cv2, self.frame()
        m = measure(cv2, self.np, frame, self.H)
        if m.get("row") is not None:
            cv2.line(frame, (0, m["row"]), (frame.shape[1], m["row"]), (0, 255, 0), 2)
        cv2.putText(frame, " ".join(f"{k}={v if not isinstance(v, float) else round(v, 1)}" for k, v in m.items() if k != "row"),
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(HERE, "dock.jpg"), frame)
        return m

    def close(self):
        self.cap.release()


def white_blob(cv2, np, frame):
    """The nearest big white, smooth thing in the lower search view (VIEW) -> None or
    {"off": cm it lies to the LEFT of the car's line (+) / right (-), "dist": cm from the arm base to
    its near edge (21 = at the bottom of the picture or nearer), "area": share of the picture, "box"}."""
    h, w = frame.shape[:2]
    m = cv2.morphologyEx(white_mask(cv2, np, frame).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m[int(0.86 * h):, int(0.2 * w): int(0.9 * w)] = 0     # what the gripper holds shows here, and a pale block passes for white
    n, _, stats, cent = cv2.connectedComponentsWithStats(m, 8)
    best = None
    for i in range(1, n):
        x, y, bw, bh, area = (int(v) for v in stats[i])
        if area < 0.04 * h * w or y + bh < 0.3 * h:       # too small, or only up in the background (walls, heater)
            continue
        if best is None or y + bh > best[1]:
            best = (i, y + bh)
    if best is None:
        return None
    i = best[0]
    x, y, bw, bh, area = (int(v) for v in stats[i])
    geom = GEOM[VIEW]
    near = floor_cm(geom, y + bh - 1, h) or 300.0
    mid = floor_cm(geom, float(cent[i][1]), h) or 300.0
    off = (mid - geom[0]) * (w / 2.0 - float(cent[i][0])) / F_PX        # sideways, at the blob's middle
    return {"off": off, "dist": near, "area": area / float(h * w), "box": (x, y, x + bw, y + bh)}


def white_mask(cv2, np, frame):
    """Pixels of a white, SMOOTH surface. Brightness alone does not do it - the light stripes of a
    patterned carpet are as bright as a white wall in shadow (2026-11-14) - but carpet pile is
    speckled (mean |pixel - 9x9 blur| around 10) and a tray wall is not (around 1)."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    smooth = cv2.blur(np.abs(gray - cv2.blur(gray, (9, 9))), (15, 15)) < 3.0
    val = hsv[:, :, 2].astype(np.float32)
    ref = float(np.median(val[~smooth])) if (~smooth).mean() > 0.05 else 0.0       # the carpet's brightness
    return smooth & (hsv[:, :, 1] < 90) & (val > ref + 25)


def measure(cv2, np, frame, H):
    """Where is the white tray in the look-down picture? ->
      {"state": "none"}                         nothing white in view (too far)
      {"state": "side", "side": +1 | -1}        white only to the left (+1) / right (-1) of the car's line
      {"state": "close"}                        white right at the bottom of the picture (too near)
      {"state": "edge", "y": cm, "deep": bool, "side": ..., "row": px}
          nearest carpet-to-white edge on the car's centre line, `y` cm ahead of the arm base;
          deep = it stays white for 10 cm beyond that edge (room to drop into)."""
    h, w = frame.shape[:2]
    m = white_mask(cv2, np, frame)
    if m.mean() < 0.015:
        return {"state": "none"}
    side = 1 if m[:, : w // 2].sum() > m[:, w // 2:].sum() else -1
    row = m[:, w // 2 - 70: w // 2 + 70].mean(axis=1) > 0.6
    edge, low = None, int(0.93 * h)                       # (below that row: what the gripper holds)
    for r in range(low, 12, -1):                          # from the car outwards: the first white stretch
        if row[r - 11: r + 1].sum() >= 10:
            edge = r
            break
    if edge is None:
        return {"state": "side", "side": side}
    if edge >= low - 6:
        return {"state": "close", "side": side}
    to_cm = lambda v: float(cv2.perspectiveTransform(np.array([[[w / 2.0, float(v)]]]), H)[0, 0, 1])
    y = to_cm(edge)
    far = next((r for r in range(edge, -1, -1) if to_cm(r) >= y + 10), 0)          # row 10 cm beyond the edge
    return {"state": "edge", "y": y, "deep": bool(row[far: edge + 1].mean() >= 0.85), "side": side, "row": edge}


def dock(hw, args, cancel, eye=None, carry_pose=None):
    """The last ~45 cm to the white tray, without ever touching it. -> (ok, detail).
    1. lower search view: the tray is found as the nearest big white smooth thing; the car slides
       sideways (mecanum) until it is on the car's line and creeps up - at most 6 cm at a time, each
       step from a fresh photo - until its near edge is about 24 cm ahead. No tray in the photo =
       no move at all.
    2. calibrated LOOK pose: the edge on the car's line in cm; creep until it is `edge_cm` (13) in
       front of the arm base (the chassis ends ~6 cm in front of the arm base: ~7 cm of air), sliding again if the
       tray is off to one side or there is not 10 cm of tray beyond the edge. At most 3 moves of
       4 cm with nothing white in view, then it gives up rather than push on blind.
    (2026-11-14: an earlier version crept 25 cm blind past the tray's corner.)"""
    want, tol = float(args.get("edge_cm", 13)), float(args.get("tol_cm", 1.5))
    cm_s, side_cm_s, u = float(args.get("cm_s", 17)), float(args.get("side_cm_s", 15)), float(args.get("u", 0.05))
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

    def slide(side, cm):
        say(f"the tray is to the {'left' if side > 0 else 'right'} - sliding {cm:.0f} cm that way")
        hw.pulse(0.0, side * u, max(0.08, cm / side_cm_s), cancel)

    try:
        eye.view()
        for i in range(16):
            if cancel.is_set():
                return finish(False, "cancelled")
            b = eye.blob()
            if b is None:
                return finish(False, "no white tray in front of the car" + (" - nothing moved" if i == 0 else " any more"))
            say(f"tray {abs(b['off']):.0f} cm to the {'left' if b['off'] > 0 else 'right'}, near edge about {b['dist']:.0f} cm ahead")
            if i == 0 and b["dist"] > float(args.get("max_start_cm", 75)):
                return finish(False, f"the tray is still about {b['dist']:.0f} cm away - seek first (nothing moved)")
            if abs(b["off"]) > 3.0:
                slide(1 if b["off"] > 0 else -1, min(6.0, 0.8 * abs(b["off"])))
            elif b["dist"] <= 25.0:
                break
            else:
                hw.pulse(u, 0.0, max(0.10, min(6.0, b["dist"] - 22.0) / cm_s), cancel)
        else:
            return finish(False, "could not line up with the tray")

        eye.look()
        blind, slides, prev = 0, 0, None
        for i in range(int(args.get("max_moves", 20))):
            if cancel.is_set():
                return finish(False, "cancelled")
            m = eye.edge()
            if m["state"] == "none":
                blind += 1
                if blind > 3:
                    return finish(False, "the tray's edge did not come into the look-down picture - stopped")
                say("nothing white under the camera yet - forward 4 cm")
                hw.pulse(u, 0.0, 4.0 / cm_s, cancel)
            elif m["state"] == "side" or (m["state"] == "edge" and not m["deep"] and slides < 8):
                slides += 1
                if slides > 8:
                    return finish(False, "the tray stays to one side of the car")
                slide(m["side"], 4.0 if m["state"] == "side" else 3.0)
            elif m["state"] == "close":
                say("white right under the camera - back 2.5 cm")
                hw.pulse(-u, 0.0, 2.5 / cm_s, cancel)
            else:
                err = m["y"] - want
                say(f"edge {m['y']:.1f} cm ahead (want {want:.0f})")
                if prev is not None and err * prev < 0:   # went past it: the smallest move is too big for this
                    tol = min(3.0, tol + 0.75)            # tolerance - accept a little more rather than rock to and fro
                prev = err
                if abs(err) <= tol:
                    return finish(True, f"docked: edge {m['y']:.1f} cm in front of the arm base"
                                        + ("" if m["deep"] else " (not much tray behind it)"))
                hw.pulse(math.copysign(u, err), 0.0, min(0.33, max(0.10, 0.8 * abs(err) / cm_s)), cancel)
        return finish(False, "could not settle at the edge")
    except Exception as e:
        return finish(False, f"{type(e).__name__}: {e}")
