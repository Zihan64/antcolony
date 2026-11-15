"""Arm-car skill `seek`: find a named thing with the car's own camera and drive up to it.

    {"type": "skill", "name": "seek", "args": {"target": "white box", "stop_cm": 22}}

Used to bring a held object to a landmark ("put the block next to the white box"): seek, then
drop. No tracker. The loop is look - move - look:

  look    arm raised so the camera looks out across the floor (VIEW; the gripper servo is not
          touched: what it holds stays held), one photo, the VLM on the laptop boxes the target (~3 s)
  not seen    the ARM looks 50 deg left and right (the car stands still); still nothing: the car
          turns a third of a circle and looks again (gives up after a full turn per camera view)
  off centre  turn towards it (short timed pulse; the turn rate - and its sign - are learned
          from how far the target moved in the picture)
  centred     far from it: one smooth move, steering, with the target looked for in every camera
          frame (no VLM: its colour, or the white patch the VLM just named); near it: short pulses,
          the front sonar read at rest before each one, and a look after each

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


def vivid_mask(hsv):
    """Strongly coloured pixels: saturated, or - for a light block in bright light - less saturated but
    very bright (2026-11-14: a sky-blue block read S 113, V 247 and was missed at S > 140; the carpet's
    teal stripes and a blue-white bottle have the same hue but V under 175)."""
    return ((hsv[:, :, 1] > 140) & (hsv[:, :, 2] > 110)) | ((hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 200))


def colour_blob(cv2, np, frame, name):
    """Biggest vivid blob of the colour named in `name` ("red block"; no colour word = any vivid
    colour). -> (x1, y1, x2, y2) or None. Carpet, wood, paper and white things are not vivid.
    A blob with white all round it is skipped: that is an object already lying in the tray."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
    vivid = vivid_mask(hsv)
    rng = next((v for k, v in HUES.items() if k in name.lower()), None)
    if rng:
        hue = np.zeros(vivid.shape, bool)
        for lo, hi in rng:
            hue |= (hsv[:, :, 0] >= lo) & (hsv[:, :, 0] <= hi)
        vivid &= hue
    m = cv2.morphologyEx(vivid.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cs = sorted((c for c in cs if 40 <= cv2.contourArea(c) <= 0.25 * m.size), key=cv2.contourArea, reverse=True)
    white, (H, W) = None, m.shape
    for c in cs:
        x, y, w, h = cv2.boundingRect(c)
        if white is None:                                 # smooth and not coloured = tray (lit or in the block's shadow);
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)          # carpet is speckled
            white = (cv2.blur(np.abs(gray - cv2.blur(gray, (9, 9))), (15, 15)) < 3.0) & (hsv[:, :, 1] < 90)
        gx, gy = max(6, int(0.6 * w)), max(6, int(0.6 * h))
        X1, Y1, X2, Y2 = max(0, x - gx), max(0, y - gy), min(W, x + w + gx), min(H, y + h + gy)
        ring = white[Y1:Y2, X1:X2].sum() - white[y:y + h, x:x + w].sum()
        if ring / float(max(1, (X2 - X1) * (Y2 - Y1) - w * h)) > IN_TRAY:
            continue                                      # lies in the white tray: delivered, not to be fetched
        return x, y, x + w, y + h
    return None


def in_tray(cv2, np, frame, box):
    """Does the thing in `box` lie in the white tray? (The same test as in colour_blob: smooth,
    uncoloured surface all round it.)"""
    H, W = frame.shape[:2]
    x, y, x2, y2 = (int(v) for v in box)
    w, h = max(1, x2 - x), max(1, y2 - y)
    gx, gy = max(6, int(0.6 * w)), max(6, int(0.6 * h))
    X1, Y1, X2, Y2 = max(0, x - gx), max(0, y - gy), min(W, x + w + gx), min(H, y + h + gy)
    part = frame[Y1:Y2, X1:X2]
    gray = cv2.cvtColor(part, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sat = cv2.cvtColor(part, cv2.COLOR_BGR2HSV)[:, :, 1]
    white = (cv2.blur(np.abs(gray - cv2.blur(gray, (9, 9))), (15, 15)) < 3.0) & (sat < 90)
    inner = white[max(0, y - Y1): y - Y1 + h, max(0, x - X1): x - X1 + w].sum()
    return (white.sum() - inner) / float(max(1, white.size - w * h)) > IN_TRAY


def thing_colour(cv2, np, frame, box):
    """The colour word (HUES) of the thing in `box`, from the vivid pixels in its middle; None = not coloured."""
    x1, y1, x2, y2 = (int(v) for v in box)
    w, h = x2 - x1, y2 - y1
    part = frame[max(0, y1 + h // 5): max(y1 + h // 5 + 1, y2 - h // 5), max(0, x1 + w // 5): max(x1 + w // 5 + 1, x2 - w // 5)]
    if part.size == 0:
        return None
    hsv = cv2.cvtColor(part, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    vivid = hsv[(hsv[:, 1] > 80) & (hsv[:, 2] > 60)]
    if len(vivid) < 0.2 * len(hsv):
        return None
    hue = vivid[:, 0].astype(np.float32)
    if ((hue < 12) | (hue > 168)).mean() > 0.5:            # red wraps round 0 / 180: the median would land in the middle
        return "red"
    return colour_name(float(np.median(hue)))


def nearest_object(cv2, np, frame, name, yolo, conf=0.5, classes=None):
    """As nearest_thing(), but WHAT counts as an object is decided by the trained detector (shape),
    not by colour: a coloured wall, a shirt or a hand is not a block (2026-11-15: a car kept aiming
    at the red wall behind the tray). Colour only says WHICH object: "red block" = the nearest red
    one, "object not blue" = the nearest that is not blue. Things lying in the tray do not count.
    -> {"box", "colour", "label", "conf"} or None."""
    low = name.lower()
    skip = [k for k in HUES if "not " + k in low]
    want = next((k for k in HUES if k in low and k not in skip), None)
    best = None
    for x1, y1, x2, y2, s, n in yolo.infer(frame):
        if s < 0.6 * conf or (classes and n.lower() not in classes):
            continue
        box = (int(x1), int(y1), int(x2), int(y2))
        col = thing_colour(cv2, np, frame, box)
        # What the cars' own photos showed (2026-11-15, 26 real blocks): a block's box is about square
        # (0.80-1.39 wide / high), never cut by the picture's edge when it is aimed at, and coloured; far
        # ones (~25 px) score only 0.3-0.45. False "blocks": a wall panel at the edge (0.57, 0.50 wide /
        # high, no colour), a grey panel (0.43, no colour), the gripper's red tips (0.45, 4 wide / high).
        bw, bh = box[2] - box[0], box[3] - box[1]
        if not 0.55 <= bw / float(max(1, bh)) <= 1.8:
            continue
        if box[0] <= 2 or box[1] <= 2 or box[2] >= frame.shape[1] - 3:      # cut off: judge it when it is in view
            continue
        if col is None and s < max(conf, 0.75):          # uncoloured: only when the detector is very sure
            continue
        if (want and col != want) or (col in skip) or in_tray(cv2, np, frame, box):
            continue
        if best is None or box[3] > best["box"][3]:
            best = {"box": box, "colour": col, "label": n, "conf": float(s)}
    return best


IN_TRAY = 0.35                                # share of smooth, uncoloured surface around a coloured blob that means "it lies
                                              # in the tray" (photos 2026-11-14: in the tray 0.75-0.82, on the carpet 0.00-0.04)


def colour_name(hue):
    """OpenCV hue (0..180) -> the colour word used in HUES."""
    return next((k for k, v in HUES.items() if any(lo <= hue <= hi for lo, hi in v)), "coloured")


def vivid_blobs(cv2, np, frame):
    """Every vivid thing in the picture that is not lying in the tray: [{"colour", "box", "area"}], biggest first."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
    m = cv2.morphologyEx(vivid_mask(hsv).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if not 40 <= area <= 0.25 * m.size:
            continue
        name = colour_name(float(np.median(hsv[:, :, 0][lab == i])))
        if colour_blob(cv2, np, frame[max(0, y - h): y + 2 * h, max(0, x - w): x + 2 * w], name) is None:
            continue                                      # white all round it: already in the tray
        out.append({"colour": name, "box": (x, y, x + w, y + h), "area": area})
    return sorted(out, key=lambda b: -b["area"])


def nearest_thing(cv2, np, frame, name):
    """The NEAREST vivid thing (lowest in the picture) of the colour named in `name`; no colour word =
    any colour. Things lying in the tray do not count. -> {"box", "colour"} or None.
    "object not red" / "object not red not blue": any colour but those (two cars: the other car has
    claimed the red one, antcolony/duo.py)."""
    low = name.lower()
    skip = [k for k in HUES if "not " + k in low]
    want = next((k for k in HUES if k in low and k not in skip), None)
    best = None
    for b in vivid_blobs(cv2, np, frame):
        if (want and b["colour"] != want) or b["colour"] in skip:
            continue
        if not want and b["colour"] == "orange":          # skin and varnished wood read as vivid orange (chair legs, a
            continue                                      # hand): never "any object" - an orange thing must be asked for by name
        if best is None or b["box"][3] > best["box"][3]:
            best = b
    return best


def white_things(cv2, np, frame, min_area=0.003):
    """Every white, smooth patch: [(x1, y1, x2, y2)] - for following the tray from frame to frame."""
    h, w = frame.shape[:2]
    m = cv2.morphologyEx(white_mask(cv2, np, frame).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m[int(0.86 * h):, int(0.2 * w): int(0.9 * w)] = 0     # what the gripper holds
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    return [(int(x), int(y), int(x + bw), int(y + bh)) for x, y, bw, bh, a in stats[1:] if a >= min_area * h * w]


def overlap(a, b):
    """Intersection over union of two boxes."""
    ix, iy = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / float((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


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
        self.yolo, self.conf, self.classes = None, 0.5, None
        self.min_width_cm = 15.0 if "tray" in str(target).lower() or "box" in str(target).lower() else 0.0
        if detector == "yolo":                            # the team's trained model, as installed for the grab (detector.py)
            try:
                import detector as det
                cfg = det.load_config(os.path.join(HERE, "detector.warehouse.json"))
                self.conf = float(cfg.get("conf", 0.5))
                deb = cfg.get("debris")                   # the whitelist: classes that are debris to clear
                if isinstance(deb, list) and deb:
                    self.classes = {str(c).lower() for c in deb}
                cfg["conf"] = 0.6 * self.conf              # the network's own cut; nearest_object() decides above that
                self.yolo = det.Yolo(cfg)
            except Exception as e:                        # no model on this car: colour, as before
                print(f"  seek: YOLO not available ({e}) - finding objects by colour", flush=True)
                self.detector = "colour"
        self.tokens0 = vlm.model_size.__defaults__
        if img_tokens:                                    # (the default is bound when vlm.py is imported)
            vlm.model_size.__defaults__ = (int(img_tokens),)
        self.cap = open_cam(cv2)
        # "far" = the level view first; if a whole turn shows nothing, the lower view (a thing right
        # in front of the car is below the level view, or hidden behind what the gripper holds)
        self.views = [(VIEW_FAR, 4), (VIEW, 12)] if view == "far" else [(tuple(view), 4)] if view else [(VIEW, 12)]
        self.pan_deg, self.box, self.colour = 0.0, None, None
        self.now = self.views.pop(0)
        self.pose(*self.now)
        self.n = self.skipped = 0
        for f in glob.glob(os.path.join(HERE, "seek_[0-9][0-9].jpg")):
            os.remove(f)
        for _ in range(20):                               # exposure / white balance
            self.cap.read()

    def width_cm(self, box, w, h):
        """Real width (cm) of a thing standing on the floor, from where its lower edge meets the floor
        in this camera view and how wide it looks. None if the view's geometry does not say."""
        d = floor_cm(self.geom, box[3], h)
        if d is None:
            return None
        rng = math.hypot(d - self.geom[0], self.geom[1])        # camera -> that floor point
        return (box[2] - box[0]) * (480.0 / h) / F_PX * rng

    def nearest(self, frame, name):
        """The nearest object to fetch: by the trained detector ("yolo") or by vivid colour ("colour")."""
        if self.detector == "yolo":
            return nearest_object(self.cv2, self.np, frame, name, self.yolo, self.conf, self.classes)
        return nearest_thing(self.cv2, self.np, frame, name)

    def pose(self, view, slack, ms=1500):
        x, y, z, pitch = view
        self.geom = GEOM.get(tuple(view))
        t = math.radians(self.pan_deg)                    # the arm base turned: the camera looks pan_deg to the left
        x, y = x * math.cos(t) - y * math.sin(t), x * math.sin(t) + y * math.cos(t)
        with self.hw.lock:
            ok = self.hw.arm.move_to(x, y, z, pitch=pitch, pitch_min=pitch - slack, pitch_max=pitch + slack, ms=ms)
            if ok is None:                                # out of reach: the driving view is better than none
                print(f"  seek: view {view} out of reach - using the far driving view", flush=True)
                self.geom = None
                self.hw.arm.set_pulses(1500, dict(FAR))
                if hasattr(self.hw.arm, "pos"):
                    self.hw.arm.pos = None
        time.sleep(ms / 1000.0 + 0.4)

    def pan(self, deg):
        """Turn the ARM (not the car) so that the camera looks `deg` to the left (+) / right (-)."""
        if abs(deg - self.pan_deg) > 0.5:
            self.pan_deg = float(deg)
            self.pose(*self.now, ms=700)

    def next_view(self):
        """Switch to the next camera view, if there is one. -> switched?"""
        if not self.views:
            return False
        self.now = self.views.pop(0)
        self.pose(*self.now)
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
        if self.detector in ("colour", "yolo"):
            t = self.nearest(frame, self.target)
            box, self.colour = (t["box"], t["colour"]) if t else (None, self.colour)
        elif "white" in self.target.lower() and not white_things(self.cv2, self.np, frame, min_area=0.0003):
            # (0.0003 = ~90 px, not 0.001: a tray 2 m away in the level view is 90x32 px to the VLM but only
            # ~190 px of it pass the white mask - at 0.001 that photo was skipped and the tray not found.
            # 58 saved photos: time saved 18 of 46 instead of 19, trays lost 0 instead of 1.)
            box = None                                    # nothing white in the photo: no need to ask the VLM (3 s saved per look)
            self.skipped += 1
        else:
            dets = self.vlm.detect_all(frame, self.target, verbose=False)
            if self.min_width_cm and self.geom and abs(self.pan_deg) < 1:
                keep = []
                for d in dets:
                    wcm = self.width_cm(d["box"], w, h)
                    if wcm is None or wcm >= self.min_width_cm:
                        keep.append(d)
                    else:
                        print(f"  seek: '{self.target}' answer only {wcm:.0f} cm wide on the floor - not it "
                              f"(a corner marker cup?)", flush=True)
                dets = keep
            det = max(dets, key=lambda d: (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1])) if dets else None
            box = det["box"] if det else None
        self.box = box
        out = self.describe(box, w, h)
        if box is not None:
            x1, y1, x2, y2 = box
            self.cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        self.n += 1
        self.cv2.imwrite(os.path.join(HERE, "seek.jpg"), frame)
        self.cv2.imwrite(os.path.join(HERE, f"seek_{self.n:02d}.jpg"), frame)       # every look, for afterwards
        return out

    def describe(self, box, w, h):
        if box is None:
            return None
        x1, y1, x2, y2 = box
        fx = (w / 2) / math.tan(math.radians(HFOV_DEG / 2))
        return {"bearing": self.pan_deg + math.degrees(math.atan((w / 2 - (x1 + x2) / 2) / fx)),
                "wide": (x2 - x1) / w, "bottom": y2 / h, "colour": self.colour,
                "dist": floor_cm(self.geom, y2, h) if self.geom and abs(self.pan_deg) < 1 else None}   # its foot on the floor, cm ahead

    def glance(self):
        """A quick look while the car is MOVING: the newest frame, no VLM, nothing saved. The target is the
        thing found by the last see(): a coloured thing is found again by its colour, anything else (the
        tray) as the white patch that overlaps where it was a moment ago. -> like see(), or None = lost."""
        frame = None
        for _ in range(2):                                # the frame in the driver's queue is old
            ok, f = self.cap.read()
            if ok:
                frame = f
        if frame is None or self.box is None:
            return None
        h, w = frame.shape[:2]
        if self.detector in ("colour", "yolo"):
            t = self.nearest(frame, self.colour or self.target)
            box = t["box"] if t else None
        else:
            cands = [(overlap(self.box, b), b) for b in white_things(self.cv2, self.np, frame)]
            cands = [c for c in cands if c[0] > 0.15]
            box = max(cands)[1] if cands else None
        if box is None:
            return None
        self.box = box
        return self.describe(box, w, h)

    def survey(self):
        """One sharp photo, every vivid thing in it: [{"colour", "bearing", "dist", "px"}]. Saved like see()."""
        frame = None
        for attempt in range(4):
            for _ in range(10 if attempt == 0 else 5):
                ok, f = self.cap.read()
                if ok:
                    frame = f
            if frame is None:
                raise RuntimeError("camera: no picture (USB cable?)")
            if self.cv2.Laplacian(self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY), self.cv2.CV_64F).var() >= 200:
                break
            time.sleep(0.4)
        h, w = frame.shape[:2]
        fx, out = (w / 2) / math.tan(math.radians(HFOV_DEG / 2)), []
        for b in vivid_blobs(self.cv2, self.np, frame):
            x1, y1, x2, y2 = b["box"]
            out.append({"colour": b["colour"], "bearing": math.degrees(math.atan((w / 2 - (x1 + x2) / 2) / fx)),
                        "dist": floor_cm(self.geom, y2, h) if self.geom else None, "px": (x2 - x1, y2 - y1)})
            self.cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (255, 255, 255), 1)
            self.cv2.putText(frame, b["colour"], (x1 - 3, max(12, y1 - 6)), self.cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        self.n += 1
        self.cv2.imwrite(os.path.join(HERE, "seek.jpg"), frame)
        self.cv2.imwrite(os.path.join(HERE, f"seek_{self.n:02d}.jpg"), frame)
        return out

    def close(self):
        self.cap.release()
        self.vlm.model_size.__defaults__ = self.tokens0


def run(hw, args, cancel, eye=None, carry_pose=None):
    """-> (ok, detail). `eye` and `carry_pose` are replaced in tests/test_seek.py."""
    target = str(args.get("target", "white box"))
    stop_cm = float(args.get("stop_cm", 22))
    tol = float(args.get("tol_deg", 7))
    # Turning on the spot: nothing happens for the first ~0.12 s of a pulse, then about 250 deg/s (car B on
    # carpet, 2026-11-14: 0.18 s pulses turned 9 deg, 0.27 s pulses 36 deg). The rate is re-measured after every turn.
    rate, dead = float(args.get("turn_deg_s", 250)), float(args.get("turn_dead_s", 0.12))
    w_cmd, u = float(args.get("w", 0.3)), float(args.get("u", 0.05))
    cm_s, step = float(args.get("cm_s", 17)), float(args.get("step_s", 0.33))
    seg, max_cm = float(args.get("seg_cm", 30)), float(args.get("max_cm", 200))
    made = eye is None                                    # the camera first: what it can do decides how to search
    if made:                                              # (2026-11-14: these checks ran on eye=None, so on the car
        eye = Eye(hw, target, args.get("view"), args.get("detector", "vlm"), args.get("img_tokens"))   # neither the
                                                          # arm pan nor the smooth rolling was ever switched on)
        if args.get("classes") and hasattr(eye, "classes"):      # detector "yolo": only these classes are objects to fetch
            eye.classes = {str(c).lower() for c in args["classes"]}
    # Searching: the ARM looks left and right first (the camera rides on it) - the car stands still and
    # three looks cover the 170 deg in front of it; only if nothing is there does the car turn, by a
    # third of a circle. Without an arm that can pan (tests): the car turns 30 deg per look as before.
    pans = [float(v) for v in args.get("pans", (50, -50))] if hasattr(eye, "pan") and args.get("pan", True) else []
    scan_deg = float(args.get("scan_deg", 120 if pans else 30))
    max_scans = int(args.get("max_scans", 3 if pans else 15))
    if args.get("scan") == "right":                      # search turning right instead of left
        scan_deg = -scan_deg
    follow = hasattr(eye, "glance") and args.get("follow", True)
    steer = float(args.get("steer_deg", 150))             # gentle: 15 deg off = a tenth of full turn command; 0 = roll straight
    timeout = float(args.get("timeout", 360))
    say = lambda msg: print(f"  seek: {msg}", flush=True)
    t0, sign, drove, looks, scans, last, had = time.monotonic(), 1.0, 0.0, 0, 0, None, False
    lost_near = None
    heading, lock = [0.0], {"deg": None, "n": 0}          # the target's direction once seen twice alike
    keep_dir = args.get("detector", "vlm") == "vlm" and not args.get("look_only")

    def turn(deg):
        """Turn about `deg` (+ = left) with one timed pulse. -> seconds it lasted."""
        heading[0] += deg                                 # where the car points now, as far as it knows
        t = min(0.6, dead + max(0.02, abs(deg) / rate))
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
        if args.get("survey"):                            # look all round, drive nowhere: what coloured things are there?
            seen = {}
            for k in range(int(args.get("views", 14))):
                if cancel.is_set():
                    return finish(False, "cancelled")
                for b in eye.survey():
                    seen[b["colour"]] = seen.get(b["colour"], 0) + 1
                    say(f"view {k + 1}: {b['colour']} thing {abs(b['bearing']):.0f} deg {'left' if b['bearing'] > 0 else 'right'}, "
                        f"about {b['dist'] if b['dist'] is None else round(b['dist'])} cm, {b['px'][0]}x{b['px'][1]} px")
                turn(scan_deg)
            return finish(True, "survey: " + (", ".join(f"{c} in {n} views" for c, n in seen.items()) or "nothing coloured"))
        while True:
            if cancel.is_set():
                return finish(False, f"cancelled after {drove:.0f} cm")
            if time.monotonic() - t0 > timeout:
                return finish(False, f"timeout after {looks} looks, {drove:.0f} cm")
            s = eye.see()
            looks += 1
            if s is None and had and last is None and not args.get("look_only"):
                had = False                               # it was there a moment ago and the car has not turned:
                say(f"look {looks}: '{target}' not found this time - looking again")      # the VLM misses now and then
                s = eye.see()
                looks += 1
            if s is None and pans and not args.get("look_only"):
                for pan in pans:                          # not straight ahead: look to the sides with the arm
                    eye.pan(pan)
                    s = eye.see()
                    looks += 1
                    if s is not None:
                        break
                eye.pan(0)
            if args.get("look_only"):
                d = hw.sonar_med()
                return finish(s is not None, f"{target}: " + ("not in the picture" if s is None else
                              f"{abs(s['bearing']):.0f} deg {'left' if s['bearing'] > 0 else 'right'}, "
                              f"{100 * s['wide']:.0f} % of the width") + f"; sonar {d if d is None else round(d)} cm")
            if s is not None and keep_dir:               # 2026-11-15, car B: the 'tray' 40 deg left, then 59 deg right,
                wb = heading[0] + s["bearing"]            # then left again - two white things, and the car turned between
                off = abs(((wb - lock["deg"] + 180) % 360) - 180) if lock["deg"] is not None else 0.0   # them for minutes
                if lock["deg"] is None or (lock["n"] < 2 and off > 40):
                    lock["deg"], lock["n"] = wb, 1
                elif off <= 40:
                    lock["deg"], lock["n"] = 0.5 * (lock["deg"] + wb), lock["n"] + 1
                else:
                    say(f"look {looks}: a '{target}' {abs(s['bearing']):.0f} deg {'left' if s['bearing'] > 0 else 'right'} - "
                        f"not where I saw it twice before; ignoring it")
                    s = None
            if s is None and keep_dir and lock["n"] >= 2 and not args.get("look_only"):
                back = ((lock["deg"] - heading[0] + 180) % 360) - 180
                if abs(back) > 15:
                    say(f"look {looks}: turning back to where the {target} was ({abs(back):.0f} deg {'left' if back > 0 else 'right'})")
                    turn(back)
                    lock["n"] = 1                         # it must be seen there again
                    continue
            if s is None and lost_near is not None and not args.get("look_only"):
                back = -0.5 * lost_near
                say(f"look {looks}: lost the {target} right after turning - turning back {abs(back):.0f} deg")
                turn(back)
                lost_near = None
                continue
            if s is None:
                if scans >= max_scans:
                    if hasattr(eye, "next_view") and eye.next_view():
                        say("nothing in a whole turn with this view - once more with the other camera view")
                        scans = 0
                        continue
                    return finish(False, f"'{target}' not found in a full turn ({looks} looks)")
                say(f"look {looks}: '{target}' not in the picture - turning {'left' if scan_deg > 0 else 'right'}")
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
                    rate = max(60.0, min(500.0, 0.5 * rate + 0.5 * abs(did) / max(0.02, secs - dead)))
                    if b * b0 < 0 and abs(b) <= 14:       # the shortest pulse overshoots: this is as good as it gets
                        tol = max(tol, abs(b) + 1)        # (the sonar beam is +-15 deg, and we look again on the way)
                last = None
            d = hw.sonar_med()
            say(f"look {looks}: '{target}' {abs(b):.0f} deg {'left' if b > 0 else 'right'}, "
                f"{100 * s['wide']:.0f} % wide, sonar {d if d is None else round(d)} cm")
            lost_near = None
            if abs(b) > tol:
                near_now = s["wide"] >= 0.35
                ask = (0.5 if near_now else 0.8) * b      # aim short; near, a big turn throws it out of the picture
                last = (ask, turn(ask), b)
                if near_now:
                    lost_near = ask
                continue
            if args.get("stop") == "aim":
                return finish(True, f"facing the {target}: {abs(b):.0f} deg off, {100 * s['wide']:.0f} % wide, "
                                    f"lower edge at {100 * s['bottom']:.0f} % of the picture, {looks} looks")
            bstop = args.get("bottom_stop")               # small target on the floor: close enough when its lower
            if bstop is not None and s["bottom"] >= float(bstop):      # edge is this far down the picture
                return finish(True, f"close to the {target}: lower edge at {100 * s['bottom']:.0f} % of the picture, "
                                    f"{looks} looks, drove {drove:.0f} cm; colour={s.get('colour')}")
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
            # Far enough to roll: ONE smooth move, steering on the way. The target is looked for again in
            # every camera frame (by its colour, or as the white patch the VLM just named); the car stops as
            # soon as it is near, as soon as the target is missing from two frames in a row, or when the sonar
            # sees something. Then a proper look decides what next - never a move without the target in view.
            # (2026-11-14 on car B: a block 1 m away was approached in 8 cm hops because seg_now was capped
            # once it showed in the lower part of the picture. The per-frame check below already stops at
            # the right moment, so roll whenever the target is not yet near.)
            roll = follow and not near_by_eye and (seg_now >= 12
                                                   or (bstop is not None and s["bottom"] < float(bstop) - 0.08)
                                                   or (bstop is None and dist is not None and dist > arrive + 10))
            if roll:
                d = hw.sonar_med()
                if d is not None and d <= stop_cm + 20:
                    say(f"not rolling: the sonar reads {d:.0f} cm - short steps instead")
                if d is None or d > stop_cm + 20:
                    t_f, lost, why = time.monotonic(), 0, "long"
                    hw.set_velocity(u, 0.0, 0.0)
                    while time.monotonic() - t_f < float(args.get("follow_s", 10)):
                        if cancel.is_set():
                            why = "cancel"
                            break
                        g = eye.glance()
                        d = hw.sonar_cm()
                        if d is not None and d <= stop_cm + 12:
                            why = "sonar"
                            break
                        if g is None:
                            lost += 1
                            if lost >= 2:
                                why = "lost"
                                break
                            continue
                        lost = 0
                        if (bstop is not None and g["bottom"] >= float(bstop) - 0.04) or \
                           (bstop is None and ((g["dist"] is not None and g["dist"] <= arrive + 5) or g["wide"] >= 0.7
                                               or (g["bottom"] >= 0.9 and g["wide"] >= 0.4))):
                            why = "near"
                            break
                        if abs(g["bearing"]) > 22:        # drifting off: stop and aim again with a proper look
                            why = "off line"
                            break
                        hw.set_velocity(u, 0.0, sign * max(-0.12, min(0.12, g["bearing"] / steer)) if steer else 0.0)
                    hw.stop()
                    went = max(0.0, 22.0 * (time.monotonic() - t_f - 0.075))
                    time.sleep(0.5)
                    drove += went
                    say(f"rolled {went:.0f} cm following it ({why})")
                    if why == "cancel":
                        return finish(False, f"cancelled after {drove:.0f} cm")
                    if why == "lost" and went < 4:
                        follow = False                    # cannot follow this one: short steps and looks instead
                    last = None
                    continue
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

    def point(self, margin=4.0):
        cv2, frame = self.cv2, self.frame()
        p = drop_point(cv2, self.np, frame, self.H, margin)
        if "x" in p:                                      # draw it: cm -> picture px
            px = cv2.perspectiveTransform(self.np.array([[[p["x"], p["y"]]]], dtype=self.np.float64), self.np.linalg.inv(self.H))[0, 0]
            cv2.circle(frame, (int(px[0]), int(px[1])), 14, (0, 255, 0) if p["ok"] else (0, 165, 255), 3)
            cv2.putText(frame, f"{'drop' if p['ok'] else 'best'} x={p['x']:+.1f} y={p['y']:.1f} cm, {p['margin']:.1f} cm inside",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(HERE, "dock.jpg"), frame)
        return p

    def close(self):
        self.cap.release()


def white_blob(cv2, np, frame):
    """The nearest big white, smooth thing in the lower search view (VIEW) -> None or
    {"off": cm it lies to the LEFT of the car's line (+) / right (-), "dist": cm from the arm base to
    its near edge (21 = at the bottom of the picture or nearer), "area": share of the picture, "box"}."""
    h, w = frame.shape[:2]
    m = white_mask(cv2, np, frame).astype(np.uint8)
    m[int(0.86 * h):, int(0.2 * w): int(0.9 * w)] = 0     # what the gripper holds shows here, and a pale block passes for white
    # join the pieces of one tray first: its rim lines, a shadow or an object lying in it cut the white into
    # parts that are each too small (2026-11-14: three parts of 3.3 / 2.8 / 1.8 % of the picture, none counted;
    # the seams are ~20 px wide: the smoothness test marks a band along every edge as rough)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
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


def drop_point(cv2, np, frame, H, margin=4.0):
    """Where can the arm let go so that the object lands in the tray? Look-down picture -> floor map
    in cm (table calibration) -> the white tray as a region -> a point inside it that is at least
    `margin` cm from everything seen that is not tray, and within the arm's reach from where the car
    stands. The tray is a rectangle and any such point will do: the car is NOT lined up with an edge
    or a corner first (user, 2026-11-14: it kept sliding to one corner although it could reach).
    -> {"ok": True, "x", "y", "margin"}   x cm right of the car's line, y cm ahead of the arm base
       {"ok": False, "x", "y", "margin"}   best point seen, out of reach or too close to the rim: move towards it
       {"ok": False}                       no tray in the picture"""
    S, X0, X1, Y0, Y1 = 4.0, -13.0, 13.0, 8.0, 27.0             # canvas: 4 px per cm
    W, Hc = int((X1 - X0) * S), int((Y1 - Y0) * S)
    A = np.array([[1 / S, 0, X0], [0, -1 / S, Y1], [0, 0, 1]], dtype=np.float64)      # canvas px -> cm
    M = np.linalg.inv(H) @ A                                                         # canvas px -> picture px
    white = white_mask(cv2, np, frame).astype(np.uint8) * 255
    h, w = white.shape
    white[int(0.93 * h):, int(0.2 * w): int(0.9 * w)] = 0       # what the gripper still holds shows here
    flags = cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP
    top = cv2.warpPerspective(white, M, (W, Hc), flags=flags, borderValue=0) > 0
    seen = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (W, Hc), flags=flags, borderValue=0) > 0
    if top.sum() < 6 * S * S:                                   # less than ~6 cm2 of tray
        return {"ok": False}
    # distance (cm) from each point to the nearest thing SEEN that is not tray. Beyond the picture's edge
    # the tray is taken to go on for another 2 cm, no more: a rim may lie just outside the picture.
    dist = cv2.distanceTransform((top | ~seen).astype(np.uint8), cv2.DIST_L2, 5) / S
    dist = np.minimum(dist, cv2.distanceTransform(top.astype(np.uint8), cv2.DIST_L2, 5) / S + 2.0)
    dist[~top] = 0
    ys, xs = np.mgrid[0:Hc, 0:W]
    X, Y = X0 + xs / S, Y1 - ys / S
    r = np.hypot(X, Y)
    can = top & (r >= 12.5) & (r <= 21.0) & (np.abs(np.degrees(np.arctan2(X, Y))) <= 55)
    score = np.minimum(dist, 8.0) - 0.05 * np.abs(X) - 0.03 * np.abs(r - 18.0)
    if can.any() and float(dist[can].max()) >= margin:
        i = int(np.argmax(np.where(can & (dist >= margin), score, -1e9)))
        return {"ok": True, "x": float(X.flat[i]), "y": float(Y.flat[i]), "margin": float(dist.flat[i])}
    i = int(np.argmax(dist))
    return {"ok": False, "x": float(X.flat[i]), "y": float(Y.flat[i]), "margin": float(dist.flat[i])}


def dock(hw, args, cancel, eye=None, carry_pose=None, drop=None):
    """The last ~45 cm to the white tray without ever touching it, then - with {"drop": true} - the
    object goes in. -> (ok, detail).
    1. lower search view: the tray is the nearest big white smooth thing; the car creeps up, at most
       6 cm per fresh photo, until its near edge is about 24 cm ahead; it slides sideways only if the
       tray is more than 8 cm off the car's line. No tray in the photo = no move at all.
    2. calibrated LOOK pose: drop_point() looks for a spot inside the tray that the arm can reach from
       here. Found = done (no lining up with an edge or a corner). Not found = one small move towards
       the best spot seen (forward at most 4 cm, or sideways), and look again. At most 3 moves with
       no tray in view, then it gives up rather than push on blind."""
    cm_s, side_cm_s, u = float(args.get("cm_s", 17)), float(args.get("side_cm_s", 15)), float(args.get("u", 0.05))
    margin = float(args.get("margin_cm", 4.0))
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
        say(f"sliding {cm:.0f} cm to the {'left' if side > 0 else 'right'}")
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
            if abs(b["off"]) > 8.0:                       # the arm reaches 8 cm to either side: only then
                slide(1 if b["off"] > 0 else -1, min(6.0, abs(b["off"]) - 4.0))
            elif b["dist"] <= 25.0:
                break
            else:
                hw.pulse(u, 0.0, max(0.10, min(6.0, b["dist"] - 22.0) / cm_s), cancel)
        else:
            return finish(False, "could not get in front of the tray")

        eye.look()
        blind = 0
        for i in range(int(args.get("max_moves", 14))):
            if cancel.is_set():
                return finish(False, "cancelled")
            p = eye.point(margin)
            if p["ok"]:
                say(f"can reach into the tray from here: {abs(p['x']):.0f} cm {'right' if p['x'] > 0 else 'left'}, "
                    f"{p['y']:.0f} cm ahead, {p['margin']:.0f} cm from the rim")
                if not args.get("drop"):
                    return finish(True, f"at the tray: drop point x={p['x']:+.1f} y={p['y']:.1f} cm ({p['margin']:.0f} cm inside)")
                hw.stop()
                if made:
                    eye.close()
                ok, detail = (drop or hw.drop)({"x_cm": round(p["x"], 1), "y_cm": round(p["y"], 1), "z_cm": float(args.get("z_cm", 10))})
                return ok, (f"in the tray: let go {abs(p['x']):.0f} cm {'right' if p['x'] > 0 else 'left'}, {p['y']:.0f} cm ahead, "
                            f"{p['margin']:.0f} cm inside the rim" if ok else detail)
            if "x" not in p:
                blind += 1
                if blind > 3:
                    return finish(False, "the tray did not come into the look-down picture - stopped")
                say("no tray under the camera yet - forward 4 cm")
                hw.pulse(u, 0.0, 4.0 / cm_s, cancel)
                continue
            say(f"best spot {abs(p['x']):.0f} cm {'right' if p['x'] > 0 else 'left'}, {p['y']:.0f} cm ahead, only {p['margin']:.1f} cm inside")
            r = math.hypot(p["x"], p["y"])
            if abs(p["x"]) > 9.0:                         # beyond the arm's sideways reach
                slide(-1 if p["x"] > 0 else 1, min(5.0, abs(p["x"]) - 5.0))
            elif r > 21.0 or p["margin"] < margin:        # too far, or only the rim is in reach: a little closer
                hw.pulse(u, 0.0, max(0.10, min(4.0, max(1.5, r - 18.0)) / cm_s), cancel)
            else:
                hw.pulse(-u, 0.0, 2.5 / cm_s, cancel)     # under the camera: a little back
        return finish(False, "found no spot in the tray that the arm can reach")
    except Exception as e:
        return finish(False, f"{type(e).__name__}: {e}")
