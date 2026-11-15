#!/usr/bin/env python3
"""
view_overlay.py

What every camera view shows, drawn the same way on the main camera and on each extra camera:

    header bar      camera name (MAIN / EXTRA), floor state, fps, lens state
    arena           the 4 pivots' outline on the floor (lightly filled), corners labelled "pivot 29"
    tags            outline of each tag THIS camera detects; label "36 robot  (0.52, 1.10) m"
                    (the fused floor position), pivots in orange, others in green
    other cameras   tags this camera does not see, located by another camera: dashed ring,
                    "36 robot  via GO 3S"
    A* path         green path on the floor with waypoints, goal star "goal  A* 1.23 m",
                    numbered route stops ("stop 2", grey when done), pins

All world geometry is projected with that camera's own pose and lens (distortion included);
points behind the camera or outside the lens model's valid field are left out, so lines stop
at the image edge instead of wrapping around (important for wide lenses like the GO 3S).
Text size follows the image width, so labels look the same after the view is scaled.
"""
import math

import cv2
import numpy as np

# BGR
PIVOT = (0, 165, 255)
TAG = (60, 220, 60)
ROBOT = (255, 80, 255)
PATH = (0, 255, 0)
STOP = (0, 200, 255)
DONE = (150, 150, 150)
REMOTE = (255, 200, 0)
ARENA = (0, 220, 255)
PIN = (0, 140, 255)


class CameraPose:
    """A snapshot of one camera's floor pose and lens, for drawing (safe to use from another
    thread: it copies everything). scale: the image it draws on is `scale` x the camera frame."""

    def __init__(self, rvec, tvec, K, D, image_size, scale=1.0):
        self.rvec = np.asarray(rvec, np.float64).reshape(3, 1).copy()
        self.tvec = np.asarray(tvec, np.float64).reshape(3, 1).copy()
        self.R, _ = cv2.Rodrigues(self.rvec)
        self.K = np.asarray(K, np.float64).copy()
        self.K[:2] *= scale
        self.D = np.asarray(D, np.float64).ravel().copy()
        self.w, self.h = int(image_size[0] * scale), int(image_size[1] * scale)
        # the widest normalized radius the lens model is valid for: the image corners, +15 %
        c = np.array([[[0, 0]], [[self.w, 0]], [[0, self.h]], [[self.w, self.h]],
                      [[self.w / 2, 0]], [[0, self.h / 2]]], np.float64)
        n = cv2.undistortPoints(c, self.K, self.D).reshape(-1, 2)
        self.r_max = 1.15 * float(np.max(np.linalg.norm(n, axis=1)))

    @classmethod
    def from_floor(cls, floor, K, D, image_size, scale=1.0):
        if floor is None or not floor.valid:
            return None
        return cls(floor.rvec, floor.tvec, K, D, image_size, scale)

    def project(self, pts):
        """World points (N, 3) -> pixels (N, 2) and a validity mask."""
        P = np.asarray(pts, np.float64).reshape(-1, 3)
        Pc = (self.R @ P.T + self.tvec).T
        z = Pc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.hypot(Pc[:, 0], Pc[:, 1]) / z
        ok = (z > 0.05) & (r < self.r_max)
        px, _ = cv2.projectPoints(P, self.rvec, self.tvec, self.K, self.D)
        px = px.reshape(-1, 2)
        ok &= np.isfinite(px).all(1)
        return px, ok


def font_scale(img):
    return 0.5 * img.shape[1] / 960.0


_PLACED = {}      # id(image) -> label rectangles already drawn (set up by draw_view)


def text(img, s, org, color, scale=None, bold=False):
    """Outlined label (readable on any background). Inside draw_view, a label that would overlap
    one already placed moves down a line (up to 3 times), so close tags stay readable."""
    fs = font_scale(img) if scale is None else scale
    th = max(1, int(round(fs * (2.4 if bold else 2.0))))
    x, y = int(org[0]), int(org[1])
    placed = _PLACED.get(id(img))
    if placed is not None:
        (tw, tht), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        step = int(tht * 1.6) + 2
        for _ in range(4):
            r = (x, y - tht - 2, x + tw, y + 4)
            if not any(r[0] < q[2] and q[0] < r[2] and r[1] < q[3] and q[1] < r[3] for q in placed):
                break
            y += step
        placed.append((x, y - tht - 2, x + tw, y + 4))
    # outline = the same text (same thickness: OpenCV's glyph spacing grows with thickness, so a
    # thicker black copy would run longer than the label) shifted around the label
    o = max(1, int(round(fs * 2.4)))
    for dx, dy in ((-o, -o), (o, o), (-o, o), (o, -o)):        # 4 diagonal copies: same outline, half the cost
        cv2.putText(img, s, (x + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fs, color, th, cv2.LINE_AA)


def header(img, title, detail="", color=(255, 255, 255), warn=False):
    """Bar across the top: title (camera name) and a detail line."""
    fs = font_scale(img) * 1.15
    hh = int(34 * fs / 0.5)
    bar = img[:hh].copy()
    bar[:] = (40, 30, 150) if warn else (25, 25, 25)
    cv2.addWeighted(bar, 0.75, img[:hh], 0.25, 0, img[:hh])
    text(img, title, (int(10 * fs / 0.5), int(23 * fs / 0.5)), color, fs, bold=True)
    if detail:
        (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
        text(img, detail, (int(10 * fs / 0.5) + tw + int(16 * fs / 0.5), int(23 * fs / 0.5)),
             (215, 215, 215), fs * 0.82)


def _polyline_runs(img, px, ok, color, thick, closed=False):
    """Draw the valid runs of a sampled world line."""
    run = []
    seq = list(range(len(px))) + ([0] if closed and len(px) else [])
    for i in seq:
        if ok[i]:
            run.append(px[i])
        elif len(run) > 1:
            cv2.polylines(img, [np.round(run).astype(np.int32)], False, color, thick, cv2.LINE_AA)
            run = []
        else:
            run = []
    if len(run) > 1:
        cv2.polylines(img, [np.round(run).astype(np.int32)], False, color, thick, cv2.LINE_AA)


def _sample_edges(corners_xy, n=40, z=0.0, closed=True):
    c = np.asarray(corners_xy, np.float64)
    seg = list(zip(c, np.roll(c, -1, 0))) if closed else list(zip(c[:-1], c[1:]))
    pts = [a + t * (b - a) for a, b in seg for t in np.linspace(0, 1, n, endpoint=False)]
    if not closed:
        pts.append(c[-1])
    return np.column_stack([np.asarray(pts), np.full(len(pts), z)])


def draw_arena(img, pose, corners_xy, ids, outline=True):
    """Outline (and light fill) of the arena between the pivots, corners labelled.
    outline=False: labels only (the main view already draws the floor grid)."""
    if pose is None or corners_xy is None or len(corners_xy) < 3:
        return
    lw = max(1, int(round(2 * font_scale(img) / 0.5)))
    if outline:
        pts = _sample_edges(corners_xy)
        px, ok = pose.project(pts)
        if ok.all():
            over = img.copy()
            cv2.fillPoly(over, [np.round(px).astype(np.int32)], ARENA)
            cv2.addWeighted(over, 0.12, img, 0.88, 0, img)
        _polyline_runs(img, px, ok, (0, 0, 0), lw + 2, closed=True)
        _polyline_runs(img, px, ok, ARENA, lw, closed=True)
    cpx, cok = pose.project(np.column_stack([np.asarray(corners_xy, np.float64), np.zeros(len(corners_xy))]))
    for (u, v), good, tid in zip(cpx, cok, ids):
        if good and 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
            text(img, f"pivot {tid}", (u + 8, v + 22 * font_scale(img) / 0.5), PIVOT)


def draw_tags(img, dets, scale, objects, pivot_ids, robot_id=None):
    """Tags this camera detects. dets: [(id, corners in camera-frame pixels)]; objects: {id: fused
    object} for the label (name and floor position)."""
    lw = max(1, int(round(2 * font_scale(img) / 0.5)))
    for tid, c in dets:
        q = np.round(np.asarray(c, np.float64) * scale).astype(np.int32)
        piv = tid in pivot_ids
        o = objects.get(tid)
        col = PIVOT if piv else ROBOT if tid == robot_id else TAG
        cv2.polylines(img, [q], True, (0, 0, 0), lw + 2, cv2.LINE_AA)
        cv2.polylines(img, [q], True, col, lw, cv2.LINE_AA)
        cv2.circle(img, tuple(q[0]), lw + 2, col, -1)          # first corner = tag's top-left
        top = q[np.argmin(q[:, 1])]
        if piv:
            continue                                            # named by draw_arena (no clutter)
        label = f"{tid}" + (f" {o['name']}" if o and o.get("name") and o["name"] != f"tag{tid}" else "")
        if o and o.get("frame") == "world":
            label += f"  ({o['x']:.2f}, {o['y']:.2f}) m"
        text(img, label, (top[0] - 10, top[1] - 8 * font_scale(img) / 0.5), col)


def draw_remote_tags(img, pose, objects, own_ids, this_name):
    """Tags this camera does NOT see, located by another camera (fused)."""
    if pose is None:
        return
    lw = max(1, int(round(2 * font_scale(img) / 0.5)))
    rad = int(16 * font_scale(img) / 0.5)
    for o in objects.values():
        if o["id"] in own_ids or o.get("frame") != "world" or o["id"] >= 1000:
            continue
        cams = [c for c in o.get("cams", []) if c != this_name]
        if not cams:
            continue
        (u, v), ok = (lambda p: (p[0][0], p[1][0]))(pose.project([[o["x"], o["y"], o.get("z", 0.0)]]))
        if not ok or not (0 <= u < img.shape[1] and 0 <= v < img.shape[0]):
            continue
        for a in range(0, 360, 30):                             # dashed ring
            cv2.ellipse(img, (int(u), int(v)), (rad, rad), 0, a, a + 18, REMOTE, lw, cv2.LINE_AA)
        name = o.get("name") if o.get("name") and o["name"] != f"tag{o['id']}" else ""
        label = f"{o['id']} {name}".strip() + f"  via {'+'.join(cams)}"
        text(img, label, (u + rad + 4, v - rad * 0.6), REMOTE)


def draw_plan(img, pose, plan, route=(), pins=()):
    """A* path (on the floor), waypoints, goal, route stops and pins."""
    if pose is None:
        return
    fs = font_scale(img)
    k = fs / 0.5
    if plan and plan.get("status") == "ok" and len(plan.get("waypoints") or []) >= 2:
        wps = np.asarray(plan["waypoints"], np.float64)
        px, ok = pose.project(_sample_edges(wps, n=16, closed=False))
        _polyline_runs(img, px, ok, (0, 0, 0), int(7 * k), closed=False)
        _polyline_runs(img, px, ok, PATH, max(2, int(4 * k)), closed=False)
        wpx, wok = pose.project(np.column_stack([wps, np.zeros(len(wps))]))
        for (u, v), good in zip(wpx[1:-1], wok[1:-1]):
            if good:
                cv2.circle(img, (int(u), int(v)), int(5 * k), (0, 120, 0), -1, cv2.LINE_AA)
        if wok[-1]:
            g = (int(wpx[-1][0]), int(wpx[-1][1]))
            cv2.drawMarker(img, g, (0, 0, 0), cv2.MARKER_STAR, int(28 * k), int(5 * k))
            cv2.drawMarker(img, g, PATH, cv2.MARKER_STAR, int(26 * k), max(2, int(3 * k)))
            # below the star: the goal tag's own label sits above its outline
            text(img, f"goal  A* {plan.get('length_m', 0):.2f} m", (g[0] + 16 * k, g[1] + 30 * k), PATH, bold=True)
        if wok[0]:
            s = (int(wpx[0][0]), int(wpx[0][1]))
            text(img, "start", (s[0] + 12 * k, s[1] + 22 * k), PATH)
    for n_, xy, done in route:
        (u, v), ok = (lambda p: (p[0][0], p[1][0]))(pose.project([[xy[0], xy[1], 0.0]]))
        if ok:
            col = DONE if done else STOP
            cv2.circle(img, (int(u), int(v)), int(14 * k), (0, 0, 0), int(5 * k), cv2.LINE_AA)
            cv2.circle(img, (int(u), int(v)), int(14 * k), col, max(2, int(3 * k)), cv2.LINE_AA)
            text(img, f"stop {n_}" + (" (done)" if done else ""), (u + 16 * k, v - 12 * k), col)
    for p in pins:
        (u, v), ok = (lambda q: (q[0][0], q[1][0]))(pose.project([p]))
        if ok:
            cv2.drawMarker(img, (int(u), int(v)), PIN, cv2.MARKER_CROSS, int(22 * k), max(2, int(2 * k)))


KNOWN = (0, 165, 255)       # one of the fine-tuned parts (trained detector): either camera
UNKNOWN = (120, 120, 120)   # anything else (appearance blob, prompt-free box): faded grey, obstacle only


def is_known(o):
    """One of the fine-tuned parts? Older results without the flag: a detector box counts as known."""
    return bool(o.get("known", o.get("method") == "detect"))


def draw_floor_objects(img, pose, floor_objects):
    """Objects without tags (from the cumulative map) outlined on the floor in this view:
    orange + label = known part, thin grey without label = unknown thing (de-emphasised)."""
    if pose is None:
        return
    lw = max(1, int(round(2 * font_scale(img) / 0.5)))
    for o in floor_objects:
        fp = o.get("footprint_poly")
        if fp is None:
            continue
        known = is_known(o)
        col = KNOWN if known else UNKNOWN
        pts = _sample_edges(np.asarray(fp, np.float64), n=8)
        px, ok = pose.project(pts)
        if not known:
            _polyline_runs(img, px, ok, col, 1, closed=True)
            continue
        _polyline_runs(img, px, ok, (0, 0, 0), lw + 2, closed=True)
        _polyline_runs(img, px, ok, col, lw, closed=True)
        if ok.any():
            top = px[ok][np.argmin(px[ok][:, 1])]
            text(img, f"{o['id']} {o.get('name') or 'object'}" if known else f"{o['id']} unknown",
                 (top[0], top[1] - 6 * font_scale(img) / 0.5), col)


def draw_view(img, pose, scale, dets, objects, own_name, pivot_ids, arena_corners, arena_ids,
              plan=None, route=(), pins=(), robot_id=None, arena_outline=True, floor_objects=()):
    """Everything except the header, in drawing order (floor first, labels last)."""
    _PLACED[id(img)] = []
    try:
        draw_arena(img, pose, arena_corners, arena_ids, arena_outline)
        draw_floor_objects(img, pose, floor_objects)
        draw_tags(img, dets, scale, objects, pivot_ids, robot_id)   # tag labels get the first spots
        draw_plan(img, pose, plan, route, pins)
        draw_remote_tags(img, pose, objects, {t for t, _ in dets}, own_name)
    finally:
        _PLACED.pop(id(img), None)
