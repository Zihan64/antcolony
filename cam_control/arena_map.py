#!/usr/bin/env python3
"""
arena_map.py

Precise floor map between four corner AprilTags, for path planning (A*) and robot control.
Uses apriltag_floor_mapper.py (same folder) for detection and pose.

Layout (seen from above, meters, right-handed, z up)

  Put the 4 pivot tags flat at the corners in ANY arrangement. The pivots are always the 4
  largest tags on the print sheet (29, 30, 31, 32 = 112 mm); any number of other tags may be in
  view during setup, they are ignored for the plane.
  The first id is the origin. Geometry decides the rest: the farthest tag is the diagonal,
  the two neighbors become +x and +y so that z points up. Example from a real floor:

      31 ---------------- 29 (origin)          arena: origin = 29
       |                   |                          +x -> 31   (tape 29->31)
       |                   |                          +y -> 30   (tape 29->30)
      32 ---------------- 30                          diagonal = 32
              (camera)

  Measure the two center-to-center distances origin->+x and origin->+y with a tape.
  The tags may be rotated in 90 degree steps.

Why it is precise
  * All 16 corners of the 4 tags, spread across the whole view, define the plane.
  * With the tape distances the scale comes from your tape, not the printed tag size, and
    without a calibration file the focal length is estimated from the taped arena itself.
  * Objects lying on the floor (or at a known height) are located by intersecting camera
    rays with the plane: precision is set by pixels (~1-3 mm), not by tag size.
  * For a fixed camera the pose is averaged over ~2 s of frames and then locked.
  * The camera image is resampled onto a metric floor grid (lens distortion included),
    so every map pixel has an exact (x, y).

Mapping beyond the tags, adaptively (GlobalMap)
  * The plane is infinite: everything the camera sees out to max_range_m is mapped, not just
    the area between the corner tags. Precision falls with distance; every cell records how
    many mm one camera pixel covered there (mm_per_camera_px.npy).
  * The map lives in fixed arena coordinates and grows (0.5 m steps, up to max_extent_m)
    when new floor comes into view; cells out of view keep their last state.
  * Obstacles accumulate evidence over frames (log-odds), so noise does not flip cells and a
    removed object clears after a few frames. The empty-floor reference adapts to slow
    lighting changes on free cells, and newly seen floor gets its own reference.
  * The floor pose locks when the camera is still and re-locks after it is moved (fusion
    pauses while it moves). Only the 4 corner (pivot) tags define the plane: other tags are
    always objects, so moving one can never tilt or shift the floor.

Pipeline per frame
  camera frame -> fused into the global floor map (color, reference, obstacle evidence)
               -> + footprints of tagged objects, - the robot itself
               -> occupancy grid -> A* (cell size grows with the map to stay fast),
                  inflated by robot radius, unseen floor blocked
               -> smoothed waypoints in arena meters (and a pure pursuit helper)

CLI
  python arena_map.py setup --tape 31=1.20,30=0.80        (no calibration needed)
      (survey the 4 corners from the live camera, write arena.json; then run the GUI)

Library
  import arena_map as am
  arena = am.load_arena("arena.json")
  amap = am.ArenaMap(arena["arena"], arena["floor_tags"])
  path = am.plan_path(occupancy, amap, (x0, y0), (x1, y1))
"""

import argparse
import heapq
import json
import math
import os
import time

import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import apriltag_floor_mapper as m  # noqa: E402

DEFAULT_ARENA_FILE = "arena.json"

ARENA_DEFAULTS = {
    "corner_ids": [29, 30, 31, 32],   # origin first; the rest in any order (geometry decides)
    "tape_m": None,                   # {"<neighbor id>": tape distance from the origin tag, m}
    "width_m": None,                  # older form: distance origin -> +x neighbor
    "height_m": None,                 # older form: distance origin -> +y neighbor
    "focal_px": None,                 # focal length estimated from the arena (if no calibration)
    "positions": None,                # or exact centers {"29": [0, 0], "30": [1.5, 0.02], ...}
    "margin_m": 0.0,                  # grow (+) or shrink (-) the map beyond the tag centers
    "max_range_m": 4.0,               # map floor up to this far from the camera (horizontal)
    "max_extent_m": 6.0,              # the global map may grow this far beyond the arena
    "min_px_per_cell": 0.35,          # skip cells the camera sees too obliquely (too coarse)
    "unknown_blocked": True,          # A* treats never-seen floor as blocked
    "map_resolution_m": None,         # None = auto (matched to the camera's floor detail)
    "plan_resolution_m": 0.01,        # A* grid: 1 cm cells
    "robot_id": None,                 # tag on the robot (global id), excluded from obstacles
    "robot_radius_m": 0.10,           # obstacles are inflated by this for planning
    "obstacle_threshold": 30,         # color difference vs empty floor that counts as obstacle
    "min_obstacle_area_m2": 0.0004,   # ignore blobs smaller than 2 x 2 cm
    "appearance_z": 5.0,              # object = this many robust std devs unlike the floor
}


def arena_settings(cfg):
    """Arena block with defaults. The corner (pivot) tags are always the 4 largest tags on the
    print sheet; a saved order is kept only if it names exactly those 4 (the order picks the
    origin tag)."""
    a = dict(ARENA_DEFAULTS)
    a.update((cfg or {}).get("arena", {}) or {})
    a.pop("learn_anchors", None)          # removed: only the pivots may define the plane
    saved = [int(i) for i in a.get("corner_ids") or []]
    piv = m.corner_tag_ids((cfg or {}).get("tag_sizes") or {})
    a["corner_ids"] = saved if (piv is None and len(saved) == 4) or (piv and sorted(saved) == piv) \
        else (piv or [int(i) for i in ARENA_DEFAULTS["corner_ids"]])
    if a.get("robot_id") is not None:
        a["robot_id"] = int(a["robot_id"])
    return a


def _wrap(deg):
    return (deg + 180.0) % 360.0 - 180.0


# --------------------------------------------------------------------------
# Setup: corner tags -> arena frame
# --------------------------------------------------------------------------

def setup_arena(samples, K, D, size_of, arena):
    """Build floor_tags (arena frame) for the 4 corner tags from a static camera.

    The corner order does not matter except for the first id, which is the origin.
    Geometry decides the rest: the tag farthest from the origin is the diagonal, the other
    two are its neighbors, and +x / +y go to the neighbors so that z points up (right-handed).
    If the second listed id is a neighbor, +x prefers it.

    Tape distances (most precise): arena["tape_m"] = {"<neighbor id>": meters from origin}.
    Older width_m / height_m are applied to the +x / +y neighbors.
    Returns (floor_tags, report). report["layout"] = {origin, x, y, opposite}."""
    ids_in = [int(i) for i in arena["corner_ids"]]
    if len(ids_in) != 4:
        raise ValueError("corner_ids needs exactly 4 tag ids")
    origin = ids_in[0]
    ft, rep = m.survey_floor(samples, K, D, size_of, set(ids_in), origin_id=origin)
    missing = [i for i in ids_in if i not in ft]
    if missing:
        raise RuntimeError(f"Corner tag(s) {missing} not seen steadily. All four must be visible.")
    warnings = list(rep["warnings"])
    c = {i: np.array([ft[i]["x"], ft[i]["y"]]) for i in ids_in}

    others = ids_in[1:]
    opp = max(others, key=lambda i: np.linalg.norm(c[i] - c[origin]))
    nbrs = [i for i in others if i != opp]
    xi = ids_in[1] if ids_in[1] in nbrs else nbrs[0]
    yi = nbrs[1] if nbrs[0] == xi else nbrs[0]
    xa = (c[xi] - c[origin]) / np.linalg.norm(c[xi] - c[origin])
    ya = np.array([-xa[1], xa[0]])
    if (c[yi] - c[origin]) @ ya < 0:         # keep z up: +y must be counter-clockwise of +x
        xi, yi = yi, xi
        xa = (c[xi] - c[origin]) / np.linalg.norm(c[xi] - c[origin])
        ya = np.array([-xa[1], xa[0]])
    ids = [origin, xi, opp, yi]              # counter-clockwise from above
    layout = {"origin": origin, "x": xi, "y": yi, "opposite": opp}

    base = math.degrees(math.atan2(xa[1], xa[0]))
    surveyed = {}
    for i in ids:
        d = c[i] - c[origin]
        surveyed[i] = (float(d @ xa), float(d @ ya), _wrap(ft[i]["yaw_deg"] - base))

    tape = {int(k): float(v) for k, v in (arena.get("tape_m") or {}).items() if v}
    if not tape and arena.get("width_m") and arena.get("height_m"):
        tape = {xi: float(arena["width_m"]), yi: float(arena["height_m"])}
    W, H = tape.get(xi), tape.get(yi)
    if arena.get("positions"):
        target = {int(k): (float(v[0]), float(v[1])) for k, v in arena["positions"].items()}
        source = "measured positions"
    elif W and H:
        target = {origin: (0.0, 0.0), xi: (W, 0.0), opp: (W, H), yi: (0.0, H)}
        source = f"tape: {origin}->{xi} = {W} m, {origin}->{yi} = {H} m"
    else:
        target = {i: surveyed[i][:2] for i in ids}
        source = ("camera only (scale from printed tag size). Enter tape distances "
                  f"{origin}->{xi} and {origin}->{yi} for a precise plane")

    floor_tags, per = {}, {}
    for i in ids:
        yaw = surveyed[i][2]
        snapped = _wrap(round(yaw / 90.0) * 90.0)
        if abs(_wrap(yaw - snapped)) <= 12.0:
            yaw_use = snapped
        else:
            yaw_use = yaw
            warnings.append(f"Tag {i} is rotated {yaw:.1f} deg, not square to the arena edges. "
                            "Using the measured angle; squaring it improves precision.")
        tx, ty = target[i]
        floor_tags[i] = {"x": round(tx, 5), "y": round(ty, 5), "yaw_deg": round(yaw_use, 3)}
        per[i] = {"surveyed_xy_m": [round(surveyed[i][0], 4), round(surveyed[i][1], 4)],
                  "used_xy_m": [round(tx, 4), round(ty, 4)],
                  "survey_vs_used_mm": round(math.hypot(surveyed[i][0] - tx, surveyed[i][1] - ty) * 1e3, 1),
                  "yaw_surveyed_deg": round(yaw, 2), "yaw_used_deg": round(yaw_use, 2)}

    sw, sh = math.hypot(*surveyed[xi][:2]), math.hypot(*surveyed[yi][:2])
    report = {"source": source, "layout": layout, "corner_ids": ids, "plane_rms_mm": rep["plane_rms_mm"],
              "camera_height_m": rep["camera_height_m"], "tags": per, "warnings": warnings,
              "camera_distance_m": {str(xi): round(sw, 4), str(yi): round(sh, 4)},
              "tape_m": {str(k): v for k, v in tape.items()}}
    if W and H and not arena.get("positions"):
        err_w, err_h = (sw - W) / W * 100, (sh - H) / H * 100
        report["scale_check_pct"] = [round(err_w, 2), round(err_h, 2)]
        if max(abs(err_w), abs(err_h)) > 1.5:
            warnings.append(f"Camera vs tape differs by {err_w:+.1f}% ({origin}->{xi}) and {err_h:+.1f}% "
                            f"({origin}->{yi}). Without a calibration file use Estimate focal; "
                            "otherwise check the tape numbers and print_scale.")
        p = {i: np.array(surveyed[i][:2]) for i in ids}
        d1, d2 = np.linalg.norm(p[opp] - p[origin]), np.linalg.norm(p[yi] - p[xi])
        if abs(d1 - d2) / max(d1, d2) > 0.02:
            warnings.append(f"Diagonals seen by the camera differ by {abs(d1 - d2) * 1e3:.0f} mm: the corners "
                            "may not form a rectangle. Measure both diagonals; if they differ, enter exact "
                            "\"positions\" in arena.json.")
    return floor_tags, report


def estimate_focal_from_tags(samples, ids, K, size_of, image_size, min_frames=3):
    """Rough focal length from the tag squares alone (no tape needed): each tag is a square of
    known size seen at an angle, which constrains the focal length. About +/-5%; the taped
    arena (estimate_focal) is much better. Returns (K_new, rms_px, f_px)."""
    acc = {}
    for dets in samples:
        for t, c in dets:
            if t in ids:
                acc.setdefault(t, []).append(np.asarray(c, np.float64))
    objs, imgs = [], []
    for t, v in acc.items():
        if len(v) >= min_frames:
            objs.append(m.tag_corners_local(size_of(t)).astype(np.float32))
            imgs.append(np.mean(v, 0).astype(np.float32))
    if len(objs) < 2:
        raise RuntimeError("Need at least 2 steady tags to estimate the focal length.")
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO |
             cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
    rms, K2, _, _, _ = cv2.calibrateCamera(objs, imgs, tuple(image_size), K.astype(np.float64).copy(),
                                           np.zeros(5), flags=flags)
    f = float(K2[0, 0])
    if not (0.25 * K[0, 0] < f < 4 * K[0, 0]):
        raise RuntimeError(f"Focal estimate {f:.0f}px is implausible.")
    return K2, float(rms), f


def estimate_focal(samples, floor_tags, size_of, ids, K, image_size, min_frames=3):
    """Focal length from the arena itself (no chessboard): the 16 tag corners have known
    floor positions once the tape distances are in, and one oblique view of a known plane
    fixes the focal length. Principal point stays at the image center, no distortion.
    Returns (K_new, rms_px, f_px)."""
    model = m.FloorModel({i: floor_tags[i] for i in ids}, size_of)
    acc = {}
    for dets in samples:
        for tid, cc in dets:
            if tid in model.world_corners:
                acc.setdefault(tid, []).append(np.asarray(cc, np.float64))
    use = [t for t in ids if len(acc.get(t, [])) >= min_frames]
    if len(use) < 4:
        raise RuntimeError("All 4 corner tags are needed to estimate the focal length.")
    obj = np.concatenate([model.world_corners[t] for t in use]).astype(np.float32)
    img = np.concatenate([np.mean(acc[t], 0) for t in use]).astype(np.float32)
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO |
             cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
    rms, K2, _, _, _ = cv2.calibrateCamera([obj], [img], tuple(image_size), K.astype(np.float64).copy(),
                                           np.zeros(5), flags=flags)
    f = float(K2[0, 0])
    if not (0.25 * K[0, 0] < f < 4 * K[0, 0]):
        raise RuntimeError(f"Focal estimate {f:.0f}px is implausible; the view may be too straight-on.")
    return K2, float(rms), f


# --------------------------------------------------------------------------
# Joint solve: 4 tags on one plane (z = 0)
# --------------------------------------------------------------------------

def _avg_corners(samples, ids, min_frames=3):
    acc = {}
    for dets in samples:
        for t, c in dets:
            if t in ids:
                acc.setdefault(t, []).append(np.asarray(c, np.float64))
    return {t: np.mean(v, 0) for t, v in acc.items() if len(v) >= min_frames}


def _tag_world(size, x, y, yaw):
    c, s_ = math.cos(yaw), math.sin(yaw)
    loc = m.tag_corners_local(size)[:, :2]
    return np.column_stack([loc[:, 0] * c - loc[:, 1] * s_ + x, loc[:, 0] * s_ + loc[:, 1] * c + y, np.zeros(4)])


def _lm(fun, p0, iters=60, eps=1e-6):
    """Small Levenberg-Marquardt with a numeric Jacobian (no scipy needed)."""
    p = np.asarray(p0, np.float64).copy()
    r = fun(p)
    cost = r @ r
    lam = 1e-3
    for _ in range(iters):
        J = np.empty((r.size, p.size))
        for j in range(p.size):
            h = eps * max(1.0, abs(p[j]))
            dp = p.copy()
            dp[j] += h
            J[:, j] = (fun(dp) - r) / h
        A, g = J.T @ J, J.T @ r
        improved = False
        for _ in range(10):
            step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -g)
            r2 = fun(p + step)
            c2 = r2 @ r2
            if c2 < cost:
                p, r, cost = p + step, r2, c2
                lam = max(lam / 3, 1e-9)
                improved = True
                break
            lam *= 4
        if not improved or np.linalg.norm(step) < 1e-10 * (1 + np.linalg.norm(p)):
            break
    return p, float(np.sqrt(cost / (r.size / 2)))   # RMS per corner (px)


def solve_plane(corners, ids, size_of, K, D, estimate_f, centers=None, init=None, estimate_k1=False):
    """Joint fit of the 4 starter tags lying on the plane z = 0.

    Unknowns: camera pose (6), focal length (if estimate_f), and per tag its in-plane pose.
      centers=None: origin tag fixed at (0, 0, yaw 0); others free (x, y, yaw).
                    Scale comes from the printed tag sizes.
      centers={id: (x, y)}: tape-measured centers fixed; only each tag's yaw is free.
    init: (rvec, tvec, {id: (x, y, yaw)}) starting point.
    Returns (rvec, tvec, K, poses {id: (x, y, yaw)}, rms_px)."""
    rvec0, tvec0, poses0 = init
    origin = ids[0]
    free = [t for t in ids if t != origin] if centers is None else list(ids)
    img = np.concatenate([corners[t] for t in ids])

    D = np.zeros(5) if D is None else np.asarray(D, np.float64).ravel()

    def unpack(p):
        rv, tv = p[0:3], p[3:6]
        k = 6
        f = K[0, 0]
        if estimate_f:
            f = p[6]
            k = 7
        Dk = D.copy()
        if estimate_k1:
            Dk[0] = p[k]
            k += 1
        poses = {}
        for t in ids:
            if centers is None:
                if t == origin:
                    poses[t] = (0.0, 0.0, 0.0)
                else:
                    poses[t] = tuple(p[k:k + 3])
                    k += 3
            else:
                poses[t] = (centers[t][0], centers[t][1], p[k])
                k += 1
        Kf = K.copy()
        Kf[0, 0] = Kf[1, 1] = f
        return rv, tv, Kf, poses, Dk

    def resid(p):
        rv, tv, Kf, poses, Dk = unpack(p)
        obj = np.concatenate([_tag_world(size_of(t), *poses[t]) for t in ids])
        proj, _ = cv2.projectPoints(obj, rv, tv, Kf, Dk)
        return (proj.reshape(-1, 2) - img).ravel()

    p0 = list(np.ravel(rvec0)) + list(np.ravel(tvec0))
    if estimate_f:
        p0.append(K[0, 0])
    if estimate_k1:
        p0.append(D[0])
    for t in free:
        x, y, yaw = poses0[t]
        p0 += [x, y, yaw] if centers is None else [yaw]
    p, rms = _lm(resid, p0)
    rv, tv, Kf, poses, Dk = unpack(p)
    if estimate_k1:
        return rv.reshape(3, 1), tv.reshape(3, 1), Kf, poses, rms, Dk
    return rv.reshape(3, 1), tv.reshape(3, 1), Kf, poses, rms


def build_arena(samples, K, D, size_of, arena, estimate_f=False):
    """Build the arena from the 4 starter tags, assumed to lie on one plane at z = 0.

    1. average each tag's corners over the samples;
    2. joint fit (solve_plane): camera pose + every tag's (x, y, yaw) on z = 0 (+ focal length
       when there is no calibration); scale from the printed tag sizes;
    3. layout: first id = origin, farthest tag = diagonal, neighbors = +x / +y (z up);
    4. with tape distances: centers fixed to the measured rectangle, fit again.
    Positions are always self-consistent with the camera model they were fitted with, so the
    per-frame floor fit cannot drift away from them.
    Without a calibration file (estimate_f=True) the focal length is fitted too, and once the
    tape fixes the centers also the main lens distortion term k1.
    Returns (floor_tags, report, K_used, D_used)."""
    ids_in = [int(i) for i in arena["corner_ids"]]
    if len(ids_in) != 4:
        raise ValueError("corner_ids needs exactly 4 tag ids")
    corners = _avg_corners(samples, set(ids_in))
    missing = [i for i in ids_in if i not in corners]
    if missing:
        raise RuntimeError(f"Corner tag(s) {missing} not seen steadily. All four must be visible.")
    origin = ids_in[0]
    warnings = []

    # initial guess: each tag's own pose relative to the origin tag
    K = np.asarray(K, np.float64)
    cand = {t: m.solve_tag_pose(corners[t], size_of(t), K, D) for t in ids_in}
    if any(v is None for v in cand.values()):
        raise RuntimeError("Could not solve a corner tag's pose.")
    R0, t0, _ = cand[origin]
    init_poses = {}
    for t in ids_in:
        R, tt, _ = cand[t]
        p = R0.T @ (tt - t0)
        xa = R0.T @ R[:, 0]
        init_poses[t] = (float(p[0]), float(p[1]), math.atan2(xa[1], xa[0]))
    rv0, _ = cv2.Rodrigues(R0)
    rv, tv, Kf, poses, rms = solve_plane(corners, ids_in, size_of, K, D, estimate_f,
                                         init=(rv0, t0.reshape(3, 1), init_poses))
    if rms > 3.0 and estimate_f:
        warnings.append(f"Plane fit {rms:.1f} px RMS is high: strong lens distortion? A chessboard "
                        "calibration (calibrate) will help.")

    # layout in the plane
    c = {t: np.array(poses[t][:2]) for t in ids_in}
    others = ids_in[1:]
    opp = max(others, key=lambda i: np.linalg.norm(c[i] - c[origin]))
    nbrs = [i for i in others if i != opp]
    xi = ids_in[1] if ids_in[1] in nbrs else nbrs[0]
    yi = nbrs[1] if nbrs[0] == xi else nbrs[0]
    xa = (c[xi] - c[origin]) / np.linalg.norm(c[xi] - c[origin])
    ya = np.array([-xa[1], xa[0]])
    if (c[yi] - c[origin]) @ ya < 0:
        xi, yi = yi, xi
        xa = (c[xi] - c[origin]) / np.linalg.norm(c[xi] - c[origin])
        ya = np.array([-xa[1], xa[0]])
    ids = [origin, xi, opp, yi]
    layout = {"origin": origin, "x": xi, "y": yi, "opposite": opp}
    th = math.atan2(xa[1], xa[0])
    # tag-frame -> arena-frame (rotate by -th about the origin tag center, z stays 0)
    Rz = np.array([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1.0]])
    Rcam, _ = cv2.Rodrigues(rv)
    Rcam_a = Rcam @ Rz                      # x_cam = Rcam (Rz x_arena) + tv
    rv_a, _ = cv2.Rodrigues(Rcam_a)
    cam_poses = {t: (float((c[t] @ xa)), float(c[t] @ ya), _wrap(math.degrees(poses[t][2] - th)))
                 for t in ids}

    tape = {int(k): float(v) for k, v in (arena.get("tape_m") or {}).items() if v}
    if not tape and arena.get("width_m") and arena.get("height_m"):
        tape = {xi: float(arena["width_m"]), yi: float(arena["height_m"])}
    W, H = tape.get(xi), tape.get(yi)
    sw, sh = math.hypot(*cam_poses[xi][:2]), math.hypot(*cam_poses[yi][:2])
    report = {"layout": layout, "corner_ids": ids, "camera_distance_m": {str(xi): round(sw, 4),
              str(yi): round(sh, 4)}, "tape_m": {str(k): v for k, v in tape.items()},
              "fit_px_free": round(rms, 3), "focal_free_px": round(float(Kf[0, 0]), 2)}
    final = {t: (cam_poses[t][0], cam_poses[t][1], math.radians(cam_poses[t][2])) for t in ids}
    Kuse, rv_use, tv_use, fit = Kf, rv_a, tv, rms
    if arena.get("positions") or (W and H):
        if arena.get("positions"):
            centers = {int(k): (float(v[0]), float(v[1])) for k, v in arena["positions"].items()}
            report["source"] = "measured positions"
        else:
            centers = {origin: (0.0, 0.0), xi: (W, 0.0), opp: (W, H), yi: (0.0, H)}
            report["source"] = f"tape: {origin}->{xi} = {W} m, {origin}->{yi} = {H} m"
            report["scale_check_pct"] = [round((sw - W) / W * 100, 2), round((sh - H) / H * 100, 2)]
        out = solve_plane(corners, ids, size_of, Kf, D, estimate_f, centers=centers,
                          init=(rv_a, tv, final), estimate_k1=estimate_f)
        if estimate_f:
            rv_use, tv_use, Kuse, final, fit, D = out
            if abs(D[0]) > 0.4:      # implausible: fall back to no distortion
                D = np.zeros(5)
                rv_use, tv_use, Kuse, final, fit = solve_plane(corners, ids, size_of, Kf, D, estimate_f,
                                                               centers=centers, init=(rv_a, tv, final))
        else:
            rv_use, tv_use, Kuse, final, fit = out
        if fit > 2.0:
            warnings.append(f"With the tape distances the tags fit to {fit:.1f} px RMS only: check the "
                            "numbers (center to center), that the tags lie flat, and the print scale.")
    else:
        report["source"] = ("printed tag sizes (no tape yet): enter tape distances "
                            f"{origin}->{xi} and {origin}->{yi} for a precise scale")
    floor_tags, per = {}, {}
    for t in ids:
        x, y, yaw = final[t]
        floor_tags[t] = {"x": round(float(x), 5), "y": round(float(y), 5), "z": 0.0,
                         "yaw_deg": round(_wrap(math.degrees(yaw)), 3)}
        per[t] = {"xy_m": [round(float(x), 4), round(float(y), 4)],
                  "camera_saw_xy_m": [round(cam_poses[t][0], 4), round(cam_poses[t][1], 4)],
                  "yaw_deg": round(_wrap(math.degrees(yaw)), 2)}
    Rc, _ = cv2.Rodrigues(rv_use)
    report.update({"tags": per, "warnings": warnings, "fit_px": round(fit, 3),
                   "focal_px": round(float(Kuse[0, 0]), 2), "focal_estimated": bool(estimate_f),
                   "k1": round(float(np.ravel(D)[0]) if D is not None and np.size(D) else 0.0, 5),
                   "camera_height_m": round(float((-Rc.T @ tv_use.reshape(3))[2]), 4),
                   "plane_rms_mm": 0.0})
    return floor_tags, report, Kuse, (np.zeros(5) if D is None else np.asarray(D, np.float64))


def averaged_lock(floor_model, samples, K, D, min_frames=3):
    """Average image corners of the corner tags over many frames, solve once, lock.
    For a camera on a tripod this removes almost all frame-to-frame jitter."""
    acc = {}
    for dets in samples:
        for tid, c in dets:
            if tid in floor_model.world_corners:
                acc.setdefault(tid, []).append(np.asarray(c, np.float64))
    avg = [(tid, np.mean(v, 0).astype(np.float32)) for tid, v in acc.items() if len(v) >= min_frames]
    floor_model.locked = False
    floor_model.R = None
    auto = floor_model.auto_lock
    floor_model.auto_lock = False
    floor_model.update(avg, K, D, time.perf_counter())
    floor_model.auto_lock = auto
    if floor_model.valid:
        floor_model.locked = True
        floor_model.lock_rms = floor_model.reproj
    return floor_model.valid


def precision_report(floor_model, amap, K, D):
    """Millimeters on the floor per camera pixel at the arena corners (worst = far side)."""
    out = {}
    for name, (x, y) in zip(("c0", "c1", "c2", "c3"), amap.corners):
        p = np.array([[x, y, 0], [x + 0.01, y, 0], [x, y + 0.01, 0]], np.float64)
        px = floor_model.world_to_pixel(p, K, D)
        mm_per_px = 10.0 / min(np.linalg.norm(px[1] - px[0]), np.linalg.norm(px[2] - px[0]))
        out[name] = round(float(mm_per_px), 2)
    return out


_PLANE_CACHE = {}


def _in_convex(poly, pts):
    """Vectorized point-in-convex-polygon test (poly in either winding order)."""
    poly = np.asarray(poly, np.float64)
    a, b = poly, np.roll(poly, -1, 0)
    cr = ((b[:, 0] - a[:, 0])[None, :] * (pts[:, 1:2] - a[None, :, 1])
          - (b[:, 1] - a[:, 1])[None, :] * (pts[:, 0:1] - a[None, :, 0]))
    return (cr >= 0).all(1) | (cr <= 0).all(1)


def _plane_geometry(shape, poly, floor, K, D, step, max_range, show_outside):
    """Pixel polylines of the floor grid; cached while the pose does not change."""
    key = (shape[:2], poly.tobytes(), floor.rvec.round(5).tobytes(), floor.tvec.round(5).tobytes(),
           K.tobytes(), np.asarray(D).tobytes(), step, max_range, show_outside)
    hit = _PLANE_CACHE.get("key")
    if hit == key:
        return _PLANE_CACHE["geo"]
    h, w = shape[:2]
    C = floor.camera_position

    def project(pts):
        zc = (floor.R @ pts.T).T[:, 2] + floor.t[2]
        px = floor.world_to_pixel(pts, K, D)
        ok = ((zc > 0.05) & (px[:, 0] >= -50) & (px[:, 0] < w + 50) & (px[:, 1] >= -50) & (px[:, 1] < h + 50)
              & (np.hypot(pts[:, 0] - C[0], pts[:, 1] - C[1]) <= max_range))
        return px, ok

    edge = np.array([[*(a + s_ * (b - a)), 0.0] for a, b in zip(poly, np.roll(poly, -1, 0))
                     for s_ in np.linspace(0, 1, 30, endpoint=False)])
    edge_px, edge_ok = project(edge)
    lo = np.minimum(poly.min(0), C[:2] - max_range) if show_outside else poly.min(0)
    hi = np.maximum(poly.max(0), C[:2] + max_range) if show_outside else poly.max(0)
    lines, labels_ = [], []
    for axis in (0, 1):
        for v in np.arange(math.ceil(lo[axis] / step) * step, hi[axis] + 1e-9, step):
            t = np.linspace(lo[1 - axis], hi[1 - axis], 400)
            pts = np.zeros((len(t), 3))
            pts[:, axis], pts[:, 1 - axis] = v, t
            px, ok = project(pts)
            inside = _in_convex(poly, pts[:, :2])
            major = abs(v / 0.5 - round(v / 0.5)) < 1e-6
            for want_inside in (True, False):
                if not show_outside and not want_inside:
                    continue
                idx = np.flatnonzero(ok & (inside == want_inside))
                if len(idx) < 2:
                    continue
                for r in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
                    if len(r) >= 2:
                        lines.append((np.round(px[r]).astype(np.int32), want_inside, major))
            onimg = ok & (px[:, 0] >= 5) & (px[:, 0] < w - 60) & (px[:, 1] >= 15) & (px[:, 1] < h - 5)
            if major and onimg.any():
                k = np.flatnonzero(onimg)[np.argmax(px[onimg][:, 1])]
                labels_.append((f"{'x' if axis == 0 else 'y'}={v:+.1f}", (int(px[k][0]) + 3, int(px[k][1]) - 4)))
    axes = floor.world_to_pixel([[0, 0, 0], [0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2]], K, D).astype(int)
    corner_px = floor.world_to_pixel(np.column_stack([poly, np.zeros(len(poly))]), K, D)
    geo = (np.round(edge_px).astype(np.int32), bool(edge_ok.all()), bool(edge_ok.any()), lines, labels_,
           axes, corner_px)
    _PLANE_CACHE["key"], _PLANE_CACHE["geo"] = key, geo
    return geo


def draw_plane(frame, corners_xy, labels, floor, K, D, step=0.1, color=(0, 220, 255),
               max_range=4.0, show_outside=True):
    """Draw the floor plane on the camera image: the arena between the corner tags (filled,
    bright grid) and, beyond it, the rest of the visible floor out to max_range (dim grid).
    Major lines every 0.5 m are labelled. Lines are sampled densely and projected with the lens
    model, so they follow distortion and stop at the horizon / image edge. The projected
    geometry is cached while the pose is unchanged (a locked camera costs ~nothing)."""
    if not floor.valid:
        return
    poly = np.asarray(corners_xy, np.float64)
    edge, edge_all, edge_any, lines, labels_, o, corner_px = _plane_geometry(
        frame.shape, poly, floor, K, D, step, max_range, show_outside)
    if edge_all:
        x0, y0 = np.maximum(edge.min(0), 0)
        x1, y1 = np.minimum(edge.max(0) + 1, [frame.shape[1], frame.shape[0]])
        if x1 > x0 and y1 > y0:
            roi = frame[y0:y1, x0:x1]
            over = roi.copy()
            cv2.fillPoly(over, [edge - [x0, y0]], color)
            cv2.addWeighted(over, 0.18, roi, 0.82, 0, roi)
    for pl, inside, major in lines:
        if inside:
            col, th = ((255, 255, 255), 2) if major else ((210, 210, 210), 1)
        else:
            col, th = ((140, 200, 200), 2) if major else ((120, 150, 150), 1)
        cv2.polylines(frame, [pl], False, col, th, cv2.LINE_AA)
    for txt, org in labels_:
        cv2.putText(frame, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    if edge_any:
        cv2.polylines(frame, [edge], True, color, 3, cv2.LINE_AA)
    for k, col in ((1, (0, 0, 255)), (2, (0, 200, 0)), (3, (255, 80, 0))):
        cv2.arrowedLine(frame, tuple(o[0]), tuple(o[k]), col, 3, cv2.LINE_AA, tipLength=0.2)
    for (x, y), lab, q in zip(poly, labels, corner_px):
        txt = f"{lab} ({x:.3f}, {y:.3f})"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        org = (int(q[0]) - tw // 2, int(q[1]) + 30)
        cv2.rectangle(frame, (org[0] - 3, org[1] - th - 4), (org[0] + tw + 3, org[1] + 5), (0, 0, 0), -1)
        cv2.putText(frame, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


# --------------------------------------------------------------------------
# Map
# --------------------------------------------------------------------------

class ArenaMap:
    """Metric bird's-eye map of the arena. Map pixel (col, row) center is at
    x = x0 + (col + 0.5) * res,  y = y1 - (row + 0.5) * res   (north up, +y up)."""

    def __init__(self, arena, floor_tags):
        self.arena = arena
        ids = arena["corner_ids"]
        self.corners = np.array([[floor_tags[i]["x"], floor_tags[i]["y"]] for i in ids], np.float64)
        mg = float(arena.get("margin_m", 0.0))
        lo = self.corners.min(0) - max(mg, 0.0)
        hi = self.corners.max(0) + max(mg, 0.0)
        self.res = float(arena["map_resolution_m"])
        self.x0, self.y0 = lo
        self.x1, self.y1 = hi
        self.W = int(math.ceil((self.x1 - self.x0) / self.res))
        self.H = int(math.ceil((self.y1 - self.y0) / self.res))
        inside = np.zeros((self.H, self.W), np.uint8)
        cv2.fillConvexPoly(inside, np.round(self.w2p(self.corners)).astype(np.int32), 255)
        r = int(round(abs(mg) / self.res))
        if r:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            inside = cv2.dilate(inside, k) if mg > 0 else cv2.erode(inside, k)
        self.inside = inside > 0
        self._remap_key = None
        self.mapx = self.mapy = None
        self.background = None

    # coordinates
    def w2p(self, xy):
        xy = np.asarray(xy, np.float64).reshape(-1, 2)
        return np.stack([(xy[:, 0] - self.x0) / self.res - 0.5, (self.y1 - xy[:, 1]) / self.res - 0.5], 1)

    def p2w(self, col, row):
        return self.x0 + (col + 0.5) * self.res, self.y1 - (row + 0.5) * self.res

    # rectification
    def update_remap(self, floor, K, D):
        if not floor.valid:
            return False
        key = (floor.rvec.tobytes(), floor.tvec.tobytes())
        if key == self._remap_key:
            return True
        cols, rows = np.meshgrid(np.arange(self.W), np.arange(self.H))
        X = self.x0 + (cols + 0.5) * self.res
        Y = self.y1 - (rows + 0.5) * self.res
        pts = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1)
        proj, _ = cv2.projectPoints(pts, floor.rvec, floor.tvec, K, D)
        proj = proj.reshape(self.H, self.W, 2).astype(np.float32)
        zc = ((floor.R @ pts.T).T + floor.t)[:, 2].reshape(self.H, self.W)
        proj[zc <= 0.01] = -1
        self.mapx, self.mapy = proj[..., 0].copy(), proj[..., 1].copy()
        self._remap_key = key
        return True

    def rectify(self, frame):
        return cv2.remap(frame, self.mapx, self.mapy, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # obstacles
    @staticmethod
    def _lab(rect):
        return cv2.GaussianBlur(cv2.cvtColor(rect, cv2.COLOR_BGR2LAB).astype(np.float32), (5, 5), 0)

    def set_background(self, rect):
        self.background = self._lab(rect)

    def obstacle_mask(self, rect, clear_circles=(), clear_polys=()):
        """Pixels that differ from the empty-floor reference. Lightness counts half so soft
        shadows matter less than color changes. clear_circles [(x, y, r)] and clear_polys
        [world polygon] are erased (the robot's own body)."""
        if self.background is None:
            return np.zeros((self.H, self.W), bool)
        d = np.abs(self._lab(rect) - self.background)
        score = 0.5 * d[..., 0] + d[..., 1] + d[..., 2]
        mask = (score > float(self.arena["obstacle_threshold"])).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        mask[~self.inside] = 0
        for x, y, r in clear_circles:
            c = self.w2p([(x, y)])[0]
            cv2.circle(mask, (int(round(c[0])), int(round(c[1]))), int(math.ceil(r / self.res)), 0, -1)
        for poly in clear_polys:
            cv2.fillConvexPoly(mask, np.round(self.w2p(poly)).astype(np.int32), 0)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        min_px = float(self.arena["min_obstacle_area_m2"]) / self.res ** 2
        keep = np.zeros(n, bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_px
        return keep[lab]

    @staticmethod
    def body_outline(x, y, radius, height, camera_pos, n=48):
        """Floor area covered, in the bird's-eye image, by an upright cylinder (the robot):
        the base circle plus its top circle projected along camera rays onto the floor.
        Tall things lean away from the camera in a floor-rectified image (parallax)."""
        th = np.linspace(0, 2 * np.pi, n, endpoint=False)
        ring = np.stack([x + radius * np.cos(th), y + radius * np.sin(th)], 1)
        C = np.asarray(camera_pos, float)
        pts = [ring]
        if height > 0 and C[2] > height + 0.01:
            top = np.column_stack([ring, np.full(n, height)])
            k = C[2] / (C[2] - height)
            pts.append((C[None, :2] + (top[:, :2] - C[None, :2]) * k))
        return cv2.convexHull(np.concatenate(pts).astype(np.float32)).reshape(-1, 2)

    def footprint_mask(self, objects, cfg, skip_ids=()):
        """Tagged objects as rotated rectangles / circles. Per object in cfg["objects"]:
        "footprint": [length_x, width_y] in the tag's frame, or "radius". Default: the tag square.
        "obstacle": false skips the object (e.g. a flat marker the robot may drive over)."""
        mask = np.zeros((self.H, self.W), np.uint8)
        for o in objects:
            if o.get("frame") != "world" or o["id"] in skip_ids:
                continue
            spec = cfg.get("objects", {}).get(o["id"], {})
            if spec.get("obstacle") is False or o.get("obstacle") is False:
                continue
            if o.get("footprint_poly") is not None:          # detected (untagged) object
                cv2.fillConvexPoly(mask, np.round(self.w2p(o["footprint_poly"])).astype(np.int32), 255)
                continue
            c = self.w2p([(o["x"], o["y"])])[0]
            if "radius" in spec:
                cv2.circle(mask, (int(round(c[0])), int(round(c[1]))),
                           int(math.ceil(spec["radius"] / self.res)), 255, -1)
                continue
            size = cfg["tag_sizes"].get(o["id"], cfg.get("tag_size", 0.1))
            lx, ly = spec.get("footprint", [size + 0.01, size + 0.01])
            h = math.radians(o.get("yaw_deg", 0.0))
            R = np.array([[math.cos(h), -math.sin(h)], [math.sin(h), math.cos(h)]])
            box = (R @ (np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * [lx / 2, ly / 2]).T).T
            poly = self.w2p(box + [o["x"], o["y"]])
            cv2.fillConvexPoly(mask, np.round(poly).astype(np.int32), 255)
        return mask > 0

    def occupancy(self, obstacles, footprints):
        """True = blocked. Outside the arena counts as blocked."""
        return obstacles | footprints | ~self.inside


# --------------------------------------------------------------------------
# Adaptive: learned anchors and the global floor map
# --------------------------------------------------------------------------

class GlobalMap:
    """Floor map in arena coordinates that covers everything the camera can see (not just the
    area between the corner tags), grows when new floor comes into view, and fuses frames:

      color      last seen color of each cell
      seen       cell has been observed at least once (unknown otherwise)
      last_seen  time of last observation (old cells are drawn faded)
      quality    best camera pixels per cell seen (how sharply the cell was observed)
      bg         empty-floor reference (LAB); learned from "Capture empty floor", extended to
                 newly seen floor, and slowly adapted on free cells (lighting changes)
      logodds    obstacle evidence: +0.9 per frame the cell differs from bg, -0.5 when it
                 matches; > 0 means obstacle. Single-frame noise does not flip a cell.

    Grid: cell (col, row) center at x = x0 + (col + .5) * res, y = y1 - (row + .5) * res."""

    def __init__(self, arena, floor_tags, res=None, floor=None, K=None, D=None):
        self.arena = arena
        ids = arena["corner_ids"]
        self.corners = np.array([[floor_tags[i]["x"], floor_tags[i]["y"]] for i in ids], np.float64)
        if not res:
            res = self.auto_resolution(floor, K, D) if floor is not None and floor.valid else 0.005
            if floor is not None and floor.valid and K is not None:
                # keep the visible part of the grid around a million cells (fast fusion)
                h_, w_ = int(round(K[1, 2] * 2)), int(round(K[0, 2] * 2))
                box = self.visible_window(floor, K, D, (h_, w_), float(arena.get("max_range_m", 4.0)))
                if box is not None:
                    area = (box[2] - box[0]) * (box[3] - box[1])
                    res = max(res, math.ceil(math.sqrt(area / 1.0e6) * 1000) / 1000)
        self.res = float(res)
        lo, hi = self.corners.min(0) - 0.2, self.corners.max(0) + 0.2
        self.x0, self.y0 = math.floor(lo[0] / 0.5) * 0.5, math.floor(lo[1] / 0.5) * 0.5
        self.x1, self.y1 = math.ceil(hi[0] / 0.5) * 0.5, math.ceil(hi[1] / 0.5) * 0.5
        self._alloc()
        self.bg_enabled = False
        self.detect_appearance = False  # objects = floor cells that look unlike the rest of the floor
        self._cache = None
        self.view = None            # (r0, r1, c0, c1, vis) of the current frame

    # ---- grid management
    @property
    def W(self):
        return int(round((self.x1 - self.x0) / self.res))

    @property
    def H(self):
        return int(round((self.y1 - self.y0) / self.res))

    def _alloc(self):
        H, W = self.H, self.W
        self.color = np.zeros((H, W, 3), np.uint8)
        self.seen = np.zeros((H, W), bool)
        self.last_seen = np.zeros((H, W), np.float32)
        self.quality = np.zeros((H, W), np.float32)
        self.bg = np.zeros((H, W, 3), np.float32)
        self.bg_ok = np.zeros((H, W), bool)
        self.logodds = np.zeros((H, W), np.float32)

    def _grow(self, x0, y0, x1, y1, limit):
        """Enlarge the grid (in 0.5 m steps, up to `limit` m from the arena) keeping all data."""
        c = self.corners
        x0 = max(x0, c[:, 0].min() - limit)
        y0 = max(y0, c[:, 1].min() - limit)
        x1 = min(x1, c[:, 0].max() + limit)
        y1 = min(y1, c[:, 1].max() + limit)
        nx0 = min(self.x0, math.floor(x0 / 0.5) * 0.5)
        ny0 = min(self.y0, math.floor(y0 / 0.5) * 0.5)
        nx1 = max(self.x1, math.ceil(x1 / 0.5) * 0.5)
        ny1 = max(self.y1, math.ceil(y1 / 0.5) * 0.5)
        if (nx0, ny0, nx1, ny1) == (self.x0, self.y0, self.x1, self.y1):
            return False
        old = {k: getattr(self, k) for k in ("color", "seen", "last_seen", "quality", "bg", "bg_ok", "logodds")}
        dc = int(round((self.x0 - nx0) / self.res))
        dr = int(round((ny1 - self.y1) / self.res))
        oh, ow = old["seen"].shape
        self.x0, self.y0, self.x1, self.y1 = nx0, ny0, nx1, ny1
        self._alloc()
        for k, v in old.items():
            getattr(self, k)[dr:dr + oh, dc:dc + ow] = v
        self._cache = None
        return True

    # ---- coordinates (same convention as ArenaMap)
    def w2p(self, xy):
        xy = np.asarray(xy, np.float64).reshape(-1, 2)
        return np.stack([(xy[:, 0] - self.x0) / self.res - 0.5, (self.y1 - xy[:, 1]) / self.res - 0.5], 1)

    def p2w(self, col, row):
        return self.x0 + (col + 0.5) * self.res, self.y1 - (row + 0.5) * self.res

    @property
    def quad_mask(self):
        m_ = np.zeros((self.H, self.W), np.uint8)
        cv2.fillConvexPoly(m_, np.round(self.w2p(self.corners)).astype(np.int32), 1)
        return m_ > 0

    def auto_resolution(self, floor, K, D):
        """Map cell size matched to how finely the camera sees the arena: the median floor
        size of one camera pixel inside the corner region, rounded to 1 mm, kept in 2..10 mm."""
        lo, hi = self.corners.min(0), self.corners.max(0)
        xs, ys = np.meshgrid(np.linspace(lo[0], hi[0], 8), np.linspace(lo[1], hi[1], 8))
        vals = []
        for x, y in zip(xs.ravel(), ys.ravel()):
            p = floor.world_to_pixel([[x, y, 0], [x + 0.01, y, 0], [x, y + 0.01, 0]], K, D)
            px = min(np.linalg.norm(p[1] - p[0]), np.linalg.norm(p[2] - p[0]))
            if px > 0:
                vals.append(10.0 / px)
        mm = float(np.median(vals)) if vals else 5.0
        return float(np.clip(round(mm), 2, 10)) / 1000.0

    # ---- visibility
    def visible_window(self, floor, K, D, shape, max_range):
        """Bounding box (world) of floor the camera sees, out to max_range from the camera."""
        h, w = shape[:2]
        n = 24
        us = np.concatenate([np.linspace(0, w - 1, n), np.full(n, w - 1), np.linspace(w - 1, 0, n), np.zeros(n)])
        vs = np.concatenate([np.zeros(n), np.linspace(0, h - 1, n), np.full(n, h - 1), np.linspace(h - 1, 0, n)])
        C = floor.camera_position
        nrm = cv2.undistortPoints(np.stack([us, vs], 1).reshape(-1, 1, 2).astype(np.float64), K, D).reshape(-1, 2)
        d = (floor.R.T @ np.column_stack([nrm, np.ones(len(nrm))]).T).T
        dh = np.linalg.norm(d[:, :2], axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            s_ = np.where(d[:, 2] < -1e-6, -C[2] / d[:, 2], np.inf)
            hit = C[None, :2] + s_[:, None] * d[:, :2]
            far = C[None, :2] + d[:, :2] / np.maximum(dh, 1e-9)[:, None] * max_range
        use_hit = np.isfinite(s_) & (np.hypot(*(hit - C[None, :2]).T) <= max_range)
        pts = np.where(use_hit[:, None], hit, far)[dh > 1e-9]
        if not len(pts):
            return None
        return pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()

    def _remap(self, floor, K, D, shape, win, max_range, min_px_per_cell):
        key = (floor.rvec.tobytes(), floor.tvec.tobytes(), win, shape[:2], self.res,
               self.x0, self.y1)
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]
        r0, r1, c0, c1 = win
        cols, rows = np.meshgrid(np.arange(c0, c1), np.arange(r0, r1))
        X = self.x0 + (cols + 0.5) * self.res
        Y = self.y1 - (rows + 0.5) * self.res
        pts = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1)
        zc = ((floor.R @ pts.T).T + floor.t)[:, 2].reshape(X.shape)
        proj, _ = cv2.projectPoints(pts, floor.rvec, floor.tvec, K, D)
        proj = proj.reshape(X.shape + (2,)).astype(np.float32)
        mx, my = proj[..., 0], proj[..., 1]
        h, w = shape[:2]
        C = floor.camera_position
        rng = np.hypot(X - C[0], Y - C[1])
        # camera pixels per map cell: how sharply this cell is observed
        dxu, dxv = np.gradient(mx, axis=1), np.gradient(my, axis=1)
        dyu, dyv = np.gradient(mx, axis=0), np.gradient(my, axis=0)
        ppc = np.minimum(np.hypot(dxu, dxv), np.hypot(dyu, dyv)).astype(np.float32)
        vis = ((zc > 0.05) & (mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)
               & (rng <= max_range) & (ppc >= min_px_per_cell))
        mx[~vis] = -1
        my[~vis] = -1
        out = (mx, my, vis, ppc)
        self._cache = (key, out)
        return out

    # ---- per-frame fusion
    @staticmethod
    def _lab(img):
        return cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32), (5, 5), 0)

    def update(self, frame, floor, K, D, now, clear_polys=(), max_range=4.0, min_px_per_cell=0.35,
               capture_bg=False, adapt_rate=0.02):
        """Fuse one camera frame into the map. Returns the current window's obstacle mask."""
        if not floor.valid:
            self.view = None
            return None
        box = self.visible_window(floor, K, D, frame.shape, max_range)
        if box is None:
            self.view = None
            return None
        self._grow(*box, limit=float(self.arena.get("max_extent_m", 6.0)))
        c0 = max(0, int(math.floor((box[0] - self.x0) / self.res)))
        c1 = min(self.W, int(math.ceil((box[2] - self.x0) / self.res)))
        r0 = max(0, int(math.floor((self.y1 - box[3]) / self.res)))
        r1 = min(self.H, int(math.ceil((self.y1 - box[1]) / self.res)))
        if c1 <= c0 or r1 <= r0:
            self.view = None
            return None
        mx, my, vis, ppc = self._remap(floor, K, D, frame.shape, (r0, r1, c0, c1), max_range, min_px_per_cell)
        rect = cv2.remap(frame, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        sl = (slice(r0, r1), slice(c0, c1))
        vis3 = vis[..., None]
        np.copyto(self.color[sl], rect, where=vis3)
        self.seen[sl] |= vis
        np.copyto(self.last_seen[sl], np.float32(now), where=vis)
        np.maximum(self.quality[sl], np.where(vis, ppc, 0), out=self.quality[sl])
        lab = self._lab(rect)
        bg, bg_ok, lo = self.bg[sl], self.bg_ok[sl], self.logodds[sl]
        if capture_bg:
            np.copyto(bg, lab, where=vis3)
            bg_ok |= vis
            np.copyto(lo, np.float32(0), where=vis)
            self.bg_enabled = True
        obst = np.zeros(vis.shape, bool)
        sources = []
        if self.bg_enabled:
            new = vis & ~bg_ok           # newly seen floor: assume empty at first sight
            if new.any():
                np.copyto(bg, lab, where=new[..., None])
                bg_ok |= new
            d = cv2.absdiff(lab, bg)
            score = cv2.transform(d, np.array([[0.5, 1.0, 1.0]], np.float32))
            if score.ndim == 3:
                score = score[..., 0]
            sources.append((score > float(self.arena["obstacle_threshold"])) & vis & bg_ok)
        if self.detect_appearance:
            sources.append(appearance_mask(rect, vis, ppc, self.res, float(self.arena.get("appearance_z", 5.0))))
        if sources:
            raw = np.logical_or.reduce(sources).astype(np.uint8) * 255
            raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            for poly in clear_polys:
                pp = self.w2p(poly) - [c0, r0]
                cv2.fillConvexPoly(raw, np.round(pp).astype(np.int32), 0)
            n, labl, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
            keep = np.zeros(n, bool)
            keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= float(self.arena["min_obstacle_area_m2"]) / self.res ** 2
            obst = keep[labl] & vis
            judged = vis if self.detect_appearance else (vis & bg_ok)
            free = judged & ~obst
            lo += np.where(obst, np.float32(0.9), np.where(free, np.float32(-0.5), np.float32(0)))
            np.clip(lo, -4, 4, out=lo)
            if self.bg_enabled:
                # adapt the empty-floor reference to slow lighting changes, on confidently free cells only
                ad = free & bg_ok & (lo < -2)
                cv2.accumulateWeighted(lab, bg, adapt_rate, mask=ad.astype(np.uint8))
            # robot body, flat floor tags: never an obstacle
            if clear_polys:
                cm = np.zeros(vis.shape, np.uint8)
                for poly in clear_polys:
                    cv2.fillConvexPoly(cm, np.round(self.w2p(poly) - [c0, r0]).astype(np.int32), 1)
                lo[cm > 0] = np.minimum(lo[cm > 0], 0)
        self.view = (r0, r1, c0, c1, vis)
        return obst

    # ---- outputs
    def obstacles(self):
        return self.logodds > 0

    def objects_in_view(self):
        """Obstacle blobs in the current view as objects: centroid, footprint polygon (world),
        size. Blobs come from the fused obstacle evidence, so they are stable over frames."""
        if self.view is None:
            return []
        r0, r1, c0, c1, vis = self.view
        # Confident cells only (evidence > 1, i.e. seen as "not floor" in several updates): cells at
        # an object's rim hover around 0 and made the outline breathe. Fragments of one object
        # closer than ~2 cm are joined, so it does not split into two boxes now and then.
        ob = ((self.logodds[r0:r1, c0:c1] > 1.0) & vis).astype(np.uint8)
        k = max(3, int(round(0.02 / self.res)) | 1)
        ob = cv2.morphologyEx(ob, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        cs, _ = cv2.findContours(ob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_px = float(self.arena["min_obstacle_area_m2"]) / self.res ** 2
        out = []
        for c in cs:
            if cv2.contourArea(c) < min_px:
                continue
            mo = cv2.moments(c)
            if mo["m00"] <= 0:
                continue
            hull = cv2.convexHull(c).reshape(-1, 2).astype(np.float64)
            world = np.column_stack([self.x0 + (c0 + hull[:, 0] + 0.5) * self.res,
                                     self.y1 - (r0 + hull[:, 1] + 0.5) * self.res])
            cx, cy = mo["m10"] / mo["m00"], mo["m01"] / mo["m00"]
            (_, _), (w, h), _ = cv2.minAreaRect(c)
            if min(w, h) * self.res < 0.015:       # a sliver along an edge, not an object
                continue
            out.append({"name": "object", "conf": 1.0, "box": None,
                        "x": float(self.x0 + (c0 + cx + 0.5) * self.res),
                        "y": float(self.y1 - (r0 + cy + 0.5) * self.res),
                        "contact": None, "width_m": float(max(w, h) * self.res),
                        "depth_m": float(min(w, h) * self.res), "height_m": 0.0, "footprint": world})
        return out

    def footprint_mask(self, objects, cfg, skip_ids=()):
        return ArenaMap.footprint_mask(self, objects, cfg, skip_ids)

    def occupancy(self, footprints, unknown_blocked=True):
        occ = self.obstacles() | footprints
        if unknown_blocked:
            occ |= ~self.seen
        return occ

    def plan_resolution(self, max_cells=80000):
        """A* cell size: the configured one, enlarged if the map got very big (keeps A* fast)."""
        base = float(self.arena["plan_resolution_m"])
        area = (self.x1 - self.x0) * (self.y1 - self.y0)
        return max(base, math.sqrt(area / max_cells))

    def reset(self):
        self._alloc()
        self.bg_enabled = False
        self._cache = None

    def mm_per_px(self):
        """Floor millimeters per camera pixel for each cell (best observation), NaN if unseen."""
        with np.errstate(divide="ignore"):
            return np.where(self.quality > 0, self.res * 1000 / np.maximum(self.quality, 1e-6), np.nan)


# --------------------------------------------------------------------------
# Objects by appearance (no model, no stored reference)
# --------------------------------------------------------------------------

_FEATURE_MIN_SPREAD = (2.0, 1.5, 1.5, 0.12)    # L, a, b, log edge energy: floor noise never below this
_FEATURE_WEIGHT = (0.7, 1.0, 1.0, 1.0)         # lightness counts a bit less (shading, shadows)


def appearance_mask(rect, vis, ppc, res, z_thr=5.0, win_m=0.025):
    """Floor cells that do not look like floor, judged against the floor itself in THIS frame.

    Works for any object, never seen before, with no class list and no empty-floor image:
    the visible floor is mostly floor, so the robust statistics (median / MAD) of its local
    color (Lab mean over ~2.5 cm) and local edge energy (log of the mean gradient magnitude:
    a speckled carpet is busy, a box, a phone, paper are smooth) describe "floor". A cell
    whose features lie more than z_thr robust standard deviations away is something else: a
    cube, paper, a box, a foot. Statistics are taken separately for near / mid / far floor (by
    camera pixels per cell), so blurrier far floor is compared with far floor, not with sharp
    near floor. Returns a bool mask shaped like vis.

    Speed: gradients are taken at full map resolution (the carpet speckle is a few mm), then the
    features are averaged down to ~8 mm cells for the statistics and the test (objects of 2 cm
    and more are still several cells), and the mask is scaled back up."""
    full_shape = vis.shape
    lab = cv2.cvtColor(rect, cv2.COLOR_BGR2LAB).astype(np.float32)
    # Smoothness is the strongest cue on a patterned floor: on a real frame a gray box, a black
    # device and a red box were 9-11 robust std devs smoother than the carpet, while their mean
    # color / lightness was within 2 (gray, black) of it.
    gx = cv2.Sobel(lab[..., 0], cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(lab[..., 0], cv2.CV_32F, 0, 1)
    stack = np.dstack([lab, cv2.magnitude(gx, gy)])
    f = max(1, int(round(0.008 / res)))                    # downsample factor to ~8 mm cells
    if f > 1:
        size = (max(1, full_shape[1] // f), max(1, full_shape[0] // f))
        stack = cv2.resize(stack, size, interpolation=cv2.INTER_AREA)
        vis = cv2.resize(vis.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
        ppc = cv2.resize(ppc, size, interpolation=cv2.INTER_AREA)
        res = res * f
    k = max(3, int(round(win_m / res)) | 1)
    # stay 2 windows away from the edge of the view: border cells mix in black / smeared pixels
    valid = cv2.erode(vis.astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8)) > 0
    if valid.sum() < 300:
        return np.zeros(full_shape, bool)
    stack = cv2.blur(stack, (k, k))
    feats = (stack[..., 0], stack[..., 1], stack[..., 2], np.log(stack[..., 3] + 2.0))
    edges = np.quantile(ppc[valid], [1 / 3, 2 / 3])
    band = np.digitize(ppc, edges)
    z = np.zeros(vis.shape, np.float32)
    for b in range(3):
        sel = valid & (band == b)
        if sel.sum() < 100:
            continue
        idx = np.flatnonzero(sel)
        sub = idx[:: max(1, len(idx) // 20000)]            # statistics from a subsample: fast
        for j, (feat, wgt, lo_) in enumerate(zip(feats, _FEATURE_WEIGHT, _FEATURE_MIN_SPREAD)):
            v = feat.ravel()[sub]
            med = float(np.median(v))
            mad = max(float(np.median(np.abs(v - med))) * 1.4826, lo_)
            d = feat[sel] - med
            # edge energy counts one way only: objects are SMOOTHER than a patterned floor. Being
            # busier than the floor is just the outline of something (a tag's border, the edge
            # of the view), which would otherwise give a ring of false hits on a plain floor.
            d = np.maximum(-d, 0) if j == 3 else np.abs(d)
            z[sel] = np.maximum(z[sel], wgt * d / mad)
    mask = (z > z_thr) & valid
    if mask.shape != full_shape:
        up = cv2.resize(mask.astype(np.uint8), (full_shape[1] // f * f, full_shape[0] // f * f),
                        interpolation=cv2.INTER_NEAREST)
        mask = np.zeros(full_shape, bool)
        mask[:up.shape[0], :up.shape[1]] = up > 0
    return mask


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------

def _disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def planning_grid(occ, amap, robot_radius=None, plan_res=None):
    """Downsample the map occupancy to A* cells (a cell is blocked if any map pixel in it is)
    and inflate by the robot radius. Returns (blocked bool array, cell size).

    One A* cell is exactly k x k map pixels (k = plan_res / map res, rounded), so cell (c, r)
    covers map pixels [k*c, k*c + k) and its world position is exact. (Resizing by a
    fractional factor made the grid drift against the coordinates A* uses: up to one cell
    off, so a third of obstacle pixels were blocked in a neighbor cell instead of their own.)
    Map pixels padded past the map edge count as blocked."""
    plan_res = plan_res or amap.arena["plan_resolution_m"]
    robot_radius = amap.arena["robot_radius_m"] if robot_radius is None else robot_radius
    k = max(1, int(round(plan_res / amap.res)))
    cell = k * amap.res
    H, W = occ.shape
    h, w = -(-H // k), -(-W // k)
    pad = np.ones((h * k, w * k), bool)
    pad[:H, :W] = occ
    small = pad.reshape(h, k, w, k).any(axis=(1, 3))
    r = int(math.ceil(robot_radius / cell))
    blocked = cv2.dilate(small.astype(np.uint8), _disk(r)) > 0 if r > 0 else small
    return blocked, cell


def grid_to_map(blocked, amap, cell):
    """A* grid back at map resolution (for drawing), exactly aligned with planning_grid."""
    k = max(1, int(round(cell / amap.res)))
    H, W = int(amap.H), int(amap.W)
    return np.repeat(np.repeat(blocked, k, 0), k, 1)[:H, :W]


def _w2g(amap, xy, cell):
    # floor, not int(): points left of / above the map must map to negative (outside) cells
    return (int(math.floor((xy[0] - amap.x0) / cell)), int(math.floor((amap.y1 - xy[1]) / cell)))


def _g2w(amap, c, r, cell):
    return amap.x0 + (c + 0.5) * cell, amap.y1 - (r + 0.5) * cell


def _nearest_free(blocked, c, r):
    free = np.argwhere(~blocked)
    if not len(free):
        return None
    k = np.argmin((free[:, 0] - r) ** 2 + (free[:, 1] - c) ** 2)
    return int(free[k, 1]), int(free[k, 0])


def _line_free(blocked, a, b):
    """True if EVERY cell the straight segment between the centers of cells a and b passes
    through is free (supercover traversal). Point sampling, used before, could step over the
    corner of a blocked cell, so a shortcut could clip an (inflated) obstacle. A segment going
    exactly through a cell corner needs both side cells free, as in A* (no corner cutting)."""
    x, y = int(a[0]), int(a[1])
    x1, y1 = int(b[0]), int(b[1])
    dx, dy = abs(x1 - x), abs(y1 - y)
    sx, sy = (1 if x1 > x else -1), (1 if y1 > y else -1)
    if blocked[y, x]:
        return False
    ix = iy = 0
    while ix < dx or iy < dy:
        d = (1 + 2 * ix) * dy - (1 + 2 * iy) * dx      # which cell border the line crosses next
        if d == 0:
            if blocked[y, x + sx] or blocked[y + sy, x]:
                return False
            x, y, ix, iy = x + sx, y + sy, ix + 1, iy + 1
        elif d < 0:
            x, ix = x + sx, ix + 1
        else:
            y, iy = y + sy, iy + 1
        if blocked[y, x]:
            return False
    return True


def _segment_free(blocked, amap, cell, p, q):
    """Exact grid traversal (Amanatides-Woo) of the world segment p -> q: True if every A* cell
    it passes through is free and inside the grid."""
    H, W = blocked.shape
    u0, v0 = (p[0] - amap.x0) / cell, (amap.y1 - p[1]) / cell
    u1, v1 = (q[0] - amap.x0) / cell, (amap.y1 - q[1]) / cell
    c, r = int(math.floor(u0)), int(math.floor(v0))
    ce, re = int(math.floor(u1)), int(math.floor(v1))

    def bad(c_, r_):
        return not (0 <= c_ < W and 0 <= r_ < H) or blocked[r_, c_]
    if bad(c, r) or bad(ce, re):
        return False
    du, dv = u1 - u0, v1 - v0
    sx, sy = (1 if du > 0 else -1), (1 if dv > 0 else -1)
    tdx = abs(1.0 / du) if du else math.inf
    tdy = abs(1.0 / dv) if dv else math.inf
    tmx = ((c + 1 - u0) if du > 0 else (u0 - c)) * tdx if du else math.inf
    tmy = ((r + 1 - v0) if dv > 0 else (v0 - r)) * tdy if dv else math.inf
    for _ in range(abs(ce - c) + abs(re - r) + 2):
        if (c, r) == (ce, re):
            return True
        if abs(tmx - tmy) < 1e-12:                    # exactly through a cell corner
            if bad(c + sx, r) or bad(c, r + sy):
                return False
            c, r, tmx, tmy = c + sx, r + sy, tmx + tdx, tmy + tdy
        elif tmx < tmy:
            c, tmx = c + sx, tmx + tdx
        else:
            r, tmy = r + sy, tmy + tdy
        if bad(c, r):
            return False
    return True


def astar(blocked, start, goal):
    """8-connected A* on a bool grid (True = blocked). start/goal = (col, row).
    Returns a list of (col, row) or None. No corner cutting past blocked cells."""
    H, W = blocked.shape
    sc, sr = start
    gc, gr = goal
    if blocked[sr, sc] or blocked[gr, gc]:
        return None
    SQ2 = math.sqrt(2.0)
    g = np.full((H, W), np.inf)
    parent = np.full((H, W), -1, np.int64)
    closed = np.zeros((H, W), bool)
    g[sr, sc] = 0.0

    def h(c, r):
        dx, dy = abs(c - gc), abs(r - gr)
        return (dx + dy) + (SQ2 - 2) * min(dx, dy)
    heap = [(h(sc, sr), 0.0, sc, sr)]
    nbrs = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, SQ2), (1, -1, SQ2), (-1, 1, SQ2), (-1, -1, SQ2)]
    while heap:
        _, gc_, c, r = heapq.heappop(heap)
        if closed[r, c]:
            continue
        closed[r, c] = True
        if c == gc and r == gr:
            path, k = [], r * W + c
            while k != -1:
                path.append((k % W, k // W))
                k = parent[k // W, k % W]
            return path[::-1]
        for dc, dr, cost in nbrs:
            nc, nr = c + dc, r + dr
            if not (0 <= nc < W and 0 <= nr < H) or blocked[nr, nc] or closed[nr, nc]:
                continue
            if dc and dr and (blocked[r, nc] or blocked[nr, c]):
                continue
            ng = gc_ + cost
            if ng < g[nr, nc]:
                g[nr, nc] = ng
                parent[nr, nc] = r * W + c
                heapq.heappush(heap, (ng + h(nc, nr), ng, nc, nr))
    return None


def plan_path(occ, amap, start_xy, goal_xy, robot_radius=None, plan_res=None, blocked=None):
    """A* from start to goal in arena meters. Returns a dict with
    status ("ok", "no_path", "start_blocked"...), waypoints [(x, y)], length_m, raw cell path."""
    if blocked is None:
        blocked, cell = planning_grid(occ, amap, robot_radius, plan_res)
    else:
        # plan_res must be the cell size planning_grid returned for this grid
        cell = plan_res or amap.arena["plan_resolution_m"]
    H, W = blocked.shape
    s, g = _w2g(amap, start_xy, cell), _w2g(amap, goal_xy, cell)
    notes = []
    moved = {"start": False, "goal": False}
    for name, p in (("start", s), ("goal", g)):
        if not (0 <= p[0] < W and 0 <= p[1] < H):
            return {"status": f"{name}_outside_map", "waypoints": [], "length_m": None, "notes": notes}
    if blocked[s[1], s[0]]:
        s2 = _nearest_free(blocked, *s)
        if s2 is None:
            return {"status": "start_blocked", "waypoints": [], "length_m": None, "notes": notes}
        notes.append(f"start moved {math.hypot(s2[0] - s[0], s2[1] - s[1]) * cell * 1e3:.0f} mm to free space")
        s = s2
        moved["start"] = True
    if blocked[g[1], g[0]]:
        g2 = _nearest_free(blocked, *g)
        if g2 is None:
            return {"status": "goal_blocked", "waypoints": [], "length_m": None, "notes": notes}
        notes.append(f"goal moved {math.hypot(g2[0] - g[0], g2[1] - g[1]) * cell * 1e3:.0f} mm to free space")
        g = g2
        moved["goal"] = True
    t0 = time.perf_counter()
    cells = astar(blocked, s, g)
    dt = (time.perf_counter() - t0) * 1e3
    if cells is None:
        return {"status": "no_path", "waypoints": [], "length_m": None, "notes": notes, "ms": dt}
    # Shortcut: keep a waypoint only where the straight line would hit an obstacle.
    simple = [cells[0]]
    i = 0
    while i < len(cells) - 1:
        j = len(cells) - 1
        while j > i + 1 and not _line_free(blocked, cells[i], cells[j]):
            j -= 1
        simple.append(cells[j])
        i = j
    wps = [_g2w(amap, c, r, cell) for c, r in simple]
    # Snap the ends to the exact requested points (unless they had to move off an obstacle).
    # The first / last leg then no longer starts at a cell center, so check it again exactly;
    # if it would clip a blocked cell, go via the (free) start / goal cell's center instead.
    if not moved["start"]:
        s_xy = tuple(map(float, start_xy))
        if len(wps) > 1 and _segment_free(blocked, amap, cell, s_xy, wps[1]):
            wps[0] = s_xy
        else:
            wps.insert(0, s_xy)
    if not moved["goal"]:
        g_xy = tuple(map(float, goal_xy))
        if len(wps) > 1 and _segment_free(blocked, amap, cell, wps[-2], g_xy):
            wps[-1] = g_xy
        else:
            wps.append(g_xy)
    length = float(sum(math.dist(a, b) for a, b in zip(wps, wps[1:])))
    return {"status": "ok", "waypoints": [(round(float(x), 4), round(float(y), 4)) for x, y in wps],
            "length_m": round(length, 4), "cells": cells, "notes": notes, "ms": round(dt, 1)}


def plan_route(amap, points, plan_res=None, blocked=None, occ=None):
    """A* through several points in order (start, stops..., goal), like a route with
    intermediate stops. Every leg is planned on the same grid and the legs are joined.
    Returns plan_path's fields for the whole route plus "legs" [{waypoints, length_m}]; if a leg
    fails, status names it (e.g. "leg 2 (stop 1 -> goal): no_path")."""
    if blocked is None:
        blocked, plan_res = planning_grid(occ, amap, plan_res=plan_res)
    n = len(points)

    def name(i):
        return "start" if i == 0 else ("goal" if i == n - 1 else f"stop {i}")
    legs, wps, notes, total, ms = [], [], [], 0.0, 0.0
    for i, (a, b) in enumerate(zip(points, points[1:])):
        p = plan_path(None, amap, a, b, plan_res=plan_res, blocked=blocked)
        ms += p.get("ms") or 0.0
        notes += [f"leg {i + 1}: {t}" for t in p.get("notes", [])]
        if p["status"] != "ok":
            return {"status": f"leg {i + 1} ({name(i)} -> {name(i + 1)}): {p['status']}", "waypoints": wps,
                    "length_m": None, "legs": legs, "notes": notes, "ms": round(ms, 1)}
        legs.append({"waypoints": p["waypoints"], "length_m": p["length_m"]})
        wps += p["waypoints"] if not wps else p["waypoints"][1:]
        total += p["length_m"]
    return {"status": "ok", "waypoints": wps, "length_m": round(total, 4), "legs": legs, "notes": notes,
            "ms": round(ms, 1)}


def pure_pursuit(robot_xy, robot_heading_deg, waypoints, lookahead=0.15):
    """Steering target for a differential-drive robot following waypoints.
    Returns dict(target_xy, distance_to_goal, heading_error_deg, curvature) or None at the goal."""
    if not waypoints:
        return None
    p = np.asarray(robot_xy, float)
    pts = np.asarray(waypoints, float)
    goal_d = float(np.linalg.norm(pts[-1] - p))
    if goal_d < 1e-3:
        return None
    # first point on the path at least `lookahead` away, searching from the closest segment on
    seg_d = [np.linalg.norm(q - p) for q in pts]
    k0 = int(np.argmin(seg_d))
    target = pts[-1]
    for q in pts[k0:]:
        if np.linalg.norm(q - p) >= lookahead:
            target = q
            break
    d = target - p
    h = math.radians(robot_heading_deg)
    local_y = -math.sin(h) * d[0] + math.cos(h) * d[1]
    L = float(np.linalg.norm(d))
    err = math.degrees(math.atan2(d[1], d[0]) - h)
    return {"target_xy": (round(float(target[0]), 4), round(float(target[1]), 4)),
            "distance_to_goal_m": round(goal_d, 4), "heading_error_deg": round(_wrap(err), 2),
            "curvature": round(2 * local_y / max(L * L, 1e-6), 4)}


# --------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------

def load_arena(path=DEFAULT_ARENA_FILE):
    """arena.json is also a normal mapper config (floor_tags, objects) plus an "arena" block."""
    cfg = m.load_config(path)
    cfg["arena"] = arena_settings(cfg)
    return cfg


def save_arena(path, cfg, floor_tags, report):
    out = {k: v for k, v in cfg.items() if k not in ("tag_sizes", "sheet_loaded", "families")}
    out["tag_sizes"] = {}
    out["floor_tags"] = {str(k): v for k, v in floor_tags.items()}
    out["objects"] = {str(k): v for k, v in cfg.get("objects", {}).items()}
    out["arena"] = {k: v for k, v in cfg["arena"].items()}
    out["_arena_report"] = report
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)


def save_map(folder, amap, rect, occ, blocked, plan_cell, extra=None):
    """Write the map for other programs: PNG images, .npy grids and metadata JSON."""
    os.makedirs(folder, exist_ok=True)
    cv2.imwrite(os.path.join(folder, "rectified.png"), rect)
    cv2.imwrite(os.path.join(folder, "occupancy.png"), (~occ).astype(np.uint8) * 255)
    np.save(os.path.join(folder, "occupancy.npy"), occ)
    np.save(os.path.join(folder, "planning_blocked.npy"), blocked)
    meta = {
        "frame": "arena: origin = center of corner tag 1, +x toward corner 2, +y toward corner 4, z up",
        "map_resolution_m": amap.res, "plan_resolution_m": plan_cell,
        "x0": amap.x0, "y0": amap.y0, "x1": amap.x1, "y1": amap.y1,
        "pixel_to_world": "x = x0 + (col + 0.5) * res ; y = y1 - (row + 0.5) * res  (row 0 = top = max y)",
        "occupancy": "occupancy.npy True = blocked (raw). planning_blocked.npy = inflated by robot radius",
        "corners": amap.corners.tolist(),
    }
    meta.update(extra or {})
    with open(os.path.join(folder, "map.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return os.path.abspath(folder)


def save_global_map(folder, gm, occ, blocked, plan_cell, extra=None):
    """Write the global map: color mosaic, occupancy (raw and inflated), known-cells mask and
    the floor detail (mm per camera pixel) of every cell, plus metadata to convert pixels."""
    os.makedirs(folder, exist_ok=True)
    cv2.imwrite(os.path.join(folder, "floor_mosaic.png"), gm.color)
    cv2.imwrite(os.path.join(folder, "occupancy.png"), np.where(~gm.seen, 128, np.where(occ, 0, 255)).astype(np.uint8))
    np.save(os.path.join(folder, "occupancy.npy"), occ)
    np.save(os.path.join(folder, "known.npy"), gm.seen)
    np.save(os.path.join(folder, "planning_blocked.npy"), blocked)
    np.save(os.path.join(folder, "mm_per_camera_px.npy"), gm.mm_per_px().astype(np.float32))
    meta = {
        "frame": "arena: origin = center of the origin corner tag, +x/+y toward its neighbors, z up",
        "map_resolution_m": gm.res, "plan_resolution_m": plan_cell,
        "x0": gm.x0, "y0": gm.y0, "x1": gm.x1, "y1": gm.y1, "width_px": gm.W, "height_px": gm.H,
        "pixel_to_world": "x = x0 + (col + 0.5) * res ; y = y1 - (row + 0.5) * res  (row 0 = top = max y)",
        "occupancy": "True = blocked (obstacle evidence or tagged footprint). known.npy False = never seen. "
                     "occupancy.png: black blocked, white free, gray unknown",
        "corners": gm.corners.tolist(),
    }
    meta.update(extra or {})
    with open(os.path.join(folder, "map.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return os.path.abspath(folder)


def render_global(gm, blocked=None, foot=None, plan_cell=None, now=None, fade_s=2.0):
    """Bird's-eye picture of the global map: floor colors (dimmed where not currently in view),
    unknown dark, obstacles red, robot clearance orange, tagged footprints blue, arena outline."""
    vis = gm.color.copy()
    vis[~gm.seen] = (35, 35, 35)
    if now is not None:
        stale = gm.seen & (now - gm.last_seen > fade_s)
        vis[stale] = (vis[stale] * 0.55).astype(np.uint8)
    tint = vis.copy()
    if blocked is not None and plan_cell:
        bf = grid_to_map(blocked, gm, plan_cell)
        tint[bf & gm.seen] = (0, 165, 255)
    ob = gm.obstacles()
    tint[ob] = (0, 0, 255)
    if foot is not None:
        tint[foot & ~ob] = (255, 120, 0)
    vis = cv2.addWeighted(tint, 0.45, vis, 0.55, 0)
    for step, col in ((0.1, (70, 70, 70)), (0.5, (150, 150, 150))):
        if step / gm.res < 6:
            continue
        for gx in np.arange(math.ceil(gm.x0 / step) * step, gm.x1, step):
            c = int(round(gm.w2p([(gx, 0)])[0][0]))
            cv2.line(vis, (c, 0), (c, gm.H), col, 1)
        for gy in np.arange(math.ceil(gm.y0 / step) * step, gm.y1, step):
            r = int(round(gm.w2p([(0, gy)])[0][1]))
            cv2.line(vis, (0, r), (gm.W, r), col, 1)
    o = np.round(gm.w2p([(0, 0)])[0]).astype(int)
    cv2.line(vis, tuple(o), tuple(np.round(gm.w2p([(0.3, 0)])[0]).astype(int)), (0, 0, 255), 2)
    cv2.line(vis, tuple(o), tuple(np.round(gm.w2p([(0, 0.3)])[0]).astype(int)), (0, 200, 0), 2)
    cv2.polylines(vis, [np.round(gm.w2p(gm.corners)).astype(np.int32)], True, (0, 220, 255), 2, cv2.LINE_AA)
    return vis


# --------------------------------------------------------------------------
# CLI: setup from the live camera
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Set up the 4-corner arena from the live camera.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ps = sub.add_parser("setup", help="survey the 4 corner tags and write arena.json")
    ps.add_argument("--tape", help="tape distances from the origin tag to its two neighbors, "
                    "e.g. 30=1.20,31=0.85 (meters, center to center)")
    ps.add_argument("--corners", default="29,30,31,32", help="the 4 corner ids, origin first (any order)")
    ps.add_argument("--robot-id", type=int)
    ps.add_argument("--robot-radius", type=float, default=0.10)
    ps.add_argument("--config", help="base config with objects (optional)")
    ps.add_argument("--calib")
    ps.add_argument("--focal-mm", type=float, default=28.0)
    ps.add_argument("--sensor-width-mm", type=float, default=35.9)
    ps.add_argument("--camera", default="auto")
    ps.add_argument("--connection", default="usb", choices=m.CONNECTIONS)
    ps.add_argument("--backend", default="auto")
    ps.add_argument("--width-px", dest="width_px", type=int, default=1920)
    ps.add_argument("--height-px", dest="height_px", type=int, default=1080)
    ps.add_argument("--fps", type=int, default=30)
    ps.add_argument("--seconds", type=float, default=2.0)
    ps.add_argument("--out", default=DEFAULT_ARENA_FILE)
    args = ap.parse_args()

    cfg = m.load_config(args.config)
    cfg["arena"] = arena_settings({"arena": {
        "corner_ids": [int(v) for v in args.corners.split(",")],
        "tape_m": {kv.split("=")[0]: float(kv.split("=")[1]) for kv in (args.tape or "").split(",") if "=" in kv},
        "robot_id": args.robot_id, "robot_radius_m": args.robot_radius}})
    cam_args = argparse.Namespace(camera=args.camera, connection=args.connection, backend=args.backend,
                                  width=args.width_px, height=args.height_px, fps=args.fps)
    cam = m.open_camera(cam_args)
    K, D = m.load_intrinsics(argparse.Namespace(calib=args.calib, focal_mm=args.focal_mm,
                                                sensor_width_mm=args.sensor_width_mm), cam.width, cam.height)
    det = m.make_detector(cfg)
    print(f"Collecting {args.seconds:.1f} s. Keep the camera still.")
    samples, fid, t_end = [], 0, time.perf_counter() + args.seconds
    try:
        while time.perf_counter() < t_end:
            frame, fid, _ = cam.latest(fid)
            if frame is not None:
                samples.append(det.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
    finally:
        cam.release()
    size_of = lambda t: cfg["tag_sizes"].get(t, cfg["tag_size"])
    floor_tags, rep, K, D = build_arena(samples, K, D, size_of, cfg["arena"], estimate_f=not args.calib)
    if not args.calib:
        cfg["arena"]["focal_px"] = rep["focal_px"]
        cfg["arena"]["k1"] = rep["k1"]
        cfg["arena"]["focal_at_px"] = [cam.width, cam.height]
    cfg["arena"]["corner_ids"] = rep["corner_ids"]
    cfg["arena"]["layout"] = rep["layout"]
    floor = m.FloorModel(floor_tags, size_of)
    averaged_lock(floor, samples, K, D)
    amap = ArenaMap(cfg["arena"], floor_tags)
    if floor.valid:
        rep["floor_reproj_px"] = round(floor.reproj, 3)
        rep["mm_per_camera_pixel"] = precision_report(floor, amap, K, D)
    print(json.dumps(rep, indent=2))
    save_arena(args.out, cfg, floor_tags, rep)
    print(f"Saved {args.out}. The GUI and `run` load it automatically from this folder.")


if __name__ == "__main__":
    main()
