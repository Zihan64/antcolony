"""Waypoint follower for a robot steered from overhead-camera feedback.

Commands are normalised (vx forward, vy left, w counter-clockwise, each -1..1); the robot
agent maps them to motor duty with its own dead-band. Far from the target the robot is
driven continuously; the last few degrees of an in-place turn use short timed pulses,
because with ~0.2 s of camera latency a continuous turn overshoots.
"""
import math

from .common import ZERO, clamp, dist, sgn, to_robot, vel, wrap


class Driver:
    def __init__(self, c, holonomic=False):
        self.c, self.holo = c, holonomic
        self.rate = c["rot_rate0"]     # rad/s of an in-place turn at u_min; re-measured after every pulse
        self.reset()

    def reset(self):
        self.pulse = None              # (time the pulse has settled, heading before, duration)
        self.arrived = False           # position reached (latched until reset or pushed away)
        self.turning = False

    def turn_to(self, pose, th, now):
        """In-place turn to heading `th`. -> (command or None = send nothing, done)"""
        c = self.c
        if self.pulse:
            t_done, th0, dur = self.pulse
            if now < t_done:
                return None, False                     # pulse running / settling: stay quiet
            moved = abs(wrap(pose[2] - th0))
            if moved > 0.01:
                self.rate = clamp(0.5 * self.rate + 0.5 * moved / dur, 0.3, 5.0)
            self.pulse = None
        e = wrap(th - pose[2])
        if abs(e) < c["ang_tol"]:
            return dict(ZERO), True
        if abs(e) > c["fine_ang"]:
            return vel(0, 0, sgn(e) * clamp(0.7 * abs(e), c["u_min"], c.get("w_max", 1.0))), False
        dur = clamp(0.8 * abs(e) / self.rate, 0.06, 0.4)
        self.pulse = (now + dur + c["settle_s"], pose[2], dur)
        return vel(0, 0, sgn(e) * c["u_min"], dur=dur), False

    def follow(self, pose, path, final_th, now):
        """Drive along `path` (reached waypoints are popped), then turn to `final_th`.
        -> (command or None, done)"""
        c = self.c
        while len(path) > 1 and dist(pose, path[0]) < c["wp_tol"]:
            path.pop(0)
        if path:
            last = len(path) == 1
            ex, ey = to_robot(pose, path[0])
            d = math.hypot(ex, ey)
            if last and d < c["pos_tol"]:
                self.arrived = True
            elif self.arrived and d > 2.5 * c["pos_tol"]:
                self.arrived = False
            if not self.arrived:
                a = math.atan2(ey, ex)
                if self.holo and last and d < c.get("holo_within", 0.0):
                    self.pulse, k = None, clamp(c["k_v"] * d, c["u_min"], c["u_max"]) / d
                    return vel(k * ex, k * ey, 0), False
                if abs(a) > (c["ang_tol"] * 1.5 if self.turning else c["turn_first"]):
                    self.turning = True
                    cmd, done = self.turn_to(pose, pose[2] + a, now)
                    if not done:
                        return cmd, False
                self.turning, self.pulse = False, None
                v = clamp(c["k_v"] * d, c["u_min"], c["u_max"]) * max(0.2, math.cos(a))
                wm = c.get("w_max", 1.0)                    # control.w_max: cap on the turn command (real cars spin fast)
                return vel(v, 0, clamp(c["k_w"] * a, -wm, wm)), False
        if final_th is None:
            return dict(ZERO), True
        return self.turn_to(pose, final_th, now)
