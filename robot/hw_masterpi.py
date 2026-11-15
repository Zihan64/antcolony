"""Hardware adapter: agent.py <-> the team's masterpi_lite library and vlm_grab.py.

Lives next to vlm_grab.py. Driving uses MecanumChassis.set_velocity directly. Grabbing runs
the existing, tested `vlm_grab.py approach` as a subprocess; because that program opens the
serial port itself, this adapter closes its own Board first and re-opens it afterwards.

NOT YET RUN ON A REAL CAR (written from reading the repo on 2026-11-13). Check on the bench:
  1. the gripper keeps holding while the serial port is closed and re-opened after a grab;
  2. the carry pose and the drop pose (CARRY, and y_cm/z_cm from config.json) clear your bin;
  3. --min-duty / --rate-min really are the smallest values that move the car on the venue floor.
"""
import math
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def reach_offset():
    """cm this car's arm stops short of where it is sent: "--y-offset" in grab_defaults.json next to
    this file (car A: 1.8, measured 2026-11-14; no file = 0). grab_runner.py gives it to the grab,
    drop() adds it here - without it car A let go only 2-3 cm inside the tray's rim."""
    try:
        import json
        with open(os.path.join(HERE, "grab_defaults.json")) as f:
            return float(json.load(f).get("--y-offset", 0) or 0)
    except (OSError, ValueError, TypeError, AttributeError):
        return 0.0
CARRY = (0, 12, 12)          # cm, gripper-tip pose while driving with an object
# Pose while driving WITHOUT an object: the same forward pose, not arm.home(). HOME (0, 6, 18)
# folds the arm back over the chassis - elbow 6 cm and wrist 4 cm BEHIND the base axis, 21 cm
# up - and an overhead camera that looks at a slant then loses the tag on top of the car as
# soon as the car turns its front towards the camera (seen 2026-11-14: calibration lost the
# tag after the turn). In CARRY every joint is in front of the base axis, above the sonar's view.
TRANS_EPS, ROT_EPS = 0.02, 0.02


class MasterPi:
    def __init__(self, min_duty=30, max_duty=45, rate_min=0.20, rate_max=0.36, arm=True):
        self.min_duty, self.max_duty, self.rate_min, self.rate_max = min_duty, max_duty, rate_min, rate_max
        self.use_arm, self.holding = arm, False
        self.lock = threading.RLock()        # serial port owner
        self.board = self.car = self.arm = self.sonar = None
        self._open()
        if self.use_arm:
            self.rest(1500)

    # ------------------------------------------------------------------ serial port
    def _open(self):
        sys.path.insert(0, HERE)
        from masterpi_lite import Arm, Board, MecanumChassis
        self.board = Board()
        self.board.enable_reception()        # battery voltage
        self.car = MecanumChassis(self.board)
        self.arm = Arm(self.board) if self.use_arm else None
        if self.sonar is None:
            try:
                from masterpi_lite.sonar import Sonar
                self.sonar = Sonar()
            except Exception as e:
                print(f"  (no sonar: {e})")
                self.sonar = False

    def _close(self):
        if self.board is not None:
            try:
                self.car.stop()
            finally:
                self.board.close()
            self.board = self.car = self.arm = None

    def close(self):
        with self.lock:
            self._close()

    def rest(self, ms=1500):
        """Arm forward and compact, then gripper closed: the pose for driving (see CARRY above).
        Blocks until done. The jaws close only AFTER the arm has moved, so that right after a
        drop they do not close on the object that was just let go."""
        with self.lock:
            r = self.arm.move_to(*CARRY, pitch=-90, pitch_min=-90, pitch_max=0, ms=ms)
            if r is None:                        # should not happen: fall back to the factory pose
                self.arm.home(ms)
        time.sleep(ms / 1000.0 + 0.1)
        with self.lock:
            self.arm.gripper(True)
        time.sleep(0.5)

    def carry_pose(self, ms=1200):
        """Arm to the compact driving pose; the gripper is left as it is (open or holding)."""
        with self.lock:
            self.arm.move_to(*CARRY, pitch=-90, pitch_min=-90, pitch_max=0, ms=ms)
        time.sleep(ms / 1000.0 + 0.1)

    # ---------------------------------------------------------------------- driving
    def set_velocity(self, vx, vy, w):
        """Normalised (forward, left, counter-clockwise) -> duty / direction / rate.
        Any non-zero translation gets at least min_duty (below it the motors only hum);
        a pure turn gets at least rate_min; a turn while driving is passed through linearly."""
        m = min(1.0, math.hypot(vx, vy))
        with self.lock:
            if self.car is None:
                return
            if m > TRANS_EPS:
                duty = self.min_duty + (self.max_duty - self.min_duty) * m
                direction = math.degrees(math.atan2(vx, -vy)) % 360      # 90 = forward, 180 = left
                rate = -w * self.rate_max                                # chassis: + = clockwise
            elif abs(w) > ROT_EPS:
                duty, direction = 0, 90
                rate = -math.copysign(self.rate_min + (self.rate_max - self.rate_min) * min(1.0, abs(w)), w)
            else:
                self.car.stop()
                return
            self.car.set_velocity(duty, direction, rate)

    def stop(self):
        with self.lock:
            if self.car is not None:
                self.car.stop()

    def battery_mv(self):
        b = self.board
        return b.get_battery() if b is not None else None

    def sonar_cm(self):
        if not self.sonar:
            return None
        d = self.sonar.distance_mm()
        return None if d is None else d / 10.0

    # ----------------------------------------------------------------------- skills
    def skill(self, name, args, cancel):
        if name == "grab":
            return self.grab(args, cancel)
        if name == "drop":
            return self.drop(args)
        if name == "carry":
            return self.carry(args, cancel)
        if name == "seek":                       # find a named landmark with the camera + VLM, drive up to it
            import seek
            return seek.run(self, args, cancel)
        if name == "dock":                       # the last cm up to a low white tray (camera, calibrated look pose)
            import seek
            return seek.dock(self, args, cancel)
        if name == "open":
            with self.lock:
                self.arm.open(600)
            self.holding = False
            return True, "open"
        if name == "home":                       # "home" = the driving pose (arm forward), not arm.home()
            self.rest(1500)
            self.holding = False
            return True, "rest pose"
        if name == "fold":                       # the factory pose, arm folded back (hides the tag from a slanted camera)
            with self.lock:
                self.arm.home(1500)
            time.sleep(1.6)
            return True, "folded"
        return False, f"unknown skill {name}"

    def grab(self, args, cancel):
        """The coordinator has parked the car ~30 cm in front of the object, facing it.
        `vlm_grab.py approach` does the rest: find it, creep up, align, grip, lift, miss check."""
        if not self.use_arm:
            return False, "no arm"
        label = args.get("label") or "any"
        # grab_runner.py = vlm_grab.py with the object finder from detector.json (VLM or local YOLO)
        runner = os.path.join(HERE, "grab_runner.py")         # (an empty one = a copy that broke off: ignore it)
        script = "grab_runner.py" if os.path.isfile(runner) and os.path.getsize(runner) > 100 else "vlm_grab.py"
        cmd = [sys.executable, os.path.join(HERE, script), "approach", label,
               "--min-speed", str(args.get("min_speed", self.min_duty))]
        if args.get("obj_height"):               # a known block: skips vlm_grab's "is this a block at all?" guess,
            cmd += ["--obj-height", str(args["obj_height"])]     # which coloured light can fool (2026-11-14)
        with self.lock:
            self._close()                        # vlm_grab.py opens the serial port itself
            try:
                p = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                t0 = time.monotonic()
                while p.poll() is None:
                    if cancel.is_set() or time.monotonic() - t0 > 120:
                        p.terminate()
                        break
                    time.sleep(0.1)
                out = (p.communicate(timeout=5)[0] or "").strip().splitlines()
            finally:
                self._open()
                self.car.stop()                  # a killed vlm_grab.py (timeout / cancel) leaves the motors as they were
        last = out[-1] if out else ""
        try:                                     # everything vlm_grab printed: the agent log keeps only the last line
            with open(os.path.join(HERE, "grab.log"), "a", encoding="utf-8") as f:
                f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')}  {' '.join(cmd[1:])}  -> exit {p.returncode}\n"
                        + "\n".join(out) + "\n")
        except OSError:
            pass
        # exit 0 + a final "done..." line = holding; exit 2 = jaws closed on nothing; 1 = not seen / not reached
        ok = p.returncode == 0 and last.startswith("done")
        self.holding = ok
        if ok:
            with self.lock:                      # lift to a compact pose for the drive to the bin
                self.arm.move_to(*CARRY, pitch=-90, pitch_min=-90, pitch_max=0, ms=1200)
            time.sleep(1.3)
        else:
            self.rest(1500)                      # closes the (empty) gripper; arm forward so the tag stays visible
        return ok, last or f"exit {p.returncode}"

    def drop(self, args):
        """Reach over the bin rim and let go. Default tip target: 18 cm ahead of the arm base, 10 cm up."""
        if not self.use_arm:
            return False, "no arm"
        x, y, z = float(args.get("x_cm", 0)), float(args.get("y_cm", 18)), float(args.get("z_cm", 10))
        off = reach_offset()                     # this car's arm stops short (grab_defaults.json): reach that much
        r0 = math.hypot(x, y)                    # farther along the same line, as the grab does
        if off and r0 > 1:
            x, y = x * (r0 + off) / r0, y * (r0 + off) / r0
        with self.lock:                          # x: cm to the right of the car's line (the arm base turns)
            r = self.arm.move_to(x, y, z, pitch=-45, pitch_min=-90, pitch_max=30, ms=1500)
        if r is None:
            return False, f"drop pose ({x},{y},{z}) out of reach"
        time.sleep(1.7)
        with self.lock:
            self.arm.open(600)
        time.sleep(0.9)
        self.holding = False
        self.rest(1500)                          # lift first, close the jaws afterwards
        return True, "dropped"

    # ------------------------------------------------------- carry: forward, round obstacles
    def sonar_med(self, n=3, gap=0.04):
        """Median of n front-sonar readings in cm; None if not one came back."""
        v = []
        for _ in range(n):
            d = self.sonar_cm()
            if d is not None:
                v.append(d)
            time.sleep(gap)
        return sorted(v)[len(v) // 2] if v else None

    def pulse(self, vx, vy, secs, cancel):
        """One short timed move, then stand still (the sonar is read at rest)."""
        self.set_velocity(vx, vy, 0.0)
        t_end = time.monotonic() + secs
        while time.monotonic() < t_end and not cancel.is_set():
            time.sleep(0.01)
        self.stop()
        time.sleep(0.25)

    def cruise(self, u, cm, cancel, guard_cm=None, cm_s=22.0, ramp_s=0.075):
        """One smooth forward move of about `cm` instead of stop-and-go pulses, for stretches with
        nothing near. The front sonar is read on the way: two readings in a row under guard_cm end it.
        -> cm driven, from the time (22 cm/s at this duty once rolling; a 0.33 s pulse = 5.7 cm)."""
        secs, t0, low = cm / cm_s + ramp_s, time.monotonic(), 0
        self.set_velocity(u, 0.0, 0.0)
        while time.monotonic() - t0 < secs and not cancel.is_set():
            time.sleep(0.05)
            if guard_cm:
                d = self.sonar_cm()
                low = low + 1 if (d is not None and d < guard_cm) else 0
                if low >= 2:
                    break
        self.stop()
        t = time.monotonic() - t0
        time.sleep(0.25)
        return max(0.0, cm_s * (t - ramp_s))

    def carry(self, args, cancel):
        """Drive `dist_cm` forward (default 20) with whatever the gripper holds, going round
        anything tall in the way. No tracker: the distance is counted from the drive pulses
        (`cm_s`, measured on car B on carpet: a 0.33 s pulse at duty 30 = 5.7 cm), so expect
        +-20 %.

        Between pulses the car stands still and reads the front sonar. Something nearer than
        the rest of the way + `margin_cm` (the gripper and the object stick out in front of the
        sonar) is in the way: the car slides sideways, pulse by pulse, until the sonar is
        clear, then `extra_cm` more for its own half width, and drives on. If one side does not
        clear within `max_side_cm` of the starting line it comes back and tries the other
        side; if that fails too it returns to the line and reports failure, still holding.

        Limits: the sonar sees only things about 10 cm tall or more, straight ahead; there is
        no sensor to the sides, so the side it slides to must be free."""
        dist = float(args.get("dist_cm", 20))
        cm_s, side_cm_s = float(args.get("cm_s", 17)), float(args.get("side_cm_s", 15))
        step = float(args.get("step_s", 0.33))
        margin, extra = float(args.get("margin_cm", 15)), float(args.get("extra_cm", 10))
        max_side, look = float(args.get("max_side_cm", 30)), float(args.get("look_cm", 25))
        u = float(args.get("u", 0.05))                       # just above min_duty: slow and predictable
        first = -1 if args.get("side") == "right" else 1     # + = left
        only = args.get("side") in ("left", "right") and args.get("strict")
        t0, done, side, seen = time.monotonic(), 0.0, 0.0, None
        say = lambda msg: print(f"  carry: {msg}", flush=True)

        def read():
            for _ in range(5):                               # a failed I2C read now and then is normal
                d = self.sonar_med()
                if d is not None:
                    return d
                time.sleep(0.2)
            return None

        def slide(sgn, room):
            """Slide towards sgn until the sonar reads `room`+5 or the side limit. -> cleared?"""
            nonlocal side
            while not cancel.is_set() and side * sgn + side_cm_s * step <= max_side:
                self.pulse(0.0, sgn * u, step, cancel)
                side += sgn * side_cm_s * step
                d = read()
                if d is None or d < room + 5:
                    continue
                for _ in range(max(1, round(extra / (side_cm_s * step)))):         # half the car's width
                    if side * sgn + side_cm_s * step > max_side:
                        break
                    self.pulse(0.0, sgn * u, step, cancel)
                    side += sgn * side_cm_s * step
                d = read()
                if d is not None and d >= room:              # (a second thing behind the first: keep sliding)
                    return True
            return False

        def back_to(target):
            nonlocal side
            while not cancel.is_set() and abs(side - target) > side_cm_s * step / 2:
                sgn = 1 if target > side else -1
                self.pulse(0.0, sgn * u, step, cancel)
                side += sgn * side_cm_s * step

        d = read()
        if d is None:
            return False, "no sonar reading - not driving blind"
        if d < 6:
            return False, f"sonar reads {d:.0f} cm before moving - does the held object hang in front of it?"
        say(f"{dist:.0f} cm to go, sonar {d:.0f} cm")
        while dist - done > 1.0:
            if cancel.is_set():
                return False, f"cancelled after {done:.0f} cm"
            if time.monotonic() - t0 > float(args.get("timeout", 60)):
                return False, f"timeout after {done:.0f} cm"
            rest = dist - done
            d = read()
            if d is None:
                return False, f"sonar stopped answering after {done:.0f} cm"
            room = min(rest, look) + margin
            if d >= room:
                t = min(step, max(0.12, rest / cm_s))
                self.pulse(u, 0.0, t, cancel)
                done += cm_s * t
                say(f"forward, {done:.0f} of {dist:.0f} cm (sonar was {d:.0f})")
                continue
            seen = d if seen is None else seen
            start = side
            say(f"something {d:.0f} cm ahead (need {room:.0f}) - sliding {'left' if first > 0 else 'right'}")
            if slide(first, room):
                say(f"clear, {abs(side):.0f} cm to the {'left' if side > 0 else 'right'} of the starting line")
                continue
            if cancel.is_set():
                continue
            back_to(start)
            if not only:
                say(f"not clear on that side - trying the {'right' if first > 0 else 'left'}")
                if slide(-first, room):
                    first = -first                           # keep going round on the side that worked
                    say(f"clear, {abs(side):.0f} cm to the {'left' if side > 0 else 'right'} of the starting line")
                    continue
                if cancel.is_set():
                    continue
                back_to(start)
            return False, f"blocked {d:.0f} cm ahead after {done:.0f} cm, no way round within {max_side:.0f} cm"
        detail = f"carried {done:.0f} cm"
        if seen is not None:
            detail += f", {abs(side):.0f} cm {'left' if side > 0 else 'right'} round something {seen:.0f} cm ahead"
        return True, detail
