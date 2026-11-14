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
        y, z = float(args.get("y_cm", 18)), float(args.get("z_cm", 10))
        with self.lock:
            r = self.arm.move_to(0, y, z, pitch=-45, pitch_min=-90, pitch_max=30, ms=1500)
        if r is None:
            return False, f"drop pose (0,{y},{z}) out of reach"
        time.sleep(1.7)
        with self.lock:
            self.arm.open(600)
        time.sleep(0.9)
        self.holding = False
        self.rest(1500)                          # lift first, close the jaws afterwards
        return True, "dropped"
