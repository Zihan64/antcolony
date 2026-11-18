#!/usr/bin/env python3
"""Hardware adapter for the bulldozer: Freenove FNK0043 4WD Smart Car (ordinary wheels) + Pi.

Self-contained: talks to the car's shield over I2C directly, so Freenove's own server code
does NOT have to be installed. Needs only:  sudo apt install -y python3-smbus2 i2c-tools
and I2C switched on (sudo raspi-config nonint do_i2c 0, then reboot).
`i2cdetect -y 1` must show 40 (PCA9685 motor driver) and 48 (ADS7830 battery ADC).

    python3 hw_freenove.py wheels     # car on a box, wheels in the air: check each wheel's direction
    python3 hw_freenove.py drive      # forward, back, turn left, turn right (0.8 s each)
    python3 hw_freenove.py battery
    python3 hw_freenove.py gimbal     # pan/tilt head (camera + sonar): 7 poses, a photo + sonar reading each
    python3 hw_freenove.py sonar [secs]   # print the front ultrasonic distance (no motion)

Motor channels and the battery formula are taken from Freenove's Code/Server/motor.py and
adc.py (GitHub, commit of 2026-03-12). NOT YET RUN ON THE REAL CAR.

The car has no sideways motion: `vy` is ignored. It turns by skidding, which needs much more
duty than driving straight - that is what --rot-min / --rot-max are for.
"""
import json
import math
import os
import sys
import threading
import time

try:
    from smbus2 import SMBus
except ImportError:                      # Freenove's own setup installs the older `smbus`
    from smbus import SMBus

HERE = os.path.dirname(os.path.abspath(__file__))
DIR_FILE = os.path.join(HERE, "freenove_directions.json")
PCA, ADC = 0x40, 0x48
SONAR_MAX_M = 2.0                         # readings are capped here (no echo = "nothing near")
# wheel -> (channel held at 0, channel that gets the duty) when driving FORWARD
WHEELS = {"front-left": (0, 1), "rear-left": (3, 2), "front-right": (6, 7), "rear-right": (4, 5)}
LEFT, RIGHT = ("front-left", "rear-left"), ("front-right", "rear-right")
IR_PINS = (14, 15, 23)                    # line sensors left, middle, right (Freenove infrared.py)
PAN, TILT = 8, 9                          # PCA9685 channels of the pan/tilt head (Freenove servo '0', '1')
PAN_RANGE, TILT_RANGE = (0, 180), (60, 150)    # servo angles (after pan_center), kept off the stops


def load_directions():
    """freenove_directions.json: {"reverse": [wheels that run backwards], "flip_front": bool,
    "flip_turn": bool}.
    flip_front: the motors' 'forward' drives away from the sonar/camera end - swap front and
    back for driving (turning stays as it is: counter-clockwise is the same from either end).
    flip_turn: + w turns the car clockwise (seen from above) - mirror turning only. Car C needs
    it (2026-11-14: the camera image moved left on a 'turn left', i.e. the car turned right)."""
    try:
        with open(DIR_FILE) as f:
            d = json.load(f)
        return (set(d.get("reverse", [])), bool(d.get("flip_front", False)), bool(d.get("flip_turn", False)),
                float(d.get("pan_center", 90)))
    except (OSError, ValueError):
        return set(), False, False, 90.0


class Freenove:
    holding = False                      # no gripper

    def __init__(self, min_duty=35, max_duty=65, rot_min=55, rot_max=80, steer=25, pcb=2, bus=1,
                 sonar=True, sonar_trig=27, sonar_echo=22):
        """Duties are in percent (0..100). min_duty: smallest value that moves the car straight;
        rot_min: smallest value that turns it on the spot; steer: left/right difference at w = 1."""
        self.released = False
        self._args = dict(min_duty=min_duty, max_duty=max_duty, rot_min=rot_min, rot_max=rot_max, steer=steer,
                          pcb=pcb, bus=bus, sonar=sonar, sonar_trig=sonar_trig, sonar_echo=sonar_echo)
        self.min_duty, self.max_duty, self.rot_min, self.rot_max, self.steer = min_duty, max_duty, rot_min, rot_max, steer
        self.reverse, self.flip_front, self.flip_turn, self.pan_center = load_directions()
        self.pcb, self.lock = pcb, threading.Lock()
        self.bus_no = bus
        self.bus = SMBus(bus)
        self._pca_init()
        self.stop()
        self.look(90, 90)                    # head straight: the sonar stop only makes sense facing forward
        self.sonar = None
        if sonar:
            self._open_sonar(sonar_trig, sonar_echo)

    # ------------------------------------------------------------------ low level
    def _pca_init(self):
        self.bus.write_byte_data(PCA, 0x00, 0x00)                    # MODE1: wake up
        prescale = int(math.floor(25000000.0 / 4096.0 / 50.0 - 1.0 + 0.5))   # 50 Hz, as Freenove sets it
        old = self.bus.read_byte_data(PCA, 0x00)
        self.bus.write_byte_data(PCA, 0x00, (old & 0x7F) | 0x10)     # sleep
        self.bus.write_byte_data(PCA, 0xFE, prescale)
        self.bus.write_byte_data(PCA, 0x00, old)
        time.sleep(0.005)
        self.bus.write_byte_data(PCA, 0x00, old | 0x80)

    def _write4(self, r, duty):
        self.bus.write_byte_data(PCA, r, 0)
        self.bus.write_byte_data(PCA, r + 1, 0)
        self.bus.write_byte_data(PCA, r + 2, duty & 0xFF)
        self.bus.write_byte_data(PCA, r + 3, duty >> 8)

    def _pwm(self, ch, duty):
        """One PCA9685 channel. If the board does not answer (car C, 2026-11-15: Remote I/O error when a
        fast shove pulled the supply down and the board reset), wake it up again and write once more."""
        r = 0x06 + 4 * ch
        for attempt in range(6):
            try:
                if attempt:
                    try:
                        self.bus.close()
                    except OSError:
                        pass
                    self.bus = SMBus(self.bus_no)
                    self._pca_init()                                 # a reset board sleeps with all outputs off
                    if getattr(self, "_head", None):                 # ...the head servos too: it went limp, so
                        pan, tilt = self._head                       # every later look saw the same picture
                        self._write4(0x06 + 4 * PAN, self._servo_count(PAN, pan))
                        self._write4(0x06 + 4 * TILT, self._servo_count(TILT, tilt))
                self._write4(r, duty)
                return
            except OSError:
                if attempt == 5:
                    raise
                time.sleep(0.3)                                      # the board needs a moment after a brown-out

    def wheel(self, name, pct):
        """One wheel at pct percent duty, + = forward."""
        if name in self.reverse:
            pct = -pct
        duty = int(max(-100.0, min(100.0, pct)) * 40.95)
        a, b = WHEELS[name]
        if duty > 0:
            self._pwm(a, 0), self._pwm(b, duty)
        elif duty < 0:
            self._pwm(b, 0), self._pwm(a, -duty)
        else:
            self._pwm(a, 4095), self._pwm(b, 4095)                   # both high = brake

    def tank(self, left, right):
        with self.lock:
            for n in LEFT:
                self.wheel(n, left)
            for n in RIGHT:
                self.wheel(n, right)

    def _servo_count(self, ch, angle):
        angle = max(PAN_RANGE[0], min(PAN_RANGE[1], angle)) if ch == PAN else max(TILT_RANGE[0], min(TILT_RANGE[1], angle))
        us = 2500 - (angle + 10) / 0.09 if ch == PAN else 500 + (angle + 10) / 0.09
        return int(us * 4096 / 20000)

    def servo(self, ch, angle):
        """Pan/tilt head (camera + sonar ride on it). ch 8 = pan, 9 = tilt; 90 = straight ahead /
        level. Same angle -> pulse formula as Freenove's servo.py (pan runs mirrored)."""
        angle = max(PAN_RANGE[0], min(PAN_RANGE[1], angle)) if ch == PAN else max(TILT_RANGE[0], min(TILT_RANGE[1], angle))
        us = 2500 - (angle + 10) / 0.09 if ch == PAN else 500 + (angle + 10) / 0.09
        with self.lock:
            self._pwm(ch, int(us * 4096 / 20000))

    def line(self):
        """The three IR line sensors under the front, (left, middle, right), 1 = dark line.
        GPIO 14, 15, 23 as in Freenove's infrared.py; opened on first use."""
        if getattr(self, "_ir", None) is None:
            from gpiozero import LineSensor
            self._ir = [LineSensor(p) for p in IR_PINS]
        return tuple(1 if s.value else 0 for s in self._ir)

    def look(self, pan=90, tilt=90):
        """Point the head; pan 90 = straight ahead, < 90 = to the left of the car, tilt > 90 = up
        (checked on car C with `gimbal`). pan_center in freenove_directions.json = the servo angle
        that really faces forward (car C: the horn sits ~40 deg off)."""
        self._head = (pan + self.pan_center - 90, tilt)
        self.servo(PAN, pan + self.pan_center - 90)
        self.servo(TILT, tilt)

    # ------------------------------------------------------------- agent interface
    def set_velocity(self, vx, vy, w):
        """Normalised (forward, [left: ignored], counter-clockwise), each -1..1."""
        if self.flip_front:
            vx = -vx                                                  # see load_directions()
        if self.flip_turn:
            w = -w
        a = min(1.0, abs(vx))
        if a > 0.02:
            base = math.copysign(self.min_duty + (self.max_duty - self.min_duty) * a, vx)
            turn = max(-1.0, min(1.0, w)) * self.steer
            self.tank(base - turn, base + turn)
        elif abs(w) > 0.02:
            d = self.rot_min + (self.rot_max - self.rot_min) * min(1.0, abs(w))
            self.tank(-math.copysign(d, w), math.copysign(d, w))     # + w = turn left
        else:
            self.stop()

    def stop(self):
        if getattr(self, "released", False):                 # the clean skill's process owns the motors now
            return
        self.tank(0, 0)

    def battery_mv(self):
        """Battery voltage from the shield's ADS7830 (channel 2), or None."""
        if self.released:
            return None
        try:
            with self.lock:
                self.bus.write_byte(ADC, 0x84 | ((((2 << 2) | (2 >> 1)) & 0x07) << 4))
                for _ in range(5):
                    v1, v2 = self.bus.read_byte(ADC), self.bus.read_byte(ADC)
                    if v1 == v2:
                        break
            volts = v1 / 255.0 * (3.3 if self.pcb == 1 else 5.2) * (3 if self.pcb == 1 else 2)
            return int(volts * 1000)
        except OSError:
            return None

    def _open_sonar(self, trig, echo):
        """HC-SR04 on the head (Freenove ultrasonic.py: trigger GPIO 27, echo GPIO 22), timed here.
        Not gpiozero's DistanceSensor: when no echo comes back (open space beyond SONAR_MAX_M, or a
        slanted surface that deflects the sound) it keeps returning its last value - on car C
        (2026-11-14) it read the same 70-88 cm for minutes while the car drove, so the obstacle stop
        and the stuck check acted on a stale number."""
        try:
            from gpiozero import DigitalInputDevice, DigitalOutputDevice
            self.sonar = (DigitalOutputDevice(trig), DigitalInputDevice(echo))
        except Exception as e:                       # no gpiozero / pins busy: drive without the sonar stop
            print(f"  (no sonar: {e})")
            self.sonar = None

    def _ping(self):
        """One measurement in cm; SONAR_MAX_M*100 when no echo comes back in time."""
        trig, echo = self.sonar
        far = SONAR_MAX_M * 100
        trig.on()
        time.sleep(0.00001)
        trig.off()
        t0 = time.perf_counter()
        while not echo.value:                        # echo starts ~0.5 ms after the trigger
            if time.perf_counter() - t0 > 0.02:
                return far
        start = time.perf_counter()
        limit = start + 2 * SONAR_MAX_M / 343.0
        while echo.value:
            if time.perf_counter() > limit:
                return far
        return min(far, (time.perf_counter() - start) * 34300 / 2)

    def sonar_cm(self):
        """Distance to whatever is in front of the sonar, cm (median of 3 fresh pings, ~30 ms);
        SONAR_MAX_M*100 = nothing in range. Check after mounting the blade: a constant few cm means
        the sensor sees the blade itself, then start agent.py with --no-sonar."""
        if self.sonar is None:
            return None
        try:
            r = []
            for _ in range(3):
                r.append(self._ping())
                time.sleep(0.006)                    # let the previous ping's echoes die out
            return round(sorted(r)[1], 1)
        except Exception:
            return None

    def skill(self, name, args, cancel):
        if name == "clean":
            return self.clean(args, cancel)
        return (name in ("home", "open")), "bulldozer: no arm"

    def clean(self, args, cancel):
        """Skill "clean": cleaner.py (another process: it needs the camera, the I2C bus and the sonar
        pins, so this object lets go of them meanwhile and takes them back after). Its YOLO servers
        (yolo_pi.py, the car's own Pi) are started here if they are not up. -> (ok, "<n> pieces in the box")."""
        import re
        import signal as sig
        import subprocess
        import urllib.request
        for port, model, conf in ((11447, "models/warehouse_yolo26.onnx", 0.3), (11448, "models/box.onnx", 0.15)):
            try:
                urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/detect", b"{}"), timeout=2)
            except urllib.error.HTTPError:
                continue                                     # 500 to an empty request = the server is up
            except OSError:
                subprocess.Popen([os.path.expanduser("~/yolo_env/bin/python"), "yolo_pi.py", "serve", "--model", model,
                                  "--port", str(port), "--conf", str(conf)], cwd=HERE, start_new_session=True,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                time.sleep(6)
        cmd = [sys.executable, "-u", os.path.join(HERE, "cleaner.py"), "run", "--fast",
               "--minutes", str(float(args.get("minutes", 15))), "--peers", args.get("peers", "/tmp/antcolony_peers.json"),
               "--yolo-url", "http://127.0.0.1:11448/detect", "--yolo3-url", "http://127.0.0.1:11447/detect"]
        if args.get("brain_url"):
            cmd += ["--brain-url", args["brain_url"]]
        self.close()
        cleaned, last = 0, ""
        try:
            p = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

            def watch():                                      # a cancel must not wait for cleaner.py's next line
                while p.poll() is None:
                    if cancel.wait(0.1):
                        p.send_signal(sig.SIGTERM)            # cleaner.py stops the motors on SIGTERM
                        return
            threading.Thread(target=watch, daemon=True).start()
            for line in p.stdout:
                print("  [clean]", line.rstrip(), flush=True)
                last = line.strip() or last
                m = re.search(r"trash #(\d+) is in the box", line)
                if m:
                    cleaned = int(m[1])
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        finally:
            self.__init__(**self._args)                       # take the hardware back
        if cancel.is_set():
            return False, f"cancelled - {cleaned} pieces in the box"
        if p.returncode < 0 or "FINISHED" not in last:
            return False, f"stopped early - {cleaned} pieces in the box ({last[:80]})"
        return p.returncode == 0, f"{cleaned} pieces in the box" + ("" if p.returncode == 0 else f" ({last[:80]})")

    def close(self):
        if getattr(self, "released", False):
            return
        try:
            self.stop()
        finally:
            self.released = True
            self.bus.close()
            for d in self.sonar or ():
                d.close()
            for s in getattr(self, "_ir", None) or ():
                s.close()
            self.sonar, self._ir = None, None


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    car = Freenove()
    try:
        if cmd == "wheels":
            print("Car on a box, wheels in the air. Each wheel runs for 1 s; it should turn FORWARD.\n"
                  "FRONT = the end with the ultrasonic sensor and the camera (and later the blade).")
            rev = set()
            for n in WHEELS:
                car.reverse = set()
                with car.lock:
                    car.wheel(n, 50)
                time.sleep(1.0)
                car.stop()
                ans = input(f"  {n}: turned forward? [y = yes / n = it went backwards] ").strip().lower()
                if ans.startswith("n"):
                    rev.add(n)
            with open(DIR_FILE, "w") as f:
                json.dump({"reverse": sorted(rev), "flip_front": car.flip_front, "flip_turn": car.flip_turn,
                           "pan_center": car.pan_center}, f)
            print(f"saved {DIR_FILE}: reversed = {sorted(rev) or 'none'}. "
                  "If a DIFFERENT wheel moved than the one named, the motor plugs are swapped on the shield.")
        elif cmd == "drive":
            for name, v in (("forward", (0.5, 0, 0)), ("back", (-0.5, 0, 0)),
                            ("turn left", (0, 0, 0.5)), ("turn right", (0, 0, -0.5))):
                print(name)
                car.set_velocity(*v)
                time.sleep(0.8)
                car.stop()
                time.sleep(0.5)
        elif cmd == "sonar":
            secs = float(sys.argv[2]) if len(sys.argv) > 2 else 10
            print(f"sonar for {secs:.0f} s (move a hand or a box in front of the car)")
            t_end = time.time() + secs
            while time.time() < t_end:
                print(f"  {car.sonar_cm()} cm", flush=True)
                time.sleep(0.2)
        elif cmd == "gimbal":
            # each pose: sonar reading + a photo in gimbal/ (shows which way the head really points)
            from picamera2 import Picamera2
            os.makedirs(os.path.join(HERE, "gimbal"), exist_ok=True)
            cam = Picamera2()
            cam.configure(cam.create_still_configuration(main={"size": (1152, 648)}))
            cam.start()
            try:
                for pan, tilt in ((90, 90), (40, 90), (140, 90), (90, 90), (90, 65), (90, 130), (90, 90)):
                    car.look(pan, tilt)
                    time.sleep(0.8)
                    path = os.path.join(HERE, "gimbal", f"pan{pan}_tilt{tilt}.jpg")
                    cam.capture_file(path)
                    print(f"pan {pan:3d} tilt {tilt:3d}  sonar {car.sonar_cm()} cm  -> {path}", flush=True)
            finally:
                cam.stop()
        elif cmd == "battery":
            print(car.battery_mv(), "mV  (if this is far from the real value, try the other --pcb version in agent.py)")
        else:
            print(__doc__)
    finally:
        car.close()


if __name__ == "__main__":
    main()
