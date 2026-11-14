#!/usr/bin/env python3
"""Hardware adapter for the bulldozer: Freenove FNK0043 4WD Smart Car (ordinary wheels) + Pi.

Self-contained: talks to the car's shield over I2C directly, so Freenove's own server code
does NOT have to be installed. Needs only:  sudo apt install -y python3-smbus2 i2c-tools
and I2C switched on (sudo raspi-config nonint do_i2c 0, then reboot).
`i2cdetect -y 1` must show 40 (PCA9685 motor driver) and 48 (ADS7830 battery ADC).

    python3 hw_freenove.py wheels     # car on a box, wheels in the air: check each wheel's direction
    python3 hw_freenove.py drive      # forward, back, turn left, turn right (0.8 s each)
    python3 hw_freenove.py battery
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


def load_directions():
    """freenove_directions.json: {"reverse": [wheels that run backwards], "flip_front": bool}.
    flip_front: the motors' 'forward' drives away from the sonar/camera end - swap front and
    back for driving (turning stays as it is: counter-clockwise is the same from either end)."""
    try:
        with open(DIR_FILE) as f:
            d = json.load(f)
        return set(d.get("reverse", [])), bool(d.get("flip_front", False))
    except (OSError, ValueError):
        return set(), False


class Freenove:
    holding = False                      # no gripper

    def __init__(self, min_duty=35, max_duty=65, rot_min=55, rot_max=80, steer=25, pcb=2, bus=1,
                 sonar=True, sonar_trig=27, sonar_echo=22):
        """Duties are in percent (0..100). min_duty: smallest value that moves the car straight;
        rot_min: smallest value that turns it on the spot; steer: left/right difference at w = 1."""
        self.min_duty, self.max_duty, self.rot_min, self.rot_max, self.steer = min_duty, max_duty, rot_min, rot_max, steer
        self.reverse, self.flip_front = load_directions()
        self.pcb, self.lock = pcb, threading.Lock()
        self.bus = SMBus(bus)
        self.bus.write_byte_data(PCA, 0x00, 0x00)                    # MODE1: wake up
        prescale = int(math.floor(25000000.0 / 4096.0 / 50.0 - 1.0 + 0.5))   # 50 Hz, as Freenove sets it
        old = self.bus.read_byte_data(PCA, 0x00)
        self.bus.write_byte_data(PCA, 0x00, (old & 0x7F) | 0x10)     # sleep
        self.bus.write_byte_data(PCA, 0xFE, prescale)
        self.bus.write_byte_data(PCA, 0x00, old)
        time.sleep(0.005)
        self.bus.write_byte_data(PCA, 0x00, old | 0x80)
        self.stop()
        self.sonar = None
        if sonar:
            self._open_sonar(sonar_trig, sonar_echo)

    # ------------------------------------------------------------------ low level
    def _pwm(self, ch, duty):
        r = 0x06 + 4 * ch
        self.bus.write_byte_data(PCA, r, 0)
        self.bus.write_byte_data(PCA, r + 1, 0)
        self.bus.write_byte_data(PCA, r + 2, duty & 0xFF)
        self.bus.write_byte_data(PCA, r + 3, duty >> 8)

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

    # ------------------------------------------------------------- agent interface
    def set_velocity(self, vx, vy, w):
        """Normalised (forward, [left: ignored], counter-clockwise), each -1..1."""
        if self.flip_front:
            vx = -vx                                                  # see load_directions()
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
        self.tank(0, 0)

    def battery_mv(self):
        """Battery voltage from the shield's ADS7830 (channel 2), or None."""
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
        """HC-SR04 on the car's front (Freenove ultrasonic.py: trigger GPIO 27, echo GPIO 22).
        queue_len=3: gpiozero's default median of 9 lags a few hundred ms - too slow for a stop."""
        try:
            import warnings
            from gpiozero import DistanceSensor
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")      # "use pigpio for accuracy": fine for a stop at cm level
                self.sonar = DistanceSensor(echo=echo, trigger=trig, max_distance=SONAR_MAX_M, queue_len=3)
        except Exception as e:                       # no gpiozero / pins busy: drive without the sonar stop
            print(f"  (no sonar: {e})")
            self.sonar = None

    def sonar_cm(self):
        """Distance to whatever is in front of the sonar, cm; SONAR_MAX_M*100 = nothing in range.
        Check after mounting the blade: a constant few cm means the sensor sees the blade itself,
        then start agent.py with --no-sonar."""
        if self.sonar is None:
            return None
        try:
            return round(self.sonar.distance * 100, 1)
        except Exception:
            return None

    def skill(self, name, args, cancel):
        return (name in ("home", "open")), "bulldozer: no arm"

    def close(self):
        try:
            self.stop()
        finally:
            self.bus.close()
            if self.sonar is not None:
                self.sonar.close()


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
                json.dump({"reverse": sorted(rev), "flip_front": car.flip_front}, f)
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
        elif cmd == "battery":
            print(car.battery_mv(), "mV  (if this is far from the real value, try the other --pcb version in agent.py)")
        else:
            print(__doc__)
    finally:
        car.close()


if __name__ == "__main__":
    main()
