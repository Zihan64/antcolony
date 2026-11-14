#!/usr/bin/env python3
"""Hardware adapter for the bulldozer: Freenove FNK0043 4WD Smart Car (ordinary wheels) + Pi.

Self-contained: talks to the car's shield over I2C directly, so Freenove's own server code
does NOT have to be installed. Needs only:  sudo apt install -y python3-smbus2 i2c-tools
and I2C switched on (sudo raspi-config nonint do_i2c 0, then reboot).
`i2cdetect -y 1` must show 40 (PCA9685 motor driver) and 48 (ADS7830 battery ADC).

    python3 hw_freenove.py wheels     # car on a box, wheels in the air: check each wheel's direction
    python3 hw_freenove.py drive      # forward, back, turn left, turn right (0.8 s each)
    python3 hw_freenove.py battery

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
# wheel -> (channel held at 0, channel that gets the duty) when driving FORWARD
WHEELS = {"front-left": (0, 1), "rear-left": (3, 2), "front-right": (6, 7), "rear-right": (4, 5)}
LEFT, RIGHT = ("front-left", "rear-left"), ("front-right", "rear-right")


def load_reverse():
    try:
        with open(DIR_FILE) as f:
            return set(json.load(f).get("reverse", []))
    except (OSError, ValueError):
        return set()


class Freenove:
    holding = False                      # no gripper

    def __init__(self, min_duty=35, max_duty=65, rot_min=55, rot_max=80, steer=25, pcb=2, bus=1):
        """Duties are in percent (0..100). min_duty: smallest value that moves the car straight;
        rot_min: smallest value that turns it on the spot; steer: left/right difference at w = 1."""
        self.min_duty, self.max_duty, self.rot_min, self.rot_max, self.steer = min_duty, max_duty, rot_min, rot_max, steer
        self.pcb, self.reverse, self.lock = pcb, load_reverse(), threading.Lock()
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

    def sonar_cm(self):
        return None                      # the blade sits in front of the sensor

    def skill(self, name, args, cancel):
        return (name in ("home", "open")), "bulldozer: no arm"

    def close(self):
        try:
            self.stop()
        finally:
            self.bus.close()


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    car = Freenove()
    try:
        if cmd == "wheels":
            print("Car on a box, wheels in the air. Each wheel runs for 1 s; it should turn FORWARD.")
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
                json.dump({"reverse": sorted(rev)}, f)
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
        elif cmd == "battery":
            print(car.battery_mv(), "mV  (if this is far from the real value, try the other --pcb version in agent.py)")
        else:
            print(__doc__)
    finally:
        car.close()


if __name__ == "__main__":
    main()
