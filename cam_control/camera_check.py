#!/usr/bin/env python3
"""
camera_check.py

Checks whether a camera source is good enough for robot tracking (real time, enough resolution),
before it is added as an extra camera. Written for the Insta360 GO 3S, works for any camera.

    python camera_check.py list                                  cameras + the modes each offers
    python camera_check.py bench   --camera Insta360 --fps 30    resolution, frame rate, stalls, tags
    python camera_check.py latency --camera Insta360             glass-to-glass delay (flashing screen)
    python camera_check.py ble     [--name "GO 3S"]              Bluetooth LE: find, connect, list
                                                                 services, measure notify throughput

Insta360 GO 3S: how to get real-time frames into this computer
  USB webcam mode (USE THIS). Camera in the Action Pod, Action Pod's USB-C to the PC with a data
  cable, choose "Webcam" on the camera's screen. It shows up as a normal UVC camera (about
  1920x1080 at 30 fps), so OpenCV reads it like the Sony; expect roughly 100-150 ms glass to glass
  (measure it with `latency`). It does not charge in webcam mode: plan for battery or a powered hub.
  Bluetooth LE (BLE 5.0): control and status only. Insta360 says preview over Bluetooth is reduced
  quality; BLE moves on the order of 100 kB/s in practice, while 1080p30 video needs several Mb/s
  even as H.264 (tens of Mb/s as MJPEG). `ble` measures what the link actually delivers so the
  numbers are yours, not a claim.
  Wi-Fi (5 GHz): the phone app's preview uses a proprietary protocol, and live streaming is not a
  listed GO 3S feature; there is no documented way to pull that stream into OpenCV.

Dependencies: opencv, numpy (+ pupil_apriltags for tag counts in bench); `ble` needs  pip install bleak
"""
import argparse
import asyncio
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _open(args):
    import apriltag_floor_mapper as m
    idx, name = m.resolve_camera(args.camera, "auto")
    cam = m.CameraStream(idx, args.width, args.height, args.fps, m.backend_id(args.backend))
    print(f"camera {idx}" + (f" ({name})" if name else "") + f": asked {args.width}x{args.height}@{args.fps}, "
          f"got {cam.width}x{cam.height}")
    return cam, name


# --------------------------------------------------------------------------
# list
# --------------------------------------------------------------------------

def cmd_list(_args):
    import apriltag_floor_mapper as m
    devs = m.enumerate_cameras()
    if devs is None:
        print("camera names not available here (Windows: pip install pygrabber); try indices 0..5")
        return
    for idx, name in devs:
        modes = m.device_formats(name) or []
        tag = "   <- Insta360" if "insta" in name.lower() else ""
        print(f"{idx}: {name}{tag}")
        if modes:
            print("     modes: " + ", ".join(f"{w}x{h}@{f}" for w, h, f in modes[:8])
                  + (" ..." if len(modes) > 8 else ""))
    if not any("insta" in n.lower() for _, n in devs):
        print("\nNo Insta360 listed. GO 3S: put it in the Action Pod, connect the Pod's USB-C with a DATA cable, "
              "and pick 'Webcam' on the camera's screen when it asks (USB mode prompt).")


# --------------------------------------------------------------------------
# bench
# --------------------------------------------------------------------------

def cmd_bench(args):
    import cv2
    import apriltag_floor_mapper as m
    cam, _ = _open(args)
    det = None
    try:
        cfg = m.load_config(None)
        det = m.make_detector(cfg, nthreads=2, decimate=1.0, realtime=False)
    except Exception as e:                       # noqa: BLE001
        print(f"(tag detection skipped: {e})")
    stamps, sharp, tags, tag_px, det_ms = [], [], [], [], []
    fid, t_end = 0, time.perf_counter() + args.seconds
    print(f"measuring for {args.seconds:.0f} s ... point the camera at the arena / a printed tag")
    while time.perf_counter() < t_end:
        frame, fid, st = cam.latest(fid, timeout=1.0)
        if frame is None:
            stamps.append(None)
            continue
        stamps.append(st)
        if len(stamps) % 5 == 0:                 # image checks on every 5th frame keep up with 60 fps
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharp.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
            if det is not None:
                t0 = time.perf_counter()
                d = det.detect(g)
                det_ms.append((time.perf_counter() - t0) * 1e3)
                tags.append(len(d))
                for _, c in d:
                    tag_px.append(float(np.min(np.linalg.norm(np.roll(c, -1, 0) - c, axis=1))))
    cam.release()
    good = np.array([s for s in stamps if s is not None])
    stalls = sum(1 for s in stamps if s is None)
    if len(good) < 2:
        print("NO FRAMES. Is the camera in webcam / streaming mode and not open in another app?")
        return
    gaps = np.diff(good) * 1e3
    fps = (len(good) - 1) / (good[-1] - good[0])
    print(f"\nresolution      {cam.width}x{cam.height}")
    print(f"frame rate      {fps:.1f} fps delivered (asked {args.fps})")
    print(f"frame interval  median {np.median(gaps):.1f} ms, 95% {np.percentile(gaps, 95):.1f} ms, "
          f"max {gaps.max():.0f} ms; gaps > 3 frames: {int(np.sum(gaps > 3 * np.median(gaps)))}; "
          f"1 s stalls: {stalls}")
    if sharp:
        print(f"sharpness       {np.median(sharp):.0f} (Laplacian variance; < 50 = blurry / out of focus)")
    if tags:
        print(f"tags in view    median {int(np.median(tags))}, detection {np.median(det_ms):.0f} ms per full frame")
    if tag_px:
        print(f"tag edge        smallest {min(tag_px):.0f} px, median {np.median(tag_px):.0f} px "
              "(below ~15 px detection gets unreliable)")
    verdict = []
    if fps < 20:
        verdict.append(f"low frame rate ({fps:.0f} fps)")
    if gaps.max() > 250 or stalls:
        verdict.append("stalls: check the cable / USB port / hub power")
    if cam.width < 1280:
        verdict.append("low resolution")
    print("verdict         " + ("OK for tracking" if not verdict else "; ".join(verdict)))


# --------------------------------------------------------------------------
# latency
# --------------------------------------------------------------------------

def cmd_latency(args):
    """Glass-to-glass delay: the screen flips black <-> white, the camera films it, and the time from
    the flip until the camera's frame shows it is measured. Includes the monitor's own delay
    (~10-30 ms), so compare cameras on the same screen; it is an upper bound for the camera."""
    import cv2
    cam, _ = _open(args)
    win = "latency test: point the camera at this window (Esc = stop)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if not args.windowed:
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    black = np.zeros((480, 640, 3), np.uint8)
    white = np.full_like(black, 255)

    def level(frame):
        h, w = frame.shape[:2]
        return float(frame[h // 4:3 * h // 4, w // 4:3 * w // 4].mean())   # center: where the screen is

    def show(img, secs):
        t_end = time.perf_counter() + secs
        cv2.imshow(win, img)
        k = cv2.waitKey(1)
        t_flip = time.perf_counter()
        return t_flip, t_end, k

    def wait_levels(secs):
        fid, out, t_end = 0, [], time.perf_counter() + secs
        while time.perf_counter() < t_end:
            f, fid, _ = cam.latest(fid, timeout=0.2)
            if f is not None:
                out.append(level(f))
            if cv2.waitKey(1) == 27:
                raise KeyboardInterrupt
        return out

    try:
        print("calibrating brightness (2 s) ...")
        show(black, 0)
        lo = np.median(wait_levels(1.0) or [0])
        show(white, 0)
        hi = np.median(wait_levels(1.0) or [255])
        if hi - lo < 20:
            print(f"the camera barely sees the flashing (black {lo:.0f}, white {hi:.0f}): aim it at the screen, "
                  "fill most of its view, turn auto exposure / dim room lights down")
            return
        thr = (lo + hi) / 2
        delays = []
        state = False
        for k in range(args.flips):
            state = not state
            target = white if state else black
            t_flip, _, key = show(target, 0)
            if key == 27:
                break
            fid, seen = 0, None
            t_end = t_flip + 1.0
            while time.perf_counter() < t_end:
                f, fid, st = cam.latest(fid, timeout=0.2)
                if f is None:
                    continue
                if (level(f) > thr) == state:
                    seen = st
                    break
                cv2.waitKey(1)
            if seen is not None:
                delays.append((seen - t_flip) * 1e3)
            wait_levels(0.25 + 0.1 * (k % 3))              # irregular pace: no lock-in with the frame clock
        if not delays:
            print("no flip was seen within 1 s")
            return
        d = np.array(delays)
        print(f"\nglass-to-glass delay over {len(d)} flips: median {np.median(d):.0f} ms, "
              f"10% {np.percentile(d, 10):.0f} ms, 90% {np.percentile(d, 90):.0f} ms "
              f"(includes the monitor; frame interval {1000 / max(cam.grab_fps, 1):.0f} ms)")
        print("for a robot driving 0.3 m/s, every 100 ms of delay = 3 cm it has moved since the frame")
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        cam.release()


# --------------------------------------------------------------------------
# ble
# --------------------------------------------------------------------------

async def _ble(args):
    try:
        from bleak import BleakClient, BleakScanner
    except ImportError:
        print("pip install bleak")
        return
    print(f"scanning {args.scan:.0f} s for Bluetooth LE devices (wake the camera; close the Insta360 app on "
          "your phone, a camera usually accepts only one BLE connection) ...")
    found = await BleakScanner.discover(timeout=args.scan, return_adv=True)
    rows = sorted(found.values(), key=lambda da: -(da[1].rssi or -999))
    pat = args.name.lower()
    hits = [(d, a) for d, a in rows if pat in ((d.name or a.local_name or "").lower())]
    for d, a in rows[:25]:
        nm = d.name or a.local_name or "?"
        print(f"  {d.address}  {a.rssi:>4} dBm  {nm}" + ("   <- match" if (d, a) in hits else ""))
    if args.address:
        target = next((d for d, _ in rows if d.address.lower() == args.address.lower()), args.address)
    elif hits:
        target = hits[0][0]
    else:
        print(f"\nno device named like '{args.name}'. Insta360 cameras advertise names such as 'GO 3S XXXXXX'; "
              "use --name or --address from the list above.")
        return
    print(f"\nconnecting to {getattr(target, 'address', target)} ...")
    t0 = time.perf_counter()
    async with BleakClient(target, timeout=20.0) as client:
        print(f"connected in {time.perf_counter() - t0:.1f} s, MTU {getattr(client, 'mtu_size', '?')}")
        notify = []
        for svc in client.services:
            print(f"service {svc.uuid}  {svc.description}")
            for ch in svc.characteristics:
                props = ",".join(ch.properties)
                line = f"   char {ch.uuid}  [{props}]  {ch.description}"
                if "read" in ch.properties:
                    try:
                        v = await client.read_gatt_char(ch)
                        line += f"  = {bytes(v[:20]).hex()}" + ("..." if len(v) > 20 else "")
                    except Exception as e:      # noqa: BLE001
                        line += f"  (read: {type(e).__name__})"
                print(line)
                if "notify" in ch.properties or "indicate" in ch.properties:
                    notify.append(ch)
        if not notify or args.listen <= 0:
            return
        stats = {"bytes": 0, "n": 0}

        def cb(_sender, data):
            stats["bytes"] += len(data)
            stats["n"] += 1
        for ch in notify:
            try:
                await client.start_notify(ch, cb)
            except Exception as e:              # noqa: BLE001
                print(f"   notify {ch.uuid}: {type(e).__name__}")
        print(f"\nlistening {args.listen:.0f} s on {len(notify)} notify characteristic(s) "
              "(change a setting or press the button on the camera to make it talk) ...")
        await asyncio.sleep(args.listen)
        rate = stats["bytes"] / args.listen
        print(f"received {stats['n']} notifications, {stats['bytes']} bytes = {rate / 1000:.2f} kB/s")
        print("for comparison, per second: 1080p30 MJPEG ~4000-8000 kB, 1080p30 H.264 ~500-1000 kB, "
              "720p30 H.264 ~250 kB. BLE 5.0 at best ~100-170 kB/s of payload.")


def cmd_ble(args):
    asyncio.run(_ble(args))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    for name in ("bench", "latency"):
        p = sub.add_parser(name)
        p.add_argument("--camera", default="Insta360", help="index or part of the device name")
        p.add_argument("--width", type=int, default=1920)
        p.add_argument("--height", type=int, default=1080)
        p.add_argument("--fps", type=int, default=30)
        p.add_argument("--backend", default="auto", choices=["auto", "dshow", "msmf"])
        if name == "bench":
            p.add_argument("--seconds", type=float, default=10.0)
        else:
            p.add_argument("--flips", type=int, default=20)
            p.add_argument("--windowed", action="store_true", help="no full screen")
    p = sub.add_parser("ble")
    p.add_argument("--name", default="GO 3S", help="part of the advertised name")
    p.add_argument("--address", help="connect to this address instead of a name match")
    p.add_argument("--scan", type=float, default=8.0)
    p.add_argument("--listen", type=float, default=15.0, help="seconds to count notify traffic (0 = skip)")
    args = ap.parse_args()
    {"list": cmd_list, "bench": cmd_bench, "latency": cmd_latency, "ble": cmd_ble}[args.cmd](args)
    sys.stdout.flush()
    os._exit(0)          # pupil_apriltags detectors must not be destructed


if __name__ == "__main__":
    main()
