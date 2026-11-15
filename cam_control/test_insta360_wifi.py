#!/usr/bin/env python3
"""
Tests for insta360_wifi.py against a FAKE Insta360 camera that speaks the camera side of the Wi-Fi
protocol (sync, keepalive, StartLiveStream -> 200, length-prefixed video packets). It streams the
synthetic arena (known robot position) as a raw MPEG-4 elementary stream, because OpenCV can only
encode MPEG-4 here; a real camera sends H.265/H.264, which the same FFmpeg path decodes.

    python test_insta360_wifi.py

What this proves: framing, handshake, command/response, video extraction (two header layouts),
relay + decoding, delay, tags located through the codec, the tracker's extra-camera path
(camera=insta-wifi:host:port), and clean errors (refused start, camera disconnect).
What it cannot prove: that a real GO 3S speaks this protocol. Run `insta360_wifi.py probe` for that.
"""
import math
import os
import socket
import struct
import sys
import threading
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import apriltag_floor_mapper as m  # noqa: E402
import insta360_wifi as iw  # noqa: E402
import multi_camera as mc  # noqa: E402
from apriltag_tracker_gui import SyntheticScene  # noqa: E402

FPS = 30
TMP = os.path.join(os.environ.get("TEMP", "."), "fake_insta360.avi")


def make_stream(n=90):
    """Synthetic arena frames -> raw MPEG-4 frames (+ robot truth per frame). Frame number is coded
    in a corner patch (brightness) so the receive side can measure the delay."""
    sc = SyntheticScene()
    w = cv2.VideoWriter(TMP, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (sc.W, sc.H))
    truth = []
    for i in range(n):
        sc.t0 = time.perf_counter() - i / FPS
        f, t = sc.read()
        f[:40, :40] = (i % 50) * 5
        w.write(f)
        truth.append(t[90])
    w.release()
    avi = open(TMP, "rb").read()
    chunks, k = [], avi.index(b"movi")
    while len(chunks) < n:
        k = avi.find(b"00dc", k)
        size = struct.unpack("<I", avi[k + 4:k + 8])[0]
        if size:
            chunks.append(avi[k + 8:k + 8 + size])
        k += 8 + size + (size & 1)
    return sc, chunks, truth


class FakeInsta360:
    """Camera side of the protocol on 127.0.0.1:<port>."""

    def __init__(self, chunks, flags_len=8, refuse=False, drop_after_s=None):
        self.chunks, self.flags_len, self.refuse, self.drop_after_s = chunks, flags_len, refuse, drop_after_s
        self.srv = socket.socket()
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(4)
        self.port = self.srv.getsockname()[1]
        self.sent = {}                      # frame number (mod 50) -> time sent
        self.commands = []
        self.running = True
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while self.running:
            try:
                c, _ = self.srv.accept()
            except OSError:
                return
            threading.Thread(target=self._client, args=(c,), daemon=True).start()

    @staticmethod
    def _send(c, lock, payload):
        with lock:
            c.sendall(struct.pack("<I", len(payload) + 4) + payload)

    def _client(self, c):
        lock = threading.Lock()
        streaming = threading.Event()
        t_conn = time.perf_counter()

        def stream():
            i = 0
            while streaming.is_set() and self.running:
                if self.drop_after_s and time.perf_counter() - t_conn > self.drop_after_s:
                    c.close()                          # camera goes away (out of range / off)
                    return
                ch = self.chunks[i % len(self.chunks)]
                head = b"\x01\x00\x00" + struct.pack("<I", int(i * 1000 / FPS)) + b"\x00" * self.flags_len
                self.sent[(i % len(self.chunks)) % 50] = time.perf_counter()   # = the corner patch id
                try:
                    self._send(c, lock, head + ch)
                except OSError:
                    return
                i += 1
                time.sleep(1 / FPS)

        def recv_exact(n):
            b = b""
            while len(b) < n:
                part = c.recv(n - len(b))
                if not part:
                    raise ConnectionError
                b += part
            return b
        try:
            while self.running:
                n = struct.unpack("<I", recv_exact(4))[0]
                p = recv_exact(n - 4)
                if p == iw.PKT_SYNC:
                    self._send(c, lock, iw.PKT_SYNC)
                elif p[:3] == b"\x04\x00\x00":
                    code = struct.unpack("<H", p[3:5])[0]
                    self.commands.append(code)
                    status = iw.RESP_ERROR if (self.refuse and code == iw.CMD_START_LIVE) else iw.RESP_OK
                    self._send(c, lock, b"\x04\x00\x00" + struct.pack("<H", status) + b"\x02" + p[6:9] + b"\x80\x00\x00")
                    if code == iw.CMD_START_LIVE and not self.refuse:
                        streaming.set()
                        threading.Thread(target=stream, daemon=True).start()
                    elif code == iw.CMD_STOP_LIVE:
                        streaming.clear()
        except (ConnectionError, OSError):
            pass
        finally:
            streaming.clear()
            c.close()

    def close(self):
        self.running = False
        self.srv.close()


def test_stream_decode_delay_and_tags(sc, chunks, truth):
    for flags_len in (8, 5):                   # the two header layouts in the protocol notes
        cam = FakeInsta360(chunks, flags_len=flags_len)
        src = iw.InstaWifiStream("127.0.0.1", cam.port, log=lambda s: None)
        assert (src.width, src.height) == (sc.W, sc.H), (src.width, src.height)
        assert src.link.video_offset == 7 + flags_len, src.link.video_offset
        delays, fid, t_end = [], 0, time.perf_counter() + 4.0
        mp = m.Mapper(dict(sc.cfg, tags_on_floor=True), sc.K, np.zeros(5),
                      m.make_detector(sc.cfg, nthreads=2, decimate=1.0, realtime=False))
        errs = []
        while time.perf_counter() < t_end:
            f, fid, st = src.latest(fid, 1.0)
            if f is None:
                continue
            k = int(round(float(f[5:35, 5:35].mean()) / 5)) % 50
            if k in cam.sent:
                delays.append((st - cam.sent[k]) * 1e3)
            r = mp.process(f, st, draw=False)
            robot = next((o for o in r["objects"] if o["id"] == 90 and o.get("frame") == "world"), None)
            cands = [i for i in range(len(truth)) if i % 50 == k]
            if robot is not None and cands:
                errs.append(min(1e3 * math.hypot(robot["x"] - truth[i][0], robot["y"] - truth[i][1]) for i in cands))
        fps = src.grab_fps
        src.release()
        time.sleep(0.3)
        assert iw.CMD_STOP_LIVE in cam.commands, "StopLiveStream not sent on release"
        cam.close()
        d = np.array(delays[10:])
        print(f"  flags {flags_len} B: {sc.W}x{sc.H} @ {fps:.0f} fps, relay + decode delay median "
              f"{np.median(d):.0f} ms (95% {np.percentile(d, 95):.0f}), robot tag through the codec: "
              f"{len(errs)} frames, median error {np.median(errs):.1f} mm")
        assert np.median(d) < 120, "delay too high"
        assert len(errs) > 20 and np.median(errs) < 10, errs[:5]


def test_extra_camera_path(sc, chunks):
    cam = FakeInsta360(chunks)
    cfg = dict(sc.cfg, tags_on_floor=True)
    spec = mc.parse_spec(f"name=go3s-wifi,camera=insta-wifi:127.0.0.1:{cam.port},focal_px=1000")
    logs = []
    extra = mc.ExtraCamera(spec, cfg, log=logs.append).open().start()
    try:
        t_end, seen = time.perf_counter() + 6.0, False
        while time.perf_counter() < t_end and not seen:
            time.sleep(0.2)
            r, _ = extra.snapshot()
            seen = r is not None and extra.floor.valid and any(o["id"] == 90 for o in r["objects"])
        print(f"  tracker extra camera over 'Wi-Fi': floor {extra.status()['floor']}, tags {extra.status()['tags']}")
        assert seen, logs
    finally:
        extra.stop()
        cam.close()


def test_errors(chunks):
    cam = FakeInsta360(chunks, refuse=True)
    try:
        iw.InstaWifiStream("127.0.0.1", cam.port, log=lambda s: None)
        raise AssertionError("refused start not reported")
    except RuntimeError as e:
        print(f"  refused StartLiveStream -> {str(e)[:70]}...")
    cam.close()
    cam = FakeInsta360(chunks, drop_after_s=2.0)
    src = iw.InstaWifiStream("127.0.0.1", cam.port, log=lambda s: None)
    time.sleep(3.5)
    assert not src.running, "camera disconnect not noticed"
    print(f"  camera disconnect noticed: running={src.running}, link error {src.link.error}")
    src.release()
    cam.close()
    try:
        iw.InstaWifiStream("127.0.0.1", 1, log=lambda s: None)
        raise AssertionError("closed port not reported")
    except OSError as e:
        print(f"  nothing listening -> {type(e).__name__}")


if __name__ == "__main__":
    sc, chunks, truth = make_stream()
    print(f"[fake camera stream: {len(chunks)} frames {sc.W}x{sc.H}, {sum(map(len, chunks)) / len(chunks) / 1000:.0f} kB/frame]")
    for name, fn in (("stream, decode, delay, tags", lambda: test_stream_decode_delay_and_tags(sc, chunks, truth)),
                     ("tracker extra-camera path", lambda: test_extra_camera_path(sc, chunks)),
                     ("errors", lambda: test_errors(chunks))):
        print(f"[{name}]")
        fn()
        print("  ok")
    print("all Wi-Fi tests passed")
    sys.stdout.flush()
    os._exit(0)
