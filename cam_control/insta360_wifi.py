#!/usr/bin/env python3
"""
insta360_wifi.py

Wireless live video from an Insta360 camera over its own Wi-Fi, into OpenCV, without the phone app.

    python insta360_wifi.py probe              # step-by-step check: Wi-Fi -> port -> handshake -> video
    python insta360_wifi.py view               # live window with frame rate / delay
    (as a tracker / monitor source: camera=insta-wifi  or  camera=insta-wifi:192.168.42.1)

Protocol (reverse engineered by others, NOT documented by Insta360):
  RigacciOrg/insta360-wifi-api (ONE RS, GPLv3) and Daiki-Iijima/insta360-go-ultra-sdk (GO Ultra).
  This file is an independent implementation from their published protocol notes.
    camera = Wi-Fi access point, camera IP 192.168.42.1, TCP port 6666
    every packet: [u32 LE total length incl. these 4 bytes][payload]
    sync      payload 06 00 00 "syNceNdinS"  (sent on connect, the camera answers the same)
    keepalive payload 05 00 00               (both ways, every 2 s)
    command   payload 04 00 00 [u16 LE code] 02 [u24 LE seq] 80 00 00 [protobuf]
              responses come back with the same header layout, code 200 = OK, 500 = error
    StartLiveStream = command 1 (empty protobuf is accepted on the GO Ultra), Stop = 2
    video     payload 01 00 00 [u32 LE pts ms] [8 bytes flags] [H.265 or H.264 Annex B]

  NOT VERIFIED ON THE GO 3S: the protocol is confirmed for the ONE RS and the GO Ultra. The GO 3S
  app keeps its files under ".../Insta360OneR/...", a hint that it shares this app protocol, but
  only `probe` against your camera can tell. Run it first.

Decoding: no extra packages. The video bytes are relayed to a local TCP socket and decoded by
OpenCV's own FFmpeg with ONE decoder thread. (Measured: FFmpeg's default frame threading held
~9 frames = ~300 ms on this 8-thread laptop; with one thread the relay + decode added ~70 ms.)

Connecting:
  1. Turn the camera's Wi-Fi on (Action Pod: swipe down > Settings, Wi-Fi; or the Insta360 app
     switches it on over Bluetooth when it connects).
  2. Join its network from this PC. The name is the camera's ("GO 3S XXXXXX", often with ".OSC").
     The password is shown / changeable in the camera or app settings (many Insta360 cameras
     ship with 88888888; check yours).
  3. Close the Insta360 app on your phone: the camera serves one client at a time.
  This PC then has no other Wi-Fi: robots / coordinator on another network need a second
  adapter (USB Wi-Fi dongle or Ethernet), or run this on a Raspberry Pi bridge.
"""
import argparse
import os
import queue
import re
import socket
import struct
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

DEFAULT_HOST, DEFAULT_PORT = "192.168.42.1", 6666
PKT_SYNC = b"\x06\x00\x00syNceNdinS"
PKT_KEEPALIVE = b"\x05\x00\x00"
CMD_START_LIVE, CMD_STOP_LIVE = 1, 2
RESP_OK, RESP_ERROR = 200, 500


def parse_host(spec):
    """'insta-wifi', 'insta-wifi:192.168.42.1' or 'insta-wifi:127.0.0.1:7000' -> (host, port)."""
    rest = spec.split(":", 1)[1] if ":" in spec else ""
    host, _, port = rest.partition(":")
    return host or DEFAULT_HOST, int(port) if port else DEFAULT_PORT


def is_wifi_spec(spec):
    return str(spec).lower().startswith("insta-wifi")


def sniff_codec(annexb):
    """'h265' / 'h264' / None from the parameter-set NAL units after the start codes."""
    for m_ in re.finditer(b"\x00\x00\x01", annexb[:8192]):
        k = m_.end()
        if k + 1 >= len(annexb):
            break
        b, b2 = annexb[k], annexb[k + 1]
        if b & 0x80:
            continue                                   # forbidden bit set: not a NAL header
        if (b >> 1) & 0x3F in (32, 33, 34) and b2 == 0x01:
            return "h265"                              # VPS / SPS / PPS (layer 0, temporal id 1)
        if b & 0x1F in (7, 8) and b & 0x60:
            return "h264"                              # SPS / PPS (nal_ref_idc != 0)
    return None


def find_payload_offset(p):
    """Where the Annex B data starts in a video packet (type 3 B + pts 4 B + flags; the notes
    disagree on the flags' length, so look for the first start code)."""
    # anchor on the 3-byte start code: a zero flag byte before it would otherwise look like the
    # first byte of a 4-byte start code, and later packets' flags need not be zero
    for k in range(7, min(len(p) - 3, 48)):
        if p[k:k + 3] == b"\x00\x00\x01":
            return k
    return 15


class InstaLink:
    """The TCP control + video connection to the camera (one per camera)."""

    def __init__(self, host=DEFAULT_HOST, port=DEFAULT_PORT, on_video=None, timeout=5.0, log=None, on_close=None):
        self.host, self.port, self.on_video, self.on_close = host, port, on_video, on_close
        self.log = log or (lambda s: None)
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(None)
        self.send_lock = threading.Lock()
        self.seq = 0
        self.synced = threading.Event()
        self.responses = {}                    # seq -> (code, body)
        self.resp_cv = threading.Condition()
        self.running = True
        self.error = None
        self.video_offset = None
        self.stats = {"packets": 0, "video_packets": 0, "video_bytes": 0, "first_video": None,
                      "last_pts": None, "codec": None, "other": {}}
        self._rx = threading.Thread(target=self._recv_loop, daemon=True)
        self._rx.start()
        self._send(PKT_SYNC)
        self._ka = threading.Thread(target=self._keepalive_loop, daemon=True)
        self._ka.start()

    def _send(self, payload):
        with self.send_lock:
            self.sock.sendall(struct.pack("<I", len(payload) + 4) + payload)

    def _recv_exact(self, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("camera closed the connection")
            buf += chunk
        return bytes(buf)

    def _recv_loop(self):
        try:
            while self.running:
                n = struct.unpack("<I", self._recv_exact(4))[0]
                if n < 4 or n > 64 << 20:
                    raise ConnectionError(f"bad packet length {n}: not this protocol")
                p = self._recv_exact(n - 4)
                self.stats["packets"] += 1
                kind = p[:3]
                if p == PKT_SYNC:
                    self.synced.set()
                elif kind == PKT_KEEPALIVE:
                    pass
                elif kind == b"\x01\x00\x00" and len(p) > 7:
                    pts = struct.unpack("<I", p[3:7])[0]
                    if self.video_offset is None:
                        self.video_offset = find_payload_offset(p)
                        self.log(f"Insta360 Wi-Fi: video data starts at byte {self.video_offset} of a packet")
                    data = p[self.video_offset:]
                    st = self.stats
                    st["video_packets"] += 1
                    st["video_bytes"] += len(data)
                    st["last_pts"] = pts
                    if st["first_video"] is None:
                        st["first_video"] = time.perf_counter()
                    if st["codec"] is None:
                        st["codec"] = sniff_codec(data)
                    if self.on_video is not None:
                        self.on_video(data, pts)
                elif kind == b"\x04\x00\x00" and len(p) >= 12:
                    code = struct.unpack("<H", p[3:5])[0]
                    seq = int.from_bytes(p[6:9], "little")
                    with self.resp_cv:
                        self.responses[seq] = (code, p[12:])
                        self.resp_cv.notify_all()
                else:
                    self.stats["other"][kind.hex()] = self.stats["other"].get(kind.hex(), 0) + 1
        except Exception as e:                 # noqa: BLE001
            if self.running:
                self.error = repr(e)
                self.log(f"Insta360 Wi-Fi: {e}")
        finally:
            self.running = False
            with self.resp_cv:
                self.resp_cv.notify_all()
            if self.on_close is not None:
                self.on_close()                # e.g. end the decoder's input, so the stream notices

    def _keepalive_loop(self):
        while self.running:
            time.sleep(2.0)
            try:
                self._send(PKT_KEEPALIVE)
            except OSError:
                break

    def command(self, code, body=b"", timeout=5.0):
        """Send a command, wait for its response. Returns (response code, body) or None on timeout."""
        with self.send_lock:
            self.seq = (self.seq + 1) & 0xFFFFFF
            seq = self.seq
        self._send(b"\x04\x00\x00" + struct.pack("<H", code) + b"\x02" + seq.to_bytes(3, "little")
                   + b"\x80\x00\x00" + body)
        with self.resp_cv:
            self.resp_cv.wait_for(lambda: seq in self.responses or not self.running, timeout=timeout)
            return self.responses.pop(seq, None)

    def close(self):
        if self.running:
            try:
                self.command(CMD_STOP_LIVE, timeout=1.0)
            except OSError:
                pass
        self.running = False
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()


class _Relay:
    """Local TCP server that hands the camera's video bytes to OpenCV's FFmpeg."""

    def __init__(self):
        self.srv = socket.socket()
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(1)
        self.port = self.srv.getsockname()[1]
        self.q = queue.Queue(maxsize=600)
        self.conn = None
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def push(self, data):
        try:
            self.q.put_nowait(data)
        except queue.Full:                     # decoder gone: drop rather than block the camera link
            pass

    def _loop(self):
        try:
            self.conn, _ = self.srv.accept()
            self.conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            while self.running:
                try:
                    data = self.q.get(timeout=0.5)
                except queue.Empty:
                    continue
                self.conn.sendall(data)
        except OSError:
            pass

    def close(self):
        self.running = False
        for s in (self.conn, self.srv):
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass


class InstaWifiStream:
    """Live preview from the camera's Wi-Fi as a frame source with the same interface as
    apriltag_floor_mapper.CameraStream: latest(after_id, timeout), release(), width, height,
    grab_fps, running. Also read() -> (ok, frame) like cv2.VideoCapture."""

    def __init__(self, host=DEFAULT_HOST, port=DEFAULT_PORT, first_frame_timeout=15.0, log=None, start_body=b""):
        self.log = log or print
        self.relay = _Relay()
        self.link = InstaLink(host, port, on_video=lambda data, pts: self.relay.push(data), log=self.log,
                              on_close=self.relay.close)
        if not self.link.synced.wait(5.0):
            self.release()
            raise RuntimeError(f"{host}:{port} accepted the connection but did not answer the sync "
                               "handshake: not an Insta360 camera, or a protocol this camera does not use")
        resp = self.link.command(CMD_START_LIVE, start_body, timeout=5.0)
        if resp is None or resp[0] != RESP_OK:
            self.release()
            raise RuntimeError(f"camera refused StartLiveStream (response {resp and resp[0]}): this model may "
                               "need stream parameters, or the phone app is still connected")
        self.cond = threading.Condition()
        self.frame, self.frame_id, self.stamp = None, 0, 0.0
        self.grab_fps = 0.0
        self.running = True
        self.cap = None
        self._last_read = 0
        self._decoder = threading.Thread(target=self._decode_loop, daemon=True)
        self._decoder.start()
        with self.cond:
            self.cond.wait_for(lambda: self.frame is not None or not self.running, timeout=first_frame_timeout)
        if self.frame is None:
            st = self.link.stats
            self.release()
            raise RuntimeError(f"no decodable video within {first_frame_timeout:.0f} s (video packets "
                               f"{st['video_packets']}, {st['video_bytes']} bytes, codec {st['codec']})")
        self.height, self.width = self.frame.shape[:2]

    def _decode_loop(self):
        try:
            self.cap = cv2.VideoCapture(f"tcp://127.0.0.1:{self.relay.port}", cv2.CAP_FFMPEG,
                                        [cv2.CAP_PROP_N_THREADS, 1])
            n, t_last = 0, time.perf_counter()
            while self.running and self.cap.isOpened():
                ok, frame = self.cap.read()
                if not ok:
                    if not self.link.running:
                        break
                    time.sleep(0.005)
                    continue
                now = time.perf_counter()
                with self.cond:
                    self.frame, self.stamp = frame, now
                    self.frame_id += 1
                    self.cond.notify_all()
                n += 1
                if now - t_last >= 1.0:
                    self.grab_fps, n, t_last = n / (now - t_last), 0, now
        finally:
            self.running = False
            with self.cond:
                self.cond.notify_all()
            if self.cap is not None:
                self.cap.release()             # only here: releasing it from another thread mid-read crashes

    def latest(self, after_id=0, timeout=1.0):
        with self.cond:
            self.cond.wait_for(lambda: self.frame_id > after_id or not self.running, timeout=timeout)
            if self.frame_id <= after_id:
                return None, after_id, 0.0
            return self.frame.copy(), self.frame_id, self.stamp

    def read(self):
        frame, self._last_read, _ = self.latest(self._last_read, timeout=2.0)
        return frame is not None, frame

    def isOpened(self):
        return self.running

    def release(self):
        self.running = False
        try:
            self.link.close()
        except Exception:                      # noqa: BLE001
            pass
        self.relay.close()                     # the decoder sees end of input and releases its capture
        dec = getattr(self, "_decoder", None)
        if dec is not None and dec is not threading.current_thread():
            dec.join(timeout=3.0)

    @property
    def codec(self):
        return self.link.stats["codec"]


# --------------------------------------------------------------------------
# Generic network sources (Raspberry Pi bridge etc.)
# --------------------------------------------------------------------------

URL_PREFIXES = ("rtsp://", "rtsps://", "http://", "https://", "tcp://", "udp://", "rtmp://", "srt://")


def is_url(spec):
    return str(spec).lower().startswith(URL_PREFIXES)


def open_url(url):
    """cv2.VideoCapture for a network stream, low latency (one decoder thread, small probe)."""
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "fflags;nobuffer|flags;low_delay")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    if not cap.isOpened():
        raise RuntimeError(f"could not open {url}")
    return cap


class UrlStream:
    """A network stream (RTSP / HTTP MJPEG / ...) with the CameraStream interface."""

    def __init__(self, url):
        self.cap = open_url(url)
        ok, frame = self.cap.read()
        if not ok:
            self.cap.release()
            raise RuntimeError(f"{url} opened but sent no frame")
        self.height, self.width = frame.shape[:2]
        self.cond = threading.Condition()
        self.frame, self.frame_id, self.stamp, self.grab_fps = frame, 1, time.perf_counter(), 0.0
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        n, t_last = 0, time.perf_counter()
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            now = time.perf_counter()
            with self.cond:
                self.frame, self.stamp = frame, now
                self.frame_id += 1
                self.cond.notify_all()
            n += 1
            if now - t_last >= 1.0:
                self.grab_fps, n, t_last = n / (now - t_last), 0, now

    def latest(self, after_id=0, timeout=1.0):
        with self.cond:
            self.cond.wait_for(lambda: self.frame_id > after_id or not self.running, timeout=timeout)
            if self.frame_id <= after_id:
                return None, after_id, 0.0
            return self.frame.copy(), self.frame_id, self.stamp

    def release(self):
        self.running = False
        self.cap.release()


def open_network_source(spec, log=print):
    """'insta-wifi[:host[:port]]' or a URL -> a CameraStream-like source, else None."""
    if is_wifi_spec(spec):
        host, port = parse_host(spec)
        return InstaWifiStream(host, port, log=log)
    if is_url(spec):
        return UrlStream(spec)
    return None


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def current_wifi():
    """(ssid, signal) of this PC's Wi-Fi connection on Windows, else (None, None)."""
    if not sys.platform.startswith("win"):
        return None, None
    try:
        out = subprocess.run(["netsh", "wlan", "show", "interfaces"], capture_output=True, text=True,
                             timeout=10).stdout
    except Exception:                          # noqa: BLE001
        return None, None
    ssid = re.search(r"^\s*SSID\s*:\s*(.+)$", out, re.M)
    sig = re.search(r"^\s*Signal\s*:\s*(.+)$", out, re.M)
    return (ssid.group(1).strip() if ssid else None), (sig.group(1).strip() if sig else None)


def cmd_probe(args):
    ok = lambda s: print(f"  [ok]   {s}")
    bad = lambda s: print(f"  [FAIL] {s}")
    info = lambda s: print(f"         {s}")
    print("1. Wi-Fi")
    ssid, sig = current_wifi()
    if ssid:
        looks = any(k in ssid.lower() for k in ("go 3", "go3", "insta", ".osc"))
        (ok if looks else bad)(f"this PC is on '{ssid}' ({sig})" + ("" if looks else
                               ": that does not look like the camera's network"))
        if not looks:
            info("Turn on the camera's Wi-Fi (Action Pod settings, or let the app switch it on), then join its "
                 "network from Windows (name like 'GO 3S XXXXXX.OSC').")
    else:
        info("could not read the Wi-Fi connection (not Windows, or no Wi-Fi adapter)")
    print(f"2. TCP {args.host}:{args.port}")
    try:
        s = socket.create_connection((args.host, args.port), timeout=4.0)
        s.close()
        ok("port open")
    except OSError as e:
        bad(f"cannot connect: {e}")
        for p in (80, 8080):
            try:
                socket.create_connection((args.host, p), timeout=2.0).close()
                info(f"(port {p} is open: the camera is reachable, but not the 6666 API)")
            except OSError:
                pass
        info("Is this PC on the camera's network? Is the phone app closed? Camera awake?")
        return
    print("3. handshake")
    try:
        link = InstaLink(args.host, args.port, on_video=None)
    except OSError as e:
        bad(f"connect failed: {e}")
        return
    if link.synced.wait(5.0):
        ok("camera answered the sync handshake: it speaks this protocol")
    else:
        bad(f"no sync answer in 5 s (packets seen: {link.stats['packets']}, error {link.error})")
        link.close()
        return
    print("4. StartLiveStream")
    resp = link.command(CMD_START_LIVE, timeout=5.0)
    if resp is None:
        bad("no response")
    elif resp[0] == RESP_OK:
        ok("accepted (200)")
    else:
        bad(f"response code {resp[0]} (body {resp[1][:32].hex()})")
    print(f"5. video for {args.seconds:.0f} s")
    t0 = time.perf_counter()
    time.sleep(args.seconds)
    st = link.stats
    link.close()
    if st["video_packets"]:
        dt = time.perf_counter() - (st["first_video"] or t0)
        ok(f"{st['video_packets']} video packets, {st['video_bytes'] / max(dt, 1e-3) / 1000:.0f} kB/s, "
           f"codec {st['codec'] or 'unknown'}")
    else:
        bad(f"no video packets (other packet types seen: {st['other'] or 'none'})")
        return
    print("6. decode")
    try:
        src = InstaWifiStream(args.host, args.port, log=lambda s: None)
    except Exception as e:                     # noqa: BLE001
        bad(str(e))
        return
    fid, n, t1, stamps = 0, 0, time.perf_counter(), []
    while time.perf_counter() - t1 < 5.0:
        f, fid, st_ = src.latest(fid, 1.0)
        if f is not None:
            n += 1
            stamps.append(st_)
    w, h = src.width, src.height
    src.release()
    fps = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 2 else 0
    ok(f"{w}x{h} at {fps:.1f} fps decoded")
    verdict = "usable for tracking" if w >= 1280 and fps >= 15 else "too small / slow for tracking"
    print(f"\nVERDICT: wireless preview works, {w}x{h} @ {fps:.0f} fps -> {verdict}.")
    print("Next: python insta360_wifi.py view   (or add camera=insta-wifi as an extra camera)")


def cmd_view(args):
    src = InstaWifiStream(args.host, args.port)
    print(f"streaming {src.width}x{src.height}, codec {src.codec}; Esc / q to quit")
    fid = 0
    try:
        while True:
            f, fid, st = src.latest(fid, 2.0)
            if f is None:
                if not src.running:
                    print("stream ended:", src.link.error)
                    break
                continue
            txt = f"{src.width}x{src.height}  {src.grab_fps:4.1f} fps  frame age {1e3 * (time.perf_counter() - st):3.0f} ms"
            cv2.putText(f, txt, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
            s = min(1.0, 1280 / f.shape[1])
            cv2.imshow("Insta360 Wi-Fi", cv2.resize(f, None, fx=s, fy=s) if s < 1 else f)
            if cv2.waitKey(1) in (27, ord("q")):
                break
    finally:
        src.release()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("probe", "view"):
        p = sub.add_parser(name)
        p.add_argument("--host", default=DEFAULT_HOST)
        p.add_argument("--port", type=int, default=DEFAULT_PORT)
        if name == "probe":
            p.add_argument("--seconds", type=float, default=5.0)
    args = ap.parse_args()
    {"probe": cmd_probe, "view": cmd_view}[args.cmd](args)


if __name__ == "__main__":
    main()
