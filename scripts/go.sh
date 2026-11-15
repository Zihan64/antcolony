#!/bin/bash
# ONE-CLICK START of the whole show (no internet needed). Double-click "开始演示.command" in the
# project folder, or run:   bash scripts/go.sh          (bash scripts/go.sh fence = boundary guard on)
#
#   1. VLM server on this Mac       started in the background if it is not running (log: runs/vlm.log)
#   2. cars A and B                 found on this Wi-Fi, told this Mac's address, their programs restarted
#   3. overhead camera              prints the command for the camera computer, waits up to 2 min for its data
#   4. the show                     both cars put every object into the white box (scripts/show.sh)
# Stop at any time: Ctrl-C (both cars stop).
cd "$(dirname "$0")/.." || exit 1
mkdir -p runs
say() { printf "\n\033[1m== %s\033[0m\n" "$*"; }
fail() { printf "\n\033[31m%s\033[0m\n" "$*"; read -r -p "Press Enter to close." _; exit 1; }

say "1/4  VLM server"
if curl -s -m 2 http://127.0.0.1:8090/health | grep -q ok; then
  echo "already running"
else
  SNAP="$HOME/.cache/huggingface/hub/models--ggml-org--Qwen2.5-VL-3B-Instruct-GGUF/snapshots/5037fcf163dd95d1e41d1974465f0898ed108ca2"
  HF_HUB_OFFLINE=1 nohup llama-server -m "$SNAP/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf" --mmproj "$SNAP/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf" \
        --host 0.0.0.0 --port 8090 -ngl 99 -c 8192 --image-max-tokens 1024 > runs/vlm.log 2>&1 &
  for i in $(seq 1 30); do curl -s -m 1 http://127.0.0.1:8090/health | grep -q ok && break; sleep 1; done
  curl -s -m 2 http://127.0.0.1:8090/health | grep -q ok && echo "started" || fail "The VLM server did not start - see runs/vlm.log"
fi

say "2/4  cars A and B on this Wi-Fi"
python3 scripts/net.py || fail "The cars are not ready - read the messages just above (cars not found, or a car to confirm once with ssh). Then double-click again."
MYIP=$(ipconfig getifaddr en0 || ipconfig getifaddr en1)
bash scripts/cars.sh status

say "3/4  overhead camera"
echo "On the OVERHEAD-CAMERA computer run (if it is not running with this address yet):"
printf "\n    \033[1mpython cam_control/run_tracker.py --antcolony %s\033[0m\n\n" "$MYIP"
echo "then press Start in its window. Waiting for its data (up to 2 minutes; Ctrl-C to stop) ..."
python3 - <<'PY'
import socket, json, time, sys
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.bind(("0.0.0.0", 9100))
except OSError as e:
    print(f"cannot listen on UDP 9100 ({e}) - another program (check, tagwatch, show) is still open: close it"); sys.exit(0)
s.settimeout(1.0); t0, n, last = time.time(), 0, None
while time.time() - t0 < 120:
    try:
        last = json.loads(s.recvfrom(65535)[0]); n += 1
        if n >= 20:
            break
    except (socket.timeout, ValueError):
        if int(time.time() - t0) % 15 == 0:
            print(f"  still waiting ({int(time.time() - t0)} s) ...", flush=True)
if not last:
    print("no data from the overhead camera - the show runs WITHOUT it (cars look for the box by themselves)")
else:
    tags = sorted(last.get("tags", {}), key=int)
    print(f"overhead camera OK: tags in view {tags}")
    for want, what in (("33", "car A"), ("36", "car B"), ("54", "box (top)")):
        if want not in tags:
            print(f"  note: tag {want} ({what}) is not in view right now")
PY

say "4/4  the show"
echo "Field ready? Objects on the floor, white box with tag 54 facing up, cars side by side facing into the arena."
read -r -p "Press Enter to START (Ctrl-C to cancel) " _
bash scripts/show.sh "$@"
read -r -p "Finished. Press Enter to close." _
