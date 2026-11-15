#!/bin/bash
# The whole show: both cars put every object on the floor into the white box.
#   bash scripts/show.sh            boundary guard OFF (use until the overhead camera's arena is set up again)
#   bash scripts/show.sh fence      boundary guard ON (only after the arena was set up again and checked)
# Stop at any time: Ctrl-C in this window (both cars stop).   Log + photos: runs/show_<time>/
cd "$(dirname "$0")/.." || exit 1
R="runs/show_$(date +%H%M)"; mkdir -p "$R"
if ! curl -s -m 2 http://127.0.0.1:8090/health | grep -q ok; then
  echo "The VLM server is not running. Open another window and run:  bash scripts/vlm.sh   - then start this again."; exit 1
fi
FENCE="--fence-m 0"; [ "$1" = "fence" ] && FENCE="--fence-m 0.30"
echo "log: $R/show.log"
python3 -u -m antcolony.clear $FENCE --max-min 25 2>&1 | tee "$R/show.log"
# collect what the cars saw (for later)
for H in "zxz5500@192.168.13.106|/home/zxz5500/Desktop/Master Pi5/masterpi_lite|A" "mitra@192.168.13.49|/home/mitra/Desktop/Master Pi5/masterpi_lite|B"; do
  IFS='|' read -r U D N <<< "$H"; mkdir -p "$R/$N"
  ssh -o ConnectTimeout=6 -o BatchMode=yes "$U" "cd \"$D\" && tar cf - agent.log grab.log *.jpg 2>/dev/null" | tar xf - -C "$R/$N" 2>/dev/null
done
echo "saved: $R"
