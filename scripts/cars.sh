#!/bin/bash
# Check / start / stop the programs on cars A and B.  Run from anywhere:
#   bash scripts/cars.sh status     battery, holding, version of each car (nothing moves)
#   bash scripts/cars.sh start      (re)start the program on both cars - after a car was switched on
#   bash scripts/cars.sh stop       STOP both cars at once (wheels and arm stop where they are)
#   bash scripts/cars.sh home       both arms to the driving pose, put down whatever they hold
cd "$(dirname "$0")/.." || exit 1
ip_of() { python3 -c "import sys; sys.path.insert(0, '.'); from antcolony.common import load_config; import io, contextlib
with contextlib.redirect_stdout(io.StringIO()): c = load_config('config.json')
print(c['robots']['$1']['addr'][0])"; }
host() { [ "$1" = A ] && echo "zxz5500@$(ip_of A)" || echo "mitra@$(ip_of B)"; }
dir()  { [ "$1" = A ] && echo "/home/zxz5500/Desktop/Master Pi5/masterpi_lite" || echo "/home/mitra/Desktop/Master Pi5/masterpi_lite"; }
SSH="ssh -o ConnectTimeout=6 -o BatchMode=yes"
case "$1" in
  start)
    for N in A B; do
      H=$(host $N); D=$(dir $N); ip=${H#*@}; [ -z "$ip" ] && { echo "car $N: no address in config.json - run:  python3 scripts/net.py"; continue; }
      if ! ping -c1 -t2 "$ip" >/dev/null 2>&1; then echo "car $N ($ip): NOT ON THE NETWORK - switch it on, wait 1-2 min, try again; after a change of Wi-Fi run:  python3 scripts/net.py"; continue; fi
      $SSH "$H" "cd \"$D\" && p=\$(pgrep -f '^python3 -u agent.py'); [ -n \"\$p\" ] && kill \$p; sleep 1.5; (setsid nohup python3 -u agent.py --name $N --min-duty 30 > agent.log 2>&1 < /dev/null &); sleep 5; pgrep -f '^python3 -u agent.py' >/dev/null && echo 'car $N: program running' || echo 'car $N: PROGRAM DID NOT START - see agent.log on the car'"
    done
    "$0" status ;;
  status)
    for N in A B; do
      H=$(host $N); ip=${H#*@}
      if ! ping -c1 -t2 "$ip" >/dev/null 2>&1; then echo "[$N] NOT ON THE NETWORK ($ip) - switched off? after a change of Wi-Fi run:  python3 scripts/net.py"; continue; fi
      python3 -u -m antcolony.solo --robot $N --steps status >/dev/null 2>&1
      python3 -u -m antcolony.solo --robot $N --steps status 2>&1 | grep -E "battery|no telemetry" | sed "s/no telemetry.*/no answer - run:  bash scripts\/cars.sh start/"
    done ;;
  stop)
    python3 - <<'PY'
import socket, json, time, io, contextlib, sys
sys.path.insert(0, ".")
from antcolony.common import load_config
with contextlib.redirect_stdout(io.StringIO()):
    cfg = load_config("config.json")
tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
for k in range(5):
    for ip in [cfg["robots"][n]["addr"][0] for n in ("A", "B")]:
        tx.sendto(json.dumps({"type": "skill", "id": int(time.time() * 10) % 2000000000 + 50 + k, "name": "stop", "args": {}}).encode(), (ip, 9200))
        tx.sendto(json.dumps({"type": "vel", "vx": 0, "vy": 0, "w": 0, "dur": 0.1}).encode(), (ip, 9200))
    time.sleep(0.15)
print("stop sent to A and B")
PY
    ;;
  home)
    for N in A B; do python3 -u -m antcolony.solo --robot $N --steps drop,home --drop-y 16 --drop-z 4 2>&1 | grep -E "ok|FAILED|no telem" | tr '\n' ' '; echo; done ;;
  *) echo "usage: bash scripts/cars.sh status | start | stop | home" ;;
esac
