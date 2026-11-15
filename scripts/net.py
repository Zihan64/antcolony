#!/usr/bin/env python3
"""After a change of network (e.g. to the team's AntColony router): find the cars, point everything
at the new addresses. Run ONCE after this Mac and the cars are on the same Wi-Fi:

    python3 scripts/net.py            find the cars, update config.json and each car's VLM address
    python3 scripts/net.py --dry-run  only show what it finds and would change

It
  1. reads this Mac's IP and network,
  2. pings every address of that network and finds cars A, B, C by their network card's hardware
     (MAC) address - so it works whatever addresses the router gave them,
  3. writes those addresses into config.json as network profile "here" and makes it the active one
     (scripts/cars.sh, scripts/show.sh and all antcolony programs read them from there),
  4. sets the VLM address on cars A and B to this Mac (masterpi_lite/masterpi_lite/vlm_server.json,
     the old file is kept as vlm_server.json.before_net) and restarts their programs,
  5. prints the command for the overhead-camera computer.
No internet needed.
"""
import argparse
import ipaddress
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARS = {  # name: (MAC of the Pi's Wi-Fi, ssh user, program folder on the car or None)
    "A": ("2c:cf:67:95:d1:25", "zxz5500", "/home/zxz5500/Desktop/Master Pi5/masterpi_lite"),
    "B": ("2c:cf:67:87:c0:61", "mitra", "/home/mitra/Desktop/Master Pi5/masterpi_lite"),
    "C": ("2c:cf:67:95:aa:f6", "szk13579", None),
}


def sh(cmd, timeout=20):
    try:
        return subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, timeout=timeout).stdout
    except (subprocess.TimeoutExpired, OSError):          # OSError: too many open files etc. - treat as no answer
        return ""


def my_network():
    for ifc in ("en0", "en1"):
        ip = sh(["ipconfig", "getifaddr", ifc]).strip()
        if ip:
            mask = re.search(r"subnet_mask \(ip\): (\S+)", sh(["ipconfig", "getpacket", ifc]))
            net = ipaddress.ip_network(f"{ip}/{mask[1] if mask else '255.255.255.0'}", strict=False)
            return ifc, ip, net
    sys.exit("This Mac has no network address - is Wi-Fi on and connected?")


CAR_FILES = ["agent.py", "hw_masterpi.py", "seek.py", "grab_runner.py", "detector.py", "detector.warehouse.json"]


def ssh(host, cmd, timeout=40):
    """Run cmd on host over ssh (no questions asked: BatchMode). -> (ok, stdout + stderr)."""
    try:
        r = subprocess.run(["ssh", "-o", "ConnectTimeout=8", "-o", "BatchMode=yes", host, cmd],
                           capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0, r.stdout + r.stderr
    except (subprocess.TimeoutExpired, OSError) as e:
        return False, str(e)


HOSTKEY_HELP = """
   This Mac has not connected to car {n} at {ip} before, so ssh asks you to confirm the car once.
   Run this in a Terminal and type  yes  when it asks "Are you sure you want to continue connecting":

       ssh {user}@{ip} exit

   (If it says "REMOTE HOST IDENTIFICATION HAS CHANGED", another device had this address before: check
    that car {n} really is at {ip}, then run   ssh-keygen -R {ip}   and the ssh command above again.)
   Then double-click the start file again."""


def norm_mac(m):
    return ":".join(p.zfill(2) for p in m.lower().split(":"))


def find_cars(net, known=None):
    found = {}
    for n, ip_ in (known or {}).items():                 # the addresses we already know, if they are on this network
        if n in CARS and ip_ and ipaddress.ip_address(ip_) in net and "bytes from" in sh(["ping", "-c", "2", "-t", "2", ip_], 6):
            found[n] = ip_
    if all(n in found for n in ("A", "B")):
        print("found at the known addresses - no scan needed")
        return found
    hosts = list(net.hosts())
    if len(hosts) > 2048:
        sys.exit(f"network {net} is too big to scan")
    print(f"scanning {len(hosts)} addresses of {net} ...", flush=True)
    with ThreadPoolExecutor(32) as ex:
        list(ex.map(lambda h: sh(["ping", "-c", "1", "-t", "1", str(h)], 3), hosts))
    for line in sh(["arp", "-an"]).splitlines():
        m = re.search(r"\((\d+\.\d+\.\d+\.\d+)\) at ([0-9a-fA-F:]+)", line)
        if not m:
            continue
        for name, (mac, _, _) in CARS.items():
            if norm_mac(m[2]) == mac and name not in found:
                found[name] = m[1]
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    ifc, ip, net = my_network()
    print(f"this Mac: {ip} on {net} ({ifc})")
    with open(os.path.join(ROOT, "config.json"), encoding="utf-8") as f:
        prof0 = (json.load(f).get("networks") or {})
    known = {}
    for pname in ("hyatt", "lan", "here"):              # later profiles win
        known.update({n: v for n, v in (prof0.get(pname) or {}).items() if v})
    found = find_cars(net, known)
    for n in CARS:
        print(f"  car {n}: " + (found[n] if n in found else "NOT FOUND (switched off, or on another Wi-Fi)"))
    if not any(n in found for n in ("A", "B")):
        sys.exit("\nNeither car A nor car B is on this network. Make sure this Mac and the cars are on the same Wi-Fi, "
                 "wait 1-2 minutes after switching the cars on, and run this again.")

    cfg_path = os.path.join(ROOT, "config.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    prof = {n: found.get(n) for n in CARS}
    print(f"\nconfig.json: network profile 'here' = {prof}, made active (was '{cfg.get('network')}')")
    if not a.dry_run:
        cfg.setdefault("networks", {})["here"] = prof
        cfg["network"] = "here"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
            f.write("\n")

    url = f"http://{ip}:8090"
    not_set = []
    for n in ("A", "B"):
        if n not in found:
            continue
        _, user, d = CARS[n]
        f = f"{d}/masterpi_lite/vlm_server.json"
        print(f"car {n}: VLM address -> {url}" + ("  (dry run)" if a.dry_run else ""))
        if a.dry_run:
            print("   now: " + ssh(f"{user}@{found[n]}", f'cat "{f}"')[1].strip())
            continue
        script = (f'cd "{d}" && f=masterpi_lite/vlm_server.json && cp -p "$f" "$f.before_net" 2>/dev/null; '
                  f'python3 -c "import json,sys; p=sys.argv[1]; c=json.load(open(p)) if __import__(\'os\').path.exists(p) else {{}}; '
                  f'c[\'url\']=sys.argv[2]; c.setdefault(\'img_tokens\', 256); json.dump(c, open(p, \'w\'))" "$f" "{url}" && cat "$f"; echo; '
                  f'p=$(pgrep -f "^python3 -u agent.py"); [ -n "$p" ] && kill $p; sleep 1.5; '
                  f'(setsid nohup python3 -u agent.py --name {n} --min-duty 30 > agent.log 2>&1 < /dev/null &); sleep 5; '
                  f'pgrep -f "^python3 -u agent.py" >/dev/null && echo "   program running" || echo "   PROGRAM DID NOT START"')
        # bring the car's program up to date with this computer's copy (the fixes reach the cars without anyone
        # copying files by hand); grab_defaults.json, calibration and models on the car are not touched
        files = [os.path.join(ROOT, "robot", f) for f in CAR_FILES]
        try:
            r = subprocess.run(["scp", "-q", "-o", "ConnectTimeout=8", "-o", "BatchMode=yes", *files, f"{user}@{found[n]}:"],
                               capture_output=True, text=True, timeout=60)     # to the home folder first: the car's folder
            copied = r.returncode == 0                                       # name has a space ("Master Pi5")
            if copied:
                okm, outm = ssh(f"{user}@{found[n]}", "mv " + " ".join(f"~/{f}" for f in CAR_FILES) + f' "{d}/"')
                copied = okm
                if not okm:
                    print("   (copied, but could not move the files into place: " + outm.strip()[:200] + ")")
            if not copied:
                print("   (could not copy the newest car program: " + (r.stderr.strip() or "?")[:200] + ")")
        except (subprocess.TimeoutExpired, OSError) as e:
            copied = False
            print(f"   (could not copy the newest car program: {e})")
        if copied:
            print(f"   newest car program copied ({', '.join(CAR_FILES)})")
        ok, out = ssh(f"{user}@{found[n]}", script)
        out = out.strip()
        if "Host key verification failed" in out or "REMOTE HOST IDENTIFICATION HAS CHANGED" in out:
            print(HOSTKEY_HELP.format(n=n, ip=found[n], user=user))
            not_set.append(n)
        elif "program running" not in out:
            print("   " + (out or "no answer over ssh").replace("\n", "\n   "))
            print(f"   !! car {n} was NOT set up")
            not_set.append(n)
        else:
            print("   " + out.replace("\n", "\n   "))

    print("\nVLM server on this Mac: " + ("running" if '"ok"' in sh(["curl", "-s", "-m", "2", "http://127.0.0.1:8090/health"])
                                         else "NOT running - start it:  bash scripts/vlm.sh"))
    print("\nOn the OVERHEAD-CAMERA computer (same Wi-Fi), start the tracker with THIS address:")
    print(f"    python cam_control/run_tracker.py --antcolony {ip}")
    if not_set and not a.dry_run:
        sys.exit(f"\ncar(s) {', '.join(not_set)} NOT ready - fix that first (see above).")


if __name__ == "__main__":
    main()
