"""The wheel loader in the team run: where the arm cars are, seen from car C (antcolony.clear.peers_of),
and how car C uses it (dozer_auto: never a target, no driving into them). No hardware.

    python tests/test_dozer_team.py
"""
import json, math, os, sys, tempfile, time, types
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "robot"))
from antcolony.clear import peers_of

tags = {"A": 33, "B": 36, "C": 34}
# C at the origin facing +x; A 0.5 m straight ahead; B 1 m to C's left
snap = {"tags": {"34": [0.0, 0.0, 0.0], "33": [0.5, 0.0, 1.0], "36": [0.0, 1.0, 0.0]}}
p = {q["name"]: q for q in peers_of(snap, "C", ("A", "B"), tags, {})}
assert abs(p["A"]["dist"] - 0.5) < 1e-6 and abs(p["A"]["bearing_deg"]) < 0.1, p
assert abs(p["B"]["dist"] - 1.0) < 1e-6 and abs(p["B"]["bearing_deg"] - 90) < 0.1, p
# C turned to face +y: A is now 90 deg to its right
snap["tags"]["34"] = [0.0, 0.0, math.pi / 2]
p = {q["name"]: q for q in peers_of(snap, "C", ("A", "B"), tags, {})}
assert abs(p["A"]["bearing_deg"] + 90) < 0.1 and abs(p["B"]["bearing_deg"]) < 0.1, p
# calibration: the tag yaw is 90 deg off the car's heading
p = {q["name"]: q for q in peers_of(snap, "C", ("A", "B"), tags, {"C": {"th_off": -math.pi / 2}})}
assert abs(p["A"]["bearing_deg"]) < 0.1, p
assert peers_of({"tags": {"33": [0, 0, 0]}}, "C", ("A", "B"), tags, {}) is None
print("peers_of: ok")

# car C's side: dozer_auto.Dozer.near_peer / peer_in_the_way on a peers file
sys.modules["hw_freenove"] = types.SimpleNamespace(Freenove=None)
import dozer_auto as d
f = os.path.join(tempfile.mkdtemp(), "peers.json")
dz = d.Dozer.__new__(d.Dozer)
dz.peers_file, dz._peers, dz._peers_t = f, [], 0.0
def put(peers, age=0.0):
    json.dump({"t": time.time() - age, "peers": peers}, open(f, "w"))
    dz._peers_t = 0.0
put([{"name": "A", "dist": 0.45, "bearing_deg": 10}, {"name": "B", "dist": 2.0, "bearing_deg": -60}])
assert dz.near_peer(15) == "A" and dz.near_peer(-60) == "B" and dz.near_peer(-30) is None
assert dz.peer_in_the_way()[0] == "A"                     # 45 cm, 10 deg off the nose
put([{"name": "A", "dist": 0.45, "bearing_deg": 120}])
assert dz.peer_in_the_way() is None                       # beside / behind, not too near
put([{"name": "A", "dist": 0.25, "bearing_deg": 150}])
assert dz.peer_in_the_way()[0] == "A"                     # 25 cm: too near in any direction
put([{"name": "A", "dist": 0.45, "bearing_deg": 10}], age=5)
assert dz.peers() == [] and dz.peer_in_the_way() is None  # stale file = no information
print("dozer peers: ok")
