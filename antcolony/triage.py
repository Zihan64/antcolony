"""Triage: can the arm cars grasp an object, or is it a job for the bulldozer?

Rule first (instant, from the overhead size): the gripper opens ~7.5 cm and the arm has no
wrist roll, so anything whose short side is too wide, or that is too long, is 'push'.
Optionally a VLM looks at a crop of each new object once (cached per object) and can
overrule a 'grasp' (e.g. "a rock: too heavy", "a sheet of paper: too flat"). The VLM runs in
a background thread and never blocks the control loop. OpenAI-compatible endpoint
(llama-server, or any hosted API with the same shape).
"""
import base64
import json
import os
import queue
import re
import threading
import urllib.request

PROMPT = ("A small robot gripper (opens 6 cm, lifts about 100 g, grips things 1-6 cm wide that "
          "are at least 1 cm tall) must decide what to do with the object in this photo. "
          'Answer ONLY JSON: {"label": "<2-3 words>", "action": "grasp" or "push", "why": "<short>"}. '
          'Use "push" for anything too big, too heavy, too flat, too soft or too round to pinch.')


class Triage:
    def __init__(self, cfg):
        self.c = cfg["triage"]
        self.q, self.seen = queue.Queue(), set()
        if self.c.get("vlm_url"):
            threading.Thread(target=self._worker, daemon=True).start()

    def rule(self, o):
        return "grasp" if o.w <= self.c["grasp_max_short"] and o.l <= self.c["grasp_max_long"] else "push"

    def classify(self, o):
        """Called once when an object is confirmed. Returns the rule's answer immediately;
        the VLM's answer (if any) is written to o.cls / o.label later."""
        cls = self.rule(o)
        if self.c.get("vlm_url") and o.crop and o.id not in self.seen and os.path.isfile(o.crop):
            self.seen.add(o.id)
            self.q.put(o)
        return cls

    def ask(self, path):
        with open(path, "rb") as f:
            img = base64.b64encode(f.read()).decode()
        body = {"temperature": 0, "max_tokens": 80, "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + img}},
            {"type": "text", "text": PROMPT}]}]}
        if self.c.get("vlm_model"):
            body["model"] = self.c["vlm_model"]
        hdr = {"Content-Type": "application/json"}
        if self.c.get("vlm_key"):
            hdr["Authorization"] = "Bearer " + self.c["vlm_key"]
        req = urllib.request.Request(self.c["vlm_url"].rstrip("/") + "/v1/chat/completions",
                                     data=json.dumps(body).encode(), headers=hdr)
        with urllib.request.urlopen(req, timeout=20) as r:
            text = json.loads(r.read())["choices"][0]["message"]["content"]
        m = re.search(r"\{.*\}", text, re.S)
        return json.loads(m.group(0)) if m else {}

    def _worker(self):
        while True:
            o = self.q.get()
            try:
                ans = self.ask(o.crop)
            except Exception as e:                       # server down, bad JSON...: keep the rule's answer
                print(f"[triage] VLM failed for #{o.id}: {e}")
                continue
            if ans.get("label"):
                o.label = str(ans["label"])[:24]
            if ans.get("action") == "push" and o.state in ("free", "assigned") and o.cls == "grasp":
                o.cls = "push"                           # only ever downgrade: the size rule stays a hard limit
            print(f"[triage] #{o.id}: {ans}")
