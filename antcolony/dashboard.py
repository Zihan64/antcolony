"""Live dashboard: http://<host>:8080  (canvas map + status, polls /state 5x per second)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .common import PORT_DASH

PAGE = r"""<!doctype html><meta charset="utf-8"><title>AntColony</title>
<style>
body{margin:0;background:#11151a;color:#e6e9ee;font:14px/1.45 system-ui,sans-serif;display:flex;flex-wrap:wrap}
#map{flex:1 1 520px;min-width:320px;max-width:100vh;aspect-ratio:1}
canvas{width:100%;height:100%;display:block}
#side{flex:1 1 300px;padding:18px;box-sizing:border-box}
h1{font-size:18px;margin:0 0 4px} .big{font-size:44px;font-variant-numeric:tabular-nums;font-weight:600}
.done{color:#6fd08c} table{border-collapse:collapse;width:100%;margin:10px 0} td{padding:3px 8px 3px 0;vertical-align:top}
.k{color:#8b95a3} pre{white-space:pre-wrap;color:#aeb7c4;font:12px/1.5 ui-monospace,monospace;margin:0}
.chip{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
</style>
<div id="map"><canvas id="c" width="900" height="900"></canvas></div>
<div id="side"><h1>AntColony &mdash; cooperative floor clearing</h1>
<div class="big" id="clock">0.0 s</div><div id="count" class="k"></div>
<table id="robots"></table>
<div class="k"><span class="chip" style="background:#f2b84b"></span>arm can grasp
&nbsp;<span class="chip" style="background:#e0655f"></span>too big for the gripper
&nbsp;<span class="chip" style="background:#6fd08c"></span>being carried</div>
<h1 style="margin-top:14px">Log</h1><pre id="log"></pre></div>
<script>
const cv=document.getElementById('c'),g=cv.getContext('2d'),COL={A:'#5aa9ff',B:'#b58cff',C:'#ff9f5a'};
function draw(s){const a=s.arena,W=cv.width,pad=40,sc=(W-2*pad)/Math.max(a.xmax-a.xmin,a.ymax-a.ymin);
 const X=x=>pad+(x-a.xmin)*sc,Y=y=>W-pad-(y-a.ymin)*sc;
 g.clearRect(0,0,W,W);g.strokeStyle='#2a323c';g.lineWidth=1;
 for(let x=Math.ceil(a.xmin*4)/4;x<=a.xmax;x+=.25){g.beginPath();g.moveTo(X(x),Y(a.ymin));g.lineTo(X(x),Y(a.ymax));g.stroke()}
 for(let y=Math.ceil(a.ymin*4)/4;y<=a.ymax;y+=.25){g.beginPath();g.moveTo(X(a.xmin),Y(y));g.lineTo(X(a.xmax),Y(y));g.stroke()}
 g.strokeStyle='#56616f';g.lineWidth=2;g.strokeRect(X(a.xmin),Y(a.ymax),(a.xmax-a.xmin)*sc,(a.ymax-a.ymin)*sc);
 const b=s.bin;g.textAlign='center';
 if(b.on!==false){g.strokeStyle='#6fd08c';g.lineWidth=5;g.beginPath();
  if(b.mouth)g.arc(X(b.x),Y(b.y),b.r*sc,-(b.th+1.0),-(b.th-1.0)+2*Math.PI);else g.arc(X(b.x),Y(b.y),b.r*sc,0,7);
  g.stroke();g.fillStyle='#6fd08c';g.font='600 22px system-ui';g.fillText(s.delivered,X(b.x),Y(b.y)+8)}
 if(s.edge){const e=s.edge;g.fillStyle='rgba(111,208,140,.10)';g.fillRect(X(a.xmin),Y(a.ymax),(a.xmax-a.xmin)*sc,(a.ymax-a.ymin)*sc);
  g.clearRect(X(a.xmin+e),Y(a.ymax-e),(a.xmax-a.xmin-2*e)*sc,(a.ymax-a.ymin-2*e)*sc);
  g.strokeStyle='#6fd08c';g.lineWidth=1;g.setLineDash([5,5]);g.strokeRect(X(a.xmin+e),Y(a.ymax-e),(a.xmax-a.xmin-2*e)*sc,(a.ymax-a.ymin-2*e)*sc);g.setLineDash([]);
  for(const p of s.lying||[]){g.fillStyle='#6fd08c';g.beginPath();g.arc(X(p[0]),Y(p[1]),7,0,7);g.fill()}}
 for(const p of s.floor_tags||[]){g.strokeStyle='#8b95a3';g.lineWidth=1;g.strokeRect(X(p[0])-8,Y(p[1])-8,16,16)}
 for(const o of s.objects){g.fillStyle=o.state=='held'?'#6fd08c':o.cls=='push'?'#e0655f':'#f2b84b';
  g.beginPath();g.arc(X(o.x),Y(o.y),Math.max(6,o.r*sc),0,7);g.fill();
  if(o.owner){g.strokeStyle=COL[o.owner]||'#fff';g.lineWidth=3;g.stroke()}
  g.fillStyle='#11151a';g.font='11px system-ui';g.fillText(o.id,X(o.x),Y(o.y)+4)}
 for(const r of s.robots){const c=COL[r.name]||'#fff';
  if(r.path&&r.path.length){g.strokeStyle=c;g.globalAlpha=.45;g.lineWidth=2;g.setLineDash([6,6]);g.beginPath();g.moveTo(X(r.x),Y(r.y));
   for(const p of r.path)g.lineTo(X(p[0]),Y(p[1]));g.stroke();g.setLineDash([]);g.globalAlpha=1}
  g.fillStyle=c;g.globalAlpha=.25;g.beginPath();g.arc(X(r.x),Y(r.y),r.r*sc,0,7);g.fill();g.globalAlpha=1;
  g.strokeStyle=c;g.lineWidth=3;g.beginPath();g.arc(X(r.x),Y(r.y),r.r*sc,0,7);g.stroke();
  g.beginPath();g.moveTo(X(r.x),Y(r.y));g.lineTo(X(r.x+Math.cos(r.th)*r.r*1.3),Y(r.y+Math.sin(r.th)*r.r*1.3));g.stroke();
  g.fillStyle='#fff';g.font='600 16px system-ui';g.fillText(r.name,X(r.x),Y(r.y)-r.r*sc-6)}
 document.getElementById('clock').textContent=s.t.toFixed(1)+' s';
 document.getElementById('clock').className='big'+(s.done?' done':'');
 document.getElementById('count').textContent=s.edge?s.delivered+' object(s) put down at the edge'+(s.done?' — finished':''):s.delivered+' of '+s.total+' objects in the bin'+(s.done?' — arena clear':'');
 document.getElementById('robots').innerHTML=s.robots.map(r=>`<tr><td><span class="chip" style="background:${COL[r.name]||'#fff'}"></span><b>${r.name}</b> <span class="k">${r.role}</span></td><td>${r.state}<br><span class="k">${r.note||''}</span></td><td class="k">${r.batt?(r.batt/1000).toFixed(1)+' V':''}</td></tr>`).join('');
 document.getElementById('log').textContent=s.log.slice().reverse().join('\n')}
setInterval(()=>fetch('/state').then(r=>r.json()).then(draw).catch(()=>{}),200);
</script>"""


def serve(co, port=PORT_DASH):
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/state"):
                body, ctype = json.dumps(co.state()).encode(), "application/json"
            else:
                body, ctype = PAGE.encode(), "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("0.0.0.0", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"dashboard: http://localhost:{port}")
    return srv
