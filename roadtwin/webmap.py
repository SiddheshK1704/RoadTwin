"""Live browser map: a Leaflet page that polls twin.geojson served from the output folder."""
import functools
import http.server
import threading

MAP_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Road Digital Twin</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>
 html,body,#map{height:100%;margin:0;font-family:system-ui,sans-serif}
 .panel{position:absolute;top:10px;right:10px;z-index:1000;background:#fff;padding:10px 14px;
        border-radius:8px;box-shadow:0 2px 8px #0003;font-size:13px;min-width:180px}
 .panel h3{margin:0 0 6px;font-size:15px} .sw{display:inline-block;width:12px;height:12px;
        border-radius:50%;margin-right:6px;vertical-align:middle}
</style></head><body><div id="map"></div>
<div class="panel"><h3>Road Digital Twin</h3><div id="stats">loading...</div>
<label><input type="checkbox" id="follow" checked> follow vehicle</label></div>
<script>
const map = L.map('map').setView([0,0], 2);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
  {maxZoom: 21, maxNativeZoom: 19, attribution: '&copy; OpenStreetMap'}).addTo(map);
const COND = {good:'#3cbe5a', fair:'#e6d200', poor:'#ff8200', bad:'#dc2828'};
const KIND = {pothole:'#e62828', crack:'#ff9600', alligator:'#c800c8', bump:'#ffdc00'};
let layer = null, first = true;
async function tick(){
  try{
    const r = await fetch('twin.geojson?t=' + Date.now()); const gj = await r.json();
    if(layer) map.removeLayer(layer);
    const counts = {}; let veh = null;
    layer = L.geoJSON(gj, {
      style: f => ({color: COND[f.properties.condition] || '#888', weight: 7, opacity: .85}),
      pointToLayer: (f, ll) => {
        const p = f.properties;
        if(p.layer === 'vehicle'){ veh = ll;
          return L.circleMarker(ll, {radius: 8, color:'#fff', weight:2, fillColor:'#1e90ff', fillOpacity:1}); }
        counts[p.kind] = (counts[p.kind]||0) + 1;
        return L.circleMarker(ll, {radius: 5 + 6*p.severity, color:'#000', weight:1,
                                   fillColor: KIND[p.kind], fillOpacity:.9})
          .bindPopup(`<b>${p.kind}</b> #${p.id}<br>confidence ${p.confidence}<br>severity ${p.severity}`+
                     `<br>size ${p.size_m2} m&sup2;<br>seen ${p.hits} frames<br>chainage ${p.chainage_m} m`+
                     `<br>${ll.lat.toFixed(6)}, ${ll.lng.toFixed(6)}`);
      }}).addTo(map);
    document.getElementById('stats').innerHTML = Object.keys(KIND).map(k =>
      `<div><span class="sw" style="background:${KIND[k]}"></span>${k}: ${counts[k]||0}</div>`).join('');
    if(first && layer.getBounds().isValid()){ map.fitBounds(layer.getBounds(), {maxZoom: 18}); first = false; }
    if(veh && document.getElementById('follow').checked) map.panTo(veh, {animate: true});
  }catch(e){}
}
setInterval(tick, 1000); tick();
</script></body></html>
"""


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def start_server(directory, port=8765):
    with open(f"{directory}/map.html", "w", encoding="utf-8") as f:
        f.write(MAP_HTML)
    handler = functools.partial(_Quiet, directory=directory)
    for p in range(port, port + 10):
        try:
            srv = http.server.ThreadingHTTPServer(("127.0.0.1", p), handler)
            break
        except OSError:
            continue
    else:
        raise RuntimeError("no free port for map server")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{p}/map.html"
