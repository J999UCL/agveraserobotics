"""Drive a running sim server end to end through the client (no MuJoCo needed).

    python scripts/client_test.py [host] [port]
"""

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dcsim_client import SimClient  # noqa: E402

host = sys.argv[1] if len(sys.argv) > 1 else "localhost"
port = int(sys.argv[2]) if len(sys.argv) > 2 else 5555
sim, side = SimClient(host, port), SimClient(host, port)
t0 = time.perf_counter()
print("ping", sim.ping(), f"{(time.perf_counter() - t0) * 1000:.0f} ms")
sim.reset()
h = sim.get_server_health()
bad = [x["slot"] for x in h["dimms"] + h["fans"] if x["status"] != "ok"]
print("health:", h["overall"], "faulty:", bad)
print("pull_tray_out:", sim.pull_tray_out()["ok"])
img = side.capture_image("wrist")
print("wrist image while idle:", img.shape)

# watch the run from a second connection while the swap happens
seen = []
done = threading.Event()
def watch():
    while not done.is_set():
        st = side.get_status()
        seen.append(st["holding"])
        time.sleep(0.3)
threading.Thread(target=watch, daemon=True).start()
for slot in bad:
    r = sim.swap(slot)
    print(f"swap {slot}:", r["ok"], [(s["step"], s["ok"]) for s in r.get("steps", [])])
done.set()
print("side connection saw holding:", sorted({s for s in seen if s}))
sim.step(200)
h = sim.get_server_health()
print("health after:", h["overall"], "temps", h["cpu_temps_c"])
print("busy rejection test:", end=" ")
th = threading.Thread(target=lambda: sim.push_tray_in())
th.start(); time.sleep(0.5)
try:
    side.view_tray()
except Exception as e:
    print(e)
th.join()
print("tray:", side.get_status()["tray"])
print("done")
