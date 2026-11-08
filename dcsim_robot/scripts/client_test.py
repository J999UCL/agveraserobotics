"""Drive a running sim server through the client (no MuJoCo needed): one full shift.

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
orders = sim.get_work_orders("open")
print("open work orders:", [(w["id"], w["server"], w["slot"], w["fault"]) for w in orders])

stop_watch = threading.Event()
seen = set()


def watch():  # a second connection watching while the robot works
    while not stop_watch.is_set():
        st = side.get_status()
        seen.add((st["at_server"], st["holding"]))
        time.sleep(0.5)


threading.Thread(target=watch, daemon=True).start()
for server in dict.fromkeys(w["server"] for w in orders):
    r = sim.go_to(server)
    print(f"go_to {server}: ok={r['ok']} drive {r.get('drive_time_s')} s")
    print("  pull_tray_out:", sim.pull_tray_out()["ok"])
    for w in [w for w in orders if w["server"] == server]:
        r = sim.swap(w["slot"])
        print(f"  swap {w['slot']}: ok={r['ok']}", [(s["step"], s["ok"]) for s in r.get("steps", [])])
    sim.step(200)
    print("  health:", sim.get_server_health(server)["overall"], "| push_tray_in:", sim.push_tray_in()["ok"])
stop_watch.set()
print("work orders now:", [(w["id"], w["status"]) for w in sim.get_work_orders()])
print("go_home:", sim.go_home()["ok"], "| restock:", sim.restock().get("moved"))
print("side connection saw:", sorted(x for x in seen if x[1]))
print(f"done in {time.perf_counter() - t0:.0f} s")
