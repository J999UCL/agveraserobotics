"""Drive a running sim server through hotswap_client (any Python, no Isaac needed).

    python scripts/client_test.py [host] [port]
"""

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hotswap_client import SimClient  # noqa: E402

host = sys.argv[1] if len(sys.argv) > 1 else "localhost"
port = int(sys.argv[2]) if len(sys.argv) > 2 else 5555
out = Path(__file__).resolve().parents[1] / "outputs" / "client_test"
out.mkdir(parents=True, exist_ok=True)

sim = SimClient(host, port)
t0 = time.time()
print("ping:", sim.ping(), f"({(time.time() - t0) * 1000:.0f} ms)")
print("status:", {k: v for k, v in sim.get_status().items() if k != "joints"})
print("telemetry:", sim.get_telemetry())
print("go_to_cabinet(0):", sim.go_to_cabinet(0))
print("view_rack:", sim.view_rack())

t0 = time.time()
f = sim.get_wrist_frames()
print(f"frames: rgb {f.rgb.shape}, depth {f.depth.shape} [{f.depth[f.depth > 0].min():.3f}, {f.depth.max():.3f}] m "
      f"in {(time.time() - t0) * 1000:.0f} ms")
from PIL import Image  # noqa: E402
Image.fromarray(f.rgb).save(out / "wrist.png")
u, v = 320, 240
print("center pixel -> robot (server):", sim.get_robot_coords_at(u, v), " (client):", f.pixel_to_robot(u, v).round(4).tolist())

# stop from a second connection while a long move runs
mover = SimClient(host, port)
res = {}
th = threading.Thread(target=lambda: res.update(mover.move_hand_to([0.30, 0.15, 0.58], max_speed=0.05)))
th.start()
time.sleep(1.5)
print("busy rejection:", end=" ")
try:
    sim.view_rack()
except Exception as e:
    print(e)
print("force while moving:", sim.get_pull_force(), " stop:", sim.stop())
th.join()
print("move result after stop:", res)
print("reset:", sim.reset())
print("done")
