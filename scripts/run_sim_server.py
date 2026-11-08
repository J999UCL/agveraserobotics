"""Run the simulation as a server that other processes / machines can drive.

    Windows : C:\isaacsim\python.bat scripts\run_sim_server.py [--headless] [--port 5555]
    Linux   : python scripts/run_sim_server.py --headless          (pip-installed isaacsim)
    Modal   : modal run modal_app/sim.py                           (prints the address)

Then connect with hotswap_client.SimClient("host", port) or the isaac-mcp server.
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=5555)
parser.add_argument("--max-hours", type=float, default=0, help="exit after this long (0 = forever)")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720})

from hotswap.scene import HotSwapScene  # noqa: E402
from hotswap.server import SimServer  # noqa: E402

scene = HotSwapScene(simulation_app)
scene.build()
scene.play()
scene.step(10)
server = SimServer(scene, args.host, args.port)
print("[sim-server] READY", flush=True)

dt = scene.cfg.physics_dt
t_end = time.time() + args.max_hours * 3600 if args.max_hours else float("inf")
next_t = time.perf_counter()
while simulation_app.is_running() and time.time() < t_end:
    scene.step()  # also polls the server via the step hook
    next_t += dt  # idle at real-time speed so the GUI / clients see normal motion
    lag = next_t - time.perf_counter()
    if lag > 0:
        time.sleep(lag)
    else:
        next_t = time.perf_counter()
simulation_app.close()
