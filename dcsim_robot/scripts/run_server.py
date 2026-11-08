"""Run the data-center sim as a server.

    python scripts/run_server.py --viewer            # with the MuJoCo 3D viewer (local)
    python scripts/run_server.py                     # headless (servers / Modal)

Then connect with dcsim_client.SimClient(host, port) or the MCP server (mcp_server.py).
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dcsim.server import SimServer  # noqa: E402
from dcsim.sim import DataCenterSim  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--viewer", action="store_true", help="open the MuJoCo viewer (runs in real time)")
p.add_argument("--host", default="0.0.0.0")
p.add_argument("--port", type=int, default=5555)
p.add_argument("--fast", action="store_true", help="with --viewer: don't slow skills down to real time")
args = p.parse_args()

sim = DataCenterSim(viewer=args.viewer, realtime=args.viewer and not args.fast)
SimServer(sim, args.host, args.port)
print("[sim-server] READY", flush=True)
try:
    while sim._viewer is None or sim._viewer.is_running():
        t0 = time.perf_counter()
        sim.step()  # also serves requests through the step hook
        if not sim.realtime:  # idle at real-time speed even when headless
            time.sleep(max(0.0, sim.cfg.control_dt - (time.perf_counter() - t0)))
except KeyboardInterrupt:
    pass
finally:
    sim.close()
