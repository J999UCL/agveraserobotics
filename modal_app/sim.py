"""Run the hot-swap simulation server on a Modal GPU and expose it over a TCP tunnel.

    modal run modal_app/sim.py                    # L40S, 3 hours, prints HOST:PORT
    modal run modal_app/sim.py --gpu L4 --hours 1
    modal run --detach modal_app/sim.py           # keeps running if your laptop disconnects

Teammates then connect from anywhere:

    HOTSWAP_SIM=HOST:PORT python scripts/client_test.py          (or the isaac-mcp server)

The address is also stored in the Modal Dict "hotswap-sim" (key "address"), so
`python -m modal_app.address` prints it for anyone on the same Modal workspace.
"""

from __future__ import annotations

import modal

from common import REMOTE_ROOT, ensure_vulkan_icd, image

app = modal.App("hotswap-sim", image=image)
registry = modal.Dict.from_name("hotswap-sim", create_if_missing=True)
PORT = 5555


def _serve(hours: float) -> None:
    import subprocess
    import sys
    import time

    print(ensure_vulkan_icd(), flush=True)
    with modal.forward(PORT, unencrypted=True) as tunnel:
        host, port = tunnel.tcp_socket
        address = f"{host}:{port}"
        registry["address"] = address
        registry["started"] = time.time()
        print("=" * 60, flush=True)
        print(f"  SIM ADDRESS (share this):  {address}", flush=True)
        print(f"  export HOTSWAP_SIM={address}", flush=True)
        print("=" * 60, flush=True)
        proc = subprocess.Popen(
            [sys.executable, "scripts/run_sim_server.py", "--headless", "--port", str(PORT), "--max-hours", str(hours)],
            cwd=REMOTE_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        for line in proc.stdout:  # forward the server's own log lines (skip Kit noise)
            if "[sim-server]" in line or "Traceback" in line or "Error" in line[:120]:
                print(line, end="", flush=True)
        proc.wait()
        registry.pop("address", None)


@app.function(gpu="L40S", timeout=24 * 3600)
def serve_l40s(hours: float = 3.0) -> None:
    _serve(hours)


@app.function(gpu="L4", timeout=24 * 3600)
def serve_l4(hours: float = 3.0) -> None:
    _serve(hours)


@app.local_entrypoint()
def main(gpu: str = "L40S", hours: float = 3.0):
    fn = serve_l4 if gpu.upper() == "L4" else serve_l40s
    fn.remote(hours)
