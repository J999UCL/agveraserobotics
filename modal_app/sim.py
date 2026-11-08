"""Run the data-center sim on Modal (CPU only: MuJoCo + software rendering, no GPU needed).

    modal run modal_app/sim.py --check-only       # headless smoke test: full shift + one frame back
    modal run --detach modal_app/sim.py           # serve the sim; prints HOST:PORT (default 4 h)
    python -m modal_app.address                   # anyone on the workspace (needs `pip install modal`)

Then: DCSIM=HOST:PORT python mcp_server.py   or   SimClient("HOST", PORT)
"""

from __future__ import annotations

import pathlib

import modal

ROOT = pathlib.Path(__file__).resolve().parents[1]
REMOTE = "/root/project"
PORT = 5555

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libosmesa6", "libgl1", "libegl1", "ffmpeg")
    .pip_install("mujoco>=3.3", "numpy>=1.26", "pillow>=10", "imageio[ffmpeg]>=2.34")
    .env({"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa"})
    .add_local_dir(ROOT / "dcsim", f"{REMOTE}/dcsim", ignore=["__pycache__"])
    .add_local_dir(ROOT / "scripts", f"{REMOTE}/scripts", ignore=["__pycache__"])
    .add_local_dir(ROOT / "assets", f"{REMOTE}/assets")
    .add_local_file(ROOT / "dcsim_client.py", f"{REMOTE}/dcsim_client.py")
)
app = modal.App("datacenter-robot-sim", image=image)
registry = modal.Dict.from_name("datacenter-robot-sim", create_if_missing=True)


@app.function(cpu=4.0, memory=4096, timeout=1800)
def check() -> dict:
    import subprocess
    import sys

    sys.path.insert(0, REMOTE)
    p = subprocess.run([sys.executable, "scripts/sim_test.py"], cwd=REMOTE, capture_output=True, text=True, timeout=1500)
    frame = pathlib.Path(REMOTE, "outputs/sim_test/end_follow.png")
    return {"code": p.returncode, "log": (p.stdout + p.stderr)[-4000:], "frame": frame.read_bytes() if frame.exists() else b""}


@app.function(cpu=4.0, memory=4096, timeout=24 * 3600)
def serve(hours: float = 4.0) -> None:
    import subprocess
    import sys
    import time

    with modal.forward(PORT, unencrypted=True) as tunnel:
        host, port = tunnel.tcp_socket
        registry["address"] = f"{host}:{port}"
        print("=" * 60, f"\n  SIM ADDRESS: {host}:{port}\n  export DCSIM={host}:{port}\n" + "=" * 60, flush=True)
        proc = subprocess.Popen([sys.executable, "scripts/run_server.py", "--port", str(PORT)], cwd=REMOTE)
        t_end = time.time() + hours * 3600
        while proc.poll() is None and time.time() < t_end:
            time.sleep(5)
        proc.terminate()
        registry.pop("address", None)


@app.local_entrypoint()
def main(check_only: bool = False, hours: float = 4.0):
    if check_only:
        r = check.remote()
        print(r["log"])
        out = ROOT / "outputs" / "modal_check_follow.png"
        out.parent.mkdir(exist_ok=True)
        if r["frame"]:
            out.write_bytes(r["frame"])
            print(f"frame saved to {out}")
        return
    serve.remote(hours)
