"""Modal smoke test: can Isaac Sim render our scene's cameras on a Modal GPU?

    modal run modal_app/smoke.py            # default GPU: L4
    modal run modal_app/smoke.py --gpu L40S

Builds an image with Isaac Sim (pip), runs scripts/phase1_scene.py headless, and copies
the wrist/overview frames + logs back to outputs/modal_smoke/. The first build takes a
while (Isaac Sim is many GB). Later runs reuse the cached image.
"""

from __future__ import annotations

import pathlib

import modal

ROOT = pathlib.Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/root/project"
ISAAC_VERSION = "6.0.1.0"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        # X/GL/Vulkan client libraries Kit's renderer loads, even when headless
        "libglu1-mesa", "libgl1", "libegl1", "libxt6", "libxrandr2", "libxinerama1",
        "libxcursor1", "libxi6", "libsm6", "libice6", "libxkbcommon0", "libvulkan1",
        "vulkan-tools", "libgomp1",
    )
    .env({
        "ACCEPT_EULA": "Y",
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "PRIVACY_CONSENT": "Y",
        "NVIDIA_DRIVER_CAPABILITIES": "all",
    })
    .pip_install(f"isaacsim[all,extscache]=={ISAAC_VERSION}", extra_index_url="https://pypi.nvidia.com")
    .pip_install("pillow")
    .add_local_dir(ROOT / "hotswap", f"{REMOTE_ROOT}/hotswap")
    .add_local_dir(ROOT / "scripts", f"{REMOTE_ROOT}/scripts")
)

app = modal.App("hotswap-smoke", image=image)

NVIDIA_ICD = """{"file_format_version": "1.0.0",
 "ICD": {"library_path": "libGLX_nvidia.so.0", "api_version": "1.3.0"}}"""


def _sh(cmd: str) -> str:
    import subprocess

    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return f"$ {cmd}\n{p.stdout}{p.stderr}\n"


def _smoke(timeout_s: int = 1800) -> dict[str, bytes]:
    import os
    import subprocess

    report = _sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv")
    report += _sh("ls /usr/share/vulkan/icd.d /etc/vulkan/icd.d 2>&1; ldconfig -p | grep -E 'GLX_nvidia|nvidia-glcore|EGL_nvidia' | head")

    # Vulkan needs an NVIDIA ICD manifest. Some container runtimes don't inject one;
    # write it ourselves if the driver library is present.
    icd = pathlib.Path("/usr/share/vulkan/icd.d/nvidia_icd.json")
    if not icd.exists():
        icd.parent.mkdir(parents=True, exist_ok=True)
        icd.write_text(NVIDIA_ICD)
        report += f"(wrote {icd})\n"
    report += _sh("vulkaninfo --summary 2>&1 | head -40")

    env = dict(os.environ)
    p = subprocess.run(
        ["python", "scripts/phase1_scene.py", "--headless"],
        cwd=REMOTE_ROOT, capture_output=True, text=True, timeout=timeout_s, env=env,
    )
    log = p.stdout + p.stderr
    report += f"phase1 exit code: {p.returncode}\n"
    report += "\n".join(line for line in log.splitlines() if "[phase1]" in line or "Error" in line[:200])

    out = {"report.txt": report.encode(), "phase1_full.log": log.encode()}
    frames = pathlib.Path(REMOTE_ROOT) / "outputs" / "phase1"
    if frames.exists():
        for f in frames.iterdir():
            if f.suffix in {".png", ".json"}:
                out[f.name] = f.read_bytes()
    return out


@app.function(gpu="L4", timeout=3600)
def smoke_l4() -> dict[str, bytes]:
    return _smoke()


@app.function(gpu="L40S", timeout=3600)
def smoke_l40s() -> dict[str, bytes]:
    return _smoke()


@app.local_entrypoint()
def main(gpu: str = "L4"):
    fn = smoke_l40s if gpu.upper() == "L40S" else smoke_l4
    files = fn.remote()
    dest = ROOT / "outputs" / "modal_smoke"
    dest.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (dest / name).write_bytes(data)
    print(files["report.txt"].decode()[-4000:])
    print(f"\nSaved {len(files)} files to {dest}")
