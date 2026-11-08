"""Shared Modal image: Isaac Sim (pip) + rendering libraries + this project's code."""

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
    .add_local_dir(ROOT / "hotswap", f"{REMOTE_ROOT}/hotswap", ignore=["__pycache__"])
    .add_local_dir(ROOT / "scripts", f"{REMOTE_ROOT}/scripts", ignore=["__pycache__"])
    .add_local_python_source("common")  # this module, so modal_app/sim.py can import it remotely
)


NVIDIA_ICD = """{"file_format_version": "1.0.0",
 "ICD": {"library_path": "libGLX_nvidia.so.0", "api_version": "1.3.0"}}"""


def ensure_vulkan_icd() -> str:
    """Some container runtimes don't inject NVIDIA's Vulkan manifest; write it if missing."""
    icd = pathlib.Path("/usr/share/vulkan/icd.d/nvidia_icd.json")
    if icd.exists():
        return "vulkan icd present"
    icd.parent.mkdir(parents=True, exist_ok=True)
    icd.write_text(NVIDIA_ICD)
    return f"wrote {icd}"
