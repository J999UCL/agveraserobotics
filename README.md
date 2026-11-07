# Autonomous data-center hot-swap robot

Agentverse Hackathon, Robotics Track. A Franka Panda on a maintenance cart inspects a
42U server rack with a wrist RGB-D camera, finds the drive caddy with the amber fault
LED, and pulls it out. It reacts to falling obstacles and aborts on a jammed caddy.

Simulated in **NVIDIA Isaac Sim 6.0**, running fully locally (RTX 5060 Ti).

## Setup (once per machine)

1. Download the Isaac Sim 6.0 standalone zip and extract it to `C:\isaacsim`
   (`tar -xf isaac-sim-standalone-6.0.0-windows-x86_64.zip -C C:\isaacsim`).
2. Optional: run `C:\isaacsim\warmup.bat` once to pre-compile shaders. Otherwise the first launch is slow.

All code runs with **Isaac Sim's bundled Python**, not a venv:

```bash
C:\isaacsim\python.bat scripts\phase1_scene.py              # opens the GUI
C:\isaacsim\python.bat scripts\phase1_scene.py --headless   # saves frames to outputs\phase1\ and exits
```

## Layout

| Path | What |
|---|---|
| `hotswap/config.py` | Every dimension (rack, server, caddy, cart, camera) |
| `hotswap/rack.py` | Procedural rack, 12-bay storage server, filler servers, room |
| `hotswap/robot.py` | Franka, cart, wrist + overview cameras, ready pose |
| `hotswap/scene.py` | `HotSwapScene`: the API perception and control use |
| `hotswap/geometry.py` | Batch box builder, USD materials, quaternion helpers |
| `scripts/phase1_scene.py` | Phase 1 demo + perception round-trip check |
| `docs/interface.md` | **Read this first if you work on perception or control** |

## Status

- [x] Phase 1: Franka on cart, procedural rack, wrist RGB-D publishing, intrinsics + unprojection verified
- [ ] Phase 2: caddies on prismatic joints, arm can pull one out
- [ ] Phase 3: jam toggle + pull-force readout
- [ ] Phase 4: falling obstacle
- [ ] Phase 5: room detail, video camera paths
