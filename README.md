# Autonomous data-center hot-swap robot

Agentverse Hackathon, Robotics Track. A Franka Panda on a maintenance cart drives along an
aisle of 42U server cabinets, inspects a storage server with a wrist RGB-D camera, finds the
drive caddy with the amber fault LED, and pulls it out. It reacts to a cable falling into its
path and must abort when a caddy is jammed.

Simulated in **NVIDIA Isaac Sim 6.0**. It runs on a local RTX GPU or on a **Modal** cloud GPU,
and teammates drive it over TCP from any machine.

```
 your code / Claude (isaac-mcp)            Modal GPU (or a local PC)
 ┌──────────────────────────┐   TCP    ┌──────────────────────────────────┐
 │ hotswap_client.SimClient │ ───────► │ run_sim_server.py                │
 │  perception, FSM, LLM    │ ◄─────── │  HotSwapScene (Isaac Sim, 60 Hz) │
 └──────────────────────────┘  JSON    └──────────────────────────────────┘
```

## For the team: quick start

### 1. Get a sim running (one person)

**On Modal (recommended):**
```bash
pip install modal
modal token new
modal run --detach modal_app/sim.py
```
Wait for `SIM ADDRESS (share this): xxx.modal.host:12345` in the output (the first start takes
several minutes while shaders compile). Anyone on the same Modal workspace can also run
`python -m modal_app.address` to get it.

**On a local Windows PC with Isaac Sim at `C:\isaacsim`:**
```bash
C:\isaacsim\python.bat scripts\run_sim_server.py           # add --headless for no window
```
The address is `localhost:5555`.

### 2. Connect (everyone, any machine)

```bash
pip install -r requirements-client.txt
python scripts/client_test.py HOST PORT        # smoke test: moves the arm, grabs frames, tests stop
```

```python
from hotswap_client import SimClient
sim = SimClient("HOST", PORT)
sim.go_to_cabinet(1); sim.view_rack()
f = sim.get_wrist_frames()          # f.rgb, f.depth (m), f.K, f.pixel_to_robot(u, v)
```

The full API is in [docs/api.md](docs/api.md).

### 3. Claude Code / MCP (optional)

`isaac_mcp/server.py` exposes the sim as MCP tools named like panda-mcp (`get_status`,
`move_cartesian`, `gripper_grasp`, `capture_frame`, `get_robot_coords_at`, `stop`, ...) plus
rack tools (`go_to_cabinet`, `view_rack`, `get_pull_force`, `set_jam`, `drop_obstacle`, ...).
Add this to `.mcp.json` in your project:

```json
{
  "mcpServers": {
    "isaac-mcp": {
      "command": "python",
      "args": ["-m", "isaac_mcp.server"],
      "cwd": "/path/to/this/repo",
      "env": {"HOTSWAP_SIM": "HOST:PORT"}
    }
  }
}
```

Keep the obstacle stop and the force abort in a local control loop (`ik_step` + `get_pull_force`).
LLM calls are too slow to react inside a single move.

## Layout

| Path | What |
|---|---|
| `hotswap/config.py` | Every dimension and physics constant |
| `hotswap/rack.py` | Procedural cabinets, 12-bay storage servers, filler servers, room |
| `hotswap/physics.py` | Caddy slide joints, pull-force model, jam, release |
| `hotswap/obstacle.py` | The falling cable |
| `hotswap/robot.py` | Franka, cart, wrist + overview cameras, ready pose |
| `hotswap/scene.py` | `HotSwapScene`: the in-process API |
| `hotswap/server.py` | TCP server running inside the sim loop |
| `hotswap_client.py` | Client (no Isaac needed) |
| `isaac_mcp/` | MCP server over the client |
| `modal_app/` | Modal image, sim server deployment, smoke test |
| `scripts/` | `run_sim_server.py`, `env_test.py` (in-process test of every call), `client_test.py` |

## Credits

The MCP tool naming follows [panda-mcp](https://github.com/ratsbane/panda-mcp) (MIT licence).
The Franka model comes from NVIDIA Isaac Sim's assets.
