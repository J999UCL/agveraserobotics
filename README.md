# Autonomous data-center maintenance robot (MuJoCo)

Agentverse Hackathon, Robotics Track. A Franka Panda on a service cart diagnoses a server
from its health telemetry, pulls the server tray out of the rack, and hot-swaps failed RAM
sticks and fan modules using spares from its cart. Agents reason over diagnostics and camera
images and call the robot's skills.

```
 agents (LLM) ──MCP──► mcp_server.py ─┐
 your Python ─────────► dcsim_client ─┼─ TCP/JSON ─► scripts/run_server.py
                                      │               DataCenterSim (MuJoCo, 50 Hz control)
```

## Run it

```bash
pip install -r requirements.txt -r requirements-client.txt
python scripts/run_server.py --viewer        # MuJoCo window + server on :5555
python scripts/client_test.py                # in another terminal: full diagnose + swap via the client
```

`python scripts/sim_test.py` runs the same thing in-process (no server, a few seconds) and
saves images to `outputs/sim_test/`.

### MCP (Claude Code or any MCP client)

```json
{ "mcpServers": { "datacenter-robot": {
    "command": "python", "args": ["mcp_server.py"], "cwd": "/path/to/repo",
    "env": { "DCSIM": "localhost:5555" } } } }
```

## The scene

| Thing | Details |
|---|---|
| Robot | Franka Panda (MuJoCo Menagerie) at the origin of the cart |
| Rack | 42U cabinet, rack front 0.70 m from the robot |
| Server tray | 2U, slides out 0.32 m on rails by its red handle |
| RAM | 8 DIMMs, slots `A1`-`A8` (abstracted: thicker and wider spaced than real ones so the fingers fit) |
| Fans | 5 fan modules with blue lift tabs, slots `F1`-`F5` |
| Static | 4 front drives, 2 CPU heatsinks, PSU |
| Cart | spares shelf (`CD1`-`CD4` RAM, `CF1`-`CF2` fans) on the right, used-parts bin on the left |
| LEDs | per-DIMM fault LEDs on the board, a LED on each fan, a tray status LED: amber = fault |

Default faults: **A3** (uncorrectable ECC errors) and **F2** (stalled fan). `reset(seed=N)` randomizes them.

## Physics, briefly

- **The tray** is a drawer on a slide joint. The gripper physically drags it by the handle.
- **RAM and fans** are free bodies held by weld constraints that switch on and off:
  - **Seated** parts are welded to their slot.
  - **Pulling** a gripped part stretches a virtual spring: about 12 N unseats a healthy part.
  - **Jammed** parts keep climbing. The `pick` skill aborts at 25 N; above 40 N the socket counts as damaged.
  - **Unseated** parts attach to the gripper.
  - **Placing:** lowering a held part onto a free slot within 12 mm seats it, and the server then "verifies" it for 3 s.
  - **Discarding:** opening the gripper over the bin drops the part in.

## Layout

| Path | What |
|---|---|
| `dcsim/config.py` | All dimensions and thresholds |
| `dcsim/build.py` | Builds the MuJoCo model around the Menagerie Franka |
| `dcsim/sim.py` | `DataCenterSim`: arm control, gripper, seating physics, LEDs, cameras |
| `dcsim/health.py` | BMC-style diagnostics (ECC counts, fan RPM, temps, event log) |
| `dcsim/skills.py` | Reference skills: tray in/out, pick, place, discard, swap |
| `dcsim/kinematics.py` | IK for the gripper tip |
| `dcsim/server.py`, `dcsim_client.py`, `mcp_server.py` | Network layer + agent tools |
| `docs/api.md` | Every call, with arguments and results |

The earlier Isaac Sim version is kept in git under the tag `isaac-sim-version`.

## Credits

Franka Panda model from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
(Apache 2.0, licence in `assets/franka_emika_panda/LICENSE`). Skill-oriented MCP design inspired
by [panda-mcp](https://github.com/ratsbane/panda-mcp) (MIT).
