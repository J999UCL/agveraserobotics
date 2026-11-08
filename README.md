# Autonomous data-center maintenance robot (MuJoCo)

Agentverse Hackathon, Robotics Track. A mobile manipulator patrols a data-center aisle. It reads
server health and work orders, drives to the right rack, slides the server tray out, finds the failed
RAM stick or fan (amber LED), hot-swaps it with a spare it carries, verifies the fix, and restocks at
the depot. It stops for people in the aisle and aborts safely on stuck parts.

The AI reasoning (deciding what to fix and in what order) sits on top, using the MCP tools or the
Python client.

```
 agents (LLM) ──MCP──► mcp_server.py ─┐
 your Python ─────────► dcsim_client ─┼── TCP/JSON ──► scripts/run_server.py
                                      │                 DataCenterSim (MuJoCo, 50 Hz control)
```

## Run it

```bash
pip install -r requirements.txt -r requirements-client.txt
python scripts/run_server.py --viewer        # MuJoCo window + server on :5555
python scripts/client_test.py                # other terminal: full shift via the client
```

- `python scripts/sim_test.py`: the same shift in-process, headless, about 80 s.
- `python scripts/record_demo.py`: writes `outputs/demo.mp4` at 4× speed (follow camera, wrist inset, captions).

### MCP (Claude Code or any MCP client)

```json
{ "mcpServers": { "datacenter-robot": {
    "command": "python", "args": ["mcp_server.py"], "cwd": "/path/to/repo",
    "env": { "DCSIM": "localhost:5555" } } } }
```

## What's simulated

| Thing | Details |
|---|---|
| Room | 2 rows × 6 racks around a cold aisle, raised floor with perforated tiles, cable trays, cooling units |
| Servers | 4 serviceable 2U servers (`R1-02-U10`, `R1-05-U20`, `R2-03-U14`, `R2-06-U24`). Each has 8 DIMMs (`A1`–`A8`), 5 fans (`F1`–`F5`), and its own health and LEDs. The other servers are scenery |
| Robot | Mecanum base + lift column + Franka Panda (MuJoCo Menagerie). Onboard spares cassette and returns bin ride on the lift carriage |
| Sensors | Wrist RGB-D camera, front 2D lidar (37 rays), virtual bumper, pull-force readout |
| Depot | Dock + spares shelf at the start of the aisle |
| Diagnostics | BMC-style per-server health (ECC errors, fan RPM, temps), event logs, work orders |
| Demo triggers | Inject faults, jam a part, a technician walking into the aisle |

Default faults: **R1-05-U20 A3** (RAM, uncorrectable ECC) and **R2-03-U14 F2** (stalled fan).

## Simplifications, on purpose

- **The base drives in software** (its joints are commanded directly); there are no wheel physics.
- **Grasping and seating are assisted by constraints.** The arm must line up the part; attaching and seating are then switched on. That's standard for simulation demos.
- **Chunkier RAM:** DIMMs are thicker and wider spaced than real ones so the gripper fingers fit.
- **The tray pull tab is gripped tilted 75° down.** A level grasp at that height is outside the Franka's reach.

## Layout

| Path | What |
|---|---|
| `dcsim/config.py` | Room, rack, tray, robot dimensions; servers; thresholds |
| `dcsim/build.py` | Builds the MuJoCo model (MjSpec) around the Menagerie Franka |
| `dcsim/sim.py` | Robot base + lift + arm control, navigation, lidar, grasping/seating, LEDs, work orders |
| `dcsim/health.py` | Server health model |
| `dcsim/skills.py` | go_to, go_home, restock, tray in/out, pick, place, discard, swap |
| `dcsim/recorder.py` | MP4 recorder with overlays |
| `dcsim/server.py`, `dcsim_client.py`, `mcp_server.py` | Network layer + agent tools |
| `docs/api.md` | Full API |
| `docs/plan.md` | Build plan and status |

The earlier Isaac Sim version is kept in git under the tag `isaac-sim-version`.

## Credits

Franka Panda model from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
(Apache 2.0, licence in `assets/franka_emika_panda/LICENSE`). Skill-oriented MCP design inspired
by [panda-mcp](https://github.com/ratsbane/panda-mcp) (MIT).
