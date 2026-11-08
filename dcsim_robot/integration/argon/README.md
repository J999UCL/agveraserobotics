# Running the Argon agent on the dcsim robot

These files connect the team's reasoning agent (`robot/sandbox` on the `argon-reasoning-agent` and
`agent-perception` branches of J999UCL/agveraserobotics) to this simulator. The agent's own code
(`agent.py`, `policy.py`, `scenarios.py`, `world.py`) is unchanged.

`DcsimRack` replaces `World`, the same way their `MujocoRack` does: same methods, same return
strings. The incident (drive health, RAID, spares) stays in `World`; the robot physically carries
out every action.

## Run it

Start the simulator (add `--viewer` to watch):

```bash
python scripts/run_server.py
```

Copy the three `.py` files from this folder into `robot/sandbox/` of the team repo, then from there:

```bash
DCSIM=localhost:5555 DCSIM_REPO=/path/to/AgentVerseBob2 python run_dcsim_argon.py --only clean_failure
```

| Option | Meaning |
|---|---|
| `--brain modal` (default) | Argon on Modal; the Gemini key stays in the Modal secret |
| `--brain gemini` | Argon called from this machine; needs `GEMINI_API_KEY` |
| `--brain none` | autopilot only; anything it escalates ends in the safe fallback |
| `--mode argon` | Argon decides every step (default `local_first`) |
| `--server R2-03-U14` | which server the 8 bays live on |

Traces go to `runs/dcsim/`, camera frames to `runs/dcsim/<mode>_<scenario>/`.

For a simulator on Modal, set `DCSIM` to the address it prints. The agent needs `pillow` and
`numpy`; `dcsim_client.py` is the only file it uses from this repo.

Tests (scripted decisions, no Gemini; needs `pytest`):

```bash
DCSIM=localhost:5555 DCSIM_REPO=/path/to/AgentVerseBob2 python -m pytest test_dcsim_plugin.py -q
```

## How the agent's actions map to the robot

| Agent | Robot |
|---|---|
| 8 bays | RAM slots `A1`-`A8` of one server |
| `move_to` | drives to the server if needed, opens the tray, points the wrist camera at it |
| `pull_caddy` | `pick` the stick. A jam is physical: the force reflex aborts at 25 N |
| `stow_old` | `discard` into the returns bin |
| `pick_spare` | `pick` from the cassette. If it is empty but spares remain, drives to the depot, restocks, comes back |
| `insert_caddy` | `place` into the empty slot |
| `wait` | 3 s of sim time; the agent's RAID and obstruction counters advance as in `World` |
| obstacle | a technician blocks the aisle; the lidar stops the robot and the move returns `BLOCKED` |
| `look` / `perceive` | wrist RGB-D. Each LED is located from the slot map, checked against depth, and its colour read from pixels |

## Limits

- **The agent still talks about drives.** Its prompt says caddies, SMART and RAID. Rewording it for
  RAM (ECC errors) and fans is the next step. Telemetry and RAID state are logical, from `World`,
  not from this simulator's health model.
- **Thermal is synthetic**, inherited from `World` (as in `MujocoRack`).
- **RAM only.** Fans (`F1`-`F5`) work in the simulator but are not mapped to bays.
- **Spare stock:** `World.spares` is the agent's count. The robot starts with 2 RAM spares onboard
  and the depot has 6.
- **Start position:** the robot starts at the server with the tray open, because the LEDs are only
  visible from there. In the obstacle scenario it takes that first look, then returns to the dock.
- **Rand's `vision/` package isn't used yet.** It is tuned to the front panels of Jeet's scene; LED
  reading here is a small version written for the open-tray view.

## Test results

| | |
|---|---|
| 12 scenarios, autopilot only | same outcome as the toy world: 6 pass (`clean_failure`, `obstacle`, `raid_rebuild`, `no_spares`, `two_failures`, `same_group_double`); 6 escalate to Argon and stop safely |
| 7 scripted-decision tests | pass: swap, physical jam, two failures, technician in the aisle, dead spare, restock trip, LED reading |
| Real Gemini | not tested |

## Camera perception (vision/)

`perceive()` and the pre-pick check use `vision/tray.py` from the team repo, found automatically
when this file sits inside the team repo; otherwise set `VISION_REPO`. One wrist frame over the
open tray gives:

- **LED per bay.** Glowing blobs are matched to each slot's known LED position (`get_slot_leds`), and lights that aren't slot LEDs are ignored. A missing LED is reported as "hidden" (depth shows something in front of it) or "off" (empty slot). Lime, a new part settling in, is reported as "verifying", never a fault.
- **Blocked slots.** Depth is used to check whether anything lies in the gripper's way down to each part (`get_slot_grasp_points`). Slots blocked by the same object are grouped into one obstacle.

**Two obstacle modes:**
- **`DcsimRack(..., obstacle_mode="aisle")` (default):** the `obstacle` scenario is a technician in the aisle, and the **lidar** stops the robot.
- **`obstacle_mode="tray"`:** a loose cable lands across the open tray over the bay (`drop_cable`). Nothing physical stops the arm; the **wrist camera** sees it, so `move_to` and `pull_caddy` return `BLOCKED` until it's cleared.

The camera check before every pick is on in both modes.
