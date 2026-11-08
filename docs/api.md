# Environment API

What the simulation gives you. Perception, the state machine, and abort/avoid logic are
**not** in here. You build those on top of these calls.

There are three ways to use it. All of them expose the same calls:

| From | How |
|---|---|
| Your own Python (any machine) | `from hotswap_client import SimClient; sim = SimClient(host, port)` |
| Claude Code / an MCP client | the `isaac-mcp` server (tool names follow panda-mcp) |
| Inside Isaac Sim (scripts in this repo) | `HotSwapScene` in `hotswap/scene.py` |

## Conventions

- **Robot frame everywhere:** origin at the Franka base, +X toward the rack, +Y left, +Z up, meters.
  It moves with the cart, so your numbers don't change between cabinets.
- **Addressing:** `cabinet` 0, 1, 2 (left to right as the robot sees them), then `bay` 0–11 on the
  storage server, numbered left to right and top to bottom (bay 6 = middle row, third from left).
- **Blocking:** moves return when finished. While one runs, a **second** `SimClient` can still
  call quick commands (`stop`, `get_pull_force`, `capture_frame`, triggers). Anything else gets
  a "busy" error.

## Status and cart

| `SimClient` call | Returns / does |
|---|---|
| `ping()` | `"pong"` |
| `get_status()` | cabinet, hand position + quaternion, fingertip position, joints, gripper width, is_grasping, pull force, sim time |
| `go_to_cabinet(n)` | stows the arm, glides the cart (~3 s per cabinet) |
| `current_cabinet()` | int, or None while moving |
| `reset(seed=None)` | everything back to the start; a seed picks a random faulty bay |
| `step(n)` | advance n frames (1/60 s each) |

## Arm

| Call | Does |
|---|---|
| `move_hand_to([x,y,z], orientation=None, max_speed=0.25)` | straight-line move, `{"reached": bool}`. No orientation = gripper pointing into the rack |
| `move_relative(dx, dy, dz)` | |
| `ik_step([x,y,z], steps=1)` | one small step toward a pose, then advances `steps` frames. **Use this to build a reactive loop** (check the camera / force between steps) |
| `move_joints(q7, duration=1.5)` | |
| `view_rack(standoff=0.30)` | camera square to the storage server, about 30 cm back |
| `stow_arm()` | ready pose |
| `stop()` | from a second client: halts the running move, which returns `{"stopped": true}` |
| `open_gripper()` / `close_gripper()` / `gripper_move(w)` | |
| `gripper_width()`, `is_grasping()` | closing on a pull tab gives a width of ~0.051 and `is_grasping=True` |

**Reach:** with the gripper pointing forward, the hand must stay within z 0.55–0.65, x 0.30–0.52,
|y| < 0.16. The pull tabs are at x ≈ 0.605 (front face), z ≈ 0.612 / 0.639 / 0.666 (rows), y ≈ ±0.04 / ±0.15
(columns, tab 12 mm right of bay center). To grasp, put the fingertips ~1.2 cm into the tab:
hand x = tab front + 0.012 − 0.1034.

## Camera

| Call | Returns |
|---|---|
| `get_wrist_frames()` | `Frames`: `rgb` (480×640×3), `depth` (meters, 0 = nothing), `K`, `cam_pos`, `cam_rot`, plus `.pixel_to_robot(u, v)` |
| `capture_frame()` | just the RGB |
| `get_robot_coords_at(u, v)` | robot-frame xyz of that pixel (server-side) |
| `get_depth_at(u, v)` | meters |
| `get_camera_info()` | resolution + K (fx = fy ≈ 366.5, cx = 320, cy = 240) |
| `get_overview()` | 1280×720 third-person image for the video |

## Caddies and force

| Call | Returns |
|---|---|
| `get_pull_force()` | N resisting the caddy being pulled |
| `caddy_status(cabinet, bay)` | `extraction_m`, `force_n`, `jammed`, `released`, `connector_broken` |
| `caddy_extraction(cabinet, bay)` | meters out (0 = seated, ≥ 0.15 = free) |

Measured behaviour:
- **Healthy caddy:** peaks at about 15 N while it unseats, drops to about 0 N while sliding, and comes free at 15 cm, still in the gripper.
- **Jammed caddy:** climbs 13 → 40+ N within 0.2 s. Above **35 N** the connector counts as broken; at about 45 N the tab slips out of the gripper.
- **Suggested abort threshold: around 25 N.**

## Demo triggers

| Call | Does |
|---|---|
| `get_telemetry()` | alerts like `{"cabinet": 1, "bay": 6, "status": "drive_failure"}` plus events (`caddy_released`, `connector_broken`) |
| `set_fault(cabinet, bay)` / `set_led(cabinet, bay, "ok"\|"fault"\|"identify"\|"off")` | |
| `set_jam(cabinet, bay, on=True)` | |
| `connector_broken()` | bool |
| `drop_obstacle(position=None, delay=0)` | a dark cable drops halfway between the fingertips and the rack, in front of the camera, and stays there |
| `clear_obstacle()` | |

## Debug only

`debug_bay_pose(cabinet, bay)` returns the true tab position. **Use it for tests only.** The real
pipeline has to find the bay from the camera, or the perception score suffers.

## Not included (ask if you need it)

Video recording, keyboard shortcuts for triggers, a latch to press, and inserting a replacement drive.
