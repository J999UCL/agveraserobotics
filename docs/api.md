# API

Everything is available through `dcsim_client.SimClient` (Python, needs only numpy + pillow) and as
MCP tools in `mcp_server.py`. Skills block until done and return `{"ok": bool, ..., "status": {...}}`.
Failures come back as an `"error"` string an agent can reason about, e.g. `"part stuck: extraction
aborted to protect the socket"` or `"aisle blocked: obstacle did not clear"`.

## The world

- **Room:** one cold aisle with racks in row R1 (one side) and row R2 (the other), racks 01–06.
- **4 serviceable servers:** `R1-02-U10`, `R1-05-U20`, `R2-03-U14`, `R2-06-U24` (row-rack-U position).
- **Each server tray:** RAM slots `A1`–`A8`, fan slots `F1`–`F5`.
- **Robot storage:** onboard spares `CD1`–`CD4` (RAM) and `CF1`–`CF2` (fans), plus a returns bin.
- **Depot shelf at the dock:** `DD1`–`DD6` (RAM), `DF1`–`DF3` (fans).
- **Slot names:** short names (`A3`) refer to the server the robot is currently at (after `go_to`).

## Diagnostics

| Call | Returns |
|---|---|
| `get_work_orders(status=None)` | tickets: id, server, rack, u, slot, component, fault, severity, status (open / in_progress / resolved) |
| `get_fleet_health()` | per server: overall, problem_slots, CPU temps, rack, U |
| `get_server_health(server)` | per DIMM (ECC counts, status), per fan (rpm, status), CPU temps, usable memory, tray position |
| `get_event_log(server)` | that server's system event log |
| `get_onboard_inventory()`, `get_depot_inventory()` | slot → part id |
| `get_status()` | base x/y/heading/lift, current server, holding, pull force, last collision, onboard spares, tray extensions |

**Part status values:** `ok`, `degraded`, `failed`, `verifying` / `spinning up` (new part, ~3 s), `absent`.

**Work orders** open when a fault appears and **resolve automatically** once a replacement part verifies OK.

## Navigation

| Call | Does |
|---|---|
| `go_to(server)` | folds the arm over the base, drives down the aisle, faces the rack, lifts to the server's height, unfolds the arm |
| `go_home()` | drives back to the dock at the spares depot |
| `restock()` | at the dock, refills empty onboard spare slots from the depot shelf |

**Obstacle handling** happens while driving down the aisle:
- **Stop:** the front lidar watches the travel direction, and the robot brakes if anything is within 0.6 m.
- **Resume:** it waits, and resumes once the path has been clear for 1 s.
- **Give up:** after 20 s blocked, it gives up with an error.

The result's `nav_events` lists the stops, resumes and give-ups.

**Collision watchdog** (always on, during every motion): if any part of the robot presses on a rack,
the depot or a wall with more than 15 N, the robot stops where it is and the call returns
`"collision: <robot part> hit <object> (<force> N); motion stopped"`. `get_status()["last_collision"]`
keeps the details. Touching trays and parts is part of the job and doesn't count.

## Server work (at the current server)

| Call | Does |
|---|---|
| `pull_tray_out()` / `push_tray_in()` | slide the tray to the service position and back (gripping its pull tab from the front) |
| `view_tray()` | holds the wrist camera above the open tray; then `capture_image("wrist")` |
| `swap(slot)` | pick, drop in the returns bin, take a matching onboard spare, install it |
| `pick(slot)` / `place(slot)` / `discard()` | the building blocks. `pick` aborts above 25 N (stuck part) |
| `home()`, `stop()` | ready pose; abort the current motion (from a second connection) |

**Pull forces:**
- **Healthy parts** unseat at about 12–13 N.
- **Jammed parts** keep climbing. The robot aborts at 25 N, below the 40 N socket-damage limit.

## Sensors

| Call | Returns |
|---|---|
| `capture_image(camera)` | `"wrist"` (640×480, on the gripper), `"follow"` (behind the robot), `"overview"`, `"aisle_end"` (1280×720) |
| `capture_rgbd(camera)` | rgb, depth (m), K, camera pose |
| `pixel_to_world(u, v, depth, camera)` | world xyz |
| `get_lidar_scan()` | 37 ranges from −90° to +90° (0 = ahead); `None` = nothing within 4 m |
| `get_pull_force()` | current and peak extraction force (N) |

**Faulty parts have amber LEDs.** A DIMM's LED is on the board behind its slot; a fan's LED is on top of the module.

## Demo controls

| Call | Does |
|---|---|
| `inject_fault(server, slot, fault)` | `fault` is `dimm_uncorrectable`, `dimm_correctable`, `fan_failed` or `fan_degraded`. Opens a work order |
| `set_jam(slot, server, on)` | makes the part in that slot stuck |
| `spawn_technician(mode="cross"\|"block", x=None, delay=0, duration=4)` | a technician walks into the aisle ahead of the robot |
| `reset(seed=None)` | everything back to the start; a seed randomizes the faults |
