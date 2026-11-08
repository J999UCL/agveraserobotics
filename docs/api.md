# API

All calls are available on `dcsim_client.SimClient` (and most as MCP tools in `mcp_server.py`).
Positions are in the robot frame: +X toward the rack, +Y left, +Z up, meters.
Skills block until done and return `{"ok": bool, ..., "status": {...}}`. On failure there's an
`"error"` string an agent can reason about (e.g. `"part stuck: extraction aborted to protect the socket"`).

## Diagnostics

| Call | Returns |
|---|---|
| `get_server_health()` | `overall` (ok / warning / critical), `tray`, `usable_memory_gb`, `cpu_temps_c`, `dimms` (per slot A1–A8: present, serial, size_gb, ecc_corrected, ecc_uncorrectable, status), `fans` (per slot F1–F5: present, rpm, status) |
| `get_event_log()` | system event log: faults, parts removed and inserted, verification results, aborted extractions |
| `get_events(since)` | low-level sim events (grasped, unseated, seated, in_bin, post_complete, ...) |
| `get_installed()` | tray slot → part id |
| `get_cart_inventory()` | cart slot → part id (spares) |

Part status values: `ok`, `degraded` (e.g. high correctable ECC rate, slow fan), `failed`, `verifying` / `spinning up`
(just inserted, ~3 s), `absent`.

**Effects to know about:**
- Removing a fan raises CPU temperatures by about 7.5 °C each, and the remaining fans spin faster.
- A failed DIMM doesn't count toward `usable_memory_gb`.

## Skills

| Call | Does |
|---|---|
| `pull_tray_out()` / `push_tray_in()` | slide the server tray to its service position and back |
| `view_tray()` | wrist camera above the open tray. Follow with `capture_image("wrist")` |
| `pick(slot)` | grasp + lift the part in a tray slot (`A1`–`A8`, `F1`–`F5`) or cart slot (`CD1`–`CD4`, `CF1`–`CF2`). Watches the force and aborts above 25 N |
| `place(slot)` | install the held part in an empty, compatible slot |
| `discard()` | drop the held part in the used-parts bin |
| `swap(slot)` | pick → discard → pick a matching spare → place |
| `home()` | ready pose |
| `stop()` | from a second connection: halt the running motion |

## Low level

| Call | Does |
|---|---|
| `move_tcp(x, y, z, orientation="fingers_y")` | straight-line move of the gripper tip, pointing down. `fingers_x` is for the tray handle |
| `open_gripper()` / `close_gripper()` | close reports which part it's holding |
| `get_status()` | gripper position, holding, pull force, tray extension, joints |
| `get_pull_force()` | current and peak extraction force (N) |
| `step(n)` | advance n control ticks (20 ms each) |

## Cameras

| Call | Returns |
|---|---|
| `capture_image(camera)` | RGB numpy array. `"wrist"` (640×480, on the gripper; image up = toward the rack), `"overview"`, `"tray_top"` (1280×720) |
| `capture_rgbd(camera)` | `rgb`, `depth` (m), `K`, `cam_pos`, `cam_rot` |
| `pixel_to_world(u, v, depth, camera)` | robot-frame xyz |

Faulty parts have **amber** LEDs. A DIMM's LED is on the board just behind its slot; a fan's LED is on top of the
module. Healthy fans are green, healthy DIMM LEDs are off, and newly inserted parts glow yellow-green while verifying.

## Demo controls

`inject_fault(slot, fault)` with `dimm_uncorrectable | dimm_correctable | fan_failed | fan_degraded`,
`set_jam(slot, on)`, `reset(seed=None)`.
