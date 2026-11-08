# Build plan: environment side (everything except the AI reasoning)

Status: ☐ todo · ◐ in progress · ☑ done

## Decisions (defaults agreed)
- Room: 2 rows × 6 racks facing one cold aisle (1.8 m wide), hot aisles behind.
- Robot: generic mobile manipulator, i.e. a compact wheeled base + telescoping lift column + Franka.
  No trolley. Onboard spares cassette + returns bin ride on the lift carriage (always in reach).
- 4 serviceable servers in different racks at different heights; all other servers are static scenery.
- Sensors: 2D lidar (fan of MuJoCo rangefinders, low on the base) + bumper (touch sensor).
  Optional front depth camera. The obstacle reaction is a reflex in the nav loop, not an LLM call.
- The AI layer only calls skills (`go_to`, `swap`, ...) and reads results/telemetry.

## Phases

### 1. Mobile base, lift, onboard storage ☑
- Base joints on the Franka's `link0`: x, y, yaw (driven), lift z. Chassis body coupled to x/y/yaw.
- Lift range ~0.30–1.30 m (Franka base height). The arm-to-tray geometry stays as already tested:
  base 0.70 m from the rack front, tray floor 0.03 m above the Franka base.
- Onboard storage on the lift carriage: compact side pods (spare DIMMs/fans + returns bin).
  Parts landing in the bin get welded to the carriage so they travel with the robot.
- Refactor indices: the arm is no longer qpos[0:7] (use joint addresses).
- ✔ check: the swap still works in front of one rack.

### 2. Data-center room ☑
- Rows of racks along X with rack labels (R1-01..R1-06, R2-01..R2-06).
- Raised floor (perforated tiles in the cold aisle), CRAC units at the row ends, cable trays,
  ceiling lights, hot-aisle containment.
- 4 serviceable servers (full trays with RAM + fans, own health), e.g. R1-02 U10, R1-05 U20,
  R2-03 U14, R2-06 U24. Server IDs like `R1-05-U20`.
- Dock + spares depot at the aisle end (shelf of spare DIMMs/fans, charging pad).

### 3. Navigation ☑
- Aisle waypoint graph + shortest path. Base follows the path smoothly (sim-kinematic, no wheel physics).
- `go_to(server_id)`: stow arm, drive, turn to face the rack, stop square at 0.70 m, lift to height.
- `go_home()` (dock). Results report what happened (distance, time, waits/reroutes).

### 4. Fleet + work orders + restock ☑
- `get_fleet_health()`: every server's ID, location (row/rack/U) and health summary.
- `get_server_health(server_id)` (existing report, per server).
- Work-order flow: fault → ticket → go_to → swap → verify → close ticket → return to dock.
- `restock()`: at the depot, move spares from the depot shelf into the onboard cassette.
- MCP tools + docs/api.md updated.

### 5. Adaptation, video ☐
- Lidar (36 rays, 180°) + bumper. During `go_to`: stop if anything is within ~0.6 m ahead, wait,
  reroute if it stays blocked.
- Moving obstacle: a technician figure / cart crossing the aisle (`spawn_aisle_obstacle()` demo trigger).
- Follow camera that tracks the robot + recorder (overview + wrist side by side → mp4).
- Tray wall collision so the arm doesn't pass through the tray on camera.

### 6. Ship ☐
- Push to GitHub; Modal (CPU, headless EGL/OSMesa) test; full end-to-end run on video.

## Not in scope
- VLA / learned low-level control (IK stays). SLAM (the map is known). Real wheel dynamics.

## Progress log
- Phases 1-3 + restock/fleet health: a full work shift passes in `scripts/sim_test.py` (drive, pull tray, swap, push in, next server, home, restock, jam abort).
- Phase 4: work orders open on faults and resolve when the new part verifies; server/client/22 MCP tools updated; full shift passes over TCP (`scripts/client_test.py`).
