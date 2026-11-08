# Local MuJoCo data-centre bay

From the repository root:

```sh
./sim/view.sh --showcase
```

This opens four 42U cabinets with **twenty individual 2U sliding servers in each rack** (80 total). Three servers in the rightmost rack extend under simulated service force to expose the hardware. Omit `--showcase` to start closed.

Each server has a full-depth 760 mm chassis, drive carriers and eight drives, six cooling fans, a motherboard, two CPU heatsinks, eight RAM DIMMs, rear power supplies, connectors and two front handles. All internal parts move rigidly with their tray. These are simplified component geometry; individual RAM/drives are not independently removable yet. Server top covers and rack front doors are omitted for servicing and visibility.

## Viewer controls

| Key | Action |
| --- | --- |
| `1`–`4` | Select rack |
| Up / Down | Select the next / previous 2U server |
| `O` | Extend the selected server |
| `C` | Close the selected server |
| `M` | Release the selected server's service force, leaving a passive rail |
| `V` | Extend three servers in the right rack |
| `R` | Reset all servers and the robot |
| `H` | Hardware close-up camera |
| `F` | Return to the orbit camera |

The selected server has cyan handles; its rack and U positions print in the terminal. Rack numbering runs left to right, U numbering bottom to top. Left-drag orbits, right-drag pans, scroll zooms. Tab shows the native viewer controls. Close the window to stop.

Each rail is a bounded 0–720 mm MuJoCo slide joint with damping, friction and a 12 kg assembly mass. `O`/`C` apply a bounded 100 N service force with position/velocity feedback; this is a manual viewer aid. The robot holds its initial pose and is not being controlled to perform extraction. Robot/cart obstructions can physically block a tray; the rail controller does not teleport through them. `M` removes the service aid from the selected tray for later robot interaction.

The arm and cart stay parked. There is no Gemini, π0.5, training or remote GPU inference.

## Setup and regeneration

Python 3.11 is isolated in `.venv-mujoco`, separate from the existing project environment. To recreate it with `uv` and regenerate the scene:

```sh
./sim/setup.sh
```

`build_scene.py` converts the bundled rack GLB to textured OBJ meshes, preserves its cabinet shell, removes its fused static equipment, and composes `datacentre.xml` with the supplied Panda. `server_hardware.py` creates shared component meshes and individual sliding server bodies. Source assets are bundled; regeneration requires no asset-service login or download. Units are metres, Z-up.

On macOS the passive viewer uses `mjpython`; `view.sh` also supplies the shared-library path needed by uv's Python. [MuJoCo viewer documentation](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer).

## Validation and previews

```sh
.venv-mujoco/bin/python sim/run.py --check-trays
.venv-mujoco/bin/python sim/run.py --showcase --render sim/previews/overview.png
.venv-mujoco/bin/python sim/run.py --showcase --camera hardware --render sim/previews/hardware.png
```

The check advances ten seconds at home and then extends every rail to 700 mm and closes it under force, checking position error and MuJoCo warnings. Arm/cart contacts are temporarily disabled only during the all-rail clearance check, because the parked cart blocks some lower servers. Normal simulation and showcase rendering retain these contacts. Detailed hardware meshes are visual; tray shell/handle colliders and separate rack frame colliders provide contact. Adjacent trays use collision masks to avoid redundant checks, and the one-axis joints keep them on distinct rail levels.

Collision proxies are in group 3, hidden by default. Enable that group in the viewer to inspect them. The Panda keeps its supplied collision meshes and gripper contacts.

See [ASSETS.md](ASSETS.md) for provenance.
