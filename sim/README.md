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

Each rail is a bounded 0–720 mm MuJoCo slide joint with damping, friction and a 12 kg assembly mass. `O`/`C` apply a bounded 100 N service force with position/velocity feedback; this is a manual viewer aid. The arm is not yet controlled to perform extraction. Robot/cart obstructions can physically block a tray; the rail controller does not teleport through them. `M` removes the service aid from the selected tray for later robot interaction.

The arm and cart stay parked in this standalone asset viewer. The native OpenPI/LIBERO execution and recorded replay workflow is described below.

## Service base: rail and lift

The Panda sits on a lift (up/down, 1.55 m stroke) on a carriage that runs on a floor rail along the aisle (left/right, x from −1.5 to 1.5 m, 0.43 m in front of the rack fronts). Both are position-servoed slide joints; the lift's servo cancels the weight it carries. At home the base is parked beside rack 4 with the arm folded, clear of every tray.

`robot/arm/base.py` lines the arm up with a server: `Base(m, d).go_to(rack, u)` stands the arm 0.5 m to the right of that tray (so a fully extended tray slides past it) with its shoulder 20 cm above the handles. From there one arm posture can pull a tray all the way out; level with the handle it only manages about 24 cm. It refuses to move while a tray is open or a part is held, folds the arm first, and stops in place on any contact (reported as `BLOCKED`). Same calls, same motion: the tests check that two runs are bit-identical.

`robot/arm/arm.py` moves the hand: `move_free` (joint-space, path checked for collisions first), `move_linear` (straight line), and `guarded_move` (slow straight line that stops when the surface pushes back past a force limit). The force at the fingertips is read from the joint torques, the way the real Panda senses it, and matches the simulator's contact force. Arm poses come from a fixed list of starting postures, keeping the one furthest from the joint limits, so they are repeatable. A posture is only chosen if every straight move that follows it can be done without changing posture.

`robot/arm/skills.py` opens and closes trays with the arm: `TrayService(base, arm).open_tray(rack, u)` lines up, opens the fingers to 26 mm (the handle is 14 mm, the rack wall 45 mm away), plans one posture for reach, grip, 70 cm pull and back-off (collision-checked with the tray moving), grips the right handle and pulls. More than 30 N of resistance, or the handle slipping in the grip, is a jam: it pushes the tray back, lets go and reports `JAMMED`. `close_tray` grips the extended handle and pushes until the closed stop pushes back, then folds the arm.

To make that physically possible the scene uses MuJoCo's elliptic friction cone with `impratio=10` (its recommended setting against grasp slip), the gripper's "closed" aims 10 mm past shut so it squeezes a held handle at about 70 N like the real Franka (the stock model squeezes about 1 N), and handle contacts are stiff so the fingers don't sink in.

```sh
.venv-mujoco/bin/python -m pytest robot/arm -q     # ~5 min
.venv-mujoco/bin/mjpython robot/arm/demo.py        # watch it live (Windows: .venv-mujoco/Scripts/python)
```

The viewer keys above do not drive the base; it holds its home position. The base, arm and tray skills are deterministic: no model, training or remote inference.

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

The check advances ten seconds at home and then extends every rail to 700 mm and closes it under force, checking position error and MuJoCo warnings. Arm/cart contacts are temporarily disabled only during the all-rail clearance check, so the check covers rack clearance independently of where the base is. Normal simulation and showcase rendering retain these contacts. Detailed hardware meshes are visual; tray shell/handle colliders and separate rack frame colliders provide contact. Adjacent trays use collision masks to avoid redundant checks, and the one-axis joints keep them on distinct rail levels.

Collision proxies are in group 3, hidden by default. Enable that group in the viewer to inspect them. The Panda keeps its supplied collision meshes and gripper contacts.

See [ASSETS.md](ASSETS.md) for provenance.

## Agent execution and recorded replay

The policy path uses the pinned OpenPI `serve_policy.py --record`, its websocket
client, and LIBERO's supplied `OffScreenRenderEnv` / `env.step` APIs. LIBERO owns
the MountedPanda, OSC controller, gripper, camera rig and reset behavior. Actions
pass through unchanged. The old Menagerie controller and coordinate bridge have
been removed. This follows the [OpenPI example](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/libero/main.py) and [LIBERO scene API](https://lifelong-robot-learning.github.io/LIBERO/html/procedural_generation/scene_creation.html). The original standalone scene viewer remains an asset preview.

The rack task uses registered `MujocoXMLObject` fixtures and BDDL in the existing
`LIBERO_Tabletop_Manipulation` problem. Four floor racks contain 80 articulated
servers; the cyan handle on rack B, U21 identifies the target. A scene XML hides
the tabletop geometry while preserving the standard robot workspace anchors.
Server visuals are consolidated meshes to respect robosuite 1.4.1's renderer
capacity; the collision shells and sliding joints remain physical.

`./sim/cloud.sh validate` checks the stock controller on CPU, including a ten
second hold, XYZ response, and target projection into the supplied camera.
The GPU batch checks both rendered camera observations before starting the policy,
then runs the unmodified supplied LIBERO example followed by the rack trials.
Benchmark success and rack success are reported separately. Pretrained LIBERO
performance does not establish generalization to server extraction.

`./sim/cloud.sh resume --gpu A10 --episodes 3 --seconds 1200` continues the existing
reservation. All jobs use `j999ucl`; A10 is the default, and T4/L4 are explicit
options. The ledger in `.runtime/mujoco-agent/budget.json` retains the original
$5 cap and historical attempts. Reservations are conservative runtime allowances,
not billing measurements. Concurrent batches and provider retries are disabled.
No training is configured. `status`, `fetch RUN_ID`, and `stop` inspect, retrieve,
and stop the workload.

The official robosuite `DataCollectionWrapper` saves model XML, states and actions
at every control step (20 Hz). OpenPI's official policy recorder saves the exact
model inputs and outputs. Packaging copies referenced assets beside each model;
only file paths change. Completed recordings are under `recordings/ep_*` within
the attempt directory, with trial outcomes in `results.json`.

`./sim/replay.sh EPISODE_DIRECTORY` opens a free-camera 3D playback with the same
MuJoCo 3.2.3 version as LIBERO. It restores recorded states and calls `mj_forward`;
it never advances physics or runs inference. Space pauses, arrows seek, R restarts,
and 1–4 select 0.25×–2× speed. Mouse controls orbit, pan and zoom. Older recordings
retain their historical read-only viewer.

Run local checks with:

```sh
.venv-mujoco/bin/python -m unittest discover -s sim -p 'test_*.py'
```

The pinned `pi05_libero` config has `discrete_state_input=False`: its network
conditioning uses images and language, while numeric state remains present in
the standard API and recording. A state-sign correction alone cannot explain or
fix the old policy's first predicted action. The old rollouts predominantly
commanded downward motion; they do not prove the controller alone caused folding.
Sources: [OpenPI LIBERO example](https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479/examples/libero),
[checkpoint config](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/config.py),
[model](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py).

### Measured results, 2026-11-08

The unmodified `libero_goal` example succeeded on 10/10 tasks on A10. The native
rack task completed three 990-action trials (plus ten settling steps) with zero
extractions; the recorded maximum joint positions were 0.132, 0.226 and 0.019 mm. A later state audit established that these equal the randomized reset offsets: actual displacement was zero. CPU hold drift
was 2.5e-12 m and XYZ direction checks passed. Native rendering used 809/1000
geoms. These results validate the API and recording path, while demonstrating
that this pretrained checkpoint has not solved the rack scene.

Across the completed native trials, 59–78% of policy actions requested downward
motion and minimum gripper heights were 0.51–0.53 m, while the handle was at 1.05 m. The behavior
persists with the supplied controller. A scene/task distribution mismatch is a
plausible explanation; its exact perceptual cause has not been isolated. The
initial camera views are oblique and visually dense. No successful maintenance
behavior or adaptation has been demonstrated.

Evidence: local run `20261108-015438-69171c`, baseline in `attempt-07/baseline.log`,
completed rack results and vendor recordings in `attempt-08/`. The first native
attempt exceeded the stock episode horizon; the rollout now respects
`env.horizon`. Playback was verified against the recorded MuJoCo 3.2.3 model and
state layout. No custom controller, robot reset, camera transform, or model action
conversion is active.

### Controlled investigation

See [RACK_DIAGNOSIS.md](RACK_DIAGNOSIS.md) for 240 matched-noise inference probes and ten short native closed-loop rollouts. Familiar benchmark wording removes the large upward excursion; reducing visible clutter improves approach further. Every tested run still stops before contact with the target tray. Moving the target rack 18 cm closer makes its handle visible in the wrist camera but does not produce a grasp. The recording wrapper reproduces the unwrapped path to 2.2e-8 m, and all checked physical model arrays match between conditions. The normal scene retains four visible racks; the alternate asset directory is an explicit experiment option.

`--max-steps` bounds diagnostic rollouts and `--prompt` supplies an experimental instruction. `AGVERASE_RACK_ASSETS` selects alternate fixture XML while retaining the vendor environment. New result fields distinguish `initial_extension_m` from `max_extension_change_m`; legacy `max_pull` is the absolute maximum slider position, not travel from reset.
