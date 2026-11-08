# vision: camera perception for the robot (which drive is faulty, where to grab it, what is in the way)

The package is called `vision` rather than `perception` so it can sit next to the agent's `robot/sandbox/perception.py` without an import clash.

Takes one wrist-camera frame (RGB + depth + intrinsics + camera pose) and returns:

- **every drive bay of the server in view**, numbered left to right, top to bottom, with its LED
  state: `ok` (green), `fault` (amber/red), `identify` (blue), `off`, or `hidden` (something is in
  front of it)
- **where to grab** each bay's latch, in the robot base frame (Panda `link0`, +X towards the racks)
- **obstacles** between the gripper and the rack (e.g. a dropped cable), and which bays they block
- **reasons** for every flag, so each decision traces back to what the camera saw (rubric 2.1)

It never uses ground truth for the fault: the faulty drive is found in the image. Fully on-device
(numpy + OpenCV), about **20–30 ms per frame** on a laptop (rubric 2.3 budget: 100 ms).

## With the team's MuJoCo scene (`sim/datacentre.xml`)

```python
from vision import datacentre as dc

m, d = dc.load()                                   # adds per-drive status LEDs + a wrist camera
dc.set_led(m, rack=2, slot=6, bay=5, colour="amber")
cap = dc.capture(m, d, *dc.scan_pose(m, d, 2, 6))  # or dc.capture_wrist(m, d) once the arm moves
scan = dc.make_perception(m, d).scan(cap, target_xyz=dc.server_target(m, d, 2, 6))

scan.faults()          # [BayObs(bay=5, status="fault", tab_xyz_base=[x, y, z], ...)]
scan.obstacles         # [Obstacle(clearance_m=0.16, nearest_xyz_base=[...], ...)]
dc.bay_views(scan)     # agent format: [{"bay", "led", "temp_c", "obstructed", "grab_xyz_base", "reasons"}]
scan.summary()         # short text for logs / the reasoning agent
```

`load()` doesn't modify `sim/datacentre.xml`. It uses MjSpec at load time to add:
- **one status LED per drive carrier,** because the scene's own LEDs are a single shared mesh per server
- **a wrist camera on the Panda hand,** with api.md's intrinsics

`server_target()` is map knowledge (where a server is), which api.md allows.

## Run

```powershell
pip install -r vision/requirements.txt
python -m pytest -q vision/tests        # 41 tests
python -m vision.demo                   # fault + dropped-cable scans of the real scene -> out/
python -m vision.check_frame frame.npz  # run on a frame saved from another simulator
```

The tests check, against ground truth:
- **on the real scene:** 4 servers across 3 racks; bay numbering; LED and grab positions within 5 mm; no false obstacles from the server handles; the dropped cable is found and the bays it blocks are flagged; agent-format output; the wrist camera; the time budget
- **on stand-in scenes:** reflections ignored, "hidden" vs "off" LEDs, oblique views, and the camera-axis auto-detection used by `adapter.py`

## Files

| File | What it does |
| --- | --- |
| `hotswap.py` | Core: rack face, LED → bay grid, faults, grab points, obstacles, blocked bays |
| `datacentre.py` | Bridge to `sim/datacentre.xml`: LEDs, wrist camera, captures in the base frame, agent format |
| `adapter.py` | For api.md-style `get_wrist_frames()`; detects USD vs OpenCV camera axes |
| `leds.py` | Status-LED detection (must actually glow, so shiny latches aren't mistaken for LEDs) |
| `geometry.py` | Pixel ↔ 3D maths |
| `synthetic_hotswap.py`, `synthetic.py` | Small stand-in scenes for tests |
| `perception.py`, `tags.py`, `thermal.py` | Optional: AprilTag asset labels and a simulated thermal camera |
| `demo.py`, `check_frame.py` | Demo and saved-frame checker |
