# Scene interface (for perception + control)

Everything here comes from `hotswap/scene.py`, `hotswap/config.py` and `hotswap/robot.py`.
All units are meters and radians. Quaternions are `wxyz`.

## Frames

| | |
|---|---|
| Origin | Franka base (`panda_link0`), on top of the cart |
| +X | from the robot toward the rack |
| +Y | robot's left |
| +Z | up. The floor is at z = -0.35 |

Isaac's world frame **is** the robot base frame, so a point you unproject is already in
robot coordinates.

## Key positions (default config)

| Thing | Value |
|---|---|
| Rack front posts | x = 0.62 |
| Caddy front faces | x = 0.635 |
| Pull-tab front faces | x = 0.605 (tabs stick out 3 cm) |
| Bay rows (row 0 = top) | z ≈ 0.666, 0.639, 0.612 |
| Bay columns (col 0 = robot's left) | y ≈ +0.159, +0.053, -0.053, -0.159 (tab is 12 mm right of bay center) |
| Faulty bay | `SceneCfg.faulty_bay` = (row 1, col 2), amber LED |
| Ready pose | `READY_Q` in `robot.py`, hand at ~(0.38, 0, 0.65) pointing at the rack |
| Good working band | hand z between 0.55 and 0.65, x between 0.30 and 0.52, y within ±0.16 (all probed reachable) |

Ground-truth bay data (for debugging and scoring only, **not** for the perception
pipeline): `scene.bays[(row, col)]` is a `BayInfo` with `tab_center` and `face_center`.

## Hand orientation

`HAND_FORWARD_QUAT = [0, 0.7071, 0, 0.7071]`: gripper pointing +X into the rack, hand +X = up,
fingers close along Y. The pull tab is 50 mm wide (Y) by 18 mm tall, so it is pinched from the
sides. The gripper opens to 80 mm, which leaves about 25 mm of clearance per finger.

Don't try to point the hand forward much below z ≈ 0.5. In that posture joints 4 and 6
hit their limits. That's why the server sits at U20.

## Wrist camera (`/World/robot/panda_hand/wrist_cam`)

* RGB plus `distance_to_image_plane` depth, 640x480. The camera is 6.5 cm above the hand
  and 4.5 cm forward, looking the same way as the fingers. Image up = world up when
  the hand points at the rack.
* Intrinsics (OpenCV convention, u right, v down): fx = fy ≈ 366.5, cx = 320, cy = 240,
  so about 82° horizontal FOV. Use `scene.intrinsics()` rather than hardcoding these.
* Depth is distance along the optical axis, not ray length. Pixels that hit nothing are `inf`.

```python
frames = scene.get_wrist_frames()        # None until the first render
frames.rgb      # (480, 640, 3) uint8
frames.depth    # (480, 640) float32, meters
frames.K        # 3x3
frames.cam_pos, frames.cam_rot           # camera pose in base frame

from hotswap.scene import pixel_to_base, base_to_pixel
xyz = pixel_to_base(u, v, frames.depth[v, u], frames)   # pixel -> robot frame
```

`scripts/phase1_scene.py` checks this round trip on every run. It currently lands within
1.5 cm of the tab center. The 1.5 cm is half the tab's depth, because depth measures the
tab's front surface.

## Scene controls

```python
scene.set_led((row, col), "ok" | "fault" | "identify" | "off")   # green / amber / blue / off
scene.reset_arm()                       # snap to READY_Q
scene.ik_step(pos, quat)                # one differential-IK step for panda_hand
scene.robot.open_gripper(); scene.robot.close_gripper()
scene.step(n)                           # advance n frames (1/60 s each)
```

Coming in phase 2/3: caddies on prismatic joints, `scene.set_jam(bay, on)`,
`scene.drop_obstacle()`, and a pull-force readout.

## Perception payload (agreed contract)

```python
perception_payload = {
    "target_detected": bool,
    "target_pos_3d": [x, y, z],   # robot base frame = Isaac world frame
    "obstacle_detected": bool,
}
```

Note on the 15 cm obstacle rule: during final approach the rack itself is closer than
15 cm. Only check a corridor of the image (or compare against the expected depth),
and switch the check off once the gripper is within about 10 cm of the tab.
