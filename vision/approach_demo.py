"""Approach-path monitor on the MuJoCo data centre: a cable drops into the gripper's path mid-approach.

    python -m vision.approach_demo        # frames -> out/approach_*.png

Wrist camera rides on the gripper while it moves towards a drive latch. Each frame is checked;
the monitor says continue / slow / stop, so the arm stops when the cable appears and resumes once
the path is clear again.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import datacentre as dc
from .corridor import ApproachMonitor, annotate_path


def main() -> None:
    out = Path("out")
    out.mkdir(exist_ok=True)
    m, d = dc.load(cable_mocap=True)
    rack, slot, bay = 2, 6, 2
    dc.set_led(m, rack, slot, bay, "amber")
    goal_w = dc._tray_point(d, rack, slot, dc.grab_local(bay))
    out_w = d.body(dc.ident(rack, slot)).xmat.reshape(3, 3) @ np.array([0, -1.0, 0])
    mon = ApproachMonitor(dc.to_base(d, goal_w), confirm=2)

    dist, k = 0.32, 0
    events = {1: "drop", 4: "clear"}           # frame number -> what happens in the world
    while dist > 0.06 and k < 30:
        if events.get(k) == "drop":
            dc.place_cable(m, d, rack, slot, side_x=dc.grab_local(bay)[0] + 0.01)
        elif events.get(k) == "clear":
            dc.park_cable(m, d)
        tool_w = goal_w + dist * out_w
        cap = dc.capture(m, d, tool_w + np.array([0, 0, 0.03]), goal_w)
        action, chk = mon.update(cap, dc.to_base(d, tool_w))
        img = annotate_path(cap.rgb, chk)
        cv2.putText(img, f"frame {k}: gripper {dist * 100:.0f} cm from latch -> {action.upper()}", (10, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out / f"approach_{k:02d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"frame {k:2d}  {dist * 100:4.0f} cm  {action:8s} {chk.status:8s} {chk.latency_ms:5.1f} ms  {chk.reasons[0]}")
        if action == "continue":
            dist -= 0.04
        elif action == "slow":
            dist -= 0.015
        k += 1
    print(f"\nframes in {out.resolve()}")


if __name__ == "__main__":
    main()
