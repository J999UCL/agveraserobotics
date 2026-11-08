"""Watch the arm layer live in the MuJoCo viewer.

  .venv-mujoco/Scripts/python robot/arm/demo.py        (Windows)
  .venv-mujoco/bin/mjpython robot/arm/demo.py          (macOS)

For two servers: line the base up, grip the right handle, pull the tray 70 cm open, let go,
then grip it again and push it closed. Results print in the terminal and on screen.
Runs in real time; close the window to stop.
"""

from __future__ import annotations

import os
import sys
import time

import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.dirname(__file__))
from arm import Arm  # noqa: E402
from base import Base, load  # noqa: E402
from skills import TrayService  # noqa: E402

SERVERS = [(2, 21), (4, 9)]


def main():
    m, d = load()
    with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as v:
        with v.lock():
            v.cam.lookat[:] = [0.4, -0.6, 0.9]
            v.cam.distance, v.cam.azimuth, v.cam.elevation = 3.2, 62, -15
            v.opt.geomgroup[3] = 0
        start = time.monotonic()
        status = [""]

        def step():
            mujoco.mj_step(m, d)
            if int(round(d.time / m.opt.timestep)) % 8 == 0:
                if not v.is_running():
                    raise SystemExit
                ahead = d.time - (time.monotonic() - start)
                if ahead > 0:
                    time.sleep(ahead)          # real time
                v.sync()

        def say(msg):
            status[0] = msg
            print(f"t={d.time:5.1f}s  {msg}", flush=True)
            try:
                v.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT, f"t={d.time:.1f}s", msg))
            except AttributeError:
                pass

        base = Base(m, d, step=step)
        trays = TrayService(base, Arm(base))
        for rack, u in SERVERS:
            say(f"opening R{rack:02} U{u} ...")
            say(trays.open_tray(rack, u))
            say(f"closing R{rack:02} U{u} ...")
            say(trays.close_tray(rack, u))
        say("done - close the window to exit")
        while v.is_running():
            time.sleep(0.05)
            v.sync()
        print(f"finished at t={d.time:.1f}s sim, {time.monotonic() - start:.1f}s wall")


if __name__ == "__main__":
    main()
