"""Perception on the team's MuJoCo data centre: a faulty drive, then the same with a dropped cable.

    python -m perception.demo                       # rack 2, server slot 6, bay 5 faulty
    python -m perception.demo --rack 3 --slot 12 --bay 2

Saves annotated camera images and JSON to out/ and prints what the agent would receive.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from . import datacentre as dc


def run(m, d, rack, slot, cam_pos, target, name, out: Path):
    per = dc.make_perception(m, d)
    cap = dc.capture(m, d, cam_pos, target)
    tgt = dc.server_target(m, d, rack, slot)
    per.scan(cap, target_xyz=tgt)  # warm-up
    scan = per.scan(cap, target_xyz=tgt)
    print(f"\n=== {name}\n{scan.summary()}")
    print("agent view:", json.dumps([{k: v for k, v in b.items() if k != "reasons"} for b in dc.bay_views(scan)]))
    cv2.imwrite(str(out / f"datacentre_{name}.png"), cv2.cvtColor(dc.HotSwapPerception.annotate(cap, scan), cv2.COLOR_RGB2BGR))
    (out / f"datacentre_{name}.json").write_text(json.dumps(scan.to_dict(), indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rack", type=int, default=2)
    ap.add_argument("--slot", type=int, default=6, help="2U server slot, 0 = bottom")
    ap.add_argument("--bay", type=int, default=5, help="faulty drive, 0-7, left to right, top to bottom")
    args = ap.parse_args()
    out = Path("out")
    out.mkdir(exist_ok=True)

    m, d = dc.load()
    dc.set_led(m, args.rack, args.slot, args.bay, "amber")
    pos, tgt = dc.scan_pose(m, d, args.rack, args.slot)
    run(m, d, args.rack, args.slot, pos, tgt, "fault", out)

    outward = (pos - tgt) / np.linalg.norm(pos - tgt)
    cable = (tgt + 0.15 * outward + [0.04, 0, -0.16], tgt + 0.15 * outward + [0.06, 0, 0.16])
    m2, d2 = dc.load(cable=cable)
    dc.set_led(m2, args.rack, args.slot, args.bay, "amber")
    run(m2, d2, args.rack, args.slot, pos, tgt, "fault_and_cable", out)
    print(f"\nimages + JSON in {out.resolve()}")


if __name__ == "__main__":
    main()
