"""Run perception on frames saved from Isaac Sim, so we can test without running Isaac.

In the Isaac Sim script (Emanak's side), save a frame with:

    f = scene.get_wrist_frames()
    np.savez("frame_001.npz", rgb=f.rgb, depth=f.depth, K=f.K, cam_pos=f.cam_pos, cam_rot=f.cam_rot,
             # optional: what pixel_to_base returns for 5 pixels, so the camera convention can be checked
             check_uv=np.array([[320, 240], [100, 100], [540, 100], [100, 380], [540, 380]]),
             check_xyz=np.array([pixel_to_base(u, v, f.depth[v, u], f) for u, v in [[320, 240], [100, 100], [540, 100], [100, 380], [540, 380]]]))

Then here:
    python -m perception.check_frame frame_001.npz [--rows 3 --cols 4] [--axes usd|opencv]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from perception.adapter import capture_from_wrist_frames
from perception.hotswap import HotSwapPerception


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--axes", default="auto", choices=["auto", "usd", "opencv"])
    args = ap.parse_args()
    per = HotSwapPerception(rows=args.rows, cols=args.cols)
    out = Path("out")
    out.mkdir(exist_ok=True)
    for path in args.npz:
        z = np.load(path)
        frames = SimpleNamespace(rgb=z["rgb"], depth=z["depth"], K=z["K"], cam_pos=z["cam_pos"], cam_rot=z["cam_rot"])
        p2b = None
        if "check_uv" in z and "check_xyz" in z:
            table = {(int(u), int(v)): xyz for (u, v), xyz in zip(z["check_uv"], z["check_xyz"])}
            # only the saved pixels are answerable; make depth invalid elsewhere for the convention check
            masked = np.full_like(frames.depth, np.inf)
            for (u, v) in table:
                masked[v, u] = frames.depth[v, u]
            check = SimpleNamespace(**{**vars(frames), "depth": masked})
            from perception.adapter import detect_axes
            axes = detect_axes(check, lambda u, v, d, fr: table[(int(u), int(v))])
            print(f"{path}: camera convention detected as '{axes}'")
        else:
            axes = "usd" if args.axes == "auto" else args.axes
            print(f"{path}: no check points saved, assuming '{axes}' camera axes")
        cap = capture_from_wrist_frames(frames, cam_axes=axes)
        per.scan(cap)
        scan = per.scan(cap)
        print(scan.summary(), "\n")
        img = HotSwapPerception.annotate(cap, scan)
        dst = out / (Path(path).stem + "_perception.png")
        cv2.imwrite(str(dst), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"saved {dst}")


if __name__ == "__main__":
    main()
