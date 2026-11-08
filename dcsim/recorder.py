"""MP4 recorder: follow camera full frame + wrist camera picture-in-picture + status overlay.

    rec = Recorder(sim, "outputs/demo.mp4", speed=4.0)
    rec.caption("Driving to R1-05-U20")
    ...                      # run skills; frames are captured from the sim's step hook
    rec.close()
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .sim import DataCenterSim


def _font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


class Recorder:
    def __init__(self, sim: DataCenterSim, path: str, fps: int = 25, speed: float = 1.0,
                 size: tuple[int, int] = (1280, 720), main_camera: str = "follow") -> None:
        import imageio.v2 as imageio

        self.sim = sim
        self.size = size
        self.main_camera = main_camera
        self.every = max(1, int(round(speed / (fps * sim.cfg.control_dt))))
        self.writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=None)
        self.text = ""
        self.frames = 0
        self._tick = 0
        self.f_big, self.f_small = _font(30), _font(19)
        sim.step_hooks.append(self._hook)

    def caption(self, text: str) -> None:
        self.text = text

    def _hook(self) -> None:
        self._tick += 1
        if self._tick % self.every == 0:
            self.writer.append_data(self.frame())
            self.frames += 1

    def frame(self) -> np.ndarray:
        s = self.sim
        w, h = self.size
        main = Image.fromarray(s.render(self.main_camera, w, h))
        wrist = Image.fromarray(s.render("wrist")).resize((384, 288))
        d = ImageDraw.Draw(main, "RGBA")
        # wrist camera inset (top right)
        x0, y0 = w - 384 - 16, 16
        d.rectangle([x0 - 3, y0 - 3, x0 + 384 + 2, y0 + 288 + 2], fill=(10, 10, 12, 255))
        main.paste(wrist, (x0, y0))
        d.rectangle([x0, y0 + 288 - 26, x0 + 150, y0 + 288], fill=(0, 0, 0, 160))
        d.text((x0 + 8, y0 + 288 - 24), "wrist camera", font=self.f_small, fill=(255, 255, 255, 255))
        # caption banner (top left)
        if self.text:
            tw = d.textlength(self.text, font=self.f_big)
            d.rectangle([16, 16, 16 + tw + 28, 64], fill=(8, 60, 110, 215))
            d.text((30, 22), self.text, font=self.f_big, fill=(255, 255, 255, 255))
        # status strip (bottom)
        st = s.status()
        open_wo = sum(1 for wo in s.work_orders if wo["status"] != "resolved")
        parts = [f"t {st['time']:6.1f}s", f"at {st['at_server'] or 'aisle/dock'}", f"lift {st['base']['lift_m']:.2f} m",
                 f"holding {st['holding'] or '-'}", f"pull force {st['pull_force_n']:5.1f} N",
                 f"open work orders {open_wo}"]
        if any(e["event"] == "obstacle_stop" for e in s.nav_log[-1:]):
            parts.append("OBSTACLE: holding")
        line = "   |   ".join(parts)
        d.rectangle([0, h - 40, w, h], fill=(0, 0, 0, 170))
        d.text((16, h - 32), line, font=self.f_small, fill=(230, 235, 240, 255))
        return np.asarray(main)

    def close(self) -> None:
        if self._hook in self.sim.step_hooks:
            self.sim.step_hooks.remove(self._hook)
        self.writer.close()
