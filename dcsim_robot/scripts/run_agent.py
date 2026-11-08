"""Run the team's agent on the robot. Start the simulator first (python scripts/run_server.py --viewer).

    python scripts/run_agent.py --only doa_spare
    python scripts/run_agent.py --model gemini-4-argon          # default: gemini-3.5-flash-lite

Reads GEMINI_API_KEY from .env. Other options: see integration/argon/run_dcsim_argon.py.
The agent (robot/sandbox) and vision/ come from the team repo: either this folder is dcsim_robot/
inside it, or a copy of those two folders sits in team/.
"""

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def has(base: Path) -> bool:
    return (base / "vision" / "tray.py").is_file() and (base / "robot" / "sandbox" / "agent.py").is_file()


team = next((b for b in (ROOT / "team", ROOT.parent) if has(b)), None)
if team is None:
    sys.exit("can't find vision/ and robot/sandbox/: put a copy of them in team/ (see this file's docstring)")

env = ROOT / ".env"
if env.is_file():
    for line in env.read_text(encoding="utf-8-sig").splitlines():
        k, sep, v = line.strip().partition("=")
        if sep and k == "GEMINI_API_KEY":
            os.environ.setdefault(k, v.strip().strip("\"'"))

os.environ.setdefault("DCSIM", "localhost:5555")
os.environ["DCSIM_REPO"] = str(ROOT)
os.environ["VISION_REPO"] = str(team)
sys.path.insert(0, str(team / "robot" / "sandbox"))

args = sys.argv[1:]
if "--model" not in args and "--brain" not in args:
    args += ["--model", "gemini-3.5-flash-lite"]
script = ROOT / "integration" / "argon" / "run_dcsim_argon.py"
sys.argv = [str(script)] + args
runpy.run_path(str(script), run_name="__main__")
