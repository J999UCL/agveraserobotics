#!/bin/zsh
set -eu
SIM_DIR="${0:A:h}"
uv venv --python 3.11 --allow-existing "$SIM_DIR/../.venv-mujoco"
uv pip install --python "$SIM_DIR/../.venv-mujoco/bin/python" -r "$SIM_DIR/requirements.txt"
"$SIM_DIR/../.venv-mujoco/bin/python" "$SIM_DIR/build_scene.py"
