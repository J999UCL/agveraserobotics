#!/bin/zsh
# Resolve the project location so this also works when opened outside its directory.
set -eu
SIM_DIR="${0:A:h}"
SIM_PYTHON="$SIM_DIR/../.venv-mujoco/bin/python"
# uv's relocatable Python needs its shared library on the macOS loader path.
SIM_PYTHON_LIB="$($SIM_PYTHON -c 'import sys; print(sys.base_prefix + "/lib")')"
export DYLD_FALLBACK_LIBRARY_PATH="$SIM_PYTHON_LIB:${DYLD_FALLBACK_LIBRARY_PATH:-/usr/local/lib:/usr/lib}"
exec "$SIM_DIR/../.venv-mujoco/bin/mjpython" "$SIM_DIR/run.py" "$@"
