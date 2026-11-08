#!/bin/zsh
set -eu
SIM_DIR="${0:A:h}"
if [[ -f "$1/model.xml" ]]; then
  SIM_ENV="$SIM_DIR/../.runtime/libero-replay-env"
  if [[ ! -x "$SIM_ENV/bin/python" ]]; then
    uv venv --python 3.11 "$SIM_ENV"
    uv pip install --python "$SIM_ENV/bin/python" 'mujoco==3.2.3' 'numpy==1.26.4'
  fi
  if ! MUJOCO_GL=disable "$SIM_ENV/bin/python" -c 'import mujoco; assert mujoco.__version__ == "3.2.3"' 2>/dev/null; then
    uv pip install --python "$SIM_ENV/bin/python" 'mujoco==3.2.3' 'numpy==1.26.4'
  fi
  SIM_PYTHON_LIB="$($SIM_ENV/bin/python -c 'import sys; print(sys.base_prefix + "/lib")')"
  export DYLD_FALLBACK_LIBRARY_PATH="$SIM_PYTHON_LIB:${DYLD_FALLBACK_LIBRARY_PATH:-/usr/local/lib:/usr/lib}"
  exec "$SIM_ENV/bin/mjpython" "$SIM_DIR/replay_libero.py" "$@"
fi
SIM_PYTHON="$SIM_DIR/../.venv-mujoco/bin/python"
SIM_PYTHON_LIB="$($SIM_PYTHON -c 'import sys; print(sys.base_prefix + "/lib")')"
export DYLD_FALLBACK_LIBRARY_PATH="$SIM_PYTHON_LIB:${DYLD_FALLBACK_LIBRARY_PATH:-/usr/local/lib:/usr/lib}"
exec "$SIM_DIR/../.venv-mujoco/bin/mjpython" "$SIM_DIR/replay.py" "$@"
