#!/bin/zsh
set -eu
SIM_DIR="${0:A:h}"
export MODAL_PROFILE=j999ucl
SIM_CLOUD_PYTHON="$SIM_DIR/../.venv/bin/python"
if [[ ! -x "$SIM_CLOUD_PYTHON" ]]; then
  SIM_CLOUD_ENV="$SIM_DIR/../.runtime/mujoco-cloud-env"
  uv venv --python 3.11 --allow-existing "$SIM_CLOUD_ENV"
  uv pip install --python "$SIM_CLOUD_ENV/bin/python" 'modal==1.5.3'
  SIM_CLOUD_PYTHON="$SIM_CLOUD_ENV/bin/python"
fi
exec "$SIM_CLOUD_PYTHON" "$SIM_DIR/cloud.py" "$@"
