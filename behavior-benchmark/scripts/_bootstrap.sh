#!/usr/bin/env bash

_bb_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_bb_project_root="$(cd "${_bb_script_dir}/.." && pwd)"

bb_project_root() {
  printf '%s\n' "${_bb_project_root}"
}

bb_setup_pythonpath() {
  export PYTHONPATH="${_bb_project_root}${PYTHONPATH:+:${PYTHONPATH}}"
}

bb_load_config_exports() {
  local bootstrap_python="${PYTHON_BOOTSTRAP_BIN:-${PYTHON_BIN:-python}}"
  bb_setup_pythonpath
  local exports
  exports="$("${bootstrap_python}" - <<'PY'
from benchmark.config.settings import format_env_exports
print(format_env_exports())
PY
)"
  eval "${exports}"
}

bb_activate_conda() {
  local conda_env="${CONDA_ENV:-}"
  if [[ -z "${conda_env}" ]]; then
    return 0
  fi

  local conda_bin="${CONDA_BIN:-$HOME/anaconda3/bin/conda}"
  if [[ ! -x "${conda_bin}" ]]; then
    echo "CONDA_BIN 不存在或不可执行: ${conda_bin}" >&2
    return 2
  fi

  set +u
  eval "$("${conda_bin}" shell.bash hook)"
  conda activate "${conda_env}"
  set -u
}
