#!/usr/bin/env bash
# 将 uv 管理的 CPython 与 wheel 缓存固定到仓库所在 vepfs（与 REPO_ROOT 同盘），
# 避免多机任务里 worker 无法访问某台开发机上的 /root/.local/share/uv/...。
# 在其它 bash 脚本里于 cd 到仓库根目录之前或之后 source 本文件均可。
_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_repo_root="$(cd "${_script_dir}/.." && pwd)"
export UV_PYTHON_INSTALL_DIR="${_repo_root}/.uv/python-installs"
export UV_CACHE_DIR="${_repo_root}/.uv/cache"
mkdir -p "${UV_PYTHON_INSTALL_DIR}" "${UV_CACHE_DIR}"
unset _script_dir _repo_root
