from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

from benchmark.config.settings import bootstrap_environment, get_settings

def simple_robo_agent_root() -> Path:
    bootstrap_environment()
    return get_settings().paths.simple_robo_agent_root


def legacy_path(relative_path: str) -> Path:
    path = simple_robo_agent_root() / relative_path
    if not path.exists():
        raise FileNotFoundError(f"未找到旧仓库文件: {path}")
    return path


@lru_cache(maxsize=None)
def load_legacy_module(module_name: str, relative_path: str):
    path = legacy_path(relative_path)
    if str(simple_robo_agent_root()) not in sys.path:
        sys.path.insert(0, str(simple_robo_agent_root()))
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载旧模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
