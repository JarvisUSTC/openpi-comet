from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmark.core.io import read_json


@dataclass(slots=True)
class OpenPIServerLaunch:
    backend: str
    openpi_root: Path
    serve_script: Path
    task_mapping_path: Path
    task_name: str
    task_index: int | None
    task_prompt: str | None
    port: int
    policy_config: str
    checkpoint_dir: Path
    python_bin: str
    xla_python_client_preallocate: str | None = None
    xla_python_client_mem_fraction: str | None = None
    extra_server_args: list[str] = field(default_factory=list)
    passthrough_args: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class BackendPreset:
    default_openpi_root: str | None
    default_policy_config: str | None
    policy_config_hint: str
    checkpoint_hint: str
    extra_server_args: tuple[str, ...] = ()


def resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def resolve_path_from_base(path: str | Path, base: str | Path) -> Path:
    raw = Path(path).expanduser()
    base_path = resolve_path(base)
    return raw.resolve() if raw.is_absolute() else (base_path / raw).resolve()


def available_backends() -> list[str]:
    return ["baseline", "comet", "stop-head", "memory", "memory-git", "custom"]


def normalize_backend(raw: str) -> str:
    value = raw.strip().lower()
    alias_map = {
        "1": "baseline",
        "baseline": "baseline",
        "2": "comet",
        "comet": "comet",
        "3": "stop-head",
        "stop-head": "stop-head",
        "comet-stop-head": "stop-head",
        "4": "memory",
        "memory": "memory",
        "5": "memory-git",
        "memory-git": "memory-git",
        "6": "custom",
        "custom": "custom",
    }
    if value not in alias_map:
        raise ValueError(f"未知 BACKEND: {raw}")
    return alias_map[value]


def get_backend_preset(
    backend: str,
    *,
    default_home_root: Path,
    fine_grained_level: str,
    ruike_checkpoint_root: Path,
) -> BackendPreset:
    checkpoint_root_text = str(ruike_checkpoint_root)
    memory_args = (
        "--fine-grained-level",
        fine_grained_level,
        "--prompt-from-obs",
        "--control-mode",
        "temporal_ensemble",
        "--max-len",
        "32",
        "--action-horizon",
        "5",
        "--temporal-ensemble-max",
        "3",
        "--video-memory-frames",
        "1",
        "--video-memory-stride",
        "1",
        "--wm-in-prompt",
    )
    presets = {
        "baseline": BackendPreset(
            default_openpi_root=str(default_home_root / "openpi-comet-baseline"),
            default_policy_config="pi05_b1k-sampled_single_skill-full",
            policy_config_hint="baseline 默认是 single-skill 配置；如果训练的是别的 config，也可以直接覆盖 POLICY_CONFIG。",
            checkpoint_hint="baseline 会自动列出 OPENPI_ROOT/checkpoints 下的候选目录。",
        ),
        "comet": BackendPreset(
            default_openpi_root=str(default_home_root / "openpi-comet"),
            default_policy_config=None,
            policy_config_hint="comet 没有写死默认 config；请输入你训练时对应的 config 名。",
            checkpoint_hint=f"comet 会优先从 repo 内和 {checkpoint_root_text} 自动列出候选 checkpoint。",
        ),
        "stop-head": BackendPreset(
            default_openpi_root=str(default_home_root / "openpi-comet-stop-head"),
            default_policy_config="pi05_b1k-sampled_skill_group-stop-full-pretrained",
            policy_config_hint="stop-head 默认使用 skill-group stop 配置；可按需改成 prompt-aug 或 single-skill stop 配置。",
            checkpoint_hint=f"stop-head 会优先从 repo 内和 {checkpoint_root_text} 自动列出候选 checkpoint。",
        ),
        "memory": BackendPreset(
            default_openpi_root=str(default_home_root / "openpi-memory"),
            default_policy_config="pi05_b1k-all_skills_mem_K6",
            policy_config_hint="memory 默认使用 pi05_b1k-all_skills_mem_K6。",
            checkpoint_hint=f"memory 会优先从 repo 内和 {checkpoint_root_text} 自动列出候选 checkpoint。",
            extra_server_args=memory_args,
        ),
        "memory-git": BackendPreset(
            default_openpi_root=str(default_home_root / "openpi-memory-git"),
            default_policy_config="pi05_b1k-all_skills_mem_K6",
            policy_config_hint="memory-git 默认使用 pi05_b1k-all_skills_mem_K6。",
            checkpoint_hint=f"memory-git 会优先从 repo 内和 {checkpoint_root_text} 自动列出候选 checkpoint。",
            extra_server_args=memory_args,
        ),
        "custom": BackendPreset(
            default_openpi_root=None,
            default_policy_config=None,
            policy_config_hint="custom 模式下请填写该 repo 自己的训练 config 名。",
            checkpoint_hint="custom 模式下请填写该 repo 的 checkpoint 目录。",
        ),
    }
    return presets[backend]


def gather_policy_config_candidates(openpi_root: Path) -> list[str]:
    config_py = openpi_root / "src" / "openpi" / "training" / "config.py"
    if not config_py.is_file():
        return []
    text = config_py.read_text()
    names = re.findall(r'^\s*name="([^"]+)"\s*,?\s*$', text, flags=re.MULTILINE)
    seen: set[str] = set()
    candidates: list[str] = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        candidates.append(name)
    return candidates


def default_policy_config_choice(candidates: list[str], desired: str | None) -> str | None:
    if desired:
        for candidate in candidates:
            if candidate == desired:
                return candidate
    return candidates[0] if candidates else None


def configure_checkpoint_search_roots(
    *,
    openpi_root: Path,
    backend: str,
    ruike_checkpoint_root: Path | None,
) -> list[Path]:
    candidates: list[Path] = []

    def _add(path: Path) -> None:
        resolved = path.expanduser().resolve()
        if resolved.is_dir() and resolved not in candidates:
            candidates.append(resolved)

    _add(openpi_root / "checkpoints")
    _add(openpi_root / "outputs" / "checkpoints")
    if backend == "baseline":
        _add(openpi_root)
    elif backend in {"comet", "stop-head", "memory", "memory-git"} and ruike_checkpoint_root is not None:
        _add(ruike_checkpoint_root)
    return candidates


def gather_checkpoint_candidates(policy_config: str | None, search_roots: list[Path]) -> list[Path]:
    desired = (policy_config or "").strip().lower()
    seen: set[Path] = set()
    candidates: list[tuple[int, float, Path]] = []

    for root in search_roots:
        if not root.is_dir():
            continue
        for meta_path in root.rglob("_CHECKPOINT_METADATA"):
            checkpoint_dir = meta_path.parent.resolve()
            if checkpoint_dir in seen:
                continue
            seen.add(checkpoint_dir)
            try:
                mtime = checkpoint_dir.stat().st_mtime
            except OSError:
                mtime = 0.0
            score = 1000 if desired and desired in str(checkpoint_dir).lower() else 0
            candidates.append((score, mtime, checkpoint_dir))

    candidates.sort(key=lambda item: (-item[0], -item[1], str(item[2])))
    return [path for _, _, path in candidates]


def resolve_checkpoint_input(
    raw: str,
    *,
    openpi_root: Path,
    search_roots: list[Path],
    candidates: list[Path],
) -> Path | None:
    if not raw.strip():
        return None

    raw_path = Path(raw).expanduser()
    direct_candidates: list[Path] = []
    if raw_path.is_absolute():
        direct_candidates.append(raw_path.resolve())
    else:
        direct_candidates.append((openpi_root / raw_path).resolve())
        for root in search_roots:
            direct_candidates.append((root / raw_path).resolve())

    for candidate in direct_candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate

    raw_text = raw.strip().rstrip("/")
    raw_name = Path(raw_text).name
    matches = [
        candidate
        for candidate in candidates
        if str(candidate).endswith(raw_text) or candidate.name == raw_name
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def load_task_mapping(task_mapping_path: Path) -> dict[str, dict[str, Any]]:
    data = read_json(task_mapping_path)
    return {str(task_name): dict(meta) for task_name, meta in data.items()}


def task_menu_items(task_mapping: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for task_name, meta in task_mapping.items():
        items.append(
            {
                "task_name": task_name,
                "task_index": int(meta["task_index"]) if meta.get("task_index") is not None else None,
                "task_prompt": str(meta.get("task", "")),
            }
        )
    items.sort(key=lambda item: (item["task_index"] is None, item["task_index"], item["task_name"]))
    return items


def validate_openpi_root(openpi_root: Path) -> tuple[Path, Path]:
    resolved_root = openpi_root.expanduser().resolve()
    if not resolved_root.is_dir():
        raise FileNotFoundError(f"OPENPI_ROOT 不存在: {resolved_root}")

    serve_script = resolved_root / "scripts" / "serve_b1k.py"
    task_mapping_path = resolved_root / "scripts" / "task_mapping.json"
    if not serve_script.is_file():
        raise FileNotFoundError(f"未找到 serve_b1k.py: {serve_script}")
    if not task_mapping_path.is_file():
        raise FileNotFoundError(f"未找到 task_mapping.json: {task_mapping_path}")
    return serve_script, task_mapping_path


def build_server_command(launch: OpenPIServerLaunch) -> list[str]:
    return [
        launch.python_bin,
        str(launch.serve_script),
        "--task-name",
        launch.task_name,
        "--port",
        str(launch.port),
        *launch.extra_server_args,
        *launch.passthrough_args,
        "policy:checkpoint",
        "--policy.config",
        launch.policy_config,
        "--policy.dir",
        str(launch.checkpoint_dir),
    ]


def build_server_env(
    launch: OpenPIServerLaunch,
    *,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    env = dict(base_env or os.environ)
    pythonpath_entries = [str(launch.openpi_root), str(launch.openpi_root / "src")]
    openpi_client_src = launch.openpi_root / "packages" / "openpi-client" / "src"
    if openpi_client_src.is_dir():
        pythonpath_entries.append(str(openpi_client_src))
    if env.get("PYTHONPATH"):
        pythonpath_entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pythonpath_entries)
    if launch.xla_python_client_preallocate:
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = launch.xla_python_client_preallocate
    if launch.xla_python_client_mem_fraction:
        env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = launch.xla_python_client_mem_fraction
    return env
