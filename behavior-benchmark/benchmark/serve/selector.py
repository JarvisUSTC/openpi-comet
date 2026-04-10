from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from benchmark.config.settings import get_settings
from benchmark.core.schemas import ServerState

from .backends.openpi import (
    OpenPIServerLaunch,
    available_backends,
    build_server_command,
    build_server_env,
    configure_checkpoint_search_roots,
    default_policy_config_choice,
    gather_checkpoint_candidates,
    gather_policy_config_candidates,
    get_backend_preset,
    load_task_mapping,
    normalize_backend,
    resolve_checkpoint_input,
    resolve_path,
    resolve_path_from_base,
    task_menu_items,
    validate_openpi_root,
)
from .state import save_server_state


def _tilde(path: str | Path) -> str:
    text = str(path)
    home = str(Path.home())
    return text.replace(home, "~", 1) if text.startswith(home) else text


def _prompt_required_value(label: str, hint: str, default_value: str | None = None) -> str:
    while True:
        if hint:
            print(hint)
        if default_value:
            raw = input(f"{label} [默认={default_value}]: ").strip()
            value = raw or default_value
        else:
            value = input(f"{label}: ").strip()
        lowered = value.lower()
        if lowered in {"q", "quit", "exit"}:
            raise SystemExit(0)
        if value:
            return value
        print(f"{label} 不能为空。")


def _prompt_backend() -> str:
    while True:
        print("请选择后端 repo：")
        for index, backend in enumerate(available_backends(), start=1):
            print(f"  [{index}] {backend}")
        print("输入菜单编号 / backend 名称；输入 q 退出。")
        raw = input("> ").strip()
        if raw.lower() in {"q", "quit", "exit"}:
            raise SystemExit(0)
        try:
            return normalize_backend(raw)
        except ValueError:
            print("无效输入，请重试。")


def _prompt_policy_config(
    candidates: list[str],
    *,
    hint: str,
    default_choice: str | None,
) -> str:
    if not candidates:
        return _prompt_required_value("输入 POLICY_CONFIG", hint, default_choice)

    print("可选 POLICY_CONFIG：")
    for index, candidate in enumerate(candidates, start=1):
        print(f"  [{index}] {candidate}")
    if hint:
        print(hint)
    print("输入菜单编号 / 直接 config 名；输入 q 退出。")

    while True:
        raw = input("> ").strip()
        if not raw and default_choice:
            raw = default_choice
        lowered = raw.lower()
        if lowered in {"q", "quit", "exit"}:
            raise SystemExit(0)
        if raw.isdigit():
            index = int(raw)
            if 1 <= index <= len(candidates):
                return candidates[index - 1]
        if raw in candidates:
            return raw
        print("无效 POLICY_CONFIG，请从菜单里选，或输入完整 config 名。")


def _prompt_checkpoint_dir(
    candidates: list[Path],
    *,
    hint: str,
    default_choice: Path | None,
) -> str:
    if not candidates:
        return _prompt_required_value("输入 CHECKPOINT_DIR", hint, str(default_choice) if default_choice else None)

    print("可选 checkpoint：")
    for index, candidate in enumerate(candidates, start=1):
        print(f"  [{index}] {_tilde(candidate)}")
    if hint:
        print(hint)
    print("输入菜单编号 / 直接路径；输入 q 退出。")

    while True:
        raw = input("> ").strip()
        if not raw and default_choice is not None:
            raw = str(default_choice)
        lowered = raw.lower()
        if lowered in {"q", "quit", "exit"}:
            raise SystemExit(0)
        if raw.isdigit():
            index = int(raw)
            if 1 <= index <= len(candidates):
                return str(candidates[index - 1])
        if raw:
            return raw
        print("无效输入，请重试。")


def _prompt_task_name(task_mapping: dict[str, dict]) -> str:
    items = task_menu_items(task_mapping)
    while True:
        print("请选择真实的 BEHAVIOR task：")
        for index, item in enumerate(items, start=1):
            prompt_preview = str(item["task_prompt"]).strip().replace("\n", " ")
            if len(prompt_preview) > 100:
                prompt_preview = prompt_preview[:97] + "..."
            task_index = item["task_index"]
            if task_index is None:
                task_label = "task ????"
            else:
                task_label = f"task {int(task_index):04d}"
            print(f"  [{index}] {task_label}  {item['task_name']}  {prompt_preview}")
        print("输入菜单编号；输入 q 退出。")
        raw = input("> ").strip()
        lowered = raw.lower()
        if lowered in {"q", "quit", "exit"}:
            raise SystemExit(0)
        if raw.isdigit():
            index = int(raw)
            if 1 <= index <= len(items):
                return str(items[index - 1]["task_name"])
        print("无效输入，请重试。")


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    passthrough_args = list(argv or sys.argv[1:])
    env = os.environ

    backend_raw = env.get("BACKEND") or settings.serve.backend
    backend = normalize_backend(backend_raw) if backend_raw else _prompt_backend()

    default_home_root = resolve_path(env.get("DEFAULT_HOME_ROOT", str(Path.home() / "Jiawei")))
    fine_grained_level = env.get("FINE_GRAINED_LEVEL", "0")
    ruike_checkpoint_root = resolve_path(
        env.get("RUIKE_CHECKPOINT_ROOT", str(Path.home() / "ruike" / "task_checkpoint" / "openpi_comet"))
    )
    preset = get_backend_preset(
        backend,
        default_home_root=default_home_root,
        fine_grained_level=fine_grained_level,
        ruike_checkpoint_root=ruike_checkpoint_root,
    )

    openpi_root_raw = env.get("OPENPI_ROOT") or preset.default_openpi_root
    if not openpi_root_raw:
        openpi_root_raw = _prompt_required_value(
            "输入 OPENPI_ROOT",
            "custom 模式下请提供目标 repo 根目录。",
            None,
        )
    openpi_root = resolve_path(openpi_root_raw)
    serve_script, task_mapping_path = validate_openpi_root(openpi_root)

    policy_config_from_user = bool(env.get("POLICY_CONFIG"))
    policy_config = env.get("POLICY_CONFIG") or settings.serve.policy_config or preset.default_policy_config
    policy_config_candidates = gather_policy_config_candidates(openpi_root)
    if not policy_config_from_user and policy_config_candidates:
        default_choice = default_policy_config_choice(policy_config_candidates, policy_config)
        policy_config = _prompt_policy_config(
            policy_config_candidates,
            hint=preset.policy_config_hint,
            default_choice=default_choice,
        )
    elif not policy_config:
        policy_config = _prompt_required_value("输入 POLICY_CONFIG", preset.policy_config_hint, None)

    checkpoint_search_roots = configure_checkpoint_search_roots(
        openpi_root=openpi_root,
        backend=backend,
        ruike_checkpoint_root=ruike_checkpoint_root,
    )
    checkpoint_candidates = gather_checkpoint_candidates(policy_config, checkpoint_search_roots)

    checkpoint_dir_raw = env.get("CHECKPOINT_DIR") or settings.serve.checkpoint_dir
    if not checkpoint_dir_raw:
        checkpoint_dir_raw = _prompt_checkpoint_dir(
            checkpoint_candidates,
            hint=preset.checkpoint_hint,
            default_choice=checkpoint_candidates[0] if checkpoint_candidates else None,
        )
    resolved_checkpoint = resolve_checkpoint_input(
        checkpoint_dir_raw,
        openpi_root=openpi_root,
        search_roots=checkpoint_search_roots,
        candidates=checkpoint_candidates,
    )
    checkpoint_dir = (
        resolved_checkpoint
        if resolved_checkpoint is not None
        else resolve_path_from_base(checkpoint_dir_raw, openpi_root)
    )
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"CHECKPOINT_DIR 不存在: {checkpoint_dir}")

    task_mapping = load_task_mapping(task_mapping_path)
    task_name = env.get("TASK_NAME") or _prompt_task_name(task_mapping)
    task_meta = task_mapping.get(task_name, {})

    port = int(env.get("PORT") or settings.serve.port or 8000)
    python_bin = env.get("PYTHON_BIN") or settings.runtime.python_bin
    server_state_path = Path(env.get("SERVER_STATE_PATH", str(settings.paths.server_state_path))).expanduser().resolve()

    launch = OpenPIServerLaunch(
        backend=backend,
        openpi_root=openpi_root,
        serve_script=serve_script,
        task_mapping_path=task_mapping_path,
        task_name=task_name,
        task_index=int(task_meta["task_index"]) if task_meta.get("task_index") is not None else None,
        task_prompt=str(task_meta.get("task", "")) or None,
        port=port,
        policy_config=policy_config,
        checkpoint_dir=checkpoint_dir,
        python_bin=python_bin,
        xla_python_client_preallocate=env.get("XLA_PYTHON_CLIENT_PREALLOCATE")
        or settings.runtime.xla_python_client_preallocate,
        xla_python_client_mem_fraction=env.get("XLA_PYTHON_CLIENT_MEM_FRACTION")
        or settings.runtime.xla_python_client_mem_fraction,
        extra_server_args=list(preset.extra_server_args),
        passthrough_args=passthrough_args,
    )

    save_server_state(
        server_state_path,
        ServerState(
            task_name=launch.task_name,
            task_index=launch.task_index,
            task_prompt=launch.task_prompt,
            port=launch.port,
            checkpoint_dir=str(launch.checkpoint_dir),
            backend=launch.backend,
            repo_root=str(launch.openpi_root),
            policy_config=launch.policy_config,
            updated_at=int(time.time()),
        ),
    )

    command = build_server_command(launch)
    child_env = build_server_env(launch, base_env=dict(env))

    print(f"backend       : {launch.backend}")
    print(f"repo_root     : {launch.openpi_root}")
    print(f"task_name     : {launch.task_name}")
    print(f"port          : {launch.port}")
    print(f"policy_config : {launch.policy_config}")
    print(f"checkpoint    : {launch.checkpoint_dir}")
    print(f"xla_prealloc  : {launch.xla_python_client_preallocate}")
    print(f"xla_mem_frac  : {launch.xla_python_client_mem_fraction}")
    if launch.extra_server_args:
        print("extra_args    :" + "".join(f" {arg}" for arg in launch.extra_server_args))
    if launch.passthrough_args:
        print("passthrough   :" + "".join(f" {arg}" for arg in launch.passthrough_args))

    completed = subprocess.run(
        command,
        cwd=launch.openpi_root,
        env=child_env,
        check=False,
    )
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
