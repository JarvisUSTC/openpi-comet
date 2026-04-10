from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from benchmark.config.settings import get_settings
from benchmark.core.inventory import (
    filter_task_infos_by_category,
    get_episode_infos_for_category,
    get_global_category_infos,
    get_task_infos,
)
from benchmark.core.io import read_json
from benchmark.core.paths import build_judge_output_path, skill_dir_name, task_dir_name
from benchmark.eval.prompt_sources import load_augmented_prompts
from benchmark.serve.state import load_server_state


def _objects_preview(object_ids: list[str], limit: int = 3) -> str:
    if not object_ids:
        return "-"
    shown = object_ids[:limit]
    suffix = "" if len(object_ids) <= limit else "..."
    return ", ".join(shown) + suffix


def _prompt_task_choice(task_infos: list[dict[str, Any]]) -> dict[str, Any] | None:
    raw = input("> ").strip()
    if not raw:
        return None
    lowered = raw.lower()
    if lowered in {"q", "quit", "exit"}:
        raise SystemExit(0)
    if lowered in {"b", "back"}:
        return {"__action__": "back"}
    if lowered in {"r", "refresh"}:
        return {"__action__": "refresh"}

    if raw.isdigit():
        idx = int(raw)
        if 1 <= idx <= len(task_infos):
            return task_infos[idx - 1]

    print("无效的 task 输入，请重试。")
    return None


def _prompt_episode_choice(episode_infos: list[dict[str, Any]]) -> dict[str, Any] | None:
    raw = input("> ").strip()
    if not raw:
        return None
    lowered = raw.lower()
    if lowered in {"q", "quit", "exit"}:
        raise SystemExit(0)
    if lowered in {"b", "back"}:
        return {"__action__": "back"}
    if lowered in {"r", "refresh"}:
        return {"__action__": "refresh"}

    normalized = raw
    if normalized.endswith(".json"):
        normalized = normalized[:-5]
    if normalized.isdigit():
        padded = normalized.zfill(8)
        normalized = f"episode_{padded}"

    for episode_info in episode_infos:
        if raw == episode_info["episode_name"] or normalized == episode_info["episode_name"]:
            return episode_info
        if raw == str(episode_info["instance_id"]):
            return episode_info

    if raw.isdigit():
        idx = int(raw)
        if 1 <= idx <= len(episode_infos):
            return episode_infos[idx - 1]

    print("无效的 episode 输入，请重试。")
    return None


def _prompt_category_choice(category_infos: list[dict[str, Any]]) -> dict[str, Any] | None:
    raw = input("> ").strip()
    if not raw:
        return None
    lowered = raw.lower()
    if lowered in {"q", "quit", "exit"}:
        raise SystemExit(0)
    if lowered in {"b", "back"}:
        return {"__action__": "back"}
    if lowered in {"r", "refresh"}:
        return {"__action__": "refresh"}

    for category_info in category_infos:
        if lowered == str(category_info["category_name"]).lower():
            return category_info

    if raw.isdigit():
        idx = int(raw)
        if 1 <= idx <= len(category_infos):
            return category_infos[idx - 1]

    print("无效的 skill 类输入，请重试。")
    return None


def _resolve_skill_token(token: str, skill_infos: list[dict[str, Any]]) -> dict[str, Any] | None:
    raw = token.strip()
    if not raw:
        return None

    normalized = raw
    if normalized.lower().startswith("skill_"):
        normalized = normalized.split("_", 1)[1]

    for skill_info in skill_infos:
        if raw == skill_info["skill_dir_name"]:
            return skill_info
        if normalized.isdigit() and int(normalized) == int(skill_info["skill_idx"]):
            return skill_info

    if raw.isdigit():
        idx = int(raw)
        if 1 <= idx <= len(skill_infos):
            return skill_infos[idx - 1]

    return None


def _prompt_skill_choices(skill_infos: list[dict[str, Any]]) -> list[dict[str, Any]] | dict[str, Any] | None:
    raw = input("> ").strip()
    if not raw:
        return None
    lowered = raw.lower()
    if lowered in {"q", "quit", "exit"}:
        raise SystemExit(0)
    if lowered in {"b", "back"}:
        return {"__action__": "back"}
    if lowered in {"r", "refresh"}:
        return {"__action__": "refresh"}

    tokens: list[str] = []
    for chunk in raw.replace("，", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        tokens.extend(part for part in chunk.split() if part)

    if not tokens:
        return None

    if len(tokens) == 1 and tokens[0].lower() == "all":
        return list(skill_infos)

    selected: list[dict[str, Any]] = []
    seen_skill_idxs: set[int] = set()

    def _append_skill(skill_info: dict[str, Any]) -> None:
        skill_idx = int(skill_info["skill_idx"])
        if skill_idx in seen_skill_idxs:
            return
        seen_skill_idxs.add(skill_idx)
        selected.append(skill_info)

    for token in tokens:
        if "-" in token and token.count("-") == 1:
            left, right = token.split("-", 1)
            if left.isdigit() and right.isdigit():
                start_idx = int(left)
                end_idx = int(right)
                if start_idx <= 0 or end_idx <= 0 or start_idx > end_idx:
                    print(f"无效的范围输入: {token}")
                    return None
                for menu_idx in range(start_idx, end_idx + 1):
                    if 1 <= menu_idx <= len(skill_infos):
                        _append_skill(skill_infos[menu_idx - 1])
                    else:
                        print(f"范围超出菜单范围: {token}")
                        return None
                continue

        skill_info = _resolve_skill_token(token, skill_infos)
        if skill_info is None:
            print(f"无效的 skill 输入: {token}")
            return None
        _append_skill(skill_info)

    if not selected:
        print("没有解析到任何 skill，请重试。")
        return None
    return selected


def _print_task_menu(
    task_infos: list[dict[str, Any]],
    *,
    selected_category_name: str | None = None,
    group_by: str | None = None,
) -> None:
    if selected_category_name is None:
        print("\n可评测 task:")
    else:
        group_label = "skill_description" if group_by == "skill-description" else "skill_type"
        print(f"\n包含 `{selected_category_name}` 的可评测 task ({group_label}):")
    for idx, task_info in enumerate(task_infos, start=1):
        episode_count = int(task_info.get("matched_episode_count", task_info["episode_count"]))
        skill_count = int(task_info.get("matched_skill_count", task_info["skill_count"]))
        print(
            f"  [{idx}] task {int(task_info['task_index']):04d}  "
            f"{task_info['task_name']}  "
            f"(episodes={episode_count}, skills={skill_count})"
        )
    if selected_category_name is None:
        print("输入菜单编号；`r` 刷新，`q` 退出。")
    else:
        print("输入菜单编号；`b` 返回，`r` 刷新，`q` 退出。")


def _print_episode_menu(
    task_info: dict[str, Any],
    episode_infos: list[dict[str, Any]],
    *,
    selected_category_name: str | None = None,
    group_by: str | None = None,
) -> None:
    if selected_category_name is None:
        print(f"\nTask {int(task_info['task_index']):04d} {task_info['task_name']} 的可评测 episode:")
    else:
        group_label = "skill_description" if group_by == "skill-description" else "skill_type"
        print(
            f"\nTask {int(task_info['task_index']):04d} {task_info['task_name']} 中包含 "
            f"`{selected_category_name}` 的 episode ({group_label}):"
        )
    if task_info.get("task_prompt"):
        print(f"  prompt: {task_info['task_prompt']}")
    for idx, episode_info in enumerate(episode_infos, start=1):
        print(
            f"  [{idx}] {episode_info['episode_name']}  "
            f"(instance={episode_info['instance_id']}, skills={episode_info['skill_count']})"
        )
    print("输入 episode 名 / 8位数字 / instance_id / 菜单编号；`b` 返回，`r` 刷新，`q` 退出。")


def _print_category_menu(category_infos: list[dict[str, Any]], *, group_by: str, allow_back: bool = True) -> None:
    group_label = "skill_description" if group_by == "skill-description" else "skill_type"
    print(f"\n可选 {group_label} 类别:")
    for idx, category_info in enumerate(category_infos, start=1):
        skill_type_text = ", ".join(category_info["skill_types"]) if category_info["skill_types"] else "-"
        summary_parts = [f"count={category_info['count']}"]
        if category_info.get("task_count") is not None:
            summary_parts.append(f"tasks={category_info['task_count']}")
        if category_info.get("episode_count") is not None:
            summary_parts.append(f"episodes={category_info['episode_count']}")
        summary_parts.append(f"types={skill_type_text}")
        print(
            f"  [{idx}] {category_info['category_name']}  "
            f"({', '.join(summary_parts)})"
        )
    if allow_back:
        print("输入类别名或菜单编号；`b` 返回，`r` 刷新，`q` 退出。")
    else:
        print("输入类别名或菜单编号；`r` 刷新，`q` 退出。")


def _print_skill_menu(category_info: dict[str, Any]) -> None:
    print(f"\n类别 `{category_info['category_name']}` 下的 skill:")
    for idx, skill_info in enumerate(category_info["skill_infos"], start=1):
        skill_types = ", ".join(skill_info["skill_type"]) if skill_info["skill_type"] else "-"
        print(
            f"  [{idx}] skill_{int(skill_info['skill_idx']):02d}  "
            f"(types={skill_types}, "
            f"objects={_objects_preview(skill_info['object_ids'])}, "
            f"frame={skill_info['frame_start']}->{skill_info['frame_end']})"
        )
    print("输入 skill_idx / skill_XX / 菜单编号。")
    print("批量可用：`1,3,5`、`1-4`、`skill_10,skill_30`、`all`；`b` 返回，`r` 刷新，`q` 退出。")


def _print_augmented_prompt_menu(
    *,
    task_info: dict[str, Any],
    episode_info: dict[str, Any],
    skill_info: dict[str, Any],
    prompt_choices: list[str],
) -> None:
    print(
        f"\n为 task {int(task_info['task_index']):04d} / {episode_info['episode_name']} / "
        f"skill_{int(skill_info['skill_idx']):02d} 选择 augmented prompt:"
    )
    for idx, prompt_text in enumerate(prompt_choices, start=1):
        print(f"  [{idx}] {prompt_text}")
    print(f"  [{len(prompt_choices) + 1}] 自定义输入")
    print("输入菜单编号；也可以输入 `c` 直接手动填写 prompt；`b` 返回，`r` 刷新，`q` 退出。")


def _prompt_augmented_prompt_choice(prompt_choices: list[str]) -> str | dict[str, Any] | None:
    raw = input("> ").strip()
    if not raw:
        return None
    lowered = raw.lower()
    if lowered in {"q", "quit", "exit"}:
        raise SystemExit(0)
    if lowered in {"b", "back"}:
        return {"__action__": "back"}
    if lowered in {"r", "refresh"}:
        return {"__action__": "refresh"}
    if lowered in {"c", "custom"}:
        return {"__action__": "custom"}
    if raw.isdigit():
        idx = int(raw)
        if 1 <= idx <= len(prompt_choices):
            return prompt_choices[idx - 1]
        if idx == len(prompt_choices) + 1:
            return {"__action__": "custom"}
    print("无效的 prompt 输入，请重试。")
    return None


def _prompt_custom_prompt() -> str | dict[str, Any] | None:
    while True:
        raw = input("请输入自定义 prompt：").strip()
        if not raw:
            print("prompt 不能为空，请重试。")
            continue
        lowered = raw.lower()
        if lowered in {"q", "quit", "exit"}:
            raise SystemExit(0)
        if lowered in {"b", "back"}:
            return {"__action__": "back"}
        if lowered in {"r", "refresh"}:
            return {"__action__": "refresh"}
        return raw


def _apply_prompt_prefix(prompt_text: str, prompt_prefix: str | None) -> str:
    text = str(prompt_text).strip()
    prefix = str(prompt_prefix or "").strip()
    if not text or not prefix:
        return text
    if text.lower().startswith(prefix.lower()):
        return text
    separator = "" if prefix.endswith((" ", "\t", "\n")) else " "
    return f"{prefix}{separator}{text}".strip()


def _print_prompt_prefix_menu(prompt_preview: str) -> None:
    print("\n这次运行的 prompt prefix：")
    print("  [1] 不加 prefix")
    print("  [2] 加 `Task:`")
    print("  [3] 自定义 prefix")
    print(f"示例 prompt: {prompt_preview}")
    print("输入菜单编号；`b` 返回，`r` 刷新，`q` 退出。")


def _prompt_custom_prompt_prefix() -> str | dict[str, Any] | None:
    while True:
        raw = input("请输入自定义 prefix（例如 `Task:`）：").strip()
        if not raw:
            print("prefix 不能为空，请重试。")
            continue
        lowered = raw.lower()
        if lowered in {"q", "quit", "exit"}:
            raise SystemExit(0)
        if lowered in {"b", "back"}:
            return {"__action__": "back"}
        if lowered in {"r", "refresh"}:
            return {"__action__": "refresh"}
        return raw


def _prompt_prompt_prefix(prompt_preview: str) -> str | None | dict[str, Any]:
    while True:
        _print_prompt_prefix_menu(prompt_preview)
        raw = input("> ").strip()
        if not raw:
            continue
        lowered = raw.lower()
        if lowered in {"q", "quit", "exit"}:
            raise SystemExit(0)
        if lowered in {"b", "back"}:
            return {"__action__": "back"}
        if lowered in {"r", "refresh"}:
            return {"__action__": "refresh"}
        if raw == "1":
            return None
        if raw == "2":
            return "Task:"
        if raw == "3":
            custom_prefix = _prompt_custom_prompt_prefix()
            if custom_prefix is None:
                continue
            if isinstance(custom_prefix, dict):
                if custom_prefix.get("__action__") == "back":
                    continue
                if custom_prefix.get("__action__") == "refresh":
                    continue
            return str(custom_prefix)
        print("无效的 prefix 输入，请重试。")


def _display_path(path: Path) -> str:
    text = str(path)
    home = str(Path.home())
    return text.replace(home, "~", 1) if text.startswith(home) else text


def _log_path_candidates(current_log_path: Path) -> list[Path]:
    parent = current_log_path.parent
    if not parent.is_dir():
        return []
    candidates = [path for path in parent.iterdir() if path.is_dir()]
    candidates.sort(
        key=lambda path: (
            0 if path.resolve() == current_log_path.resolve() else 1,
            -path.stat().st_mtime_ns,
            path.name,
        )
    )
    return candidates


def _prompt_existing_log_path(current_log_path: Path) -> Path | dict[str, Any] | None:
    while True:
        candidates = _log_path_candidates(current_log_path)
        if not candidates:
            print(f"当前没有可选的已有 folder：{_display_path(current_log_path.parent)}")
            return {"__action__": "back"}

        print("\n可选保存目录：")
        for idx, candidate in enumerate(candidates, start=1):
            current_mark = "  [当前]" if candidate.resolve() == current_log_path.resolve() else ""
            print(f"  [{idx}] {_display_path(candidate)}{current_mark}")
        print("输入菜单编号；`b` 返回，`r` 刷新，`q` 退出。")

        raw = input("> ").strip()
        if not raw:
            continue
        lowered = raw.lower()
        if lowered in {"q", "quit", "exit"}:
            raise SystemExit(0)
        if lowered in {"b", "back"}:
            return {"__action__": "back"}
        if lowered in {"r", "refresh"}:
            continue
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(candidates):
                return candidates[idx - 1]
        print("无效的目录输入，请重试。")


def _prompt_manual_log_path(current_log_path: Path) -> Path | dict[str, Any] | None:
    while True:
        raw = input(f"请输入保存目录 [当前={_display_path(current_log_path)}]：").strip()
        if not raw:
            return current_log_path
        lowered = raw.lower()
        if lowered in {"q", "quit", "exit"}:
            raise SystemExit(0)
        if lowered in {"b", "back"}:
            return {"__action__": "back"}
        if lowered in {"r", "refresh"}:
            continue
        return Path(raw).expanduser().resolve()


def _prompt_log_path(current_log_path: Path) -> Path | None:
    while True:
        raw = input(
            f"保存目录 [当前={_display_path(current_log_path)}]，回车沿用，"
            "输入 m 选择已有 folder，输入 i 手动输入新路径，输入 n 取消："
        ).strip()
        if not raw:
            return current_log_path
        lowered = raw.lower()
        if lowered in {"q", "quit", "exit"}:
            raise SystemExit(0)
        if lowered in {"n", "no"}:
            return None
        if lowered in {"m", "menu", "select"}:
            selected = _prompt_existing_log_path(current_log_path)
            if selected is None:
                continue
            if isinstance(selected, dict) and selected.get("__action__") == "back":
                continue
            return Path(selected)
        if lowered in {"i", "input", "manual"}:
            selected = _prompt_manual_log_path(current_log_path)
            if selected is None:
                continue
            if isinstance(selected, dict) and selected.get("__action__") == "back":
                continue
            return Path(selected)
        print("无效输入，请输入回车 / m / i / n。")


def _prompt_max_steps(current_max_steps: int) -> int | None:
    while True:
        raw = input(f"max_steps [当前={current_max_steps}]，回车沿用，输入新数字覆盖，n 取消：").strip()
        if not raw:
            return current_max_steps
        lowered = raw.lower()
        if lowered in {"n", "no"}:
            return None
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
        print("无效的 steps 输入，请输入正整数，或直接回车沿用当前值。")


def _build_eval_command(
    args: argparse.Namespace,
    *,
    skill_dir: Path,
    max_steps: int,
    prompt_override: str | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "benchmark.eval.runner",
        "--snapshot-dir",
        str(skill_dir),
        "--max-steps",
        str(max_steps),
        "--configs-root",
        str(args.configs_root),
        "--log-path",
        str(args.log_path),
        "--vla-type",
        args.vla_type,
    ]
    effective_prompt = prompt_override if prompt_override is not None else args.prompt
    if effective_prompt is not None:
        cmd.extend(["--prompt", effective_prompt])
    if args.vla_config is not None:
        cmd.extend(["--vla-config", args.vla_config])
    if args.env_wrapper is not None:
        cmd.extend(["--env-wrapper", args.env_wrapper])
    if args.ignore_success:
        cmd.append("--ignore-success")
    if args.early_stop_enabled:
        cmd.append("--early-stop-enabled")
        cmd.extend(
            [
                "--early-stop-patience",
                str(args.early_stop_patience),
                "--early-stop-warmup-steps",
                str(args.early_stop_warmup_steps),
                "--early-stop-min-steps",
                str(args.early_stop_min_steps),
                "--early-stop-action-eps",
                str(args.early_stop_action_eps),
                "--early-stop-proprio-eps",
                str(args.early_stop_proprio_eps),
                "--early-stop-base-vel-eps",
                str(args.early_stop_base_vel_eps),
            ]
        )
    return cmd


def _build_result_dir(log_path: Path, skill_info: dict[str, Any]) -> Path:
    metrics = skill_info.get("metrics", {})
    task_index = metrics.get("task_index", None)
    skill_idx = int(skill_info["skill_idx"])
    if isinstance(task_index, int):
        task_dir = task_dir_name(task_index)
    else:
        task_dir = f"task_{metrics.get('task_name', 'task')}"
    return log_path / task_dir / skill_dir_name(skill_idx)


def _find_latest_result_path(log_path: Path, skill_info: dict[str, Any]) -> Path | None:
    metrics = skill_info.get("metrics", {})
    skill_idx = int(skill_info["skill_idx"])
    result_dir = _build_result_dir(log_path, skill_info)
    episode = str(metrics.get("episode", ""))
    candidates = sorted(
        result_dir.glob(f"result_{episode}_{skill_idx:02d}*.json"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )
    if candidates:
        return candidates[-1]
    legacy_path = result_dir / f"result_{episode}_{skill_idx:02d}.json"
    return legacy_path if legacy_path.is_file() else None


def _load_batch_run_summary(
    *,
    log_path: Path,
    skill_info: dict[str, Any],
    returncode: int,
    requested_max_steps: int,
    prompt_override: str | None,
) -> dict[str, Any]:
    result_path = _find_latest_result_path(log_path, skill_info)
    result_data: dict[str, Any] = {}
    if result_path is not None and result_path.is_file():
        try:
            result_data = read_json(result_path)
        except Exception:
            result_data = {}

    prompt_text = str(
        result_data.get("prompt_used")
        or prompt_override
        or skill_info.get("skill_description", "")
    ).strip()
    steps = result_data.get("steps", "?")
    max_steps = result_data.get("max_steps", requested_max_steps)
    video_path = str(result_data.get("video_path", "-"))
    status = "ok" if returncode == 0 else f"exit {returncode}"
    return {
        "skill_idx": int(skill_info["skill_idx"]),
        "prompt": prompt_text or "-",
        "steps": steps,
        "max_steps": max_steps,
        "video_path": video_path,
        "result_path": str(result_path) if result_path is not None else "-",
        "status": status,
        "judge_status": None,
        "judge_result_path": "-",
    }


def _run_auto_judge_for_result(
    *,
    result_path: Path,
    repo_root: Path,
) -> tuple[str, str]:
    judge_output_path = build_judge_output_path(result_path, result_path)
    cmd = [
        sys.executable,
        "-m",
        "benchmark.judge.cli",
        "--result-json",
        str(result_path),
    ]
    print("\n自动启动 judge：")
    print(f"  {shlex.join(cmd)}")
    completed = subprocess.run(cmd, cwd=repo_root, check=False)
    status = "ok" if completed.returncode == 0 else f"exit {completed.returncode}"
    return status, str(judge_output_path)


def _print_batch_run_summary(summaries: list[dict[str, Any]]) -> None:
    if not summaries:
        return
    print("\n批量结果汇总:")
    for summary in summaries:
        print(
            f"  skill_{int(summary['skill_idx']):02d}  "
            f"status={summary['status']}  "
            f"steps={summary['steps']}/{summary['max_steps']}"
        )
        print(f"    prompt: {summary['prompt']}")
        print(f"    video : {summary['video_path']}")
        if summary.get("judge_status") is not None:
            print(f"    judge : {summary['judge_status']}")
            print(f"    judge_result: {summary.get('judge_result_path', '-')}")


def _match_task_info(
    task_infos: list[dict[str, Any]],
    *,
    task_name: str | None = None,
    task_index: int | None = None,
) -> dict[str, Any] | None:
    normalized_task_name = task_name.strip().lower() if task_name else None
    for task_info in task_infos:
        if task_index is not None and int(task_info["task_index"]) == int(task_index):
            return task_info
        if normalized_task_name and str(task_info["task_name"]).strip().lower() == normalized_task_name:
            return task_info
    return None


def build_arg_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="交互式选择 task / episode / skill 类别 / skill 并启动 eval_vla_skill。",
    )
    parser.add_argument(
        "--snapshots-root",
        type=Path,
        default=settings.paths.snapshots_root,
        help="skill snapshot 根目录。",
    )
    parser.add_argument(
        "--meta-root",
        type=Path,
        default=settings.paths.meta_root,
        help="tasks.jsonl 所在目录。",
    )
    parser.add_argument(
        "--server-state-path",
        type=Path,
        default=settings.paths.server_state_path,
        help="server 端当前 task 的共享状态文件路径。",
    )
    parser.add_argument(
        "--ignore-server-task",
        action="store_true",
        help="忽略 server 端共享状态，仍然手动选择 task。",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default=None,
        help="直接指定 task_name，跳过 server / task 菜单。",
    )
    parser.add_argument(
        "--task-index",
        type=int,
        default=None,
        help="直接指定 task_index，跳过 server / task 菜单。",
    )
    parser.add_argument(
        "--group-by",
        type=str,
        choices=("skill-description", "skill-type"),
        default="skill-description",
        help="skill 分类方式。",
    )
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument(
        "--augmented-annotations-root",
        type=Path,
        default=settings.paths.augmented_annotations_root,
        help="augmented_subtask 标注根目录。交互模式下若未显式传 --prompt，则从这里为每个 skill 选择 prompt。",
    )
    parser.add_argument(
        "--prompt-prefix",
        type=str,
        default=None,
        help="可选 prompt 前缀，例如 `Task:`。不传时，交互模式下会询问是否添加 prefix。",
    )
    parser.add_argument("--max-steps", type=int, default=settings.eval.default_max_steps)
    parser.add_argument(
        "--configs-root",
        type=Path,
        default=settings.eval.configs_root,
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=settings.eval.log_path,
    )
    parser.add_argument("--vla-type", type=str, default=settings.eval.vla_type)
    parser.add_argument("--vla-config", type=str, default=None)
    parser.add_argument(
        "--env-wrapper",
        type=str,
        default=settings.eval.env_wrapper,
    )
    parser.add_argument("--ignore-success", action="store_true")
    parser.add_argument(
        "--auto-judge",
        action=argparse.BooleanOptionalAction,
        default=settings.eval.auto_judge,
        help="eval 成功后自动对本次新生成的 result json 调用 judge。",
    )
    parser.add_argument("--early-stop-enabled", action="store_true")
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--early-stop-warmup-steps", type=int, default=20)
    parser.add_argument("--early-stop-min-steps", type=int, default=15)
    parser.add_argument("--early-stop-action-eps", type=float, default=0.002)
    parser.add_argument("--early-stop-proprio-eps", type=float, default=0.001)
    parser.add_argument("--early-stop-base-vel-eps", type=float, default=0.01)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv_list = list(argv or sys.argv[1:])
    parser = build_arg_parser()
    args = parser.parse_args(argv_list)
    log_path_locked = any(
        token == "--log-path" or token.startswith("--log-path=") for token in argv_list
    )

    args.snapshots_root = args.snapshots_root.expanduser().resolve()
    args.meta_root = args.meta_root.expanduser().resolve()
    args.server_state_path = args.server_state_path.expanduser().resolve()
    args.augmented_annotations_root = args.augmented_annotations_root.expanduser().resolve()
    args.configs_root = args.configs_root.expanduser().resolve()
    args.log_path = args.log_path.expanduser().resolve()
    if args.prompt_prefix is not None:
        args.prompt_prefix = args.prompt_prefix.strip() or None
    args.log_path.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[2]

    while True:
        task_infos = get_task_infos(
            snapshots_root=args.snapshots_root,
            meta_root=args.meta_root,
        )
        if not task_infos:
            print("没有找到可评测的 task（需要已有 snapshot.json + replay_metrics.json）。")
            return 0

        scoped_task_infos = task_infos
        fixed_task_choice: dict[str, Any] | None = None
        if args.task_name is not None or args.task_index is not None:
            fixed_task_choice = _match_task_info(
                task_infos,
                task_name=args.task_name,
                task_index=args.task_index,
            )
            if fixed_task_choice is None:
                requested = args.task_name if args.task_name is not None else f"{int(args.task_index):04d}"
                print(f"没有找到指定的 task: {requested}")
                return 0
            scoped_task_infos = [fixed_task_choice]
            print(
                f"\n使用命令行指定 task: "
                f"{int(fixed_task_choice['task_index']):04d} {fixed_task_choice['task_name']}"
            )
        elif not args.ignore_server_task:
            server_state = load_server_state(args.server_state_path)
            if server_state is not None:
                server_task_choice = _match_task_info(
                    task_infos,
                    task_name=server_state.task_name,
                    task_index=server_state.task_index,
                )
                if server_task_choice is not None:
                    print(
                        f"\n当前 server task 供参考: "
                        f"{int(server_task_choice['task_index']):04d} {server_task_choice['task_name']}"
                    )

        category_infos = get_global_category_infos(scoped_task_infos, args.group_by)
        if not category_infos:
            print("当前范围内没有可选的 skill 类别。")
            return 0
        _print_category_menu(category_infos, group_by=args.group_by, allow_back=False)
        category_choice = _prompt_category_choice(category_infos)
        if category_choice is None:
            continue
        if category_choice.get("__action__") == "back":
            continue
        if category_choice.get("__action__") == "refresh":
            continue

        selected_category_name = str(category_choice["category_name"])

        while True:
            fresh_task_infos = get_task_infos(
                snapshots_root=args.snapshots_root,
                meta_root=args.meta_root,
            )
            fresh_scoped_task_infos = (
                [info for info in fresh_task_infos if int(info["task_index"]) == int(fixed_task_choice["task_index"])]
                if fixed_task_choice is not None
                else fresh_task_infos
            )
            matching_task_infos = filter_task_infos_by_category(
                fresh_scoped_task_infos,
                category_name=selected_category_name,
                group_by=args.group_by,
            )
            if not matching_task_infos:
                print("当前没有包含该 skill 类别的 task 了，返回类别列表。")
                break

            if fixed_task_choice is not None:
                task_info = matching_task_infos[0]
            else:
                _print_task_menu(
                    matching_task_infos,
                    selected_category_name=selected_category_name,
                    group_by=args.group_by,
                )
                task_choice = _prompt_task_choice(matching_task_infos)
                if task_choice is None:
                    continue
                if task_choice.get("__action__") == "back":
                    break
                if task_choice.get("__action__") == "refresh":
                    continue
                task_info = next(
                    (
                        info
                        for info in matching_task_infos
                        if int(info["task_index"]) == int(task_choice["task_index"])
                    ),
                    None,
                )
                if task_info is None:
                    print("这个 task 当前不再包含所选 skill 类别，返回 task 列表。")
                    continue

            while True:
                episode_infos = get_episode_infos_for_category(
                    task_info,
                    category_name=selected_category_name,
                    group_by=args.group_by,
                )
                if not episode_infos:
                    print("这个 task 当前没有包含所选 skill 类别的 episode 了，返回 task 列表。")
                    break

                _print_episode_menu(
                    task_info,
                    episode_infos,
                    selected_category_name=selected_category_name,
                    group_by=args.group_by,
                )
                episode_choice = _prompt_episode_choice(episode_infos)
                if episode_choice is None:
                    continue
                if episode_choice.get("__action__") == "back":
                    break
                if episode_choice.get("__action__") == "refresh":
                    continue

                while True:
                    fresh_episode_infos = get_episode_infos_for_category(
                        task_info,
                        category_name=selected_category_name,
                        group_by=args.group_by,
                    )
                    episode_info = next(
                        (
                            info
                            for info in fresh_episode_infos
                            if info["episode_name"] == episode_choice["episode_name"]
                        ),
                        None,
                    )
                    if episode_info is None:
                        print("这个 episode 当前不可用，返回 episode 列表。")
                        break

                    category_view = {
                        "category_name": selected_category_name,
                        "skill_infos": episode_info["skill_infos"],
                    }
                    _print_skill_menu(category_view)
                    skill_choices = _prompt_skill_choices(episode_info["skill_infos"])
                    if skill_choices is None:
                        continue
                    if isinstance(skill_choices, dict) and skill_choices.get("__action__") == "back":
                        break
                    if isinstance(skill_choices, dict) and skill_choices.get("__action__") == "refresh":
                        continue

                    selected_runs: list[dict[str, Any]] = []
                    selected_prompt_prefix = args.prompt_prefix
                    if args.prompt is not None:
                        selected_runs = [
                            {
                                "skill_info": skill_choice,
                                "prompt_text": args.prompt,
                            }
                            for skill_choice in skill_choices
                        ]
                    else:
                        prompt_selection_cancelled = False
                        for skill_choice in skill_choices:
                            prompt_choices: list[str] = []
                            try:
                                prompt_choices = load_augmented_prompts(
                                    args.augmented_annotations_root,
                                    task_index=int(task_info["task_index"]),
                                    episode_name=str(episode_info["episode_name"]),
                                    skill_info=skill_choice,
                                )
                            except Exception as exc:
                                print(
                                    f"\n[WARN] 无法为 skill_{int(skill_choice['skill_idx']):02d} 加载 augmented prompt: {exc}"
                                )
                                print("将只提供手动输入 prompt。")

                            while True:
                                _print_augmented_prompt_menu(
                                    task_info=task_info,
                                    episode_info=episode_info,
                                    skill_info=skill_choice,
                                    prompt_choices=prompt_choices,
                                )
                                prompt_choice = _prompt_augmented_prompt_choice(prompt_choices)
                                if prompt_choice is None:
                                    continue
                                if isinstance(prompt_choice, dict) and prompt_choice.get("__action__") == "back":
                                    prompt_selection_cancelled = True
                                    break
                                if isinstance(prompt_choice, dict) and prompt_choice.get("__action__") == "refresh":
                                    continue
                                if isinstance(prompt_choice, dict) and prompt_choice.get("__action__") == "custom":
                                    custom_prompt = _prompt_custom_prompt()
                                    if custom_prompt is None:
                                        continue
                                    if isinstance(custom_prompt, dict) and custom_prompt.get("__action__") == "back":
                                        continue
                                    if isinstance(custom_prompt, dict) and custom_prompt.get("__action__") == "refresh":
                                        continue
                                    selected_runs.append(
                                        {
                                            "skill_info": skill_choice,
                                            "prompt_text": str(custom_prompt),
                                        }
                                    )
                                    break
                                selected_runs.append(
                                    {
                                        "skill_info": skill_choice,
                                        "prompt_text": str(prompt_choice),
                                    }
                                )
                                break

                            if prompt_selection_cancelled:
                                break

                        if prompt_selection_cancelled:
                            continue

                    if selected_runs and selected_prompt_prefix is None:
                        prompt_preview = str(selected_runs[0]["prompt_text"])
                        prefix_choice = _prompt_prompt_prefix(prompt_preview)
                        if prefix_choice is None:
                            selected_prompt_prefix = None
                        elif isinstance(prefix_choice, dict):
                            if prefix_choice.get("__action__") in {"back", "refresh"}:
                                continue
                        else:
                            selected_prompt_prefix = str(prefix_choice)

                    if selected_runs:
                        selected_runs = [
                            {
                                **run,
                                "prompt_text": _apply_prompt_prefix(
                                    str(run["prompt_text"]),
                                    selected_prompt_prefix,
                                ),
                            }
                            for run in selected_runs
                        ]

                    if not log_path_locked:
                        selected_log_path = _prompt_log_path(args.log_path)
                        if selected_log_path is None:
                            continue
                        args.log_path = selected_log_path.expanduser().resolve()
                        args.log_path.mkdir(parents=True, exist_ok=True)

                    selected_max_steps = _prompt_max_steps(args.max_steps)
                    if selected_max_steps is None:
                        continue
                    args.max_steps = selected_max_steps

                    if len(selected_runs) == 1:
                        summary = f"skill_{int(selected_runs[0]['skill_info']['skill_idx']):02d}"
                    else:
                        summary = ", ".join(
                            f"skill_{int(run['skill_info']['skill_idx']):02d}" for run in selected_runs
                        )
                    print(f"\n本次将执行 {len(selected_runs)} 个 skill: {summary}")
                    print(f"自动 judge: {'开启' if args.auto_judge else '关闭'}")
                    confirm = input("按回车开始，输入 n 取消：").strip().lower()
                    if confirm in {"n", "no"}:
                        continue

                    failures: list[tuple[int, int]] = []
                    judge_failures: list[tuple[int, str]] = []
                    run_summaries: list[dict[str, Any]] = []
                    for run_idx, selected_run in enumerate(selected_runs, start=1):
                        skill_choice = selected_run["skill_info"]
                        prompt_text = selected_run["prompt_text"]
                        cmd = _build_eval_command(
                            args,
                            skill_dir=skill_choice["skill_dir"],
                            max_steps=selected_max_steps,
                            prompt_override=prompt_text,
                        )
                        print("\n即将执行：")
                        print(f"  [{run_idx}/{len(selected_runs)}] {shlex.join(cmd)}")
                        result = subprocess.run(cmd, cwd=repo_root)
                        summary_row = _load_batch_run_summary(
                            log_path=args.log_path,
                            skill_info=skill_choice,
                            returncode=int(result.returncode),
                            requested_max_steps=selected_max_steps,
                            prompt_override=prompt_text,
                        )
                        if args.auto_judge:
                            if result.returncode != 0:
                                summary_row["judge_status"] = "skipped(eval_failed)"
                            else:
                                result_path_text = str(summary_row.get("result_path", "-")).strip()
                                if not result_path_text or result_path_text == "-":
                                    summary_row["judge_status"] = "missing-result"
                                    judge_failures.append(
                                        (int(skill_choice["skill_idx"]), "missing-result")
                                    )
                                else:
                                    judge_status, judge_result_path = _run_auto_judge_for_result(
                                        result_path=Path(result_path_text).expanduser().resolve(),
                                        repo_root=repo_root,
                                    )
                                    summary_row["judge_status"] = judge_status
                                    summary_row["judge_result_path"] = judge_result_path
                                    if judge_status != "ok":
                                        judge_failures.append(
                                            (int(skill_choice["skill_idx"]), judge_status)
                                        )
                        run_summaries.append(summary_row)
                        if result.returncode == 0:
                            print(f"\nskill_{int(skill_choice['skill_idx']):02d} 评测结束。")
                        else:
                            print(f"\nskill_{int(skill_choice['skill_idx']):02d} 退出码为 {result.returncode}。")
                            failures.append((int(skill_choice["skill_idx"]), int(result.returncode)))

                    _print_batch_run_summary(run_summaries)
                    if failures:
                        failure_text = ", ".join(
                            f"skill_{skill_idx:02d}:{code}" for skill_idx, code in failures
                        )
                        print(f"\n批量执行完成，但有失败项: {failure_text}")
                    elif judge_failures:
                        judge_failure_text = ", ".join(
                            f"skill_{skill_idx:02d}:{status}" for skill_idx, status in judge_failures
                        )
                        print(f"\n评测完成，但自动 judge 有失败项: {judge_failure_text}")
                    else:
                        print("\n批量执行完成，全部成功。")

            if fixed_task_choice is not None:
                break
            continue

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
