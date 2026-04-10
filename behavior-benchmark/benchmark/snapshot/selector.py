from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from benchmark.config.settings import get_settings

from .discovery import collect_task_infos


def _prompt_choice(
    items: list[dict[str, Any]],
    *,
    label: str,
    key_name: str,
    alt_key_name: str | None = None,
) -> dict[str, Any] | None:
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
        if 1 <= idx <= len(items):
            return items[idx - 1]

    normalized_raw = raw
    if key_name == "episode_name":
        if normalized_raw.endswith(".json"):
            normalized_raw = normalized_raw[:-5]
        if normalized_raw.isdigit():
            normalized_raw = f"episode_{normalized_raw}"

    for item in items:
        if str(item[key_name]) == raw or str(item[key_name]) == normalized_raw:
            return item
        if alt_key_name is not None and str(item[alt_key_name]) == raw:
            return item

    print(f"无效的{label}输入，请重试。")
    return None


def _print_task_menu(task_infos: list[dict[str, Any]]) -> None:
    print("\n可选 task（held-out 中仍有未完整切出的 episode）:")
    for idx, info in enumerate(task_infos, start=1):
        print(
            f"  [{idx}] task {int(info['task_index']):04d}  "
            f"{info['task_name']}  "
            f"(missing={len(info['missing_episode_infos'])}, "
            f"complete={info['complete_episode_count']}, "
            f"ready={info['ready_episode_count']})"
        )
    print("输入编号或 task index；`r` 刷新，`q` 退出。")


def _print_episode_menu(task_info: dict[str, Any]) -> None:
    episodes = task_info["missing_episode_infos"]
    print(f"\nTask {int(task_info['task_index']):04d} {task_info['task_name']} 的可选 episode:")
    if task_info["task_prompt"]:
        print(f"  prompt: {task_info['task_prompt']}")
    for idx, info in enumerate(episodes, start=1):
        print(
            f"  [{idx}] {info['episode_name']}  "
            f"(instance={info['instance_id']}, "
            f"snapshot={info['snapshot_count']}/{info['expected_skill_count']}, "
            f"metrics={info['metrics_count']}/{info['expected_skill_count']})"
        )
    print("输入编号或 episode 名；`b` 返回 task 列表，`r` 刷新，`q` 退出。")


def _build_generate_command(
    args: argparse.Namespace,
    *,
    task_index: int,
    episode_name: str,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "benchmark.snapshot.engine",
        "--annotation-root",
        str(args.annotation_root),
        "--meta-root",
        str(args.meta_root),
        "--raw-root",
        str(args.raw_root),
        "--configs-root",
        str(args.configs_root),
        "--task-index",
        str(task_index),
        "--episode-name",
        episode_name,
        "--playback-mode",
        args.playback_mode,
        "--seed",
        str(args.seed),
    ]
    if args.output_root is not None:
        cmd.extend(["--output-root", str(args.output_root)])
    if args.max_skills_per_episode != -1:
        cmd.extend(["--max-skills-per-episode", str(args.max_skills_per_episode)])
    if args.no_require_strict_alignment:
        cmd.append("--no-require-strict-alignment")
    if args.record_skills:
        cmd.append("--record-skills")
    return cmd


def _build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Interactive selector for task / episode snapshot generation.",
    )
    parser.add_argument(
        "--annotation-root",
        type=Path,
        default=settings.snapshot.annotation_root,
    )
    parser.add_argument(
        "--meta-root",
        type=Path,
        default=settings.paths.meta_root,
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=settings.snapshot.raw_root,
    )
    parser.add_argument(
        "--configs-root",
        type=Path,
        default=settings.snapshot.configs_root,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=settings.paths.snapshots_root,
    )
    parser.add_argument(
        "--max-skills-per-episode",
        type=int,
        default=settings.snapshot.default_max_skills_per_episode,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=settings.snapshot.default_seed,
    )
    parser.add_argument(
        "--no-require-strict-alignment",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--record-skills",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--playback-mode",
        type=str,
        choices=("replay", "state"),
        default=settings.snapshot.default_playback_mode,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = _build_parser()
    args = parser.parse_args(argv)

    args.annotation_root = args.annotation_root.expanduser().resolve()
    args.meta_root = args.meta_root.expanduser().resolve()
    args.raw_root = args.raw_root.expanduser().resolve()
    args.configs_root = args.configs_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()

    require_strict_alignment = not args.no_require_strict_alignment

    while True:
        task_infos = collect_task_infos(
            annotation_root=args.annotation_root,
            raw_root=args.raw_root,
            meta_root=args.meta_root,
            output_dir=args.output_root,
            require_strict_alignment=require_strict_alignment,
            max_skills=args.max_skills_per_episode,
        )
        if not task_infos:
            print("没有找到可选 task。所有 held-out episode 可能都已经完整切出。")
            return 0

        _print_task_menu(task_infos)
        task_choice = _prompt_choice(
            task_infos,
            label="task",
            key_name="task_index",
        )
        if task_choice is None:
            continue
        if task_choice.get("__action__") in {"refresh", "back"}:
            continue

        while True:
            refreshed_tasks = collect_task_infos(
                annotation_root=args.annotation_root,
                raw_root=args.raw_root,
                meta_root=args.meta_root,
                output_dir=args.output_root,
                require_strict_alignment=require_strict_alignment,
                max_skills=args.max_skills_per_episode,
            )
            task_info = next(
                (
                    info
                    for info in refreshed_tasks
                    if int(info["task_index"]) == int(task_choice["task_index"])
                ),
                None,
            )
            if task_info is None:
                print("这个 task 已经没有可选 episode 了，返回 task 列表。")
                break

            _print_episode_menu(task_info)
            episode_choice = _prompt_choice(
                task_info["missing_episode_infos"],
                label="episode",
                key_name="episode_name",
                alt_key_name="instance_id",
            )
            if episode_choice is None:
                continue
            action = episode_choice.get("__action__")
            if action == "refresh":
                continue
            if action == "back":
                break

            cmd = _build_generate_command(
                args,
                task_index=int(task_info["task_index"]),
                episode_name=str(episode_choice["episode_name"]),
            )
            print("\n即将执行：")
            print(f"  {shlex.join(cmd)}")
            confirm = input("按回车开始，输入 n 取消：").strip().lower()
            if confirm in {"n", "no"}:
                continue

            result = subprocess.run(cmd, cwd=settings.paths.repo_root, check=False)
            if result.returncode == 0:
                print("\n本次生成命令已结束。将刷新 episode 列表。")
            else:
                print(
                    f"\n生成命令退出码为 {result.returncode}。"
                    " 这通常表示中途报错或环境退出异常；将刷新列表。"
                )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
