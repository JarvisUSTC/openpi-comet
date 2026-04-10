from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .io import read_json, read_jsonl


def load_task_mapping(tasks_jsonl_path: Path) -> dict[int, dict[str, Any]]:
    mapping: dict[int, dict[str, Any]] = {}
    for item in read_jsonl(tasks_jsonl_path):
        mapping[int(item["task_index"])] = item
    return mapping


def parse_instance_id(episode_name: str) -> int:
    digits = episode_name.replace("episode_", "").split(".")[0]
    rest = int(digits[4:])
    return rest // 10


def get_skill_infos(episode_dir: Path) -> list[dict[str, Any]]:
    skill_infos: list[dict[str, Any]] = []
    for skill_dir in sorted(episode_dir.glob("skill_*")):
        if not skill_dir.is_dir():
            continue
        snapshot_path = skill_dir / "snapshot.json"
        metrics_path = skill_dir / "replay_metrics.json"
        if not snapshot_path.exists() or not metrics_path.exists():
            continue
        metrics = read_json(metrics_path)
        skill_infos.append(
            {
                "skill_dir": skill_dir,
                "skill_dir_name": skill_dir.name,
                "skill_idx": int(metrics.get("skill_idx", -1)),
                "skill_description": str(metrics.get("skill_description", "")),
                "skill_type": [str(x) for x in metrics.get("skill_type", [])],
                "object_ids": [str(x) for x in metrics.get("object_ids", [])],
                "frame_start": int(metrics.get("frame_start", -1)),
                "frame_end": int(metrics.get("frame_end", -1)),
                "metrics": metrics,
            }
        )
    skill_infos.sort(key=lambda info: (int(info["skill_idx"]), str(info["skill_dir_name"])))
    return skill_infos


def get_episode_infos(task_info: dict[str, Any]) -> list[dict[str, Any]]:
    episode_infos: list[dict[str, Any]] = []
    for episode_dir in sorted(task_info["task_dir"].glob("episode_*")):
        if not episode_dir.is_dir():
            continue
        skill_infos = get_skill_infos(episode_dir)
        if not skill_infos:
            continue
        episode_name = episode_dir.name
        episode_infos.append(
            {
                "episode_name": episode_name,
                "instance_id": parse_instance_id(episode_name),
                "episode_dir": episode_dir,
                "skill_count": len(skill_infos),
                "skill_infos": skill_infos,
            }
        )
    return episode_infos


def get_task_infos(*, snapshots_root: Path, meta_root: Path) -> list[dict[str, Any]]:
    task_mapping = load_task_mapping(meta_root / "tasks.jsonl")
    task_infos: list[dict[str, Any]] = []
    for task_dir in sorted(snapshots_root.glob("task_*")):
        if not task_dir.is_dir():
            continue
        try:
            task_index = int(task_dir.name.split("_")[-1])
        except ValueError:
            continue
        episode_dirs = [path for path in sorted(task_dir.glob("episode_*")) if path.is_dir()]
        episode_count = 0
        skill_count = 0
        for episode_dir in episode_dirs:
            n_skills = len(get_skill_infos(episode_dir))
            if n_skills <= 0:
                continue
            episode_count += 1
            skill_count += n_skills
        if episode_count <= 0:
            continue
        task_meta = task_mapping.get(task_index, {})
        task_infos.append(
            {
                "task_index": task_index,
                "task_name": task_meta.get("task_name", f"task_{task_index:04d}"),
                "task_prompt": task_meta.get("task", ""),
                "task_dir": task_dir,
                "episode_count": episode_count,
                "skill_count": skill_count,
            }
        )
    return task_infos


def skill_group_key(skill_info: dict[str, Any], group_by: str) -> str:
    if group_by == "skill-type":
        skill_types = skill_info.get("skill_type") or []
        return ", ".join(skill_types) if skill_types else "<none>"
    value = str(skill_info.get("skill_description", "")).strip()
    return value if value else "<empty>"


def get_global_category_infos(task_infos: list[dict[str, Any]], group_by: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[int, str, dict[str, Any]]]] = defaultdict(list)
    for task_info in task_infos:
        for episode_info in get_episode_infos(task_info):
            for skill_info in episode_info["skill_infos"]:
                grouped[skill_group_key(skill_info, group_by)].append(
                    (int(task_info["task_index"]), str(episode_info["episode_name"]), skill_info)
                )
    category_infos: list[dict[str, Any]] = []
    for name, items in grouped.items():
        skill_types = sorted(
            {
                skill_type
                for _, _, skill_info in items
                for skill_type in skill_info.get("skill_type", [])
                if skill_type
            }
        )
        category_infos.append(
            {
                "category_name": name,
                "count": len(items),
                "task_count": len({task_index for task_index, _, _ in items}),
                "episode_count": len({episode_name for _, episode_name, _ in items}),
                "skill_types": skill_types,
            }
        )
    category_infos.sort(key=lambda info: (str(info["category_name"]), int(info["count"])))
    return category_infos


def filter_task_infos_by_category(
    task_infos: list[dict[str, Any]], *, category_name: str, group_by: str
) -> list[dict[str, Any]]:
    matched_task_infos: list[dict[str, Any]] = []
    for task_info in task_infos:
        matched_episode_count = 0
        matched_skill_count = 0
        for episode_info in get_episode_infos(task_info):
            matched_skills = [
                skill_info
                for skill_info in episode_info["skill_infos"]
                if skill_group_key(skill_info, group_by) == category_name
            ]
            if not matched_skills:
                continue
            matched_episode_count += 1
            matched_skill_count += len(matched_skills)
        if matched_skill_count <= 0:
            continue
        matched_task_info = dict(task_info)
        matched_task_info["matched_episode_count"] = matched_episode_count
        matched_task_info["matched_skill_count"] = matched_skill_count
        matched_task_infos.append(matched_task_info)
    return matched_task_infos


def get_episode_infos_for_category(
    task_info: dict[str, Any], *, category_name: str, group_by: str
) -> list[dict[str, Any]]:
    filtered_episode_infos: list[dict[str, Any]] = []
    for episode_info in get_episode_infos(task_info):
        matched_skills = [
            skill_info
            for skill_info in episode_info["skill_infos"]
            if skill_group_key(skill_info, group_by) == category_name
        ]
        if not matched_skills:
            continue
        filtered_episode_info = dict(episode_info)
        filtered_episode_info["skill_infos"] = matched_skills
        filtered_episode_info["skill_count"] = len(matched_skills)
        filtered_episode_infos.append(filtered_episode_info)
    return filtered_episode_infos
