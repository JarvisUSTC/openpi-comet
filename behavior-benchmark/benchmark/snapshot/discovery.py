from __future__ import annotations

from pathlib import Path
from random import Random
from typing import Any

import h5py

from benchmark.core.inventory import load_task_mapping, parse_instance_id
from benchmark.core.io import read_json

NUM_TRAIN_EPISODES = 180


def get_available_task_dirs(annotation_root: Path, raw_root: Path) -> list[tuple[int, Path, Path]]:
    results: list[tuple[int, Path, Path]] = []
    for ann_task_dir in sorted(annotation_root.glob("task-*")):
        if not ann_task_dir.is_dir():
            continue
        try:
            task_index = int(ann_task_dir.name.split("-")[-1])
        except ValueError:
            continue
        raw_task_dir = raw_root / ann_task_dir.name
        if raw_task_dir.is_dir():
            results.append((task_index, ann_task_dir, raw_task_dir))
    return results


def is_single_segment(frame_duration: Any) -> bool:
    if not frame_duration or not isinstance(frame_duration, (list, tuple)):
        return False
    return not isinstance(frame_duration[0], (list, tuple))


def episode_has_only_single_segment_skills(annotation: dict[str, Any]) -> bool:
    for skill in annotation.get("skill_annotation", []):
        if not is_single_segment(skill.get("frame_duration")):
            return False
    return True


def episode_strict_alignment_ok(annotation: dict[str, Any], action_len: int) -> bool:
    single_segment = [
        skill
        for skill in annotation.get("skill_annotation", [])
        if is_single_segment(skill.get("frame_duration"))
    ]
    if not single_segment:
        return False
    return all(int(skill["frame_duration"][0]) < action_len for skill in single_segment)


def get_heldout_ann_paths(ann_task_dir: Path) -> tuple[list[Path], bool, int]:
    all_ann_paths = sorted(ann_task_dir.glob("episode_*.json"))
    heldout_ann_paths = all_ann_paths[NUM_TRAIN_EPISODES:]
    used_all_episodes = False
    if not heldout_ann_paths:
        heldout_ann_paths = all_ann_paths
        used_all_episodes = True
    return heldout_ann_paths, used_all_episodes, len(all_ann_paths)


def load_annotation(annotation_path: Path) -> dict[str, Any]:
    return read_json(annotation_path)


def try_get_action_len(hdf5_path: Path) -> int | None:
    try:
        with h5py.File(str(hdf5_path), "r") as hdf5_file:
            return int(hdf5_file["data/demo_0/action"].shape[0])
    except Exception:
        return None


def count_valid_skills(annotation: dict[str, Any], action_len: int) -> int:
    count = 0
    for skill in annotation.get("skill_annotation", []):
        frame_duration = skill.get("frame_duration")
        if not is_single_segment(frame_duration):
            continue
        if int(frame_duration[0]) >= action_len:
            continue
        count += 1
    return count


def count_existing_skill_outputs(output_dir: Path, task_index: int, episode_name: str) -> tuple[int, int]:
    episode_dir = output_dir / f"task_{task_index:04d}" / episode_name
    if not episode_dir.is_dir():
        return 0, 0
    skill_dirs = [path for path in episode_dir.glob("skill_*") if path.is_dir()]
    snapshot_count = sum((path / "snapshot.json").exists() for path in skill_dirs)
    metrics_count = sum((path / "replay_metrics.json").exists() for path in skill_dirs)
    return snapshot_count, metrics_count


def collect_skill_targets(
    annotation: dict[str, Any],
    action_len: int,
    *,
    max_skills: int = -1,
    rng: Random | None = None,
) -> list[tuple[int, dict[str, Any]]]:
    valid: list[tuple[int, dict[str, Any]]] = []
    for skill in annotation.get("skill_annotation", []):
        frame_duration = skill.get("frame_duration")
        if not is_single_segment(frame_duration):
            continue
        frame_start = int(frame_duration[0])
        if frame_start >= action_len:
            continue
        valid.append((frame_start, skill))

    if max_skills > 0 and len(valid) > max_skills:
        valid = rng.sample(valid, max_skills) if rng is not None else valid[:max_skills]

    valid.sort(key=lambda item: item[0])
    return valid


def collect_heldout_episode_infos(
    *,
    task_index: int,
    ann_task_dir: Path,
    raw_task_dir: Path,
    output_dir: Path,
    require_strict_alignment: bool,
    max_skills: int,
) -> tuple[list[dict[str, Any]], bool, int]:
    heldout_ann_paths, used_all_episodes, total_ann_count = get_heldout_ann_paths(ann_task_dir)
    infos: list[dict[str, Any]] = []

    for ann_path in heldout_ann_paths:
        episode_name = ann_path.stem
        instance_id = parse_instance_id(episode_name)
        hdf5_path = raw_task_dir / ann_path.name.replace(".json", ".hdf5")
        info: dict[str, Any] = {
            "episode_name": episode_name,
            "instance_id": instance_id,
            "ann_path": ann_path,
            "hdf5_path": hdf5_path,
            "status": "unknown",
        }

        if not hdf5_path.exists():
            info["status"] = "missing_hdf5"
            infos.append(info)
            continue

        annotation = load_annotation(ann_path)
        info["annotation"] = annotation
        if not annotation.get("skill_annotation"):
            info["status"] = "no_skill_annotation"
            infos.append(info)
            continue
        if not episode_has_only_single_segment_skills(annotation):
            info["status"] = "multi_segment"
            infos.append(info)
            continue

        action_len = try_get_action_len(hdf5_path)
        if action_len is None:
            info["status"] = "invalid_hdf5"
            infos.append(info)
            continue
        info["action_len"] = action_len

        if require_strict_alignment and not episode_strict_alignment_ok(annotation, action_len):
            info["status"] = "misaligned"
            infos.append(info)
            continue

        total_skill_targets = collect_skill_targets(
            annotation,
            action_len,
            max_skills=-1,
            rng=None,
        )
        if not total_skill_targets:
            info["status"] = "no_valid_skills"
            infos.append(info)
            continue

        expected_skill_count = len(total_skill_targets)
        if max_skills > 0:
            expected_skill_count = min(expected_skill_count, max_skills)

        snapshot_count, metrics_count = count_existing_skill_outputs(
            output_dir,
            task_index,
            episode_name,
        )
        info.update(
            {
                "status": "ready",
                "expected_skill_count": expected_skill_count,
                "snapshot_count": snapshot_count,
                "metrics_count": metrics_count,
                "is_complete": (
                    snapshot_count >= expected_skill_count and metrics_count >= expected_skill_count
                ),
            }
        )
        infos.append(info)

    return infos, used_all_episodes, total_ann_count


def collect_task_infos(
    *,
    annotation_root: Path,
    raw_root: Path,
    meta_root: Path,
    output_dir: Path,
    require_strict_alignment: bool,
    max_skills: int,
) -> list[dict[str, Any]]:
    task_mapping = load_task_mapping(meta_root / "tasks.jsonl")
    task_infos: list[dict[str, Any]] = []

    for task_index, ann_task_dir, raw_task_dir in get_available_task_dirs(annotation_root, raw_root):
        infos, used_all_episodes, total_ann_count = collect_heldout_episode_infos(
            task_index=task_index,
            ann_task_dir=ann_task_dir,
            raw_task_dir=raw_task_dir,
            output_dir=output_dir,
            require_strict_alignment=require_strict_alignment,
            max_skills=max_skills,
        )
        missing_infos = [info for info in infos if info["status"] == "ready" and not info["is_complete"]]
        complete_infos = [info for info in infos if info["status"] == "ready" and info["is_complete"]]
        if not missing_infos:
            continue

        task_meta = task_mapping.get(task_index, {})
        task_infos.append(
            {
                "task_index": task_index,
                "task_name": task_meta.get("task_name", f"task_{task_index:04d}"),
                "task_prompt": task_meta.get("task", ""),
                "ann_task_dir": ann_task_dir,
                "raw_task_dir": raw_task_dir,
                "episode_infos": infos,
                "missing_episode_infos": missing_infos,
                "complete_episode_count": len(complete_infos),
                "ready_episode_count": len(missing_infos) + len(complete_infos),
                "used_all_episodes": used_all_episodes,
                "total_ann_count": total_ann_count,
            }
        )

    task_infos.sort(key=lambda info: int(info["task_index"]))
    return task_infos


def episode_status_reason(info: dict[str, Any]) -> str:
    status = str(info.get("status", "unknown"))
    if status == "missing_hdf5":
        return "缺少同名 HDF5"
    if status == "invalid_hdf5":
        return "HDF5 读取失败"
    if status == "no_skill_annotation":
        return "skill_annotation 为空"
    if status == "multi_segment":
        return "包含多段 frame_duration skill"
    if status == "misaligned":
        return "HDF5 action 与标注帧号不严格对齐"
    if status == "no_valid_skills":
        return "没有可切的有效 skill"
    if status == "ready" and info.get("is_complete"):
        return "已完整切出"
    if status == "ready":
        return "可用"
    return f"未知状态: {status}"


def print_episode_inventory(
    *,
    infos: list[dict[str, Any]],
    used_all_episodes: bool,
    total_ann_count: int,
    require_strict_alignment: bool,
) -> None:
    if used_all_episodes:
        print(
            f"  [WARN] 只有 {total_ann_count} 个 annotation，不足 {NUM_TRAIN_EPISODES}，使用全部 episode"
        )

    status_counts: dict[str, int] = {}
    for info in infos:
        status = str(info["status"])
        status_counts[status] = status_counts.get(status, 0) + 1

    if status_counts.get("missing_hdf5", 0) > 0:
        print(f"  [INFO] 跳过 {status_counts['missing_hdf5']} 个缺少 HDF5 的 episode")
    if status_counts.get("invalid_hdf5", 0) > 0:
        print(f"  [INFO] 跳过 {status_counts['invalid_hdf5']} 个无效 HDF5 的 episode")
    if status_counts.get("multi_segment", 0) > 0:
        print(f"  [INFO] 跳过 {status_counts['multi_segment']} 个含多段帧 skill 的 episode")
    if status_counts.get("no_skill_annotation", 0) > 0:
        print(f"  [INFO] 跳过 {status_counts['no_skill_annotation']} 个没有 skill_annotation 的 episode")
    if status_counts.get("no_valid_skills", 0) > 0:
        print(f"  [INFO] 跳过 {status_counts['no_valid_skills']} 个没有有效 skill 的 episode")
    if require_strict_alignment and status_counts.get("misaligned", 0) > 0:
        print(
            "  [INFO] 跳过 "
            f"{status_counts['misaligned']} 个 HDF5 与标注帧号不对齐的 episode (require_strict_alignment)"
        )

    ready_infos = [info for info in infos if info["status"] == "ready"]
    complete_ready_infos = [info for info in ready_infos if info["is_complete"]]
    missing_ready_infos = [info for info in ready_infos if not info["is_complete"]]
    print(
        "  [INFO] held-out episode 状态: "
        f"ready={len(ready_infos)}, missing={len(missing_ready_infos)}, complete={len(complete_ready_infos)}"
    )


def choose_episode(
    *,
    task_index: int,
    ann_task_dir: Path,
    raw_task_dir: Path,
    output_dir: Path,
    rng: Random,
    require_strict_alignment: bool = True,
    max_skills: int = -1,
    episode_name: str | None = None,
) -> dict[str, Any]:
    infos, used_all_episodes, total_ann_count = collect_heldout_episode_infos(
        task_index=task_index,
        ann_task_dir=ann_task_dir,
        raw_task_dir=raw_task_dir,
        output_dir=output_dir,
        require_strict_alignment=require_strict_alignment,
        max_skills=max_skills,
    )
    print_episode_inventory(
        infos=infos,
        used_all_episodes=used_all_episodes,
        total_ann_count=total_ann_count,
        require_strict_alignment=require_strict_alignment,
    )

    ready_infos = [info for info in infos if info["status"] == "ready"]
    missing_ready_infos = [info for info in ready_infos if not info["is_complete"]]

    if episode_name is not None:
        for info in infos:
            if info["episode_name"] != episode_name:
                continue
            if info["status"] != "ready":
                raise RuntimeError(
                    f"指定 episode {episode_name} 不可用：{episode_status_reason(info)}"
                )
            if info["is_complete"]:
                raise RuntimeError(
                    f"指定 episode {episode_name} 已完整切出；可先运行 "
                    f"`--task-index {task_index} --list-missing-episodes` 查看仍可选的 held-out episode"
                )
            return info

        explicit_ann_path = ann_task_dir / f"{episode_name}.json"
        if explicit_ann_path.exists():
            raise RuntimeError(f"指定 episode {episode_name} 存在，但不在当前 held-out 候选集合中")
        raise RuntimeError(f"指定 episode {episode_name} 不存在于 {ann_task_dir}")

    if missing_ready_infos:
        return rng.choice(missing_ready_infos)
    if ready_infos:
        return rng.choice(ready_infos)

    extra = (
        "（严格对齐过滤下 held-out 中可能无对齐 episode；可用 --no-require-strict-alignment 关闭）"
        if require_strict_alignment
        else ""
    )
    raise RuntimeError(f"后 20 个 episode 中未找到可用 episode: {ann_task_dir}{extra}")
