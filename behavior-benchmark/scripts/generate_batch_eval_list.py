#!/usr/bin/env python3
"""
生成批量评测任务列表。

功能：
1. 扫描所有可用的 skill snapshots
2. 按 skill_description 分组
3. 选择 34 个不同的 skill 类型
4. 每个 skill 选 4 个不同的 [task + episode + skill_id] 组合
5. 优先选择：分散在不同 task/episode、复杂度高的任务

输出：JSON 文件，包含所有执行所需的信息
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# 配置路径
SNAPSHOTS_ROOT = Path("/home/simpleai/Jiawei/SimpleRoboAgent/logs/skill_snapshots")
OUTPUT_PATH = Path("/home/simpleai/Jiawei/behavior-benchmark/config/batch_eval_list_34x4.json")

# 随机种子，保证可复现
RANDOM_SEED = 42


def load_replay_metrics(skill_dir: Path) -> dict[str, Any] | None:
    """加载单个 skill 的 replay_metrics.json"""
    metrics_path = skill_dir / "replay_metrics.json"
    if not metrics_path.exists():
        return None
    try:
        with open(metrics_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def compute_complexity(metrics: dict[str, Any]) -> float:
    """
    计算任务复杂度分数，用于排序选择。
    分数越高表示任务越复杂，越可能被选中。
    """
    score = 0.0

    # 1. task prompt 长度（越长通常越复杂）
    task_prompt = metrics.get("task_prompt", "")
    score += len(task_prompt) / 100.0

    # 2. object_ids 数量（涉及物体越多越复杂）
    object_ids = metrics.get("object_ids", [])
    score += len(object_ids) * 2.0

    # 3. manipulating_object_ids 数量（操作物体越多越复杂）
    manipulating_ids = metrics.get("manipulating_object_ids", [])
    score += len(manipulating_ids) * 3.0

    # 4. skill 持续时间（越长可能越复杂）
    frame_start = metrics.get("frame_start", 0)
    frame_end = metrics.get("frame_end", 0)
    duration = frame_end - frame_start
    score += duration / 100.0

    # 5. 特定复杂 skill type 加分
    skill_desc = metrics.get("skill_description", "")
    complex_keywords = ["place on next to", "place in next to", "place on", "place in", "open", "close"]
    for keyword in complex_keywords:
        if keyword in skill_desc.lower():
            score += 5.0
            break

    return score


def scan_all_skills(root: Path) -> dict[str, list[dict[str, Any]]]:
    """
    扫描所有 skill，按 skill_description 分组。
    返回: {skill_description: [skill_items, ...]}
    """
    skill_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    if not root.exists():
        print(f"错误: 路径不存在 {root}")
        return skill_groups

    # 遍历 task_* / episode_* / skill_*
    for task_dir in sorted(root.glob("task_*")):
        if not task_dir.is_dir():
            continue

        task_index_str = task_dir.name.split("_")[-1]
        try:
            task_index = int(task_index_str)
        except ValueError:
            continue

        for episode_dir in sorted(task_dir.glob("episode_*")):
            if not episode_dir.is_dir():
                continue

            episode_name = episode_dir.name

            for skill_dir in sorted(episode_dir.glob("skill_*")):
                if not skill_dir.is_dir():
                    continue

                metrics = load_replay_metrics(skill_dir)
                if metrics is None:
                    continue

                # 构建完整信息
                skill_item = {
                    # 核心路径
                    "snapshot_dir": str(skill_dir),
                    "snapshot_json": str(skill_dir / "snapshot.json"),
                    "replay_metrics_json": str(skill_dir / "replay_metrics.json"),
                    "before_image": str(skill_dir / "before.png"),

                    # 基础元数据
                    "task_index": metrics.get("task_index", task_index),
                    "task_name": metrics.get("task_name", ""),
                    "task_prompt": metrics.get("task_prompt", ""),
                    "episode": episode_name,
                    "instance_id": metrics.get("instance_id", 0),
                    "skill_idx": metrics.get("skill_idx", -1),
                    "skill_id": metrics.get("skill_id", []),
                    "skill_description": metrics.get("skill_description", "unknown"),
                    "skill_type": metrics.get("skill_type", []),

                    # 物体信息
                    "object_ids": metrics.get("object_ids", []),
                    "manipulating_object_ids": metrics.get("manipulating_object_ids", []),
                    "memory_prefix": metrics.get("memory_prefix", []),
                    "spatial_prefix": metrics.get("spatial_prefix", []),

                    # 时间范围
                    "frame_start": metrics.get("frame_start", 0),
                    "frame_end": metrics.get("frame_end", 0),
                    "replay_frames": metrics.get("replay_frames", 0),

                    # 复杂度评分
                    "complexity_score": compute_complexity(metrics),
                }

                skill_desc = skill_item["skill_description"]
                if not skill_desc or skill_desc == "unknown":
                    skill_desc = "_unknown"

                skill_groups[skill_desc].append(skill_item)

    return skill_groups


def select_diverse_samples(
    items: list[dict[str, Any]],
    n_select: int = 4,
) -> list[dict[str, Any]]:
    """
    从同一 skill 的多个实例中，选择 n_select 个分散的样本。
    策略：
    1. 优先选择不同 task
    2. 其次选择不同 episode
    3. 在同等条件下选择复杂度高的
    """
    if len(items) <= n_select:
        return items

    # 按 (task_index, complexity) 分组和排序
    task_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        task_groups[item["task_index"]].append(item)

    # 对每个 task 内的 items 按复杂度排序
    for task_idx in task_groups:
        task_groups[task_idx].sort(key=lambda x: x["complexity_score"], reverse=True)

    selected = []
    task_indices = list(task_groups.keys())

    # 轮询选择，确保分散性
    round_idx = 0
    while len(selected) < n_select and task_groups:
        made_progress = False

        for task_idx in task_indices:
            if task_idx not in task_groups:
                continue

            group = task_groups[task_idx]
            if round_idx < len(group):
                selected.append(group[round_idx])
                made_progress = True

                if len(selected) >= n_select:
                    break
            else:
                # 该 task 已经没有更多样本了
                del task_groups[task_idx]

        if not made_progress:
            break

        round_idx += 1

    # 如果数量不够，从剩下的里选复杂度最高的
    if len(selected) < n_select:
        remaining = []
        for group in task_groups.values():
            for item in group[round_idx:]:
                remaining.append(item)

        remaining.sort(key=lambda x: x["complexity_score"], reverse=True)
        needed = n_select - len(selected)
        selected.extend(remaining[:needed])

    return selected


def format_prompt(skill_item: dict[str, Any]) -> str:
    """
    格式化 prompt，参考 runner.py 中的 _format_training_prompt 逻辑。
    """
    skill_desc = skill_item["skill_description"]
    object_ids = skill_item["object_ids"]
    memory_prefix = skill_item.get("memory_prefix", [])

    # 清理物体名称
    def sanitize(name: str) -> str:
        # 移除末尾的 _数字 后缀
        parts = name.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            name = parts[0]
        return name.replace("_", " ").strip()

    objs = [sanitize(o) for o in object_ids if o]

    # 处理特殊 skill 格式
    if skill_desc == "place on next to" and len(objs) >= 3:
        return f"place {objs[0]} on {objs[1]} next to {objs[2]}"
    if skill_desc == "place in next to" and len(objs) >= 3:
        return f"place {objs[0]} in {objs[1]} next to {objs[2]}"

    # 常见介词模式
    prep_patterns = [
        (" from", 5), (" next to", 8), (" into", 5), (" onto", 5),
        (" under", 6), (" on", 3), (" in", 3), (" to", 3), (" off", 4), (" with", 5),
    ]

    if len(objs) >= 2:
        for prep, strip_len in prep_patterns:
            if skill_desc.endswith(prep):
                verb = skill_desc[:-strip_len].strip()
                return f"{verb} {objs[0]} {prep.strip()} {objs[1]}"

    # 默认：直接拼接
    if objs:
        return f"{skill_desc} {' '.join(objs)}".strip()
    return skill_desc


def generate_batch_list(
    n_skills: int = 34,
    n_samples_per_skill: int = 4,
) -> dict[str, Any]:
    """
    生成完整的批量评测列表。
    """
    print(f"扫描路径: {SNAPSHOTS_ROOT}")

    # 1. 扫描所有 skills
    skill_groups = scan_all_skills(SNAPSHOTS_ROOT)

    print(f"\n发现 {len(skill_groups)} 个不同的 skill 类型:")
    for desc, items in sorted(skill_groups.items(), key=lambda x: -len(x[1])):
        n_tasks = len(set(i["task_index"] for i in items))
        n_episodes = len(set(i["episode"] for i in items))
        print(f"  - {desc!r}: {len(items)} 个实例 (跨 {n_tasks} tasks, {n_episodes} episodes)")

    # 2. 选择 skill 类型
    # 策略：优先选择实例数量多、跨 task 多的 skill
    def skill_diversity_score(items: list[dict[str, Any]]) -> float:
        n_tasks = len(set(i["task_index"] for i in items))
        n_episodes = len(set(i["episode"] for i in items))
        return len(items) * 0.5 + n_tasks * 3.0 + n_episodes * 1.0

    sorted_skills = sorted(
        skill_groups.items(),
        key=lambda x: skill_diversity_score(x[1]),
        reverse=True
    )

    # 确保有足够的 skill
    if len(sorted_skills) < n_skills:
        print(f"\n警告: 只有 {len(sorted_skills)} 个 skill 类型，少于要求的 {n_skills}")
        selected_skills = sorted_skills
    else:
        # 前 80% 按分数选，后 20% 随机选（保证多样性）
        n_top = int(n_skills * 0.8)
        n_random = n_skills - n_top

        selected_skills = sorted_skills[:n_top]
        remaining = sorted_skills[n_top:]

        # 随机打乱剩余部分
        random.shuffle(remaining)
        selected_skills.extend(remaining[:n_random])

    # 3. 为每个 skill 选择样本
    final_items = []
    skill_summaries = []

    for skill_desc, items in selected_skills[:n_skills]:
        # 选择分散的样本
        selected = select_diverse_samples(items, n_samples_per_skill)

        skill_summary = {
            "skill_description": skill_desc,
            "total_available": len(items),
            "n_tasks": len(set(i["task_index"] for i in items)),
            "n_episodes": len(set(i["episode"] for i in items)),
            "selected_count": len(selected),
        }
        skill_summaries.append(skill_summary)

        for item in selected:
            # 添加格式化后的 prompt
            item["formatted_prompt"] = format_prompt(item)

            # 构建 exec_config（执行时需要的配置）
            item["exec_config"] = {
                "snapshot_dir": item["snapshot_dir"],
                "prompt": item["formatted_prompt"],
                "task_index": item["task_index"],
                "episode": item["episode"],
                "skill_idx": item["skill_idx"],
            }

            final_items.append(item)

    # 4. 构建最终输出
    output = {
        "metadata": {
            "generated_at": str(Path(__file__).stat().st_mtime),
            "snapshots_root": str(SNAPSHOTS_ROOT),
            "n_skills": len(selected_skills[:n_skills]),
            "n_samples_per_skill": n_samples_per_skill,
            "total_samples": len(final_items),
            "random_seed": RANDOM_SEED,
        },
        "skill_summaries": skill_summaries,
        "items": final_items,
    }

    return output


def main() -> int:
    random.seed(RANDOM_SEED)

    # 生成列表
    print("=" * 60)
    print("生成批量评测任务列表")
    print("=" * 60)

    batch_data = generate_batch_list(n_skills=34, n_samples_per_skill=4)

    # 确保输出目录存在
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 保存 JSON
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(batch_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"生成完成！")
    print(f"输出文件: {OUTPUT_PATH}")
    print(f"总计: {batch_data['metadata']['n_skills']} 个 skills × "
          f"{batch_data['metadata']['n_samples_per_skill']} 个样本 = "
          f"{batch_data['metadata']['total_samples']} 条评测任务")
    print(f"{'=' * 60}")

    # 打印摘要
    print("\n选择的 Skill 摘要:")
    for summary in batch_data["skill_summaries"]:
        print(f"  - {summary['skill_description']!r}")
        print(f"    从 {summary['total_available']} 可用实例中选 {summary['selected_count']} 个")
        print(f"    覆盖 {summary['n_tasks']} tasks, {summary['n_episodes']} episodes")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
