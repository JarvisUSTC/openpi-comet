#!/usr/bin/env python3
"""
Generate orchestrator directories from existing annotations (skill_annotation).
Use when your dataset has annotations/ with episode_*.json containing skill_annotation
but no orchestrators/ directory. Creates orchestrators/task-XXXX/episode_YYYY/ with
task_annotated.json and subtask_*_annotated.json so that the dataloader can load
from disk, or you can keep using the in-memory path (dataloader now builds level 1/2
from annotations automatically).

Usage:
  uv run scripts/generate_orchestrators_from_annotations.py --root /path/to/behavior/dataset
  # e.g. --root ~/Training/DATASETS/behavior/2025-challenge-demos
"""
import argparse
import json
from pathlib import Path

from lerobot.datasets.utils import load_jsonlines

from behavior.learning.datas.dataset import (
    ANNOTATIONS_PATH,
    ORCHESTRATORS_PATH,
    build_orchestrator_levels_from_annotations,
)
from lerobot.datasets.utils import EPISODES_PATH, TASKS_PATH


def load_meta(root: Path):
    tasks = load_jsonlines(root / TASKS_PATH)
    task_names = {item["task_index"]: item["task_name"] for item in sorted(tasks, key=lambda x: x["task_index"])}
    tasks_dict = {item["task_index"]: item["task"] for item in sorted(tasks, key=lambda x: x["task_index"])}
    episodes = load_jsonlines(root / EPISODES_PATH)
    episodes_dict = {
        item["episode_index"]: item
        for item in sorted(episodes, key=lambda x: x["episode_index"])
        if item["tasks"][0] in tasks_dict
    }
    annotations_dir = root / ANNOTATIONS_PATH
    annotations = {}
    if annotations_dir.exists():
        for task_dir in sorted(annotations_dir.iterdir()):
            if not task_dir.is_dir() or not task_dir.name.startswith("task-"):
                continue
            task_id = int(task_dir.name[5:])
            if task_id not in tasks_dict:
                continue
            for ep_file in sorted(task_dir.glob("episode_*.json")):
                ep_id = int(ep_file.stem[8:])
                with open(ep_file) as f:
                    annotations[ep_id] = json.load(f)
    return tasks_dict, episodes_dict, annotations


def main():
    parser = argparse.ArgumentParser(description="Generate orchestrators from annotations")
    parser.add_argument("--root", type=Path, required=True, help="Behavior dataset root (e.g. .../2025-challenge-demos)")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be written")
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")

    tasks_dict, episodes_dict, annotations = load_meta(root)
    out_dir = root / ORCHESTRATORS_PATH
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for ep_id, ep_data in episodes_dict.items():
        ann = annotations.get(ep_id)
        if not ann or not ann.get("skill_annotation"):
            continue
        task_idx = ep_data["tasks"][0]
        ep_len = ep_data["length"]
        level_0_task = tasks_dict.get(task_idx, "task")
        orch = build_orchestrator_levels_from_annotations(ep_id, ep_len, ann["skill_annotation"], level_0_task)
        task_id_str = f"task-{task_idx:04d}"
        ep_dir = out_dir / task_id_str / f"episode_{ep_id:08d}"
        if not args.dry_run:
            ep_dir.mkdir(parents=True, exist_ok=True)
        task_annotated = {
            "cot_task_description": level_0_task,
            "cot_subtask_description_list": [seg["task"] for seg in orch[1]],
        }
        if not args.dry_run:
            with open(ep_dir / "task_annotated.json", "w") as f:
                json.dump(task_annotated, f, indent=2)
        for i, seg in enumerate(orch[1]):
            raw_desc = ann["skill_annotation"][i].get("skill_description")
            if isinstance(raw_desc, list):
                raw_desc = (raw_desc or [""])[0]
            else:
                raw_desc = str(raw_desc) if raw_desc else ""
            subtask_annotated = {
                "cot_subtask_description": seg["task"],
                "start_frame": seg["start_frame"],
                "end_frame": seg["end_frame"] + 1,
            }
            subtask_annotated["skill_description"] = raw_desc
            if not args.dry_run:
                with open(ep_dir / f"subtask_{i}_annotated.json", "w") as f:
                    json.dump(subtask_annotated, f, indent=2)
        written += 1
        if written <= 2:
            print(f"Example: {ep_dir} level_0={level_0_task!r} num_skills={len(orch[1])}")

    print(f"Wrote orchestrators for {written} episodes under {out_dir}")


if __name__ == "__main__":
    main()
