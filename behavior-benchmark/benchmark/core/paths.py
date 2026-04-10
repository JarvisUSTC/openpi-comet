from __future__ import annotations

import re
from pathlib import Path


def task_dir_name(task_index: int) -> str:
    return f"task_{int(task_index):04d}"


def skill_dir_name(skill_idx: int) -> str:
    return f"skill_{int(skill_idx):02d}"


def build_result_filename(episode: str, skill_idx: int, suffix: str = "*") -> str:
    return f"result_{episode}_{int(skill_idx):02d}{suffix}.json"


def build_result_glob(episode: str, skill_idx: int) -> str:
    return build_result_filename(episode, skill_idx, "*")


def build_judge_output_path(source_result_path: Path | None, video_path: Path) -> Path:
    if source_result_path is not None:
        name = source_result_path.name
        if name.startswith("result_"):
            return source_result_path.with_name("judge_" + name)
        return source_result_path.with_name(f"judge_{source_result_path.stem}.json")
    return video_path.with_name(f"judge_{video_path.stem}.json")


def parse_video_metadata_from_name(video_path: str | Path) -> tuple[str | None, int | None]:
    name = Path(video_path).name
    episode_match = re.search(r"(episode_\d+)", name)
    skill_match = re.search(r"_s(\d+)_", name)
    episode = episode_match.group(1) if episode_match else None
    skill_idx = int(skill_match.group(1)) if skill_match else None
    return episode, skill_idx
