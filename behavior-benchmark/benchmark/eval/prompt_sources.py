from __future__ import annotations

from pathlib import Path

from benchmark.core.augmented_prompts import load_augmented_prompt_choices


def load_augmented_prompts(
    annotations_root: Path,
    *,
    task_index: int,
    episode_name: str,
    skill_info: dict,
) -> list[str]:
    return load_augmented_prompt_choices(
        annotations_root,
        task_index=task_index,
        episode_name=episode_name,
        skill_info=skill_info,
    )
