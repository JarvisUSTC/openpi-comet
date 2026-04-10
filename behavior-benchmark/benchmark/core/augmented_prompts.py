from __future__ import annotations

from pathlib import Path

from .io import read_json


def load_augmented_prompt_choices(
    annotations_root: Path,
    *,
    task_index: int,
    episode_name: str,
    skill_info: dict,
) -> list[str]:
    annotation_path = annotations_root / f"task-{task_index:04d}" / f"{episode_name}.json"
    if not annotation_path.is_file():
        raise FileNotFoundError(f"未找到增广标注文件: {annotation_path}")

    annotation = read_json(annotation_path)
    skill_annotations = annotation.get("skill_annotation")
    if not isinstance(skill_annotations, list):
        raise ValueError(f"增广标注格式不正确，缺少 `skill_annotation`: {annotation_path}")

    target_skill_idx = int(skill_info["skill_idx"])
    matched_skill = next(
        (item for item in skill_annotations if int(item.get("skill_idx", -1)) == target_skill_idx),
        None,
    )
    if matched_skill is None:
        raise ValueError(f"增广标注中未找到 skill_{target_skill_idx:02d}: {annotation_path}")

    prompt_choices: list[str] = []
    seen_prompts: set[str] = set()
    for candidate in matched_skill.get("augmented_subtask", []) or []:
        text = str(candidate).strip()
        if not text or text in seen_prompts:
            continue
        seen_prompts.add(text)
        prompt_choices.append(text)

    if not prompt_choices:
        raise ValueError(f"skill_{target_skill_idx:02d} 没有可用的 augmented_subtask: {annotation_path}")
    return prompt_choices
