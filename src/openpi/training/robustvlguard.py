from __future__ import annotations

import json
import pathlib
from collections.abc import Iterable
from typing import Any

import openpi.training.vqa_schema as vqa_schema

DEFAULT_TASK_FAMILY = "robustvlguard"
DEFAULT_TASK_NAME = "comprehensive_4k"
DEFAULT_SOURCE = "RobustVLGuard"
DEFAULT_OUTPUT_FILENAME = "comprehensive_4k_openpi_schema.jsonl"


def convert_record(
    record: dict[str, Any],
    *,
    image_root: pathlib.Path,
    annotation_path: pathlib.Path,
    task_family: str = DEFAULT_TASK_FAMILY,
    task_name: str = DEFAULT_TASK_NAME,
    source: str = DEFAULT_SOURCE,
    validate_image: bool = True,
) -> vqa_schema.VQASample:
    raw_image = record.get("image")
    if not raw_image:
        raise ValueError(f"RobustVLGuard record {record.get('id')} is missing image.")

    prompt, answer = vqa_schema.conversations_to_prompt_answer(record["conversations"])
    image_path = (image_root / str(raw_image)).resolve()
    if validate_image and not image_path.is_file():
        raise FileNotFoundError(f"Image not found for record {record.get('id')}: {image_path}")

    return vqa_schema.VQASample(
        sample_id=str(record.get("id", image_path.stem)),
        prompt=prompt,
        answer=answer,
        images=(vqa_schema.VQAImageRef(storage="path", path=str(image_path)),),
        task_family=task_family,
        task_name=task_name,
        source=source,
        metadata={
            "annotation_path": str(annotation_path),
            "raw_image": str(raw_image),
        },
    )


def iter_converted_samples(
    input_path: pathlib.Path,
    *,
    image_root: pathlib.Path | None = None,
    task_family: str = DEFAULT_TASK_FAMILY,
    task_name: str = DEFAULT_TASK_NAME,
    source: str = DEFAULT_SOURCE,
    validate_images: bool = True,
) -> Iterable[vqa_schema.VQASample]:
    resolved_image_root = input_path.parent if image_root is None else pathlib.Path(image_root)
    with input_path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                yield convert_record(
                    record,
                    image_root=resolved_image_root,
                    annotation_path=input_path,
                    task_family=task_family,
                    task_name=task_name,
                    source=source,
                    validate_image=validate_images,
                )
            except Exception as exc:
                raise type(exc)(f"{exc} (input={input_path}, line={line_number})") from exc


def convert_jsonl(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    *,
    image_root: pathlib.Path | None = None,
    task_family: str = DEFAULT_TASK_FAMILY,
    task_name: str = DEFAULT_TASK_NAME,
    source: str = DEFAULT_SOURCE,
    validate_images: bool = True,
) -> int:
    samples = list(
        iter_converted_samples(
            input_path,
            image_root=image_root,
            task_family=task_family,
            task_name=task_name,
            source=source,
            validate_images=validate_images,
        )
    )
    vqa_schema.dump_samples_to_jsonl(samples, output_path)
    return len(samples)
