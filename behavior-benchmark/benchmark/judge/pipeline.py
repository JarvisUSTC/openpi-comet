from __future__ import annotations

import argparse
import base64
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import httpx

from benchmark.config.settings import get_settings
from benchmark.core.io import read_json, write_json
from benchmark.core.paths import build_judge_output_path, parse_video_metadata_from_name
from benchmark.core.schemas import JudgeInput
from .prompts import build_system_prompt, build_user_text

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_API_KEY_ENV = "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"
MODEL_MAP = {
    "seed-2.0-lite": "bytedance-seed/seed-2.0-lite",
    "qwen3-vl-8b-instruct": "qwen/qwen3-vl-8b-instruct",
    "qwen3-vl-30b-a3b-instruct": "qwen/qwen3-vl-30b-a3b-instruct",
    "qwen3-vl-235b-a22b-instruct": "qwen/qwen3-vl-235b-a22b-instruct",
    "gemini-2.5-flash-lite": "google/gemini-2.5-flash-lite",
}
TAIL_DENSE_WINDOW_SECONDS = 3.0
TAIL_DENSE_SAMPLE_SECONDS = 1.0


@dataclass(slots=True)
class JudgeMenuEntry:
    entry_kind: str
    label: str
    judge_input: JudgeInput
    output_path: Path
    existing_judge_success: bool | None
    existing_judge_path: Path | None


@dataclass(slots=True)
class FolderMenuEntry:
    entry_kind: str
    label: str
    dir_path: Path
    task_count: int
    result_count: int


def _read_result_input(path: Path) -> JudgeInput:
    data = read_json(path)
    video_path = Path(str(data["video_path"])).expanduser().resolve()
    prompt_used = str(data.get("prompt_used", "")).strip()
    if not prompt_used:
        raise ValueError(f"`prompt_used` is empty in {path}")
    return JudgeInput(
        source_result_path=path.resolve(),
        video_path=video_path,
        before_image_path=Path(str(data["before_image"])).expanduser().resolve() if data.get("before_image") else None,
        prompt_used=prompt_used,
        task_name=str(data.get("task_name", "")).strip() or None,
        task_prompt=str(data.get("task_prompt", "")).strip() or None,
        task_index=int(data["task_index"]) if data.get("task_index") is not None else None,
        episode=str(data.get("episode", "")).strip() or None,
        skill_idx=int(data["skill_idx"]) if data.get("skill_idx") is not None else None,
        skill_description=str(data.get("skill_description", "")).strip() or None,
        object_ids=[str(x) for x in data.get("object_ids", [])],
        manipulating_object_ids=[str(x) for x in data.get("manipulating_object_ids", [])],
        max_steps=int(data["max_steps"]) if data.get("max_steps") is not None else None,
        steps=int(data["steps"]) if data.get("steps") is not None else None,
    )


def _available_video_paths(skill_dir: Path) -> list[Path]:
    return [path.resolve() for path in sorted(skill_dir.glob("*.mp4")) if path.is_file()]


def _resolve_result_video_path(item: JudgeInput, available_video_paths: list[Path] | None = None) -> Path | None:
    expected_path = item.video_path.expanduser().resolve()
    if expected_path.is_file():
        return expected_path

    if item.source_result_path is None:
        return expected_path if expected_path.is_file() else None

    candidate_paths = available_video_paths
    if candidate_paths is None:
        candidate_paths = _available_video_paths(item.source_result_path.parent)

    if item.episode is not None and item.skill_idx is not None:
        matched = [
            path
            for path in candidate_paths
            if parse_video_metadata_from_name(path) == (item.episode, item.skill_idx)
        ]
        if len(matched) == 1:
            return matched[0]
        if len(matched) > 1:
            return None

    if item.episode is not None:
        matched = [path for path in candidate_paths if parse_video_metadata_from_name(path)[0] == item.episode]
        if len(matched) == 1:
            return matched[0]
        if len(matched) > 1:
            return None

    return None


def _prepare_result_input(path: Path, available_video_paths: list[Path] | None = None) -> JudgeInput | None:
    item = _read_result_input(path)
    resolved_video_path = _resolve_result_video_path(item, available_video_paths)
    if resolved_video_path is None:
        return None
    item.video_path = resolved_video_path
    return item


def _gather_inputs(args: argparse.Namespace) -> list[JudgeInput]:
    inputs: list[JudgeInput] = []

    for result_json in args.result_jsons:
        result_path = result_json.expanduser().resolve()
        item = _prepare_result_input(result_path)
        if item is not None:
            inputs.append(item)

    if args.video_path is not None:
        if not args.prompt:
            raise ValueError("Using --video-path requires --prompt.")
        inputs.append(
            JudgeInput(
                source_result_path=None,
                video_path=args.video_path.expanduser().resolve(),
                before_image_path=None,
                prompt_used=args.prompt.strip(),
                task_name=args.task_name,
                task_prompt=None,
                task_index=args.task_index,
                episode=args.episode,
                skill_idx=args.skill_idx,
                skill_description=None,
                object_ids=None,
                manipulating_object_ids=None,
                max_steps=args.max_steps,
                steps=args.steps,
            )
        )

    if args.log_path is not None:
        log_root = args.log_path.expanduser().resolve()
        result_paths = sorted(log_root.rglob("result_*.json"))
        for result_path in result_paths:
            item = _prepare_result_input(result_path)
            if item is not None:
                inputs.append(item)

    deduped: list[JudgeInput] = []
    seen_keys: set[tuple[str, str | None]] = set()
    for item in inputs:
        key = (str(item.video_path), str(item.source_result_path) if item.source_result_path else None)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(item)

    if args.limit is not None:
        deduped = deduped[: args.limit]

    if not deduped:
        raise ValueError("No inputs found. Provide --result-json, --video-path + --prompt, or --log-path.")
    return deduped


def _video_metadata(video_path: Path) -> tuple[float, float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    if fps > 0 and frame_count > 0:
        duration = frame_count / fps
    else:
        duration = 0.0
    return fps, duration, frame_count


def _compute_sample_timestamps(
    duration: float,
    sample_every_seconds: float,
    max_frames: int | None,
) -> list[float]:
    if sample_every_seconds <= 0:
        raise ValueError("--sample-every-seconds must be > 0.")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("--max-frames must be > 0.")

    if duration <= 0:
        return [0.0]

    timestamps = [0.0]
    current = sample_every_seconds
    while current < duration:
        timestamps.append(round(current, 3))
        current += sample_every_seconds

    tail_start = max(duration - TAIL_DENSE_WINDOW_SECONDS, 0.0)
    tail_current = tail_start
    while tail_current < duration:
        timestamps.append(round(tail_current, 3))
        tail_current += TAIL_DENSE_SAMPLE_SECONDS

    last_timestamp = max(duration - 1e-3, 0.0)
    if not timestamps or abs(timestamps[-1] - last_timestamp) > 1e-2:
        timestamps.append(round(last_timestamp, 3))

    timestamps = sorted({round(ts, 3) for ts in timestamps})
    if max_frames is None or len(timestamps) <= max_frames:
        return timestamps

    tail_dense_timestamps = {
        round(ts, 3)
        for ts in timestamps
        if ts >= round(max(duration - TAIL_DENSE_WINDOW_SECONDS, 0.0), 3)
    }
    keep_indices = {
        idx
        for idx, timestamp in enumerate(timestamps)
        if idx == 0 or idx == len(timestamps) - 1 or timestamp in tail_dense_timestamps
    }
    if len(keep_indices) >= max_frames:
        if 0 in keep_indices and max_frames > 1:
            nonzero_indices = [idx for idx in sorted(keep_indices) if idx != 0]
            keep_indices = {0, *nonzero_indices[-(max_frames - 1) :]}
        else:
            keep_indices = set(sorted(keep_indices)[-max_frames:])
        return [timestamps[idx] for idx in sorted(keep_indices)]

    interior_needed = max_frames - len(keep_indices)
    if interior_needed > 0:
        for idx in range(1, interior_needed + 1):
            pos = idx * (len(timestamps) - 1) / (interior_needed + 1)
            keep_indices.add(int(round(pos)))
    return [timestamps[idx] for idx in sorted(keep_indices)]


def _sample_video_frames(video_path: Path, timestamps: list[float]) -> list[dict[str, Any]]:
    meta_cap = cv2.VideoCapture(str(video_path))
    if not meta_cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    fps = float(meta_cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(meta_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    meta_cap.release()

    sampled: list[dict[str, Any]] = []
    for timestamp in timestamps:
        frame = None
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to reopen video: {video_path}")

        if fps > 0 and frame_count > 0:
            frame_index = min(max(int(round(timestamp * fps)), 0), frame_count - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, image = cap.read()
            if ok and image is not None:
                frame = image

        if frame is None:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(timestamp, 0.0) * 1000.0)
            ok, image = cap.read()
            if ok and image is not None:
                frame = image

        cap.release()

        if frame is None:
            raise ValueError(f"Failed to decode frame at {timestamp:.3f}s from {video_path}")

        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            raise ValueError(f"Failed to encode sampled frame at {timestamp:.3f}s from {video_path}")
        image_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
        sampled.append(
            {
                "timestamp_sec": round(timestamp, 3),
                "image_base64": image_b64,
            }
        )
    return sampled


def _encode_reference_image(image_path: Path | None) -> dict[str, Any] | None:
    if image_path is None:
        return None
    resolved_path = image_path.expanduser().resolve()
    if not resolved_path.is_file():
        return None

    image = cv2.imread(str(resolved_path))
    if image is None:
        return None

    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        return None

    return {
        "image_path": str(resolved_path),
        "image_base64": base64.b64encode(encoded.tobytes()).decode("ascii"),
    }


def _extract_text_content(message_content: Any) -> str:
    if isinstance(message_content, str):
        return message_content
    if isinstance(message_content, list):
        parts: list[str] = []
        for chunk in message_content:
            if isinstance(chunk, dict) and chunk.get("type") == "text":
                parts.append(str(chunk.get("text", "")))
        return "\n".join(parts).strip()
    return str(message_content)


def _extract_json_payload(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1).strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not match:
            raise ValueError(f"Model response does not contain JSON: {text}")
        return json.loads(match.group(0))


def _resolve_model_name(model_name: str) -> str:
    normalized = model_name.strip()
    return MODEL_MAP.get(normalized, normalized)


def _coerce_optional_bool(value: Any) -> bool | None:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "success", "successful"}:
            return True
        if lowered in {"false", "no", "fail", "failed"}:
            return False
        return None
    if isinstance(value, bool):
        return value
    return None


def _normalize_judge_response(
    payload: dict[str, Any],
) -> tuple[bool | None, float | None, str, dict[str, bool | None] | None]:
    success_value = payload.get("success", payload.get("judge_success"))
    confidence_value = payload.get("confidence", payload.get("judge_confidence"))
    reason = str(payload.get("reason", payload.get("judge_reason", ""))).strip()
    success_value = _coerce_optional_bool(success_value)

    if confidence_value is not None:
        try:
            confidence_value = float(confidence_value)
        except (TypeError, ValueError):
            confidence_value = None

    checklist_value = payload.get("checklist")
    checklist: dict[str, bool | None] | None = None
    if isinstance(checklist_value, dict):
        checklist = {
            "grounded_target_tracked": _coerce_optional_bool(
                checklist_value.get("grounded_target_tracked")
            ),
            "final_relation_satisfied": _coerce_optional_bool(
                checklist_value.get("final_relation_satisfied")
            ),
            "gripper_state_consistent": _coerce_optional_bool(
                checklist_value.get("gripper_state_consistent")
            ),
            "object_state_stable": _coerce_optional_bool(
                checklist_value.get("object_state_stable")
            ),
            "evidence_sufficient": _coerce_optional_bool(
                checklist_value.get("evidence_sufficient")
            ),
        }
        all_present = all(value is not None for value in checklist.values())
        if any(value is False for value in checklist.values()):
            success_value = False
        elif success_value is None and all_present:
            success_value = True

    return success_value, confidence_value, reason, checklist


def _call_openai_compatible_api(
    *,
    base_url: str,
    api_key: str,
    model: str,
    timeout_seconds: float,
    prompt_text: str,
    reference_image: dict[str, Any] | None,
    sampled_frames: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
) -> tuple[str, dict[str, Any]]:
    model = _resolve_model_name(model)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if "openrouter.ai" in base_url:
        headers["HTTP-Referer"] = "https://cursor.sh"
        headers["X-Title"] = "SimpleRoboAgent VLM Judge"
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
    if reference_image is not None:
        content.append(
            {
                "type": "text",
                "text": "Reference image before rollout: use this image only to identify the exact grounded object instance or destination anchor if multiple similar objects are visible.",
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{reference_image['image_base64']}",
                },
            }
        )
    for idx, frame in enumerate(sampled_frames, start=1):
        content.append(
            {
                "type": "text",
                "text": f"Frame {idx} at {frame['timestamp_sec']:.3f}s",
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame['image_base64']}",
                },
            }
        )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": content},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    chat_url = base_url.rstrip("/") + "/chat/completions"
    with httpx.Client(timeout=timeout_seconds) as client:
        response = client.post(chat_url, json=payload, headers=headers)
        response.raise_for_status()
        raw_response = response.json()

    message = raw_response["choices"][0]["message"]["content"]
    return _extract_text_content(message), raw_response


def _judge_one(
    item: JudgeInput,
    *,
    sample_every_seconds: float,
    max_frames: int | None,
    dry_run: bool,
    model: str,
    base_url: str | None,
    api_key_env: str,
    timeout_seconds: float,
    api_max_tokens: int,
    api_temperature: float,
) -> dict[str, Any]:
    if not item.video_path.is_file():
        raise FileNotFoundError(f"Video not found: {item.video_path}")

    fps, duration, frame_count = _video_metadata(item.video_path)
    timestamps = _compute_sample_timestamps(duration, sample_every_seconds, max_frames)
    sampled_frames = _sample_video_frames(item.video_path, timestamps)
    reference_image = _encode_reference_image(item.before_image_path)

    result: dict[str, Any] = {
        "source_result_path": str(item.source_result_path) if item.source_result_path else None,
        "video_path": str(item.video_path),
        "before_image_path": str(item.before_image_path) if item.before_image_path else None,
        "before_image_used": reference_image is not None,
        "prompt_used": item.prompt_used,
        "task_prompt": item.task_prompt,
        "task_name": item.task_name,
        "task_index": item.task_index,
        "episode": item.episode,
        "skill_idx": item.skill_idx,
        "steps": item.steps,
        "max_steps": item.max_steps,
        "sample_every_seconds": sample_every_seconds,
        "max_frames": max_frames,
        "tail_dense_sample_every_seconds": TAIL_DENSE_SAMPLE_SECONDS,
        "sample_timestamps": [frame["timestamp_sec"] for frame in sampled_frames],
        "sampled_frame_count": len(sampled_frames),
        "video_duration_sec": round(duration, 3),
        "video_fps": round(fps, 3),
        "video_frame_count": frame_count,
        "model": _resolve_model_name(model),
        "judge_success": None,
        "judge_confidence": None,
        "judge_reason": None,
        "judge_checklist": None,
        "raw_response": None,
        "judged_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
    }

    if dry_run:
        return result

    if not base_url:
        raise ValueError("Real judge mode requires --base-url.")
    api_key = os.getenv(api_key_env, "").strip()
    if not api_key:
        raise ValueError(f"Environment variable {api_key_env} is not set.")

    prompt_text = build_user_text(item, sampled_frames)
    response_text, raw_response = _call_openai_compatible_api(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout_seconds=timeout_seconds,
        prompt_text=prompt_text,
        reference_image=reference_image,
        sampled_frames=sampled_frames,
        max_tokens=api_max_tokens,
        temperature=api_temperature,
    )
    payload = _extract_json_payload(response_text)
    judge_success, judge_confidence, judge_reason, judge_checklist = _normalize_judge_response(
        payload
    )

    result["judge_success"] = judge_success
    result["judge_confidence"] = judge_confidence
    result["judge_reason"] = judge_reason
    result["judge_checklist"] = judge_checklist
    result["raw_response"] = raw_response
    return result


def _print_item_summary(judge_result: dict[str, Any]) -> None:
    skill_label = judge_result.get("skill_idx")
    if skill_label is None:
        title = Path(str(judge_result["video_path"])).name
    else:
        title = f"skill_{int(skill_label):02d}"
    status = judge_result["judge_success"]
    print(
        f"{title}: success={status} "
        f"frames={judge_result['sampled_frame_count']} "
        f"prompt={judge_result['prompt_used']}"
    )


def _load_existing_judge_success(output_path: Path) -> bool | None:
    if not output_path.is_file():
        return None
    try:
        data = read_json(output_path)
    except Exception:
        return None
    value = data.get("judge_success")
    return value if isinstance(value, bool) else None


def _collect_task_infos(log_root: Path) -> list[dict[str, Any]]:
    task_infos: list[dict[str, Any]] = []
    for task_dir in sorted(log_root.glob("task_*")):
        if not task_dir.is_dir():
            continue
        skill_dirs = [path for path in sorted(task_dir.glob("skill_*")) if path.is_dir()]
        if not skill_dirs:
            continue
        try:
            task_index = int(task_dir.name.split("_")[-1])
        except ValueError:
            task_index = -1
        entry_count = 0
        for skill_dir in skill_dirs:
            entry_count += len(list(skill_dir.glob("result_*.json")))
        task_infos.append(
            {
                "task_dir": task_dir,
                "task_name": task_dir.name,
                "task_index": task_index,
                "skill_count": len(skill_dirs),
                "entry_count": entry_count,
            }
        )
    return task_infos


def _has_judgeable_results(task_dir: Path) -> bool:
    if not task_dir.is_dir() or not task_dir.name.startswith("task_"):
        return False
    return any(task_dir.rglob("result_*.json"))


def _count_task_results(task_dir: Path) -> int:
    return len(list(task_dir.rglob("result_*.json")))


@lru_cache(maxsize=None)
def _count_subtree_stats_cached(dir_path_text: str) -> tuple[int, int]:
    dir_path = Path(dir_path_text)
    task_count = 0
    result_count = 0
    for task_dir in sorted(dir_path.rglob("task_*")):
        if not _has_judgeable_results(task_dir):
            continue
        task_count += 1
        result_count += _count_task_results(task_dir)
    return task_count, result_count


def _collect_folder_menu_entries(current_dir: Path) -> list[FolderMenuEntry]:
    entries: list[FolderMenuEntry] = []

    direct_task_dirs = [
        task_dir
        for task_dir in sorted(current_dir.glob("task_*"))
        if _has_judgeable_results(task_dir)
    ]
    if direct_task_dirs:
        entries.append(
            FolderMenuEntry(
                entry_kind="use_current",
                label=f"使用当前目录 `{current_dir.name}`",
                dir_path=current_dir.resolve(),
                task_count=len(direct_task_dirs),
                result_count=sum(_count_task_results(task_dir) for task_dir in direct_task_dirs),
            )
        )

    for child_dir in sorted(current_dir.iterdir()):
        if not child_dir.is_dir() or child_dir.name.startswith(".") or child_dir.name.startswith("task_"):
            continue
        task_count, result_count = _count_subtree_stats_cached(str(child_dir.resolve()))
        if task_count <= 0:
            continue
        entries.append(
            FolderMenuEntry(
                entry_kind="folder",
                label=child_dir.name,
                dir_path=child_dir.resolve(),
                task_count=task_count,
                result_count=result_count,
            )
        )
    return entries


def _collect_skill_infos(task_dir: Path) -> list[dict[str, Any]]:
    skill_infos: list[dict[str, Any]] = []
    for skill_dir in sorted(task_dir.glob("skill_*")):
        if not skill_dir.is_dir():
            continue
        result_count = len(list(skill_dir.glob("result_*.json")))
        video_count = len(list(skill_dir.glob("*.mp4")))
        if result_count <= 0:
            continue
        try:
            skill_idx = int(skill_dir.name.split("_")[-1])
        except ValueError:
            skill_idx = -1
        prompt_preview = ""
        sample_result = next(iter(sorted(skill_dir.glob("result_*.json"))), None)
        if sample_result is not None:
            try:
                prompt_preview = str(read_json(sample_result).get("prompt_used", "")).strip()
            except Exception:
                prompt_preview = ""
        skill_infos.append(
            {
                "skill_dir": skill_dir,
                "skill_idx": skill_idx,
                "result_count": result_count,
                "video_count": video_count,
                "prompt_preview": prompt_preview,
            }
        )
    return skill_infos


def _build_video_only_input(video_path: Path, result_inputs: list[JudgeInput]) -> JudgeInput | None:
    episode, skill_idx = parse_video_metadata_from_name(video_path)
    match: JudgeInput | None = None
    if episode is not None and skill_idx is not None:
        match = next(
            (
                item
                for item in result_inputs
                if item.episode == episode and item.skill_idx == skill_idx
            ),
            None,
        )
    if match is None and episode is not None:
        match = next((item for item in result_inputs if item.episode == episode), None)
    if match is None and result_inputs:
        match = result_inputs[0]
    if match is None or not match.prompt_used:
        return None
    return JudgeInput(
        source_result_path=None,
        video_path=video_path.resolve(),
        before_image_path=match.before_image_path,
        prompt_used=match.prompt_used,
        task_name=match.task_name,
        task_prompt=match.task_prompt,
        task_index=match.task_index,
        episode=episode or match.episode,
        skill_idx=skill_idx if skill_idx is not None else match.skill_idx,
        skill_description=match.skill_description,
        object_ids=match.object_ids,
        manipulating_object_ids=match.manipulating_object_ids,
        max_steps=match.max_steps,
        steps=match.steps,
    )


def _collect_menu_entries(skill_dir: Path) -> list[JudgeMenuEntry]:
    entries: list[JudgeMenuEntry] = []
    available_video_paths = _available_video_paths(skill_dir)

    for result_path in sorted(skill_dir.glob("result_*.json")):
        try:
            judge_input = _prepare_result_input(result_path, available_video_paths)
        except Exception:
            continue
        if judge_input is None:
            continue
        output_path = build_judge_output_path(judge_input.source_result_path, judge_input.video_path)
        existing_success = _load_existing_judge_success(output_path)
        entries.append(
            JudgeMenuEntry(
                entry_kind="result",
                label=result_path.name,
                judge_input=judge_input,
                output_path=output_path,
                existing_judge_success=existing_success,
                existing_judge_path=output_path if output_path.exists() else None,
            )
        )

    entries.sort(
        key=lambda entry: (
            int(entry.judge_input.skill_idx or -1),
            str(entry.judge_input.episode or ""),
            str(entry.label),
        )
    )
    return entries


def _collect_task_inputs(task_dir: Path) -> list[JudgeInput]:
    inputs: list[JudgeInput] = []
    seen: set[tuple[str, str | None]] = set()
    for skill_info in _collect_skill_infos(task_dir):
        for entry in _collect_menu_entries(skill_info["skill_dir"]):
            item = entry.judge_input
            key = (
                str(item.video_path.resolve()),
                str(item.source_result_path.resolve()) if item.source_result_path is not None else None,
            )
            if key in seen:
                continue
            seen.add(key)
            inputs.append(item)
    return inputs


def _resolve_index_choices(raw: str, items: list[Any]) -> list[int] | None:
    tokens: list[str] = []
    for chunk in raw.replace("，", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        tokens.extend(part for part in chunk.split() if part)
    if not tokens:
        return None

    indices: list[int] = []
    seen: set[int] = set()
    for token in tokens:
        if "-" in token and token.count("-") == 1:
            left, right = token.split("-", 1)
            if left.isdigit() and right.isdigit():
                start_idx = int(left)
                end_idx = int(right)
                if start_idx <= 0 or end_idx <= 0 or start_idx > end_idx:
                    return None
                for menu_idx in range(start_idx, end_idx + 1):
                    if 1 <= menu_idx <= len(items):
                        idx0 = menu_idx - 1
                        if idx0 not in seen:
                            seen.add(idx0)
                            indices.append(idx0)
                    else:
                        return None
                continue

        if token.isdigit():
            menu_idx = int(token)
            if 1 <= menu_idx <= len(items):
                idx0 = menu_idx - 1
                if idx0 not in seen:
                    seen.add(idx0)
                    indices.append(idx0)
                continue

        return None
    return indices


def _prompt_task_choice(task_infos: list[dict[str, Any]]) -> dict[str, Any] | None:
    raw = input("> ").strip()
    if not raw:
        return None
    lowered = raw.lower()
    if lowered in {"q", "quit", "exit"}:
        raise SystemExit(0)
    if lowered in {"r", "refresh"}:
        return {"__action__": "refresh"}
    if raw.isdigit():
        idx = int(raw)
        if 1 <= idx <= len(task_infos):
            return task_infos[idx - 1]
    print("无效的 task 输入，请输入菜单编号。")
    return None


def _prompt_folder_choice(folder_infos: list[FolderMenuEntry], *, allow_back: bool) -> FolderMenuEntry | dict[str, Any] | None:
    raw = input("> ").strip()
    if not raw:
        return None
    lowered = raw.lower()
    if lowered in {"q", "quit", "exit"}:
        raise SystemExit(0)
    if lowered in {"b", "back"} and allow_back:
        return {"__action__": "back"}
    if lowered in {"r", "refresh"}:
        return {"__action__": "refresh"}
    if raw.isdigit():
        idx = int(raw)
        if 1 <= idx <= len(folder_infos):
            return folder_infos[idx - 1]
    print("无效的 folder 输入，请输入菜单编号。")
    return None


def _prompt_skill_choice(skill_infos: list[dict[str, Any]]) -> dict[str, Any] | None:
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
        if 1 <= idx <= len(skill_infos):
            return skill_infos[idx - 1]
    print("无效的 skill 输入，请输入菜单编号。")
    return None


def _prompt_entry_choices(entries: list[JudgeMenuEntry]) -> list[JudgeMenuEntry] | dict[str, Any] | None:
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

    indices = _resolve_index_choices(raw, entries)
    if indices is None:
        print("无效的 result json 输入，请输入菜单编号或编号范围。")
        return None
    return [entries[idx] for idx in indices]


def _print_task_menu(task_infos: list[dict[str, Any]]) -> None:
    print("\n可选 task:")
    for idx, task_info in enumerate(task_infos, start=1):
        print(
            f"  [{idx}] task_{int(task_info['task_index']):04d}  "
            f"(skills={task_info['skill_count']}, results={task_info['entry_count']})"
        )
    print("输入菜单编号；`r` 刷新，`q` 退出。")


def _print_folder_menu(folder_infos: list[FolderMenuEntry], *, search_root: Path, current_dir: Path) -> None:
    try:
        rel_path = str(current_dir.relative_to(search_root))
    except ValueError:
        rel_path = str(current_dir)
    if rel_path == ".":
        rel_path = current_dir.name

    print(f"\n可选 log folder（当前目录: {rel_path}）:")
    for idx, folder_info in enumerate(folder_infos, start=1):
        print(
            f"  [{idx}] {folder_info.label}  "
            f"(tasks={folder_info.task_count}, results={folder_info.result_count})"
        )
    if current_dir != search_root:
        print("输入菜单编号；`b` 返回上一层，`r` 刷新，`q` 退出。")
    else:
        print("输入菜单编号；`r` 刷新，`q` 退出。")


def _print_skill_menu(task_info: dict[str, Any], skill_infos: list[dict[str, Any]]) -> None:
    print(f"\nTask `{task_info['task_dir'].name}` 下的 skill:")
    for idx, skill_info in enumerate(skill_infos, start=1):
        prompt_preview = skill_info["prompt_preview"] or "-"
        if len(prompt_preview) > 60:
            prompt_preview = prompt_preview[:57] + "..."
        print(
            f"  [{idx}] skill_{int(skill_info['skill_idx']):02d}  "
            f"(results={skill_info['result_count']}, prompt={prompt_preview})"
        )
    print("输入菜单编号；`b` 返回，`r` 刷新，`q` 退出。")


def _print_entry_menu(skill_info: dict[str, Any], entries: list[JudgeMenuEntry]) -> None:
    print(f"\n`{skill_info['skill_dir'].name}` 下的 result json:")
    for idx, entry in enumerate(entries, start=1):
        judge_flag = (
            "-"
            if entry.existing_judge_success is None
            else ("success" if entry.existing_judge_success else "fail")
        )
        prompt_preview = entry.judge_input.prompt_used
        if len(prompt_preview) > 60:
            prompt_preview = prompt_preview[:57] + "..."
        episode = entry.judge_input.episode or "-"
        print(
            f"  [{idx}] {entry.label}  episode={episode}  judge={judge_flag}"
        )
        print(f"      prompt={prompt_preview}")
    print("输入菜单编号 / `1,3,5` / `1-4`；`b` 返回，`r` 刷新，`q` 退出。")


def _run_judge_batch(inputs: list[JudgeInput], args: argparse.Namespace) -> list[dict[str, Any]]:
    failures: list[str] = []
    summaries: list[dict[str, Any]] = []

    print(f"Found {len(inputs)} input(s) for VLM judge.")
    for idx, item in enumerate(inputs, start=1):
        print(f"\n[{idx}/{len(inputs)}] Processing {item.video_path}")
        try:
            judge_result = _judge_one(
                item,
                sample_every_seconds=args.sample_every_seconds,
                max_frames=args.max_frames,
                dry_run=args.dry_run,
                model=args.model,
                base_url=args.base_url,
                api_key_env=args.api_key_env,
                timeout_seconds=args.timeout_seconds,
                api_max_tokens=args.api_max_tokens,
                api_temperature=args.api_temperature,
            )
            output_path = build_judge_output_path(item.source_result_path, item.video_path)
            write_json(output_path, judge_result)
            print(f"  wrote: {output_path}")
            _print_item_summary(judge_result)
            summaries.append(
                {
                    "skill_idx": item.skill_idx,
                    "episode": item.episode or "-",
                    "video_name": item.video_path.name,
                    "judge_success": judge_result.get("judge_success"),
                    "output_path": str(output_path),
                }
            )
        except Exception as exc:
            failures.append(f"{item.video_path}: {exc}")
            print(f"  failed: {exc}")

    if summaries:
        print("\nJudge 结果汇总:")
        for summary in summaries:
            skill_text = "-" if summary["skill_idx"] is None else f"skill_{int(summary['skill_idx']):02d}"
            print(
                f"  {skill_text}  episode={summary['episode']}  "
                f"success={summary['judge_success']}  video={summary['video_name']}"
            )
            print(f"    judge_result: {summary['output_path']}")

    if failures:
        print("\nFailures:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)

    print("\nAll VLM judge runs completed successfully.")
    return summaries


def _confirm_and_run_batch(inputs: list[JudgeInput], args: argparse.Namespace) -> bool:
    if not inputs:
        print("没有找到可 judge 的条目。")
        return False
    print(f"\n本次将 judge {len(inputs)} 个条目。")
    confirm = input("按回车开始，输入 n 取消：").strip().lower()
    if confirm in {"n", "no"}:
        return False
    _run_judge_batch(inputs, args)
    return True


def _resolve_interactive_root(log_path: Path | None) -> Path:
    root = (log_path or get_settings().paths.logs_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"log path does not exist: {root}")
    return root


def _run_interactive(args: argparse.Namespace) -> None:
    root = _resolve_interactive_root(args.log_path)
    fixed_task_dir: Path | None = None
    fixed_skill_dir: Path | None = None
    fixed_log_root: Path | None = None
    folder_cursor: Path | None = None

    if root.name.startswith("skill_") and root.is_dir():
        fixed_skill_dir = root
        if root.parent.name.startswith("task_"):
            fixed_task_dir = root.parent
    elif root.name.startswith("task_") and root.is_dir():
        fixed_task_dir = root
        fixed_log_root = root.parent
    elif any(_has_judgeable_results(task_dir) for task_dir in root.glob("task_*")):
        fixed_log_root = root

    while True:
        if fixed_task_dir is not None:
            task_infos = [
                {
                    "task_dir": fixed_task_dir,
                    "task_name": fixed_task_dir.name,
                    "task_index": int(fixed_task_dir.name.split("_")[-1]) if fixed_task_dir.name.split("_")[-1].isdigit() else -1,
                    "skill_count": len([path for path in fixed_task_dir.glob("skill_*") if path.is_dir()]),
                    "entry_count": len(list(fixed_task_dir.rglob("result_*.json"))),
                }
            ]
            task_choice = task_infos[0]
        else:
            if fixed_log_root is None:
                if folder_cursor is None:
                    folder_cursor = root
                folder_infos = _collect_folder_menu_entries(folder_cursor)
                if not folder_infos:
                    print("没有找到包含可 judge 结果的 log folder。")
                    if folder_cursor == root:
                        return
                    folder_cursor = folder_cursor.parent
                    continue
                _print_folder_menu(folder_infos, search_root=root, current_dir=folder_cursor)
                folder_choice = _prompt_folder_choice(folder_infos, allow_back=folder_cursor != root)
                if folder_choice is None:
                    continue
                if isinstance(folder_choice, dict) and folder_choice.get("__action__") == "refresh":
                    continue
                if isinstance(folder_choice, dict) and folder_choice.get("__action__") == "back":
                    folder_cursor = folder_cursor.parent if folder_cursor != root else root
                    continue
                assert isinstance(folder_choice, FolderMenuEntry)
                if folder_choice.entry_kind == "use_current":
                    current_log_root = folder_choice.dir_path
                else:
                    folder_cursor = folder_choice.dir_path
                    continue
            else:
                current_log_root = fixed_log_root

            task_infos = _collect_task_infos(current_log_root)
            if not task_infos:
                print("没有在该 log 路径下找到可 judge 的 task。")
                if fixed_log_root is not None:
                    return
                continue
            _print_task_menu(task_infos)
            task_choice = _prompt_task_choice(task_infos)
            if task_choice is None:
                continue
            if task_choice.get("__action__") == "refresh":
                continue

        while True:
            if fixed_skill_dir is not None:
                skill_infos = [
                    {
                        "skill_dir": fixed_skill_dir,
                        "skill_idx": int(fixed_skill_dir.name.split("_")[-1]) if fixed_skill_dir.name.split("_")[-1].isdigit() else -1,
                        "result_count": len(list(fixed_skill_dir.glob("result_*.json"))),
                        "video_count": len(list(fixed_skill_dir.glob("*.mp4"))),
                        "prompt_preview": "",
                    }
                ]
                skill_choice = skill_infos[0]
            else:
                skill_infos = _collect_skill_infos(task_choice["task_dir"])
                if not skill_infos:
                    print("这个 task 下没有可 judge 的 skill。")
                    break
                _print_skill_menu(task_choice, skill_infos)
                skill_choice = _prompt_skill_choice(skill_infos)
                if skill_choice is None:
                    continue
                if skill_choice.get("__action__") == "back":
                    break
                if skill_choice.get("__action__") == "refresh":
                    continue

            while True:
                entries = _collect_menu_entries(skill_choice["skill_dir"])
                if not entries:
                    print("这个 skill 下没有可 judge 的 result json。")
                    break
                _print_entry_menu(skill_choice, entries)
                entry_choices = _prompt_entry_choices(entries)
                if entry_choices is None:
                    continue
                if isinstance(entry_choices, dict) and entry_choices.get("__action__") == "back":
                    break
                if isinstance(entry_choices, dict) and entry_choices.get("__action__") == "refresh":
                    continue

                assert isinstance(entry_choices, list)
                _confirm_and_run_batch([entry.judge_input for entry in entry_choices], args)

            if fixed_skill_dir is not None:
                return
        if fixed_task_dir is not None:
            return
        if fixed_log_root is not None:
            return


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="对已有 skill eval 视频按时间抽帧，并调用 VLM 判断该 skill 是否成功。",
    )
    parser.add_argument(
        "--result-json",
        action="append",
        dest="result_jsons",
        type=Path,
        default=[],
        help="单个评测结果 JSON，可重复传入多次。",
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        default=None,
        help="直接指定单个视频路径；此模式需同时传 --prompt。",
    )
    parser.add_argument("--prompt", type=str, default=None, help="单个视频模式下的 skill prompt。")
    parser.add_argument("--task-name", type=str, default=None)
    parser.add_argument("--task-index", type=int, default=None)
    parser.add_argument("--episode", type=str, default=None)
    parser.add_argument("--skill-idx", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--log-path",
        type=Path,
        default=None,
        help="递归扫描该目录下所有 result_*.json，形成批量 judge 模式。",
    )
    parser.add_argument("--limit", type=int, default=None, help="最多处理多少个输入。")
    parser.add_argument("--sample-every-seconds", type=float, default=3.0)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="可选的抽帧上限；默认不限制，按视频长度和采样间隔自动决定。",
    )
    parser.add_argument("--model", type=str, default="seed-2.0-lite")
    parser.add_argument(
        "--base-url",
        type=str,
        default=DEFAULT_BASE_URL,
        help="OpenAI 兼容接口 base_url，例如 https://openrouter.ai/api/v1",
    )
    parser.add_argument("--api-key-env", type=str, default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--api-max-tokens", type=int, default=256)
    parser.add_argument("--api-temperature", type=float, default=0.0)
    parser.add_argument("--interactive", action="store_true", help="进入 terminal 交互选择模式。")
    parser.add_argument("--dry-run", action="store_true", help="只做输入解析和抽帧，不真正请求模型。")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.log_path = args.log_path.expanduser().resolve() if args.log_path is not None else None

    if args.interactive:
        _run_interactive(args)
        return 0

    try:
        inputs = _gather_inputs(args)
    except Exception as exc:
        raise SystemExit(f"Failed to gather inputs: {exc}")

    _run_judge_batch(inputs, args)
    return 0


def run(argv: list[str] | None = None) -> int:
    return main(argv)
