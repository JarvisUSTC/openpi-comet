from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TaskRef:
    task_index: int
    task_name: str
    task_prompt: str | None = None


@dataclass(slots=True)
class EpisodeRef:
    task_index: int
    episode_name: str
    instance_id: int


@dataclass(slots=True)
class SkillRef:
    task_index: int
    episode_name: str
    skill_idx: int
    skill_dir: Path
    skill_description: str = ""
    object_ids: list[str] = field(default_factory=list)
    skill_type: list[str] = field(default_factory=list)
    frame_start: int = -1
    frame_end: int = -1


@dataclass(slots=True)
class JudgeInput:
    source_result_path: Path | None
    video_path: Path
    before_image_path: Path | None
    prompt_used: str
    task_name: str | None = None
    task_prompt: str | None = None
    task_index: int | None = None
    episode: str | None = None
    skill_idx: int | None = None
    skill_description: str | None = None
    object_ids: list[str] | None = None
    manipulating_object_ids: list[str] | None = None
    max_steps: int | None = None
    steps: int | None = None


@dataclass(slots=True)
class SkillEvalResult:
    payload: dict[str, Any]

    @property
    def prompt_used(self) -> str | None:
        value = self.payload.get("prompt_used")
        return str(value) if value is not None else None

    @property
    def video_path(self) -> str | None:
        value = self.payload.get("video_path")
        return str(value) if value is not None else None

    @property
    def task_index(self) -> int | None:
        value = self.payload.get("task_index")
        return int(value) if value is not None else None


@dataclass(slots=True)
class ServerState:
    task_name: str
    task_index: int | None = None
    task_prompt: str | None = None
    port: int | None = None
    checkpoint_dir: str | None = None
    backend: str | None = None
    repo_root: str | None = None
    policy_config: str | None = None
    updated_at: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
