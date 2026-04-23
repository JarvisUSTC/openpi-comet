import dataclasses
import json
import pathlib
from collections.abc import Iterable, Sequence
from typing import Any, Literal


ImageStorage = Literal["path", "zip"]


@dataclasses.dataclass(frozen=True)
class VQAImageRef:
    """Reference to an image stored on disk or inside a zip archive."""

    storage: ImageStorage
    path: str | None = None
    archive_path: str | None = None
    member_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VQAImageRef":
        return cls(
            storage=payload["storage"],
            path=payload.get("path"),
            archive_path=payload.get("archive_path"),
            member_path=payload.get("member_path"),
        )


@dataclasses.dataclass(frozen=True)
class VQASample:
    """Normalized sample schema used by local VQA preprocessors and loaders."""

    sample_id: str
    prompt: str
    answer: str
    images: tuple[VQAImageRef, ...]
    task_family: str
    task_name: str
    source: str
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "prompt": self.prompt,
            "answer": self.answer,
            "images": [image.to_dict() for image in self.images],
            "task_family": self.task_family,
            "task_name": self.task_name,
            "source": self.source,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VQASample":
        return cls(
            sample_id=payload["sample_id"],
            prompt=payload["prompt"],
            answer=payload["answer"],
            images=tuple(VQAImageRef.from_dict(image) for image in payload["images"]),
            task_family=payload["task_family"],
            task_name=payload["task_name"],
            source=payload["source"],
            metadata=dict(payload.get("metadata", {})),
        )


def normalize_prompt(prompt: str) -> str:
    prompt = prompt.replace("<image>", " ")
    return " ".join(prompt.split())


def conversations_to_prompt_answer(conversations: Sequence[dict[str, Any]]) -> tuple[str, str]:
    human_turns: list[str] = []
    assistant_turns: list[str] = []
    for turn in conversations:
        role = str(turn.get("from", "")).lower()
        value = str(turn.get("value", "")).strip()
        if role == "human":
            human_turns.append(value)
        elif role in {"gpt", "assistant"}:
            assistant_turns.append(value)

    if not human_turns:
        raise ValueError("VQA sample conversations must include a human prompt.")
    if not assistant_turns:
        raise ValueError("VQA sample conversations must include an assistant answer.")
    return normalize_prompt(human_turns[-1]), assistant_turns[-1]


def load_samples_from_path(path: pathlib.Path) -> list[VQASample]:
    if path.suffix == ".jsonl":
        return [VQASample.from_dict(json.loads(line)) for line in path.read_text().splitlines() if line.strip()]

    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        payload = payload.get("samples", [])
    if not isinstance(payload, list):
        raise ValueError(f"Unsupported VQA schema payload in {path}.")
    return [VQASample.from_dict(item) for item in payload]


def dump_samples_to_jsonl(samples: Iterable[VQASample], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
