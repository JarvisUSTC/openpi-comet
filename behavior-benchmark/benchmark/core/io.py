from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def read_json(path: str | Path) -> dict[str, Any]:
    resolved = resolve_path(path)
    with resolved.open() as f:
        return json.load(f)


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return resolved


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    resolved = resolve_path(path)
    items: list[dict[str, Any]] = []
    with resolved.open() as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            items.append(json.loads(text))
    return items
