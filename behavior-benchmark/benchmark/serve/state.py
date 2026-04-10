from __future__ import annotations

from pathlib import Path

from benchmark.core.io import read_json, write_json
from benchmark.core.schemas import ServerState


def load_server_state(path: str | Path) -> ServerState | None:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        return None
    data = read_json(resolved)
    return ServerState(
        task_name=str(data.get("task_name", "")),
        task_index=int(data["task_index"]) if data.get("task_index") is not None else None,
        task_prompt=str(data.get("task_prompt", "")) or None,
        port=int(data["port"]) if data.get("port") is not None else None,
        checkpoint_dir=str(data.get("checkpoint_dir", "")) or None,
        backend=str(data.get("backend", "")) or None,
        repo_root=str(data.get("repo_root", "")) or None,
        policy_config=str(data.get("policy_config", "")) or None,
        updated_at=int(data["updated_at"]) if data.get("updated_at") is not None else None,
    )


def save_server_state(path: str | Path, state: ServerState) -> Path:
    return write_json(path, state.to_dict())
