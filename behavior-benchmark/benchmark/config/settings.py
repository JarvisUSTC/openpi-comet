from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _strip_quotes(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    return text


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        values[key] = _strip_quotes(value)
    return values


@lru_cache(maxsize=1)
def _env_file_values() -> dict[str, str]:
    root = repo_root()
    merged: dict[str, str] = {}
    merged.update(_read_env_file(root / ".env"))
    merged.update(_read_env_file(root / ".env.local"))
    return merged


def _get(name: str, default: str | None = None) -> str | None:
    if name in os.environ and os.environ[name] != "":
        return os.environ[name]
    return _env_file_values().get(name, default)


def _get_bool(name: str, default: bool = False) -> bool:
    value = _get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def bootstrap_environment() -> None:
    os.environ.setdefault("BEHAVIOR_BENCHMARK_ROOT", str(repo_root()))
    for key, value in _env_file_values().items():
        os.environ.setdefault(key, value)
    os.environ.setdefault(
        "CONDA_BIN",
        _get("CONDA_BIN", os.path.expanduser("~/anaconda3/bin/conda"))
        or os.path.expanduser("~/anaconda3/bin/conda"),
    )
    default_conda_env = _get("CONDA_ENV", "behavior-comet")
    if default_conda_env:
        os.environ.setdefault("CONDA_ENV", default_conda_env)
    os.environ.setdefault("PYTHON_BIN", _get("PYTHON_BIN", "python") or "python")
    os.environ.setdefault(
        "SERVER_STATE_PATH",
        _get("SERVER_STATE_PATH", "~/.cache/simpleai/openpi_b1k_server_state.json")
        or "~/.cache/simpleai/openpi_b1k_server_state.json",
    )
    xla_preallocate = _get("XLA_PYTHON_CLIENT_PREALLOCATE")
    if xla_preallocate:
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", xla_preallocate)
    xla_mem_fraction = _get("XLA_PYTHON_CLIENT_MEM_FRACTION")
    if xla_mem_fraction:
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", xla_mem_fraction)
    serve_port = _get("SERVE_PORT")
    if serve_port:
        os.environ.setdefault("PORT", serve_port)
    serve_backend = _get("SERVE_BACKEND")
    if serve_backend:
        os.environ.setdefault("BACKEND", serve_backend)
    serve_policy_config = _get("SERVE_POLICY_CONFIG")
    if serve_policy_config:
        os.environ.setdefault("POLICY_CONFIG", serve_policy_config)
    serve_checkpoint_dir = _get("SERVE_CHECKPOINT_DIR")
    if serve_checkpoint_dir:
        os.environ.setdefault("CHECKPOINT_DIR", serve_checkpoint_dir)


@dataclass(frozen=True, slots=True)
class PathSettings:
    repo_root: Path
    simple_robo_agent_root: Path
    logs_root: Path
    snapshots_root: Path
    meta_root: Path
    augmented_annotations_root: Path
    server_state_path: Path


@dataclass(frozen=True, slots=True)
class JudgeSettings:
    model: str
    base_url: str
    api_key_env: str
    timeout_seconds: int
    api_key_present: bool


@dataclass(frozen=True, slots=True)
class EvalSettings:
    default_max_steps: int
    configs_root: Path
    log_path: Path
    vla_type: str
    env_wrapper: str | None
    auto_judge: bool


@dataclass(frozen=True, slots=True)
class SnapshotSettings:
    annotation_root: Path
    raw_root: Path
    configs_root: Path
    default_max_skills_per_episode: int
    default_seed: int
    default_playback_mode: str


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    conda_bin: str
    conda_env: str | None
    python_bin: str
    xla_python_client_preallocate: str | None
    xla_python_client_mem_fraction: str | None


@dataclass(frozen=True, slots=True)
class ServeSettings:
    port: int
    backend: str | None
    policy_config: str | None
    checkpoint_dir: str | None


@dataclass(frozen=True, slots=True)
class BenchmarkSettings:
    paths: PathSettings
    judge: JudgeSettings
    eval: EvalSettings
    snapshot: SnapshotSettings
    runtime: RuntimeSettings
    serve: ServeSettings


def get_settings() -> BenchmarkSettings:
    bootstrap_environment()
    sra_root = Path(_get("SRA_ROOT", "/home/simpleai/Jiawei/SimpleRoboAgent") or "").expanduser().resolve()
    logs_root = Path(_get("LOGS_ROOT", str(sra_root / "logs")) or "").expanduser().resolve()
    snapshots_root = Path(
        _get("SNAPSHOTS_ROOT", str(logs_root / "skill_snapshots")) or ""
    ).expanduser().resolve()
    meta_root = Path(_get("META_ROOT", "/home/simpleai/Data/B1K/meta") or "").expanduser().resolve()
    augmented_root = Path(
        _get("AUGMENTED_ANNOTATIONS_ROOT", "/home/simpleai/Jiawei/annotations_aug_v3") or ""
    ).expanduser().resolve()
    server_state_path = Path(
        _get("SERVER_STATE_PATH", "~/.cache/simpleai/openpi_b1k_server_state.json") or ""
    ).expanduser().resolve()
    judge_api_env = _get("JUDGE_API_KEY_ENV", "OPENROUTER_API_KEY") or "OPENROUTER_API_KEY"

    return BenchmarkSettings(
        paths=PathSettings(
            repo_root=repo_root(),
            simple_robo_agent_root=sra_root,
            logs_root=logs_root,
            snapshots_root=snapshots_root,
            meta_root=meta_root,
            augmented_annotations_root=augmented_root,
            server_state_path=server_state_path,
        ),
        judge=JudgeSettings(
            model=_get("JUDGE_MODEL", "seed-2.0-lite") or "seed-2.0-lite",
            base_url=_get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            or "https://openrouter.ai/api/v1",
            api_key_env=judge_api_env,
            timeout_seconds=int(_get("JUDGE_TIMEOUT_SECONDS", "120") or "120"),
            api_key_present=bool(_get(judge_api_env)),
        ),
        eval=EvalSettings(
            default_max_steps=int(_get("EVAL_MAX_STEPS", "300") or "300"),
            configs_root=Path(
                _get(
                    "EVAL_CONFIGS_ROOT",
                    "/home/simpleai/ruike/BEHAVIOR-1K/OmniGibson/omnigibson/learning/configs",
                )
                or "/home/simpleai/ruike/BEHAVIOR-1K/OmniGibson/omnigibson/learning/configs"
            )
            .expanduser()
            .resolve(),
            log_path=Path(
                _get("EVAL_LOG_PATH", str(logs_root / "vla_stopaug_eval"))
                or str(logs_root / "vla_stopaug_eval")
            )
            .expanduser()
            .resolve(),
            vla_type=_get("EVAL_VLA_TYPE", "openpi-comet") or "openpi-comet",
            env_wrapper=_get("EVAL_ENV_WRAPPER", "omnigibson.learning.wrappers.RGBWrapper"),
            auto_judge=_get_bool("EVAL_AUTO_JUDGE", False),
        ),
        snapshot=SnapshotSettings(
            annotation_root=Path(
                _get("SNAPSHOT_ANNOTATION_ROOT", "/home/simpleai/Data/vla_datagen/b1k/annotations")
                or "/home/simpleai/Data/vla_datagen/b1k/annotations"
            )
            .expanduser()
            .resolve(),
            raw_root=Path(_get("SNAPSHOT_RAW_ROOT", "/home/simpleai/Data/b1k_rawdata") or "")
            .expanduser()
            .resolve(),
            configs_root=Path(
                _get(
                    "SNAPSHOT_CONFIGS_ROOT",
                    _get(
                        "EVAL_CONFIGS_ROOT",
                        "/home/simpleai/ruike/BEHAVIOR-1K/OmniGibson/omnigibson/learning/configs",
                    ),
                )
                or "/home/simpleai/ruike/BEHAVIOR-1K/OmniGibson/omnigibson/learning/configs"
            )
            .expanduser()
            .resolve(),
            default_max_skills_per_episode=int(
                _get("SNAPSHOT_MAX_SKILLS_PER_EPISODE", "-1") or "-1"
            ),
            default_seed=int(_get("SNAPSHOT_SEED", "7") or "7"),
            default_playback_mode=_get("SNAPSHOT_PLAYBACK_MODE", "replay") or "replay",
        ),
        runtime=RuntimeSettings(
            conda_bin=_get("CONDA_BIN", os.path.expanduser("~/anaconda3/bin/conda"))
            or os.path.expanduser("~/anaconda3/bin/conda"),
            conda_env=_get("CONDA_ENV", "behavior-comet"),
            python_bin=_get("PYTHON_BIN", "python") or "python",
            xla_python_client_preallocate=_get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            xla_python_client_mem_fraction=_get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
        ),
        serve=ServeSettings(
            port=int(_get("SERVE_PORT", "8000") or "8000"),
            backend=_get("SERVE_BACKEND"),
            policy_config=_get("SERVE_POLICY_CONFIG"),
            checkpoint_dir=_get("SERVE_CHECKPOINT_DIR"),
        ),
    )


def _has_flag(argv: list[str], *flags: str) -> bool:
    return any(flag in argv for flag in flags)


def apply_judge_cli_defaults(argv: list[str] | None) -> list[str]:
    args = list(argv or [])
    settings = get_settings()
    if not _has_flag(args, "--model"):
        args.extend(["--model", settings.judge.model])
    if not _has_flag(args, "--base-url"):
        args.extend(["--base-url", settings.judge.base_url])
    if not _has_flag(args, "--api-key-env"):
        args.extend(["--api-key-env", settings.judge.api_key_env])
    if not _has_flag(args, "--timeout-seconds"):
        args.extend(["--timeout-seconds", str(settings.judge.timeout_seconds)])
    return args


def format_env_exports() -> str:
    settings = get_settings()
    lines = [
        f"export SRA_ROOT={shlex.quote(str(settings.paths.simple_robo_agent_root))}",
        f"export LOGS_ROOT={shlex.quote(str(settings.paths.logs_root))}",
        f"export SNAPSHOTS_ROOT={shlex.quote(str(settings.paths.snapshots_root))}",
        f"export META_ROOT={shlex.quote(str(settings.paths.meta_root))}",
        f"export AUGMENTED_ANNOTATIONS_ROOT={shlex.quote(str(settings.paths.augmented_annotations_root))}",
        f"export SERVER_STATE_PATH={shlex.quote(str(settings.paths.server_state_path))}",
        f"export JUDGE_MODEL={shlex.quote(settings.judge.model)}",
        f"export OPENROUTER_BASE_URL={shlex.quote(settings.judge.base_url)}",
        f"export JUDGE_API_KEY_ENV={shlex.quote(settings.judge.api_key_env)}",
        f"export JUDGE_TIMEOUT_SECONDS={shlex.quote(str(settings.judge.timeout_seconds))}",
        f"export EVAL_MAX_STEPS={shlex.quote(str(settings.eval.default_max_steps))}",
        f"export EVAL_CONFIGS_ROOT={shlex.quote(str(settings.eval.configs_root))}",
        f"export EVAL_LOG_PATH={shlex.quote(str(settings.eval.log_path))}",
        f"export EVAL_VLA_TYPE={shlex.quote(settings.eval.vla_type)}",
        f"export EVAL_AUTO_JUDGE={shlex.quote('true' if settings.eval.auto_judge else 'false')}",
        f"export SNAPSHOT_ANNOTATION_ROOT={shlex.quote(str(settings.snapshot.annotation_root))}",
        f"export SNAPSHOT_RAW_ROOT={shlex.quote(str(settings.snapshot.raw_root))}",
        f"export SNAPSHOT_CONFIGS_ROOT={shlex.quote(str(settings.snapshot.configs_root))}",
        "export SNAPSHOT_MAX_SKILLS_PER_EPISODE="
        + shlex.quote(str(settings.snapshot.default_max_skills_per_episode)),
        f"export SNAPSHOT_SEED={shlex.quote(str(settings.snapshot.default_seed))}",
        f"export SNAPSHOT_PLAYBACK_MODE={shlex.quote(settings.snapshot.default_playback_mode)}",
        f"export CONDA_BIN={shlex.quote(settings.runtime.conda_bin)}",
        f"export PYTHON_BIN={shlex.quote(settings.runtime.python_bin)}",
        f"export SERVE_PORT={shlex.quote(str(settings.serve.port))}",
    ]
    if settings.eval.env_wrapper:
        lines.append(f"export EVAL_ENV_WRAPPER={shlex.quote(settings.eval.env_wrapper)}")
    if settings.runtime.conda_env:
        lines.append(f"export CONDA_ENV={shlex.quote(settings.runtime.conda_env)}")
    if settings.runtime.xla_python_client_preallocate:
        lines.append(
            "export XLA_PYTHON_CLIENT_PREALLOCATE="
            + shlex.quote(settings.runtime.xla_python_client_preallocate)
        )
    if settings.runtime.xla_python_client_mem_fraction:
        lines.append(
            "export XLA_PYTHON_CLIENT_MEM_FRACTION="
            + shlex.quote(settings.runtime.xla_python_client_mem_fraction)
        )
    if settings.serve.backend:
        lines.append(f"export SERVE_BACKEND={shlex.quote(settings.serve.backend)}")
    if settings.serve.policy_config:
        lines.append(f"export SERVE_POLICY_CONFIG={shlex.quote(settings.serve.policy_config)}")
    if settings.serve.checkpoint_dir:
        lines.append(f"export SERVE_CHECKPOINT_DIR={shlex.quote(settings.serve.checkpoint_dir)}")
    return "\n".join(lines)
