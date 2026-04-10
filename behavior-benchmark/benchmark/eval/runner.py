from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import MethodType
from typing import Any, Dict, Optional

import numpy as np
from omegaconf import OmegaConf

import omnigibson as og
from omnigibson.learning.eval import Evaluator
from omnigibson.learning.utils.eval_utils import PROPRIOCEPTION_INDICES
from omnigibson.learning.utils.config_utils import (
    register_omegaconf_resolvers as _register_resolvers,
)
from omnigibson.learning.utils.obs_utils import create_video_writer
from omnigibson.macros import gm

from benchmark.config.settings import get_settings
from .config import build_behavior_b1k_config

try:
    from simple_robo_agent.vla.groot_b1k_vla import GR00TB1KPolicyWrapper
    from gr00t.policy.server_client import PolicyClient

    GROOT_AVAILABLE = True
except Exception:
    GR00TB1KPolicyWrapper = None
    PolicyClient = None
    GROOT_AVAILABLE = False

import re as _re


def _sanitize_object_name(name: str) -> str:
    name = _re.sub(r"_\d+$", "", name)
    return name.replace("_", " ").split()[0] if name else name


def _format_training_prompt(skill_desc: str, object_ids: list) -> str:
    objs = [_sanitize_object_name(o) for o in (object_ids or []) if o]
    if not objs:
        return skill_desc

    if skill_desc == "place on next to" and len(objs) >= 3:
        return f"place {objs[0]} on {objs[1]} next to {objs[2]}"
    if skill_desc == "place in next to" and len(objs) >= 3:
        return f"place {objs[0]} in {objs[1]} next to {objs[2]}"

    prep_patterns = [
        (" from", 5),
        (" next to", 8),
        (" into", 5),
        (" onto", 5),
        (" under", 6),
        (" on", 3),
        (" in", 3),
        (" to", 3),
        (" off", 4),
        (" with", 5),
    ]
    if len(objs) >= 2:
        for prep, strip_len in prep_patterns:
            if skill_desc.endswith(prep):
                verb = skill_desc[:-strip_len].strip()
                return f"{verb} {objs[0]} {prep.strip()} {objs[1]}"
    return f"{skill_desc} {' '.join(objs)}".strip()


def _load_skill_context(snapshot_dir: Path) -> Dict[str, Any]:
    ctx_path = snapshot_dir / "replay_metrics.json"
    if not ctx_path.exists():
        raise FileNotFoundError(f"replay_metrics.json not found in {snapshot_dir}")
    with ctx_path.open() as f:
        return json.load(f)


def _build_video_path(log_path: Path, context: Dict[str, Any], vla_type: str) -> Path:
    task_name = context.get("task_name", "task")
    skill_idx = int(context.get("skill_idx", 0))
    prompt = context.get("prompt_used", "")
    episode = context.get("episode", "episode")
    task_index = context.get("task_index", None)

    safe = lambda s: "".join(c if c.isalnum() or c in "-_" else "_" for c in str(s))
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    ms = int((time.time() % 1) * 1000)
    filename = f"skill_{safe(vla_type)}_{safe(task_name)}_{safe(prompt)}_{episode}_s{skill_idx:02d}_{ts}_{ms:03d}.mp4"

    if isinstance(task_index, int):
        task_dir = f"task_{task_index:04d}"
    else:
        task_dir = f"task_{safe(task_name)}"

    skill_dir = f"skill_{skill_idx:02d}"
    return (log_path / task_dir / skill_dir / filename).resolve()


def _build_result_path(log_path: Path, context: Dict[str, Any], video_path: Path) -> Path:
    task_index = context.get("task_index", None)
    skill_idx = int(context.get("skill_idx", 0))
    episode = str(context.get("episode", ""))

    if isinstance(task_index, int):
        task_dir = f"task_{task_index:04d}"
    else:
        task_dir = f"task_{context.get('task_name', 'task')}"

    timestamp_match = _re.search(r"_(\d{8}_\d{6})_(\d{3})$", video_path.stem)
    if timestamp_match:
        suffix = f"_{timestamp_match.group(1)}_{timestamp_match.group(2)}"
    else:
        suffix = f"_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}_{int((time.time() % 1) * 1000):03d}"

    skill_dir = f"skill_{skill_idx:02d}"
    filename = f"result_{episode}_{skill_idx:02d}{suffix}.json"
    return (log_path / task_dir / skill_dir / filename).resolve()


def _install_early_stop_patch(
    evaluator: Evaluator,
    *,
    patience: int,
    warmup_steps: int,
    min_steps: int,
    action_eps: float,
    proprio_eps: float,
    base_vel_eps: float,
) -> None:
    original_run_vla_episode = evaluator.run_vla_episode

    def _patched_run_vla_episode(
        self,
        prompt: Optional[str] = None,
        max_steps: Optional[int] = None,
        return_preprocessed: bool = False,
        subtask_done_fn=None,
        write_video_every_n: int = 1,
    ):
        stable_count = 0
        prev_action = None
        prev_proprio = None
        early_stopped = False

        base_qvel_idx = PROPRIOCEPTION_INDICES["R1Pro"]["base_qvel"]

        def _to_numpy(x):
            if x is None:
                return None
            if hasattr(x, "detach"):
                return x.detach().cpu().numpy()
            return np.asarray(x)

        def _early_stop_detector(_raw_obs, step_count: int) -> bool:
            nonlocal stable_count, prev_action, prev_proprio, early_stopped
            if step_count < warmup_steps or step_count < min_steps:
                return False

            action_vec = _to_numpy(self.robot_action)
            proprio_vec = _to_numpy(self.obs.get("robot_r1::proprio"))
            if action_vec is None or proprio_vec is None:
                return False

            action_vec = action_vec.reshape(-1)
            proprio_vec = proprio_vec.reshape(-1)
            base_vel_norm = float(np.linalg.norm(proprio_vec[base_qvel_idx]))

            if prev_action is None or prev_proprio is None:
                prev_action = action_vec.copy()
                prev_proprio = proprio_vec.copy()
                return False

            action_delta = float(np.linalg.norm(action_vec - prev_action))
            proprio_delta = float(np.linalg.norm(proprio_vec - prev_proprio))
            prev_action = action_vec.copy()
            prev_proprio = proprio_vec.copy()

            is_stable = (
                action_delta <= action_eps
                and proprio_delta <= proprio_eps
                and base_vel_norm <= base_vel_eps
            )
            stable_count = stable_count + 1 if is_stable else 0
            if stable_count >= patience:
                early_stopped = True
                print(
                    f"[early-stop] triggered at step={step_count}, "
                    f"stable_count={stable_count}, action_delta={action_delta:.6f}, "
                    f"proprio_delta={proprio_delta:.6f}, base_vel={base_vel_norm:.6f}"
                )
                return True
            return False

        if subtask_done_fn is None:
            merged_subtask_done_fn = _early_stop_detector
        else:
            def merged_subtask_done_fn(raw_obs, step_count: int) -> bool:
                return bool(subtask_done_fn(raw_obs, step_count) or _early_stop_detector(raw_obs, step_count))

        result = original_run_vla_episode(
            prompt=prompt,
            max_steps=max_steps,
            return_preprocessed=return_preprocessed,
            subtask_done_fn=merged_subtask_done_fn,
            write_video_every_n=write_video_every_n,
        )
        result["early_stopped"] = bool(early_stopped)
        return result

    evaluator.run_vla_episode = MethodType(_patched_run_vla_episode, evaluator)


def _run_skill_eval(
    snapshot_dir: Path,
    prompt: Optional[str],
    configs_root: Path,
    log_path: Path,
    max_steps: int,
    vla_type: str,
    vla_config: Optional[Dict[str, Any]],
    env_wrapper: Optional[str],
    ignore_task_success: bool,
    early_stop_enabled: bool,
    early_stop_patience: int,
    early_stop_warmup_steps: int,
    early_stop_min_steps: int,
    early_stop_action_eps: float,
    early_stop_proprio_eps: float,
    early_stop_base_vel_eps: float,
) -> None:
    _register_resolvers()

    context = _load_skill_context(snapshot_dir)
    snapshot_path = snapshot_dir / "snapshot.json"
    if not snapshot_path.exists():
        raise FileNotFoundError(f"snapshot.json not found in {snapshot_dir}")

    task_name = context["task_name"]
    skill_desc = context.get("skill_description", "")
    if prompt is not None:
        effective_prompt = prompt
    else:
        effective_prompt = _format_training_prompt(skill_desc, context.get("object_ids", []))
    context["prompt_used"] = effective_prompt

    print(f"task          : {task_name}")
    print(f"episode       : {context.get('episode')}")
    print(f"skill_idx     : {context.get('skill_idx')}")
    print(f"skill_desc    : {skill_desc}")
    print(f"prompt        : {effective_prompt}")
    print(f"snapshot      : {snapshot_path}")
    print(f"vla_type      : {vla_type}")
    print(f"max_steps     : {max_steps}")
    print(f"ignore_success: {ignore_task_success}")
    print(f"early_stop    : {early_stop_enabled}")

    cfg = build_behavior_b1k_config(
        configs_root=configs_root,
        task_name=task_name,
        log_path=log_path,
        vla_type=vla_type,
    )

    if vla_config and vla_type == "openpi-comet":
        if vla_config.get("host") is not None:
            cfg.model.host = vla_config["host"]
        if vla_config.get("port") is not None:
            cfg.model.port = int(vla_config["port"])

    if env_wrapper is not None:
        print(f"env_wrapper   : {env_wrapper}")
        cfg.env_wrapper._target_ = env_wrapper

    try:
        gm.HEADLESS = bool(cfg.headless)
    except Exception:
        gm.HEADLESS = True

    with Evaluator(cfg) as evaluator:
        evaluator.reset()
        if early_stop_enabled:
            _install_early_stop_patch(
                evaluator,
                patience=early_stop_patience,
                warmup_steps=early_stop_warmup_steps,
                min_steps=early_stop_min_steps,
                action_eps=early_stop_action_eps,
                proprio_eps=early_stop_proprio_eps,
                base_vel_eps=early_stop_base_vel_eps,
            )

        if ignore_task_success:
            predicate_term = getattr(evaluator.env.task, "_termination_conditions", {}).get("predicate")
            if predicate_term is None:
                print("[WARN] ignore_task_success=True but predicate termination not found")
            else:
                original_predicate_step = predicate_term._step

                def _ignore_success_step(self, task, env, action):
                    original_predicate_step(task, env, action)
                    return False

                predicate_term._step = MethodType(_ignore_success_step, predicate_term)
                print("predicate success termination disabled for this rollout")

        print("restoring snapshot ...")
        og.sim.restore([str(snapshot_path)])

        for _ in range(5):
            og.sim.render()
        raw_obs, _ = evaluator.env.get_obs()
        evaluator.obs = evaluator._preprocess_obs(raw_obs)
        evaluator.policy.reset()
        print("snapshot restored, obs refreshed")

        original_policy = None
        if vla_type == "groot":
            if not GROOT_AVAILABLE:
                raise ImportError("vla_type='groot' but GR00T modules not found.")
            assert GR00TB1KPolicyWrapper is not None and PolicyClient is not None

            host = (vla_config or {}).get("host", "localhost")
            port = int((vla_config or {}).get("port", 8000))
            timeout_ms = int((vla_config or {}).get("timeout_ms", 15000))
            strict = bool((vla_config or {}).get("strict", False))

            print(f"connecting GR00T server {host}:{port}")
            client = PolicyClient(host=host, port=port, timeout_ms=timeout_ms, strict=strict)
            policy_wrapper = GR00TB1KPolicyWrapper(client=client)
            original_policy = getattr(evaluator, "policy", None)
            evaluator.policy = policy_wrapper
            try:
                policy_wrapper.reset()
            except Exception as e:
                print(f"[WARN] GR00T policy reset failed: {e}")

        video_path = _build_video_path(log_path, context, vla_type)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"video         : {video_path}")

        try:
            evaluator.video_writer = create_video_writer(
                fpath=str(video_path),
                resolution=(448, 672),
            )

            result = evaluator.run_vla_episode(
                prompt=effective_prompt,
                max_steps=max_steps,
                return_preprocessed=True,
                subtask_done_fn=None,
            )
        except Exception as eval_err:
            import traceback as tb

            print("\n" + "=" * 60, flush=True)
            print(f"[FATAL] run_vla_episode raised: {eval_err}", flush=True)
            tb.print_exc()
            print("=" * 60 + "\n", flush=True)
            import sys as _sys
            _sys.stdout.flush()
            _sys.stderr.flush()
            raise
        finally:
            evaluator.video_writer = None
            if original_policy is not None:
                evaluator.policy = original_policy

        steps = int(result.get("steps", 0))
        terminated = bool(result.get("terminated", False))
        truncated = bool(result.get("truncated", False))
        reached_max = bool(result.get("reached_max_steps", False))
        task_success = result.get("task_success")
        early_stopped = bool(result.get("early_stopped", False))

        print("\n==== Skill Episode Result ====")
        print(f"  steps            : {steps}")
        print(f"  terminated       : {terminated}")
        print(f"  truncated        : {truncated}")
        print(f"  reached_max_steps: {reached_max}")
        print(f"  task_success     : {task_success}")
        print(f"  early_stopped    : {early_stopped}")
        print(f"  video            : {video_path}")

        result_data = {
            **{k: v for k, v in context.items() if k != "obs"},
            "prompt_used": effective_prompt,
            "vla_type": vla_type,
            "max_steps": max_steps,
            "ignore_task_success": ignore_task_success,
            "steps": steps,
            "terminated": terminated,
            "truncated": truncated,
            "reached_max_steps": reached_max,
            "task_success": task_success,
            "early_stopped": early_stopped,
            "video_path": str(video_path),
        }
        result_path = _build_result_path(log_path, context, video_path)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result_data, indent=2, ensure_ascii=False))
        print(f"  result           : {result_path}")
        import sys as _sys
        _sys.stdout.flush()


def build_arg_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="基于 snapshot 恢复场景，评估 VLA 在单个 skill 上的执行能力。",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=str,
        required=True,
        help="skill snapshot 目录（含 snapshot.json + replay_metrics.json）。",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="传给 VLA 的 prompt。不传则使用 replay_metrics.json 中的 skill_description。",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=settings.eval.default_max_steps,
        help="单个 skill 的最大步数。",
    )
    parser.add_argument(
        "--configs-root",
        type=str,
        default=str(settings.eval.configs_root),
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default=str(settings.eval.log_path),
    )
    parser.add_argument(
        "--vla-type",
        type=str,
        default=settings.eval.vla_type,
        help="VLA 类型（openpi-comet / groot 等）。",
    )
    parser.add_argument(
        "--vla-config",
        type=str,
        default=None,
        help='VLA 配置 JSON。groot: host/port 等；openpi-comet: 可传 {"port":8010} 覆盖 websocket.yaml 默认 8000。',
    )
    parser.add_argument(
        "--env-wrapper",
        type=str,
        default=settings.eval.env_wrapper,
        help="Env wrapper target，如 omnigibson.learning.wrappers.RGBWrapper。",
    )
    parser.add_argument(
        "--ignore-success",
        action="store_true",
        help="忽略 OmniGibson 的整任务成功终止，只按 max_steps / 外部 skill 判定继续评估。",
    )
    parser.add_argument("--early-stop-enabled", action="store_true", help="启用 evaluator 侧 early-stop（monkey-patch）。")
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--early-stop-warmup-steps", type=int, default=20)
    parser.add_argument("--early-stop-min-steps", type=int, default=15)
    parser.add_argument("--early-stop-action-eps", type=float, default=0.002)
    parser.add_argument("--early-stop-proprio-eps", type=float, default=0.001)
    parser.add_argument("--early-stop-base-vel-eps", type=float, default=0.01)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    snapshot_dir = Path(args.snapshot_dir).expanduser().resolve()
    configs_root = Path(args.configs_root).expanduser().resolve()
    log_path = Path(args.log_path).expanduser().resolve()
    log_path.mkdir(parents=True, exist_ok=True)

    vla_config = None
    if args.vla_config:
        try:
            vla_config = json.loads(args.vla_config)
        except json.JSONDecodeError as e:
            raise SystemExit(f"Error parsing VLA config JSON: {e}")

    _run_skill_eval(
        snapshot_dir=snapshot_dir,
        prompt=args.prompt,
        configs_root=configs_root,
        log_path=log_path,
        max_steps=args.max_steps,
        vla_type=args.vla_type,
        vla_config=vla_config,
        env_wrapper=args.env_wrapper,
        ignore_task_success=args.ignore_success,
        early_stop_enabled=args.early_stop_enabled,
        early_stop_patience=args.early_stop_patience,
        early_stop_warmup_steps=args.early_stop_warmup_steps,
        early_stop_min_steps=args.early_stop_min_steps,
        early_stop_action_eps=args.early_stop_action_eps,
        early_stop_proprio_eps=args.early_stop_proprio_eps,
        early_stop_base_vel_eps=args.early_stop_base_vel_eps,
    )
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
