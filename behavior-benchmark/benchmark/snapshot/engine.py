from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import torch as th
from omnigibson.learning.eval import Evaluator
from omnigibson.learning.utils.config_utils import (
    register_omegaconf_resolvers as _register_resolvers,
)
from omnigibson.learning.utils.eval_utils import ROBOT_CAMERA_NAMES
from omnigibson.macros import gm

import omnigibson as og

from benchmark.config.settings import get_settings
from benchmark.core.inventory import load_task_mapping
from benchmark.core.paths import skill_dir_name, task_dir_name
from benchmark.eval.config import build_behavior_b1k_config

from .discovery import (
    choose_episode,
    collect_heldout_episode_infos,
    collect_skill_targets,
    get_available_task_dirs,
    print_episode_inventory,
)


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, th.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _get_camera_keys() -> tuple[str, str, str]:
    return (
        ROBOT_CAMERA_NAMES["R1Pro"]["left_wrist"] + "::rgb",
        ROBOT_CAMERA_NAMES["R1Pro"]["right_wrist"] + "::rgb",
        ROBOT_CAMERA_NAMES["R1Pro"]["head"] + "::rgb",
    )


def _obs_to_montage(
    obs: dict[str, Any],
    *,
    wrist_size: int = 448,
    head_size: int = 896,
) -> np.ndarray:
    left_key, right_key, head_key = _get_camera_keys()
    left_wrist_rgb = cv2.resize(_to_numpy(obs[left_key]), (wrist_size, wrist_size))
    right_wrist_rgb = cv2.resize(_to_numpy(obs[right_key]), (wrist_size, wrist_size))
    head_rgb = cv2.resize(_to_numpy(obs[head_key]), (head_size, head_size))
    montage = np.hstack([np.vstack([left_wrist_rgb, right_wrist_rgb]), head_rgb])
    return np.clip(montage, 0, 255).astype(np.uint8)


def _save_rgb_image(image_rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def _normalize_nested_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        results: list[str] = []
        for item in value:
            results.extend(_normalize_nested_strings(item))
        return results
    return [str(value)]


def _parse_task_indices(values: list[str] | None) -> list[int] | None:
    if not values:
        return None

    parsed: list[int] = []
    for value in values:
        for part in value.split(","):
            task_index_str = part.strip()
            if task_index_str:
                parsed.append(int(task_index_str))

    return list(dict.fromkeys(parsed)) if parsed else None


def _normalize_episode_name(value: str) -> str:
    episode_name = value.strip()
    if episode_name.endswith(".json"):
        episode_name = episode_name[:-5]
    if not episode_name.startswith("episode_"):
        raise ValueError(
            f"--episode-name 必须形如 episode_00291400 或 episode_00291400.json，收到: {value!r}"
        )
    return episode_name


def _load_hdf5_episode(hdf5_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    with h5py.File(str(hdf5_path), "r") as hdf5_file:
        demo = hdf5_file["data/demo_0"]
        actions = demo["action"][:].astype(np.float32)
        states = demo["state"][:]
        state_sizes = demo["state_size"][:]
        transitions_str = demo.attrs.get("transitions", "{}")
        transitions = json.loads(transitions_str) if transitions_str else {}
    return actions, states, state_sizes, transitions


def _apply_transitions_at_frame(scene: Any, cur_transitions: dict[str, Any]) -> None:
    try:
        from omnigibson.utils.python_utils import create_object_from_init_info
    except ImportError:
        from omnigibson.objects.object_base import create_object_from_init_info

    systems = cur_transitions.get("systems") or {}
    objects = cur_transitions.get("objects") or {}

    for add_sys_name in systems.get("add", []) or []:
        try:
            scene.get_system(add_sys_name, force_init=True)
        except Exception as exc:
            print(f"  [WARN] transition add system failed: {add_sys_name}: {exc}")
    for remove_sys_name in systems.get("remove", []) or []:
        try:
            scene.clear_system(remove_sys_name)
        except Exception as exc:
            print(f"  [WARN] transition remove system failed: {remove_sys_name}: {exc}")
    for remove_obj_name in objects.get("remove", []) or []:
        obj = scene.object_registry("name", remove_obj_name)
        if obj is None:
            continue
        try:
            scene.remove_object(obj)
        except Exception as exc:
            print(f"  [WARN] transition remove object failed: {remove_obj_name}: {exc}")
    for index, add_obj_info in enumerate(objects.get("add", []) or []):
        try:
            obj = create_object_from_init_info(add_obj_info)
            scene.add_object(obj)
            obj.set_position(th.ones(3) * 100.0 + th.ones(3) * 5 * index)
        except Exception as exc:
            print(f"  [WARN] transition add object failed (idx={index}): {exc}")
    og.sim.step()


VIDEO_FPS = 10


def _record_skill_video(
    *,
    evaluator: Any,
    actions: np.ndarray,
    states: np.ndarray,
    state_sizes: np.ndarray,
    transitions: dict[str, Any],
    frame_start: int,
    frame_end: int,
    video_path: Path,
) -> int:
    total_action_len = len(actions)
    actual_end = min(frame_end, total_action_len, len(states) - 1)

    first_frame = True
    writer: cv2.VideoWriter | None = None
    n_frames = 0

    for frame_index in range(frame_start, actual_end):
        transition = transitions.get(str(frame_index))
        if transition is not None:
            _apply_transitions_at_frame(og.sim.scenes[0], transition)

        state_index = frame_index + 1
        if state_index < len(states):
            state_size = int(state_sizes[state_index])
            og.sim.load_state(th.from_numpy(states[state_index, :state_size].copy()), serialized=True)

        evaluator.env.step(th.from_numpy(actions[frame_index]), n_render_iterations=1)
        for _ in range(2):
            og.sim.render()
        raw_obs, _ = evaluator.env.get_obs()
        obs = evaluator._preprocess_obs(raw_obs)
        montage = _obs_to_montage(obs, wrist_size=360, head_size=720)

        if first_frame:
            height, width = montage.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(video_path), fourcc, VIDEO_FPS, (width, height))
            first_frame = False

        assert writer is not None
        writer.write(cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
        n_frames += 1

    if writer is not None:
        writer.release()
    return n_frames


def _generate_snapshots_for_episode(
    *,
    task_index: int,
    task_name: str,
    task_prompt: str,
    annotation_path: Path,
    hdf5_path: Path,
    instance_id: int,
    skill_targets: list[tuple[int, dict[str, Any]]],
    configs_root: Path,
    output_dir: Path,
    record_skills: bool = False,
    playback_mode: str = "replay",
) -> list[dict[str, Any]]:
    actions, states, state_sizes, transitions = _load_hdf5_episode(hdf5_path)
    total_action_len = len(actions)
    print(f"  HDF5 state: {states.shape}, transitions: {len(transitions)} entries")

    if playback_mode not in {"replay", "state"}:
        raise ValueError(f"Unknown --playback-mode: {playback_mode}")
    if playback_mode == "state" and record_skills:
        raise ValueError("--record-skills requires --playback-mode replay")

    cfg = build_behavior_b1k_config(
        configs_root=configs_root,
        task_name=task_name,
        log_path=output_dir,
        vla_type="groot",
    )
    if getattr(cfg, "env_wrapper", None) is not None:
        cfg.env_wrapper._target_ = "omnigibson.learning.wrappers.RGBWrapper"
    try:
        gm.HEADLESS = bool(cfg.headless)
    except Exception:
        gm.HEADLESS = True

    results: list[dict[str, Any]] = []
    with Evaluator(cfg) as evaluator:
        evaluator.reset()
        evaluator.load_task_instance(instance_id, test_hidden=cfg.test_hidden)

        if len(states) > 0:
            state_size = int(state_sizes[0])
            try:
                og.sim.load_state(th.from_numpy(states[0, :state_size].copy()), serialized=True)
                if playback_mode == "replay":
                    og.sim.step()
                print(f"  initial state loaded (size={state_size}, mode={playback_mode})")
            except Exception as exc:
                print(f"  [WARN] failed to load initial state: {exc}")

        current_frame = 0
        for frame_start, skill in skill_targets:
            skill_idx = int(skill.get("skill_idx", -1))
            frame_end = int(skill["frame_duration"][1])
            skill_t0 = time.time()

            if frame_start >= len(states):
                print(
                    f"  [WARN] skill {skill_idx}: frame_start={frame_start} exceeds state length {len(states)}"
                )
                continue

            steps_needed = frame_start - current_frame
            if steps_needed < 0:
                print(
                    f"  [WARN] skill {skill_idx}: frame_start={frame_start} "
                    f"< current_frame={current_frame}, skipping"
                )
                continue

            if playback_mode == "replay":
                if steps_needed > 0:
                    print(f"  replaying {current_frame} -> {frame_start} ({steps_needed} steps) ...")
                    t0 = time.time()
                    for frame_index in range(current_frame, frame_start):
                        if frame_index >= total_action_len:
                            print(f"  [WARN] action 越界 at frame {frame_index}, stopping replay")
                            break
                        transition = transitions.get(str(frame_index))
                        if transition is not None:
                            _apply_transitions_at_frame(og.sim.scenes[0], transition)
                        state_index = frame_index + 1
                        if state_index < len(states):
                            state_size = int(state_sizes[state_index])
                            og.sim.load_state(
                                th.from_numpy(states[state_index, :state_size].copy()),
                                serialized=True,
                            )
                        evaluator.env.step(th.from_numpy(actions[frame_index]), n_render_iterations=1)
                    elapsed = time.time() - t0
                    fps = steps_needed / elapsed if elapsed > 0 else float("inf")
                    print(f"  replay done in {elapsed:.1f}s ({fps:.1f} fps)")
                    current_frame = frame_start

                if steps_needed == 0:
                    state_size = int(state_sizes[frame_start])
                    gt_state = th.from_numpy(states[frame_start, :state_size].copy())
                    try:
                        og.sim.load_state(gt_state, serialized=True)
                        og.sim.step()
                        print(f"  state correction applied @ frame {frame_start} (size={state_size})")
                    except Exception as exc:
                        print(f"  [WARN] state correction failed @ frame {frame_start}: {exc}")
            else:
                try:
                    for frame_index in range(current_frame, frame_start):
                        transition = transitions.get(str(frame_index))
                        if transition is not None:
                            _apply_transitions_at_frame(og.sim.scenes[0], transition)
                        state_index = frame_index + 1
                        if state_index < len(states):
                            state_size = int(state_sizes[state_index])
                            og.sim.load_state(
                                th.from_numpy(states[state_index, :state_size].copy()),
                                serialized=True,
                            )
                    current_frame = frame_start
                    state_size = int(state_sizes[frame_start])
                    print(f"  reached frame {frame_start} via state playback (size={state_size})")
                except Exception as exc:
                    print(f"  [WARN] failed to reach frame {frame_start} via state playback: {exc}")
                    continue

            skill_dir = output_dir / task_dir_name(task_index) / annotation_path.stem / skill_dir_name(skill_idx)
            skill_dir.mkdir(parents=True, exist_ok=True)

            snapshot_path = skill_dir / "snapshot.json"
            og.sim.save([str(snapshot_path)])

            for _ in range(5):
                og.sim.render()
            raw_obs, _ = evaluator.env.get_obs()
            obs = evaluator._preprocess_obs(raw_obs)
            before_path = skill_dir / "before.png"
            _save_rgb_image(_obs_to_montage(obs), before_path)

            video_path: Path | None = None
            if record_skills:
                video_path = skill_dir / "replay.mp4"
                print(f"  recording skill {skill_idx:02d} ({frame_start} -> {frame_end}) ...")
                t0 = time.time()
                n_video_frames = _record_skill_video(
                    evaluator=evaluator,
                    actions=actions,
                    states=states,
                    state_sizes=state_sizes,
                    transitions=transitions,
                    frame_start=frame_start,
                    frame_end=frame_end,
                    video_path=video_path,
                )
                elapsed_rec = time.time() - t0
                print(f"  recorded {n_video_frames} frames in {elapsed_rec:.1f}s -> {video_path}")
                current_frame = min(frame_end, total_action_len)

            skill_descriptions = _normalize_nested_strings(skill.get("skill_description", []))
            skill_desc = ",".join(skill_descriptions) or f"skill_{skill_idx}"
            metrics = {
                "task_index": task_index,
                "task_name": task_name,
                "task_prompt": task_prompt,
                "episode": annotation_path.stem,
                "instance_id": instance_id,
                "skill_idx": skill_idx,
                "skill_id": skill.get("skill_id", []),
                "playback_mode": playback_mode,
                "skill_description": skill_desc,
                "skill_type": skill.get("skill_type", []),
                "frame_start": frame_start,
                "frame_end": frame_end,
                "object_ids": _normalize_nested_strings(skill.get("object_id", [])),
                "manipulating_object_ids": _normalize_nested_strings(
                    skill.get("manipulating_object_id", [])
                ),
                "memory_prefix": _normalize_nested_strings(skill.get("memory_prefix", [])),
                "spatial_prefix": _normalize_nested_strings(skill.get("spatial_prefix", [])),
                "hdf5_path": str(hdf5_path),
                "snapshot_path": str(snapshot_path),
                "before_image": str(before_path),
                "replay_video": str(video_path) if video_path is not None else None,
                "replay_frames": frame_start,
                "skill_wall_time_sec": round(time.time() - skill_t0, 2),
            }
            metrics_path = skill_dir / "replay_metrics.json"
            metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
            results.append(metrics)

            print(f"  skill {skill_idx:02d} @ frame {frame_start}: saved")
            print(f"    snapshot: {snapshot_path}")
            print(f"    before  : {before_path}")

    return results


def _build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Phase 1: 通过 action replay 为每个 skill 生成完整场景快照。",
    )
    parser.add_argument(
        "--annotation-root",
        type=str,
        default=str(settings.snapshot.annotation_root),
        help="Annotation 根目录（含 task-XXXX/*.json）。",
    )
    parser.add_argument(
        "--meta-root",
        type=str,
        default=str(settings.paths.meta_root),
        help="Meta 目录（含 tasks.jsonl）。",
    )
    parser.add_argument(
        "--raw-root",
        type=str,
        default=str(settings.snapshot.raw_root),
        help="Raw HDF5 数据根目录（含 task-XXXX/*.hdf5）。",
    )
    parser.add_argument(
        "--configs-root",
        type=str,
        default=str(settings.snapshot.configs_root),
        help="BEHAVIOR-1K learning/configs 目录。",
    )
    parser.add_argument(
        "--task-index",
        nargs="+",
        type=str,
        default=None,
        help="指定一个或多个 task index。不指定则处理所有 task。",
    )
    parser.add_argument(
        "--episode-name",
        type=str,
        default=None,
        help="显式指定要切的 held-out episode，仅支持单个 task-index。",
    )
    parser.add_argument(
        "--list-missing-episodes",
        action="store_true",
        default=False,
        help="列出所选 task 中尚未完整切出的 held-out episode 并退出。",
    )
    parser.add_argument(
        "--max-skills-per-episode",
        type=int,
        default=settings.snapshot.default_max_skills_per_episode,
        help="每个 episode 最多处理的 skill 数量（-1 = 全部）。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=settings.snapshot.default_seed,
        help="随机种子（用于 episode 选择）。",
    )
    parser.add_argument(
        "--no-require-strict-alignment",
        action="store_true",
        default=False,
        help="关闭 held-out 与 HDF5 的严格帧对齐过滤。",
    )
    parser.add_argument(
        "--require-strict-alignment",
        action="store_true",
        default=False,
        help="兼容旧参数；严格对齐已默认开启。",
    )
    parser.add_argument(
        "--record-skills",
        action="store_true",
        default=False,
        help="录制每个 skill 的 GT replay 视频。",
    )
    parser.add_argument(
        "--playback-mode",
        type=str,
        choices=("replay", "state"),
        default=settings.snapshot.default_playback_mode,
        help="回放模式：replay=action replay；state=仅 transitions + load_state 跳帧。",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(settings.paths.snapshots_root),
        help="输出根目录。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.require_strict_alignment and args.no_require_strict_alignment:
        parser.error("不能同时指定 --require-strict-alignment 与 --no-require-strict-alignment")

    require_strict_alignment = not args.no_require_strict_alignment
    try:
        task_indices = _parse_task_indices(args.task_index)
    except ValueError as exc:
        parser.error(f"--task-index 包含非法整数: {exc}")
    try:
        episode_name = _normalize_episode_name(args.episode_name) if args.episode_name else None
    except ValueError as exc:
        parser.error(str(exc))

    _register_resolvers()

    annotation_root = Path(args.annotation_root).expanduser().resolve()
    meta_root = Path(args.meta_root).expanduser().resolve()
    raw_root = Path(args.raw_root).expanduser().resolve()
    configs_root = Path(args.configs_root).expanduser().resolve()
    run_output_dir = Path(args.output_root).expanduser().resolve()
    run_output_dir.mkdir(parents=True, exist_ok=True)

    task_mapping = load_task_mapping(meta_root / "tasks.jsonl")
    available_task_dirs = get_available_task_dirs(annotation_root=annotation_root, raw_root=raw_root)

    if task_indices is not None:
        requested_task_indices = set(task_indices)
        chosen_tasks = [
            (task_index, ann_task_dir, raw_task_dir)
            for task_index, ann_task_dir, raw_task_dir in available_task_dirs
            if task_index in requested_task_indices
        ]
        available_task_indices = {task_index for task_index, _, _ in available_task_dirs}
        missing_task_indices = [
            task_index for task_index in task_indices if task_index not in available_task_indices
        ]
        if missing_task_indices:
            raise RuntimeError(
                f"task-index {missing_task_indices} 不在可用列表中 (可用: {[t[0] for t in available_task_dirs]})"
            )
        chosen_task_map = {
            task_index: (ann_task_dir, raw_task_dir)
            for task_index, ann_task_dir, raw_task_dir in chosen_tasks
        }
        chosen_tasks = [
            (task_index, *chosen_task_map[task_index]) for task_index in task_indices
        ]
    else:
        chosen_tasks = available_task_dirs

    if episode_name is not None and (task_indices is None or len(task_indices) != 1):
        parser.error("--episode-name 需要且仅支持一个 --task-index")
    if args.list_missing_episodes and task_indices is None:
        parser.error("--list-missing-episodes 需要配合 --task-index 使用")

    rng = random.Random(args.seed)

    print(f"annotation_root         : {annotation_root}")
    print(f"raw_root                : {raw_root}")
    print(f"seed                    : {args.seed}")
    print(f"task_index              : {task_indices or 'all'}")
    print(f"episode_name            : {episode_name or 'auto'}")
    print(f"num_tasks               : {len(chosen_tasks)}")
    print(f"max_skills_per_episode  : {args.max_skills_per_episode}")
    print(f"require_strict_alignment: {require_strict_alignment}")
    print(f"record_skills           : {args.record_skills}")
    print(f"output_dir              : {run_output_dir}")
    print("=" * 80)

    if args.list_missing_episodes:
        for task_index, ann_task_dir, raw_task_dir in chosen_tasks:
            task_name = task_mapping[task_index]["task_name"]
            infos, used_all_episodes, total_ann_count = collect_heldout_episode_infos(
                task_index=task_index,
                ann_task_dir=ann_task_dir,
                raw_task_dir=raw_task_dir,
                output_dir=run_output_dir,
                require_strict_alignment=require_strict_alignment,
                max_skills=args.max_skills_per_episode,
            )
            print(f"\n[Task {task_index:04d}] {task_name}")
            print_episode_inventory(
                infos=infos,
                used_all_episodes=used_all_episodes,
                total_ann_count=total_ann_count,
                require_strict_alignment=require_strict_alignment,
            )
            missing_infos = [
                info for info in infos if info["status"] == "ready" and not info["is_complete"]
            ]
            if not missing_infos:
                print("  可选 episode: 无")
                continue
            print("  可选 episode（held-out 且未完整切出）:")
            for info in missing_infos:
                print(
                    "   "
                    f"{info['episode_name']}  "
                    f"(instance={info['instance_id']}, "
                    f"snapshot={info['snapshot_count']}/{info['expected_skill_count']}, "
                    f"metrics={info['metrics_count']}/{info['expected_skill_count']})"
                )
        return 0

    all_results: list[dict[str, Any]] = []
    for task_index, ann_task_dir, raw_task_dir in chosen_tasks:
        task_meta = task_mapping.get(task_index, {})
        task_name = str(task_meta.get("task_name", f"task_{task_index:04d}"))
        task_prompt = str(task_meta.get("task", ""))

        episode_info = choose_episode(
            task_index=task_index,
            ann_task_dir=ann_task_dir,
            raw_task_dir=raw_task_dir,
            output_dir=run_output_dir,
            rng=rng,
            require_strict_alignment=require_strict_alignment,
            max_skills=args.max_skills_per_episode,
            episode_name=episode_name,
        )
        ann_path = Path(episode_info["ann_path"])
        annotation = episode_info["annotation"]
        hdf5_path = Path(episode_info["hdf5_path"])
        instance_id = int(episode_info["instance_id"])
        action_len = int(episode_info["action_len"])
        skill_targets = collect_skill_targets(
            annotation,
            action_len,
            max_skills=args.max_skills_per_episode,
            rng=rng,
        )

        if not skill_targets:
            print(f"\n[Task {task_index:04d}] {task_name} — 无有效 skill，跳过")
            continue

        print(f"\n[Task {task_index:04d}] {task_name}")
        print(f"  prompt  : {task_prompt}")
        print(f"  episode : {ann_path.stem}  (instance={instance_id})")
        print(f"  actions : {action_len}")
        print(f"  hdf5    : {hdf5_path}")
        print(
            "  existing: "
            f"snapshot={episode_info['snapshot_count']}/{episode_info['expected_skill_count']}, "
            f"metrics={episode_info['metrics_count']}/{episode_info['expected_skill_count']}"
        )
        print(f"  skills  : {len(skill_targets)} targets")
        print(f"  frames  : {[frame_start for frame_start, _ in skill_targets]}")

        try:
            results = _generate_snapshots_for_episode(
                task_index=task_index,
                task_name=task_name,
                task_prompt=task_prompt,
                annotation_path=ann_path,
                hdf5_path=hdf5_path,
                instance_id=instance_id,
                skill_targets=skill_targets,
                configs_root=configs_root,
                output_dir=run_output_dir,
                record_skills=args.record_skills,
                playback_mode=args.playback_mode,
            )
            all_results.extend(results)
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 80)
    print(f"Summary: {len(all_results)} snapshots generated")
    for result in all_results:
        print(
            f"  [{result['episode']}] skill {int(result['skill_idx']):02d} "
            f"@ frame {result['frame_start']}: {result['skill_description']}"
        )

    summary_path = run_output_dir / "summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False) + "\n")
    print(f"\nsummary saved to: {summary_path}")
    return 0


def run(argv: list[str] | None = None) -> int:
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
