import bisect
from collections import defaultdict
import logging
from collections.abc import Callable, Iterable
import json
import os
from pathlib import Path
import random
import re

import datasets
from datasets import load_dataset
from huggingface_hub import snapshot_download
from lerobot.constants import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import EPISODES_PATH
from lerobot.datasets.utils import EPISODES_STATS_PATH
from lerobot.datasets.utils import STATS_PATH
from lerobot.datasets.utils import TASKS_PATH
from lerobot.datasets.utils import backward_compatible_episodes_stats
from lerobot.datasets.utils import cast_stats_to_numpy
from lerobot.datasets.utils import check_delta_timestamps
from lerobot.datasets.utils import check_timestamps_sync
from lerobot.datasets.utils import check_version_compatibility
from lerobot.datasets.utils import get_delta_indices
from lerobot.datasets.utils import get_episode_data_index
from lerobot.datasets.utils import get_safe_version
from lerobot.datasets.utils import is_valid_version
from lerobot.datasets.utils import load_info
from lerobot.datasets.utils import load_json
from lerobot.datasets.utils import load_jsonlines
from lerobot.datasets.video_utils import get_safe_default_codec
import numpy as np
from omnigibson.learning.utils.eval_utils import ROBOT_CAMERA_NAMES
from omnigibson.learning.utils.eval_utils import TASK_NAMES_TO_INDICES
from omnigibson.learning.utils.lerobot_utils import aggregate_stats
from omnigibson.learning.utils.lerobot_utils import decode_video_frames
from omnigibson.learning.utils.lerobot_utils import hf_transform_to_torch
from omnigibson.learning.utils.obs_utils import OBS_LOADER_MAP
from omnigibson.learning.utils.obs_utils import instance_id_to_instance
from omnigibson.utils.ui_utils import create_module_logger
import packaging.version
import torch as th
from torch.utils.data import Dataset
from torch.utils.data import get_worker_info

ANNOTATIONS_PATH = "annotations"
ORCHESTRATORS_PATH = "orchestrators"
logger = create_module_logger("BehaviorLeRobotDataset")

from behavior.learning.datas.skill_prompt import _flatten_objs
from behavior.learning.datas.skill_prompt import _sanitize_object_name
from behavior.learning.datas.skill_prompt import _skill_desc_text
from behavior.learning.datas.skill_prompt import format_skill_prompt




def build_orchestrator_levels_from_annotations(
    episode_key: int,
    episode_len: int,
    skill_annotation: list,
    level_0_task: str,
) -> dict:
    """
    Build orchestrator levels 0, 1, 2 from skill_annotation when no orchestrator files exist.
    - level 0: one segment, whole episode, level_0_task.
    - level 1 & 2: one segment per skill; task text = format_skill_prompt(skill).
    Uses frame_duration [start, end] per skill if present (end exclusive); else equal split.
    """
    output_data = defaultdict(list)
    output_data[0].append({
        "task": level_0_task,
        "start_frame": 0,
        "end_frame": episode_len - 1,
    })
    if not skill_annotation:
        output_data[1] = list(output_data[0])
        output_data[2] = list(output_data[0])
        output_data[3] = list(output_data[0])
        return output_data

    def _parse_frame_duration(fd) -> tuple[int, int] | None:
        """Parse frame_duration as [start, end]; return (start, end) or None.
        Multi-segment format like [[0,50],[100,150]] is not supported and returns None (will be filtered out).
        """
        if fd is None or not isinstance(fd, (list, tuple)) or len(fd) < 2:
            return None
        a, b = fd[0], fd[1]
        # 多段格式 [[0,50],[100,150]] 直接过滤：首元素是 list 表示多段，不解析
        if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
            return None
        try:
            return int(a), int(b)
        except (TypeError, ValueError):
            return None

    n = len(skill_annotation)
    for i, s in enumerate(skill_annotation):
        skill_desc = _skill_desc_text(s)
        task_text = format_skill_prompt(s)
        parsed = _parse_frame_duration(s.get("frame_duration")) if "frame_duration" in s else None
        # 有 frame_duration 但解析为 None（多段或无效）时跳过该 skill，不加入 orchestrator
        if "frame_duration" in s and parsed is None:
            continue
        if parsed is not None:
            start_f, end_f = parsed
            end_frame = min(end_f - 1, episode_len - 1) if end_f > 0 else episode_len - 1
            start_frame = max(0, start_f)
        else:
            start_frame = (i * episode_len) // n
            end_frame = ((i + 1) * episode_len - 1) // n if i < n - 1 else episode_len - 1
        output_data[1].append({
            "task": task_text,
            "skill": skill_desc,
            "start_frame": start_frame,
            "end_frame": end_frame,
        })
        output_data[2].append({
            "task": task_text,
            "skill": skill_desc,
            "start_frame": start_frame,
            "end_frame": end_frame,
        })
    # 若所有 skill 均被过滤（如均为多段 frame_duration），则 level 1/2 退化为整段 episode
    if not output_data[1]:
        output_data[1] = list(output_data[0])
        output_data[2] = list(output_data[0])
    output_data[3] = list(output_data[2])
    return output_data


class BehaviorLeRobotDataset(LeRobotDataset):
    """
    BehaviorLeRobotDataset is a customized dataset class for loading and managing LeRobot datasets,
    with additional filtering and loading options tailored for the BEHAVIOR-1K benchmark.
    This class extends LeRobotDataset and introduces the following customizations:
        - Task-based filtering: Load only episodes corresponding to specific tasks.
        - Modality and camera selection: Load only specified modalities (e.g., "rgb", "depth", "seg_instance_id")
          and cameras (e.g., "left_wrist", "right_wrist", "head").
        - Ability to download and use additional annotation and metainfo files.
        - Local-only mode: Optionally restrict dataset usage to local files, disabling downloads.
        - Optional batch streaming using keyframe for faster access.
    These customizations allow for more efficient and targeted dataset usage in the context of B1K tasks
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = "pyav",
        batch_encoding_size: int = 1,
        # === Customized arguments for BehaviorLeRobotDataset ===
        tasks: Iterable[str] = None,
        modalities: Iterable[str] = None,
        cameras: Iterable[str] = None,
        local_only: bool = False,
        check_timestamp_sync: bool = True,
        chunk_streaming_using_keyframe: bool = True,
        shuffle: bool = True,
        seed: int = 42,
        fine_grained_level: int = 0,  # 0, 1, 2, 3
        train_rgb_type: str = "regular",  # regular | bbox | point
        return_seg_instance: bool = False,
        skill_list: list[str] = ["all"],
    ):
        """
        Custom args:
            episodes (List[int]): list of episodes to use PER TASK.
                NOTE: This is different from the actual episode indices in the dataset.
                Rather, this is meant to be used for train/val split, or loading a specific amount of partial data.
                If set to None, all episodes will be loaded for a given task.
            tasks (List[str]): list of task names to load. If None, all tasks will be loaded.
            modalities (List[str]): list of modality names to load. If None, all modalities will be loaded.
                must be a subset of ["rgb", "depth", "seg_instance_id"]
            cameras (List[str]): list of camera names to load. If None, all cameras will be loaded.
                must be a subset of ["left_wrist", "right_wrist", "head"]
            local_only (bool): whether to only use local data (not download from HuggingFace).
                NOTE: set this to False and force_cache_sync to True if you want to force re-syncing the local cache with the remote dataset.
                For more details, please refer to the `force_cache_sync` argument in the base class.
            check_timestamp_sync (bool): whether to check timestamp synchronization between different modalities and the state/action data.
                While it is set to True in the original LeRobotDataset and is set to True here by default, it can be set to False to skip the check for faster loading.
                This will especially save time if you are loading the complete challenge demo dataset.
            chunk_streaming_using_keyframe (bool): whether to use chunk streaming mode for loading the dataset using keyframes.
                When this is enabled, the dataset will pseudo-randomly load data in chunks based on keyframes, allowing for faster access to the data.
                NOTE: As B1K challenge demos has GOP size of 250 frames for efficient storage, it is STRONGLY recommended to set this to True if you don't need true frame-level random access.
                When this is enabled, it is recommended to set shuffle to True for better randomness in chunk selection.
                We also enforce that segmentation instance ID videos can only be loaded in chunk_streaming_using_keyframe mode for faster access.
            shuffle (bool): whether to shuffle the chunks after loading. This ONLY applies in chunk streaming mode. Recommended to be set to True for better randomness in chunk selection.
            seed (int): random seed for shuffling chunks.
            fine_grained_level (int): fine-grained level of orchestrators to use for training.
            train_rgb_type (str): type of rgb to use for training.
            return_seg_instance (bool): whether to return seg instance.
            skill_list (list[str]):
                - Filter mode: e.g. ["move to", "pick up from"] (keep only matching `skill_description`).
                - Weight mode: e.g. ["all", "move to:0.5", "place in:0.2"] (default 1.0 when "all" is present).
        """
        Dataset.__init__(self)
        self.repo_id = repo_id
        self.root = Path(os.path.expanduser(str(root))) if root else HF_LEROBOT_HOME / repo_id
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.tolerance_s = tolerance_s
        self.revision = revision or CODEBASE_VERSION
        self.video_backend = video_backend or get_safe_default_codec()
        self.delta_indices = None
        self.batch_encoding_size = batch_encoding_size
        self.episodes_since_last_encoding = 0
        self.return_seg_instance = return_seg_instance
        self.train_rgb_type = train_rgb_type
        self.skill_list = skill_list

        # Unused attributes
        self.image_writer = None
        self.episode_buffer = None

        self.root.mkdir(exist_ok=True, parents=True)

        # ========== Customizations ==========
        self.seed = seed
        if modalities is None:
            modalities = ["rgb", "depth", "seg_instance_id"]
        if "seg_instance_id" in modalities:
            assert chunk_streaming_using_keyframe, "For the sake of data loading speed, please use chunk_streaming_using_keyframe=True when loading segmentation instance ID videos."
        if "depth" in modalities:
            assert self.video_backend == "pyav", (
                "Depth videos can only be decoded with the 'pyav' backend. "
                "Please set video_backend='pyav' when initializing the dataset."
            )
        if cameras is None:
            cameras = ["head", "left_wrist", "right_wrist"]
        self.task_names = set(tasks) if tasks is not None else set(TASK_NAMES_TO_INDICES.keys())
        self.task_indices = [TASK_NAMES_TO_INDICES[task] for task in self.task_names]
        # Load metadata
        self.meta = BehaviorLerobotDatasetMetadata(
            repo_id=self.repo_id,
            root=self.root,
            revision=self.revision,
            force_cache_sync=force_cache_sync,
            tasks=self.task_names,
            modalities=modalities,
            cameras=cameras,
        )
        # overwrite episode based on task
        all_episodes = load_jsonlines(self.root / EPISODES_PATH)
        # get the episodes grouped by task
        epi_by_task = defaultdict(list)
        for item in all_episodes:
            if item["episode_index"] // 1e4 in self.meta.tasks:
                epi_by_task[item["episode_index"] // 1e4].append(item["episode_index"])
        # sort and cherrypick episodes within each task
        for task_id, ep_indices in epi_by_task.items():
            epi_by_task[task_id] = sorted(ep_indices)
            if episodes is not None:
                epi_by_task[task_id] = [epi_by_task[task_id][i] for i in episodes if i < len(epi_by_task[task_id])]
        # now put episodes back together
        self.episodes = sorted([ep for eps in epi_by_task.values() for ep in eps])

        # Optional prefilter: drop episodes that cannot produce any samples under skill_list.
        # This reduces expensive parquet scanning/loading when training a rare skill.
        if not self._skill_list_is_trivial_all():
            before = len(self.episodes)
            kept = []
            dropped = 0
            for ep_idx in self.episodes:
                segs = None
                try:
                    # meta.orchestrators[ep_idx][1] is skill-level segments (built from annotations when available).
                    segs = self.meta.orchestrators.get(ep_idx, {}).get(1)
                except Exception:
                    segs = None
                if not segs:
                    # If we can't determine skill segments, keep the episode to avoid over-filtering.
                    kept.append(ep_idx)
                    continue
                ok = False
                for seg in segs:
                    try:
                        skill = seg.get("skill") or seg["task"]
                    except Exception:
                        continue
                    if float(skill_weight(skill, self.skill_list)) > 0.0:
                        ok = True
                        break
                if ok:
                    kept.append(ep_idx)
                else:
                    dropped += 1
            self.episodes = kept
            after = len(self.episodes)
            if dropped > 0:
                logger.info(
                    "Prefiltered episodes by skill_list: %d -> %d (dropped %d). skill_list=%s",
                    before,
                    after,
                    dropped,
                    self.skill_list,
                )
        # handle streaming mode and shuffling of episodes
        self._chunk_streaming_using_keyframe = chunk_streaming_using_keyframe
        if self._chunk_streaming_using_keyframe:
            if not shuffle:
                logger.warning(
                    "chunk_streaming_using_keyframe mode is enabled but shuffle is set to False. This may lead to less randomness in chunk selection."
                )
            self.chunks = self._get_keyframe_chunk_indices()
            # Now, we randomly permute the episodes if shuffle is True
            if shuffle:
                self.current_streaming_chunk_idx = None
                self.current_streaming_frame_idx = None
            else:
                self.current_streaming_chunk_idx = 0
                self.current_streaming_frame_idx = self.chunks[self.current_streaming_chunk_idx][0]
            self.obs_loaders = dict()
            self._should_obs_loaders_reload = True
        # record the positional index of each episode index within self.episodes
        self.episode_data_index_pos = {ep_idx: i for i, ep_idx in enumerate(self.episodes)}
        logger.info(f"Total episodes: {len(self.episodes)}")
        # ====================================

        if self.episodes is not None and self.meta._version >= packaging.version.parse("v2.1") and self.meta.stats is not None:
            episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes if ep_idx in self.meta.episodes_stats]
            if episodes_stats:
                self.stats = aggregate_stats(episodes_stats)

        # Load actual data
        try:
            if force_cache_sync:
                raise FileNotFoundError
            for fpath in self.get_episodes_file_paths():
                assert (self.root / fpath).is_file(), f"Missing file: {self.root / fpath}"
            self.hf_dataset = self.load_hf_dataset()
        except (AssertionError, FileNotFoundError, NotADirectoryError) as e:
            if local_only:
                raise e
            self.revision = get_safe_version(self.repo_id, self.revision)
            self.download_episodes(download_videos)
            self.hf_dataset = self.load_hf_dataset()

        self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)

        # Check timestamps
        if check_timestamp_sync:
            timestamps = th.stack(self.hf_dataset["timestamp"]).numpy()
            episode_indices = th.stack(self.hf_dataset["episode_index"]).numpy()
            ep_data_index_np = {k: t.numpy() for k, t in self.episode_data_index.items()}
            check_timestamps_sync(timestamps, episode_indices, ep_data_index_np, self.fps, self.tolerance_s)

        # Setup delta_indices
        if self.delta_timestamps is not None:
            check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

        self.prepare_task(fine_grained_level)

        self.omnigibson_mapping = {ep_idx: defaultdict(dict) for ep_idx in self.episodes}
        self._skill_stream = None
        self._skill_stream_rng = None
        self._skill_stream_rng_worker_id = None
        self._skill_stream_range_end = None
        if self._chunk_streaming_using_keyframe:
            # Skill filtering/weighting can be extremely slow if implemented as "decode frames then discard".
            # Build segment-range sampling so we can stream within skill segments (seek only at segment boundaries).
            self._maybe_build_skill_stream()

    def _skill_list_is_trivial_all(self) -> bool:
        # Default case: no filtering/weighting.
        return bool(self.skill_list) and len(self.skill_list) == 1 and str(self.skill_list[0]).strip() == "all"

    def _maybe_build_skill_stream(self) -> None:
        """
        Precompute weighted skill-segment ranges for streaming when skill_list filtering is active.

        When skill_list is restrictive (e.g. only "move to"), naive rejection sampling in
        streaming mode can end up decoding and discarding a huge number of frames.
        Instead, we stream directly within eligible skill segments:
          - sample a (episode, segment) range proportional to keep_prob * segment_length
          - seek video loaders once to the segment start
          - decode sequentially within the segment
        """
        if self._skill_list_is_trivial_all():
            self._skill_stream = None
            return

        # We rely on orchestrators (skill segments) to identify eligible frame ranges.
        # meta.orchestrators[ep_idx][1] corresponds to skill-level segments.
        ranges: list[tuple[int, int, int, int, bool]] = []  # (global_start, global_end, ep_idx, local_start, is_keyframe)
        weights: list[float] = []
        for ep_idx in self.episodes:
            try:
                segments = self.meta.orchestrators[ep_idx][1]
            except Exception:
                continue
            if not segments:
                continue

            for seg in segments:
                try:
                    skill = seg.get("skill") or seg["task"]
                    start = int(seg["start_frame"])
                    end = int(seg["end_frame"])
                except Exception:
                    continue
                if end < start:
                    continue
                keep_prob = float(skill_weight(skill, self.skill_list))
                if keep_prob <= 0.0:
                    continue
                length = end - start + 1
                seg_weight = keep_prob * float(length)
                if seg_weight <= 0.0:
                    continue

                ep_pos = self.episode_data_index_pos[ep_idx]
                global_from = int(self.episode_data_index["from"][ep_pos].item())
                global_to = int(self.episode_data_index["to"][ep_pos].item())
                global_start = global_from + max(0, start)
                global_end = min(global_to, global_from + end + 1)
                if global_end <= global_start:
                    continue
                local_start = max(0, start)
                is_keyframe = (local_start % 250) == 0
                ranges.append((global_start, global_end, ep_idx, local_start, is_keyframe))
                weights.append(seg_weight)

        if not ranges:
            logger.warning(
                "skill_list filtering requested but no eligible skill segments were found. "
                "Falling back to sequential streaming. skill_list=%s",
                self.skill_list,
            )
            self._skill_stream = None
            return

        weights_np = np.asarray(weights, dtype=np.float64)
        probs = weights_np / weights_np.sum()
        self._skill_stream = {
            "ranges": ranges,
            "probs": probs,
        }

    def _sample_skill_stream_range(self, *, rng: np.random.Generator) -> tuple[int, int, int, int, bool]:
        assert self._skill_stream is not None
        ranges: list[tuple[int, int, int, int, bool]] = self._skill_stream["ranges"]
        probs: np.ndarray = self._skill_stream["probs"]
        i = int(rng.choice(len(ranges), p=probs))
        return ranges[i]

    def prepare_task(self, fine_grained_level: int):
        """set train subtask mode for lerobot dataset"""
        self.fine_grained_level = fine_grained_level

        # calculate the start and end indices of each episode
        self.task_sizes = {}
        try:
            for ep_id, ep_orch in self.meta.orchestrators.items():
                self.task_sizes[ep_id] = [task_info["end_frame"] for task_info in ep_orch[fine_grained_level]]
        except Exception as e:
            print(f"[warn] {self.repo_id} failed to calculate episode subtask cumulate: {e}")

        print(f"prepare task with fine_grained_level {self.fine_grained_level} for {self.root}")

    def get_episodes_file_paths(self) -> list[str]:
        """
        Overwrite the original method to use the episodes indices instead of range(self.meta.total_episodes)
        """
        episodes = self.episodes if self.episodes is not None else list(self.meta.episodes.keys())
        fpaths = [str(self.meta.get_data_file_path(ep_idx)) for ep_idx in episodes]
        # append metainfo and language annotations
        fpaths += [str(self.meta.get_metainfo_path(ep_idx)) for ep_idx in episodes]
        # TODO: add this back once we have all the language annotations
        # fpaths += [str(self.meta.get_annotation_path(ep_idx)) for ep_idx in episodes]
        if len(self.meta.video_keys) > 0:
            video_files = [
                str(self.meta.get_video_file_path(ep_idx, vid_key))
                for vid_key in self.meta.video_keys
                for ep_idx in episodes
            ]
            fpaths += video_files

        return fpaths

    def download_episodes(self, download_videos: bool = True) -> None:
        """
        Overwrite base method to allow more flexible pattern matching.
        Here, we do coarse filtering based on tasks, cameras, and modalities.
        We do this instead of filename patterns to speed up pattern checking and download speed.
        """
        allow_patterns = []
        if set(self.task_indices) != set(TASK_NAMES_TO_INDICES.values()):
            for task in self.task_indices:
                allow_patterns.append(f"**/task-{task:04d}/**")
        if len(self.meta.modalities) != 3:
            for modality in self.meta.modalities:
                if len(self.meta.camera_names) != 3:
                    for camera in self.meta.camera_names:
                        allow_patterns.append(f"**/observation.images.{modality}.{camera}/**")
                else:
                    allow_patterns.append(f"**/observation.images.{modality}.*/**")
        elif len(self.meta.camera_names) != 3:
            for camera in self.meta.camera_names:
                allow_patterns.append(f"**/observation.images.*.{camera}/**")
        ignore_patterns = []
        if not download_videos:
            ignore_patterns.append("videos/")
        if set(self.task_indices) != set(TASK_NAMES_TO_INDICES.values()):
            for task in set(TASK_NAMES_TO_INDICES.values()).difference(self.task_indices):
                ignore_patterns.append(f"**/task-{task:04d}/**")

        allow_patterns = None if allow_patterns == [] else allow_patterns
        ignore_patterns = None if ignore_patterns == [] else ignore_patterns
        self.pull_from_repo(allow_patterns=allow_patterns, ignore_patterns=ignore_patterns)

    def pull_from_repo(
        self,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
    ) -> None:
        """
        Overwrite base class to increase max workers to num of CPUs - 2
        """
        logger.info(f"Pulling dataset {self.repo_id} from HuggingFace hub...")
        snapshot_download(
            self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            local_dir=self.root,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            max_workers=os.cpu_count() - 2,
        )

    def load_hf_dataset(self) -> datasets.Dataset:
        """hf_dataset contains all the observations, states, actions, rewards, etc."""
        if self.episodes is not None and len(self.episodes) == 0:
            raise ValueError(
                "No episodes selected after filtering (episodes=[]). "
                f"repo_id={self.repo_id!r} root={str(self.root)!r} tasks={self.tasks!r} skill_list={self.skill_list!r}. "
                "Check your --data.skill-list / --data.tasks / episodes_index and dataset contents."
            )

        if self.episodes is None:
            path = str(self.root / "data")
            hf_dataset = load_dataset("parquet", data_dir=path, split="train")
        else:
            files = [str(self.root / self.meta.get_data_file_path(ep_idx)) for ep_idx in self.episodes]
            hf_dataset = load_dataset("parquet", data_files=files, split="train")

        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def __getitem__(self, idx) -> dict:
        if not self._chunk_streaming_using_keyframe:
            item = super().__getitem__(idx)
            item["task"] = self._get_fine_grained_task(item)
            ep_idx = item["episode_index"].item()
            frame_index = round(item["timestamp"].item() * self.fps)
            skill_end = self._get_skill_end_frame(ep_idx, frame_index)
            if skill_end is not None:
                self._mask_action_chunks_to_skill_end(item, frame_index, skill_end)
            return item

        # Skill streaming path: sample an eligible skill segment, seek once to its start,
        # then decode sequentially within the segment. This avoids both heavy rejection
        # sampling and expensive per-sample random seeking.
        if self._skill_stream is not None:
            worker_info = get_worker_info()
            worker_id = 0 if worker_info is None else worker_info.id
            if self._skill_stream_rng is None or self._skill_stream_rng_worker_id != worker_id:
                self._skill_stream_rng = np.random.default_rng(self.seed + worker_id)
                self._skill_stream_rng_worker_id = worker_id
            rng = self._skill_stream_rng

            if self._skill_stream_range_end is None or self.current_streaming_frame_idx is None:
                self.current_streaming_frame_idx = None

            if self.current_streaming_frame_idx is None or self._skill_stream_range_end is None:
                start, end, ep_idx, local_start, is_keyframe = self._sample_skill_stream_range(rng=rng)
                self.current_streaming_frame_idx = start
                self._skill_stream_range_end = end
                self.current_streaming_episode_idx = None
                self._should_obs_loaders_reload = True
                self._skill_stream_local_start = local_start
                self._skill_stream_is_keyframe = is_keyframe
                self._skill_stream_ep_idx = ep_idx
            elif self.current_streaming_frame_idx >= self._skill_stream_range_end:
                start, end, ep_idx, local_start, is_keyframe = self._sample_skill_stream_range(rng=rng)
                self.current_streaming_frame_idx = start
                self._skill_stream_range_end = end
                self.current_streaming_episode_idx = None
                self._should_obs_loaders_reload = True
                self._skill_stream_local_start = local_start
                self._skill_stream_is_keyframe = is_keyframe
                self._skill_stream_ep_idx = ep_idx
            else:
                ep_idx = self._skill_stream_ep_idx

            item = self.hf_dataset[self.current_streaming_frame_idx]
            item.pop("observation.task_info", None)
            ep_idx_from_item = item["episode_index"].item()
            if ep_idx_from_item != ep_idx:
                # Safety check: if indices ever desync, fall back to item-reported episode.
                ep_idx = ep_idx_from_item
                self._should_obs_loaders_reload = True
                self.current_streaming_episode_idx = None

            if self._should_obs_loaders_reload:
                for loader in self.obs_loaders.values():
                    loader.close()
                self.obs_loaders = dict()
                task_id = item["task_index"].item()
                self.current_streaming_episode_idx = ep_idx
                for vid_key in self.meta.video_keys:
                    kwargs = {}
                    if "seg_instance_id" in vid_key:
                        with open(
                            self.root / "meta/episodes" / f"task-{task_id:04d}" / f"episode_{ep_idx:08d}.json",
                        ) as f:
                            meta = json.load(f)
                            instance_id_mapping = json.loads(meta["ins_id_mapping"])
                            instance_id_mapping = {int(k): v for k, v in instance_id_mapping.items()}
                            self.omnigibson_mapping[ep_idx]["instance_id_mapping"] = instance_id_mapping
                            self.omnigibson_mapping[ep_idx]["unique_ins_ids"][vid_key.split(".")[-1]] = meta[
                                f"{ROBOT_CAMERA_NAMES['R1Pro'][vid_key.split('.')[-1]]}::unique_ins_ids"
                            ]
                            kwargs["id_list"] = th.tensor(
                                self.omnigibson_mapping[ep_idx]["unique_ins_ids"][vid_key.split(".")[-1]]
                            )
                    if "rgb" in vid_key:
                        kwargs["train_rgb_type"] = self.train_rgb_type
                    self.obs_loaders[vid_key] = iter(
                        OBS_LOADER_MAP[vid_key.split(".")[2]](
                            data_path=self.root,
                            task_id=task_id,
                            camera_id=vid_key.split(".")[-1],
                            demo_id=f"{ep_idx:08d}",
                            start_idx=self._skill_stream_local_start,
                            start_idx_is_keyframe=self._skill_stream_is_keyframe,
                            batch_size=1,
                            stride=1,
                            **kwargs,
                        )
                    )
                self._should_obs_loaders_reload = False

            if self.delta_indices is not None:
                query_indices, padding = self._get_query_indices(self.current_streaming_frame_idx, ep_idx)
                query_result = self._query_hf_dataset(query_indices)
                item = {**item, **padding}
                for key, val in query_result.items():
                    item[key] = val

            # load visual observations
            for key in self.meta.video_keys:
                try:
                    item[key] = next(self.obs_loaders[key])[0]
                except StopIteration:
                    # obs_loader exhausted before skill segment ended
                    # (annotation end_frame > actual video frames). Skip this episode.
                    logging.warning(
                        f"obs_loader exhausted early for ep_idx={ep_idx}, key={key}. "
                        "Skipping episode and resampling."
                    )
                    self.current_streaming_frame_idx = None
                    self._skill_stream_range_end = None
                    self._should_obs_loaders_reload = True
                    return self.__getitem__(idx)

                if self.return_seg_instance and "seg_instance_id" in key:
                    seg_instance, instance_mapping = instance_id_to_instance(
                        obs=item[key],
                        instance_id_mapping=self.omnigibson_mapping[ep_idx]["instance_id_mapping"],
                        unique_ins_ids=np.array(self.omnigibson_mapping[ep_idx]["unique_ins_ids"][key.split(".")[-1]]),
                    )
                    instance_mapping = {instance_name: id for id, instance_name in instance_mapping.items()}

                    frame_index = round(item["timestamp"].item() * self.fps)
                    sub_idx = bisect.bisect_right(self.task_sizes[ep_idx], frame_index, hi=len(self.task_sizes[ep_idx]) - 1)
                    skill_annotation = self.meta.annotations[ep_idx]["skill_annotation"]
                    relative_obj_names = _flatten_objs(skill_annotation[sub_idx].get("object_id") or [])
                    relative_obj_names = [str(o) for o in relative_obj_names if o is not None]
                    for i, relative_obj_name in enumerate(relative_obj_names):
                        instance_id = instance_mapping.get(relative_obj_name)
                        if instance_id is None:
                            continue
                        seg_instance[seg_instance == instance_id] = -(i + 1)
                    seg_instance[seg_instance > 0] = 0
                    seg_instance *= -1
                    item[key.replace("seg_instance_id", "seg_instance")] = seg_instance

            if self.image_transforms is not None:
                image_keys = self.meta.camera_keys
                for cam in image_keys:
                    item[cam] = self.image_transforms(item[cam])

            # Add task as a string + mask action beyond skill end.
            item["task"] = self._get_fine_grained_task(item)
            frame_index = round(item["timestamp"].item() * self.fps)
            skill_end = self._get_skill_end_frame(ep_idx, frame_index)
            if skill_end is not None:
                self._mask_action_chunks_to_skill_end(item, frame_index, skill_end)

            self.current_streaming_frame_idx += 1
            return item

        # Streaming mode: we will load the episode at the current streaming index, and then increment the index for next call
        # NOTE: skill_list filtering/weighting is applied in streaming mode by skipping frames.
        # Use a loop (not recursion) to avoid stack growth when most frames are filtered out.
        max_skip_attempts = 50_000
        skip_attempts = 0
        last_task_skill = None
        while True:
            # Randomize chunk index on first call
            if self.current_streaming_chunk_idx is None:
                worker_info = get_worker_info()
                worker_id = 0 if worker_info is None else worker_info.id
                num_workers = 1 if worker_info is None else worker_info.num_workers
                if not hasattr(self, "_active_chunks") or self._active_chunks is None:
                    indices = list(range(worker_id, len(self.chunks), num_workers))
                    worker_chunks = [self.chunks[i] for i in indices]
                    rng = np.random.default_rng(self.seed + worker_id)
                    rng.shuffle(worker_chunks)
                    self._active_chunks = worker_chunks
                rng = np.random.default_rng(self.seed + worker_id)
                self.current_streaming_chunk_idx = rng.integers(0, len(self._active_chunks)).item()
                self.current_streaming_frame_idx = self._active_chunks[self.current_streaming_chunk_idx][0]

            # Current chunk iterated, move to next chunk
            if self.current_streaming_frame_idx >= self._active_chunks[self.current_streaming_chunk_idx][1]:
                self.current_streaming_chunk_idx += 1
                # All data iterated, restart from beginning
                if self.current_streaming_chunk_idx >= len(self._active_chunks):
                    self.current_streaming_chunk_idx = 0
                self.current_streaming_frame_idx = self._active_chunks[self.current_streaming_chunk_idx][0]
                self._should_obs_loaders_reload = True

            item = self.hf_dataset[self.current_streaming_frame_idx]
            item.pop("observation.task_info")
            ep_idx = item["episode_index"].item()

            if self._should_obs_loaders_reload:
                for loader in self.obs_loaders.values():
                    loader.close()
                self.obs_loaders = dict()
                # reload video loaders for new episode
                self.current_streaming_episode_idx = ep_idx
                for vid_key in self.meta.video_keys:
                    kwargs = {}
                    task_id = item["task_index"].item()
                    if "seg_instance_id" in vid_key:
                        # load id list
                        with open(
                            self.root / "meta/episodes" / f"task-{task_id:04d}" / f"episode_{ep_idx:08d}.json",
                        ) as f:
                            meta = json.load(f)
                            instance_id_mapping = json.loads(meta["ins_id_mapping"])
                            instance_id_mapping = {int(k): v for k, v in instance_id_mapping.items()}
                            self.omnigibson_mapping[ep_idx]["instance_id_mapping"] = instance_id_mapping
                            self.omnigibson_mapping[ep_idx]["unique_ins_ids"][vid_key.split(".")[-1]] = meta[
                                f"{ROBOT_CAMERA_NAMES['R1Pro'][vid_key.split('.')[-1]]}::unique_ins_ids"
                            ]
                            kwargs["id_list"] = th.tensor(
                                self.omnigibson_mapping[ep_idx]["unique_ins_ids"][vid_key.split(".")[-1]]
                            )
                    if "rgb" in vid_key:
                        kwargs["train_rgb_type"] = self.train_rgb_type
                    self.obs_loaders[vid_key] = iter(
                        OBS_LOADER_MAP[vid_key.split(".")[2]](
                            data_path=self.root,
                            task_id=task_id,
                            camera_id=vid_key.split(".")[-1],
                            demo_id=f"{ep_idx:08d}",
                            start_idx=self._active_chunks[self.current_streaming_chunk_idx][2],
                            start_idx_is_keyframe=False,
                            batch_size=1,
                            stride=1,
                            **kwargs,
                        )
                    )
                self._should_obs_loaders_reload = False

            if self.delta_indices is not None:
                query_indices, padding = self._get_query_indices(self.current_streaming_frame_idx, ep_idx)
                query_result = self._query_hf_dataset(query_indices)
                item = {**item, **padding}
                for key, val in query_result.items():
                    item[key] = val

            last_task_skill = self._get_current_task_skill(item)
            weight = skill_weight(last_task_skill, self.skill_list)
            if random.choices([True, False], weights=[weight, 1 - weight])[0]:
                break

            # Skip this frame (and advance video iterators to keep alignment).
            self.current_streaming_frame_idx += 1
            for key in self.meta.video_keys:
                next(self.obs_loaders[key])[0]
            skip_attempts += 1
            if skip_attempts >= max_skip_attempts:
                raise RuntimeError(
                    "Exceeded max skip attempts while applying skill_list filtering/weighting. "
                    f"skill_list={self.skill_list}, last_task_skill={last_task_skill!r}"
                )

        # load visual observations
        for key in self.meta.video_keys:
            item[key] = next(self.obs_loaders[key])[0]

            if self.return_seg_instance and "seg_instance_id" in key:
                seg_instance, instance_mapping = instance_id_to_instance(
                    obs=item[key],
                    instance_id_mapping=self.omnigibson_mapping[ep_idx]["instance_id_mapping"],
                    unique_ins_ids=np.array(self.omnigibson_mapping[ep_idx]["unique_ins_ids"][key.split(".")[-1]]),
                )
                instance_mapping = {instance_name: id for id, instance_name in instance_mapping.items()}

                frame_index = round(item["timestamp"].item() * self.fps)
                sub_idx = bisect.bisect_right(self.task_sizes[ep_idx], frame_index, hi=len(self.task_sizes[ep_idx]) - 1)
                skill_annotation = self.meta.annotations[ep_idx]["skill_annotation"]
                relative_obj_names = _flatten_objs(skill_annotation[sub_idx].get("object_id") or [])
                # Be robust to missing / stale instance names.
                relative_obj_names = [str(o) for o in relative_obj_names if o is not None]
                for i, relative_obj_name in enumerate(relative_obj_names):
                    instance_id = instance_mapping.get(relative_obj_name)
                    if instance_id is None:
                        continue
                    seg_instance[seg_instance == instance_id] = -(i + 1)
                seg_instance[seg_instance > 0] = 0
                seg_instance *= -1
                item[key.replace("seg_instance_id", "seg_instance")] = seg_instance

        if self.image_transforms is not None:
            image_keys = self.meta.camera_keys
            for cam in image_keys:
                item[cam] = self.image_transforms(item[cam])

        # Add task as a string
        item["task"] = self._get_fine_grained_task(item)
        # Mask action chunk beyond current skill end so model learns "stop" not next-skill actions
        ep_idx = item["episode_index"].item()
        frame_index = round(item["timestamp"].item() * self.fps)
        skill_end = self._get_skill_end_frame(ep_idx, frame_index)
        if skill_end is not None:
            self._mask_action_chunks_to_skill_end(item, frame_index, skill_end)
        self.current_streaming_frame_idx += 1

        return item

    def _get_current_task_skill(self, item: dict) -> str:
        ep_idx = item["episode_index"].item()
        frame_index = round(item["timestamp"].item() * self.fps)
        sub_idx = bisect.bisect_right(self.task_sizes[ep_idx], frame_index, hi=len(self.task_sizes[ep_idx]) - 1)
        seg = self.meta.orchestrators[ep_idx][1][sub_idx]
        # Prefer canonical skill label (annotation skill_description) when available.
        return seg.get("skill") or seg["task"]

    def _get_fine_grained_task(self, item: dict) -> str:
        ep_idx = item["episode_index"].item()
        task_idx = item["task_index"].item()
        frame_index = round(item["timestamp"].item() * self.fps)
        try:
            sub_idx = bisect.bisect_right(self.task_sizes[ep_idx], frame_index, hi=len(self.task_sizes[ep_idx]) - 1)
            task_text = self.meta.orchestrators[ep_idx][self.fine_grained_level][sub_idx]["task"]

        except Exception as e:
            logger.warning(
                "%s fine_grained_level=%d failed to get subtask (fallback to global task): ep_idx=%s task_idx=%s frame=%s error=%s",
                self.repo_id,
                self.fine_grained_level,
                ep_idx,
                task_idx,
                frame_index,
                e,
            )
            task_text = self.meta.tasks[task_idx]
        return task_text

    def _get_skill_end_frame(self, ep_idx: int, frame_index: int) -> int | None:
        """Return the end_frame of the skill segment containing this frame, or None if not using segments."""
        if self.fine_grained_level < 1 or ep_idx not in self.task_sizes:
            return None
        try:
            sub_idx = bisect.bisect_right(
                self.task_sizes[ep_idx], frame_index, hi=len(self.task_sizes[ep_idx]) - 1
            )
            return self.meta.orchestrators[ep_idx][self.fine_grained_level][sub_idx]["end_frame"]
        except Exception:
            return None

    def _mask_action_chunks_to_skill_end(
        self, item: dict, frame_index: int, skill_end: int
    ) -> None:
        """
        For frames near the end of a skill segment, the action chunk would otherwise extend
        into the next skill. Overwrite chunk positions beyond skill_end with the action at
        skill_end (repeat last action of the skill) so the model learns to predict "stop"
        instead of the next skill's actions. Modifies item in place for keys in delta_indices.
        """
        if self.delta_indices is None:
            return
        for key in self.delta_indices:
            if key not in item:
                continue
            arr = item[key]
            try:
                delta_list = list(self.delta_indices[key])
            except Exception:
                continue
            H = len(delta_list)
            # Chunk shape must be (H, ...); skip if single-step or wrong shape
            arr_shape = getattr(arr, "shape", None) or (len(arr),)
            if not arr_shape or arr_shape[0] != H:
                continue
            end_offset = skill_end - frame_index
            if end_offset < 0 or end_offset >= H:
                continue
            # Clone so we don't mutate shared/cached data
            if hasattr(arr, "clone"):
                arr = arr.clone()
            else:
                arr = np.copy(arr)
            # For i > end_offset: chunk[i] = chunk[end_offset] (repeat last action of skill)
            last_action = arr[end_offset]
            for i in range(end_offset + 1, H):
                if hasattr(last_action, "clone"):
                    arr[i] = last_action.clone()
                else:
                    arr[i] = np.copy(last_action) if isinstance(arr, np.ndarray) else last_action.copy()
            item[key] = arr

    def _get_query_indices(self, idx: int, ep_idx: int) -> tuple[dict[str, list[int | bool]]]:
        ep_idx = self.episode_data_index_pos[ep_idx]
        ep_start = self.episode_data_index["from"][ep_idx]
        ep_end = self.episode_data_index["to"][ep_idx]
        query_indices = {
            key: [max(ep_start.item(), min(ep_end.item() - 1, idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {  # Pad values outside of current episode range
            f"{key}_is_pad": th.BoolTensor(
                [(idx + delta < ep_start.item()) | (idx + delta >= ep_end.item()) for delta in delta_idx]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, th.Tensor]:
        """Note: When using data workers (e.g. DataLoader with num_workers>0), do not call this function
        in the main process (e.g. by using a second Dataloader with num_workers=0). It will result in a
        Segmentation Fault. This probably happens because a memory reference to the video loader is created in
        the main process and a subprocess fails to access it.
        """
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
            frames = decode_video_frames(video_path, query_ts, self.tolerance_s, self.video_backend)
            item[vid_key] = frames.squeeze(0)

        return item

    def _get_keyframe_chunk_indices(self, chunk_size=250) -> list[tuple[int, int, int]]:
        """
        Divide each episode into chunks of data based on GOP of the data (here for B1K, GOP size is 250 frames).
        Args:
            chunk_size (int): size of each chunk in number of frames. Default is 250 for B1K. Should be the GOP size of the video data.
        Returns:
            List of tuples, where each tuple contains (start_index, end_index, local_start_index) for each chunk.
        """
        episode_lengths = {ep_idx: ep_dict["length"] for ep_idx, ep_dict in self.meta.episodes.items()}
        episode_lengths = [episode_lengths[ep_idx] for ep_idx in self.episodes]
        chunks = []
        offset = 0
        for L in episode_lengths:
            local_starts = list(range(0, L, chunk_size))
            local_ends = local_starts[1:] + [L]
            for ls, le in zip(local_starts, local_ends):
                chunks.append((offset + ls, offset + le, ls))
            offset += L
        return chunks


class BehaviorLerobotDatasetMetadata(LeRobotDatasetMetadata):
    """
    BehaviorLerobotDatasetMetadata extends LeRobotDatasetMetadata with the following customizations:
        1. Restricts the set of allowed modalities to {"rgb", "depth", "seg_instance_id"}.
        2. Restricts the set of allowed camera names to those defined in ROBOT_CAMERA_NAMES["R1Pro"].
        3. Provides a filtered view of dataset features, including only those corresponding to the selected modalities and camera names.
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        revision: str | None = None,
        force_cache_sync: bool = False,
        # === Customized arguments for BehaviorLeRobotDataset ===
        tasks: Iterable[str] = None,
        modalities: Iterable[str] = None,
        cameras: Iterable[str] = None,
    ):
        # ========== Customizations ==========
        self.task_name_candidates = set(tasks) if tasks is not None else set(TASK_NAMES_TO_INDICES.keys())
        self.modalities = set(modalities)
        self.camera_names = set(cameras)
        assert self.modalities.issubset(
            {"rgb", "depth", "seg_instance_id"}
        ), f"Modalities must be a subset of ['rgb', 'depth', 'seg_instance_id'], but got {self.modalities}"
        assert self.camera_names.issubset(
            ROBOT_CAMERA_NAMES["R1Pro"]
        ), f"Camera names must be a subset of {ROBOT_CAMERA_NAMES['R1Pro']}, but got {self.camera_names}"
        # ===================================

        self.repo_id = repo_id
        self.revision = revision or CODEBASE_VERSION
        self.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id

        try:
            if force_cache_sync:
                raise FileNotFoundError
            self.load_metadata()
        except (FileNotFoundError, NotADirectoryError):
            if is_valid_version(self.revision):
                self.revision = get_safe_version(self.repo_id, self.revision)

            (self.root / "meta").mkdir(exist_ok=True, parents=True)
            self.pull_from_repo(allow_patterns="meta/**", ignore_patterns="meta/episodes/**")
            self.load_metadata()

    def load_metadata(self):
        self.info = load_info(self.root)
        check_version_compatibility(self.repo_id, self._version, CODEBASE_VERSION)
        self.tasks, self.task_to_task_index, self.task_names = self.load_tasks(self.root)
        # filter based on self.task_name_candidates
        valid_task_indices = [idx for idx, name in self.task_names.items() if name in self.task_name_candidates]
        self.task_names = set([self.task_names[idx] for idx in valid_task_indices])
        self.tasks = {idx: self.tasks[idx] for idx in valid_task_indices}
        self.task_to_task_index = {v: k for k, v in self.tasks.items()}

        self.episodes = self.load_episodes(self.root)
        self.annotations = self.load_annotations(self.root)
        self.orchestrators = self.load_orchestrators(self.root)
        if self._version < packaging.version.parse("v2.1"):
            self.stats = self.load_stats(self.root)
            self.episodes_stats = backward_compatible_episodes_stats(self.stats, self.episodes)
        else:
            try:
                self.episodes_stats = self.load_episodes_stats(self.root)
                if self.episodes_stats:
                    self.stats = aggregate_stats(list(self.episodes_stats.values()))
                else:
                    logger.warning(
                        "Loaded %s but found no matching episode stats for the current episode set. "
                        "This usually happens when a local subset uses different `episode_index` values than the source "
                        "dataset. Metadata stats will be unavailable.",
                        EPISODES_STATS_PATH,
                    )
                    self.stats = None
            except FileNotFoundError:
                # Common when users create a local subset and forget to carry over meta/episodes_stats.jsonl.
                # Training can proceed (norm stats are loaded from `assets/`), but metadata stats will be missing.
                logger.warning(
                    "Missing %s under dataset root %s. "
                    "This file contains per-episode summary stats (min/max/mean/std/quantiles) for "
                    "`action` and `observation.state`, used for quick normalization-stats computation. "
                    "To fix permanently, copy it from the source dataset or regenerate it for your subset.",
                    EPISODES_STATS_PATH,
                    self.root,
                )
                self.episodes_stats = {}
                self.stats = None
        logger.info(f"Loaded metadata for {len(self.episodes)} episodes.")

    def load_tasks(self, local_dir: Path) -> tuple[dict, dict]:
        tasks = load_jsonlines(local_dir / TASKS_PATH)
        task_names = {item["task_index"]: item["task_name"] for item in sorted(tasks, key=lambda x: x["task_index"])}
        tasks = {item["task_index"]: item["task"] for item in sorted(tasks, key=lambda x: x["task_index"])}
        task_to_task_index = {task: task_index for task_index, task in tasks.items()}
        return tasks, task_to_task_index, task_names

    def load_episodes(self, local_dir: Path) -> dict:
        episodes = load_jsonlines(local_dir / EPISODES_PATH)
        return {
            item["episode_index"]: item
            for item in sorted(episodes, key=lambda x: x["episode_index"])
            if item["episode_index"] // 1e4 in self.tasks
        }

    def load_stats(self, local_dir: Path) -> dict[str, dict[str, np.ndarray]]:
        if not (local_dir / STATS_PATH).exists():
            return None
        stats = load_json(local_dir / STATS_PATH)
        return cast_stats_to_numpy(stats)

    def load_episodes_stats(self, local_dir: Path) -> dict:
        episodes_stats = load_jsonlines(local_dir / EPISODES_STATS_PATH)
        return {
            item["episode_index"]: cast_stats_to_numpy(item["stats"])
            for item in sorted(episodes_stats, key=lambda x: x["episode_index"])
            if item["episode_index"] in self.episodes
        }

    def load_annotations(self, local_dir: Path) -> dict:
        annotations = local_dir / ANNOTATIONS_PATH
        task_list = [task_id for task_id in annotations.iterdir() if task_id.is_dir()]
        return {
            int(episode.stem[8:]): load_json(episode)
            for task_id in task_list
            if int(task_id.name[5:]) in self.tasks
            for episode in sorted(task_id.iterdir())
        }

    def load_orchestrators(self, local_dir: Path) -> dict:
        orchestrators_path = local_dir / ORCHESTRATORS_PATH
        orchestrators = {
            episode_key: load_orchestrators_data(episode_data["tasks"][0], episode_data["length"])
            for episode_key, episode_data in sorted(self.episodes.items())
        }
        if orchestrators_path.exists():
            for task in self.tasks:
                if (orchestrators_path / f"task-{task:04d}").exists():
                    orchestrators.update(
                        {
                            int(episode.stem[8:]): load_orchestrators_data(
                                episode, self.episodes[int(episode.stem[8:])]["length"]
                            )
                            for episode in sorted((orchestrators_path / f"task-{task:04d}").iterdir())
                        }
                    )
        # When annotations have skill_annotation, build level 1/2 from them (no orchestrator files needed)
        for ep_id, ep_data in self.episodes.items():
            ann = self.annotations.get(ep_id)
            if not ann:
                continue
            skill_ann = ann.get("skill_annotation")
            if not skill_ann:
                continue
            ep_len = ep_data["length"]
            task_idx = ep_data["tasks"][0]
            level_0_task = self.tasks.get(task_idx, "task")
            orchestrators[ep_id] = build_orchestrator_levels_from_annotations(
                ep_id, ep_len, skill_ann, level_0_task
            )
        return orchestrators

    def get_annotation_path(self, ep_index: int) -> Path:
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.annotation_path.format(episode_chunk=ep_chunk, episode_index=ep_index)
        return Path(fpath)

    def get_metainfo_path(self, ep_index: int) -> Path:
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.metainfo_path.format(episode_chunk=ep_chunk, episode_index=ep_index)
        return Path(fpath)

    @property
    def annotation_path(self) -> str | None:
        """Formattable string for the annotation files."""
        return self.info["annotation_path"]

    @property
    def metainfo_path(self) -> str | None:
        """Formattable string for the metainfo files."""
        return self.info["metainfo_path"]

    @property
    def features(self) -> dict[str, dict]:
        """All features contained in the dataset."""
        features = dict()
        # pop not required features
        for name in self.info["features"].keys():
            if (
                name.startswith("observation.images.")
                and name.split(".")[-1] in self.camera_names
                and name.split(".")[-2] in self.modalities
            ):
                features[name] = self.info["features"][name]
        return features


def load_orchestrators_data(episode_path_or_level_0_task, episode_len):
    output_data = defaultdict(list)
    if type(episode_path_or_level_0_task) == str:
        for i in range(4):
            output_data[i] = [
                {
                    "task": episode_path_or_level_0_task,
                    "start_frame": 0,
                    "end_frame": episode_len - 1,
                }
            ]
        return output_data
    episode_path = episode_path_or_level_0_task
    task_annotated_data = load_json(episode_path / "task_annotated.json")
    level_0_task = task_annotated_data["cot_task_description"]
    output_data[0].append(
        {
            "task": level_0_task,
            "start_frame": 0,
            "end_frame": episode_len - 1,
        }
    )
    try:
        num_level1_tasks = len(task_annotated_data["cot_subtask_description_list"])
        for i in range(num_level1_tasks):
            subtask_data = load_json(episode_path / f"subtask_{i}_annotated.json")
            subtask = subtask_data["cot_subtask_description"]
            start_frame, end_frame = subtask_data["start_frame"], subtask_data["end_frame"] - 1
            skill = subtask_data["skill_description"]
            output_data[1].append(
                {
                    "task": skill,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                }
            )
            output_data[2].append(
                {
                    "task": subtask,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                }
            )
            for event_data_path in sorted(episode_path.glob(f"event_{i}_*_annotated.json")):
                event_data = load_json(event_data_path)
                event_task = event_data["subtask_answer_detailed"]
                start_frame, end_frame = event_data["start_frame"], event_data["end_frame"] - 1
                output_data[3].append(
                    {
                        "task": event_task,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                    }
                )
    except Exception as e:
        print(f"[warn] {episode_path} failed to load orchestrators data: {e}, falling back to default task.")
        for i in range(len(output_data)):
            output_data[i] = output_data[0]
    return output_data


def skill_weight(cur_skill, skill_list: list[str]) -> float:
    """
    Compute keep probability for a given `cur_skill`.

    Supported skill_list formats:
      - ["move to", "pick up from"] => filter mode (match => 1.0, else => 0.0)
      - ["all", "move to:0.5"] => weight mode with default 1.0 for unspecified skills
      - ["move to:0.2", "pick up from:1.0"] => weight mode with default 0.0 for unspecified skills
    """
    if not skill_list:
        return 1.0
    default = 1.0 if "all" in skill_list else 0.0
    weights: dict[str, float] = {}
    for raw_item in skill_list:
        if raw_item == "all":
            continue
        item = str(raw_item).strip()
        if not item:
            continue
        if ":" in item:
            skill, weight_str = item.split(":", 1)
            skill = skill.strip()
            try:
                weight = float(weight_str.strip())
            except ValueError as e:
                raise ValueError(f"Invalid skill_list entry {raw_item!r}; expected 'skill' or 'skill:weight'.") from e
        else:
            skill = item
            weight = 1.0
        weights[skill] = weight
    return weights.get(cur_skill, default)


class MultiBehaviorLeRobotDataset:
    def __init__(self, datasets: list[BehaviorLeRobotDataset], sample_weights: list[float] | None = None):
        if sample_weights is None:
            sample_weights = [1.0 / len(datasets)] * len(datasets)
        assert len(datasets) == len(sample_weights), "Length of datasets and sample weights must be the same"
        if sum(sample_weights) != 1.0:
            sample_weights = [weight / sum(sample_weights) for weight in sample_weights]

        self.datasets = datasets
        self.sample_weights = sample_weights

    def __len__(self):
        return max(len(dataset) for dataset in self.datasets)

    def __getitem__(self, idx):
        index = np.random.choice(range(len(self.datasets)), p=self.sample_weights)
        return self.datasets[index][idx]
