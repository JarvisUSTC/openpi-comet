from collections.abc import Iterator, Sequence
import logging
import multiprocessing
import os
import typing
from typing import Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


class VideoMemoryDataset(Dataset):
    """Wraps a dataset to provide K historical frames per camera.

    Maintains a per-episode frame buffer with LRU eviction. The returned item
    contains extra keys like ``{cam_key}_history`` with a list of K-1 historical
    frames (oldest first). If fewer than K-1 frames are available, the earliest
    frame is repeated.

    NOTE: This relies on the underlying BehaviorLeRobotDataset streaming mode
    which returns frames sequentially within 250-frame chunks (ignoring the
    ``idx`` parameter passed by DataLoader). The buffer therefore accumulates
    correctly within each chunk regardless of DataLoader shuffle settings.
    At chunk boundaries there is a ~(K-1)*stride frame warm-up where history
    is partially padded — this is expected and typically covers <5% of frames.
    """

    _DEFAULT_CAMERA_KEYS = (
        "observation.images.rgb.head",
        "observation.images.rgb.left_wrist",
        "observation.images.rgb.right_wrist",
    )
    _MAX_EPISODES_CACHED = 32

    def __init__(self, dataset: Dataset, num_frames: int, stride: int = 1, camera_keys: Sequence[str] | None = None):
        self._dataset = dataset
        self._num_frames = num_frames
        self._stride = max(1, stride)
        self._camera_keys = camera_keys or self._DEFAULT_CAMERA_KEYS
        from collections import OrderedDict
        self._buffers: "OrderedDict[int, dict[str, list]]" = OrderedDict()
        self._last_frame_idx: dict[int, int] = {}
        self._max_buffer = (num_frames - 1) * self._stride + 1
        self._stats_total = 0
        self._stats_valid = 0

    def _get_frame_idx(self, item) -> int:
        ts = item.get("timestamp")
        if ts is not None:
            return round(float(ts.item() if hasattr(ts, "item") else ts) * 30)
        return -1

    def __getitem__(self, index):
        item = self._dataset[index]
        if self._num_frames <= 1:
            return item

        ep_idx = item.get("episode_index")
        if hasattr(ep_idx, "item"):
            ep_idx = ep_idx.item()

        frame_idx = self._get_frame_idx(item)

        if ep_idx in self._buffers:
            self._buffers.move_to_end(ep_idx)
            if ep_idx in self._last_frame_idx and frame_idx >= 0:
                gap = abs(frame_idx - self._last_frame_idx[ep_idx])
                if gap > self._stride + 1:
                    self._buffers[ep_idx] = {}
        else:
            self._buffers[ep_idx] = {}
            while len(self._buffers) > self._MAX_EPISODES_CACHED:
                oldest_key = next(iter(self._buffers))
                del self._buffers[oldest_key]
                self._last_frame_idx.pop(oldest_key, None)

        if frame_idx >= 0:
            self._last_frame_idx[ep_idx] = frame_idx

        for cam_key in self._camera_keys:
            if cam_key not in item:
                continue
            if cam_key not in self._buffers[ep_idx]:
                self._buffers[ep_idx][cam_key] = []

            buf = self._buffers[ep_idx][cam_key]
            frame = item[cam_key]
            buf.append(frame.clone() if hasattr(frame, "clone") else np.copy(frame))

            if len(buf) > self._max_buffer:
                buf.pop(0)

            K = self._num_frames
            available = buf[:-1]
            sampled = []
            valid_flags = []
            for i in range(K - 1, 0, -1):
                idx = len(available) - i * self._stride
                if idx < 0:
                    sampled.append(available[0] if available else buf[-1])
                    valid_flags.append(False)
                else:
                    sampled.append(available[idx])
                    valid_flags.append(True)

            item[f"{cam_key}_history"] = sampled
            item[f"{cam_key}_history_valid"] = valid_flags

        self._stats_total += 1
        if valid_flags and all(valid_flags):
            self._stats_valid += 1
        if self._stats_total > 0 and self._stats_total % 5000 == 0:
            pct = self._stats_valid / self._stats_total * 100
            logging.info(
                "VideoMemoryDataset: %d/%d (%.1f%%) samples have full history",
                self._stats_valid, self._stats_total, pct,
            )

        return item

    def __len__(self):
        return len(self._dataset)


def create_behavior_dataset(data_config: _config.DataConfig, action_horizon: int, video_memory_frames: int = 1, video_memory_stride_s: float = 1.0) -> Dataset:
    """Create a dataset for training."""
    from behavior.learning.datas.dataset import BehaviorLeRobotDataset

    args = {}

    if data_config.skill_list != ["all"]:
        args["skill_list"] = data_config.skill_list

    dataset = BehaviorLeRobotDataset(
        repo_id=data_config.repo_id,
        root=data_config.behavior_dataset_root,
        tolerance_s=data_config.tolerance_s,
        tasks=data_config.tasks,
        modalities=data_config.modalities,
        local_only=True,
        delta_timestamps={key: [t / 30.0 for t in range(action_horizon)] for key in data_config.action_sequence_keys},
        episodes=data_config.episodes_index,
        chunk_streaming_using_keyframe=True,
        shuffle=False,
        fine_grained_level=data_config.fine_grained_level,
        return_seg_instance=data_config.return_seg_instance,
        train_rgb_type=data_config.train_rgb_type,
        check_timestamp_sync=False,
        **args,
    )

    # fixed prompt hard coding
    dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotItem()])

    # MEM: wrap with video memory buffer if K > 1
    if video_memory_frames > 1:
        fps = 30  # B1K dataset FPS
        stride_frames = max(1, int(video_memory_stride_s * fps))
        dataset = VideoMemoryDataset(dataset, num_frames=video_memory_frames, stride=stride_frames)

    return dataset


def _get_behavior_dataset(dataset):
    """Unwrap nested dataset wrappers to find the underlying BehaviorLeRobotDataset."""
    from behavior.learning.datas.dataset import BehaviorLeRobotDataset

    d = dataset
    while d is not None and not isinstance(d, BehaviorLeRobotDataset):
        d = getattr(d, "_dataset", None)
    return d


def _rebalance_chunks_by_skill(dataset):
    """Rebalance dataset chunks so rare skills are oversampled (inverse-sqrt-frequency).

    Operates on BehaviorLeRobotDataset.chunks in-place. Only effective when
    chunk_streaming_using_keyframe=True.
    """
    import bisect
    import math
    from collections import defaultdict

    bds = _get_behavior_dataset(dataset)
    if bds is None:
        logging.warning("Skill resampling: could not find BehaviorLeRobotDataset; skipped.")
        return
    if not getattr(bds, "_chunk_streaming_using_keyframe", False) or not getattr(bds, "chunks", None):
        logging.warning("Skill resampling: dataset has no chunks; skipped.")
        return

    chunks = bds.chunks
    episodes = bds.episodes
    task_sizes = getattr(bds, "task_sizes", {})
    orchestrators = bds.meta.orchestrators
    fg_level = bds.fine_grained_level

    chunk_ep_list = []
    for ep_idx in episodes:
        L = bds.meta.episodes[ep_idx]["length"]
        n_chunks = len(range(0, L, 250))
        chunk_ep_list.extend([ep_idx] * n_chunks)

    if len(chunk_ep_list) != len(chunks):
        logging.warning(
            "Skill resampling: chunk/episode mismatch (%d vs %d); skipped.",
            len(chunk_ep_list), len(chunks),
        )
        return

    skill_per_chunk = []
    skill_frame_total = defaultdict(int)

    for i, chunk in enumerate(chunks):
        ep_idx = chunk_ep_list[i]
        local_mid = chunk[2] + (chunk[1] - chunk[0]) // 2

        skill = "unknown"
        if ep_idx in task_sizes and task_sizes[ep_idx]:
            sizes = task_sizes[ep_idx]
            si = bisect.bisect_right(sizes, local_mid, hi=len(sizes) - 1)
            try:
                skill = orchestrators[ep_idx][fg_level][si].get("task", "unknown")
            except Exception:
                pass

        skill_per_chunk.append(skill)
        skill_frame_total[skill] += chunk[1] - chunk[0]

    if not skill_frame_total:
        return

    max_frames = max(skill_frame_total.values())
    skill_weight = {
        skill: math.sqrt(max_frames / max(frames, 1))
        for skill, frames in skill_frame_total.items()
    }

    new_chunks = []
    skill_new_count = defaultdict(int)
    for i, chunk in enumerate(chunks):
        copies = max(1, round(skill_weight[skill_per_chunk[i]]))
        new_chunks.extend([chunk] * copies)
        skill_new_count[skill_per_chunk[i]] += copies

    logging.info(
        "Skill resampling: %d -> %d chunks (%.1fx)",
        len(chunks), len(new_chunks), len(new_chunks) / len(chunks),
    )
    for skill in sorted(skill_frame_total, key=skill_frame_total.get, reverse=True):
        old_c = sum(1 for s in skill_per_chunk if s == skill)
        logging.info(
            "  %-35s  frames=%10d  chunks %5d -> %5d  (weight=%.2f)",
            skill, skill_frame_total[skill], old_c, skill_new_count[skill], skill_weight[skill],
        )

    bds.chunks = new_chunks


def create_multi_behavior_dataset(
    data_configs: list[_config.DataConfig], sample_weights: list[float] | None, action_horizon: int
) -> Dataset:
    from behavior.learning.datas.dataset import MultiBehaviorLeRobotDataset

    datasets = [create_behavior_dataset(data_config, action_horizon) for data_config in data_configs]
    return MultiBehaviorLeRobotDataset(datasets, sample_weights=sample_weights)


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_behavior_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    seed_shift: int = 0,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    vm_frames = getattr(config.model, "video_memory_frames", 1)
    vm_stride = getattr(config.model, "video_memory_stride_s", 1.0)
    if isinstance(config.data, list):
        data_configs = [config_.create(config.assets_dirs, config.model) for config_ in config.data]
        dataset = create_multi_behavior_dataset(
            data_configs,
            sample_weights=config.sample_weights,
            action_horizon=config.model.action_horizon,
        )
        data_config = data_configs[0]
    else:
        data_config = config.data.create(config.assets_dirs, config.model)
        dataset = create_behavior_dataset(data_config, action_horizon=config.model.action_horizon, video_memory_frames=vm_frames, video_memory_stride_s=vm_stride)

    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    if getattr(config, "skill_resampling", False):
        _rebalance_chunks_by_skill(dataset)

    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed + seed_shift,
    )

    return DataLoaderImpl(data_config, data_loader)


def _inject_ddp_rank(dataset, rank: int, world_size: int) -> None:
    """Inject DDP rank/world_size into BehaviorLeRobotDataset instances.

    Worker processes spawned by DataLoader don't have torch.distributed
    initialized, so we store rank info as attributes on the dataset object
    before the DataLoader pickles it to workers.
    """
    from behavior.learning.datas.dataset import BehaviorLeRobotDataset, MultiBehaviorLeRobotDataset

    # Unwrap TransformedDataset layers
    inner = dataset
    while hasattr(inner, "_dataset"):
        inner = inner._dataset

    if isinstance(inner, MultiBehaviorLeRobotDataset):
        for ds in inner.datasets:
            sub = ds
            while hasattr(sub, "_dataset"):
                sub = sub._dataset
            if isinstance(sub, BehaviorLeRobotDataset):
                sub._ddp_rank = rank
                sub._ddp_world_size = world_size
    elif isinstance(inner, BehaviorLeRobotDataset):
        inner._ddp_rank = rank
        inner._ddp_world_size = world_size


def create_torch_behavior_data_loader(
    config: _config.TrainConfig,
    action_horizon: int,
    batch_size: int,
    *,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_workers: int = 0,
    seed: int = 0,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    vm_frames = getattr(config.model, "video_memory_frames", 1)
    vm_stride = getattr(config.model, "video_memory_stride_s", 1.0)
    if isinstance(config.data, list):
        data_configs = [config_.create(config.assets_dirs, config.model) for config_ in config.data]
        dataset = create_multi_behavior_dataset(
            data_configs,
            sample_weights=config.sample_weights,
            action_horizon=config.model.action_horizon,
        )
        data_config = data_configs[0]
    else:
        data_config = config.data.create(config.assets_dirs, config.model)
        dataset = create_behavior_dataset(data_config, action_horizon=config.model.action_horizon, video_memory_frames=vm_frames, video_memory_stride_s=vm_stride)

    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    if getattr(config, "skill_resampling", False):
        _rebalance_chunks_by_skill(dataset)

    sampler = None
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=True,
        )
        local_batch_size = batch_size // world_size
        # Inject DDP rank into the underlying BehaviorLeRobotDataset so that
        # worker processes (where torch.distributed is NOT initialized) can
        # still differentiate data across GPUs.
        _inject_ddp_rank(dataset, rank, world_size)
    else:
        local_batch_size = batch_size

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_workers=num_workers,
        seed=seed,
        framework="pytorch",
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        from behavior.learning.datas.dataset import MultiBehaviorLeRobotDataset

        if jax.process_count() > 1:
            logging.info(f"Subsetting dataset for process {jax.process_index()}.")
            if isinstance(dataset._dataset, MultiBehaviorLeRobotDataset):
                for dataset_index in range(len(dataset._dataset.datasets)):
                    indices = list(
                        range(
                            jax.process_index(),
                            len(dataset._dataset.datasets[dataset_index]._dataset.chunks),
                            jax.process_count(),
                        )
                    )
                    dataset._dataset.datasets[dataset_index]._dataset.chunks = [
                        dataset._dataset.datasets[dataset_index]._dataset.chunks[i] for i in indices
                    ]
                total_chunks = sum(
                    len(dataset._dataset.datasets[dataset_index]._dataset.chunks)
                    for dataset_index in range(len(dataset._dataset.datasets))
                )
                logging.info(f"[P{jax.process_index()}] After subset, Dataset has {total_chunks} chunks.")
            else:
                indices = list(range(jax.process_index(), len(dataset._dataset._dataset.chunks), jax.process_count()))
                dataset._dataset._dataset.chunks = [dataset._dataset._dataset.chunks[i] for i in indices]
                logging.info(
                    f"[P{jax.process_index()}] After subset, Dataset has {len(dataset._dataset._dataset.chunks)} chunks."
                )

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        # For multi-process JAX training, each process should have a different seed
        process_seed = seed + jax.process_index()
        generator = torch.Generator()
        generator.manual_seed(process_seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration as e:
                    logging.info(f"Stop Iteration ... {e}")
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
