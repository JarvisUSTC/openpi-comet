from collections.abc import Iterator, Sequence
import bisect
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import logging
import mmap
import multiprocessing
import os
import pathlib
import typing
from typing import Protocol, SupportsIndex, TypeVar
import zipfile

import jax
import jax.numpy as jnp
import numpy as np
import filelock
from PIL import Image
import torch

import openpi.models.model as _model
import openpi.shared.download as _download
import openpi.training.config as _config
import openpi.training.vqa_schema as vqa_schema
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


class WeightedMultiDataset(Dataset):
    def __init__(self, datasets: Sequence[Dataset], sample_weights: list[float] | None = None):
        if sample_weights is None:
            sample_weights = [1.0 / len(datasets)] * len(datasets)
        if len(datasets) != len(sample_weights):
            raise ValueError("Length of datasets and sample_weights must match.")
        probs = np.asarray(sample_weights, dtype=np.float64)
        self.datasets = list(datasets)
        self.sample_weights = (probs / probs.sum()).tolist()
        self._num_datasets = len(self.datasets)
        self._dataset_lengths = [len(dataset) for dataset in self.datasets]
        self._max_len = max(self._dataset_lengths)
        self._encoded_index_base = self._max_len

    def __getitem__(self, index: SupportsIndex):
        def _describe_dataset(dataset: object) -> str:
            parts: list[str] = []
            current = dataset
            for _ in range(8):
                parts.append(type(current).__name__)
                inner = getattr(current, "_dataset", None)
                if inner is None:
                    break
                current = inner

            repo_id = getattr(current, "repo_id", None) or getattr(current, "_repo_id", None)
            if repo_id is not None:
                parts.append(f"repo_id={repo_id}")
            return "->".join(parts)

        raw_index = index.__index__()
        if raw_index >= self._encoded_index_base:
            encoded_index = raw_index - self._encoded_index_base
            dataset_index = encoded_index % self._num_datasets
            dataset = self.datasets[dataset_index]
            item_index = (encoded_index // self._num_datasets) % len(dataset)
        else:
            dataset_index = np.random.choice(range(self._num_datasets), p=self.sample_weights)
            dataset = self.datasets[dataset_index]
            item_index = raw_index % len(dataset)
        try:
            return dataset[item_index]
        except Exception as e:
            raise ValueError(
                "WeightedMultiDataset sample fetch failed: "
                f"raw_index={raw_index} encoded_index_base={self._encoded_index_base} "
                f"dataset_index={dataset_index} item_index={item_index} "
                f"dataset={_describe_dataset(dataset)} dataset_len={len(dataset)}"
            ) from e

    def __len__(self) -> int:
        return self._max_len

    @property
    def dataset_lengths(self) -> list[int]:
        return self._dataset_lengths

    @property
    def encoded_index_base(self) -> int:
        return self._encoded_index_base


class WeightedBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Sample one dataset source per batch for better locality.

    When ``mix_sources=True``, each batch contains samples from multiple
    dataset sources proportionally to ``sample_weights``, which can produce
    more stable gradient estimates for joint training.
    """

    def __init__(
        self,
        dataset: WeightedMultiDataset,
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
        num_batches: int | None = None,
        mix_sources: bool = False,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        self._dataset = dataset
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._seed = seed
        self._num_batches = num_batches
        self._mix_sources = mix_sources
        self._steps_per_epoch = max(1, max(dataset.dataset_lengths) // batch_size)
        self._cursors = [0 for _ in dataset.dataset_lengths]
        self._epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self._seed + self._epoch)
        num_batches = self._steps_per_epoch if self._num_batches is None else self._num_batches
        num_datasets = len(self._dataset.datasets)
        dataset_lengths = self._dataset.dataset_lengths
        cursors = list(self._cursors)
        encoded_index_base = self._dataset.encoded_index_base
        weights = self._dataset.sample_weights

        for _ in range(num_batches):
            if self._mix_sources:
                # Distribute batch_size across dataset sources proportionally.
                counts = rng.multinomial(self._batch_size, weights)
                indices: list[int] = []
                for ds_idx, count in enumerate(counts):
                    if count == 0:
                        continue
                    ds_len = dataset_lengths[ds_idx]
                    if self._shuffle:
                        ds_indices = rng.integers(0, ds_len, size=count)
                    else:
                        start = cursors[ds_idx]
                        ds_indices = [(start + offset) % ds_len for offset in range(count)]
                        cursors[ds_idx] = (start + count) % ds_len
                    indices.extend(
                        encoded_index_base + int(i) * num_datasets + ds_idx for i in ds_indices
                    )
                # Shuffle within the batch so sources are interleaved.
                rng.shuffle(indices)
                yield indices
            else:
                dataset_index = int(rng.choice(num_datasets, p=weights))
                dataset_length = dataset_lengths[dataset_index]
                if self._shuffle:
                    sample_indices = rng.integers(0, dataset_length, size=self._batch_size)
                else:
                    start = cursors[dataset_index]
                    sample_indices = [(start + offset) % dataset_length for offset in range(self._batch_size)]
                    cursors[dataset_index] = (start + self._batch_size) % dataset_length
                yield [encoded_index_base + int(sample_index) * num_datasets + dataset_index for sample_index in sample_indices]

        self._cursors = cursors
        self._epoch += 1

    def __len__(self) -> int:
        return self._steps_per_epoch if self._num_batches is None else self._num_batches


class HuggingFaceVQADataset(Dataset):
    def __init__(self, data_config: _config.DataConfig):
        from datasets import load_dataset

        if data_config.hf_dataset_name is None:
            raise ValueError("hf_dataset_name must be provided for HuggingFace VQA datasets.")

        self._dataset = load_dataset(
            data_config.hf_dataset_name,
            data_config.hf_dataset_config_name,
            split=data_config.hf_dataset_split,
            cache_dir=str(_download.get_hf_datasets_cache_dir()),
        )
        self._indices = np.arange(len(self._dataset))
        self._image_column = data_config.hf_image_column
        self._question_column = data_config.hf_question_column
        self._answer_column = data_config.hf_answer_column

    def subset_for_process(self, process_index: int, process_count: int) -> None:
        self._indices = self._indices[process_index::process_count]

    def __getitem__(self, index: SupportsIndex) -> dict:
        sample = self._dataset[int(self._indices[index.__index__() % len(self._indices)])]
        return {
            "image": sample[self._image_column],
            "prompt": sample[self._question_column],
            "answer": sample[self._answer_column],
        }

    def __len__(self) -> int:
        return len(self._indices)


class LocalVQASchemaDataset(Dataset):
    def __init__(self, data_config: _config.DataConfig):
        if not data_config.local_vqa_schema_paths:
            raise ValueError("local_vqa_schema_paths must be provided for local VQA schema datasets.")
        self._samples = []
        for path in data_config.local_vqa_schema_paths:
            self._samples.extend(vqa_schema.load_samples_from_path(pathlib.Path(path)))
        self._indices = np.arange(len(self._samples))

    def subset_for_process(self, process_index: int, process_count: int) -> None:
        self._indices = self._indices[process_index::process_count]

    def __getitem__(self, index: SupportsIndex) -> dict:
        sample = self._samples[int(self._indices[index.__index__() % len(self._indices)])]
        return {
            "images": _load_vqa_images(sample.images),
            "prompt": sample.prompt,
            "answer": sample.answer,
            "sample_id": sample.sample_id,
            "task_family": sample.task_family,
            "task_name": sample.task_name,
            "source": sample.source,
            "metadata": sample.metadata,
        }

    def __len__(self) -> int:
        return len(self._indices)


class _SplitArchiveReader(io.BufferedIOBase):
    def __init__(self, parts: Sequence[pathlib.Path]):
        self._files = [part.open("rb") for part in parts]
        self._sizes = [part.stat().st_size for part in parts]
        self._offsets = [0]
        for size in self._sizes:
            self._offsets.append(self._offsets[-1] + size)
        self._position = 0
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        for f in self._files:
            f.close()
        self._closed = True
        super().close()

    @property
    def closed(self) -> bool:
        return self._closed

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            new_pos = offset
        elif whence == io.SEEK_CUR:
            new_pos = self._position + offset
        elif whence == io.SEEK_END:
            new_pos = self._offsets[-1] + offset
        else:
            raise ValueError(f"Unsupported whence: {whence}")
        self._position = max(0, new_pos)
        return self._position

    def read(self, size: int = -1) -> bytes:
        if size == 0:
            return b""
        total_size = self._offsets[-1]
        if size < 0 or self._position + size > total_size:
            size = total_size - self._position
        if size <= 0:
            return b""

        chunks: list[bytes] = []
        remaining = size
        while remaining > 0:
            file_index = bisect.bisect_right(self._offsets, self._position) - 1
            file_offset = self._position - self._offsets[file_index]
            to_read = min(remaining, self._sizes[file_index] - file_offset)
            f = self._files[file_index]
            f.seek(file_offset)
            chunk = f.read(to_read)
            if not chunk:
                break
            chunks.append(chunk)
            chunk_size = len(chunk)
            self._position += chunk_size
            remaining -= chunk_size
        return b"".join(chunks)


def _find_split_zip_parts(archive_path: pathlib.Path) -> list[pathlib.Path]:
    if archive_path.exists():
        return [archive_path]

    split_parts = []
    for part in archive_path.parent.glob(f"{archive_path.name}.*"):
        suffix = part.suffix.removeprefix(".")
        if suffix.isdigit():
            split_parts.append(part)
    return sorted(split_parts, key=lambda path: int(path.suffix.removeprefix(".")))


def _open_zip_archive(archive_path: str) -> zipfile.ZipFile:
    archive = pathlib.Path(archive_path)
    parts = _find_split_zip_parts(archive)
    if not parts:
        raise FileNotFoundError(f"Zip archive not found: {archive_path}")
    if len(parts) == 1 and parts[0] == archive:
        return zipfile.ZipFile(archive)
    return zipfile.ZipFile(_SplitArchiveReader(parts))


def _load_vqa_image(image_ref: vqa_schema.VQAImageRef, archive_cache: dict[str, zipfile.ZipFile] | None = None) -> np.ndarray:
    if image_ref.storage == "path":
        if image_ref.path is None:
            raise ValueError("Path-backed VQA image refs require `path`.")
        with Image.open(image_ref.path) as image:
            return np.asarray(image.convert("RGB"))

    if image_ref.storage == "zip":
        if image_ref.archive_path is None or image_ref.member_path is None:
            raise ValueError("Zip-backed VQA image refs require archive_path and member_path.")
        if archive_cache is None:
            with _open_zip_archive(image_ref.archive_path) as archive:
                with archive.open(image_ref.member_path) as f:
                    with Image.open(io.BytesIO(f.read())) as image:
                        return np.asarray(image.convert("RGB"))
        archive = archive_cache.get(image_ref.archive_path)
        if archive is None:
            archive = _open_zip_archive(image_ref.archive_path)
            archive_cache[image_ref.archive_path] = archive
        with archive.open(image_ref.member_path) as f:
            with Image.open(io.BytesIO(f.read())) as image:
                return np.asarray(image.convert("RGB"))

    raise ValueError(f"Unsupported VQA image storage: {image_ref.storage}")


def _load_vqa_images(image_refs: Sequence[vqa_schema.VQAImageRef]) -> list[np.ndarray]:
    archive_cache: dict[str, zipfile.ZipFile] = {}
    try:
        return [_load_vqa_image(image_ref, archive_cache=archive_cache) for image_ref in image_refs]
    finally:
        for archive in archive_cache.values():
            archive.close()


def _infer_robointer_image_path_from_annotation(annotation_relative_path: str, image_path: str) -> str | None:
    annotation_parts = pathlib.PurePosixPath(annotation_relative_path).parts
    if len(annotation_parts) < 5 or annotation_parts[1] != "meta":
        return None

    annotation_name = pathlib.PurePosixPath(annotation_parts[-1])
    if annotation_name.suffix != ".json":
        return None

    image_parts = pathlib.PurePosixPath(image_path.lstrip("./")).parts
    if not image_parts:
        return None

    inferred_parts = (
        annotation_parts[0],
        "image",
        *annotation_parts[2:-1],
        annotation_name.stem,
        *image_parts,
    )
    return pathlib.PurePosixPath(*inferred_parts).as_posix()


def _resolve_robointer_image_ref(
    root: pathlib.Path, image_path: str, *, annotation_relative_path: str | None = None
) -> vqa_schema.VQAImageRef:
    normalized = image_path.lstrip("./")
    candidate_paths = [normalized]
    if annotation_relative_path is not None:
        inferred = _infer_robointer_image_path_from_annotation(annotation_relative_path, normalized)
        if inferred is not None and inferred != normalized:
            candidate_paths.append(inferred)

    for candidate in candidate_paths:
        direct_path = root / candidate
        if direct_path.exists():
            return vqa_schema.VQAImageRef(storage="path", path=str(direct_path))

    saw_supported_candidate = False
    for candidate in candidate_paths:
        path_parts = pathlib.PurePosixPath(candidate).parts
        if len(path_parts) < 5:
            continue
        saw_supported_candidate = True

        for archive_prefix_len in range(len(path_parts) - 1, 3, -1):
            archive_path = root / pathlib.Path(*path_parts[:archive_prefix_len]).with_suffix(".zip")
            member_path = pathlib.PurePosixPath(*path_parts[archive_prefix_len - 1 :]).as_posix()
            if _find_split_zip_parts(archive_path):
                return vqa_schema.VQAImageRef(
                    storage="zip",
                    archive_path=str(archive_path),
                    member_path=member_path,
                )

    if not saw_supported_candidate:
        raise ValueError(
            f"Unsupported RoboInter image path: {image_path} "
            f"(annotation_relative_path={annotation_relative_path})"
        )

    raise FileNotFoundError(
        f"Could not resolve RoboInter image {image_path} under {root}. "
        f"Tried candidates={candidate_paths} annotation_relative_path={annotation_relative_path}"
    )


class _RoboInterAnnotationShard:
    def __init__(
        self,
        root: pathlib.Path,
        relative_path: str,
        *,
        max_samples: int | None,
    ):
        self._root = root
        self.relative_path = relative_path
        self.path = root / relative_path
        self._cache_dir = _download.get_dataset_cache_dir(root) / "robointer_offsets"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        path_parts = pathlib.PurePosixPath(relative_path).parts
        if len(path_parts) < 4:
            raise ValueError(f"Unexpected RoboInter annotation path: {relative_path}")
        self.task_family = path_parts[0].lower()
        self.task_name = path_parts[-1].removesuffix(".json")
        self._offsets = self._load_or_build_offsets(max_samples=max_samples)

    def _offset_cache_path(self, *, max_samples: int | None) -> pathlib.Path:
        stat = self.path.stat()
        cache_key = {
            "path": str(self.path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "max_samples": max_samples,
        }
        digest = hashlib.sha256(json.dumps(cache_key, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        return self._cache_dir / f"{digest[:16]}.npy"

    def _load_or_build_offsets(self, *, max_samples: int | None) -> list[tuple[int, int]]:
        cache_path = self._offset_cache_path(max_samples=max_samples)
        lock = filelock.FileLock(str(cache_path) + ".lock")
        with lock:
            if cache_path.exists():
                offsets = np.load(cache_path, allow_pickle=False)
                logging.info("Loaded RoboInter offsets from cache %s (count=%s)", cache_path, len(offsets))
                return [tuple(int(v) for v in pair) for pair in offsets.tolist()]

            offsets = self._build_offsets(max_samples=max_samples)
            tmp_path = cache_path.with_suffix(".tmp.npy")
            with tmp_path.open("wb") as f:
                np.save(f, np.asarray(offsets, dtype=np.int64), allow_pickle=False)
            tmp_path.replace(cache_path)
            logging.info("Saved RoboInter offsets to cache %s (count=%s)", cache_path, len(offsets))
            return offsets

    def _build_offsets(self, *, max_samples: int | None) -> list[tuple[int, int]]:
        offsets: list[tuple[int, int]] = []
        with self.path.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            pos = 0
            size = len(mm)
            while pos < size and chr(mm[pos]).isspace():
                pos += 1
            if pos >= size or mm[pos] != ord("["):
                raise ValueError(f"Expected top-level JSON array in {self.path}")
            pos += 1

            while pos < size:
                while pos < size and chr(mm[pos]).isspace():
                    pos += 1
                if pos < size and mm[pos] == ord("]"):
                    break
                if pos < size and mm[pos] == ord(","):
                    pos += 1
                    continue
                if pos >= size:
                    break

                start = pos
                depth = 0
                in_string = False
                escaped = False
                while pos < size:
                    char = mm[pos]
                    if in_string:
                        if escaped:
                            escaped = False
                        elif char == ord("\\"):
                            escaped = True
                        elif char == ord('"'):
                            in_string = False
                    else:
                        if char == ord('"'):
                            in_string = True
                        elif char in (ord("{"), ord("[")):
                            depth += 1
                        elif char in (ord("}"), ord("]")):
                            depth -= 1
                            if depth == 0:
                                pos += 1
                                offsets.append((start, pos))
                                break
                    pos += 1
                if max_samples is not None and len(offsets) >= max_samples:
                    break
            return offsets

    def __len__(self) -> int:
        return len(self._offsets)

    def load_record(self, local_index: int) -> dict:
        start, end = self._offsets[local_index]
        with self.path.open("rb") as f:
            f.seek(start)
            return json.loads(f.read(end - start))


class RoboInterVQADataset(Dataset):
    def __init__(self, data_config: _config.DataConfig):
        if data_config.local_vqa_root is None:
            raise ValueError("local_vqa_root must be provided for RoboInter-VQA datasets.")
        if not data_config.robointer_annotation_paths:
            raise ValueError("robointer_annotation_paths must be provided for RoboInter-VQA datasets.")

        self._root = pathlib.Path(data_config.local_vqa_root)
        # Build shard offsets in parallel to speed up cold-start initialization.
        max_samples = data_config.robointer_max_samples_per_annotation
        paths = data_config.robointer_annotation_paths
        if len(paths) > 1:
            with ThreadPoolExecutor(max_workers=min(8, len(paths))) as executor:
                self._shards = list(executor.map(
                    lambda p: _RoboInterAnnotationShard(self._root, p, max_samples=max_samples),
                    paths,
                ))
        else:
            self._shards = [
                _RoboInterAnnotationShard(self._root, p, max_samples=max_samples)
                for p in paths
            ]
        shard_lengths = [len(shard) for shard in self._shards]
        self._shard_offsets = np.cumsum([0, *shard_lengths])
        self._indices = np.arange(int(self._shard_offsets[-1]))

    def subset_for_process(self, process_index: int, process_count: int) -> None:
        self._indices = self._indices[process_index::process_count]

    def _decode_sample(self, global_index: int) -> dict:
        shard_index = int(np.searchsorted(self._shard_offsets, global_index, side="right") - 1)
        shard = self._shards[shard_index]
        local_index = global_index - int(self._shard_offsets[shard_index])
        record = shard.load_record(local_index)
        prompt, answer = vqa_schema.conversations_to_prompt_answer(record["conversations"])
        raw_images = record.get("images")
        if raw_images is None:
            raise ValueError(f"RoboInter sample {record.get('id')} is missing images.")
        if isinstance(raw_images, str):
            raw_images = [raw_images]
        image_refs = tuple(
            _resolve_robointer_image_ref(
                self._root,
                image_path,
                annotation_relative_path=shard.relative_path,
            )
            for image_path in raw_images
        )
        return {
            "images": _load_vqa_images(image_refs),
            "prompt": prompt,
            "answer": answer,
            "sample_id": str(record.get("id", f"{shard.relative_path}:{local_index}")),
            "task_family": shard.task_family,
            "task_name": shard.task_name,
            "source": "RoboInter-VQA",
            "metadata": {
                "annotation_path": shard.relative_path,
                "raw_task": record.get("task"),
                "ground_truth": record.get("gt"),
            },
        }

    def __getitem__(self, index: SupportsIndex) -> dict:
        global_index = int(self._indices[index.__index__() % len(self._indices)])
        return self._decode_sample(global_index)

    def __len__(self) -> int:
        return len(self._indices)


def create_behavior_dataset(data_config: _config.DataConfig, action_horizon: int) -> Dataset:
    """Create a dataset for training."""
    from behavior.learning.datas.dataset import BehaviorLeRobotDataset

    args = {}

    if data_config.skill_list != ["all"]:
        args["skill_list"] = data_config.skill_list

    dataset = BehaviorLeRobotDataset(
        repo_id=data_config.repo_id,
        root=data_config.behavior_dataset_root,
        hf_cache_dir=str(_download.get_hf_datasets_cache_dir()),
        tolerance_s=data_config.tolerance_s,
        check_timestamp_sync=data_config.check_timestamp_sync,
        tasks=data_config.tasks,
        modalities=data_config.modalities,
        local_only=True,
        delta_timestamps={key: [t / 30.0 for t in range(action_horizon)] for key in data_config.action_sequence_keys},
        episodes=data_config.episodes_index,
        chunk_streaming_using_keyframe=True,
        shuffle=True,
        fine_grained_level=data_config.fine_grained_level,
        return_seg_instance=data_config.return_seg_instance,
        train_rgb_type=data_config.train_rgb_type,
        **args,
    )

    # fixed prompt hard coding
    dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotItem()])

    return dataset


def create_multi_behavior_dataset(
    data_configs: list[_config.DataConfig], sample_weights: list[float] | None, action_horizon: int
) -> Dataset:
    datasets = [create_behavior_dataset(data_config, action_horizon) for data_config in data_configs]
    return WeightedMultiDataset(datasets, sample_weights=sample_weights)


def create_vqa_dataset(data_config: _config.DataConfig) -> Dataset:
    if data_config.dataset_type == "hf_vqa":
        return HuggingFaceVQADataset(data_config)
    if data_config.dataset_type == "local_vqa_schema":
        return LocalVQASchemaDataset(data_config)
    if data_config.dataset_type == "robointer_vqa":
        return RoboInterVQADataset(data_config)
    raise ValueError(f"Unsupported VQA dataset type: {data_config.dataset_type}")


def create_dataset(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    *,
    num_samples: int,
) -> Dataset:
    if data_config.dataset_type == "fake":
        return FakeDataset(model_config, num_samples=num_samples)
    if data_config.dataset_type in {"hf_vqa", "local_vqa_schema", "robointer_vqa"}:
        return create_vqa_dataset(data_config)
    return create_behavior_dataset(data_config, action_horizon=model_config.action_horizon)


def create_mixed_dataset(
    data_configs: Sequence[_config.DataConfig],
    model_config: _model.BaseModelConfig,
    *,
    sample_weights: list[float] | None,
    num_samples: int,
    skip_norm_stats: bool = False,
) -> Dataset:
    datasets = [
        transform_dataset(create_dataset(data_config, model_config, num_samples=num_samples), data_config, skip_norm_stats=skip_norm_stats)
        for data_config in data_configs
    ]
    if len(datasets) == 1:
        return datasets[0]
    return WeightedMultiDataset(datasets, sample_weights=sample_weights)


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.requires_norm_stats and data_config.repo_id != "fake" and not skip_norm_stats:
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
    if data_config.requires_norm_stats and data_config.repo_id != "fake" and not skip_norm_stats:
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
    data_configs = (
        [config_.create(config.assets_dirs, config.model) for config_ in config.data]
        if isinstance(config.data, list)
        else [config.data.create(config.assets_dirs, config.model)]
    )
    dataset = create_mixed_dataset(
        data_configs,
        config.model,
        sample_weights=config.sample_weights,
        num_samples=config.batch_size,
        skip_norm_stats=skip_norm_stats,
    )
    data_config = data_configs[0]

    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed + seed_shift,
        mix_sources=config.mix_sources,
    )

    return DataLoaderImpl(data_config, data_loader)


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
    data_configs = (
        [config_.create(config.assets_dirs, config.model) for config_ in config.data]
        if isinstance(config.data, list)
        else [config.data.create(config.assets_dirs, config.model)]
    )
    dataset = create_mixed_dataset(
        data_configs,
        config.model,
        sample_weights=config.sample_weights,
        num_samples=batch_size,
        skip_norm_stats=skip_norm_stats,
    )
    data_config = data_configs[0]

    sampler = None
    if torch.distributed.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
            shuffle=shuffle,
            drop_last=True,
        )
        local_batch_size = batch_size // torch.distributed.get_world_size()
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
        mix_sources=config.mix_sources,
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
        mix_sources: bool = False,
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
        if jax.process_count() > 1:
            logging.info(f"Subsetting dataset for process {jax.process_index()}.")
            _subset_dataset_for_process(dataset, jax.process_index(), jax.process_count())

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

        collate_fn = _collate_fn if framework != "pytorch" else _torch_collate_fn
        extra_kwargs: dict[str, typing.Any] = {}
        if num_workers > 0:
            # Reduce the chance of dataloader starvation. Only valid when num_workers > 0.
            extra_kwargs["prefetch_factor"] = 2
        if framework == "pytorch":
            # Enables faster host->device transfers when the training step runs on CUDA.
            extra_kwargs["pin_memory"] = torch.cuda.is_available()

        # If we spawn dataloader workers, ensure they do not try to initialize JAX on GPU.
        # The main process already imported JAX, so changing these env vars here won't
        # affect the training process, but will be inherited by the spawned workers.
        prev_jax_platforms: str | None = None
        prev_xla_prealloc: str | None = None
        prev_xla_alloc: str | None = None
        if num_workers > 0:
            prev_jax_platforms = os.environ.get("JAX_PLATFORMS")
            prev_xla_prealloc = os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE")
            prev_xla_alloc = os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR")
            os.environ["JAX_PLATFORMS"] = "cpu"
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

        # For multi-process JAX training, each process should have a different seed
        process_seed = seed + jax.process_index()
        generator = torch.Generator()
        generator.manual_seed(process_seed)
        try:
            if isinstance(dataset, WeightedMultiDataset) and sampler is None:
                batch_sampler = WeightedBatchSampler(
                    dataset,
                    batch_size=local_batch_size,
                    shuffle=shuffle,
                    seed=process_seed,
                    num_batches=num_batches,
                    mix_sources=mix_sources,
                )
                self._data_loader = torch.utils.data.DataLoader(
                    typing.cast(torch.utils.data.Dataset, dataset),
                    batch_sampler=batch_sampler,
                    num_workers=num_workers,
                    multiprocessing_context=mp_context,
                    persistent_workers=num_workers > 0,
                    collate_fn=collate_fn,
                    worker_init_fn=_worker_init_fn,
                    **extra_kwargs,
                )
            else:
                self._data_loader = torch.utils.data.DataLoader(
                    typing.cast(torch.utils.data.Dataset, dataset),
                    batch_size=local_batch_size,
                    shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
                    sampler=sampler,
                    num_workers=num_workers,
                    multiprocessing_context=mp_context,
                    persistent_workers=num_workers > 0,
                    collate_fn=collate_fn,
                    worker_init_fn=_worker_init_fn,
                    drop_last=True,
                    generator=generator,
                    **extra_kwargs,
                )
        finally:
            if num_workers > 0:
                if prev_jax_platforms is None:
                    os.environ.pop("JAX_PLATFORMS", None)
                else:
                    os.environ["JAX_PLATFORMS"] = prev_jax_platforms
                if prev_xla_prealloc is None:
                    os.environ.pop("XLA_PYTHON_CLIENT_PREALLOCATE", None)
                else:
                    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = prev_xla_prealloc
                if prev_xla_alloc is None:
                    os.environ.pop("XLA_PYTHON_CLIENT_ALLOCATOR", None)
                else:
                    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = prev_xla_alloc

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


def _torch_collate_fn(items):
    """Collate the batch elements into batched torch tensors (enables pin_memory)."""
    return jax.tree.map(lambda *xs: torch.stack([torch.as_tensor(x) for x in xs], dim=0), *items)


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


def _subset_dataset_for_process(dataset, process_index: int, process_count: int) -> None:
    if isinstance(dataset, WeightedMultiDataset):
        for inner_dataset in dataset.datasets:
            _subset_dataset_for_process(inner_dataset, process_index, process_count)
        return
    if isinstance(dataset, TransformedDataset):
        _subset_dataset_for_process(dataset._dataset, process_index, process_count)
        return
    if hasattr(dataset, "subset_for_process"):
        dataset.subset_for_process(process_index, process_count)
        return
    if hasattr(dataset, "chunks"):
        indices = list(range(process_index, len(dataset.chunks), process_count))
        dataset.chunks = [dataset.chunks[i] for i in indices]
