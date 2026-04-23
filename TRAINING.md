# Training README

This file documents the default training/cache behavior used by this repository, especially for multi-node BEHAVIOR-1K training.

## Default Cache Behavior

If you do not set any cache-related environment variables, the code now prefers the shared mount:

```text
/vepfs-C/.cache/openpi
```

When `/vepfs-C` is not available, it falls back to:

```text
~/.cache/openpi
```

Current default cache layout:

- Shared download/cache root: `/vepfs-C/.cache/openpi`
- Hugging Face datasets cache: `/vepfs-C/.cache/openpi/hf_datasets`
- BEHAVIOR chunk/skill cache: `<behavior_dataset_root>/.openpi_cache/chunks`
  - Example: `/vepfs-C/dataset/Behavior-1k/.openpi_cache/chunks`
- RoboInter annotation offset cache: `<local_vqa_root>/.openpi_cache/robointer_offsets`
  - Example: `/vepfs-C/dataset/RoboInter-VQA/.openpi_cache/robointer_offsets`

The BEHAVIOR chunk cache stores reusable metadata such as:

- keyframe chunk indices
- skill-whitelist filtered chunk indices

The RoboInter offset cache stores:

- precomputed JSON record byte offsets for large annotation shards

This avoids re-scanning multi-GB annotation JSON files on every training start.

When multiple datasets are mixed, the loader now samples one dataset source per batch instead of mixing sources at per-sample granularity. This improves locality for dataset-specific decode paths such as RoboInter annotations and image archives.

This reduces repeated startup work when restarting training with the same dataset selection.

## Default BEHAVIOR Validation Behavior

Training now defaults to:

- `check_timestamp_sync=False` for BEHAVIOR training datasets

Reason:

- on the full BEHAVIOR cached parquet dataset, the timestamp synchronization check performs a very large full-column scan
- this is useful for dataset validation, but expensive for normal repeated training runs

If you are using a stable local/shared BEHAVIOR dataset that has already been validated, leaving this disabled is usually the right default.

If you need to re-enable it for debugging a new dataset export or path migration, set:

```python
DataConfig(
    ...,
    check_timestamp_sync=True,
)
```

or change the corresponding training config in `src/openpi/training/config.py`.

## Recommended Training Environment

For multi-node or repeated experiments, use a shared or persistent path for all of the following:

- BEHAVIOR dataset root
- Hugging Face datasets cache
- OpenPI shared cache
- FAST tokenizer directory

Recommended example:

```bash
export OPENPI_DATA_HOME=/vepfs-C/.cache/openpi
export OPENPI_FAST_TOKENIZER_PATH=/vepfs-C/Jiawei/fast
```

In most cases you do not need to set `OPENPI_DATA_HOME`, because the code already defaults to `/vepfs-C/.cache/openpi` when `/vepfs-C` exists. It is still useful to export it explicitly in job scripts for clarity and portability.

## Cache-Related Environment Variables

You can override the defaults with the following environment variables:

| Variable | Meaning |
| --- | --- |
| `OPENPI_DATA_HOME` | Root cache dir for OpenPI downloads and shared caches |
| `OPENPI_HF_DATASETS_CACHE` | Explicit cache dir for `datasets.load_dataset(...)` |
| `HF_DATASETS_CACHE` | Standard Hugging Face datasets cache override |
| `HF_HOME` | Hugging Face home dir; if set and `OPENPI_HF_DATASETS_CACHE` is unset, datasets cache becomes `$HF_HOME/datasets` |
| `OPENPI_BEHAVIOR_CACHE_DIR` | Override BEHAVIOR chunk metadata cache directory |
| `OPENPI_FAST_TOKENIZER_PATH` | Local FAST tokenizer path for offline environments |

Priority rules:

1. Explicit OpenPI env vars win.
2. Then Hugging Face env vars are used where relevant.
3. Otherwise the repository defaults are used.

## Precompute Reusable Caches

You can warm up caches before launching a real training job:

```bash
PYTHONPATH=src:packages/openpi-client/src \
.venv/bin/python scripts/precompute_data_cache.py \
  pi05_b1k-knowledge_insulation-vqa-joint-skill-pick-up-from \
  --fast-tokenizer-path /vepfs-C/Jiawei/fast
```

What this precompute step helps with:

- builds/reuses Hugging Face parquet dataset cache
- builds/reuses BEHAVIOR chunk index cache
- builds/reuses BEHAVIOR skill-filtered chunk cache

Optional stricter check:

```bash
PYTHONPATH=src:packages/openpi-client/src \
.venv/bin/python scripts/precompute_data_cache.py \
  <config_name> \
  --fast-tokenizer-path /path/to/fast \
  --touch-first-sample
```

This is slower, but it can catch broken local paths or sample decode problems earlier.

## New Machine or New Path

If you move to a new environment, decide first where caches should live.

### Case 1: New machine, still using `/vepfs-C`

Usually no code change is needed. Just make sure:

- `/vepfs-C` is mounted
- the dataset exists under the configured path
- the FAST tokenizer directory exists

Recommended:

```bash
export OPENPI_FAST_TOKENIZER_PATH=/vepfs-C/Jiawei/fast
```

Then run precompute once per node or shared environment.

### Case 2: No `/vepfs-C`, use another shared mount

For example:

```bash
export OPENPI_DATA_HOME=/mnt/shared/openpi-cache
export OPENPI_HF_DATASETS_CACHE=/mnt/shared/openpi-cache/hf_datasets
export OPENPI_BEHAVIOR_CACHE_DIR=/mnt/shared/openpi-cache/behavior_chunks
export OPENPI_FAST_TOKENIZER_PATH=/mnt/shared/models/fast
```

If your BEHAVIOR dataset root is also moved, update the training config accordingly, e.g. `behavior_dataset_root="/mnt/shared/dataset/Behavior-1k"`.

### Case 3: Personal workstation, only local disk available

Use local persistent directories explicitly:

```bash
export OPENPI_DATA_HOME=/data/openpi-cache
export OPENPI_HF_DATASETS_CACHE=/data/openpi-cache/hf_datasets
export OPENPI_FAST_TOKENIZER_PATH=/data/models/fast
```

This avoids accidentally writing large caches into a small home partition.

## Practical Notes

- If startup repeatedly shows `Generating train split`, first check whether the active HF cache path is really on persistent/shared storage.
- If startup repeatedly shows BEHAVIOR chunk filtering for the same config, check whether `<behavior_dataset_root>/.openpi_cache/chunks` is writable and preserved across runs.
- If offline, always set `OPENPI_FAST_TOKENIZER_PATH` or pass `--fast-tokenizer-path` to the precompute script.
- If you change `episodes`, `fine_grained_level`, dataset root, or `skill_list`, BEHAVIOR chunk cache keys will change and a new cache entry will be generated. This is expected.

## Suggested Job Script Snippet

```bash
export PYTHONPATH=src:packages/openpi-client/src
export OPENPI_DATA_HOME=/vepfs-C/.cache/openpi
export OPENPI_FAST_TOKENIZER_PATH=/vepfs-C/Jiawei/fast
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

PYTHONPATH=src:packages/openpi-client/src \
.venv/bin/python scripts/precompute_data_cache.py \
  pi05_b1k-knowledge_insulation-vqa-joint-skill-pick-up-from \
  --fast-tokenizer-path /vepfs-C/Jiawei/fast
```

Then start the actual training job.
