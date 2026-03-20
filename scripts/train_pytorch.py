"""
PyTorch training entrypoint for PI0/PI05 with multi-GPU and multi-node (DDP) support.
This script mirrors the behavior of the JAX trainer (`scripts/train.py`) but runs
entirely in PyTorch using the `PI0Pytorch` model and your existing config/data
pipeline from `src/openpi/training/config.py` and `src/openpi/training/data_loader.py`.

Usage
Single GPU:
  python scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>
  Example:
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test --resume  # Resume from latest checkpoint
Multi-GPU (single node):
  torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>
  Example:
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume
Multi-Node Training:
	torchrun \
    --nnodes=<num_nodes> --nproc_per_node=<gpus_per_node> --node_rank=<rank_of_node> \
    --master_addr=<master_ip> --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>

"""

import dataclasses
import datetime
import gc
import logging
import os
import platform
import shutil
import time

import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.parallel
import tqdm
import wandb

import openpi.models.pi0_config
import openpi.models.tokenizer as _tokenizer
import openpi.models_pytorch.pi0_pytorch
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader

_PROMPT_LOG_INTERVAL = 10


def _validation_is_enabled(config: _config.TrainConfig) -> bool:
    if config.val_log_interval <= 0 or config.val_num_batches <= 0:
        return False
    return (config.val_repo_id is not None) or (config.val_episodes_index is not None)


def _override_factory_for_val(
    factory: _config.DataConfigFactory, config: _config.TrainConfig
) -> _config.DataConfigFactory:
    if config.val_repo_id is not None:
        factory = dataclasses.replace(factory, repo_id=config.val_repo_id)
    if config.val_episodes_index is not None:
        base = factory.base_config or _config.DataConfig()
        base = dataclasses.replace(base, episodes_index=config.val_episodes_index)
        factory = dataclasses.replace(factory, base_config=base)
    return factory


def _make_val_config(config: _config.TrainConfig) -> _config.TrainConfig:
    val_batch_size = config.batch_size if config.val_batch_size is None else config.val_batch_size
    if isinstance(config.data, list):
        val_data = [_override_factory_for_val(f, config) for f in config.data]
    else:
        val_data = _override_factory_for_val(config.data, config)
    return dataclasses.replace(config, batch_size=val_batch_size, data=val_data)


def _freeze_backbone(model, config, is_main: bool = True):
    """Apply freeze_filter from config: freeze matching params, train the rest.

    For non-LoRA configs (freeze_filter == nnx.Nothing) all params are trainable.
    For LoRA configs the base LLM params (except LoRA adapters) are frozen.
    """
    import re
    import flax.nnx as nnx

    raw_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

    freeze_filter = config.freeze_filter
    # nnx.Nothing means "freeze nothing" → all params trainable
    is_full_param = isinstance(freeze_filter, type) and issubclass(freeze_filter, nnx.Nothing)
    if not is_full_param:
        try:
            is_full_param = isinstance(freeze_filter, nnx.Nothing)
        except TypeError:
            pass

    total = 0
    frozen = 0

    if is_full_param:
        for param in raw_model.parameters():
            total += param.numel()
            param.requires_grad = True
    else:
        # Extract regex patterns from the Flax freeze_filter for PyTorch params.
        # Convention: LoRA configs freeze params matching "llm" but not "lora".
        freeze_patterns: list[re.Pattern] = []
        unfreeze_patterns: list[re.Pattern] = []
        _extract_patterns(freeze_filter, freeze_patterns, unfreeze_patterns)

        for name, param in raw_model.named_parameters():
            total += param.numel()
            should_freeze = any(p.search(name) for p in freeze_patterns)
            if should_freeze and unfreeze_patterns:
                should_freeze = not any(p.search(name) for p in unfreeze_patterns)
            param.requires_grad = not should_freeze
            if should_freeze:
                frozen += param.numel()

    trainable = total - frozen
    if is_main:
        logging.info(
            "Freeze setup: total=%.1fM, trainable=%.1fM (%.1f%%), frozen=%.1fM",
            total / 1e6, trainable / 1e6,
            100.0 * trainable / max(total, 1),
            frozen / 1e6,
        )


def _extract_patterns(flax_filter, freeze_patterns, unfreeze_patterns):
    """Best-effort extraction of regex patterns from Flax NNX filter objects."""
    import re

    try:
        import openpi.shared.nnx_utils as nnx_utils
    except ImportError:
        nnx_utils = None

    if nnx_utils and isinstance(flax_filter, nnx_utils.PathRegex):
        pat = flax_filter.pattern
        freeze_patterns.append(pat if isinstance(pat, re.Pattern) else re.compile(pat))
        return

    # Handle nnx.All(filter1, filter2, ...)
    if hasattr(flax_filter, "filters"):
        for f in flax_filter.filters:
            _extract_patterns(f, freeze_patterns, unfreeze_patterns)
        return

    # Handle nnx.Not(filter)
    if hasattr(flax_filter, "filter"):
        inner_freeze: list[re.Pattern] = []
        inner_unfreeze: list[re.Pattern] = []
        _extract_patterns(flax_filter.filter, inner_freeze, inner_unfreeze)
        unfreeze_patterns.extend(inner_freeze)
        freeze_patterns.extend(inner_unfreeze)
        return


def _decode_prompt_for_logging(observation, tokenizer: _tokenizer.PaligemmaTokenizer) -> str:
    tokens = observation.tokenized_prompt[0]
    token_mask = getattr(observation, "tokenized_prompt_mask", None)
    token_mask = None if token_mask is None else token_mask[0].to(torch.bool)

    token_ar_mask = getattr(observation, "token_ar_mask", None)
    token_ar_mask = None if token_ar_mask is None else token_ar_mask[0]
    if token_ar_mask is not None:
        prompt_mask = token_ar_mask == 0
        if token_mask is not None:
            prompt_mask = prompt_mask & token_mask
        return tokenizer.decode(tokens.detach().cpu().numpy(), mask=prompt_mask.detach().cpu().numpy())

    if token_mask is not None:
        return tokenizer.decode(tokens.detach().cpu().numpy(), mask=token_mask.detach().cpu().numpy())
    return tokenizer.decode(tokens.detach().cpu().numpy())


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    """Initialize wandb logging."""
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name, entity="haoranjia66-university-of-waterloo")
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
            entity="haoranjia66-university-of-waterloo",
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=datetime.timedelta(minutes=30),
        )

        # Set up debugging environment variables for DDP issues
        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return use_ddp, local_rank, device


def cleanup_ddp():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def set_seed(seed: int, local_rank: int):
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


def build_datasets(config: _config.TrainConfig):
    # Use the unified data loader with PyTorch framework
    # data_loader = _data_loader.create_data_loader(config, framework="pytorch", shuffle=True)
    data_loader = _data_loader.create_torch_behavior_data_loader(
        config,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        skip_norm_stats=False,
        shuffle=True,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    return data_loader, data_loader.data_config()


def build_val_loader(config: _config.TrainConfig):
    """Build a validation data loader from the training config."""
    val_config = _make_val_config(config)
    val_batch_size = config.batch_size if config.val_batch_size is None else config.val_batch_size
    val_loader = _data_loader.create_torch_behavior_data_loader(
        val_config,
        action_horizon=config.model.action_horizon,
        batch_size=val_batch_size,
        skip_norm_stats=False,
        shuffle=False,
        num_workers=0,
        seed=config.seed + 1000,
    )
    return val_loader


def reset_val_loader(val_loader):
    """Reset the streaming dataset pointer and VideoMemoryDataset buffer so
    that every validation pass reads exactly the same data."""
    from behavior.learning.datas.dataset import BehaviorLeRobotDataset

    def _unwrap(ds):
        """Recursively unwrap dataset wrappers, resetting each layer."""
        if isinstance(ds, _data_loader.VideoMemoryDataset):
            ds._buffers.clear()
            ds._last_frame_idx.clear()
            ds._stats_total = 0
            ds._stats_valid = 0
            _unwrap(ds._dataset)
        elif isinstance(ds, BehaviorLeRobotDataset):
            if hasattr(ds, "_active_chunks") and ds._active_chunks:
                ds.current_streaming_chunk_idx = 0
                ds.current_streaming_frame_idx = ds._active_chunks[0][0]
                ds._should_obs_loaders_reload = True
        elif hasattr(ds, "_dataset"):
            _unwrap(ds._dataset)

    torch_ds = val_loader._data_loader.torch_loader.dataset
    _unwrap(torch_ds)


@torch.no_grad()
def validate(model, val_loader, device, config):
    """Run validation and return a dict of metrics."""
    raw_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    was_training = raw_model.training
    raw_model.eval()

    flow_losses = []
    action_mses = []
    action_maes = []
    action_cosine_sims = []
    first_action_mses = []

    for batch_idx, (observation, actions) in enumerate(val_loader):
        if batch_idx >= config.val_num_batches:
            break

        observation = jax.tree.map(lambda x: x.to(device), observation)
        actions = actions.to(torch.float32).to(device)

        per_element_loss = raw_model(observation, actions)
        flow_losses.append(per_element_loss.mean().item())

        pred_actions = raw_model.sample_actions(device, observation, num_steps=config.val_denoise_steps)

        action_error = pred_actions - actions
        action_mses.append(torch.mean(action_error**2).item())
        action_maes.append(torch.mean(torch.abs(action_error)).item())

        pred_flat = pred_actions.reshape(pred_actions.shape[0], -1)
        gt_flat = actions.reshape(actions.shape[0], -1)
        cos_sim = torch.sum(pred_flat * gt_flat, dim=-1) / (
            torch.norm(pred_flat, dim=-1) * torch.norm(gt_flat, dim=-1) + 1e-8
        )
        action_cosine_sims.append(cos_sim.mean().item())

        first_action_mses.append(torch.mean((pred_actions[:, 0] - actions[:, 0]) ** 2).item())

        if batch_idx == 0:
            n_samples = min(3, pred_actions.shape[0])
            n_dims = min(8, pred_actions.shape[-1])
            for i in range(n_samples):
                pred_t0 = pred_actions[i, 0, :n_dims].cpu().tolist()
                gt_t0 = actions[i, 0, :n_dims].cpu().tolist()
                err_t0 = (pred_actions[i, 0, :n_dims] - actions[i, 0, :n_dims]).abs().cpu().tolist()
                pred_str = ", ".join(f"{v:+.4f}" for v in pred_t0)
                gt_str = ", ".join(f"{v:+.4f}" for v in gt_t0)
                err_str = ", ".join(f"{v:.4f}" for v in err_t0)
                logging.info("[VAL sample %d] pred: [%s]", i, pred_str)
                logging.info("[VAL sample %d]   gt: [%s]", i, gt_str)
                logging.info("[VAL sample %d]  err: [%s]  cos=%.4f", i, err_str, cos_sim[i].item())
            pred_mean = pred_actions[:n_samples, 0, :n_dims].mean(dim=0).cpu().tolist()
            gt_std = actions[:n_samples, 0, :n_dims].std(dim=0).cpu().tolist()
            logging.info("[VAL] pred_mean: [%s]", ", ".join(f"{v:+.4f}" for v in pred_mean))
            logging.info("[VAL]   gt_std: [%s]", ", ".join(f"{v:.4f}" for v in gt_std))

    if was_training:
        raw_model.train()

    return {
        "val_loss": float(np.mean(flow_losses)),
        "val/flow_loss": float(np.mean(flow_losses)),
        "val/action_mse": float(np.mean(action_mses)),
        "val/action_mae": float(np.mean(action_maes)),
        "val/action_cosine_sim": float(np.mean(action_cosine_sims)),
        "val/first_action_mse": float(np.mean(first_action_mses)),
    }


def get_model_state_dict(model):
    """Get state dict from model, handling DDP wrapper."""
    return (
        model.module.state_dict()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.state_dict()
    )


def get_model_parameters(model):
    """Get parameters from model, handling DDP wrapper."""
    return (
        model.module.parameters()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.parameters()
    )


def save_checkpoint(model, optimizer, global_step, config, is_main, data_config):
    """Save a checkpoint with model state, optimizer state, and metadata."""
    if not is_main:
        return

    # Only save if it's time to save or if it's the final step
    if (global_step % config.save_interval == 0 and global_step > 0) or global_step == config.num_train_steps - 1:
        # Create temporary directory for atomic checkpoint saving
        final_ckpt_dir = config.checkpoint_dir / f"{global_step}"
        tmp_ckpt_dir = config.checkpoint_dir / f"tmp_{global_step}"

        # Remove any existing temp directory and create new one
        if tmp_ckpt_dir.exists():
            shutil.rmtree(tmp_ckpt_dir)
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model state using safetensors (handle shared tensors)
        model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        safetensors.torch.save_model(model_to_save, tmp_ckpt_dir / "model.safetensors")

        # Save optimizer state using PyTorch format
        torch.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")

        # Save training metadata (avoid saving full config to prevent JAX/Flax compatibility issues)
        metadata = {
            "global_step": global_step,
            "config": dataclasses.asdict(config),
            "timestamp": time.time(),
        }
        torch.save(metadata, tmp_ckpt_dir / "metadata.pt")

        # save norm stats
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(tmp_ckpt_dir / "assets" / data_config.asset_id, norm_stats)

        # Atomically move temp directory to final location
        if final_ckpt_dir.exists():
            shutil.rmtree(final_ckpt_dir)
        tmp_ckpt_dir.rename(final_ckpt_dir)

        logging.info(f"Saved checkpoint at step {global_step} -> {final_ckpt_dir}")

        # Log checkpoint to wandb
        if config.wandb_enabled:
            wandb.log({"checkpoint_step": global_step}, step=global_step)


def load_checkpoint(model, optimizer, checkpoint_dir, device):
    """Load the latest checkpoint and return the global step."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]

    if not checkpoint_steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    latest_step = max(checkpoint_steps)
    ckpt_dir = checkpoint_dir / f"{latest_step}"

    # Clear memory before loading checkpoints
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "before_loading_checkpoint")

    try:
        # Load model state with error handling
        logging.info("Loading model state...")
        safetensors_path = ckpt_dir / "model.safetensors"

        if safetensors_path.exists():
            model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            safetensors.torch.load_model(model_to_load, safetensors_path, device=str(device), strict=False)
            logging.info("Loaded model state from safetensors format")
        else:
            raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")

        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_model")

        # Load optimizer state with error handling
        logging.info("Loading optimizer state...")
        optimizer_path = ckpt_dir / "optimizer.pt"

        if optimizer_path.exists():
            optimizer_state_dict = torch.load(optimizer_path, map_location=device, weights_only=False)
            logging.info("Loaded optimizer state from pt format")
        else:
            raise FileNotFoundError(f"No optimizer checkpoint found at {ckpt_dir}")

        optimizer.load_state_dict(optimizer_state_dict)
        del optimizer_state_dict
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_optimizer")

        # Load metadata
        logging.info("Loading metadata...")
        metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device, weights_only=False)
        global_step = metadata.get("global_step", latest_step)
        del metadata
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_metadata")

        logging.info(f"Successfully loaded all checkpoint components from step {latest_step}")
        return global_step

    except RuntimeError as e:
        if "out of memory" in str(e):
            # Clear memory and provide detailed error message
            torch.cuda.empty_cache()
            gc.collect()
            logging.error(f"Out of memory error while loading checkpoint: {e!s}")
            log_memory_usage(device, latest_step, "after_oom_error")
            raise RuntimeError(
                "Out of memory while loading checkpoint. Try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            ) from e
        raise


def get_latest_checkpoint_step(checkpoint_dir):
    """Get the latest checkpoint step number from a checkpoint directory."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    return max(checkpoint_steps) if checkpoint_steps else None


def log_memory_usage(device, step, phase="unknown"):
    """Log detailed memory usage information."""
    if not torch.cuda.is_available():
        return

    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    memory_free = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    memory_free = memory_free / 1e9

    # Get more detailed memory info
    memory_stats = torch.cuda.memory_stats(device)
    max_memory_allocated = memory_stats.get("allocated_bytes.all.peak", 0) / 1e9
    max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9

    # Get DDP info if available
    ddp_info = ""
    if dist.is_initialized():
        ddp_info = f" | DDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}"

    logging.info(
        f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB, free: {memory_free:.2f}GB, peak_allocated: {max_memory_allocated:.2f}GB, peak_reserved: {max_memory_reserved:.2f}GB{ddp_info}"
    )


def train_loop(config: _config.TrainConfig):
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    set_seed(config.seed, local_rank)

    # Initialize checkpoint directory and wandb
    resuming = False
    if config.resume:
        # Find checkpoint directory based on experiment name
        exp_checkpoint_dir = config.checkpoint_dir
        if exp_checkpoint_dir.exists():
            # Use validation to find the latest working checkpoint
            latest_step = get_latest_checkpoint_step(exp_checkpoint_dir)
            if latest_step is not None:
                resuming = True
                logging.info(
                    f"Resuming from experiment checkpoint directory: {exp_checkpoint_dir} at step {latest_step}"
                )
            else:
                raise FileNotFoundError(f"No valid checkpoints found in {exp_checkpoint_dir} for resume")
        else:
            raise FileNotFoundError(f"Experiment checkpoint directory {exp_checkpoint_dir} does not exist for resume")
    elif config.overwrite and config.checkpoint_dir.exists():
        shutil.rmtree(config.checkpoint_dir)
        logging.info(f"Overwriting checkpoint directory: {config.checkpoint_dir}")

    # Create checkpoint directory with experiment name
    if not resuming:
        # For new runs, create experiment-specific checkpoint directory
        exp_checkpoint_dir = config.checkpoint_dir
        exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created experiment checkpoint directory: {exp_checkpoint_dir}")
    else:
        # For resume, checkpoint_dir is already set to the experiment directory
        logging.info(f"Using existing experiment checkpoint directory: {config.checkpoint_dir}")

    # Initialize wandb (only on main process)
    if is_main:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # Build data loader using the unified data loader
    # Calculate effective batch size per GPU for DDP
    # For N GPUs, each GPU should get batch_size/N samples, so total across all GPUs is batch_size
    world_size = torch.distributed.get_world_size() if use_ddp else 1
    effective_batch_size = config.batch_size // world_size
    logging.info(
        f"Using batch size per GPU: {effective_batch_size} (total batch size across {world_size} GPUs: {config.batch_size})"
    )

    # Pass the original batch size to data loader - it will handle DDP splitting internally
    logging.info("[DEBUG] [1/6] Building datasets...")
    loader, data_config = build_datasets(config)
    logging.info("[DEBUG] [1/6] Datasets built OK.")

    # # Log sample images to wandb on first batch
    # if is_main and config.wandb_enabled and not resuming:
    #     # Create a separate data loader for sample batch to avoid consuming the main loader
    #     sample_data_loader = _data_loader.create_data_loader(config, framework="pytorch", shuffle=False)
    #     sample_batch = next(iter(sample_data_loader))
    #     # Convert observation and actions to torch tensors
    #     observation, actions = sample_batch
    #     sample_batch = observation.to_dict()
    #     sample_batch["actions"] = actions

    #     # Create sample images for wandb
    #     images_to_log = []
    #     # Get batch size from the first image tensor
    #     batch_size = next(iter(sample_batch["image"].values())).shape[0]
    #     for i in range(min(5, batch_size)):
    #         # Concatenate all camera views horizontally for this batch item
    #         # Convert from NCHW to NHWC format for wandb
    #         img_concatenated = torch.cat([img[i].permute(1, 2, 0) for img in sample_batch["image"].values()], axis=1)
    #         img_concatenated = img_concatenated.cpu().numpy()
    #         images_to_log.append(wandb.Image(img_concatenated))

    #     wandb.log({"camera_views": images_to_log}, step=0)

    #     # Clear sample batch from memory aggressively
    #     del sample_batch, observation, actions, images_to_log, img_concatenated
    #     del sample_data_loader  # Also delete the sample data loader
    #     gc.collect()
    #     if torch.cuda.is_available():
    #         torch.cuda.empty_cache()
    #     logging.info("Cleared sample batch and data loader from memory")

    # Build model
    if not isinstance(config.model, openpi.models.pi0_config.Pi0Config):
        # Convert dataclass to Pi0Config if needed
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
        )
    else:
        model_cfg = config.model
        # Update dtype to match pytorch_training_precision
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    logging.info("[DEBUG] [2/6] Creating PI0Pytorch model on CPU...")
    _t0 = time.time()
    model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg)
    logging.info("[DEBUG] [2/6] Model created on CPU in %.1fs, moving to %s ...", time.time() - _t0, device)
    _t0 = time.time()
    model = model.to(device)
    logging.info("[DEBUG] [2/6] Model moved to device in %.1fs.", time.time() - _t0)

    if hasattr(model, "gradient_checkpointing_enable"):
        enable_gradient_checkpointing = True
        model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing for memory optimization")
    else:
        enable_gradient_checkpointing = False
        logging.info("Gradient checkpointing is not supported for this model")

    # Log initial memory usage after model creation
    if is_main and torch.cuda.is_available():
        log_memory_usage(device, 0, "after_model_creation")

    # Enable memory optimizations for large-scale training
    if world_size >= 8:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set memory allocation configuration
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        logging.info("Enabled memory optimizations for 8+ GPU training")

    if use_ddp:
        logging.info("[DEBUG] [3/6] Wrapping model with DDP...")
        _t0 = time.time()
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
            static_graph=False,
        )
        logging.info("[DEBUG] [3/6] DDP wrapped in %.1fs.", time.time() - _t0)
    else:
        logging.info("[DEBUG] [3/6] Single GPU, skipping DDP.")

    # Load weights from weight_loader if specified (for fine-tuning)
    logging.info("[DEBUG] [4/6] pytorch_weight_path=%s", config.pytorch_weight_path)
    # NOTE: The JAX training pipeline supports `config.weight_loader` (Orbax/JAX params).
    # This PyTorch training script currently only supports loading PyTorch weights from
    # `config.pytorch_weight_path` (expects a `model.safetensors`).
    if config.pytorch_weight_path is None:
        try:
            import openpi.training.weight_loaders as _weight_loaders

            if not isinstance(config.weight_loader, _weight_loaders.NoOpWeightLoader):
                logging.warning(
                    "PyTorch training: `config.weight_loader=%s` will be ignored because "
                    "`pytorch_weight_path` is None. You are likely training from scratch. "
                    "Convert your JAX/Orbax checkpoint to a PyTorch `model.safetensors` and set "
                    "`--pytorch_weight_path`, or use the JAX training script instead.",
                    type(config.weight_loader).__name__,
                )
        except Exception:
            # Best-effort warning only; do not block training if optional deps are missing.
            pass
    if config.pytorch_weight_path is not None:
        logging.info(f"Loading weights from: {config.pytorch_weight_path}")

        model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
        model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        missing, unexpected = safetensors.torch.load_model(model_to_load, model_path, strict=False)
        if missing:
            logging.info(f"Missing keys: {len(missing)} keys")
            for k in missing[:10]:
                logging.info(f"  {k}")
        if unexpected:
            logging.warning(f"Unexpected keys in checkpoint: {unexpected}")
        logging.info(f"Loaded PyTorch weights from {config.pytorch_weight_path}")

    _freeze_backbone(model, config, is_main)

    # Optimizer + learning rate schedule from config
    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    logging.info("[DEBUG] [5/6] Creating optimizer...")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if is_main:
        trainable_numel = sum(p.numel() for p in trainable_params)
        logging.info("Trainable parameters: %.1fM (%d tensors)", trainable_numel / 1e6, len(trainable_params))
    optim = torch.optim.AdamW(
        trainable_params,
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )
    logging.info("[DEBUG] [5/6] Optimizer created.")

    # Load checkpoint if resuming
    global_step = 0
    if resuming:
        global_step = load_checkpoint(model, optim, config.checkpoint_dir, device)
        logging.info(f"Resumed training from step {global_step}")

    def lr_schedule(step: int):
        if step < warmup_steps:
            # Match JAX behavior: start from peak_lr / (warmup_steps + 1)
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        # cosine decay
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    model.train()
    start_time = time.time()
    infos = []  # Collect stats over log interval
    if is_main:
        logging.info(
            f"Running on: {platform.node()} | world_size={torch.distributed.get_world_size() if use_ddp else 1}"
        )
        logging.info(
            f"Training config: batch_size={config.batch_size}, effective_batch_size={effective_batch_size}, num_train_steps={config.num_train_steps}"
        )
        logging.info(f"Memory optimizations: gradient_checkpointing={enable_gradient_checkpointing}")
        logging.info(
            f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, decay_steps={decay_steps}, end_lr={end_lr:.2e}"
        )
        logging.info(
            f"Optimizer: {type(config.optimizer).__name__}, weight_decay={config.optimizer.weight_decay}, clip_norm={config.optimizer.clip_gradient_norm}"
        )
        logging.info("EMA is not supported for PyTorch training")
        logging.info(f"Training precision: {model_cfg.dtype}")

    prompt_tokenizer = _tokenizer.PaligemmaTokenizer(config.model.max_token_len) if is_main else None

    val_loader = None
    if is_main and _validation_is_enabled(config):
        val_loader = build_val_loader(config)
        logging.info(
            "Validation enabled: val_repo_id=%s val_episodes_index=%s val_batch_size=%s val_num_batches=%s val_log_interval=%s",
            config.val_repo_id,
            config.val_episodes_index,
            config.val_batch_size,
            config.val_num_batches,
            config.val_log_interval,
        )

    # Training loop - iterate until we reach num_train_steps
    pbar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not is_main)
        if is_main
        else None
    )

    logging.info("[DEBUG] [6/6] Entering training loop, fetching first batch...")
    _t0 = time.time()
    _first_batch = True

    while global_step < config.num_train_steps:
        # Set epoch for distributed training
        if use_ddp and hasattr(loader, "set_epoch"):
            loader.set_epoch(global_step // len(loader))

        for observation, actions in loader:
            if _first_batch:
                logging.info("[DEBUG] [6/6] First batch fetched in %.1fs. Training started!", time.time() - _t0)
                _first_batch = False

            # Check if we've reached the target number of steps
            if global_step >= config.num_train_steps:
                break

            if prompt_tokenizer is not None and (global_step % _PROMPT_LOG_INTERVAL == 0):
                try:
                    prompt_text = _decode_prompt_for_logging(observation, prompt_tokenizer)
                    if pbar is not None:
                        pbar.write(f"[prompt step={global_step}] {prompt_text}")
                    else:
                        logging.info("[prompt step=%s] %s", global_step, prompt_text)
                except Exception:
                    logging.exception("Failed to decode/log prompt at step=%s", global_step)

            is_val_step = (global_step % config.val_log_interval == 0) and _validation_is_enabled(config)
            if is_val_step:
                if val_loader is not None:
                    try:
                        val_metrics = validate(model, val_loader, device, config)
                        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in val_metrics.items() if not k.startswith("val/"))
                        if pbar is not None:
                            pbar.write(f"Step {global_step}: {metrics_str}")
                        else:
                            logging.info("Step %s: %s", global_step, metrics_str)
                        if config.wandb_enabled:
                            wandb.log(val_metrics, step=global_step)
                    except Exception:
                        logging.exception("Validation failed at step=%s", global_step)
                if use_ddp:
                    dist.barrier()

            # The unified data loader returns (observation, actions) tuple
            observation = jax.tree.map(lambda x: x.to(device), observation)  # noqa: PLW2901
            actions = actions.to(torch.float32)  # noqa: PLW2901
            actions = actions.to(device)  # noqa: PLW2901

            base_lr = lr_schedule(global_step)
            for pg in optim.param_groups:
                pg["lr"] = base_lr

            # Forward pass
            losses = model(observation, actions)
            # Ensure losses is a tensor and handle different return types
            if isinstance(losses, list | tuple):
                losses = torch.stack(losses)
            elif not isinstance(losses, torch.Tensor):
                losses = torch.tensor(losses, device=device, dtype=torch.float32)

            loss = losses.mean()

            # 训练诊断：每 100 步打印 pred vs gt
            if is_main and global_step % 100 == 0:
                with torch.no_grad():
                    raw_m = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
                    train_pred = raw_m.sample_actions(device, observation, num_steps=10)
                    n_s = min(2, train_pred.shape[0])
                    n_d = min(8, train_pred.shape[-1])
                    for i in range(n_s):
                        p = train_pred[i, 0, :n_d].cpu().tolist()
                        g = actions[i, 0, :n_d].cpu().tolist()
                        cos = torch.nn.functional.cosine_similarity(
                            train_pred[i].reshape(1, -1), actions[i].reshape(1, -1)
                        ).item()
                        logging.info("[TRAIN step=%d sample %d] pred: [%s]", global_step, i, ", ".join(f"{v:+.4f}" for v in p))
                        logging.info("[TRAIN step=%d sample %d]   gt: [%s]  cos=%.4f", global_step, i, ", ".join(f"{v:+.4f}" for v in g), cos)

            # Backward pass
            loss.backward()

            # Log memory usage after backward pass
            if global_step < 5 and is_main and torch.cuda.is_available():
                log_memory_usage(device, global_step, "after_backward")

            _clip_norm = config.optimizer.clip_gradient_norm
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=_clip_norm)

            # Optimizer step
            optim.step()
            optim.zero_grad(set_to_none=True)

            # Clear gradients more aggressively
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad = None

            # Aggregate loss across all GPUs for more accurate logging
            reduced_loss = loss.detach().clone()
            if use_ddp:
                torch.distributed.all_reduce(reduced_loss)
                reduced_loss = reduced_loss / torch.distributed.get_world_size()

            # Collect stats
            if is_main:
                infos.append(
                    {
                        "loss": reduced_loss.item(),
                        "learning_rate": base_lr,
                        "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    }
                )

            if is_main and (global_step % config.log_interval == 0):
                elapsed = time.time() - start_time

                # Average stats over log interval
                avg_loss = sum(info["loss"] for info in infos) / len(infos)
                avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)

                def _avg_key(key):
                    vals = [info[key] for info in infos if key in info and info[key] is not None]
                    return sum(vals) / len(vals) if vals else None

                avg_grad_norm = _avg_key("grad_norm")

                parts = [f"step={global_step}", f"loss={avg_loss:.4f}", f"lr={avg_lr:.2e}"]
                if avg_grad_norm is not None:
                    parts.append(f"grad={avg_grad_norm:.2f}")
                parts.append(f"time={elapsed:.1f}s")
                logging.info(" ".join(parts))

                # Log to wandb
                if config.wandb_enabled and len(infos) > 0:
                    log_payload = {
                        "loss": avg_loss,
                        "learning_rate": avg_lr,
                        "step": global_step,
                        "time_per_step": elapsed / config.log_interval,
                    }
                    if avg_grad_norm is not None:
                        log_payload["grad_norm"] = avg_grad_norm
                    wandb.log(log_payload, step=global_step)

                start_time = time.time()
                infos = []  # Reset stats collection

            global_step += 1
            # Save checkpoint using the new mechanism
            save_checkpoint(model, optim, global_step, config, is_main, data_config)

            # Update progress bar
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "lr": f"{optim.param_groups[0]['lr']:.2e}", "step": global_step}
                )

    # Close progress bar
    if pbar is not None:
        pbar.close()

    # Finish wandb run
    if is_main and config.wandb_enabled:
        wandb.finish()

    cleanup_ddp()


def main():
    init_logging()
    config = _config.cli()
    train_loop(config)


if __name__ == "__main__":
    main()
