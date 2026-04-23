import dataclasses
import functools
import logging
import os
import platform
import time
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.models.tokenizer as _tokenizer
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints_dist as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.jax_train_step as _jax_train_step
import openpi.training.optimizer as _optimizer
import openpi.training.wandb_log as _wandb_log
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders

_PROMPT_LOG_INTERVAL = 10


def _leaf_size_bytes(leaf: Any) -> int:
    if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
        return int(np.prod(leaf.shape)) * np.dtype(leaf.dtype).itemsize
    if hasattr(leaf, "size") and hasattr(leaf, "dtype"):
        return int(leaf.size) * np.dtype(leaf.dtype).itemsize
    return 0


def _tree_size_bytes(tree: Any) -> int:
    return sum(_leaf_size_bytes(leaf) for leaf in jax.tree.leaves(tree))


def _batch_leading_row_numpy(x: Any) -> Any:
    """First row of batch dim as host numpy; supports multi-host sharded jax.Array."""
    if x is None:
        return None
    if isinstance(x, jax.Array):
        if x.is_fully_addressable:
            return np.asarray(jax.device_get(x))[0]
        shards = x.addressable_shards
        if not shards:
            return np.asarray(jax.device_get(x))[0]
        local = np.asarray(shards[0].data)
        return local[0]
    return np.asarray(x)[0]


def _decode_prompt_for_logging(observation: Any, tokenizer: _tokenizer.PaligemmaTokenizer) -> str:
    tokens = _batch_leading_row_numpy(observation.tokenized_prompt)
    token_mask = getattr(observation, "tokenized_prompt_mask", None)
    token_mask = None if token_mask is None else _batch_leading_row_numpy(token_mask).astype(bool)

    token_ar_mask = getattr(observation, "token_ar_mask", None)
    token_ar_mask = None if token_ar_mask is None else _batch_leading_row_numpy(token_ar_mask)
    if token_ar_mask is not None:
        prompt_mask = token_ar_mask == 0
        if token_mask is not None:
            prompt_mask = prompt_mask & token_mask
        return tokenizer.decode(tokens, mask=prompt_mask)

    if token_mask is not None:
        return tokenizer.decode(tokens, mask=token_mask)
    return tokenizer.decode(tokens)


def _broadcast_str_from_primary(s: str, max_len: int = 512) -> str:
    """Broadcast a string from process 0 to all processes."""
    if jax.process_count() == 1:
        return s
    from jax.experimental.multihost_utils import broadcast_one_to_all

    # Encode to fixed-size numpy array
    encoded = np.zeros(max_len, dtype=np.uint8)
    s_bytes = s.encode("utf-8")[:max_len]
    encoded[: len(s_bytes)] = list(s_bytes)
    # Broadcast and decode (using int(x) to avoid jax.Array.tobytes() quirks)
    broadcasted = broadcast_one_to_all(encoded)
    return bytes(int(x) for x in broadcasted).rstrip(b"\x00").decode("utf-8")


def _validation_is_enabled(config: _config.TrainConfig) -> bool:
    if config.val_log_interval <= 0 or config.val_num_batches <= 0:
        return False
    return (config.val_repo_id is not None) or (config.val_episodes_index is not None)


def _override_factory_for_val(factory: _config.DataConfigFactory, config: _config.TrainConfig) -> _config.DataConfigFactory:
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


@at.typecheck
def eval_step(
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> at.Float[at.Array, ""]:
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    observation, actions = batch
    chunked_loss = model.compute_loss(rng, observation, actions, train=False)
    return jnp.mean(chunked_loss)


def init_logging():
    """Custom logging format for better readability."""
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
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(name=config.exp_name, config=dataclasses.asdict(config), project=config.project_name, group="openpi")
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)
    accum_steps = int(getattr(config, "gradient_accumulation_steps", 1))
    if accum_steps > 1:
        tx = optax.MultiSteps(tx, every_k_schedule=accum_steps, use_grad_mean=True, accumulator_dtype=jnp.bfloat16)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)
    trainable_params_shape = train_state_shape.params.filter(config.trainable_filter)
    logging.info(
        "[P%s] Host-side train state estimate: trainable_params=%.2f GiB full_params=%.2f GiB opt_state=%.2f GiB ema_params=%.2f GiB",
        jax.process_index(),
        _tree_size_bytes(trainable_params_shape) / 2**30,
        _tree_size_bytes(train_state_shape.params) / 2**30,
        _tree_size_bytes(train_state_shape.opt_state) / 2**30,
        0.0 if train_state_shape.ema_params is None else _tree_size_bytes(train_state_shape.ema_params) / 2**30,
    )

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


def main(config: _config.TrainConfig):
    init_logging()
    num_local_devices = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    jax.distributed.initialize(
        coordinator_address=f"{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
        process_id=int(os.environ["WORLD_RANK"]),
        num_processes=int(os.environ["WORLD_SIZE"]),
        local_device_ids=list(range(num_local_devices)),
    )
    logging.info(f"Running on: {platform.node()}")
    logging.info(f"JAX process index: {jax.process_index()}")
    logging.info(f"JAX process count: {jax.process_count()}")
    logging.info(f"JAX local device count: {jax.local_device_count()}")
    logging.info(f"JAX global device count: {jax.device_count()}")

    config = dataclasses.replace(config, exp_name=_broadcast_str_from_primary(config.exp_name))
    logging.info(f"[P{jax.process_index()}] checkpoint_dir: {config.checkpoint_dir}")

    accum_steps = int(getattr(config, "gradient_accumulation_steps", 1))
    if accum_steps < 1:
        raise ValueError(f"gradient_accumulation_steps must be >= 1, got {accum_steps}.")
    if config.batch_size % accum_steps != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by gradient_accumulation_steps={accum_steps}."
        )
    micro_batch_size = config.batch_size // accum_steps
    if micro_batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Micro batch size {micro_batch_size} must be divisible by the number of devices {jax.device_count()} "
            f"(batch_size={config.batch_size}, gradient_accumulation_steps={accum_steps})."
        )
    logging.info(
        "[P%s] Gradient accumulation: steps=%s micro_batch_size=%s effective_batch_size=%s",
        jax.process_index(),
        accum_steps,
        micro_batch_size,
        config.batch_size,
    )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    base_rng = jax.random.fold_in(jax.random.key(config.seed), jax.process_index())
    train_rng, _ = jax.random.split(base_rng)

    init_rng = jax.random.key(config.seed)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )

    if jax.process_index() == 0:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    train_data_config = dataclasses.replace(config, batch_size=micro_batch_size) if accum_steps > 1 else config
    data_loader = _data_loader.create_behavior_data_loader(
        train_data_config, sharding=data_sharding, shuffle=True, skip_norm_stats=False, seed_shift=int(jax.process_index())
    )
    data_iter = iter(data_loader)
    micro_batch = next(data_iter)

    prompt_tokenizer = None
    if jax.process_index() == 0:
        prompt_tokenizer = _tokenizer.PaligemmaTokenizer(config.model.max_token_len)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    trainable_params = train_state.params.filter(config.trainable_filter)
    logging.info(
        "[P%s] Train state sizes: trainable_params=%.2f GiB full_params=%.2f GiB opt_state=%.2f GiB ema_params=%.2f GiB",
        jax.process_index(),
        _tree_size_bytes(trainable_params) / 2**30,
        _tree_size_bytes(train_state.params) / 2**30,
        _tree_size_bytes(train_state.opt_state) / 2**30,
        0.0 if train_state.ema_params is None else _tree_size_bytes(train_state.ema_params) / 2**30,
    )
    logging.info(
        f"[P{jax.process_index()}] Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}"
    )

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)
        logging.info(f"[P{jax.process_index()}] Restored train state from checkpoint")

    ptrain_step = jax.jit(
        functools.partial(_jax_train_step.train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    val_loader = None
    peval_step = None
    if _validation_is_enabled(config):
        val_config = _make_val_config(config)
        val_loader = _data_loader.create_behavior_data_loader(
            val_config,
            sharding=data_sharding,
            shuffle=False,
            num_batches=val_config.val_num_batches,
            skip_norm_stats=False,
        )
        peval_step = jax.jit(
            eval_step,
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=replicated_sharding,
        )
        logging.info(
            "[P%s] Validation enabled: val_repo_id=%s val_episodes_index=%s val_batch_size=%s val_num_batches=%s val_log_interval=%s",
            jax.process_index(),
            val_config.val_repo_id,
            val_config.val_episodes_index,
            val_config.batch_size,
            val_config.val_num_batches,
            val_config.val_log_interval,
        )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    log_window_t0 = time.monotonic()

    N = len(data_loader._data_loader._data_loader)
    logging.info(f"{type(data_loader._data_loader), type(data_loader._data_loader._data_loader)}")
    logging.info(f"[P{jax.process_index()}] Steps per epoch: {N}")

    for step in pbar:
        if prompt_tokenizer is not None and (step % _PROMPT_LOG_INTERVAL == 0):
            try:
                observation, _actions = micro_batch
                prompt_text = _decode_prompt_for_logging(observation, prompt_tokenizer)
                pbar.write(f"[prompt step={step}] {prompt_text}")
            except Exception:
                logging.exception("Failed to decode/log prompt at step=%s", step)

        if val_loader is not None and peval_step is not None and (step % config.val_log_interval == 0):
            try:
                val_losses = []
                val_rng = jax.random.fold_in(train_rng, 1_000_000 + step)
                for val_batch in val_loader:
                    with sharding.set_mesh(mesh):
                        val_loss = peval_step(val_rng, train_state, val_batch)
                    val_losses.append(val_loss)
                val_stack = jax.device_get(jnp.stack(val_losses))
                val_loss_mean = float(jnp.mean(val_stack))
                val_loss_std = float(jnp.std(val_stack))
                pbar.write(f"Step {step}: val_loss={val_loss_mean:.4f} (std={val_loss_std:.4f})")
                if jax.process_index() == 0:
                    wandb.log(
                        {
                            "val_loss": val_loss_mean,
                            "val_loss_std": val_loss_std,
                            "val_loss_min": float(jnp.min(val_stack)),
                            "val_loss_max": float(jnp.max(val_stack)),
                            "val_num_batches": float(len(val_losses)),
                        },
                        step=step,
                    )
            except Exception:
                logging.exception("Validation failed at step=%s", step)

        if accum_steps > 1:
            micro_infos = []
            for _micro_i in range(accum_steps):
                with sharding.set_mesh(mesh):
                    train_state, micro_info = ptrain_step(train_rng, train_state, micro_batch)
                micro_infos.append(micro_info)
                micro_batch = next(data_iter)
            stacked = common_utils.stack_forest(micro_infos)
            info = jax.tree.map(lambda x: jnp.mean(x, axis=0), stacked)
            # Override update-related metrics with the last micro-step, since
            # optax.MultiSteps returns zero updates on intermediate steps.
            info["update_norm"] = micro_infos[-1]["update_norm"]
            info["update_param_ratio"] = micro_infos[-1]["update_param_ratio"]
            info["grad_param_ratio"] = micro_infos[-1]["grad_param_ratio"]
        else:
            with sharding.set_mesh(mesh):
                train_state, info = ptrain_step(train_rng, train_state, micro_batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            n_steps = len(infos)
            dt = max(time.monotonic() - log_window_t0, 1e-9)
            log_window_t0 = time.monotonic()
            wandb_payload = _wandb_log.jax_stacked_infos_to_wandb(
                stacked_infos,
                extra={
                    "wall_time_per_step_ms": 1000.0 * dt / max(n_steps, 1),
                    "throughput_samples_per_sec": (n_steps * config.batch_size) / dt,
                    "throughput_steps_per_sec": n_steps / dt,
                    "log_window_steps": float(n_steps),
                },
            )
            if jax.process_index() == 0:
                reduced_info = {str(k): wandb_payload[str(k)] for k in stacked_infos}
                info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
                pbar.write(f"Step {step}: {info_str}")
                wandb.log(wandb_payload, step=step)
            infos = []
        if accum_steps <= 1:
            micro_batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    checkpoint_manager.wait_until_finished()
    checkpoint_manager.close()


if __name__ == "__main__":
    main(_config.cli())
