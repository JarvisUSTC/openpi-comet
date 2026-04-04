import dataclasses
import functools
import logging
import os
import platform
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
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders

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
        if config.val_stop_balanced_sampling:
            base = dataclasses.replace(
                base,
                stop_balanced_sampling=True,
                stop_balanced_pos_prob=float(config.val_stop_balanced_pos_prob),
                stop_balanced_cycle=bool(config.val_stop_balanced_cycle),
            )
        factory = dataclasses.replace(factory, base_config=base)
    return factory


def _make_val_config(config: _config.TrainConfig) -> _config.TrainConfig:
    val_batch_size = config.batch_size if config.val_batch_size is None else config.val_batch_size
    if isinstance(config.data, list):
        val_data = [_override_factory_for_val(f, config) for f in config.data]
    else:
        val_data = _override_factory_for_val(config.data, config)
    return dataclasses.replace(config, batch_size=val_batch_size, data=val_data)


def _decode_prompt_for_logging(observation: Any, tokenizer: _tokenizer.PaligemmaTokenizer) -> str:
    def _first_example_to_host(x: Any) -> np.ndarray | None:
        if x is None:
            return None
        if isinstance(x, jax.Array):
            # In multi-host runs, x may be sharded across non-addressable devices, and
            # jax.device_get(x) will fail. Use any local shard instead.
            try:
                if getattr(x, "is_fully_addressable", False):
                    return np.asarray(jax.device_get(x)[0])
            except Exception:
                pass
            shards = getattr(x, "addressable_shards", None)
            if shards:
                return np.asarray(jax.device_get(shards[0].data)[0])
            # Best-effort fallback.
            return None
        try:
            return np.asarray(x)[0]
        except Exception:
            return None

    tokens = _first_example_to_host(observation.tokenized_prompt)
    if tokens is None:
        return "<prompt_unavailable>"

    token_mask = _first_example_to_host(getattr(observation, "tokenized_prompt_mask", None))
    token_mask = None if token_mask is None else token_mask.astype(bool)

    token_ar_mask = _first_example_to_host(getattr(observation, "token_ar_mask", None))
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
    # Allow partial restores when the current model has new parameters that are not present in the checkpoint.
    expected_flat = traverse_util.flatten_dict(params_shape)
    loaded_flat = traverse_util.flatten_dict(loaded_params)

    # If the checkpoint predates our stop modules (or stop head structure changed),
    # drop any loaded stop parameters so they get randomly initialized.
    expected_top = {k[0] for k in expected_flat.keys() if k}
    loaded_top = {k[0] for k in loaded_flat.keys() if k}
    missing_top = expected_top - loaded_top
    if any(str(t).startswith("stop") for t in missing_top):
        before = len(loaded_flat)
        loaded_flat = {k: v for k, v in loaded_flat.items() if not (k and str(k[0]).startswith("stop"))}
        after = len(loaded_flat)
        logging.info("Weight loader: dropped %d stop params from checkpoint (reset stop head).", before - after)

    extra_keys = set(loaded_flat.keys()) - set(expected_flat.keys())
    if extra_keys:
        extra_preview = sorted(extra_keys)[:10]
        raise ValueError(
            f"Loaded checkpoint has unexpected parameter keys (showing up to 10): {extra_preview}"
        )

    missing_keys = set(expected_flat.keys()) - set(loaded_flat.keys())
    if missing_keys:
        logging.info(
            "Weight loader: %d missing keys (will init randomly): %s",
            len(missing_keys),
            ", ".join("/".join(map(str, k)) for k in sorted(missing_keys)[:5]),
        )
        for k in missing_keys:
            loaded_flat[k] = expected_flat[k]
        loaded_params = traverse_util.unflatten_dict(loaded_flat)

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


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        if hasattr(model, "compute_loss_and_metrics"):
            chunked_loss, metrics = model.compute_loss_and_metrics(rng, observation, actions, train=True)  # type: ignore[attr-defined]
        else:
            chunked_loss = model.compute_loss(rng, observation, actions, train=True)
            metrics = {}
        return jnp.mean(chunked_loss), metrics

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, metrics), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    if isinstance(metrics, dict):
        info.update(metrics)
    return new_state, info


def eval_step(
    num_denoise_steps: int,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> dict[str, at.Float[at.Array, ""]]:
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    observation, actions = batch

    loss_rng, sample_rng = jax.random.split(rng)

    metrics: dict[str, at.Float[at.Array, ""]] = {}
    if hasattr(model, "compute_loss_and_metrics"):
        chunked_loss, metrics = model.compute_loss_and_metrics(loss_rng, observation, actions, train=False)  # type: ignore[attr-defined]
    else:
        chunked_loss = model.compute_loss(loss_rng, observation, actions, train=False)

    total_loss = jnp.mean(chunked_loss)
    flow_loss = metrics.get("flow/loss", total_loss)

    pred_actions = model.sample_actions(sample_rng, observation, num_steps=num_denoise_steps)

    action_error = pred_actions - actions
    action_mse = jnp.mean(jnp.square(action_error))
    action_mae = jnp.mean(jnp.abs(action_error))

    pred_flat = pred_actions.reshape(pred_actions.shape[0], -1)
    gt_flat = actions.reshape(actions.shape[0], -1)
    cos_sim = jnp.sum(pred_flat * gt_flat, axis=-1) / (
        jnp.linalg.norm(pred_flat, axis=-1) * jnp.linalg.norm(gt_flat, axis=-1) + 1e-8
    )

    first_action_mse = jnp.mean(jnp.square(pred_actions[:, 0] - actions[:, 0]))

    out: dict[str, at.Float[at.Array, ""]] = {
        "val_loss": total_loss,
        "val/total_loss": total_loss,
        "val/flow_loss": flow_loss,
        "val/action_mse": action_mse,
        "val/action_mae": action_mae,
        "val/action_cosine_sim": jnp.mean(cos_sim),
        "val/first_action_mse": first_action_mse,
    }
    if metrics:
        for k, v in metrics.items():
            if k.startswith("stop/"):
                out[f"val/{k}"] = v
    return out


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

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
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

    data_loader = _data_loader.create_behavior_data_loader(
        config, sharding=data_sharding, shuffle=True, skip_norm_stats=False, seed_shift=int(jax.process_index())
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)

    prompt_tokenizer = None
    if jax.process_index() == 0:
        prompt_tokenizer = _tokenizer.PaligemmaTokenizer(config.model.max_token_len)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(
        f"[P{jax.process_index()}] Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}"
    )

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)
        logging.info(f"[P{jax.process_index()}] Restored train state from checkpoint")

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
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
            functools.partial(eval_step, config.val_denoise_steps),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=replicated_sharding,
        )
        logging.info(
            "Validation enabled: val_repo_id=%s val_episodes_index=%s val_batch_size=%s val_num_batches=%s val_log_interval=%s",
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

    N = len(data_loader._data_loader._data_loader)
    logging.info(f"{type(data_loader._data_loader), type(data_loader._data_loader._data_loader)}")
    logging.info(f"[P{jax.process_index()}] Steps per epoch: {N}")

    for step in pbar:
        if jax.process_index() == 0 and prompt_tokenizer is not None and (step % _PROMPT_LOG_INTERVAL == 0):
            try:
                observation, _actions = batch
                prompt_text = _decode_prompt_for_logging(observation, prompt_tokenizer)
                pbar.write(f"[prompt step={step}] {prompt_text}")
            except Exception:
                logging.exception("Failed to decode/log prompt at step=%s", step)

        if val_loader is not None and peval_step is not None and (step % config.val_log_interval == 0):
            try:
                # Reset stateful streaming datasets so validation is comparable across steps.
                if hasattr(val_loader, "_data_loader") and hasattr(val_loader._data_loader, "reset_dataset_state"):
                    val_loader._data_loader.reset_dataset_state()
                val_metrics_list = []
                val_rng = jax.random.fold_in(train_rng, 1_000_000 + step)
                for val_batch in val_loader:
                    with sharding.set_mesh(mesh):
                        metrics = peval_step(val_rng, train_state, val_batch)
                    val_metrics_list.append(metrics)
                stacked = common_utils.stack_forest(val_metrics_list)
                val_metrics = {k: float(v) for k, v in jax.device_get(jax.tree.map(jnp.mean, stacked)).items()}
                if jax.process_index() == 0:
                    # Print a small, stable subset to logs; full dict still goes to wandb.
                    keys = [
                        "val_loss",
                        "val/flow_loss",
                        "val/stop/loss_weighted",
                        "val/stop/pos_frac",
                        "val/stop/prob_mean_pos",
                        "val/stop/prob_mean_neg",
                        "val/stop/prob_mean_hard_pos",
                        "val/stop/prob_mean_hard_neg",
                        "val/stop/pos_weight",
                        "val/action_mse",
                    ]
                    metrics_str = ", ".join(f"{k}={val_metrics[k]:.4f}" for k in keys if k in val_metrics)
                    pbar.write(f"Step {step}: {metrics_str}")
                    wandb.log(val_metrics, step=step)
            except Exception:
                logging.exception("Validation failed at step=%s", step)

        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            if jax.process_index() == 0:
                info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
                pbar.write(f"Step {step}: {info_str}")
                wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    checkpoint_manager.wait_until_finished()
    checkpoint_manager.close()


if __name__ == "__main__":
    main(_config.cli())
