import dataclasses
import functools
import logging
import platform
import time
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import optax
import tqdm_loggable.auto as tqdm
import wandb
import numpy as np

import openpi.models.model as _model
import openpi.models.tokenizer as _tokenizer
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders

_PROMPT_LOG_INTERVAL = 10


def _count_array_elements_and_bytes(x: Any) -> tuple[int, int]:
    if isinstance(x, jax.ShapeDtypeStruct):
        shape = x.shape
        dtype = np.dtype(x.dtype)
    elif hasattr(x, "shape") and hasattr(x, "dtype"):
        shape = x.shape
        dtype = np.dtype(x.dtype)
    else:
        return 0, 0
    numel = int(np.prod(shape)) if len(shape) > 0 else 1
    return numel, numel * int(dtype.itemsize)


def _count_params_in_state(state: nnx.State) -> tuple[int, int, dict[str, int]]:
    """Returns (num_params, num_bytes, path->num_params)."""
    flat = traverse_util.flatten_dict(state.to_pure_dict())
    num_params = 0
    num_bytes = 0
    per_path: dict[str, int] = {}
    for key, value in flat.items():
        path = "/".join(str(p) for p in key)
        n, b = _count_array_elements_and_bytes(value)
        num_params += n
        num_bytes += b
        if n:
            per_path[path] = n
    return num_params, num_bytes, per_path


def _log_trainable_params_summary(config: _config.TrainConfig, state: training_utils.TrainState) -> None:
    total_n, total_b, _ = _count_params_in_state(state.params)
    trainable_state = state.params.filter(config.trainable_filter)
    train_n, train_b, train_paths = _count_params_in_state(trainable_state)

    lora_n = sum(n for p, n in train_paths.items() if "lora" in p.lower())
    lora_frac = (lora_n / train_n) if train_n else 0.0
    frac = (train_n / total_n) if total_n else 0.0

    logging.info(
        "Params: trainable=%s (%.3f%%) total=%s | trainable_bytes=%.2f GiB total_bytes=%.2f GiB | trainable_lora_coverage=%.1f%%",
        f"{train_n:,}",
        100.0 * frac,
        f"{total_n:,}",
        train_b / (1024**3),
        total_b / (1024**3),
        100.0 * lora_frac,
    )

    largest = sorted(train_paths.items(), key=lambda kv: kv[1], reverse=True)[:15]
    if largest:
        logging.info("Largest trainable params (top 15):")
        for path, n in largest:
            logging.info("  %s: %s", path, f'{n:,}')

    if "lora" in getattr(config.model, "paligemma_variant", "").lower() or "lora" in getattr(
        config.model, "action_expert_variant", ""
    ).lower():
        if frac > 0.10:
            logging.warning(
                "Trainable parameter fraction looks high for LoRA training (%.2f%%). Check `freeze_filter`.",
                100.0 * frac,
            )

    try:
        wandb.log(
            {
                "params/total": int(total_n),
                "params/trainable": int(train_n),
                "params/trainable_fraction": float(frac),
                "params/trainable_lora_coverage": float(lora_frac),
                "params/total_gib": float(total_b / (1024**3)),
                "params/trainable_gib": float(train_b / (1024**3)),
            },
            step=int(jax.device_get(state.step)),
        )
    except Exception:
        logging.exception("Failed to log params summary to wandb")


def _make_frozen_spotcheck(
    config: _config.TrainConfig, state: training_utils.TrainState, *, max_tensors: int = 5, max_numel: int = 2048
) -> list[tuple[tuple[Any, ...], str, np.ndarray]]:
    """Captures a few small frozen tensors so we can verify frozen weights stay unchanged."""
    if config.freeze_filter is nnx.Nothing or getattr(getattr(config.freeze_filter, "__class__", None), "__name__", "") == "Nothing":
        return []

    frozen_state = state.params.filter(config.freeze_filter)
    flat = traverse_util.flatten_dict(frozen_state.to_pure_dict())
    chosen: list[tuple[tuple[Any, ...], str, np.ndarray]] = []
    for key, value in flat.items():
        n, _b = _count_array_elements_and_bytes(value)
        if n <= 0 or n > max_numel:
            continue
        path = "/".join(str(p) for p in key)
        baseline = np.array(jax.device_get(value))
        chosen.append((key, path, baseline))
        if len(chosen) >= max_tensors:
            break

    if chosen:
        logging.info("Frozen spotcheck tensors (n=%d, max_numel=%d):", len(chosen), max_numel)
        for _key, path, base in chosen:
            logging.info("  %s: shape=%s dtype=%s", path, base.shape, base.dtype)
    return chosen


def _decode_prompt_for_logging(observation: Any, tokenizer: _tokenizer.PaligemmaTokenizer) -> str:
    tokens = jax.device_get(observation.tokenized_prompt)[0]
    token_mask = getattr(observation, "tokenized_prompt_mask", None)
    token_mask = None if token_mask is None else jax.device_get(token_mask)[0].astype(bool)

    token_ar_mask = getattr(observation, "token_ar_mask", None)
    token_ar_mask = None if token_ar_mask is None else jax.device_get(token_ar_mask)[0]
    if token_ar_mask is not None:
        prompt_mask = token_ar_mask == 0
        if token_mask is not None:
            prompt_mask = prompt_mask & token_mask
        return tokenizer.decode(tokens, mask=prompt_mask)

    if token_mask is not None:
        return tokenizer.decode(tokens, mask=token_mask)
    return tokenizer.decode(tokens)

def _validation_is_enabled(config: _config.TrainConfig) -> bool:
    if config.val_log_interval <= 0 or config.val_num_batches <= 0:
        return False
    # Require an explicit validation split/dataset override to avoid silently "validating" on the train set.
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
        # Backward-compat: keep `val_loss` but make it total loss (flow + stop supervision if enabled).
        "val_loss": total_loss,
        "val/total_loss": total_loss,
        # `val/flow_loss` is flow-matching loss only.
        "val/flow_loss": flow_loss,
        "val/action_mse": action_mse,
        "val/action_mae": action_mae,
        "val/action_cosine_sim": jnp.mean(cos_sim),
        "val/first_action_mse": first_action_mse,
    }
    if metrics:
        # Mirror stop metrics under `val/` so dashboards are unambiguous.
        for k, v in metrics.items():
            if k.startswith("stop/"):
                out[f"val/{k}"] = v
    return out


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
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
            group="openpi",
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    # Allow partial restores when the current model has new parameters that are not present in the checkpoint
    # (e.g. adding a new head). We validate that:
    # - every loaded key exists in the expected tree and matches shape/dtype
    # - missing keys are filled with ShapeDtypeStruct placeholders (and removed afterwards)
    expected_flat = traverse_util.flatten_dict(params_shape)
    loaded_flat = traverse_util.flatten_dict(loaded_params)

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
        model: _model.BaseModel,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
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
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                state.ema_params,
                new_params,
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
    trainable_param_norm = optax.global_norm(params)
    update_norm = optax.global_norm(updates)
    grad_norm = optax.global_norm(grads)
    info = {
        "loss": loss,
        "grad_norm": grad_norm,
        "update_norm": update_norm,
        "trainable_param_norm": trainable_param_norm,
        "grad_to_param": grad_norm / (trainable_param_norm + 1e-8),
        "update_to_param": update_norm / (trainable_param_norm + 1e-8),
        "param_norm": optax.global_norm(kernel_params),
    }
    if isinstance(metrics, dict):
        info.update(metrics)
    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")
    logging.info(f"JAX process index: {jax.process_index()}")
    logging.info(f"JAX process count: {jax.process_count()}")
    logging.info(f"JAX local device count: {jax.local_device_count()}")
    logging.info(f"JAX global device count: {jax.device_count()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_behavior_data_loader(
        config, sharding=data_sharding, shuffle=True, skip_norm_stats=False
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(
        "Data config: fine_grained_level=%s (0=global task, 1=subtask, 2=skill)",
        getattr(data_loader.data_config(), "fine_grained_level", "?"),
    )
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    prompt_tokenizer = None
    if jax.process_index() == 0:
        prompt_tokenizer = _tokenizer.PaligemmaTokenizer(config.model.max_token_len)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    try:
        _log_trainable_params_summary(config, train_state)
    except Exception:
        logging.exception("Failed to compute/log trainable params summary")

    frozen_spotcheck = []
    try:
        frozen_spotcheck = _make_frozen_spotcheck(config, train_state)
    except Exception:
        logging.exception("Failed to initialize frozen spotcheck")

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    lr_schedule = config.lr_schedule.create()

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
    data_time_accum_s = 0.0
    last_log_t = time.perf_counter()
    last_log_step = start_step
    for step in pbar:
        if prompt_tokenizer is not None and (step % _PROMPT_LOG_INTERVAL == 0):
            try:
                observation, _actions = batch
                prompt_text = _decode_prompt_for_logging(observation, prompt_tokenizer)
                pbar.write(f"[prompt step={step}] {prompt_text}")
            except Exception:
                logging.exception("Failed to decode/log prompt at step=%s", step)

        if val_loader is not None and peval_step is not None and (step % config.val_log_interval == 0):
            try:
                val_metrics_list = []
                val_rng = jax.random.fold_in(train_rng, 1_000_000 + step)
                for val_batch in val_loader:
                    with sharding.set_mesh(mesh):
                        metrics = peval_step(val_rng, train_state, val_batch)
                    val_metrics_list.append(metrics)
                stacked = common_utils.stack_forest(val_metrics_list)
                val_metrics = {k: float(v) for k, v in jax.device_get(jax.tree.map(jnp.mean, stacked)).items()}
                # Print a small, stable subset to logs; full dict still goes to wandb.
                keys = [
                    "val_loss",
                    "val/flow_loss",
                    "val/stop/loss_weighted",
                    "val/stop/pos_frac",
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
            jax.block_until_ready(train_state.step)

            now = time.perf_counter()
            interval_steps = (step - last_log_step) + 1
            interval_s = now - last_log_t
            time_per_step_s = interval_s / max(interval_steps, 1)
            steps_per_sec = interval_steps / max(interval_s, 1e-9)

            data_time_per_step_s = data_time_accum_s / max(interval_steps, 1)
            data_fraction = data_time_per_step_s / max(time_per_step_s, 1e-9)
            lr = float(jax.device_get(lr_schedule(step)))

            reduced_info = dict(reduced_info)
            reduced_info.update(
                {
                    "lr": lr,
                    "time/step_s": float(time_per_step_s),
                    "time/steps_per_sec": float(steps_per_sec),
                    "time/examples_per_sec": float(steps_per_sec * config.batch_size),
                    "time/data_time_s": float(data_time_per_step_s),
                    "time/data_fraction": float(data_fraction),
                }
            )

            if frozen_spotcheck:
                try:
                    frozen_flat = traverse_util.flatten_dict(train_state.params.filter(config.freeze_filter).to_pure_dict())
                    max_abs = 0.0
                    for key, _path, base in frozen_spotcheck:
                        cur = np.array(jax.device_get(frozen_flat[key]))
                        max_abs = max(max_abs, float(np.max(np.abs(cur - base))))
                    reduced_info["frozen/spotcheck_max_abs_diff"] = max_abs
                except Exception:
                    logging.exception("Frozen spotcheck failed at step=%s", step)

            try:
                ms = jax.devices()[0].memory_stats()
                if isinstance(ms, dict):
                    if "bytes_in_use" in ms:
                        reduced_info["memory/bytes_in_use_gib"] = float(ms["bytes_in_use"] / (1024**3))
                    if "peak_bytes_in_use" in ms:
                        reduced_info["memory/peak_bytes_in_use_gib"] = float(ms["peak_bytes_in_use"] / (1024**3))
            except Exception:
                pass

            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
            data_time_accum_s = 0.0
            last_log_t = now
            last_log_step = step + 1
        data_t0 = time.perf_counter()
        batch = next(data_iter)
        data_time_accum_s += time.perf_counter() - data_t0

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
