"""Shared JAX training step used by ``scripts/train.py`` and ``scripts/train_dist.py``."""

from __future__ import annotations

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.config as _config
import openpi.training.utils as training_utils


def _compute_batch_metrics(
    config: _config.TrainConfig,
    state: training_utils.TrainState,
    observation: _model.Observation,
    actions: _model.Actions,
    updates: nnx.State,
    grads: nnx.State,
    param_norm: at.Array,
) -> dict[str, at.Array]:
    info: dict[str, at.Array] = {}

    update_norm = optax.global_norm(updates)
    grad_norm = optax.global_norm(grads)
    info["update_norm"] = update_norm
    info["grad_param_ratio"] = grad_norm / jnp.clip(param_norm, 1e-8)
    info["update_param_ratio"] = update_norm / jnp.clip(param_norm, 1e-8)
    info["lr"] = config.lr_schedule.create()(state.step)

    state_norm = jnp.linalg.norm(observation.state, axis=-1)
    action_norm = jnp.linalg.norm(actions, axis=-1)
    info["state_norm_mean"] = jnp.mean(state_norm)
    info["action_norm_mean"] = jnp.mean(action_norm)

    if observation.tokenized_prompt_mask is not None:
        token_lengths = jnp.sum(observation.tokenized_prompt_mask.astype(jnp.int32), axis=-1)
        max_tokens = observation.tokenized_prompt_mask.shape[-1]
        info["prompt_token_len_mean"] = jnp.mean(token_lengths.astype(jnp.float32))
        info["prompt_token_len_max"] = jnp.max(token_lengths.astype(jnp.float32))
        info["prompt_at_capacity_ratio"] = jnp.mean((token_lengths >= max_tokens).astype(jnp.float32))

    if observation.flow_tokenized_prompt_mask is not None:
        flow_token_lengths = jnp.sum(observation.flow_tokenized_prompt_mask.astype(jnp.int32), axis=-1)
        max_flow_tokens = observation.flow_tokenized_prompt_mask.shape[-1]
        info["flow_prompt_token_len_mean"] = jnp.mean(flow_token_lengths.astype(jnp.float32))
        info["flow_prompt_token_len_max"] = jnp.max(flow_token_lengths.astype(jnp.float32))
        info["flow_prompt_at_capacity_ratio"] = jnp.mean((flow_token_lengths >= max_flow_tokens).astype(jnp.float32))

    if observation.token_loss_mask is not None:
        supervised_token_count = jnp.sum(observation.token_loss_mask[..., 1:].astype(jnp.float32), axis=-1)
        info["supervised_token_count_mean"] = jnp.mean(supervised_token_count)
        max_supervised = observation.token_loss_mask[..., 1:].shape[-1]
        info["supervised_token_at_capacity_ratio"] = jnp.mean(
            (supervised_token_count >= max_supervised).astype(jnp.float32)
        )

    if observation.is_vqa is not None:
        is_vqa = observation.is_vqa.astype(jnp.float32)
        info["vqa_sample_ratio"] = jnp.mean(is_vqa)
        info["b1k_sample_ratio"] = 1.0 - info["vqa_sample_ratio"]

    if observation.flow_loss_mask is not None:
        info["flow_active_ratio"] = jnp.mean(observation.flow_loss_mask.astype(jnp.float32))

    image_valid = [jnp.mean(mask.astype(jnp.float32)) for mask in observation.image_masks.values()]
    if image_valid:
        info["image_valid_ratio_mean"] = jnp.mean(jnp.stack(image_valid))

    return info


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
            chunked_loss, aux_metrics = model.compute_loss_and_metrics(rng, observation, actions, train=True)
        else:
            chunked_loss = model.compute_loss(rng, observation, actions, train=True)
            aux_metrics = {}
        return jnp.mean(chunked_loss), aux_metrics

    train_rng = jax.random.fold_in(rng, state.step)
    micro_step = getattr(state.opt_state, "mini_step", None)
    if micro_step is not None:
        train_rng = jax.random.fold_in(train_rng, jnp.asarray(micro_step, dtype=jnp.uint32))
    observation, actions = batch

    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, aux_metrics), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_step = getattr(new_opt_state, "gradient_step", None)
    new_step = state.step + 1 if new_step is None else jnp.asarray(new_step, dtype=state.step.dtype)
    new_state = dataclasses.replace(state, step=new_step, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        # Only update EMA when parameters actually update (e.g., with optax.MultiSteps).
        should_update_ema = new_step != state.step
        new_ema = jax.lax.cond(
            should_update_ema,
            lambda _: jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                state.ema_params,
                new_params,
            ),
            lambda _: state.ema_params,
            operand=None,
        )
        new_state = dataclasses.replace(new_state, ema_params=new_ema)

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
    info.update(aux_metrics)
    info.update(_compute_batch_metrics(config, state, observation, actions, updates, grads, info["param_norm"]))
    return new_state, info


@at.typecheck
def compute_grads(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
    micro_step: int | at.ArrayLike = 0,
) -> tuple[nnx.State, dict[str, at.Array]]:
    """Compute gradients for a single micro-batch (no optimizer update)."""
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
            chunked_loss, aux_metrics = model.compute_loss_and_metrics(rng, observation, actions, train=True)
        else:
            chunked_loss = model.compute_loss(rng, observation, actions, train=True)
            aux_metrics = {}
        return jnp.mean(chunked_loss), aux_metrics

    observation, actions = batch
    micro_step_u32 = jnp.asarray(micro_step, dtype=jnp.uint32)
    train_rng = jax.random.fold_in(jax.random.fold_in(rng, state.step), micro_step_u32)
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, aux_metrics), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions
    )
    info = {"loss": loss}
    info.update(aux_metrics)
    return grads, info


@at.typecheck
def apply_grads(
    config: _config.TrainConfig,
    state: training_utils.TrainState,
    grads: nnx.State,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """Apply a (possibly accumulated) gradient update and advance `state.step` by 1."""
    model = nnx.merge(state.model_def, state.params)
    model.train()

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

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

    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
        "update_norm": optax.global_norm(updates),
    }
    return new_state, info


@at.typecheck
def train_step_accumulated(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    micro_batches: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """Gradient accumulation: compute grads per micro-batch, average, then update once.

    Unlike scan-inside-value_and_grad, this keeps only one micro-batch's activations
    in memory at a time, avoiding OOM on large accumulation counts.
    """
    observations, actions_stacked = micro_batches
    n_micro = jax.tree.leaves(observations)[0].shape[0]

    def _micro_grad(rng_key, obs, act):
        model = nnx.merge(state.model_def, state.params)
        model.train()

        def loss_fn(mdl, rng_k, o, a):
            if hasattr(mdl, "compute_loss_and_metrics"):
                cl, aux = mdl.compute_loss_and_metrics(rng_k, o, a, train=True)
            else:
                cl = mdl.compute_loss(rng_k, o, a, train=True)
                aux = {}
            return jnp.mean(cl), aux

        diff = nnx.DiffState(0, config.trainable_filter)
        (loss, aux), grads = nnx.value_and_grad(loss_fn, argnums=diff, has_aux=True)(
            model, rng_key, obs, act
        )
        return grads, loss, aux

    def _slice_micro(tree, idx: jax.Array):
        return jax.tree.map(lambda x: jax.lax.dynamic_index_in_dim(x, idx, axis=0, keepdims=False), tree)

    rng_key = jax.random.fold_in(rng, state.step)
    micro_rngs = jax.random.split(rng_key, n_micro)

    obs_0 = _slice_micro(observations, jnp.array(0, dtype=jnp.int32))
    act_0 = _slice_micro(actions_stacked, jnp.array(0, dtype=jnp.int32))
    grads_acc, total_loss, first_aux = _micro_grad(micro_rngs[0], obs_0, act_0)

    def _body(i, carry):
        grads_sum, loss_sum = carry
        obs_i = _slice_micro(observations, i)
        act_i = _slice_micro(actions_stacked, i)
        grads_i, loss_i, _aux_i = _micro_grad(micro_rngs[i], obs_i, act_i)
        grads_sum = jax.tree.map(jnp.add, grads_sum, grads_i)
        loss_sum = loss_sum + loss_i
        return grads_sum, loss_sum

    grads_acc, total_loss = jax.lax.fori_loop(
        lower=jnp.array(1, dtype=jnp.int32),
        upper=jnp.asarray(n_micro, dtype=jnp.int32),
        body_fun=_body,
        init_val=(grads_acc, total_loss),
    )

    grads_acc = jax.tree.map(lambda g: g / n_micro, grads_acc)
    total_loss = total_loss / n_micro

    model = nnx.merge(state.model_def, state.params)
    model.train()

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads_acc, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

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

    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": total_loss,
        "grad_norm": optax.global_norm(grads_acc),
        "param_norm": optax.global_norm(kernel_params),
    }
    for k, v in first_aux.items():
        info[k] = v

    first_obs = jax.tree.map(lambda x: x[0], observations)
    first_act = jax.tree.map(lambda x: x[0], actions_stacked)
    info.update(_compute_batch_metrics(config, state, first_obs, first_act, updates, grads_acc, info["param_norm"]))
    return new_state, info
