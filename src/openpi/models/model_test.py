from flax import nnx
import jax
import jax.numpy as jnp
import ml_collections
import pytest

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import pi0_fast
from openpi.shared import download
from openpi.shared import nnx_utils


def test_pi0_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=8,
        action_horizon=4,
        max_token_len=16,
    )
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = model.compute_loss(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = model.sample_actions(key, obs, num_steps=2)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_model_without_pi05():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(
        pi05=False,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=8,
        action_horizon=4,
        max_token_len=16,
    )
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = model.compute_loss(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)


def test_pi0_lora_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=8,
        action_horizon=4,
        max_token_len=16,
    )
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = model.compute_loss(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = model.sample_actions(key, obs, num_steps=2)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_fast_model():
    def _dummy_fast_config(_):
        return ml_collections.ConfigDict(
            {
                "variant": "dummy",
                "width": 64,
                "depth": 2,
                "mlp_dim": 128,
                "num_heads": 4,
                "num_kv_heads": 1,
                "head_dim": 16,
                "norm_eps": 1e-6,
                "vocab_size": 257_152,
                "scan": True,
                "remat_policy": "nothing_saveable",
            }
        )

    pi0_fast._gemma.get_config = _dummy_fast_config
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig(
        paligemma_variant="gemma_2b",
        action_dim=8,
        action_horizon=4,
        max_token_len=16,
    )
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = model.compute_loss(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = model.sample_actions(key, obs, max_decoding_steps=16)
    assert actions.shape == (batch_size, 16)


def test_pi0_fast_lora_model():
    def _dummy_fast_config(_):
        return ml_collections.ConfigDict(
            {
                "variant": "dummy",
                "width": 64,
                "depth": 2,
                "mlp_dim": 128,
                "num_heads": 4,
                "num_kv_heads": 1,
                "head_dim": 16,
                "norm_eps": 1e-6,
                "vocab_size": 257_152,
                "scan": True,
                "remat_policy": "nothing_saveable",
            }
        )

    pi0_fast._gemma.get_config = _dummy_fast_config
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig(
        paligemma_variant="gemma_2b",
        action_dim=8,
        action_horizon=4,
        max_token_len=16,
    )
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = model.compute_loss(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = model.sample_actions(key, obs, max_decoding_steps=16)
    assert actions.shape == (batch_size, 16)


def test_pi0_ki_dummy_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(
        pi05=True,
        knowledge_insulation=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=8,
        action_horizon=4,
        max_token_len=16,
    )
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)
    obs = obs.replace(
        token_loss_mask=jnp.ones((batch_size, config.max_token_len), dtype=jnp.bool_),
        token_ar_mask=jnp.concatenate(
            [
                jnp.zeros((batch_size, config.max_token_len // 2), dtype=jnp.int32),
                jnp.ones((batch_size, config.max_token_len - config.max_token_len // 2), dtype=jnp.int32),
            ],
            axis=1,
        ),
        flow_loss_mask=jnp.asarray([True, False], dtype=jnp.bool_),
        is_vqa=jnp.asarray([False, True], dtype=jnp.bool_),
    )

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)


@pytest.mark.manual
def test_model_restore():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    model = config.load(
        _model.restore_params(download.maybe_download("gs://openpi-assets/checkpoints/pi0_base/params"))
    )

    loss = model.compute_loss(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = model.sample_actions(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)
