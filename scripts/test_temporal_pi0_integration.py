"""Integration test for Pi0 temporal prefix path (K>1)."""

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from openpi.models.model import Observation
from openpi.models.pi0 import Pi0
from openpi.models.pi0 import make_attn_mask
from openpi.models.pi0_config import Pi0Config
from openpi.shared import array_typing as at


def main():
    # Small config to keep CPU test lightweight.
    cfg = Pi0Config(
        pi05=True,
        action_horizon=4,
        action_dim=8,
        max_token_len=16,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        dtype="float32",
    )
    object.__setattr__(cfg, "video_memory_frames", 3)
    model = Pi0(cfg, rngs=nnx.Rngs(jax.random.key(0)))
    model.eval()

    bsize, num_frames = 2, 3
    # Keep default image resolution to match model lazy_init shape.
    img_h, img_w = 224, 224
    # K>1 images are flattened to BK in Observation.
    bk = bsize * num_frames
    obs = Observation(
        images={
            "base_0_rgb": jnp.ones((bk, img_h, img_w, 3), dtype=jnp.float32),
            "left_wrist_0_rgb": jnp.ones((bk, img_h, img_w, 3), dtype=jnp.float32),
            "right_wrist_0_rgb": jnp.ones((bk, img_h, img_w, 3), dtype=jnp.float32),
        },
        image_masks={
            # Expanded [B*K] mask as produced by from_dict for temporal inputs.
            "base_0_rgb": jnp.array([True, True, True, False, False, False], dtype=jnp.bool_),
            "left_wrist_0_rgb": jnp.array([True, True, True, False, False, False], dtype=jnp.bool_),
            "right_wrist_0_rgb": jnp.array([True, True, True, False, False, False], dtype=jnp.bool_),
        },
        state=jnp.zeros((bsize, cfg.action_dim), dtype=jnp.float32),
        tokenized_prompt=jnp.ones((bsize, cfg.max_token_len), dtype=jnp.int32),
        tokenized_prompt_mask=jnp.ones((bsize, cfg.max_token_len), dtype=jnp.bool_),
        temporal_mask=jnp.array([[False, True, True], [True, True, True]], dtype=jnp.bool_),
    )

    # Run real prefix embedding path.
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(obs)
    assert prefix_tokens.shape[0] == bsize, f"Expected B={bsize}, got {prefix_tokens.shape[0]}"
    assert prefix_mask.shape[0] == bsize, f"Expected B={bsize}, got {prefix_mask.shape[0]}"
    assert prefix_ar_mask.ndim == 1 and prefix_ar_mask.shape[0] == prefix_tokens.shape[1]

    # Ensure attention-mask/positions path is connected.
    attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    assert attn_mask.shape == (bsize, prefix_tokens.shape[1], prefix_tokens.shape[1])
    assert positions.shape == (bsize, prefix_tokens.shape[1])

    # One lightweight llm prefix call to verify end-to-end compatibility.
    _prefix_out, _kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=attn_mask, positions=positions)
    print("[PASS] Pi0 temporal prefix integration (K>1) end-to-end")


if __name__ == "__main__":
    with at.disable_typechecking():
        main()
