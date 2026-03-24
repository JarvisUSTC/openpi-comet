"""Sanity tests for temporal SigLIP integration in JAX."""

import jax
import jax.numpy as jnp

from openpi.models import siglip


def main():
    model = siglip.Module(variant="mu", pool_type="none")
    bsize, num_frames = 2, 3
    h, w, c = 32, 32, 3

    img_k1 = jnp.ones((bsize, h, w, c), dtype=jnp.float32)
    params = model.init(jax.random.key(0), img_k1, train=False)

    # Test 1: K=1 should keep no-op equivalence.
    out_ref, _ = model.apply(params, img_k1, train=False)
    out_k1, _ = model.apply(
        params,
        img_k1,
        train=False,
        num_frames=1,
        temporal_mask=jnp.ones((bsize, 1), dtype=bool),
    )
    max_diff = float(jnp.max(jnp.abs(out_ref - out_k1)))
    assert max_diff < 1e-6, f"K=1 regression detected, max_diff={max_diff}"
    print(f"[PASS] K=1 no-op equivalence (max_diff={max_diff:.3e})")

    # Test 2: K=3 output shape should collapse back to batch size B.
    img_k3 = jnp.ones((bsize * num_frames, h, w, c), dtype=jnp.float32)
    mask_k3 = jnp.array([[False, True, True], [True, True, True]], dtype=bool)
    out_k3, _ = model.apply(params, img_k3, train=False, num_frames=num_frames, temporal_mask=mask_k3)
    assert out_k3.shape[0] == bsize, f"Expected batch {bsize}, got {out_k3.shape[0]}"
    print(f"[PASS] K=3 shape check -> {out_k3.shape}")

    # Test 3: temporal_mask with False values should run and remain finite.
    out_full_true, _ = model.apply(
        params,
        img_k3,
        train=False,
        num_frames=num_frames,
        temporal_mask=jnp.ones((bsize, num_frames), dtype=bool),
    )
    assert jnp.isfinite(out_k3).all(), "Output contains non-finite values with partial temporal_mask"
    assert jnp.isfinite(out_full_true).all(), "Output contains non-finite values with full temporal_mask"
    print("[PASS] temporal_mask stability check")

    print("ALL PASS")


if __name__ == "__main__":
    main()
