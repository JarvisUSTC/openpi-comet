"""Unit tests for temporal image-mask alignment in Pi0 prefix embedding."""

import jax.numpy as jnp

from openpi.models.pi0 import _align_image_mask_with_token_batch


def main():
    # Case 1: K=1 should be no-op.
    mask_k1 = jnp.array([True, False], dtype=bool)
    out_k1 = _align_image_mask_with_token_batch(mask_k1, token_batch=2, num_frames=1)
    assert out_k1.shape == (2,)
    assert jnp.array_equal(out_k1, mask_k1)
    print("[PASS] K=1 no-op")

    # Case 2: K=3, mask expanded [B*K] should be reduced to [B].
    # Expanded by np.repeat([True, False], 3) -> [T,T,T,F,F,F]
    mask_expanded = jnp.array([True, True, True, False, False, False], dtype=bool)
    out_reduced = _align_image_mask_with_token_batch(mask_expanded, token_batch=2, num_frames=3)
    expected = jnp.array([True, False], dtype=bool)
    assert out_reduced.shape == (2,)
    assert jnp.array_equal(out_reduced, expected)
    print("[PASS] K=3 expanded mask reduced to batch B")

    # Case 3: K=3, already [B] should stay unchanged.
    mask_b = jnp.array([True, False], dtype=bool)
    out_b = _align_image_mask_with_token_batch(mask_b, token_batch=2, num_frames=3)
    assert out_b.shape == (2,)
    assert jnp.array_equal(out_b, mask_b)
    print("[PASS] K=3 with already-aligned mask stays unchanged")

    # Case 4: Mismatched size should be preserved (no silent truncation).
    odd_mask = jnp.array([True, False, True, False], dtype=bool)
    out_odd = _align_image_mask_with_token_batch(odd_mask, token_batch=2, num_frames=3)
    assert out_odd.shape == (4,)
    assert jnp.array_equal(out_odd, odd_mask)
    print("[PASS] mismatch case preserved (no reduction)")

    print("ALL PASS")


if __name__ == "__main__":
    main()
