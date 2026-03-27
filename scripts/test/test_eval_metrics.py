"""Unit tests for eval_step action metrics (action_mse, action_cosine_sim).

Pure numpy tests — no JAX, no model, no GPU required.
"""

import numpy as np


def compute_action_mse(pred_actions, gt_actions):
    """Mirrors the action_mse computation in eval_step."""
    action_error = pred_actions - gt_actions
    return np.mean(np.square(action_error))


def compute_cosine_sim(pred_actions, gt_actions):
    """Mirrors the cosine_sim computation in eval_step."""
    pred_flat = pred_actions.reshape(pred_actions.shape[0], -1)
    gt_flat = gt_actions.reshape(gt_actions.shape[0], -1)
    cos_sim = np.sum(pred_flat * gt_flat, axis=-1) / (
        np.linalg.norm(pred_flat, axis=-1) * np.linalg.norm(gt_flat, axis=-1) + 1e-8
    )
    return np.mean(cos_sim)


def test_mse_identical():
    """MSE should be 0 when predicted == GT."""
    actions = np.random.randn(4, 32, 23).astype(np.float32)
    mse = compute_action_mse(actions, actions)
    assert mse == 0.0, f"Expected 0, got {mse}"
    print("PASS test_mse_identical")


def test_mse_known_value():
    """MSE with a known offset should match hand-computed value."""
    gt = np.zeros((2, 4, 3), dtype=np.float32)
    pred = np.ones((2, 4, 3), dtype=np.float32)
    mse = compute_action_mse(pred, gt)
    assert np.isclose(mse, 1.0), f"Expected 1.0, got {mse}"
    print("PASS test_mse_known_value")


def test_mse_positive():
    """MSE should always be non-negative."""
    pred = np.random.randn(8, 32, 23).astype(np.float32)
    gt = np.random.randn(8, 32, 23).astype(np.float32)
    mse = compute_action_mse(pred, gt)
    assert mse >= 0, f"MSE should be non-negative, got {mse}"
    print("PASS test_mse_positive")


def test_cosine_sim_identical():
    """Cosine similarity should be 1.0 when predicted == GT."""
    actions = np.random.randn(4, 32, 23).astype(np.float32)
    sim = compute_cosine_sim(actions, actions)
    assert np.isclose(sim, 1.0, atol=1e-6), f"Expected 1.0, got {sim}"
    print("PASS test_cosine_sim_identical")


def test_cosine_sim_opposite():
    """Cosine similarity should be -1.0 when predicted == -GT."""
    gt = np.random.randn(4, 32, 23).astype(np.float32)
    pred = -gt
    sim = compute_cosine_sim(pred, gt)
    assert np.isclose(sim, -1.0, atol=1e-6), f"Expected -1.0, got {sim}"
    print("PASS test_cosine_sim_opposite")


def test_cosine_sim_orthogonal():
    """Cosine similarity should be ~0 for orthogonal vectors."""
    # Construct two orthogonal vectors in 2D, batch=1, horizon=1
    pred = np.array([[[1.0, 0.0]]], dtype=np.float32)
    gt = np.array([[[0.0, 1.0]]], dtype=np.float32)
    sim = compute_cosine_sim(pred, gt)
    assert np.isclose(sim, 0.0, atol=1e-6), f"Expected 0.0, got {sim}"
    print("PASS test_cosine_sim_orthogonal")


def test_cosine_sim_range():
    """Cosine similarity should be in [-1, 1]."""
    pred = np.random.randn(8, 32, 23).astype(np.float32)
    gt = np.random.randn(8, 32, 23).astype(np.float32)
    sim = compute_cosine_sim(pred, gt)
    assert -1.0 - 1e-6 <= sim <= 1.0 + 1e-6, f"Cosine sim out of range: {sim}"
    print("PASS test_cosine_sim_range")


def test_batch_shapes():
    """Verify metrics work with typical B1K shapes: [B, action_horizon, action_dim]."""
    B, AH, AD = 8, 32, 23
    pred = np.random.randn(B, AH, AD).astype(np.float32)
    gt = np.random.randn(B, AH, AD).astype(np.float32)
    mse = compute_action_mse(pred, gt)
    sim = compute_cosine_sim(pred, gt)
    assert mse.shape == (), f"MSE should be scalar, got shape {mse.shape}"
    assert sim.shape == (), f"Cosine sim should be scalar, got shape {sim.shape}"
    print("PASS test_batch_shapes")


if __name__ == "__main__":
    np.random.seed(42)
    test_mse_identical()
    test_mse_known_value()
    test_mse_positive()
    test_cosine_sim_identical()
    test_cosine_sim_opposite()
    test_cosine_sim_orthogonal()
    test_cosine_sim_range()
    test_batch_shapes()
    print("\nAll 8 tests passed!")
