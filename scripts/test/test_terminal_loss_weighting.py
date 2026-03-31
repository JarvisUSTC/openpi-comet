"""Unit tests for terminal loss weighting logic.

Pure numpy tests — no JAX/OpenPI imports required.
Run locally: python scripts/test/test_terminal_loss_weighting.py
"""

import numpy as np


def compute_terminal_loss_weight(
    *,
    frame_index: int,
    skill_end: int | None,
    action_chunk=None,
    final_window_frames: int = 5,
    near_window_frames: int = 15,
    final_weight: float = 4.0,
    near_weight: float = 2.0,
    max_base_motion_norm: float = 0.05,
) -> np.float32:
    """Mirror the dataset-side terminal weighting helper."""
    if skill_end is None:
        return np.float32(1.0)

    d = int(skill_end) - int(frame_index)
    if action_chunk is None:
        return np.float32(1.0)

    arr = np.asarray(action_chunk)
    if arr.ndim < 2 or arr.shape[0] == 0 or arr.shape[-1] < 3:
        return np.float32(1.0)

    end_offset = max(0, min(int(d), arr.shape[0] - 1))
    base_motion_norm = np.linalg.norm(arr[end_offset, :3])
    if base_motion_norm > float(max_base_motion_norm):
        return np.float32(1.0)

    if d <= int(final_window_frames):
        return np.float32(final_weight)
    if d <= int(near_window_frames):
        return np.float32(near_weight)
    return np.float32(1.0)


def reduce_chunked_loss(chunked_loss: np.ndarray, sample_weights: np.ndarray | None) -> np.float32:
    """Mirror the train-time weighted reduction over [B, H] chunk losses."""
    if sample_weights is None:
        return np.float32(np.mean(chunked_loss))

    sample_loss = np.mean(chunked_loss, axis=-1)
    return np.float32(np.sum(sample_loss * sample_weights) / (np.sum(sample_weights) + 1e-8))


def test_terminal_loss_weight_windows():
    quiet_chunk = np.zeros((32, 23), dtype=np.float32)
    noisy_chunk = np.zeros((32, 23), dtype=np.float32)
    noisy_chunk[14, 0] = 0.2

    assert compute_terminal_loss_weight(frame_index=80, skill_end=100, action_chunk=quiet_chunk) == np.float32(1.0)
    assert compute_terminal_loss_weight(frame_index=86, skill_end=100, action_chunk=quiet_chunk) == np.float32(2.0)
    assert compute_terminal_loss_weight(frame_index=95, skill_end=100, action_chunk=quiet_chunk) == np.float32(4.0)
    assert compute_terminal_loss_weight(frame_index=100, skill_end=100, action_chunk=quiet_chunk) == np.float32(4.0)
    assert compute_terminal_loss_weight(frame_index=86, skill_end=100, action_chunk=noisy_chunk) == np.float32(1.0)
    assert compute_terminal_loss_weight(frame_index=50, skill_end=None, action_chunk=quiet_chunk) == np.float32(1.0)
    print("PASS test_terminal_loss_weight_windows")


def test_weighted_loss_reduction():
    chunked_loss = np.array(
        [
            [1.0, 3.0],
            [2.0, 4.0],
            [5.0, 7.0],
        ],
        dtype=np.float32,
    )
    weights = np.array([1.0, 2.0, 4.0], dtype=np.float32)

    # Sample means are [2, 3, 6]; weighted mean = (2 + 6 + 24) / 7
    reduced = reduce_chunked_loss(chunked_loss, weights)
    assert np.isclose(reduced, 32.0 / 7.0, atol=1e-6), f"Unexpected weighted loss: {reduced}"
    print("PASS test_weighted_loss_reduction")


if __name__ == "__main__":
    test_terminal_loss_weight_windows()
    test_weighted_loss_reduction()
    print("\nAll 2 terminal loss weighting tests passed!")
