"""Strict unit tests for VideoMemoryDataset.

Tests the data pipeline migration (Phase 1).
- Tests 1-9: Pure logic tests (no JAX/Flax dependency, run anywhere)
- Tests 10-11: Import tests (require JAX/Flax, skipped if unavailable)

Run locally:  python scripts/test_video_memory_dataset.py
Run on server: python scripts/test_video_memory_dataset.py
"""

import sys
import os
import numpy as np
from collections import OrderedDict
from collections.abc import Sequence

# ── Standalone VideoMemoryDataset (no JAX import) ────────────────────────────

class _StandaloneVideoMemoryDataset:
    """Copy of VideoMemoryDataset logic for testing without JAX imports."""

    _DEFAULT_CAMERA_KEYS = (
        "observation.images.rgb.head",
        "observation.images.rgb.left_wrist",
        "observation.images.rgb.right_wrist",
    )
    _MAX_EPISODES_CACHED = 32

    def __init__(self, dataset, num_frames, stride=1, camera_keys=None):
        self._dataset = dataset
        self._num_frames = num_frames
        self._stride = max(1, stride)
        self._camera_keys = camera_keys or self._DEFAULT_CAMERA_KEYS
        self._buffers: "OrderedDict[int, dict[str, list]]" = OrderedDict()
        self._last_frame_idx: dict[int, int] = {}
        self._max_buffer = (num_frames - 1) * self._stride + 1
        self._stats_total = 0
        self._stats_valid = 0

    def _get_frame_idx(self, item) -> int:
        ts = item.get("timestamp")
        if ts is not None:
            return round(float(ts.item() if hasattr(ts, "item") else ts) * 30)
        return -1

    def __getitem__(self, index):
        item = self._dataset[index]
        if self._num_frames <= 1:
            return item

        ep_idx = item.get("episode_index")
        if hasattr(ep_idx, "item"):
            ep_idx = ep_idx.item()

        frame_idx = self._get_frame_idx(item)

        if ep_idx in self._buffers:
            self._buffers.move_to_end(ep_idx)
            if ep_idx in self._last_frame_idx and frame_idx >= 0:
                gap = abs(frame_idx - self._last_frame_idx[ep_idx])
                if gap > self._stride + 1:
                    self._buffers[ep_idx] = {}
        else:
            self._buffers[ep_idx] = {}
            while len(self._buffers) > self._MAX_EPISODES_CACHED:
                oldest_key = next(iter(self._buffers))
                del self._buffers[oldest_key]
                self._last_frame_idx.pop(oldest_key, None)

        if frame_idx >= 0:
            self._last_frame_idx[ep_idx] = frame_idx

        valid_flags = []
        for cam_key in self._camera_keys:
            if cam_key not in item:
                continue
            if cam_key not in self._buffers[ep_idx]:
                self._buffers[ep_idx][cam_key] = []

            buf = self._buffers[ep_idx][cam_key]
            frame = item[cam_key]
            buf.append(np.copy(frame))

            if len(buf) > self._max_buffer:
                buf.pop(0)

            K = self._num_frames
            available = buf[:-1]
            sampled = []
            valid_flags = []
            for i in range(K - 1, 0, -1):
                idx = len(available) - i * self._stride
                if idx < 0:
                    sampled.append(available[0] if available else buf[-1])
                    valid_flags.append(False)
                else:
                    sampled.append(available[idx])
                    valid_flags.append(True)

            item[f"{cam_key}_history"] = sampled
            item[f"{cam_key}_history_valid"] = valid_flags

        self._stats_total += 1
        if valid_flags and all(valid_flags):
            self._stats_valid += 1

        return item

    def __len__(self):
        return len(self._dataset)


# ── Helpers ──────────────────────────────────────────────────────────────────

class FakeFrameDataset:
    """Simulates BehaviorLeRobotDataset streaming behavior."""

    def __init__(self, episodes, cam_keys=None):
        self._cam_keys = cam_keys or [
            "observation.images.rgb.head",
            "observation.images.rgb.left_wrist",
            "observation.images.rgb.right_wrist",
        ]
        self._items = []
        for ep_idx, frames in enumerate(episodes):
            for f in frames:
                item = {
                    "episode_index": np.int64(ep_idx),
                    "timestamp": np.float64(f / 30.0),
                }
                for cam in self._cam_keys:
                    # Use float32 to avoid uint8 overflow for large frame indices
                    img = np.full((3, 224, 224), fill_value=f, dtype=np.float32)
                    img[0, 0, 0] = ep_idx
                    item[cam] = img
                self._items.append(item)

    def __getitem__(self, index):
        return {k: (np.copy(v) if isinstance(v, np.ndarray) else v)
                for k, v in self._items[index % len(self._items)].items()}

    def __len__(self):
        return len(self._items)


def _frame_value(img):
    return int(img[1, 0, 0])


def _episode_value(img):
    return int(img[0, 0, 0])


# ── Pure Logic Tests (no JAX needed) ─────────────────────────────────────────

def test_k1_passthrough():
    """K=1: VideoMemoryDataset should be a pure passthrough."""
    ds = FakeFrameDataset(episodes=[list(range(100))])
    vmd = _StandaloneVideoMemoryDataset(ds, num_frames=1, stride=1)

    assert len(vmd) == len(ds)

    for i in range(len(ds)):
        item = vmd[i]
        orig = ds[i]
        for cam in ds._cam_keys:
            assert f"{cam}_history" not in item, "K=1 should not add history keys"
            assert np.array_equal(item[cam], orig[cam]), "K=1 should return identical images"

    print("  PASS: test_k1_passthrough")


def test_k3_history_shape():
    """K=3: Should produce exactly K-1=2 history frames per camera."""
    ds = FakeFrameDataset(episodes=[list(range(100))])
    vmd = _StandaloneVideoMemoryDataset(ds, num_frames=3, stride=1)

    for i in range(len(ds)):
        item = vmd[i]

    cam = "observation.images.rgb.head"
    assert f"{cam}_history" in item
    assert len(item[f"{cam}_history"]) == 2
    assert len(item[f"{cam}_history_valid"]) == 2

    print("  PASS: test_k3_history_shape")


def test_k3_stride1_frame_order():
    """K=3, stride=1: History frames should be in chronological order."""
    ds = FakeFrameDataset(episodes=[list(range(100))])
    vmd = _StandaloneVideoMemoryDataset(ds, num_frames=3, stride=1)

    cam = "observation.images.rgb.head"
    for i in range(len(ds)):
        item = vmd[i]

    hist = item[f"{cam}_history"]
    valid = item[f"{cam}_history_valid"]

    assert _frame_value(hist[0]) == 97, f"Expected 97, got {_frame_value(hist[0])}"
    assert _frame_value(hist[1]) == 98, f"Expected 98, got {_frame_value(hist[1])}"
    assert all(valid)

    print("  PASS: test_k3_stride1_frame_order")


def test_k3_stride30_frame_order():
    """K=3, stride=30: Should sample every 30th frame."""
    ds = FakeFrameDataset(episodes=[list(range(200))])
    vmd = _StandaloneVideoMemoryDataset(ds, num_frames=3, stride=30)

    cam = "observation.images.rgb.head"
    for i in range(len(ds)):
        item = vmd[i]

    hist = item[f"{cam}_history"]
    valid = item[f"{cam}_history_valid"]

    assert _frame_value(hist[0]) == 139, f"Expected 139, got {_frame_value(hist[0])}"
    assert _frame_value(hist[1]) == 169, f"Expected 169, got {_frame_value(hist[1])}"
    assert all(valid)

    print("  PASS: test_k3_stride30_frame_order")


def test_warmup_padding():
    """Early frames should have valid_flags=False (padding)."""
    ds = FakeFrameDataset(episodes=[list(range(100))])
    vmd = _StandaloneVideoMemoryDataset(ds, num_frames=3, stride=1)

    cam = "observation.images.rgb.head"

    item0 = vmd[0]
    assert not any(item0[f"{cam}_history_valid"]), f"Frame 0 should have no valid history"

    item1 = vmd[1]
    assert item1[f"{cam}_history_valid"] == [False, True], f"Frame 1: expected [False, True]"

    item2 = vmd[2]
    assert all(item2[f"{cam}_history_valid"]), f"Frame 2: expected all valid"

    print("  PASS: test_warmup_padding")


def test_multi_episode_isolation():
    """Different episodes should have independent buffers."""
    ds = FakeFrameDataset(episodes=[list(range(50)), list(range(50))])
    vmd = _StandaloneVideoMemoryDataset(ds, num_frames=3, stride=1)

    cam = "observation.images.rgb.head"
    for i in range(len(ds)):
        item = vmd[i]

    # Last item is ep1, frame 49
    assert _episode_value(item[cam]) == 1
    hist = item[f"{cam}_history"]
    assert _episode_value(hist[0]) == 1, "History should be from episode 1"
    assert _episode_value(hist[1]) == 1, "History should be from episode 1"

    print("  PASS: test_multi_episode_isolation")


def test_lru_eviction():
    """Buffer should evict oldest episodes when exceeding _MAX_EPISODES_CACHED."""
    episodes = [list(range(10)) for _ in range(35)]
    ds = FakeFrameDataset(episodes=episodes)
    vmd = _StandaloneVideoMemoryDataset(ds, num_frames=3, stride=1)

    for i in range(len(ds)):
        vmd[i]

    assert len(vmd._buffers) <= 32, f"LRU should keep <= 32, got {len(vmd._buffers)}"

    print("  PASS: test_lru_eviction")


def test_gap_detection_clears_buffer():
    """Large frame gaps should clear the buffer."""
    frames = list(range(10)) + list(range(100, 110))
    ds = FakeFrameDataset(episodes=[frames])
    vmd = _StandaloneVideoMemoryDataset(ds, num_frames=3, stride=1)

    cam = "observation.images.rgb.head"
    for i in range(len(ds)):
        item = vmd[i]

    hist = item[f"{cam}_history"]
    assert _frame_value(hist[0]) >= 100, f"After gap, history should be post-gap"
    assert _frame_value(hist[1]) >= 100, f"After gap, history should be post-gap"

    print("  PASS: test_gap_detection_clears_buffer")


def test_k6_stride30():
    """K=6, stride=30: Full video memory configuration."""
    ds = FakeFrameDataset(episodes=[list(range(500))])
    vmd = _StandaloneVideoMemoryDataset(ds, num_frames=6, stride=30)

    cam = "observation.images.rgb.head"
    for i in range(len(ds)):
        item = vmd[i]

    hist = item[f"{cam}_history"]
    valid = item[f"{cam}_history_valid"]

    expected = [349, 379, 409, 439, 469]
    actual = [_frame_value(h) for h in hist]

    assert len(hist) == 5, f"K=6 should have 5 history frames"
    assert actual == expected, f"Expected {expected}, got {actual}"
    assert all(valid)

    print("  PASS: test_k6_stride30")


def test_stats_tracking():
    """Should track valid history statistics."""
    ds = FakeFrameDataset(episodes=[list(range(100))])
    vmd = _StandaloneVideoMemoryDataset(ds, num_frames=3, stride=1)

    for i in range(len(ds)):
        vmd[i]

    assert vmd._stats_total == 100, f"Expected 100 total, got {vmd._stats_total}"
    assert vmd._stats_valid == 98, f"Expected 98 valid, got {vmd._stats_valid}"

    print("  PASS: test_stats_tracking")


# ── Import Tests (require JAX/Flax) ─────────────────────────────────────────

def test_config_fields_exist():
    """pi0_config should have video_memory fields."""
    from openpi.models.pi0_config import Pi0Config

    cfg = Pi0Config(pi05=True, action_horizon=32)
    assert cfg.video_memory_frames == 1
    assert cfg.video_memory_stride_s == 1.0

    cfg3 = Pi0Config(pi05=True, action_horizon=32, video_memory_frames=3)
    assert cfg3.video_memory_frames == 3

    print("  PASS: test_config_fields_exist")


def test_create_behavior_dataset_signature():
    """create_behavior_dataset should accept video_memory params."""
    import inspect
    from openpi.training.data_loader import create_behavior_dataset

    params = list(inspect.signature(create_behavior_dataset).parameters.keys())
    assert "video_memory_frames" in params
    assert "video_memory_stride_s" in params

    print("  PASS: test_create_behavior_dataset_signature")


def test_imported_vmd_matches_standalone():
    """Imported VideoMemoryDataset should produce same results as standalone."""
    from openpi.training.data_loader import VideoMemoryDataset

    ds = FakeFrameDataset(episodes=[list(range(100))])

    vmd_standalone = _StandaloneVideoMemoryDataset(ds, num_frames=3, stride=1)
    vmd_imported = VideoMemoryDataset(ds, num_frames=3, stride=1)

    cam = "observation.images.rgb.head"
    for i in range(len(ds)):
        item_s = vmd_standalone[i]
        item_i = vmd_imported[i]

        if i >= 2:
            for j in range(2):
                assert np.array_equal(
                    item_s[f"{cam}_history"][j],
                    item_i[f"{cam}_history"][j],
                ), f"Frame {i}, history[{j}] mismatch"
            assert item_s[f"{cam}_history_valid"] == item_i[f"{cam}_history_valid"]

    print("  PASS: test_imported_vmd_matches_standalone")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(project_root, "src"))

    # Pure logic tests (always run)
    pure_tests = [
        test_k1_passthrough,
        test_k3_history_shape,
        test_k3_stride1_frame_order,
        test_k3_stride30_frame_order,
        test_warmup_padding,
        test_multi_episode_isolation,
        test_lru_eviction,
        test_gap_detection_clears_buffer,
        test_k6_stride30,
        test_stats_tracking,
    ]

    # Import tests (need JAX/Flax)
    import_tests = [
        test_config_fields_exist,
        test_create_behavior_dataset_signature,
        test_imported_vmd_matches_standalone,
    ]

    print(f"\n{'='*50}")
    print(f"Phase 1: VideoMemoryDataset Tests")
    print(f"{'='*50}")

    passed = 0
    failed = 0
    skipped = 0

    print(f"\n[Pure Logic Tests] ({len(pure_tests)} tests)")
    for test in pure_tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test.__name__}: {e}")
            failed += 1

    # Check if JAX is available
    try:
        import jax  # noqa: F401
        has_jax = True
    except ImportError:
        has_jax = False

    print(f"\n[Import Tests] ({len(import_tests)} tests) {'[JAX available]' if has_jax else '[JAX not found, skipping]'}")
    if has_jax:
        for test in import_tests:
            try:
                test()
                passed += 1
            except Exception as e:
                print(f"  FAIL: {test.__name__}: {e}")
                failed += 1
    else:
        skipped = len(import_tests)

    total = len(pure_tests) + len(import_tests)
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped, {total} total")
    if failed > 0:
        sys.exit(1)
    else:
        print("All runnable tests passed!")


if __name__ == "__main__":
    main()
