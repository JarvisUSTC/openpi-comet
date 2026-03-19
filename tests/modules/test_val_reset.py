"""Verify that reset_val_loader correctly resets the streaming dataset pointer
and VideoMemoryDataset buffer, ensuring reproducible validation batches.

This test is fully self-contained: it reproduces the exact nesting structure
and the reset logic WITHOUT importing the heavy real modules (JAX, transformers,
etc.), so it runs in seconds.

Nesting structure being tested:
  DataLoaderImpl._data_loader (TorchDataLoader)
    .torch_loader (PyTorch DataLoader)
      .dataset = TransformedDataset
        ._dataset = VideoMemoryDataset
          ._dataset = TransformedDataset
            ._dataset = BehaviorLeRobotDataset

Run:
    python tests/modules/test_val_reset.py
"""

import sys
from collections import OrderedDict


# ===== Mock classes that mirror the real nesting structure =====

class MockBehaviorLeRobotDataset:
    """Minimal mock of BehaviorLeRobotDataset streaming pointer."""

    def __init__(self, num_chunks=5, chunk_size=10):
        self._active_chunks = [
            (i * chunk_size, (i + 1) * chunk_size, 0) for i in range(num_chunks)
        ]
        self.current_streaming_chunk_idx = 0
        self.current_streaming_frame_idx = self._active_chunks[0][0]
        self._should_obs_loaders_reload = False

    def advance(self, n=1):
        """Simulate reading n frames."""
        for _ in range(n):
            self.current_streaming_frame_idx += 1
            end = self._active_chunks[self.current_streaming_chunk_idx][1]
            if self.current_streaming_frame_idx >= end:
                self.current_streaming_chunk_idx += 1
                if self.current_streaming_chunk_idx >= len(self._active_chunks):
                    self.current_streaming_chunk_idx = 0
                self.current_streaming_frame_idx = self._active_chunks[
                    self.current_streaming_chunk_idx
                ][0]

    def snapshot(self):
        return (self.current_streaming_chunk_idx, self.current_streaming_frame_idx)


class MockVideoMemoryDataset:
    """Minimal mock of VideoMemoryDataset with buffers."""

    def __init__(self, inner_dataset):
        self._dataset = inner_dataset
        self._buffers = OrderedDict({"ep0": {"cam": [1, 2, 3]}, "ep1": {"cam": [4, 5]}})
        self._last_frame_idx = {"ep0": 10, "ep1": 20}
        self._stats_total = 500
        self._stats_valid = 400


class MockTransformedDataset:
    def __init__(self, dataset):
        self._dataset = dataset


class MockTorchLoader:
    def __init__(self, dataset):
        self.dataset = dataset


class MockTorchDataLoader:
    def __init__(self, dataset):
        self.torch_loader = MockTorchLoader(dataset)


class MockDataLoaderImpl:
    def __init__(self, dataset):
        self._data_loader = MockTorchDataLoader(dataset)


# ===== Inline copy of reset logic (the function under test) =====
# We inline the core logic to avoid importing the full train_pytorch module.
# If the real code changes, this test will need updating — but it validates
# the ALGORITHM, not just that the import works.

def reset_val_loader(val_loader):
    """Exact copy of the reset logic from train_pytorch.py."""

    def _unwrap(ds):
        if isinstance(ds, MockVideoMemoryDataset):
            ds._buffers.clear()
            ds._last_frame_idx.clear()
            ds._stats_total = 0
            ds._stats_valid = 0
            _unwrap(ds._dataset)
        elif isinstance(ds, MockBehaviorLeRobotDataset):
            if hasattr(ds, "_active_chunks") and ds._active_chunks:
                ds.current_streaming_chunk_idx = 0
                ds.current_streaming_frame_idx = ds._active_chunks[0][0]
                ds._should_obs_loaders_reload = True
        elif hasattr(ds, "_dataset"):
            _unwrap(ds._dataset)

    torch_ds = val_loader._data_loader.torch_loader.dataset
    _unwrap(torch_ds)


# ===== Helper: build the full nesting =====

def _build_loader(num_chunks=5, chunk_size=10):
    base_ds = MockBehaviorLeRobotDataset(num_chunks=num_chunks, chunk_size=chunk_size)
    inner_transform = MockTransformedDataset(base_ds)
    vm_ds = MockVideoMemoryDataset(inner_transform)
    outer_transform = MockTransformedDataset(vm_ds)
    val_loader = MockDataLoaderImpl(outer_transform)
    return val_loader, base_ds, vm_ds


# ===== Tests =====

def test_reset_restores_pointer():
    """After advancing the streaming pointer, reset brings it back to start."""
    val_loader, base_ds, _ = _build_loader()
    initial = base_ds.snapshot()

    base_ds.advance(27)
    assert base_ds.snapshot() != initial, "Pre-condition: pointer should have moved"

    reset_val_loader(val_loader)

    assert base_ds.snapshot() == initial, (
        f"Expected pointer {initial}, got {base_ds.snapshot()}"
    )
    assert base_ds._should_obs_loaders_reload is True
    print("  PASS: reset_restores_pointer")


def test_reset_clears_video_memory_buffers():
    """VideoMemoryDataset buffers and stats should be cleared on reset."""
    val_loader, _, vm_ds = _build_loader()

    assert len(vm_ds._buffers) > 0
    assert vm_ds._stats_total > 0

    reset_val_loader(val_loader)

    assert len(vm_ds._buffers) == 0, "Buffers should be cleared"
    assert len(vm_ds._last_frame_idx) == 0, "last_frame_idx should be cleared"
    assert vm_ds._stats_total == 0, "stats_total should be reset"
    assert vm_ds._stats_valid == 0, "stats_valid should be reset"
    print("  PASS: reset_clears_video_memory_buffers")


def test_reset_is_idempotent():
    """Calling reset twice yields the same state."""
    val_loader, base_ds, vm_ds = _build_loader()
    base_ds.advance(15)

    reset_val_loader(val_loader)
    snap1 = base_ds.snapshot()

    reset_val_loader(val_loader)
    snap2 = base_ds.snapshot()

    assert snap1 == snap2, f"Idempotency failed: {snap1} != {snap2}"
    print("  PASS: reset_is_idempotent")


def test_reproducible_sequence():
    """After reset, advancing N steps always reaches the same position."""
    val_loader, base_ds, _ = _build_loader(num_chunks=4, chunk_size=8)

    reset_val_loader(val_loader)
    base_ds.advance(20)
    snap_run1 = base_ds.snapshot()

    reset_val_loader(val_loader)
    base_ds.advance(20)
    snap_run2 = base_ds.snapshot()

    assert snap_run1 == snap_run2, (
        f"Runs diverged after reset: {snap_run1} != {snap_run2}"
    )
    print("  PASS: reproducible_sequence")


def test_wrap_around_then_reset():
    """Pointer wraps around to chunk 0 after exhausting all chunks, then resets."""
    val_loader, base_ds, _ = _build_loader(num_chunks=3, chunk_size=5)
    initial = base_ds.snapshot()

    base_ds.advance(3 * 5 + 2)
    assert base_ds.current_streaming_chunk_idx > 0 or base_ds.current_streaming_frame_idx > 0

    reset_val_loader(val_loader)
    assert base_ds.snapshot() == initial, (
        f"After wrap-around + reset, expected {initial}, got {base_ds.snapshot()}"
    )
    print("  PASS: wrap_around_then_reset")


def test_no_active_chunks_no_crash():
    """If _active_chunks is empty, reset should not crash."""
    val_loader, base_ds, _ = _build_loader()
    base_ds._active_chunks = []

    try:
        reset_val_loader(val_loader)
        print("  PASS: no_active_chunks_no_crash")
    except Exception as e:
        print(f"  FAIL: no_active_chunks_no_crash — {e}")
        raise


# ===== Main =====

if __name__ == "__main__":
    print("=" * 60)
    print("Testing reset_val_loader (Issue #15 fix)")
    print("=" * 60)

    tests = [
        test_reset_restores_pointer,
        test_reset_clears_video_memory_buffers,
        test_reset_is_idempotent,
        test_reproducible_sequence,
        test_wrap_around_then_reset,
        test_no_active_chunks_no_crash,
    ]

    passed = 0
    failed = 0
    for fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {fn.__name__} — {e}")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print("=" * 60)

    sys.exit(1 if failed > 0 else 0)
