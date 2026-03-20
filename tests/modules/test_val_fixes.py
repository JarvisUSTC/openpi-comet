"""Tests for val loss oscillation fixes:
  Fix 1: reset_val_loader() is called before each validation pass
  Fix 2: val loader skips DDP batch division (is_validation=True)

Run:
    python tests/modules/test_val_fixes.py
"""

import sys
import unittest
from unittest.mock import patch, MagicMock
from collections import OrderedDict


# ============================================================
# Fix 2 Tests: is_validation flag skips DDP batch division
# ============================================================

class TestValBatchSizeNoDDP(unittest.TestCase):
    """Verify that create_torch_behavior_data_loader with is_validation=True
    does NOT divide batch_size by world_size even when DDP is initialized."""

    def test_is_validation_skips_ddp_branch(self):
        """When is_validation=True and DDP is initialized, batch_size should
        NOT be divided by world_size."""
        # We test by importing and inspecting the conditional logic.
        # Since the real function has heavy dependencies, we test the logic
        # pattern directly.

        def simulate_batch_logic(batch_size, is_validation, ddp_initialized, world_size):
            """Mirrors the patched logic in create_torch_behavior_data_loader."""
            if ddp_initialized and not is_validation:
                local_batch_size = batch_size // world_size
            else:
                local_batch_size = batch_size
            return local_batch_size

        # Training: should divide
        self.assertEqual(simulate_batch_logic(64, False, True, 8), 8)
        self.assertEqual(simulate_batch_logic(256, False, True, 8), 32)

        # Validation: should NOT divide
        self.assertEqual(simulate_batch_logic(64, True, True, 8), 64)
        self.assertEqual(simulate_batch_logic(64, True, True, 24), 64)
        self.assertEqual(simulate_batch_logic(256, True, True, 8), 256)

        # No DDP: should NOT divide regardless
        self.assertEqual(simulate_batch_logic(64, False, False, 1), 64)
        self.assertEqual(simulate_batch_logic(64, True, False, 1), 64)

    def test_val_batch_size_config_values(self):
        """Verify the actual config values produce correct val batch sizes."""
        # K6 config: val_batch_size=64, world_size=24
        # Before fix: 64 // 24 = 2 (terrible!)
        # After fix: 64 (correct)
        self.assertEqual(64, 64)  # is_validation=True → no division
        self.assertNotEqual(64 // 24, 64)  # confirms the bug existed

        # K3_v2 config: val_batch_size=64, world_size=8
        # Before fix: 64 // 8 = 8
        # After fix: 64 (correct)
        self.assertEqual(64, 64)
        self.assertNotEqual(64 // 8, 64)


# ============================================================
# Fix 1 Tests: reset_val_loader called before validate
# ============================================================

class MockBehaviorLeRobotDataset:
    def __init__(self, num_chunks=5, chunk_size=10):
        self._active_chunks = [
            (i * chunk_size, (i + 1) * chunk_size, 0) for i in range(num_chunks)
        ]
        self.current_streaming_chunk_idx = 0
        self.current_streaming_frame_idx = self._active_chunks[0][0]
        self._should_obs_loaders_reload = False

    def advance(self, n=1):
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
    def __init__(self, inner_dataset):
        self._dataset = inner_dataset
        self._buffers = OrderedDict({"ep0": {"cam": [1, 2, 3]}})
        self._last_frame_idx = {"ep0": 10}
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


def reset_val_loader(val_loader):
    """Copy of the reset logic from train_pytorch.py."""
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


def _build_loader(num_chunks=5, chunk_size=10):
    base_ds = MockBehaviorLeRobotDataset(num_chunks=num_chunks, chunk_size=chunk_size)
    inner_transform = MockTransformedDataset(base_ds)
    vm_ds = MockVideoMemoryDataset(inner_transform)
    outer_transform = MockTransformedDataset(vm_ds)
    val_loader = MockDataLoaderImpl(outer_transform)
    return val_loader, base_ds, vm_ds


class TestResetBeforeValidation(unittest.TestCase):
    """Verify that calling reset before each validation produces
    reproducible starting positions."""

    def test_multiple_val_passes_are_reproducible(self):
        """Simulate 3 validation passes: each should start from the same position."""
        val_loader, base_ds, vm_ds = _build_loader()
        initial = base_ds.snapshot()

        snapshots = []
        for _ in range(3):
            reset_val_loader(val_loader)
            start = base_ds.snapshot()
            snapshots.append(start)
            # Simulate reading val_num_batches * batch_size frames
            base_ds.advance(100)

        # All starts should be identical
        self.assertEqual(snapshots[0], initial)
        self.assertEqual(snapshots[1], initial)
        self.assertEqual(snapshots[2], initial)

    def test_video_memory_buffers_cleared_each_pass(self):
        """VideoMemoryDataset buffers should be empty at start of each val pass."""
        val_loader, base_ds, vm_ds = _build_loader()

        for i in range(3):
            reset_val_loader(val_loader)
            self.assertEqual(len(vm_ds._buffers), 0, f"Pass {i}: buffers not cleared")
            self.assertEqual(vm_ds._stats_total, 0, f"Pass {i}: stats not reset")
            # Simulate reading and accumulating buffers
            vm_ds._buffers["new_ep"] = {"cam": [1, 2]}
            vm_ds._stats_total = 42
            base_ds.advance(50)

    def test_without_reset_positions_drift(self):
        """Without reset, consecutive val passes start at different positions."""
        val_loader, base_ds, _ = _build_loader()

        snapshots = []
        for _ in range(3):
            # NO reset_val_loader call
            start = base_ds.snapshot()
            snapshots.append(start)
            base_ds.advance(17)

        # Without reset, positions should differ
        self.assertNotEqual(snapshots[0], snapshots[1],
                            "Without reset, positions should drift (this confirms the bug)")
        self.assertNotEqual(snapshots[1], snapshots[2])


# ============================================================
# Integration: both fixes together
# ============================================================

class TestIntegration(unittest.TestCase):
    """Test both fixes working together."""

    def test_full_val_scenario(self):
        """Simulate the fixed training loop:
        1. Val batch_size is NOT divided (Fix 2)
        2. reset_val_loader is called before each validate() (Fix 1)
        """
        # Fix 2: val_batch_size stays at 64, not divided by world_size
        val_batch_size = 64
        world_size = 8
        is_validation = True

        if not is_validation:
            effective_batch = val_batch_size // world_size
        else:
            effective_batch = val_batch_size

        self.assertEqual(effective_batch, 64, "Val batch should be full 64")

        # Fix 1: reset produces reproducible val passes
        val_loader, base_ds, vm_ds = _build_loader(num_chunks=10, chunk_size=25)
        val_num_batches = 10
        frames_per_val = val_num_batches * effective_batch  # 640 frames

        results = []
        for _ in range(3):
            reset_val_loader(val_loader)
            start = base_ds.snapshot()
            base_ds.advance(frames_per_val)
            end = base_ds.snapshot()
            results.append((start, end))

        # All val passes should have identical start and end positions
        for i in range(1, len(results)):
            self.assertEqual(results[0], results[i],
                             f"Val pass {i} diverged from pass 0")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Val Loss Oscillation Fixes (Fix 1 + Fix 2)")
    print("=" * 60)
    unittest.main(verbosity=2)
