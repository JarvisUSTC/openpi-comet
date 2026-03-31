import numpy as np

from behavior.learning.datas.stop_supervision import compute_stop_label_and_mask


def test_compute_stop_label_and_mask_skill_end_none():
    label, mask = compute_stop_label_and_mask(frame_index=10, skill_end=None)
    assert label.dtype == np.float32
    assert mask.dtype == np.bool_
    assert bool(mask) is False


def test_compute_stop_label_and_mask_positive():
    label, mask = compute_stop_label_and_mask(frame_index=100, skill_end=100, pos_margin_frames=0, neg_margin_frames=15)
    assert float(label) == 1.0
    assert bool(mask) is True


def test_compute_stop_label_and_mask_negative_far():
    label, mask = compute_stop_label_and_mask(frame_index=50, skill_end=100, pos_margin_frames=0, neg_margin_frames=15)
    assert float(label) == 0.0
    assert bool(mask) is True


def test_compute_stop_label_and_mask_ignore_window():
    # d=10 is within (pos=0, neg=15) => ignore
    label, mask = compute_stop_label_and_mask(frame_index=90, skill_end=100, pos_margin_frames=0, neg_margin_frames=15)
    assert bool(mask) is False


def test_compute_stop_label_and_mask_soft_label_ramp():
    # soft label ramps linearly within (0, pos] toward 1 at d=0
    label, mask = compute_stop_label_and_mask(
        frame_index=90,
        skill_end=100,  # d=10
        pos_margin_frames=60,
        neg_margin_frames=180,
        soft_labels=True,
    )
    assert bool(mask) is True
    assert 0.0 < float(label) < 1.0
    assert np.isclose(float(label), 1.0 - (10.0 / 60.0), atol=1e-6)
