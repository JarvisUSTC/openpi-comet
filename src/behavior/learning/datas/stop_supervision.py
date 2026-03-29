import numpy as np


def compute_stop_label_and_mask(
    *,
    frame_index: int,
    skill_end: int | None,
    pos_margin_frames: int = 0,
    neg_margin_frames: int = 15,
) -> tuple[np.float32, np.bool_]:
    """Compute weak stop supervision for 'skill end' detection.

    Semantics:
      - stop_label==1 indicates the current frame is at/after the skill end (within pos margin).
      - stop_label==0 indicates the current frame is definitely not near the end (before neg margin).
      - stop_mask==False indicates the label is ignored (uncertain window near the end).
    """
    if skill_end is None:
        return np.float32(0.0), np.bool_(False)

    d = int(skill_end) - int(frame_index)
    if d <= int(pos_margin_frames):
        return np.float32(1.0), np.bool_(True)
    if d >= int(neg_margin_frames):
        return np.float32(0.0), np.bool_(True)
    return np.float32(0.0), np.bool_(False)

