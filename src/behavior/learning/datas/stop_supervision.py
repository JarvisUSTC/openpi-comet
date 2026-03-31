import numpy as np


def compute_stop_label_and_mask(
    *,
    frame_index: int,
    skill_end: int | None,
    pos_margin_frames: int = 0,
    neg_margin_frames: int = 15,
    soft_labels: bool = False,
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
        if not soft_labels:
            return np.float32(1.0), np.bool_(True)
        pos = int(pos_margin_frames)
        if pos <= 0 or d <= 0:
            return np.float32(1.0), np.bool_(True)
        # Linear ramp within (0, pos]: closer to end => label closer to 1.
        # Clamp for safety in case of inconsistent annotations.
        y = 1.0 - (float(d) / float(pos))
        y = float(np.clip(y, 0.0, 1.0))
        return np.float32(y), np.bool_(True)
    if d >= int(neg_margin_frames):
        return np.float32(0.0), np.bool_(True)
    return np.float32(0.0), np.bool_(False)
