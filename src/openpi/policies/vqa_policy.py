import dataclasses
import math

import numpy as np
from PIL import Image

from openpi import transforms


def _parse_image(image) -> np.ndarray:
    if isinstance(image, Image.Image):
        image = image.convert("RGB")
    image = np.asarray(image)
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            image = np.clip(image, 0.0, 1.0)
            image = (255.0 * image).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    return image


def _resize_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    pil = Image.fromarray(image)
    return np.asarray(pil.resize(size, Image.BILINEAR))


def _make_image_grid(images: list[np.ndarray]) -> np.ndarray:
    if len(images) == 1:
        return images[0]

    target_h, target_w = images[0].shape[:2]
    resized = [_resize_image(image, (target_w, target_h)) for image in images]
    cols = math.ceil(math.sqrt(len(resized)))
    rows = math.ceil(len(resized) / cols)
    canvas = np.zeros((rows * target_h, cols * target_w, 3), dtype=np.uint8)

    for idx, image in enumerate(resized):
        row = idx // cols
        col = idx % cols
        canvas[row * target_h : (row + 1) * target_h, col * target_w : (col + 1) * target_w] = image
    return canvas


def _pack_images(images: list[np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.bool_]]:
    if not images:
        raise ValueError("At least one image is required for VQA input packing.")

    parsed = [_parse_image(image) for image in images]
    dummy = np.zeros_like(parsed[0])
    slots = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]

    groups = [[] for _ in range(len(slots))]
    for idx, image in enumerate(parsed):
        groups[idx % len(slots)].append(image)

    packed_images: dict[str, np.ndarray] = {}
    packed_masks: dict[str, np.bool_] = {}
    for slot, group in zip(slots, groups, strict=True):
        if group:
            packed_images[slot] = _make_image_grid(group)
            packed_masks[slot] = np.True_
        else:
            packed_images[slot] = dummy
            packed_masks[slot] = np.False_
    return packed_images, packed_masks


def _normalize_answer(answer) -> str:
    if isinstance(answer, str):
        return answer
    if isinstance(answer, np.ndarray):
        if answer.ndim == 0:
            return str(answer.item())
        answer = answer.tolist()
    if isinstance(answer, dict):
        for key in ("answer", "text", "label"):
            if key in answer:
                return _normalize_answer(answer[key])
        return str(answer)
    if isinstance(answer, (list, tuple)):
        if not answer:
            return ""
        if isinstance(answer[0], dict):
            counts: dict[str, int] = {}
            for item in answer:
                value = _normalize_answer(item)
                counts[value] = counts.get(value, 0) + 1
            return max(counts, key=counts.get)
        return _normalize_answer(answer[0])
    return str(answer)


@dataclasses.dataclass(frozen=True)
class VQAInputs(transforms.DataTransformFn):
    action_dim: int
    action_horizon: int

    def __call__(self, data: dict) -> dict:
        raw_images = data.get("images")
        if raw_images is None:
            raw_images = [data["image"]]
        elif not isinstance(raw_images, (list, tuple)):
            raw_images = [raw_images]

        images, image_masks = _pack_images(list(raw_images))
        return {
            "state": np.zeros((self.action_dim,), dtype=np.float32),
            "actions": np.zeros((self.action_horizon, self.action_dim), dtype=np.float32),
            "image": images,
            "image_mask": image_masks,
            "prompt": data["prompt"],
            "answer": _normalize_answer(data["answer"]),
        }
