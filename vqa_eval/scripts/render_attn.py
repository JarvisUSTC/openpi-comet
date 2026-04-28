"""Render attention heatmaps captured by ``extract_attn.py``.

For each captured case we now produce four families of outputs:

1. ``<id>__triptych.png``      original | last-layer heatmap | overlay
2. ``<id>__per_layer.png``     grid (3x6) of all 18 layers' base-camera
                               attention, each individually min-max normalized
3. ``<id>__agg_<name>.png``    aggregate triptychs:
                                 - ``mean_all``    raw mean across all layers
                                 - ``mean_mid``    raw mean across layers 4..12
                                 - ``norm_mean``   per-layer min-max normalized,
                                                   then mean across all layers
                                 - ``rolloutlite`` cumulative product of
                                                   ``0.5*A_layer + 0.5*I``
                                                   evaluated only over the 16x16
                                                   image-patch axis (cheap
                                                   approximation of attention
                                                   rollout that doesn't require
                                                   the full 800x800 matrices)
4. ``summary.md``              per-image grouping with all paired prompts

Usage::

    cd /b1k/Jiawei/openpi-comet-clean
    /b1k/hyn/openpi-codebase/.venv/bin/python \\
        vqa_eval/scripts/render_attn.py \\
            --attn-dir vqa_eval/outputs/<timestamp>/attn
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from PIL import Image

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
TARGET_RESOLUTION = (224, 224)
MIDDLE_LAYER_RANGE = (4, 13)  # inclusive-exclusive: layers 4..12

logger = logging.getLogger("attn_render")


def _load_image_resized(path: pathlib.Path, size: tuple[int, int]) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize(size, Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def _normalize(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def _upsample(attn_2d: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Bilinearly upsample a 2D attention map to ``size``.

    Input is expected to be already normalized to [0, 1].
    """
    arr = np.clip(attn_2d, 0.0, 1.0)
    img8 = (arr * 255.0).astype(np.uint8)
    return np.asarray(Image.fromarray(img8).resize(size, Image.BILINEAR), dtype=np.float32) / 255.0


def _heatmap_rgba(attn_up: np.ndarray, cmap_name: str = "jet") -> np.ndarray:
    cmap = colormaps[cmap_name]
    return (cmap(attn_up) * 255.0).astype(np.uint8)


def _overlay(image_rgb: np.ndarray, attn_up: np.ndarray, alpha: float = 0.5,
             cmap_name: str = "jet") -> np.ndarray:
    rgba = _heatmap_rgba(attn_up, cmap_name=cmap_name)
    rgb = rgba[..., :3].astype(np.float32) / 255.0
    base = image_rgb.astype(np.float32) / 255.0
    weight = alpha * attn_up[..., None]
    blended = np.clip(base * (1.0 - weight) + rgb * weight, 0.0, 1.0)
    return (blended * 255.0).astype(np.uint8)


def _save_triptych(out_path: pathlib.Path, image_rgb: np.ndarray,
                   attn_2d: np.ndarray, title: str, subtitle: str = "") -> None:
    attn_norm = _normalize(attn_2d)
    attn_up = _upsample(attn_norm, TARGET_RESOLUTION)
    overlay = _overlay(image_rgb, attn_up, alpha=0.5)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    axes[0].imshow(image_rgb)
    axes[0].set_title("input (resized)")
    axes[0].axis("off")
    im = axes[1].imshow(attn_up, cmap="jet", vmin=0.0, vmax=1.0)
    axes[1].set_title("attention (16x16 -> 224x224, min-max norm)")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    axes[2].imshow(overlay)
    axes[2].set_title("overlay (alpha=0.5)")
    axes[2].axis("off")
    full_title = f"{title}\n{subtitle}" if subtitle else title
    fig.suptitle(full_title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _save_per_layer_grid(out_path: pathlib.Path, image_rgb: np.ndarray,
                         per_layer: np.ndarray, title: str) -> None:
    """Render a 4x5 (or sized to depth) grid of per-layer attention maps.

    Each layer is min-max normalized independently so we can see WHERE each
    layer is focusing rather than the absolute attention mass (which collapses
    onto text in the deep layers).
    """
    depth = per_layer.shape[0]
    cols = 6
    rows = int(np.ceil((depth + 1) / cols))  # +1 cell for the original image
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.4 * rows))
    axes = np.atleast_2d(axes)

    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title("input", fontsize=9)
    axes[0, 0].axis("off")

    for layer_idx in range(depth):
        slot = layer_idx + 1
        r, c = divmod(slot, cols)
        attn_norm = _normalize(per_layer[layer_idx])
        attn_up = _upsample(attn_norm, TARGET_RESOLUTION)
        overlay = _overlay(image_rgb, attn_up, alpha=0.55)
        axes[r, c].imshow(overlay)
        axes[r, c].set_title(f"layer {layer_idx}", fontsize=9)
        axes[r, c].axis("off")

    for slot in range(depth + 1, rows * cols):
        r, c = divmod(slot, cols)
        axes[r, c].axis("off")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _aggregate_views(per_layer: np.ndarray) -> dict[str, np.ndarray]:
    """Compute several layer-aggregate attention maps.

    All outputs are 2D (16x16). Min-max normalization is applied INSIDE the
    layer-aggregating step where it makes sense; further normalization for
    visualization happens in ``_save_triptych``.
    """
    depth = per_layer.shape[0]

    mean_all = per_layer.mean(axis=0)

    lo, hi = MIDDLE_LAYER_RANGE
    mid_slice = per_layer[lo:hi]
    mean_mid = mid_slice.mean(axis=0) if mid_slice.size > 0 else mean_all

    # Per-layer normalize (so deep layers with tiny absolute mass get equal
    # voting weight as shallow layers), then mean.
    layer_max = per_layer.reshape(depth, -1).max(axis=1).reshape(depth, 1, 1)
    layer_max = np.maximum(layer_max, 1e-12)
    norm_per_layer = per_layer / layer_max
    norm_mean = norm_per_layer.mean(axis=0)

    # "Rollout-lite": cumulative product of (0.5*A + 0.5*I) restricted to the
    # 16x16 image patch axis. This is NOT the strict Abnar & Zuidema rollout
    # (which requires full TxT matrices); it's a cheap surrogate that still
    # spreads activation across layers and dampens stuck-anchor patches.
    flat = per_layer.reshape(depth, -1)  # (depth, 256)
    flat_norm = flat / np.maximum(flat.sum(axis=1, keepdims=True), 1e-12)
    n = flat.shape[1]
    eye = np.ones(n) / n
    cumulative = flat_norm[0] + 0.5 * eye
    cumulative /= cumulative.sum()
    for layer_idx in range(1, depth):
        mixed = 0.5 * flat_norm[layer_idx] + 0.5 * eye
        cumulative = cumulative * mixed
        cumulative /= max(cumulative.sum(), 1e-12)
    rollout_lite = cumulative.reshape(per_layer.shape[1:])

    return {
        "mean_all": mean_all,
        "mean_mid": mean_mid,
        "norm_mean": norm_mean,
        "rolloutlite": rollout_lite,
    }


def _resolve_image_path(image_field: str) -> pathlib.Path:
    p = pathlib.Path(image_field)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


def _write_summary(attn_dir: pathlib.Path, cases: list[dict]) -> None:
    by_image: dict[str, list[dict]] = {}
    for c in cases:
        by_image.setdefault(c["image"], []).append(c)

    lines = ["# Attention heatmap summary",
             "",
             f"Aggregations: ``mean_all``, ``mean_mid`` (layers {MIDDLE_LAYER_RANGE[0]}..{MIDDLE_LAYER_RANGE[1] - 1}), "
             "``norm_mean`` (per-layer normalized then averaged), ``rolloutlite``.",
             ""]
    for image, group in by_image.items():
        rel = pathlib.Path(image)
        try:
            rel = rel.relative_to(REPO_ROOT)
        except ValueError:
            pass
        lines.append(f"## `{rel}`")
        lines.append("")
        for c in group:
            lines.append(f"### {c['id']} -- Q: {c['question']}")
            if c.get("note"):
                lines.append(f"_note:_ {c['note']}")
            lines.append("")
            mass = c.get("attention_mass", {})
            lines.append(
                f"- prefix_len={c['prefix_len']} (image={c['image_len']}, text={c['text_len']}) "
                f"| depth={c['depth']}, K={c['num_kv_heads']}, G={c['group_size']}"
            )
            lines.append(
                f"- last-layer mass: base_camera={mass.get('base_camera', 0.0):.3f}, "
                f"text_tokens={mass.get('text_tokens', 0.0):.3f}"
            )
            lines.append("")
            lines.append(f"![per_layer]({c['id']}__per_layer.png)")
            lines.append("")
            for name in ("mean_all", "mean_mid", "norm_mean", "rolloutlite"):
                lines.append(f"![{name}]({c['id']}__agg_{name}.png)")
            lines.append("")
            lines.append(f"![last]({c['id']}__triptych.png)")
            lines.append("")
        lines.append("")
    (attn_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attn-dir", type=pathlib.Path, required=True,
                        help="Directory containing <id>__per_layer.npy and <id>__meta.json")
    parser.add_argument("--cmap", type=str, default="jet")
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")

    attn_dir = args.attn_dir
    if not attn_dir.is_dir():
        raise NotADirectoryError(attn_dir)

    metas = sorted(p for p in attn_dir.glob("*__meta.json") if p.name != "meta.json")
    if not metas:
        raise FileNotFoundError(f"No <id>__meta.json files in {attn_dir}")

    cases: list[dict] = []
    for meta_path in metas:
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        case_id = meta["id"]
        per_layer_path = attn_dir / f"{case_id}__per_layer.npy"
        last_path = attn_dir / f"{case_id}__last.npy"
        if not per_layer_path.exists():
            logger.warning("Missing %s; falling back to last-layer only", per_layer_path)
            if not last_path.exists():
                continue
            per_layer = np.load(last_path)[None, ...]
        else:
            per_layer = np.load(per_layer_path)

        img_path = _resolve_image_path(meta["image"])
        image_rgb = _load_image_resized(img_path, TARGET_RESOLUTION)

        title = f"{case_id} -- Q: {meta['question']}"

        _save_triptych(
            attn_dir / f"{case_id}__triptych.png",
            image_rgb, per_layer[-1], title,
            subtitle=f"layer {per_layer.shape[0] - 1} (last)",
        )
        _save_per_layer_grid(
            attn_dir / f"{case_id}__per_layer.png",
            image_rgb, per_layer, title,
        )

        agg = _aggregate_views(per_layer)
        for name, arr in agg.items():
            _save_triptych(
                attn_dir / f"{case_id}__agg_{name}.png",
                image_rgb, arr, title, subtitle=f"aggregate: {name}",
            )

        cases.append(meta)
        logger.info("Rendered %s (Q=%r) | depth=%d", case_id, meta["question"], per_layer.shape[0])

    _write_summary(attn_dir, cases)
    logger.info("Done. Summary: %s", attn_dir / "summary.md")


if __name__ == "__main__":
    main()
