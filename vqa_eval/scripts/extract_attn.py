"""Extract last-layer attention heatmaps from the pi05 KI VQA-joint checkpoint.

Reuses ``eval_vqa.py``'s loading / preprocessing / prefix construction; the
ONLY difference is that we install ``attn_patch`` before the forward pass and
read back the captured attention probabilities, then save them as numpy
``.npz`` files alongside per-case metadata. Visualization is done by
``render_attn.py`` so this script stays close to the model and is easy to debug.

Per-case output (``vqa_eval/outputs/<ts>/attn/``)::

    <id>__last.npy        # (256,) attention from the last prompt token to the
                          # 256 base-camera image patches, head-averaged
    <id>__meta.json       # image path, question, prefix lengths, layer count

A single shared ``meta.json`` at the top level captures global info (depth,
num_kv_heads, group_size, image-token layout).

Usage::

    cd /b1k/Jiawei/openpi-comet-clean
    OPENPI_DATA_HOME=/b1k/.cache/openpi \\
        /b1k/hyn/openpi-codebase/.venv/bin/python \\
            vqa_eval/scripts/extract_attn.py \\
                --ckpt outputs/checkpoints/pi05_b1k-ki-vqa-joint-pick-up-from-no-task-planning/pi05_ki_joint_pick_up_from_vqa_no_task_planning_ga1/5000 \\
                --samples vqa_eval/data/attn_samples.jsonl \\
                --out-dir vqa_eval/outputs
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import logging
import pathlib
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import sentencepiece
from PIL import Image

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT / "src", REPO_ROOT / "packages/openpi-client/src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _install_omnigibson_stub() -> None:
    import types

    if "omnigibson.learning.utils.eval_utils" in sys.modules:
        return
    pkg = types.ModuleType("omnigibson")
    pkg.__path__ = []
    learning = types.ModuleType("omnigibson.learning")
    learning.__path__ = []
    utils = types.ModuleType("omnigibson.learning.utils")
    utils.__path__ = []
    eval_utils = types.ModuleType("omnigibson.learning.utils.eval_utils")
    eval_utils.PROPRIOCEPTION_INDICES = {}
    sys.modules.update(
        {
            "omnigibson": pkg,
            "omnigibson.learning": learning,
            "omnigibson.learning.utils": utils,
            "omnigibson.learning.utils.eval_utils": eval_utils,
        }
    )


_install_omnigibson_stub()

# IMPORTANT: install the attention patch BEFORE constructing the model so that
# the very first forward pass (and any future ones) ship probs out via the
# debug callback. The patch is on the linen Module class itself, so it survives
# the nnx_bridge.ToNNX wrapping.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import attn_patch  # noqa: E402

from openpi.models import model as _model  # noqa: E402
from openpi.models.pi0 import make_attn_mask  # noqa: E402
from openpi.policies import vqa_policy  # noqa: E402
from openpi.training import config as _config  # noqa: E402

CONFIG_NAME = "pi05_b1k-knowledge_insulation-vqa-joint-skill-pick-up-from-no-task-planning"
DEFAULT_CKPT = (
    REPO_ROOT
    / "outputs/checkpoints/pi05_b1k-ki-vqa-joint-pick-up-from-no-task-planning"
    / "pi05_ki_joint_pick_up_from_vqa_no_task_planning_ga1/5000"
)
DEFAULT_SAMPLES = REPO_ROOT / "vqa_eval/data/attn_samples.jsonl"
DEFAULT_OUT_DIR = REPO_ROOT / "vqa_eval/outputs"

IMAGE_RESOLUTION = (224, 224)
PATCH_GRID = 16  # 224/14 (SigLIP So400m/14) -> 16x16 = 256 patches per camera
PATCHES_PER_CAMERA = PATCH_GRID * PATCH_GRID  # 256
NUM_CAMERA_SLOTS = 3  # base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb

logger = logging.getLogger("attn_eval")


def _load_image_array(path: pathlib.Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _build_observation(
    image_np: np.ndarray,
    prompt_token_ids: np.ndarray,
    prompt_token_mask: np.ndarray,
    action_dim: int,
) -> _model.Observation:
    packed_images, packed_masks = vqa_policy._pack_images([image_np])
    resized = {
        k: (v if v.shape[:2] == IMAGE_RESOLUTION else vqa_policy._resize_image(v, (IMAGE_RESOLUTION[1], IMAGE_RESOLUTION[0])))
        for k, v in packed_images.items()
    }
    obs_dict = {
        "image": {k: v[None] for k, v in resized.items()},
        "image_mask": {k: np.asarray([bool(packed_masks[k])], dtype=bool) for k in resized},
        "state": np.zeros((1, action_dim), dtype=np.float32),
        "tokenized_prompt": prompt_token_ids[None].astype(np.int32),
        "tokenized_prompt_mask": prompt_token_mask[None].astype(bool),
    }
    return _model.Observation.from_dict(obs_dict)


def _build_prompt(
    question: str,
    sp: sentencepiece.SentencePieceProcessor,
) -> tuple[np.ndarray, np.ndarray]:
    cleaned = question.lower().strip().replace("_", " ")
    prefix_text = f"Task: {cleaned}, State: <no_state>;\n"
    token_ids = sp.encode(prefix_text, add_bos=True)
    return np.asarray(token_ids, dtype=np.int32), np.ones(len(token_ids), dtype=bool)


def _find_object_token_range(
    prompt_ids: np.ndarray,
    sp: sentencepiece.SentencePieceProcessor,
    object_name: str,
) -> tuple[int, int] | None:
    """Locate the contiguous token range in ``prompt_ids`` that encodes
    ``object_name`` (e.g. ``apple`` -> tokens for "▁apple").

    Returns ``(start, end)`` (end exclusive) within the prompt, or ``None`` if
    the object cannot be found. We try a few tokenization variants because
    SentencePiece is sensitive to whether the object word is preceded by
    whitespace or punctuation in the original prompt.
    """
    obj = object_name.strip().lower().replace("_", " ")
    candidates: list[list[int]] = []
    for variant in (" " + obj, obj):
        toks = sp.encode(variant)
        if toks:
            candidates.append(list(toks))

    pid_list = [int(x) for x in prompt_ids.tolist()]
    for cand in candidates:
        L = len(cand)
        for start in range(len(pid_list) - L + 1):
            if pid_list[start : start + L] == cand:
                return start, start + L
    return None


def _run_prefix_forward(model, observation: _model.Observation):
    """Run the prefix-fill forward through PaliGemma.llm.

    The patched ``Attention.__call__`` will fire ``jax.debug.callback`` once per
    layer in scan order, pushing the last query row's softmax probs into
    ``attn_patch._attn_buffer``.
    """
    observation = _model.preprocess_observation(None, observation, train=False)
    observation = dataclasses.replace(
        observation,
        tokenized_prompt=jnp.asarray(observation.tokenized_prompt, dtype=jnp.int32),
        tokenized_prompt_mask=jnp.asarray(observation.tokenized_prompt_mask, dtype=jnp.bool_),
    )
    visual = model._embed_visual_prefix(observation)
    text_emb = model.PaliGemma.llm(observation.tokenized_prompt, method="embed")

    image_tokens = jnp.concatenate(visual[0], axis=1)
    image_mask = jnp.concatenate(visual[1], axis=1)
    image_len = int(image_tokens.shape[1])
    text_len = int(jnp.sum(observation.tokenized_prompt_mask).item())

    full_tokens = jnp.concatenate([image_tokens, text_emb], axis=1)
    full_mask = jnp.concatenate([image_mask, observation.tokenized_prompt_mask], axis=1)
    ar_mask = jnp.zeros(full_tokens.shape[:2], dtype=jnp.bool_)
    attn_mask = make_attn_mask(full_mask, ar_mask)
    positions = jnp.cumsum(full_mask, axis=1) - 1

    (hidden, _), _ = model.PaliGemma.llm(
        [full_tokens, None],
        mask=attn_mask,
        positions=positions,
        adarms_cond=[None, None],
    )
    # Force any pending callbacks to fire by materializing one device value.
    _ = jax.device_get(hidden[..., 0:1])
    return image_len, text_len


def _resolve_image_path(image_field: str) -> pathlib.Path:
    p = pathlib.Path(image_field)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


def _resolve_tokenizer_path(arg_path: str | None) -> pathlib.Path:
    if arg_path:
        p = pathlib.Path(arg_path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"--tokenizer path does not exist: {p}")
        return p
    candidates = [
        pathlib.Path("/b1k/.cache/openpi/big_vision/paligemma_tokenizer.model"),
        pathlib.Path.home() / ".cache/openpi/big_vision/paligemma_tokenizer.model",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "Could not find paligemma_tokenizer.model. Pass --tokenizer or set OPENPI_DATA_HOME."
    )


def _load_samples(samples_path: pathlib.Path) -> list[dict]:
    items: list[dict] = []
    with samples_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=pathlib.Path, default=DEFAULT_CKPT)
    parser.add_argument("--samples", type=pathlib.Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")

    params_dir = args.ckpt / "params"
    if not params_dir.exists():
        raise FileNotFoundError(f"checkpoint params/ dir not found: {params_dir}")

    tokenizer_path = _resolve_tokenizer_path(args.tokenizer)
    logger.info("Loading sentencepiece tokenizer from %s", tokenizer_path)
    with tokenizer_path.open("rb") as f:
        sp = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    # Patch BEFORE the model is constructed/loaded.
    attn_patch.apply_attention_patch()
    logger.info("Attention patch installed.")

    logger.info("Building train config: %s", CONFIG_NAME)
    train_config = _config.get_config(CONFIG_NAME)
    action_dim = train_config.model.action_dim

    logger.info("Restoring params from %s", params_dir)
    t_load = time.monotonic()
    params = _model.restore_params(params_dir, dtype=jnp.bfloat16)
    model = train_config.model.load(params)
    logger.info("Model loaded in %.1fs", time.monotonic() - t_load)

    samples = _load_samples(args.samples)
    if args.limit is not None:
        samples = samples[: args.limit]
    logger.info("Loaded %d samples from %s", len(samples), args.samples)

    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    attn_dir = args.out_dir / timestamp / "attn"
    attn_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Writing attention captures to %s", attn_dir)

    img_cache: dict[str, np.ndarray] = {}
    global_meta: dict | None = None

    for i, sample in enumerate(samples):
        img_path = _resolve_image_path(sample["image"])
        if str(img_path) not in img_cache:
            img_cache[str(img_path)] = _load_image_array(img_path)
        image_np = img_cache[str(img_path)]

        prompt_ids, prompt_mask = _build_prompt(sample["question"], sp)
        obs = _build_observation(image_np, prompt_ids, prompt_mask, action_dim=action_dim)

        attn_patch.reset_attn_buffer()
        t0 = time.monotonic()
        image_len, text_len = _run_prefix_forward(model, obs)
        elapsed = time.monotonic() - t0

        layers = attn_patch.get_attn_layers()
        if not layers:
            raise RuntimeError(
                "No attention probs were captured. Patch may not have been applied "
                "before model construction."
            )

        # layers[k] now has shape (B=1, K, G, n_q, T_k) where
        # n_q = min(prefix_len, attn_patch.N_QUERY_ROWS).
        depth = len(layers)
        b, k_heads, g_heads, n_q, t_k = layers[-1].shape
        assert b == 1, f"unexpected batch size {b}"
        assert t_k == image_len + text_len, (
            f"attention key length {t_k} != image_len {image_len} + text_len {text_len}"
        )
        prefix_len = image_len + text_len
        # The captured rows correspond to ABSOLUTE query positions in
        # [prefix_len - n_q, prefix_len). To map an absolute position p to a
        # row index r in the captured slice: r = p - (prefix_len - n_q).
        row_offset = prefix_len - n_q

        # Head-average + drop batch -> (depth, n_q, T_k).
        stacked = np.stack(layers, axis=0)  # (depth, 1, K, G, n_q, T_k)
        head_avg = stacked[:, 0, :, :, :, :].mean(axis=(1, 2))  # (depth, n_q, T_k)

        # ------- "last query" view (legacy: query = the trailing `\n`) -------
        last_attn_over_keys = head_avg[:, -1, :]  # (depth, T_k)
        last_base_per_layer = last_attn_over_keys[:, :PATCHES_PER_CAMERA].reshape(
            depth, PATCH_GRID, PATCH_GRID
        )

        # ------- "object-name query" view (the real semantic-grounding signal) -------
        # Locate the object's tokens in the prompt. The prompt was built as
        # ``Task: {cleaned}, State: <no_state>;\n`` so absolute positions of
        # the object tokens are: image_len + (offset within prompt_ids).
        obj_name = (sample.get("object") or "").strip()
        if not obj_name:
            cleaned_q = sample["question"].lower().strip().replace("_", " ")
            for prefix in ("pick up the ", "pick up "):
                if cleaned_q.startswith(prefix):
                    obj_name = cleaned_q[len(prefix):].strip()
                    break
            else:
                obj_name = cleaned_q.split()[-1] if cleaned_q else ""

        obj_attn_per_layer = None
        obj_token_range_abs: tuple[int, int] | None = None
        obj_query_capture_status = "ok"
        if obj_name:
            tok_range = _find_object_token_range(prompt_ids, sp, obj_name)
            if tok_range is None:
                obj_query_capture_status = f"object {obj_name!r} not found in prompt tokens"
            else:
                obj_start_in_prompt, obj_end_in_prompt = tok_range
                obj_start_abs = image_len + obj_start_in_prompt
                obj_end_abs = image_len + obj_end_in_prompt
                obj_token_range_abs = (obj_start_abs, obj_end_abs)
                row_lo = obj_start_abs - row_offset
                row_hi = obj_end_abs - row_offset
                if row_lo < 0 or row_hi > n_q:
                    obj_query_capture_status = (
                        f"object query rows [{row_lo}, {row_hi}) out of captured slice "
                        f"[0, {n_q}); increase attn_patch.N_QUERY_ROWS"
                    )
                else:
                    # MAX over object tokens (so multi-token objects like
                    # "cereal box" don't get washed out by averaging).
                    obj_window = head_avg[:, row_lo:row_hi, :]  # (depth, n_obj, T_k)
                    obj_attn_per_layer = obj_window.max(axis=1)  # (depth, T_k)
        else:
            obj_query_capture_status = "no object name available"

        if obj_attn_per_layer is None:
            obj_attn_per_layer = last_attn_over_keys
            logger.warning("[%s] %s -- falling back to trailing-newline query",
                           sample.get("id"), obj_query_capture_status)

        obj_base_per_layer = obj_attn_per_layer[:, :PATCHES_PER_CAMERA].reshape(
            depth, PATCH_GRID, PATCH_GRID
        )

        # Per-layer mass diagnostics for the object-query view.
        per_layer_image_mass = obj_attn_per_layer[:, : NUM_CAMERA_SLOTS * PATCHES_PER_CAMERA].sum(axis=1)
        per_layer_text_mass = obj_attn_per_layer[:, NUM_CAMERA_SLOTS * PATCHES_PER_CAMERA :].sum(axis=1)
        obj_last = obj_attn_per_layer[-1]
        img_mass = float(obj_last[: NUM_CAMERA_SLOTS * PATCHES_PER_CAMERA].sum())
        base_mass = float(obj_last[:PATCHES_PER_CAMERA].sum())
        text_mass = float(obj_last[NUM_CAMERA_SLOTS * PATCHES_PER_CAMERA :].sum())

        case_id = sample.get("id") or f"case_{i:03d}"
        # Primary outputs (object-name query). These are what render_attn.py reads.
        np.save(attn_dir / f"{case_id}__per_layer.npy", obj_base_per_layer)
        np.save(attn_dir / f"{case_id}__per_layer_full.npy", obj_attn_per_layer)
        np.save(attn_dir / f"{case_id}__last.npy", obj_base_per_layer[-1])
        np.save(attn_dir / f"{case_id}__last_full.npy", obj_attn_per_layer[-1])
        # Diagnostic: keep the trailing-newline view too so we can compare.
        np.save(attn_dir / f"{case_id}__newline_per_layer.npy", last_base_per_layer)
        np.save(attn_dir / f"{case_id}__newline_per_layer_full.npy", last_attn_over_keys)

        case_meta = {
            "id": case_id,
            "image": str(img_path),
            "question": sample["question"],
            "object": obj_name or None,
            "object_token_range_abs": list(obj_token_range_abs) if obj_token_range_abs else None,
            "object_query_capture_status": obj_query_capture_status,
            "query_kind": "object_token_max" if obj_token_range_abs else "trailing_newline",
            "note": sample.get("note"),
            "image_len": image_len,
            "text_len": text_len,
            "prefix_len": image_len + text_len,
            "depth": depth,
            "num_kv_heads": int(k_heads),
            "group_size": int(g_heads),
            "n_query_rows_captured": int(n_q),
            "patch_grid": PATCH_GRID,
            "patches_per_camera": PATCHES_PER_CAMERA,
            "num_camera_slots": NUM_CAMERA_SLOTS,
            "attention_mass": {
                "all_cameras": img_mass,
                "base_camera": base_mass,
                "text_tokens": text_mass,
            },
            "per_layer_attention_mass": {
                "image": [float(x) for x in per_layer_image_mass],
                "text": [float(x) for x in per_layer_text_mass],
            },
            "elapsed_s": elapsed,
        }
        with (attn_dir / f"{case_id}__meta.json").open("w", encoding="utf-8") as f:
            json.dump(case_meta, f, ensure_ascii=False, indent=2)

        logger.info(
            "[%d/%d] %s | Q=%r | depth=%d K=%d G=%d | base_mass=%.3f text_mass=%.3f (%.2fs)",
            i + 1, len(samples), case_id, sample["question"],
            depth, k_heads, g_heads, base_mass, text_mass, elapsed,
        )

        if global_meta is None:
            global_meta = {
                "ckpt": str(args.ckpt),
                "config": CONFIG_NAME,
                "depth": depth,
                "num_kv_heads": int(k_heads),
                "group_size": int(g_heads),
                "image_resolution": list(IMAGE_RESOLUTION),
                "patch_grid": PATCH_GRID,
                "patches_per_camera": PATCHES_PER_CAMERA,
                "num_camera_slots": NUM_CAMERA_SLOTS,
                "patched": "openpi.models.gemma.Attention.__call__",
            }

    if global_meta is not None:
        with (attn_dir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(global_meta, f, ensure_ascii=False, indent=2)

    logger.info("Done. Attention captures in %s", attn_dir)


if __name__ == "__main__":
    main()
