"""Run VQA inference on the pi05 KI VQA-joint checkpoint.

Loads the pi05 checkpoint manually (bypasses ``create_trained_policy`` because
the training config references ``/vepfs-C/...`` paths that don't exist on the
inference host), reuses ``VQAInputs`` packing for image slots, builds a FAST-style
prefix (``Task: ..., State: ...;\\nAnswer:``) and greedily samples the answer
tokens one-by-one through ``Pi0.PaliGemma.llm`` while reusing the KV cache.

Outputs go to ``vqa_eval/outputs/<timestamp>/``:
- ``results.jsonl`` -- one record per (image, question) with the generated answer
- ``report.md`` -- markdown report with embedded image references

Usage::

    cd /b1k/Jiawei/openpi-comet-clean
    OPENPI_DATA_HOME=/b1k/.cache/openpi \\
        python vqa_eval/scripts/eval_vqa.py \\
            --ckpt outputs/checkpoints/pi05_b1k-ki-vqa-joint-pick-up-from-no-task-planning/pi05_ki_joint_pick_up_from_vqa_no_task_planning_ga1/5000 \\
            --samples vqa_eval/data/samples.jsonl \\
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
    pkg.__path__ = []  # mark as package
    learning = types.ModuleType("omnigibson.learning")
    learning.__path__ = []
    utils = types.ModuleType("omnigibson.learning.utils")
    utils.__path__ = []
    eval_utils = types.ModuleType("omnigibson.learning.utils.eval_utils")
    eval_utils.PROPRIOCEPTION_INDICES = {}  # not used during VQA inference
    sys.modules.update(
        {
            "omnigibson": pkg,
            "omnigibson.learning": learning,
            "omnigibson.learning.utils": utils,
            "omnigibson.learning.utils.eval_utils": eval_utils,
        }
    )


_install_omnigibson_stub()

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
DEFAULT_SAMPLES = REPO_ROOT / "vqa_eval/data/samples.jsonl"
DEFAULT_OUT_DIR = REPO_ROOT / "vqa_eval/outputs"

IMAGE_RESOLUTION = (224, 224)

logger = logging.getLogger("vqa_eval")


def _load_image_array(path: pathlib.Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _build_observation(
    image_np: np.ndarray,
    prompt_token_ids: np.ndarray,
    prompt_token_mask: np.ndarray,
    action_dim: int,
) -> _model.Observation:
    """Pack a single image + prompt into a batch-1 Observation."""
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
    *,
    action_dim: int,  # noqa: ARG001  # kept for API symmetry; unused for VQA
) -> tuple[np.ndarray, np.ndarray]:
    """Build VQA prefix tokens.

    Matches ``FASTTokenizer._tokenize_prefix`` for VQA samples:
    ``transforms.TokenizeFASTInputs`` forces ``state=None`` whenever ``answer``
    is present, which produces ``"Task: <q>, State: <no_state>;\\n"``.
    The answer (``Answer: <text>`` + EOS) is the supervised postfix that the
    model is expected to generate -- we MUST NOT prepend ``Answer:`` here.
    """
    cleaned = question.lower().strip().replace("_", " ")
    prefix_text = f"Task: {cleaned}, State: <no_state>;\n"
    token_ids = sp.encode(prefix_text, add_bos=True)
    return np.asarray(token_ids, dtype=np.int32), np.ones(len(token_ids), dtype=bool)


def _vqa_generate(
    model,
    observation: _model.Observation,
    sp: sentencepiece.SentencePieceProcessor,
    *,
    max_new_tokens: int,
) -> tuple[list[int], str]:
    """Greedy autoregressive VQA decoding.

    Returns the list of generated token ids (excluding the EOS) and the decoded
    answer string. Decoding stops at the SentencePiece EOS id or when
    ``max_new_tokens`` is reached.
    """
    eos_id = sp.eos_id()

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

    (hidden, _), kv_cache = model.PaliGemma.llm(
        [full_tokens, None],
        mask=attn_mask,
        positions=positions,
        adarms_cond=[None, None],
    )

    last_text_idx = image_len + text_len - 1
    last_hidden = hidden[:, last_text_idx : last_text_idx + 1]
    logits = model.PaliGemma.llm(last_hidden, method="decode_logits")
    next_token = int(jnp.argmax(logits[0, 0]).item())

    generated: list[int] = []
    if next_token == eos_id:
        return generated, ""
    generated.append(next_token)

    next_position = int(jnp.sum(full_mask).item())  # absolute RoPE pos for the new token

    for _ in range(max_new_tokens - 1):
        token_arr = jnp.asarray([[next_token]], dtype=jnp.int32)
        emb = model.PaliGemma.llm(token_arr, method="embed")

        gen_mask = jnp.ones((1, len(generated)), dtype=jnp.bool_)
        full_mask_1d = jnp.concatenate([full_mask, gen_mask], axis=1)
        new_mask = full_mask_1d[:, None, :]
        new_positions = jnp.asarray([[next_position]], dtype=jnp.int32)

        (hidden, _), kv_cache = model.PaliGemma.llm(
            [emb, None],
            mask=new_mask,
            positions=new_positions,
            kv_cache=kv_cache,
            adarms_cond=[None, None],
        )
        logits = model.PaliGemma.llm(hidden, method="decode_logits")
        next_token = int(jnp.argmax(logits[0, 0]).item())
        if next_token == eos_id:
            break
        generated.append(next_token)
        next_position += 1

    answer = sp.decode(generated).strip()
    return generated, answer


def _load_samples(samples_path: pathlib.Path) -> list[dict]:
    items: list[dict] = []
    with samples_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _resolve_image_path(image_field: str) -> pathlib.Path:
    p = pathlib.Path(image_field)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


def _write_markdown(out_dir: pathlib.Path, results: list[dict]) -> None:
    lines: list[str] = ["# pi05 KI VQA evaluation\n"]
    by_image: dict[str, list[dict]] = {}
    for r in results:
        by_image.setdefault(r["image"], []).append(r)
    for image, group in by_image.items():
        rel = pathlib.Path(image)
        try:
            rel = rel.relative_to(REPO_ROOT)
        except ValueError:
            pass
        lines.append(f"## `{rel}`\n")
        lines.append(f"![]({pathlib.Path('..') / pathlib.Path('..') / rel})\n")
        for r in group:
            lines.append(f"- **Q:** {r['question']}")
            lines.append(f"  - **A:** {r['answer']}")
            if r.get("note"):
                lines.append(f"  - _note:_ {r['note']}")
        lines.append("")
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


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
        "Could not find paligemma_tokenizer.model locally. Pass --tokenizer or set "
        "OPENPI_DATA_HOME to a directory containing big_vision/paligemma_tokenizer.model."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=pathlib.Path, default=DEFAULT_CKPT, help="Checkpoint dir containing params/")
    parser.add_argument("--samples", type=pathlib.Path, default=DEFAULT_SAMPLES, help="JSONL with image+question pairs")
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR, help="Where results go")
    parser.add_argument("--tokenizer", type=str, default=None, help="Optional explicit path to paligemma_tokenizer.model")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of samples to run")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")

    params_dir = args.ckpt / "params"
    if not params_dir.exists():
        raise FileNotFoundError(f"checkpoint params/ dir not found: {params_dir}")

    tokenizer_path = _resolve_tokenizer_path(args.tokenizer)
    logger.info("Loading sentencepiece tokenizer from %s", tokenizer_path)
    with tokenizer_path.open("rb") as f:
        sp = sentencepiece.SentencePieceProcessor(model_proto=f.read())

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
    out_dir = args.out_dir / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Writing results to %s", out_dir)

    results: list[dict] = []
    img_cache: dict[str, np.ndarray] = {}
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as fout:
        for i, sample in enumerate(samples):
            img_path = _resolve_image_path(sample["image"])
            if str(img_path) not in img_cache:
                img_cache[str(img_path)] = _load_image_array(img_path)
            image_np = img_cache[str(img_path)]

            prompt_ids, prompt_mask = _build_prompt(
                sample["question"], sp, action_dim=action_dim
            )
            obs = _build_observation(image_np, prompt_ids, prompt_mask, action_dim=action_dim)

            t0 = time.monotonic()
            _, answer = _vqa_generate(model, obs, sp, max_new_tokens=args.max_new_tokens)
            elapsed = time.monotonic() - t0
            logger.info(
                "[%d/%d] %s | Q=%r | A=%r (%.2fs)",
                i + 1, len(samples), sample.get("id", img_path.name), sample["question"], answer, elapsed,
            )

            record = {
                "id": sample.get("id"),
                "image": str(img_path),
                "question": sample["question"],
                "answer": answer,
                "note": sample.get("note"),
                "elapsed_s": elapsed,
                "prompt_token_len": int(prompt_ids.size),
            }
            results.append(record)
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

    _write_markdown(out_dir, results)
    logger.info("Done. Results: %s", out_dir / "results.jsonl")
    logger.info("Report: %s", out_dir / "report.md")


if __name__ == "__main__":
    main()
