#!/usr/bin/env python3
"""Verify that denoise_step() produces the same velocity as forward().

The training path uses PaliGemmaWithExpertModel.forward(inputs_embeds=[prefix, suffix])
with joint attention. The inference path (sample_actions -> denoise_step) caches prefix
KV from paligemma, then manually runs expert layers with that cache. This test checks
that both paths produce identical velocity output for the same input.

Usage:
    cd /root/Training/memory0.1
    uv run python scripts/test_denoise_consistency.py
"""
import os
import sys

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, ".venv", "lib", "python3.11", "site-packages"))

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import shutil
import glob as _glob

replace_src = os.path.join(REPO_ROOT, "src", "openpi", "models_pytorch", "transformers_replace")
site_transformers = os.path.join(REPO_ROOT, ".venv", "lib", "python3.11", "site-packages", "transformers")
for src_file in _glob.glob(os.path.join(replace_src, "**"), recursive=True):
    if os.path.isfile(src_file):
        rel = os.path.relpath(src_file, replace_src)
        dst = os.path.join(site_transformers, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src_file, dst)

import torch
import numpy as np
from openpi.models import pi0_config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks
from openpi.models.model import Observation

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}  {detail}")


def make_dummy_observation(config, batch_size=1, device="cpu"):
    """Create a minimal dummy observation for testing."""
    num_frames = getattr(config, "video_memory_frames", 1)
    img_shape = (batch_size, 3, 224, 224)
    state_dim = config.action_dim

    images = {
        "base_0_rgb": torch.randn(*img_shape, device=device),
        "left_wrist_0_rgb": torch.randn(*img_shape, device=device),
        "right_wrist_0_rgb": torch.randn(*img_shape, device=device),
    }
    image_masks = {
        "base_0_rgb": torch.ones(batch_size, dtype=torch.bool, device=device),
        "left_wrist_0_rgb": torch.ones(batch_size, dtype=torch.bool, device=device),
        "right_wrist_0_rgb": torch.ones(batch_size, dtype=torch.bool, device=device),
    }

    max_token_len = config.max_token_len
    tokenized_prompt = torch.randint(0, 1000, (batch_size, max_token_len), device=device)
    tokenized_prompt_mask = torch.ones(batch_size, max_token_len, dtype=torch.bool, device=device)
    state = torch.randn(batch_size, state_dim, device=device, dtype=torch.float32)

    return Observation(
        images=images,
        image_masks=image_masks,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
        state=state,
    )


def test_denoise_vs_forward_consistency(device="cpu"):
    """Core test: denoise_step velocity must match forward velocity for the same input."""
    print(f"\n== Test: denoise_step vs forward consistency (device={device}) ==")

    config = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=32,
        video_memory_frames=1,
        dtype="float32",
    )

    torch.manual_seed(42)
    model = PI0Pytorch(config)
    model.to(device)
    model.eval()

    B = 1
    torch.manual_seed(123)
    observation = make_dummy_observation(config, batch_size=B, device=device)

    actions = torch.randn(B, config.action_horizon, config.action_dim, device=device)
    timestep_val = 0.5
    time = torch.full((B,), timestep_val, dtype=torch.float32, device=device)

    noise = torch.randn_like(actions)
    time_expanded = time[:, None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    u_t = noise - actions

    # --- Path A: forward() (training path) ---
    images, img_masks, lang_tokens, lang_masks, state, temporal_mask = model._preprocess_observation(
        observation, train=False
    )
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, temporal_mask=temporal_mask
    )
    suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = model.embed_suffix(state, x_t, time)

    if model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
        suffix_embs_fwd = suffix_embs.clone().to(dtype=torch.bfloat16)
        prefix_embs_fwd = prefix_embs.clone().to(dtype=torch.bfloat16)
    else:
        suffix_embs_fwd = suffix_embs.clone()
        prefix_embs_fwd = prefix_embs.clone()

    pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
    att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
    att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
    position_ids = torch.cumsum(pad_masks, dim=1) - 1
    att_2d_masks_4d = model._prepare_attention_masks_4d(att_2d_masks)

    with torch.no_grad():
        (_, suffix_out_fwd), _ = model.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs_fwd, suffix_embs_fwd],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
    suffix_out_fwd = suffix_out_fwd[:, -config.action_horizon:]
    v_t_forward = model.action_out_proj(suffix_out_fwd.to(dtype=torch.float32))

    # --- Path B: sample_actions / denoise_step (inference path) ---
    # Use joint forward with use_cache=True to get prefix KV that matches training exactly
    with torch.no_grad():
        (_, _), past_key_values = model.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs_fwd.clone(), suffix_embs_fwd.clone()],
            use_cache=True,
            adarms_cond=[None, adarms_cond],
        )

    with torch.no_grad():
        v_t_denoise = model.denoise_step(
            past_key_values=past_key_values,
            prefix_pad_masks=prefix_pad_masks,
            state=state,
            x_t=x_t,
            timestep=time,
        )

    # --- Compare ---
    v_fwd = v_t_forward.detach().float()
    v_den = v_t_denoise.detach().float()

    abs_diff = (v_fwd - v_den).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    rel_diff = (abs_diff / (v_fwd.abs() + 1e-8)).mean().item()

    cos_sim = torch.nn.functional.cosine_similarity(
        v_fwd.reshape(1, -1), v_den.reshape(1, -1)
    ).item()

    print(f"    max_abs_diff:  {max_diff:.6e}")
    print(f"    mean_abs_diff: {mean_diff:.6e}")
    print(f"    mean_rel_diff: {rel_diff:.6e}")
    print(f"    cosine_sim:    {cos_sim:.6f}")
    print(f"    v_fwd norm:    {v_fwd.norm().item():.4f}")
    print(f"    v_den norm:    {v_den.norm().item():.4f}")

    check("cosine similarity > 0.999", cos_sim > 0.999, f"cos_sim={cos_sim:.6f}")
    check("max abs diff < 0.01", max_diff < 0.01, f"max_diff={max_diff:.6e}")
    check("mean abs diff < 0.001", mean_diff < 0.001, f"mean_diff={mean_diff:.6e}")
    check("output norms are similar", abs(v_fwd.norm().item() - v_den.norm().item()) / (v_fwd.norm().item() + 1e-8) < 0.01,
          f"fwd={v_fwd.norm().item():.4f} den={v_den.norm().item():.4f}")


def test_denoise_vs_forward_bfloat16():
    """Same test in bfloat16 (actual training precision) on GPU."""
    if not torch.cuda.is_available():
        print("\n== Test: bfloat16 consistency (SKIPPED - no CUDA) ==")
        return

    print("\n== Test: denoise_step vs forward consistency (bfloat16, cuda) ==")

    config = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=32,
        video_memory_frames=1,
        dtype="bfloat16",
    )

    device = "cuda"
    torch.manual_seed(42)
    model = PI0Pytorch(config)
    model.to(device)
    model.eval()

    B = 2
    torch.manual_seed(789)
    observation = make_dummy_observation(config, batch_size=B, device=device)

    actions = torch.randn(B, config.action_horizon, config.action_dim, device=device)
    time = torch.full((B,), 0.7, dtype=torch.float32, device=device)

    noise = torch.randn_like(actions)
    time_expanded = time[:, None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions

    images, img_masks, lang_tokens, lang_masks, state, temporal_mask = model._preprocess_observation(
        observation, train=False
    )
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, temporal_mask=temporal_mask
    )
    suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = model.embed_suffix(state, x_t, time)

    dtype = model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
    suffix_embs_fwd = suffix_embs.clone().to(dtype=dtype)
    prefix_embs_fwd = prefix_embs.clone().to(dtype=dtype)

    pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
    att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
    att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
    position_ids = torch.cumsum(pad_masks, dim=1) - 1
    att_2d_masks_4d = model._prepare_attention_masks_4d(att_2d_masks)

    with torch.no_grad():
        (_, suffix_out_fwd), _ = model.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs_fwd, suffix_embs_fwd],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
    suffix_out_fwd = suffix_out_fwd[:, -config.action_horizon:]
    v_t_forward = model.action_out_proj(suffix_out_fwd.to(dtype=torch.float32))

    with torch.no_grad():
        (_, _), past_key_values = model.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs_fwd.clone(), suffix_embs_fwd.clone()],
            use_cache=True,
            adarms_cond=[None, adarms_cond],
        )

    with torch.no_grad():
        v_t_denoise = model.denoise_step(
            past_key_values=past_key_values,
            prefix_pad_masks=prefix_pad_masks,
            state=state,
            x_t=x_t,
            timestep=time,
        )

    v_fwd = v_t_forward.detach().float()
    v_den = v_t_denoise.detach().float()

    abs_diff = (v_fwd - v_den).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        v_fwd.reshape(1, -1), v_den.reshape(1, -1)
    ).item()

    print(f"    max_abs_diff:  {max_diff:.6e}")
    print(f"    mean_abs_diff: {mean_diff:.6e}")
    print(f"    cosine_sim:    {cos_sim:.6f}")
    print(f"    v_fwd norm:    {v_fwd.norm().item():.4f}")
    print(f"    v_den norm:    {v_den.norm().item():.4f}")

    # bfloat16 has lower precision, so relax thresholds
    check("bf16 cosine similarity > 0.99", cos_sim > 0.99, f"cos_sim={cos_sim:.6f}")
    check("bf16 max abs diff < 0.1", max_diff < 0.1, f"max_diff={max_diff:.6e}")
    check("bf16 mean abs diff < 0.01", mean_diff < 0.01, f"mean_diff={mean_diff:.6e}")


def test_multiple_timesteps():
    """Verify consistency holds across different timestep values."""
    if not torch.cuda.is_available():
        print("\n== Test: multi-timestep (SKIPPED - no CUDA) ==")
        return

    print("\n== Test: consistency across multiple timesteps ==")

    config = pi0_config.Pi0Config(pi05=True, action_horizon=32, video_memory_frames=1, dtype="bfloat16")
    device = "cuda"

    torch.manual_seed(42)
    model = PI0Pytorch(config)
    model.to(device)
    model.eval()

    B = 1
    torch.manual_seed(99)
    observation = make_dummy_observation(config, batch_size=B, device=device)
    actions = torch.randn(B, config.action_horizon, config.action_dim, device=device)
    noise = torch.randn_like(actions)

    images, img_masks, lang_tokens, lang_masks, state, temporal_mask = model._preprocess_observation(
        observation, train=False
    )
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, temporal_mask=temporal_mask
    )

    dtype = model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
    with torch.no_grad():
        # Use first timestep's suffix for the joint forward to get prefix KV
        first_time = torch.full((B,), 0.1, dtype=torch.float32, device=device)
        first_x_t = first_time[:, None, None] * noise + (1 - first_time[:, None, None]) * actions
        first_suffix, first_spad, first_satt, first_adarms = model.embed_suffix(state, first_x_t, first_time)

        first_pad = torch.cat([prefix_pad_masks, first_spad], dim=1)
        first_att = torch.cat([prefix_att_masks, first_satt], dim=1)
        first_att_2d = make_att_2d_masks(first_pad, first_att)
        first_pos = torch.cumsum(first_pad, dim=1) - 1
        first_att_4d = model._prepare_attention_masks_4d(first_att_2d)

        (_, _), past_key_values = model.paligemma_with_expert.forward(
            attention_mask=first_att_4d,
            position_ids=first_pos,
            past_key_values=None,
            inputs_embeds=[prefix_embs.clone().to(dtype=dtype), first_suffix.to(dtype=dtype)],
            use_cache=True,
            adarms_cond=[None, first_adarms],
        )

    all_pass = True
    for t_val in [0.1, 0.3, 0.5, 0.7, 0.9]:
        time = torch.full((B,), t_val, dtype=torch.float32, device=device)
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions

        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = model.embed_suffix(state, x_t, time)
        suffix_embs_fwd = suffix_embs.clone().to(dtype=dtype)
        prefix_embs_fwd = prefix_embs.clone().to(dtype=dtype)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = model._prepare_attention_masks_4d(att_2d_masks)

        with torch.no_grad():
            (_, suffix_out_fwd), _ = model.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs_fwd, suffix_embs_fwd],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
        suffix_out_fwd = suffix_out_fwd[:, -config.action_horizon:]
        v_fwd = model.action_out_proj(suffix_out_fwd.to(dtype=torch.float32))

        with torch.no_grad():
            v_den = model.denoise_step(
                past_key_values=past_key_values,
                prefix_pad_masks=prefix_pad_masks,
                state=state,
                x_t=x_t,
                timestep=time,
            )

        cos_sim = torch.nn.functional.cosine_similarity(
            v_fwd.reshape(1, -1).float(), v_den.reshape(1, -1).float()
        ).item()
        max_diff = (v_fwd.float() - v_den.float()).abs().max().item()
        status = "✅" if cos_sim > 0.99 else "❌"
        if cos_sim <= 0.99:
            all_pass = False
        print(f"    t={t_val:.1f}  cos_sim={cos_sim:.6f}  max_diff={max_diff:.6e}  {status}")

    check("all timesteps consistent (cos > 0.99)", all_pass)


if __name__ == "__main__":
    print("=" * 60)
    print("Denoise Step vs Forward Consistency Test")
    print("=" * 60)

    test_denoise_vs_forward_consistency(device="cpu")
    test_denoise_vs_forward_bfloat16()
    test_multiple_timesteps()

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed")
    print("=" * 60)
    sys.exit(1 if FAIL > 0 else 0)
