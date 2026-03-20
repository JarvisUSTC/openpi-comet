#!/usr/bin/env python3
"""Layer-by-layer debug: find where forward vs denoise_step diverge."""
import os, sys
REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, ".venv", "lib", "python3.11", "site-packages"))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import shutil, glob as _glob
replace_src = os.path.join(REPO_ROOT, "src", "openpi", "models_pytorch", "transformers_replace")
site_transformers = os.path.join(REPO_ROOT, ".venv", "lib", "python3.11", "site-packages", "transformers")
for f in _glob.glob(os.path.join(replace_src, "**"), recursive=True):
    if os.path.isfile(f):
        rel = os.path.relpath(f, replace_src)
        dst = os.path.join(site_transformers, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(f, dst)

import torch, math
from transformers.models.gemma import modeling_gemma
from openpi.models import pi0_config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks
from openpi.models.model import Observation

def make_obs(config, B=1, device="cpu"):
    return Observation(
        images={
            "base_0_rgb": torch.randn(B, 3, 224, 224, device=device),
            "left_wrist_0_rgb": torch.randn(B, 3, 224, 224, device=device),
            "right_wrist_0_rgb": torch.randn(B, 3, 224, 224, device=device),
        },
        image_masks={
            "base_0_rgb": torch.ones(B, dtype=torch.bool, device=device),
            "left_wrist_0_rgb": torch.ones(B, dtype=torch.bool, device=device),
            "right_wrist_0_rgb": torch.ones(B, dtype=torch.bool, device=device),
        },
        tokenized_prompt=torch.randint(0, 1000, (B, config.max_token_len), device=device),
        tokenized_prompt_mask=torch.ones(B, config.max_token_len, dtype=torch.bool, device=device),
        state=torch.randn(B, config.action_dim, device=device, dtype=torch.float32),
    )

def main():
    config = pi0_config.Pi0Config(pi05=True, action_horizon=32, video_memory_frames=1, dtype="float32")
    torch.manual_seed(42)
    model = PI0Pytorch(config)
    model.eval()

    B = 1
    torch.manual_seed(123)
    obs = make_obs(config, B=B)
    actions = torch.randn(B, config.action_horizon, config.action_dim)
    time = torch.full((B,), 0.5, dtype=torch.float32)
    noise = torch.randn_like(actions)
    time_exp = time[:, None, None]
    x_t = time_exp * noise + (1 - time_exp) * actions

    images, img_masks, lang_tokens, lang_masks, state, temporal_mask = model._preprocess_observation(obs, train=False)
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks, temporal_mask=temporal_mask)
    suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = model.embed_suffix(state, x_t, time)

    # ===== PATH A: Joint forward (layer by layer, matching compute_layer_complete) =====
    pwe = model.paligemma_with_expert
    pal_lm = pwe.paligemma.language_model
    expert = pwe.gemma_expert.model
    num_layers = pwe.paligemma.config.text_config.num_hidden_layers

    pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
    att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
    att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
    position_ids = torch.cumsum(pad_masks, dim=1) - 1
    att_2d_masks_4d = model._prepare_attention_masks_4d(att_2d_masks)

    prefix_len = prefix_embs.shape[1]
    fwd_prefix_hs = prefix_embs.clone()
    fwd_suffix_hs = suffix_embs.clone()

    # ===== PATH B: Cache prefix, then run denoise layer by layer =====
    prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_pos_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_4d = model._prepare_attention_masks_4d(prefix_att_2d)
    pal_lm.config._attn_implementation = "eager"

    with torch.no_grad():
        _, past_kv = pwe.forward(
            attention_mask=prefix_att_4d,
            position_ids=prefix_pos_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs.clone(), None],
            use_cache=True,
        )

    suffix_position_ids = torch.sum(prefix_pad_masks, dim=-1)[:, None] + torch.cumsum(suffix_pad_masks, dim=1) - 1
    suffix_att_2d = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
    prefix_pad_2d = prefix_pad_masks[:, None, :].expand(B, suffix_pad_masks.shape[1], prefix_pad_masks.shape[1])
    full_att_2d = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)
    full_att_4d = model._prepare_attention_masks_4d(full_att_2d)

    den_suffix_hs = suffix_embs.clone()

    print(f"num_layers = {num_layers}")
    print(f"prefix_len = {prefix_len}, suffix_len = {suffix_embs.shape[1]}")
    print()

    def _layer_kv(cache, idx):
        if hasattr(cache, "key_cache"):
            return cache.key_cache[idx], cache.value_cache[idx]
        return cache[idx][0], cache[idx][1]

    with torch.no_grad():
        for li in range(num_layers):
            # ---- Forward path (joint) ----
            pal_layer = pal_lm.layers[li]
            exp_layer = expert.layers[li]

            # prefix through paligemma
            fwd_pre_normed, fwd_pre_gate = pal_layer.input_layernorm(fwd_prefix_hs, cond=None)
            # suffix through expert
            fwd_suf_normed, fwd_suf_gate = exp_layer.input_layernorm(fwd_suffix_hs, cond=adarms_cond)

            head_dim = pal_layer.self_attn.head_dim
            pre_shape = (*fwd_pre_normed.shape[:-1], -1, head_dim)
            suf_shape = (*fwd_suf_normed.shape[:-1], -1, head_dim)

            fwd_pre_q = pal_layer.self_attn.q_proj(fwd_pre_normed).view(pre_shape).transpose(1, 2)
            fwd_pre_k = pal_layer.self_attn.k_proj(fwd_pre_normed).view(pre_shape).transpose(1, 2)
            fwd_pre_v = pal_layer.self_attn.v_proj(fwd_pre_normed).view(pre_shape).transpose(1, 2)

            fwd_suf_q = exp_layer.self_attn.q_proj(fwd_suf_normed).view(suf_shape).transpose(1, 2)
            fwd_suf_k = exp_layer.self_attn.k_proj(fwd_suf_normed).view(suf_shape).transpose(1, 2)
            fwd_suf_v = exp_layer.self_attn.v_proj(fwd_suf_normed).view(suf_shape).transpose(1, 2)

            fwd_q = torch.cat([fwd_pre_q, fwd_suf_q], dim=2)
            fwd_k = torch.cat([fwd_pre_k, fwd_suf_k], dim=2)
            fwd_v = torch.cat([fwd_pre_v, fwd_suf_v], dim=2)

            dummy = torch.zeros(fwd_q.shape[0], fwd_q.shape[2], fwd_q.shape[-1], device=fwd_q.device, dtype=fwd_q.dtype)
            cos, sin = pwe.paligemma.model.language_model.rotary_emb(dummy, position_ids)
            fwd_q, fwd_k = modeling_gemma.apply_rotary_pos_emb(fwd_q, fwd_k, cos, sin, unsqueeze_dim=1)

            scaling = pal_layer.self_attn.scaling
            fwd_attn_out, _ = modeling_gemma.eager_attention_forward(pal_layer.self_attn, fwd_q, fwd_k, fwd_v, att_2d_masks_4d, scaling)

            num_heads = pal_layer.self_attn.q_proj.out_features // head_dim
            fwd_attn_out_r = fwd_attn_out.reshape(B, -1, num_heads * head_dim)

            fwd_pre_attn = pal_layer.self_attn.o_proj(fwd_attn_out_r[:, :prefix_len])
            fwd_suf_attn = exp_layer.self_attn.o_proj(fwd_attn_out_r[:, prefix_len:])

            fwd_prefix_hs = modeling_gemma._gated_residual(fwd_pre_normed, fwd_pre_attn, fwd_pre_gate)
            fwd_pre_after = fwd_prefix_hs.clone()
            fwd_prefix_hs, fwd_pre_gate2 = pal_layer.post_attention_layernorm(fwd_prefix_hs, cond=None)
            fwd_prefix_hs = pal_layer.mlp(fwd_prefix_hs)
            fwd_prefix_hs = modeling_gemma._gated_residual(fwd_pre_after, fwd_prefix_hs, fwd_pre_gate2)

            fwd_suffix_hs = modeling_gemma._gated_residual(fwd_suf_normed, fwd_suf_attn, fwd_suf_gate)
            fwd_suf_after = fwd_suffix_hs.clone()
            fwd_suffix_hs, fwd_suf_gate2 = exp_layer.post_attention_layernorm(fwd_suffix_hs, cond=adarms_cond)
            fwd_suffix_hs = exp_layer.mlp(fwd_suffix_hs)
            fwd_suffix_hs = modeling_gemma._gated_residual(fwd_suf_after, fwd_suffix_hs, fwd_suf_gate2)

            # ---- Denoise path (suffix only, with cached prefix KV) ----
            den_normed, den_gate = exp_layer.input_layernorm(den_suffix_hs, cond=adarms_cond)
            den_shape = (*den_normed.shape[:-1], -1, head_dim)

            den_q = exp_layer.self_attn.q_proj(den_normed).view(den_shape).transpose(1, 2)
            den_k = exp_layer.self_attn.k_proj(den_normed).view(den_shape).transpose(1, 2)
            den_v = exp_layer.self_attn.v_proj(den_normed).view(den_shape).transpose(1, 2)

            dummy2 = torch.zeros(den_q.shape[0], den_q.shape[2], den_q.shape[-1], device=den_q.device, dtype=den_q.dtype)
            cos2, sin2 = pwe.paligemma.model.language_model.rotary_emb(dummy2, suffix_position_ids)
            den_q, den_k = modeling_gemma.apply_rotary_pos_emb(den_q, den_k, cos2, sin2, unsqueeze_dim=1)

            pk, pv = _layer_kv(past_kv, li)
            den_full_k = torch.cat([pk, den_k], dim=2)
            den_full_v = torch.cat([pv, den_v], dim=2)

            den_attn_out, _ = modeling_gemma.eager_attention_forward(pal_layer.self_attn, den_q, den_full_k, den_full_v, full_att_4d, scaling)
            den_attn_out_r = den_attn_out.reshape(B, -1, num_heads * head_dim)
            # Note: using the same reshape as forward (no extra transpose)

            den_suf_attn = exp_layer.self_attn.o_proj(den_attn_out_r)
            den_suffix_hs = modeling_gemma._gated_residual(den_normed, den_suf_attn, den_gate)
            den_after = den_suffix_hs.clone()
            den_suffix_hs, den_gate2 = exp_layer.post_attention_layernorm(den_suffix_hs, cond=adarms_cond)
            den_suffix_hs = exp_layer.mlp(den_suffix_hs)
            den_suffix_hs = modeling_gemma._gated_residual(den_after, den_suffix_hs, den_gate2)

            # ---- Compare suffix hidden states after this layer ----
            diff = (fwd_suffix_hs - den_suffix_hs).abs()
            cos = torch.nn.functional.cosine_similarity(fwd_suffix_hs.reshape(1, -1), den_suffix_hs.reshape(1, -1)).item()

            # Also compare the cached prefix K/V vs forward prefix K/V
            # Forward prefix K after RoPE
            fwd_pre_k_rope = fwd_k[:, :, :prefix_len, :]  # [B, heads, prefix_len, head_dim]
            cached_k = pk
            kv_diff = (fwd_pre_k_rope - cached_k).abs().max().item() if fwd_pre_k_rope.shape == cached_k.shape else -1

            print(f"Layer {li:2d}: suffix cos={cos:.6f}  max_diff={diff.max():.6e}  mean_diff={diff.mean():.6e}  |  prefix_k_diff={kv_diff:.6e}")

            if cos < 0.99 and li == 0:
                print(f"  → DIVERGENCE AT LAYER 0!")
                print(f"  fwd_suf_normed norm: {fwd_suf_normed.norm():.4f}")
                print(f"  den_normed norm:     {den_normed.norm():.4f}")
                print(f"  fwd_suf_q norm:      {fwd_suf_q.norm():.4f}")
                print(f"  den_q norm:          {den_q[:,:,:fwd_suf_q.shape[2],:].norm():.4f}")

                # Check if prefix KV cache matches
                print(f"  cached pk shape: {pk.shape}")
                print(f"  fwd pre k shape: {fwd_pre_k_rope.shape}")
                if fwd_pre_k_rope.shape == cached_k.shape:
                    print(f"  prefix K max diff: {(fwd_pre_k_rope - cached_k).abs().max():.6e}")
                    print(f"  prefix K cos sim: {torch.nn.functional.cosine_similarity(fwd_pre_k_rope.reshape(1,-1), cached_k.reshape(1,-1)).item():.6f}")

    print()
    # Final norm
    fwd_suffix_hs, _ = expert.norm(fwd_suffix_hs, cond=adarms_cond)
    den_suffix_hs, _ = expert.norm(den_suffix_hs, cond=adarms_cond)
    cos_final = torch.nn.functional.cosine_similarity(fwd_suffix_hs.reshape(1,-1), den_suffix_hs.reshape(1,-1)).item()
    print(f"After final norm: cos={cos_final:.6f}")

if __name__ == "__main__":
    main()
