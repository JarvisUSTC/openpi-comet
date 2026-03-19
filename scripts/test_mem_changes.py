#!/usr/bin/env python3
"""Test script to verify MEM temporal attention and freeze logic before training.

Tests:
  1. temporal_causal_attention: shapes, causality, temporal_mask, gradient flow
  2. SiglipEncoderLayer integration: temporal attention actually changes output
  3. SiglipEncoder: history frame discard
  4. _freeze_backbone: full-param vs LoRA freeze logic
  5. End-to-end forward pass with dummy SigLIP config
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import torch
import torch.nn as nn
import numpy as np

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(REPO_ROOT, ".venv", "lib", "python3.11", "site-packages"))

import importlib, shutil, glob as _glob
replace_src = os.path.join(REPO_ROOT, "src", "openpi", "models_pytorch", "transformers_replace")
site_transformers = os.path.join(REPO_ROOT, ".venv", "lib", "python3.11", "site-packages", "transformers")
for src_dir in _glob.glob(os.path.join(replace_src, "**"), recursive=True):
    if os.path.isfile(src_dir):
        rel = os.path.relpath(src_dir, replace_src)
        dst = os.path.join(site_transformers, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src_dir, dst)

from transformers.models.siglip.modeling_siglip import (
    temporal_causal_attention,
    temporal_posemb_sincos,
    SiglipEncoderLayer,
    SiglipEncoder,
)
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig

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


def test_temporal_causal_attention_shapes():
    """Test output shapes and basic properties."""
    print("\n== Test 1: temporal_causal_attention shapes ==")
    B, K, n, d, num_heads = 2, 6, 16, 64, 8
    BK = B * K

    hidden = torch.randn(BK, n, d)
    ln = nn.LayerNorm(d)
    q_proj = nn.Linear(d, d)
    k_proj = nn.Linear(d, d)
    v_proj = nn.Linear(d, d)
    out_proj = nn.Linear(d, d)
    t_mask = torch.tensor([[False, False, False, False, False, True],
                           [True,  True,  True,  True,  True,  True]])

    out = temporal_causal_attention(
        hidden, num_frames=K, num_heads=num_heads,
        layer_norm=ln, q_proj=q_proj, k_proj=k_proj, v_proj=v_proj,
        out_proj=out_proj, temporal_mask=t_mask,
    )
    check("output shape", out.shape == (BK, n, d), f"got {out.shape}")
    check("output dtype", out.dtype == hidden.dtype)
    check("no NaN", not torch.isnan(out).any().item())
    check("no Inf", not torch.isinf(out).any().item())

    out_single = temporal_causal_attention(
        torch.randn(B, n, d), num_frames=1, num_heads=num_heads,
        layer_norm=ln, q_proj=q_proj, k_proj=k_proj, v_proj=v_proj,
        out_proj=out_proj,
    )
    check("K=1 returns zeros", torch.all(out_single == 0).item())


def test_temporal_causal_attention_causality():
    """Verify causal masking: changing future frames shouldn't affect past frames."""
    print("\n== Test 2: Causality ==")
    B, K, n, d, num_heads = 1, 4, 8, 32, 4

    torch.manual_seed(42)
    ln = nn.LayerNorm(d)
    q_proj = nn.Linear(d, d)
    k_proj = nn.Linear(d, d)
    v_proj = nn.Linear(d, d)
    out_proj = nn.Linear(d, d)

    hidden_a = torch.randn(B * K, n, d)
    hidden_b = hidden_a.clone()
    hidden_b[3] = torch.randn(n, d)  # change frame 3 (last frame)

    out_a = temporal_causal_attention(
        hidden_a, num_frames=K, num_heads=num_heads,
        layer_norm=ln, q_proj=q_proj, k_proj=k_proj, v_proj=v_proj,
        out_proj=out_proj,
    )
    out_b = temporal_causal_attention(
        hidden_b, num_frames=K, num_heads=num_heads,
        layer_norm=ln, q_proj=q_proj, k_proj=k_proj, v_proj=v_proj,
        out_proj=out_proj,
    )

    check("frame 0 unaffected by future change", torch.allclose(out_a[0], out_b[0], atol=1e-6))
    check("frame 1 unaffected by future change", torch.allclose(out_a[1], out_b[1], atol=1e-6))
    check("frame 2 unaffected by future change", torch.allclose(out_a[2], out_b[2], atol=1e-6))
    check("frame 3 IS affected", not torch.allclose(out_a[3], out_b[3], atol=1e-5))


def test_temporal_mask_blocks_invalid_frames():
    """With temporal_mask=[F,F,F,T], only frame 3 is valid. Frame 3 should only attend to itself."""
    print("\n== Test 3: temporal_mask blocks invalid frames ==")
    B, K, n, d, num_heads = 1, 4, 8, 32, 4

    torch.manual_seed(0)
    ln = nn.LayerNorm(d)
    q_proj = nn.Linear(d, d)
    k_proj = nn.Linear(d, d)
    v_proj = nn.Linear(d, d)
    out_proj = nn.Linear(d, d)

    hidden = torch.randn(B * K, n, d)
    mask_all_valid = torch.ones(B, K, dtype=torch.bool)
    mask_only_last = torch.tensor([[False, False, False, True]])

    out_all = temporal_causal_attention(
        hidden, num_frames=K, num_heads=num_heads,
        layer_norm=ln, q_proj=q_proj, k_proj=k_proj, v_proj=v_proj,
        out_proj=out_proj, temporal_mask=mask_all_valid,
    )
    out_masked = temporal_causal_attention(
        hidden, num_frames=K, num_heads=num_heads,
        layer_norm=ln, q_proj=q_proj, k_proj=k_proj, v_proj=v_proj,
        out_proj=out_proj, temporal_mask=mask_only_last,
    )
    check("masked output differs from all-valid", not torch.allclose(out_all[3], out_masked[3], atol=1e-5))
    check("masked output has no NaN", not torch.isnan(out_masked).any().item())


def test_gradient_flow():
    """Ensure gradients flow through temporal attention to the shared projections."""
    print("\n== Test 4: Gradient flow through shared projections ==")
    B, K, n, d, num_heads = 1, 4, 8, 32, 4

    ln = nn.LayerNorm(d)
    q_proj = nn.Linear(d, d)
    k_proj = nn.Linear(d, d)
    v_proj = nn.Linear(d, d)
    out_proj = nn.Linear(d, d)

    hidden = torch.randn(B * K, n, d, requires_grad=True)

    out = temporal_causal_attention(
        hidden, num_frames=K, num_heads=num_heads,
        layer_norm=ln, q_proj=q_proj, k_proj=k_proj, v_proj=v_proj,
        out_proj=out_proj,
    )
    loss = out.sum()
    loss.backward()

    check("grad on hidden_states", hidden.grad is not None and hidden.grad.abs().sum() > 0)
    check("grad on q_proj.weight", q_proj.weight.grad is not None and q_proj.weight.grad.abs().sum() > 0)
    check("grad on k_proj.weight", k_proj.weight.grad is not None and k_proj.weight.grad.abs().sum() > 0)
    check("grad on v_proj.weight", v_proj.weight.grad is not None and v_proj.weight.grad.abs().sum() > 0)
    check("grad on out_proj.weight", out_proj.weight.grad is not None and out_proj.weight.grad.abs().sum() > 0)
    check("grad on layer_norm.weight", ln.weight.grad is not None and ln.weight.grad.abs().sum() > 0)


def test_temporal_changes_output():
    """With projections, temporal attention should produce non-trivial output even for identical frames."""
    print("\n== Test 5: Temporal attention NOT invisible (unlike old version) ==")
    B, K, n, d, num_heads = 1, 4, 8, 32, 4

    torch.manual_seed(123)
    ln = nn.LayerNorm(d)
    q_proj = nn.Linear(d, d)
    k_proj = nn.Linear(d, d)
    v_proj = nn.Linear(d, d)
    out_proj = nn.Linear(d, d)

    single_frame = torch.randn(1, n, d)
    hidden = single_frame.repeat(K, 1, 1)  # all K frames identical

    out = temporal_causal_attention(
        hidden, num_frames=K, num_heads=num_heads,
        layer_norm=ln, q_proj=q_proj, k_proj=k_proj, v_proj=v_proj,
        out_proj=out_proj,
    )

    cos_sim = torch.nn.functional.cosine_similarity(
        out.reshape(K, -1), hidden.reshape(K, -1), dim=-1
    ).mean()
    check(
        "output NOT parallel to input (cos_sim < 0.99)",
        cos_sim.item() < 0.99,
        f"cos_sim={cos_sim.item():.4f} — if ≈1.0, temporal attention is invisible",
    )
    check("output is non-zero", out.abs().mean().item() > 1e-6)


def test_encoder_layer_integration():
    """SiglipEncoderLayer with has_temporal_attn=True produces different output than False."""
    print("\n== Test 6: SiglipEncoderLayer integration ==")
    cfg = SiglipVisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=8,
        image_size=32,
        patch_size=4,
    )
    layer = SiglipEncoderLayer(cfg)
    layer.eval()

    B, K, n = 1, 4, 16
    d = cfg.hidden_size
    hidden = torch.randn(B * K, n, d)
    attn_mask = torch.zeros(B * K, 1, n, n)

    out_no_temporal = layer(
        hidden, attn_mask, output_attentions=False,
        num_frames=K, layer_idx=0, has_temporal_attn=False,
    )[0]
    out_with_temporal = layer(
        hidden, attn_mask, output_attentions=False,
        num_frames=K, layer_idx=0, has_temporal_attn=True,
    )[0]

    check("output shapes match", out_no_temporal.shape == out_with_temporal.shape)
    check(
        "temporal attention changes output",
        not torch.allclose(out_no_temporal, out_with_temporal, atol=1e-5),
    )


def test_encoder_discards_history():
    """SiglipEncoder should discard history frames and return only [B, n, d]."""
    print("\n== Test 7: SiglipEncoder discards history frames ==")
    cfg = SiglipVisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=8,
        num_attention_heads=8,
        image_size=32,
        patch_size=4,
    )
    encoder = SiglipEncoder(cfg)
    encoder.eval()

    B, K, n, d = 2, 6, 16, cfg.hidden_size
    hidden = torch.randn(B * K, n, d)
    attn_mask = torch.zeros(B * K, 1, n, n)
    t_mask = torch.ones(B, K, dtype=torch.bool)

    result = encoder(
        hidden, attention_mask=attn_mask,
        num_frames=K, temporal_mask=t_mask,
    )
    out = result[0] if isinstance(result, tuple) else result.last_hidden_state

    check("output shape [B, n, d]", out.shape == (B, n, d), f"got {out.shape}")
    check("no NaN", not torch.isnan(out).any().item())


def test_freeze_backbone_full_param():
    """_freeze_backbone with nnx.Nothing should set all params trainable."""
    print("\n== Test 8: _freeze_backbone full-param mode ==")
    sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
    from train_pytorch import _freeze_backbone

    import flax.nnx as nnx

    model = nn.Sequential(nn.Linear(32, 64), nn.Linear(64, 32))
    for p in model.parameters():
        p.requires_grad = False

    class FakeConfig:
        freeze_filter = nnx.Nothing

    _freeze_backbone(model, FakeConfig(), is_main=False)
    all_trainable = all(p.requires_grad for p in model.parameters())
    check("all params trainable with nnx.Nothing", all_trainable)


def test_temporal_posemb():
    """Verify positional embedding properties."""
    print("\n== Test 9: temporal_posemb_sincos ==")
    K, d = 6, 64
    pe = temporal_posemb_sincos(K, d, device="cpu")
    check("pe shape", pe.shape == (K, d), f"got {pe.shape}")
    check("pe bounded [-2, 2]", pe.abs().max().item() <= 2.0 + 1e-6,
          f"max abs = {pe.abs().max().item():.4f}")
    check("current frame (last) is zero vector", torch.all(pe[-1] == 0).item())
    check("history frames are non-zero", pe[:-1].abs().sum().item() > 0)


def test_bfloat16_forward():
    """Temporal attention should work in bfloat16 (training dtype)."""
    print("\n== Test 10: bfloat16 forward ==")
    if not torch.cuda.is_available():
        print("  ⚠️  Skipping bfloat16 test (no CUDA)")
        return

    B, K, n, d, num_heads = 1, 4, 8, 64, 8
    device = "cuda"

    ln = nn.LayerNorm(d).to(device, dtype=torch.bfloat16)
    q_proj = nn.Linear(d, d).to(device, dtype=torch.bfloat16)
    k_proj = nn.Linear(d, d).to(device, dtype=torch.bfloat16)
    v_proj = nn.Linear(d, d).to(device, dtype=torch.bfloat16)
    out_proj = nn.Linear(d, d).to(device, dtype=torch.bfloat16)
    hidden = torch.randn(B * K, n, d, device=device, dtype=torch.bfloat16)
    t_mask = torch.ones(B, K, dtype=torch.bool, device=device)

    out = temporal_causal_attention(
        hidden, num_frames=K, num_heads=num_heads,
        layer_norm=ln, q_proj=q_proj, k_proj=k_proj, v_proj=v_proj,
        out_proj=out_proj, temporal_mask=t_mask,
    )
    check("bfloat16 output dtype", out.dtype == torch.bfloat16)
    check("bfloat16 no NaN", not torch.isnan(out).any().item())
    check("bfloat16 no Inf", not torch.isinf(out).any().item())


if __name__ == "__main__":
    print("=" * 60)
    print("MEM Temporal Attention — Verification Tests")
    print("=" * 60)

    test_temporal_causal_attention_shapes()
    test_temporal_causal_attention_causality()
    test_temporal_mask_blocks_invalid_frames()
    test_gradient_flow()
    test_temporal_changes_output()
    test_encoder_layer_integration()
    test_encoder_discards_history()
    test_freeze_backbone_full_param()
    test_temporal_posemb()
    test_bfloat16_forward()

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed")
    print("=" * 60)
    sys.exit(1 if FAIL > 0 else 0)
