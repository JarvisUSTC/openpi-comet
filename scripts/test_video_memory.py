"""
MEM Video Memory 逐步验证脚本。

用法:
    # Step 0: 记录基线
    python scripts/test_video_memory.py --step 0

    # Step 1: 验证配置改动不影响结果
    python scripts/test_video_memory.py --step 1

    # Step 2: 验证接口改动不影响结果
    python scripts/test_video_memory.py --step 2

    # Step 3: 独立测试 temporal attention 函数
    python scripts/test_video_memory.py --step 3

    # Step 4: 验证 temporal attention 接入 ViT
    python scripts/test_video_memory.py --step 4

    # Step 5: 验证 embed_prefix 端到端
    python scripts/test_video_memory.py --step 5
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

TMP_DIR = os.path.join(os.path.dirname(__file__), "..", "_tmp")
BASELINE_PATH = os.path.join(TMP_DIR, "video_memory_baseline.pt")
WEIGHTS_PATH = os.path.join(TMP_DIR, "video_memory_weights.pt")

MODEL_SEED = 0
DATA_SEED = 42


def _create_model_deterministic(config):
    """Create model with fixed seed for reproducibility."""
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    torch.manual_seed(MODEL_SEED)
    model = PI0Pytorch(config)
    model.eval()
    return model


def _create_and_save_model(config):
    """Create model, save weights, return model."""
    model = _create_model_deterministic(config)
    os.makedirs(TMP_DIR, exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS_PATH)
    return model


def _load_model_with_saved_weights(config):
    """Create model and load saved weights for exact reproducibility."""
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    model = PI0Pytorch(config)
    state_dict = torch.load(WEIGHTS_PATH, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


# ============================================================
# Step 0 & 1 & 2: 基线 / 配置 / 接口 验证
# ============================================================

def run_step0(args):
    """Step 0: 记录基线——用现有模型跑一次 forward，保存结果和权重。"""
    print("=" * 60)
    print("Step 0: 记录基线")
    print("=" * 60)

    from openpi.models import pi0_config

    config = pi0_config.Pi0Config(pi05=True, action_horizon=32)
    print(f"Config: pi05={config.pi05}, action_horizon={config.action_horizon}")

    model = _create_and_save_model(config)
    print(f"  权重已保存到 {WEIGHTS_PATH}")

    torch.manual_seed(DATA_SEED)
    B = 1
    obs, actions = _make_fake_observation(config, B, device="cpu")

    print(f"  Images shape: {obs.images['base_0_rgb'].shape}")
    print(f"  State shape: {obs.state.shape}")
    print(f"  Actions shape: {actions.shape}")

    with torch.no_grad():
        loss = model(obs, actions)

    loss_val = loss.mean().item()
    print(f"  Baseline loss: {loss_val}")

    torch.save({
        "loss": loss_val,
        "loss_tensor": loss.cpu(),
    }, BASELINE_PATH)
    print(f"  基线已保存到 {BASELINE_PATH}")
    print("  ✅ Step 0 完成")


def run_step1(args):
    """Step 1: 验证 pi0_config.py 配置改动不影响结果。"""
    print("=" * 60)
    print("Step 1: 验证配置改动 (video_memory_frames=1 应该和基线一致)")
    print("=" * 60)

    baseline = _load_baseline()
    from openpi.models import pi0_config

    config = pi0_config.Pi0Config(pi05=True, action_horizon=32)

    if not hasattr(config, "video_memory_frames"):
        print("  ❌ Pi0Config 缺少 video_memory_frames 字段")
        return False

    print(f"  video_memory_frames = {config.video_memory_frames}")
    print(f"  video_memory_stride_s = {config.video_memory_stride_s}")
    assert config.video_memory_frames == 1, "默认值应该是 1"

    model = _load_model_with_saved_weights(config)

    torch.manual_seed(DATA_SEED)
    obs, actions = _make_fake_observation(config, B=1, device="cpu")

    with torch.no_grad():
        loss = model(obs, actions)

    loss_val = loss.mean().item()
    diff = abs(loss_val - baseline["loss"])
    print(f"  当前 loss: {loss_val}")
    print(f"  基线 loss: {baseline['loss']}")
    print(f"  差异: {diff}")

    if diff < 1e-5:
        print("  ✅ Step 1 通过: loss 与基线一致")
        return True
    else:
        print("  ❌ Step 1 失败: loss 与基线不一致")
        return False


def run_step2(args):
    """Step 2: 验证 SigLIP 接口改动不影响结果（只加参数，不加逻辑）。"""
    print("=" * 60)
    print("Step 2: 验证 SigLIP 接口改动 (num_frames 参数透传)")
    print("=" * 60)

    baseline = _load_baseline()
    from openpi.models import pi0_config

    config = pi0_config.Pi0Config(pi05=True, action_horizon=32)
    model = _load_model_with_saved_weights(config)

    torch.manual_seed(DATA_SEED)
    obs, actions = _make_fake_observation(config, B=1, device="cpu")

    with torch.no_grad():
        loss = model(obs, actions)

    loss_val = loss.mean().item()
    diff = abs(loss_val - baseline["loss"])
    print(f"  当前 loss: {loss_val}")
    print(f"  基线 loss: {baseline['loss']}")
    print(f"  差异: {diff}")

    if diff < 1e-5:
        print("  ✅ Step 2 通过: 接口改动不影响结果")
        return True
    else:
        print("  ❌ Step 2 失败: 接口改动破坏了结果")
        return False


# ============================================================
# Step 3: 独立测试 temporal attention 函数
# ============================================================

def run_step3(args):
    """Step 3: 独立测试 temporal_posemb_sincos 和 temporal_causal_attention。"""
    print("=" * 60)
    print("Step 3: 独立测试 temporal attention 函数")
    print("=" * 60)

    passed = 0
    total = 4

    # --- Test 1: Temporal PE shape 和 t=0 为 0 ---
    print("\n  Test 1: temporal PE 在当前帧 (t=0) 时为 0")
    try:
        from transformers.models.siglip.modeling_siglip import (
            temporal_posemb_sincos,
        )

        pe = temporal_posemb_sincos(K=6, width=1152, device=torch.device("cpu"))
        assert pe.shape == (6, 1152), f"Shape 错误: {pe.shape}"
        assert torch.allclose(pe[-1], torch.zeros(1152), atol=1e-7), "当前帧 PE 不为 0"
        assert not torch.allclose(pe[0], torch.zeros(1152), atol=1e-3), "历史帧 PE 不应为 0"
        print("    ✅ 通过")
        passed += 1
    except Exception as e:
        print(f"    ❌ 失败: {e}")

    # --- Test 2: K=1 时 temporal attention 输出为 0 ---
    print("\n  Test 2: K=1 时 temporal attention 输出为 0")
    try:
        from transformers.models.siglip.modeling_siglip import (
            TemporalCausalAttentionModule,
        )
        from transformers.models.siglip.configuration_siglip import SiglipVisionConfig

        cfg = SiglipVisionConfig(hidden_size=1152, num_attention_heads=16)
        module = TemporalCausalAttentionModule(cfg)
        module.eval()

        x = torch.randn(2, 256, 1152)  # [B*1, n, d]
        with torch.no_grad():
            out = module(x, num_frames=1)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-5), \
            f"K=1 时输出不为 0, max abs = {out.abs().max().item()}"
        print("    ✅ 通过")
        passed += 1
    except Exception as e:
        print(f"    ❌ 失败: {e}")

    # --- Test 3: 输出 shape 正确 ---
    print("\n  Test 3: 输出 shape 正确")
    try:
        from transformers.models.siglip.modeling_siglip import (
            TemporalCausalAttentionModule,
        )
        from transformers.models.siglip.configuration_siglip import SiglipVisionConfig

        cfg = SiglipVisionConfig(hidden_size=1152, num_attention_heads=16)
        module = TemporalCausalAttentionModule(cfg)
        module.eval()

        B, K, n, d = 2, 6, 256, 1152
        x = torch.randn(B * K, n, d)
        with torch.no_grad():
            out = module(x, num_frames=K)
        assert out.shape == (B * K, n, d), f"Shape 错误: {out.shape}"
        print("    ✅ 通过")
        passed += 1
    except Exception as e:
        print(f"    ❌ 失败: {e}")

    # --- Test 4: 因果性——修改未来帧不影响过去帧 ---
    print("\n  Test 4: 因果性 (修改未来帧不影响过去帧)")
    try:
        from transformers.models.siglip.modeling_siglip import (
            TemporalCausalAttentionModule,
        )
        from transformers.models.siglip.configuration_siglip import SiglipVisionConfig

        cfg = SiglipVisionConfig(hidden_size=1152, num_attention_heads=16)
        module = TemporalCausalAttentionModule(cfg)
        # out_proj is zero-init by design; set to non-zero to test causality
        torch.manual_seed(777)
        nn.init.xavier_uniform_(module.out_proj.weight)
        module.eval()

        B, K, n, d = 1, 4, 64, 1152
        torch.manual_seed(123)
        x = torch.randn(B * K, n, d)
        with torch.no_grad():
            out1 = module(x.clone(), num_frames=K)

        x_modified = x.clone()
        x_modified[-1] = torch.randn(n, d)
        with torch.no_grad():
            out2 = module(x_modified, num_frames=K)

        assert torch.allclose(out1[:3], out2[:3], atol=1e-5), \
            f"因果性违反: 前 3 帧 max diff = {(out1[:3] - out2[:3]).abs().max().item()}"
        assert not torch.allclose(out1[-1:], out2[-1:], atol=1e-3), \
            "最后一帧应该变化但没变"
        print("    ✅ 通过")
        passed += 1
    except Exception as e:
        print(f"    ❌ 失败: {e}")

    print(f"\n  结果: {passed}/{total} 通过")
    if passed == total:
        print("  ✅ Step 3 全部通过")
        return True
    else:
        print("  ❌ Step 3 部分失败")
        return False


# ============================================================
# Step 4: 验证 temporal attention 接入 ViT
# ============================================================

def run_step4(args):
    """Step 4: 验证 temporal attention 接入 ViT 后的行为。"""
    print("=" * 60)
    print("Step 4: 验证 temporal attention 接入 ViT")
    print("=" * 60)

    baseline = _load_baseline()
    from openpi.models import pi0_config

    passed = 0
    total = 3

    # --- Test A: K=1 退化 ---
    print("\n  Test A: K=1 退化 (loss 与基线一致)")
    try:
        config = pi0_config.Pi0Config(pi05=True, action_horizon=32, video_memory_frames=1)
        model = _load_model_with_saved_weights(config)

        torch.manual_seed(DATA_SEED)
        obs, actions = _make_fake_observation(config, B=1, device="cpu")

        with torch.no_grad():
            loss = model(obs, actions)

        loss_val = loss.mean().item()
        diff = abs(loss_val - baseline["loss"])
        print(f"    当前 loss: {loss_val}, 基线 loss: {baseline['loss']}, 差异: {diff}")

        assert diff < 1e-5, f"K=1 退化失败, diff={diff}"
        print("    ✅ 通过")
        passed += 1
    except Exception as e:
        print(f"    ❌ 失败: {e}")

    # --- Test B: K>1 SigLIP 输出 shape 正确 ---
    print("\n  Test B: K>1 SigLIP 输出 shape 正确")
    try:
        config = pi0_config.Pi0Config(pi05=True, action_horizon=32, video_memory_frames=6)
        model = _load_model_with_saved_weights(config)

        B, K = 2, 6
        fake_images = torch.randn(B * K, 3, 224, 224)

        with torch.no_grad():
            output = model.paligemma_with_expert.embed_image(fake_images, num_frames=K)

        expected_shape = (B, 256, output.shape[-1])
        assert output.shape == expected_shape, \
            f"Shape 错误: got {output.shape}, expected {expected_shape}"
        print(f"    输出 shape: {output.shape} ✓")
        print("    ✅ 通过")
        passed += 1
    except Exception as e:
        print(f"    ❌ 失败: {e}")

    # --- Test C: 同一图重复 K 次，输出应接近单帧 ---
    # NOTE: Must call _init_temporal_from_spatial to zero out_proj, because
    # HuggingFace post_init() overwrites __init__'s zero-init with lecun_normal_.
    print("\n  Test C: 同一图重复 K 次 vs 单帧 (需先 _init_temporal_from_spatial)")
    try:
        config = pi0_config.Pi0Config(pi05=True, action_horizon=32, video_memory_frames=6)
        model = _load_model_with_saved_weights(config)
        siglip_encoder = model.paligemma_with_expert.paligemma.vision_tower.vision_model.encoder
        siglip_encoder._init_temporal_from_spatial()

        B, K = 1, 6
        torch.manual_seed(99)
        single_image = torch.randn(B, 3, 224, 224)
        repeated = single_image.repeat(K, 1, 1, 1)  # [K, 3, 224, 224]

        with torch.no_grad():
            out_single = model.paligemma_with_expert.embed_image(single_image, num_frames=1)
            out_multi = model.paligemma_with_expert.embed_image(repeated, num_frames=K)

        diff = (out_single - out_multi).abs().mean().item()
        print(f"    同图差异: {diff}")
        assert diff < 0.05, f"同图差异过大: {diff}"
        print("    ✅ 通过")
        passed += 1
    except Exception as e:
        print(f"    ❌ 失败: {e}")

    print(f"\n  结果: {passed}/{total} 通过")
    if passed == total:
        print("  ✅ Step 4 全部通过")
        return True
    else:
        print("  ❌ Step 4 部分失败")
        return False


# ============================================================
# Step 5: 验证 embed_prefix 端到端
# ============================================================

def run_step5(args):
    """Step 5: 验证 embed_prefix 改动后端到端行为。"""
    print("=" * 60)
    print("Step 5: 验证 embed_prefix 端到端")
    print("=" * 60)

    baseline = _load_baseline()
    from openpi.models import pi0_config

    passed = 0
    total = 2

    # --- Test A: K=1 退化 ---
    print("\n  Test A: K=1 端到端退化")
    try:
        config = pi0_config.Pi0Config(pi05=True, action_horizon=32, video_memory_frames=1)
        model = _load_model_with_saved_weights(config)

        torch.manual_seed(DATA_SEED)
        obs, actions = _make_fake_observation(config, B=1, device="cpu")

        with torch.no_grad():
            loss = model(obs, actions)

        loss_val = loss.mean().item()
        diff = abs(loss_val - baseline["loss"])
        print(f"    当前 loss: {loss_val}, 基线: {baseline['loss']}, 差异: {diff}")
        assert diff < 1e-5, f"退化失败, diff={diff}"
        print("    ✅ 通过")
        passed += 1
    except Exception as e:
        print(f"    ❌ 失败: {e}")

    # --- Test B: K>1 端到端 forward 不报错 ---
    print("\n  Test B: K>1 端到端 forward")
    try:
        K = 6
        config = pi0_config.Pi0Config(pi05=True, action_horizon=32, video_memory_frames=K)
        model = _load_model_with_saved_weights(config)

        torch.manual_seed(DATA_SEED)
        obs, actions = _make_fake_observation(config, B=1, device="cpu", num_frames=K)

        with torch.no_grad():
            loss = model(obs, actions)

        loss_val = loss.mean().item()
        print(f"    K={K} loss: {loss_val}")
        assert np.isfinite(loss_val), f"loss 不是有限值: {loss_val}"
        assert loss.shape[-1] == config.action_horizon, f"loss shape 错误: {loss.shape}"
        print("    ✅ 通过")
        passed += 1
    except Exception as e:
        print(f"    ❌ 失败: {e}")

    print(f"\n  结果: {passed}/{total} 通过")
    if passed == total:
        print("  ✅ Step 5 全部通过")
        return True
    else:
        print("  ❌ Step 5 部分失败")
        return False


# ============================================================
# 辅助函数
# ============================================================

def _make_fake_observation(config, B, device="cpu", num_frames=1):
    """构造假的 Observation 和 Actions，用于验证。"""
    from openpi.models.model import Observation
    from openpi.shared import array_typing as at

    H, W = 224, 224

    img_batch = B * num_frames if num_frames > 1 else B
    images = {
        "base_0_rgb": torch.randn(img_batch, 3, H, W, device=device),
        "left_wrist_0_rgb": torch.randn(img_batch, 3, H, W, device=device),
        "right_wrist_0_rgb": torch.randn(img_batch, 3, H, W, device=device),
    }

    image_masks = {
        "base_0_rgb": torch.ones(img_batch, dtype=torch.bool, device=device),
        "left_wrist_0_rgb": torch.ones(img_batch, dtype=torch.bool, device=device),
        "right_wrist_0_rgb": torch.ones(img_batch, dtype=torch.bool, device=device),
    }

    state = torch.randn(B, config.action_dim, device=device)

    prompt_len = config.max_token_len
    tokenized_prompt = torch.ones(B, prompt_len, dtype=torch.long, device=device)
    tokenized_prompt_mask = torch.ones(B, prompt_len, dtype=torch.bool, device=device)

    with at.disable_typechecking():
        obs = Observation(
            images=images,
            image_masks=image_masks,
            state=state,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
        )

    actions = torch.randn(B, config.action_horizon, config.action_dim, device=device)
    return obs, actions


def _load_baseline():
    """加载 Step 0 保存的基线。"""
    if not os.path.exists(BASELINE_PATH):
        print(f"  ❌ 基线文件不存在: {BASELINE_PATH}")
        print("  请先运行: python scripts/test_video_memory.py --step 0")
        sys.exit(1)
    return torch.load(BASELINE_PATH, weights_only=True)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="MEM Video Memory 逐步验证")
    parser.add_argument("--step", type=int, required=True, choices=[0, 1, 2, 3, 4, 5],
                        help="要运行的验证步骤 (0-5)")
    args = parser.parse_args()

    runners = {
        0: run_step0,
        1: run_step1,
        2: run_step2,
        3: run_step3,
        4: run_step4,
        5: run_step5,
    }

    start = time.time()
    result = runners[args.step](args)
    elapsed = time.time() - start
    print(f"\n耗时: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
