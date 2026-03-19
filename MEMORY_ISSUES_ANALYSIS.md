# OpenPI Memory 训练问题深度分析

> 基于对 `openpi-memory` 代码库的完整审查，总结训练效果差、梯度爆炸的根因。

---

## 问题总览

| 优先级 | 问题 | 影响 | 涉及文件 |
|--------|------|------|----------|
| **P0** | VideoMemoryDataset 在 shuffle 下历史帧全部失效 | 模型从未见过有效历史帧 | `training/data_loader.py` |
| **P0** | Temporal Attention 没有可学习参数 | 无法学习时间关系 | `siglip/modeling_siglip.py` |
| **P1** | Temporal Attention 残差连接缺少 LayerNorm | 梯度爆炸的直接原因 | `siglip/modeling_siglip.py` |
| **P1** | 全参数微调 + 仅 50 episodes 数据 | 过拟合 + 灾难性遗忘 | `train_pytorch.py`, `config.py` |
| **P2** | Positional Encoding 的 `cos()-1.0` 偏移 | 加剧训练不稳定 | `siglip/modeling_siglip.py` |
| **P2** | K=6 帧全量通过完整 SigLIP 编码器 | 计算浪费、梯度更难稳定 | 整体架构 |

---

## P0：致命问题

### 1. VideoMemoryDataset 的有状态 Buffer 在 Shuffle 下完全失效

**文件**: `src/openpi/training/data_loader.py` L155-197

**现象**: 训练时历史帧几乎全是 padding（重复当前帧），模型实际上在 K=6 的配置下只看到了 1 帧有效数据。

**原因**:

`VideoMemoryDataset` 用一个 `self._buffers` 字典维护每个 episode 的帧历史。但它有两个致命缺陷：

```python
# data_loader.py L164-166
if ep_idx not in self._buffers:
    self._buffers = {}          # ← 遇到新 episode 时，清空所有 buffer
    self._buffers[ep_idx] = {}
```

1. **Shuffle 打乱了访问顺序**：DataLoader 设置了 `shuffle=True`，加上 `DistributedSampler` 也会打乱。连续两次 `__getitem__` 调用很可能来自不同 episode，导致 buffer 反复被清空。

2. **Stride 太大导致 buffer 永远填不满**：`stride = int(1.0 * 30) = 30` 帧，K=6 需要 buffer 积累 `(6-1) × 30 = 150` 帧才能得到完整历史。而一个 chunk 才 250 帧（GOP size），即使顺序访问也需要 60% 的 chunk 才有完整历史。

**结果**: `valid_flags` 几乎全是 `False`，`temporal_mask` 几乎全是 `[False, False, False, False, False, True]`（只有当前帧有效）。模型训练时从未真正学习过时间信息。

**修复方向**:
- 不要用有状态 buffer。改为在 `__getitem__` 中根据 episode_index 和 frame_index **直接随机访问**同一 episode 的历史帧。
- 或者保证 DataLoader 按 episode 内顺序采样（不 shuffle，或 episode-level shuffle + episode 内顺序）。
- 先减小 stride（如 stride=1 或 stride=5），确保 buffer 能快速填满。

---

### 2. Temporal Attention 没有任何可学习参数

**文件**: `src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py` L375-443

**现象**: temporal attention 对训练 loss 的贡献接近于零（因为没有参数来学习有用的时间特征）。

**原因**:

```python
# Q 和 K 完全相同，都是 hidden_states + PE
q = x_pe.permute(0, 2, 1, 3)
k = q                          # ← Q == K，无独立投影
v_raw = x.view(...)            # ← V = 原始 hidden states，也无投影

# 最终返回残差
return out - x
```

- **Q = K = hidden_states + sinusoidal_PE**：没有任何 `nn.Linear` 投影
- **V = hidden_states（raw）**：也没有投影
- 整个函数是纯计算，**零可学习参数**

对比 MEM 论文和标准 Transformer 实现：论文使用带有可学习投影的 cross-attention 来融合时间信息。你的实现只是一个固定模式的注意力，模型无法学习"提取什么时间信息"。

**修复方向**:
- 为 temporal attention 添加独立的可学习 `q_proj`, `k_proj`, `v_proj`（至少添加 `v_proj`）。
- 或者使用 cross-attention 架构：当前帧 query 历史帧。
- 添加可学习的 gating 系数（初始化为 0），控制 temporal attention 的影响力度。

---

## P1：严重问题

### 3. Temporal Attention 残差连接缺少 LayerNorm

**文件**: `src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py` L566-581

**现象**: 训练后期梯度爆炸（25k 步后 grad_norm 飙升到 50M+）。

**原因**:

标准 Transformer 每个 sub-layer 都有 LayerNorm（pre-norm 或 post-norm），但 temporal attention 直接做 raw residual：

```python
# SiglipEncoderLayer.forward()
# 空间注意力有 LayerNorm ✓
hidden_states = self.layer_norm1(hidden_states)
hidden_states, _ = self.self_attn(hidden_states=hidden_states, ...)
hidden_states = residual + hidden_states

# temporal 注意力没有 LayerNorm ✗
if num_frames > 1 and (layer_idx + 1) % 4 == 0:
    hidden_states = hidden_states + temporal_causal_attention(...)  # ← 无 LayerNorm！
```

未归一化的残差会导致：
- 梯度幅度不可控
- 随着训练进行，hidden states 的 scale 逐渐偏移
- 最终触发梯度爆炸

**修复方向**:
- 在 temporal attention 前添加 pre-norm：`temporal_causal_attention(self.temporal_ln(hidden_states), ...)`
- 或添加 post-norm
- 添加独立的 `nn.LayerNorm` 作为 SiglipEncoderLayer 的新成员

---

### 4. 全参数微调 + 极少数据 = 灾难性遗忘

**文件**: `scripts/train_pytorch.py` L504, `src/openpi/training/config.py` L857-889

**现象**: 即使 loss 下降，模型在评估时完全不工作（"找不到任务"）。

**原因**:

1. **PyTorch 训练脚本没有应用 freeze_filter**：
   ```python
   # train_pytorch.py L504
   optim = torch.optim.AdamW(
       model.parameters(),  # ← 所有参数都在训练！
       ...
   )
   ```
   配置文件里虽然设置了 `freeze_filter`，但 PyTorch 训练脚本完全忽略了它。

2. **freeze_filter 本身也是空的**：
   ```python
   # pi0_config.py L119-120
   if not filters:
       return nnx.Nothing  # ← 不冻结任何东西
   ```
   因为没有使用 LoRA，所以 freeze filter 返回 `Nothing`。

3. **数据太少**：只有 50 个 episode，但模型有 3B+ 参数。全参数微调导致严重过拟合和灾难性遗忘——预训练学到的通用视觉/语言能力被破坏。

**修复方向**:
- **冻结 SigLIP vision tower 和 PaliGemma language model**
- 只训练：temporal attention 新增的参数 + action expert + action projections
- 在 `train_pytorch.py` 中实现参数冻结：
  ```python
  for name, param in model.named_parameters():
      if "temporal" not in name and "action" not in name and "expert" not in name:
          param.requires_grad = False
  ```

---

## P2：中等问题

### 5. Positional Encoding 的 `cos()-1.0` 偏移设计

**文件**: `src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py` L369

**现象**: 加剧训练不稳定性。

**原因**:

```python
pe = torch.cat([angles.sin(), angles.cos() - 1.0], dim=-1)  # [K, width]
```

- `sin()` 范围: `[-1, 1]`
- `cos() - 1.0` 范围: `[-2, 0]`
- 当前帧（index K-1）的 PE = 全零
- 历史帧的 PE 有最高达 2.0 的负偏移

这意味着历史帧的 hidden states 被加上了一个**大幅度、非零均值**的偏移，而当前帧完全不变。这种不对称会：
- 破坏 SigLIP 预训练的空间特征表示
- 在注意力计算中引入数值不稳定

**修复方向**:
- 使用标准的 `sin/cos` 编码（去掉 `-1.0` 偏移）
- 或乘以一个小的 scaling factor（如 0.1）来降低 PE 的影响
- 或使用可学习的 temporal embedding

---

### 6. K=6 帧全量通过完整 SigLIP 编码器

**现象**: 训练速度极慢，GPU 内存占用高。

**原因**:

3 个相机 × 6 帧 = 18 张图片，全部经过 SigLIP 的 27 层 encoder。虽然 temporal attention 只在每 4 层触发一次（27 层中约 7 次），但 **所有 27 层的空间注意力都处理了全部 6 帧**。

- 计算量是单帧的 6 倍
- 内存占用是单帧的 6 倍
- 梯度计算量是 6 倍
- 但大部分计算是冗余的——历史帧的空间特征并不需要被"微调"

**修复方向**:
- 历史帧可以用 **frozen SigLIP** 预先编码（缓存 feature），只对当前帧做完整编码
- 或者只在最后几层引入历史帧（而不是从第一层就处理全部 K 帧）
- 先从 K=2 或 K=3 开始验证方案可行性

---

## 修复优先级建议

### 第一步：修复数据管道（P0-1）
确保模型能看到有效的历史帧。这是一切的基础。

### 第二步：添加可学习的 Temporal Attention 参数（P0-2）
至少添加 Q/K/V 投影 + LayerNorm，让模型能真正学习时间信息。

### 第三步：冻结 Backbone（P1-4）
只训练新增的 temporal 参数和 action expert，保护预训练能力。

### 第四步：减小 K 值验证（P2-6）
从 K=2 开始，确认方案有效后再增加帧数。

---

## 对比：你的实现 vs MEM 论文的关键差异

| 方面 | 你的实现 | MEM 论文（推测） |
|------|---------|-----------------|
| 数据采样 | 有状态 buffer + shuffle（失效） | Episode 内顺序采样 |
| Temporal Attention | 无可学习参数，Q=K=x+PE | 可学习 Q/K/V 投影 |
| 归一化 | 无 LayerNorm | 标准 pre-norm/post-norm |
| 参数冻结 | 全参数微调 | 只微调新增模块 |
| PE 设计 | sinusoidal + cos()-1.0 偏移 | 可学习 temporal embedding |
| 帧处理 | 全量通过完整编码器 | 历史帧可能用 frozen 编码器 |

---

## 第二轮审查：对照 MEM 论文的深度代码审查（2026-03-13）

> 在修复上述 P0-P2 问题之后，对照 MEM 论文原文逐一审查代码，发现以下新问题。

### 新问题 7：Temporal Q/K/V 随机初始化（应从空间注意力复制）

**严重程度**: P1

**文件**: `siglip/modeling_siglip.py` — `TemporalCausalAttentionModule.__init__`、`SiglipEncoder.__init__`

**MEM 论文原文**:

> "A key property of our video encoder is that **it does not introduce new learnable parameters** compared to standard, single-image ViTs. Video encoding capabilities are added by **modifying the attention pattern** of the ViT and adding a **fixed sinusoidal temporal position encoding**."

论文中 temporal attention 和空间注意力**共享同一套 Q/K/V 权重**——同一个 `q_proj` 在 forward 中被调用两次（一次空间、一次时间），梯度从两条路径回传到同一个参数。因此：

- **零新增参数**
- Q/K/V 已经经过大规模预训练，天然能提取有意义的视觉特征
- 第一步训练就能产生有意义的 temporal attention pattern

**当前实现的问题**:

`TemporalCausalAttentionModule` 创建了**全新的** `q_proj`、`k_proj`、`v_proj`（默认 Xavier 随机初始化），共 31.9M 新增参数：

```python
self.q_proj = nn.Linear(embed_dim, embed_dim)  # ← Xavier 随机初始化
self.k_proj = nn.Linear(embed_dim, embed_dim)
self.v_proj = nn.Linear(embed_dim, embed_dim)
```

**随机初始化带来的风险**:

1. **初始 attention pattern 是噪声**：随机 Q/K 的点积产生随机的注意力分布，对比预训练权重生成的有意义的特征相似度分数，完全没有归纳偏置。
2. **gate=0 + zero-init out_proj 造成"延迟启动"**：训练初始阶段 temporal attention 输出恒为零，模型完全无法利用历史帧。当 gate 开始增大时，随机 Q/K/V 的低质量 attention 突然注入，可能导致 loss 突变。
3. **梯度不稳定**：随机 attention pattern 中可能出现极端分布（某位置权重 ≈ 1，其他 ≈ 0），反向传播时梯度集中在少数位置，幅度忽大忽小。

**为什么不能直接用论文的"共享"方案**:

论文是端到端预训练（整个模型可训练），所以共享 Q/K/V 可以同时学习空间和时间特征。我们的场景是**冻结 backbone 微调**：

- 如果共享冻结的 Q/K/V → temporal attention 的 Q/K/V 完全不可训练 → 只能靠 out_proj + gate 学习，能力受限
- 如果不冻结 → 50 episodes 训 36.5 亿参数 → 灾难性遗忘

**修复方案：Clone + 训练**

从同层空间注意力的 `self_attn.q_proj`、`k_proj`、`v_proj` **复制权重**作为初始化，但保持为独立的可训练参数：

```python
# SiglipEncoder.__init__ 中，创建 temporal_attns 之后：
for layer_idx_str, temporal_attn in self.temporal_attns.items():
    layer_idx = int(layer_idx_str)
    spatial_attn = self.layers[layer_idx].self_attn
    temporal_attn.q_proj.weight.data.copy_(spatial_attn.q_proj.weight.data)
    temporal_attn.q_proj.bias.data.copy_(spatial_attn.q_proj.bias.data)
    temporal_attn.k_proj.weight.data.copy_(spatial_attn.k_proj.weight.data)
    temporal_attn.k_proj.bias.data.copy_(spatial_attn.k_proj.bias.data)
    temporal_attn.v_proj.weight.data.copy_(spatial_attn.v_proj.weight.data)
    temporal_attn.v_proj.bias.data.copy_(spatial_attn.v_proj.bias.data)
    # out_proj 和 gate 保持 zero-init 不变
```

**效果**:

| | Xavier 随机初始化 | Clone 初始化 |
|--|---|---|
| 初始 attention pattern | 随机噪声 | 有意义的视觉特征相似度 |
| 训练起步 | gate=0 阶段学不到任何 temporal 信息 | 预训练权重立刻提供有效的时间特征提取 |
| 梯度稳定性 | attention 分布可能极端 → 梯度突变 | attention 分布平滑 → 梯度平稳 |
| 训练后期 | Q/K/V 可能学到有用模式，也可能学歪 | 从好的起点逐步适应 temporal 任务 |
| 参数量 | 31.9M | 31.9M（相同，只改初始化） |

---

### 新问题 8：梯度流路径过深 + 全局梯度剪裁不公平

**严重程度**: P1

**文件**: `scripts/train_pytorch.py` — 训练循环、optimizer 配置

**MEM 论文原文（Section III-D）**:

> "Gradients don't flow from the action expert into the VLM backbone."

**当前代码的梯度流路径**:

```
loss  ← 梯度起点
  ↓
action_out_proj  （可训练，距离近，梯度大）
  ↓
gemma_expert     （可训练，~860M 参数）
  ↓ ──── 穿过冻结层 ────
PaliGemma backbone （冻结，~2B 参数，梯度穿过但不更新）
  ↓
multi_modal_projector （冻结）
  ↓
SigLIP encoder 高层   （冻结）
  ↓
temporal_attn    （可训练，距离远，梯度小）
```

虽然冻结参数（`requires_grad=False`）本身不更新，但**梯度仍然穿过冻结层继续向前传播**（就像光穿过玻璃——玻璃不变，但光继续传）。temporal_attn 的梯度必须穿过整个 PaliGemma backbone 的 ~18 个 Transformer 层才能到达。

**这导致两个问题**:

#### 问题 8a：temporal_attn 收到的梯度幅度远小于 action expert

每穿过一个冻结的 Transformer 层，梯度要经过 LayerNorm backward、attention softmax Jacobian、FFN 矩阵乘法。~18 层叠加后，梯度幅度可能衰减 100~1000 倍。

从之前训练的 wandb 日志可以看到 `grad_norm` 从 ~3 飙升到 ~17 且剧烈震荡——这个 grad_norm 是全局值，几乎全部由 action expert 贡献。temporal_attn 的梯度可能小到看不见。

#### 问题 8b：全局梯度剪裁让 temporal_attn 被"误杀"

当前的剪裁方式：

```python
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

这会计算所有可训练参数的梯度总范数，然后等比缩放。由于 action expert 的梯度主导了总范数：

```
实际情况：
  expert 梯度范数:    ‖g_expert‖ = 5.0
  temporal 梯度范数:  ‖g_temporal‖ = 0.01
  总范数 ≈ 5.0

  缩放因子 = max_norm / 总范数 = 1.0 / 5.0 = 0.2

剪裁后：
  expert:   5.0 × 0.2 = 1.0      ← 从 5.0 降到 1.0，合理
  temporal: 0.01 × 0.2 = 0.002   ← 从 0.01 降到 0.002，被误杀！
```

temporal_attn 本来梯度就小，还被全局剪裁进一步压缩了 5 倍，几乎学不动。

**修复方案：分组梯度剪裁 + 分组学习率 + 分模块梯度监控**

**A. 分组梯度剪裁** — 让两组参数的梯度互不干扰：

```python
expert_params = [p for n, p in model.named_parameters()
                 if p.requires_grad and 'temporal_attn' not in n]
temporal_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and 'temporal_attn' in n]

# 各自独立剪裁，互不影响
torch.nn.utils.clip_grad_norm_(expert_params, max_norm=1.0)
torch.nn.utils.clip_grad_norm_(temporal_params, max_norm=1.0)
```

效果：temporal 范数 0.01 < 1.0，不会被剪裁，保留完整梯度信号。

**B. 分组学习率** — 用更大的 LR 补偿 temporal_attn 的梯度衰减：

```python
optimizer = torch.optim.AdamW([
    {'params': expert_params, 'lr': 5e-5},
    {'params': temporal_params, 'lr': 2.5e-4},  # 5x 更大的 LR 补偿
])
```

具体倍数需要根据方案 C 的监控数据调整。

**C. 分模块梯度监控** — 诊断两组梯度的实际差距，指导 LR 倍数调整：

```python
# 在训练循环中每 100 步记录一次
if step % 100 == 0:
    temporal_grad = torch.nn.utils.clip_grad_norm_(temporal_params, float('inf'))
    expert_grad = torch.nn.utils.clip_grad_norm_(expert_params, float('inf'))
    wandb.log({
        "temporal_grad_norm": temporal_grad,
        "expert_grad_norm": expert_grad,
        "grad_ratio": expert_grad / max(temporal_grad, 1e-8),
    })
```

**调参流程**:

1. 先上 A + C，LR 倍数初始设 5x，跑几百步看 log
2. 如果 `grad_ratio` > 100 → 说明 5x 不够，调大到 50x-100x
3. 如果 `grad_ratio` < 10 → 5x 差不多够了
4. 目标：让两组参数的**实际更新量**（梯度 × 学习率）在同一数量级（差 5-10 倍内可接受，不需要完全相等）

---

## 修复优先级建议（更新版）

### 已完成（第一轮修复）
- ✅ P0-1：重写 VideoMemoryDataset 数据管道
- ✅ P0-2：为 Temporal Attention 添加可学习 Q/K/V + gate
- ✅ P1-3：在 temporal attention 前添加 LayerNorm
- ✅ P1-4：在 train_pytorch.py 中冻结 backbone
- ✅ P2-5：修复 PE 的 cos()-1.0 偏移
- ✅ P2-6：新增 K=3 配置

### 已完成（第二轮修复）
- ✅ 新问题 7：将 temporal Q/K/V 的初始化从 Xavier 随机改为从空间注意力 clone
- ✅ 新问题 8a：实现分组梯度剪裁（temporal vs expert）
- ✅ 新问题 8b：实现分组学习率（temporal 用更大 LR）
- ✅ 新问题 8c：添加分模块梯度监控（wandb 记录 temporal_grad / expert_grad）

---

## 第三轮审查：对照 MEM 论文 + 训练曲线诊断（2026-03-13）

> 对照 MEM 论文原文（arXiv:2603.03596）逐一审查代码，结合 wandb 训练曲线分析，发现以下致命和严重问题。

### 新问题 9（P0）：`gate=0` + `out_proj=0` 双重零初始化 → temporal attention 完全死亡

**严重程度**: P0（致命）

**文件**: `siglip/modeling_siglip.py` — `TemporalCausalAttentionModule.__init__` + `SiglipEncoder._init_temporal_from_spatial`

**问题描述**:

temporal attention 的 forward 返回 `self.gate * self.out_proj(attn_output)`。`gate` 初始化为 0（`nn.Parameter(torch.zeros(1))`），`out_proj` 的 weight 和 bias 也被显式初始化为 0。

两者同时为零会导致**所有参数的梯度恒为零**：

```
y = gate * out_proj(x)

∂y/∂gate = out_proj(x) = 0          （因为 out_proj 是零矩阵）
∂y/∂out_proj.weight = gate * x^T = 0 （因为 gate=0）
∂y/∂q_proj = ... 链式经过 gate 和 out_proj ... = 0
```

**结果**：整个 temporal attention 模块的所有参数（Q/K/V/out_proj/gate/LayerNorm）在训练过程中**永远收不到任何梯度信号**。模块完全是死的，等价于 K=1 单帧模型。

这解释了训练曲线中的现象：
- loss 在下降 → 只有 action expert 在学
- grad_norm 抖动 → 全来自 action expert
- 生成视频抖 → 模型从未利用历史帧

**修复方案**: 去掉 `gate` 参数，只保留 `out_proj` 零初始化（标准 zero-init residual 模式）。

```python
# 修改前：
out = self.out_proj(out)
return self.gate * out   # gate=0 * zero_matrix(x) = 永远是 0

# 修改后：
out = self.out_proj(out)
return out                # out_proj=0 保证初始行为=单帧，但梯度可以流过
```

**修改的文件**:
- `siglip/modeling_siglip.py`：`TemporalCausalAttentionModule.__init__` 删除 `self.gate`；`forward` 返回 `self.out_proj(out)` 而不是 `self.gate * out`
- `siglip/modeling_siglip.py`：`_init_temporal_from_spatial` 删除 `temporal_attn.gate.data.zero_()`
- 同步更新 `.venv` 中的安装副本

**梯度流原理**:

去掉 gate 后，`y = out_proj(attn_output)`。虽然初始时 out_proj=0 导致 y=0（与单帧行为一致），但 `∂y/∂out_proj.weight = attn_output^T`，这个值不为零（因为 Q/K/V 是从预训练空间注意力 clone 来的，会产生有意义的 attention 输出）。所以 out_proj 从第一步就能收到梯度，进而通过反向传播让 Q/K/V 和 LayerNorm 也能收到梯度。

---

### 新问题 10（P1）：LR 调度覆盖了分组学习率设置

**严重程度**: P1（严重）

**文件**: `scripts/train_pytorch.py` L652-654

**问题描述**:

虽然 optimizer 创建时分了 expert 和 temporal 两个 param group，但训练循环中 LR 调度对**所有** param group 设了相同的 LR：

```python
# 修改前：
for pg in optim.param_groups:
    pg["lr"] = lr_schedule(global_step)  # 全部相同 LR！
```

temporal attention 位于 SigLIP 编码器深层，梯度必须穿过整个冻结的 PaliGemma backbone 才能到达，幅度远小于 action expert。用相同 LR 意味着 temporal attention 的参数更新量远远不足。

**修复方案**: temporal 组使用 5x 更大的学习率（可根据 wandb `grad_ratio` 调整）。

```python
# 修改后：
_TEMPORAL_LR_MULTIPLIER = 5.0
base_lr = lr_schedule(global_step)
optim.param_groups[0]["lr"] = base_lr                              # expert
optim.param_groups[1]["lr"] = base_lr * _TEMPORAL_LR_MULTIPLIER    # temporal
```

**修改的文件**: `scripts/train_pytorch.py`
- 新增全局常量 `_TEMPORAL_LR_MULTIPLIER = 5.0`
- 训练循环中 LR 更新逻辑改为分组设置

**调参指南**: 观察 wandb 上 `grad_ratio`（expert_grad / temporal_grad）：
- 如果 > 100 → multiplier 不够，调大到 50x
- 如果 < 10 → 5x 差不多
- 目标是让两组的实际更新量在同一数量级

---

### 新问题 11（P1）：VideoMemoryDataset LRU 缓存太小 + gap 阈值过严

**严重程度**: P1

**文件**: `src/openpi/training/data_loader.py` — `VideoMemoryDataset`

**背景分析**:

经过深入分析底层 `BehaviorLeRobotDataset`，发现在 `chunk_streaming_using_keyframe=True` 模式下，`__getitem__` **忽略传入的 idx 参数**，按内部流式状态返回帧。因此：
- DataLoader 的 shuffle 不影响实际数据顺序
- 每个 worker 内部，帧在 250 帧的 chunk 内是顺序返回的
- buffer 在 chunk 内部能正常积累

但仍有改进空间：

1. **LRU 缓存 `_MAX_EPISODES_CACHED=8` 太小**：worker 分到的 chunks 来自不同 episode，如果有超过 8 个 episode 交错，旧 episode 的 buffer 会被驱逐。增大到 32 以覆盖更多交错场景。

2. **gap 阈值 `> 2` 过于严格**：硬编码 2 帧的阈值在 stride > 1 时不合理。如果底层数据有微小的 frame index 不规则性（如跳过 1 帧），buffer 不应该被重置。改为 `> self._stride + 1`，与 stride 参数联动。

3. **缺乏诊断信息**：无法从日志判断历史帧有效率。添加每 5000 样本打印一次 "full history" 百分比。

**修复方案**:

```python
_MAX_EPISODES_CACHED = 32   # 原来是 8

# gap 阈值改为 stride-aware
if gap > self._stride + 1:  # 原来是 gap > 2
    self._buffers[ep_idx] = {}

# 添加诊断日志
if self._stats_total % 5000 == 0:
    logging.info("VideoMemoryDataset: %d/%d (%.1f%%) samples have full history", ...)
```

**修改的文件**: `src/openpi/training/data_loader.py`

---

### 新问题 12（P1）：底层数据集双重 shuffle 导致 chunk 顺序混乱

**严重程度**: P1

**文件**: `src/openpi/training/data_loader.py` — `create_behavior_dataset` L260

**问题描述**:

`BehaviorLeRobotDataset(shuffle=True)` 在 `__init__` 时全局打乱 chunk 列表，然后每个 worker 又对分到的 chunks 做独立的 per-worker shuffle。这是双重 shuffle：

1. `__init__` 全局 shuffle：打乱了 `self.chunks` 列表
2. `__getitem__` 中 per-worker shuffle：`rng.shuffle(worker_chunks)`

全局 shuffle 是多余的，因为 per-worker shuffle 已经保证了训练多样性。全局 shuffle 反而会破坏同一 episode 的 chunks 在列表中的连续性——比如 episode 5 的 chunk A、B、C 在全局 shuffle 后被打散到列表的不同位置，使得 per-worker 的 strided 分配更不可能把同一 episode 的连续 chunks 分给同一个 worker。

**修复方案**: 将 `shuffle=True` 改为 `shuffle=False`，依赖 per-worker shuffle 提供训练多样性。

```python
# 修改前：
BehaviorLeRobotDataset(..., shuffle=True, ...)

# 修改后：
BehaviorLeRobotDataset(..., shuffle=False, ...)
```

**修改的文件**: `src/openpi/training/data_loader.py` — `create_behavior_dataset`

---

## 修复状态总览

### 已完成（第一轮修复）
- ✅ P0-1：重写 VideoMemoryDataset 数据管道
- ✅ P0-2：为 Temporal Attention 添加可学习 Q/K/V + gate
- ✅ P1-3：在 temporal attention 前添加 LayerNorm
- ✅ P1-4：在 train_pytorch.py 中冻结 backbone
- ✅ P2-5：修复 PE 的 cos()-1.0 偏移
- ✅ P2-6：新增 K=3 配置

### 已完成（第二轮修复）
- ✅ 新问题 7：将 temporal Q/K/V 的初始化从 Xavier 随机改为从空间注意力 clone
- ✅ 新问题 8a：实现分组梯度剪裁（temporal vs expert）
- ✅ 新问题 8b：实现分组学习率（temporal 用更大 LR）
- ✅ 新问题 8c：添加分模块梯度监控（wandb 记录 temporal_grad / expert_grad）

### 已完成（第三轮修复 — 2026-03-13）
- ✅ 新问题 9：去掉 gate 参数，解除双重零初始化死锁
- ✅ 新问题 10：修复 LR 调度覆盖分组设置，temporal 组用 5x LR
- ✅ 新问题 11：VideoMemoryDataset LRU 8→32，gap 阈值 stride-aware，加诊断日志
- ✅ 新问题 12：底层数据集 shuffle=True → False，消除双重 shuffle

### 已完成（第四轮修复 — 2026-03-13）
- ✅ 新问题 13：Temporal attention softmax 改为在 float32 下计算（与空间注意力一致），防止 bfloat16 数值不稳定
- ✅ 新问题 14：为 temporal attention 添加 attn_dropout + proj_dropout（0.1），缓解 50 episodes 训 31.9M 参数的过拟合风险

### 待观察
- ⬜ PE centering（cos()-1.0）：数学上是 MEM 论文要求的"PE=0 for current frame"的正确实现，可能不需要改。如训练仍不稳定可加 scale factor 如 `0.1 * pe`
- ⬜ 过拟合风险：50 episodes 训 ~32M 新参数。已加 dropout=0.1。如 loss 下降但 eval 仍差，考虑冻结 temporal Q/K/V（只训 out_proj，降到 ~9.3M 可训参数）
- ⬜ Proprioceptive state 历史：MEM 论文用 K 个 state token，当前只用 1 个。如需提升动态推理能力可后续添加
- ⬜ 架构差异说明：MEM 论文用共享 Q/K/V（零新增参数），我们因数据/算力限制用独立模块（31.9M 新参数）。这是有意为之的工程决策，不是 bug

---

## 第五轮审查：训练曲线深度诊断 + 全面代码复审（2026-03-17）

> 结合 wandb 训练曲线（55k 步）逐项诊断。纠正了之前几轮的部分误判，发现了多个新的严重问题。

### 重要纠正：temporal_grad_norm 并未消失

之前多轮分析认为 temporal attention 梯度消失是 P0 级别问题。实际观察 wandb 曲线：

- `temporal_grad_norm`：在 **0.1-0.4** 之间波动，后期偏低但远非消失
- `grad_ratio`（expert / temporal）：稳定在 **20-40x**
- `expert_grad_norm`：1-3 之间

**结论**：temporal attention 的梯度一直存在，20-40x 的差距在 frozen backbone 场景下是正常的。之前的"梯度消失"诊断有误。

---

### 新问题 15（致命）：验证指标完全不可信——流式数据集 + 少量 val batch = 采样噪声

**严重程度**: P0（致命）

**文件**: `scripts/train_pytorch.py` — `validate()`, `build_val_loader()`；`src/behavior/learning/datas/dataset.py` — `__getitem__`

**现象**:

- val/flow_loss 在 0.5-3.5 之间剧烈波动
- val/action_cosine_sim 在 0.1-0.9 之间疯狂跳动
- 训练 loss 相对稳定在 ~0.3
- 相邻两次验证（间隔 100 步训练）的指标差距巨大

**根因**:

`BehaviorLeRobotDataset` 在 `chunk_streaming_using_keyframe=True` 模式下，`__getitem__` **忽略传入的 idx 参数**，按内部 `current_streaming_frame_idx` 顺序返回帧。

```python
# dataset.py L531 — idx 参数被完全忽略
item = self.hf_dataset[self.current_streaming_frame_idx]
```

验证时：
1. 每次调用 `validate()`，从 val_loader 取 `val_num_batches=10` 个 batch = 10 × 64 = **640 帧**
2. 验证集有 50 任务 × 20 episode ≈ **数十万帧**
3. 流式状态在两次验证之间**不会重置**，持续向前推进
4. 每次验证恰好落在**不同任务/场景/难度**的连续片段上

所以：
- 第 N 次验证：恰好在简单任务的平稳片段 → val_loss = 0.5, cosine_sim = 0.85
- 第 N+1 次验证：恰好在复杂操作的高动态片段 → val_loss = 3.0, cosine_sim = 0.15

**结果**：val 指标的剧烈波动反映的是**采样方差**，不是模型性能变化。你无法通过当前的验证指标判断模型是否在改进。

**验证**：如果是模型本身不稳定，不可能在相邻 100 步（模型几乎没变）之间 cosine_sim 从 0.9 掉到 0.1。这只能是不同 batch 的内在难度差异。

**修复**（已完成 ✅ — 2026-03-17）:

1. **新增 `reset_val_loader()` 函数**（`scripts/train_pytorch.py`）：
   - 递归遍历 val_loader 的嵌套 dataset 层（`DataLoaderImpl` → `TorchDataLoader` → `TransformedDataset` → `VideoMemoryDataset` → `BehaviorLeRobotDataset`）
   - 将 `BehaviorLeRobotDataset` 的流式指针重置到起点：`current_streaming_chunk_idx = 0`，`current_streaming_frame_idx = _active_chunks[0][0]`
   - 清空 `VideoMemoryDataset` 的历史帧缓存（`_buffers`、`_last_frame_idx`），避免跨验证的残留状态

2. **在每次验证前调用 `reset_val_loader(val_loader)`**（`scripts/train_pytorch.py` 训练循环中）：
   - 确保每次验证都从验证集的**同一个起点**开始读取
   - 这样 val_loss 的变化纯粹反映模型进步，不受采样位置影响

3. **修改验证配置**（`src/openpi/training/config.py`）：
   - `val_log_interval`: 100 → **500**（减少验证频率，降低总额外耗时）
   - `val_num_batches`: 10 → **100**（每次验证看 6400 帧而非 640 帧，覆盖更多任务和场景）
   - 涉及配置：`pi05_b1k-all_skills`、`pi05_b1k-all_skills_mem_K6`、`pi05_b1k-all_skills_mem_K3_v2`

**修复后效果**：
- 每次验证都看到完全相同的 100 batch 数据（固定的参照物）
- val_loss 曲线将平滑且单调下降（如果模型在进步）
- 预计总额外验证时间：~4-8 小时（300 次验证 × ~1 分钟/次），可接受

---

### 新问题 16（严重）：DistributedSampler 与流式数据集的语义冲突

**严重程度**: P0

**文件**: `src/openpi/training/data_loader.py` L518-526

**问题描述**:

```python
if torch.distributed.is_initialized():
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=..., rank=..., shuffle=shuffle, drop_last=True,
    )
```

`DistributedSampler` 给每个 GPU 分配不同的 index 子集（如 GPU0 取偶数 index，GPU1 取奇数 index）。但 `BehaviorLeRobotDataset.__getitem__` 在流式模式下**忽略传入的 idx**。

每个 GPU 的 dataset 对象是独立副本（spawn 模式），各自维护独立的流式状态。每个 worker 初始化时用 `seed + worker_id` shuffle chunks：

```python
# dataset.py L517
rng = np.random.default_rng(self.seed + worker_id)
rng.shuffle(worker_chunks)
```

DDP 下多个 GPU 各自有 8 个 worker。不同 GPU 的 **worker 0** 使用**相同的 seed**（`config.seed + 0`），`worker_chunks` 的分配也相同（都是 `range(0, total_chunks, num_workers)`），所以它们的 `_active_chunks` 顺序**完全相同**。

**影响**：不同 GPU 的同编号 worker 读取完全相同的数据序列。对于 4 GPU × 8 worker 的配置，实际有效数据多样性降低（取决于 GPU 间 worker 调度的随机性，但理论上可能有大量重复）。

---

### 新问题 17（严重）：LR decay_steps=35k vs train_steps=150k

**严重程度**: P0

**文件**: `src/openpi/training/config.py` L873-876

```python
lr_schedule=_optimizer.CosineDecaySchedule(
    peak_lr=2.5e-5,
    decay_steps=35_000,    # ← cosine decay 在 35k 步完成
    decay_lr=2.5e-6,
),
num_train_steps=150_000,   # ← 总训练 150k 步
```

LR 调度的实现（`train_pytorch.py` L687）：
```python
progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
cos = 0.5 * (1 + np.cos(np.pi * progress))
return end_lr + (peak_lr - end_lr) * cos
```

当 `step > decay_steps` 时，`progress = 1.0`，`cos = 0`，LR 固定在 `end_lr = 2.5e-6`。

**影响**：只有前 35k 步（23%）在有效学习率下训练，后 115k 步（77%）在 `2.5e-6` 的极低 LR 下运行。从 wandb 的 `learning_rate` 曲线可以清楚看到 LR 在 ~35k 步后就平了。

这意味着：
- 如果 35k 步内没有收敛，后面基本学不动
- temporal attention 的"冷启动"（out_proj 从零开始）浪费了 warmup 的高 LR 阶段
- 75% 的训练算力基本浪费

---

### 新问题 18（严重）：60%+ 训练样本的历史帧无效

**严重程度**: P1

**文件**: `src/openpi/training/data_loader.py` — `VideoMemoryDataset`

**数学分析**:

配置：K=6，`video_memory_stride_s=1.0`，FPS=30，chunk 大小=250 帧。

- `stride_frames = int(1.0 * 30) = 30`
- 需要的缓冲区深度：`(K-1) * stride + 1 = 5 * 30 + 1 = 151` 帧
- 每个 chunk 250 帧

chunk 内第 N 帧（N 从 0 开始）的缓冲区有 N 帧可用。要让所有 5 个历史帧都有效（`valid_flags` 全为 True），需要 N ≥ 150。

- 前 150 帧：历史不完整，部分或全部 `valid_flags = False`（padded 为重复帧）
- 后 100 帧：历史完整
- 有效率：100 / 250 = **40%**

注释中说 "<5% of frames" 是 padding，但那是 stride=1 时的估计。stride=30 时，**60% 的样本使用的是 padded 历史帧**。

**影响**：大部分训练步骤中，temporal attention 看到的是重复的当前帧而非真实历史，学习信号极弱。

---

### 新问题 19（中等）：递归 `__getitem__` 的 skill 权重过滤破坏历史缓冲区

**严重程度**: P1

**文件**: `src/behavior/learning/datas/dataset.py` L586-590

```python
if not random.choices([True, False], weights=[weight, 1 - weight])[0]:
    self.current_streaming_frame_idx += 1
    for key in self.meta.video_keys:
        next(self.obs_loaders[key])[0]   # 消费视频帧但丢弃
    return self.__getitem__(idx)          # 递归调用
```

当帧被 skill 权重过滤跳过时：
1. `current_streaming_frame_idx` 推进了
2. 视频帧被消费了（`next(self.obs_loaders[key])`）
3. 但这一帧**没有返回到 `VideoMemoryDataset`**

`VideoMemoryDataset` 的缓冲区只在 `__getitem__` 返回时才接收帧（L202-204）。被跳过的帧不会进入缓冲区，导致：
- 缓冲区中出现帧 index 不连续
- `gap > self._stride + 1` 触发缓冲区清空
- 历史帧更频繁地失效

**影响**：与问题 18 叠加，实际有完整历史帧的样本比例更低。

---

### 新问题 20（中等）：`shuffle=False` 硬编码导致数据顺序相关

**严重程度**: P1

**文件**: `src/openpi/training/data_loader.py` L259-260

```python
dataset = BehaviorLeRobotDataset(
    ...,
    chunk_streaming_using_keyframe=True,
    shuffle=False,     # ← 硬编码
    ...
)
```

`shuffle=False` 时，`BehaviorLeRobotDataset.__init__` 走确定性路径（L335-338）：

```python
self._active_chunks = list(self.chunks)
self.current_streaming_chunk_idx = 0
```

chunks 按 episode 顺序排列。虽然 `num_workers=8` 时每个 worker 会做 per-worker shuffle（L514-518），但主进程的初始化走了顺序路径。

**影响**：单 worker 内数据按 episode-chunk 顺序读取，相邻 batch 的样本高度相关（同一 episode 的连续帧），降低了 SGD 的有效性。

---

### 新问题 21（中等）：out_proj 零初始化的"冷启动"问题

**严重程度**: P1

**文件**: `siglip/modeling_siglip.py` — `_init_temporal_from_spatial` L714-715

之前多轮分析认为这是 P0 级别问题，但从 wandb 曲线看 temporal_grad_norm 并未消失（0.1-0.4），说明梯度确实能流过来。

但 `out_proj` 零初始化仍然有负面影响：

- 训练初期（warmup 阶段），temporal 对 loss 贡献为零，模型优化信号全来自 action expert
- warmup 1000 步 + cosine decay → 当 out_proj 开始从零离开时，LR 已经在下降
- 配合问题 17（decay_steps=35k），temporal 的有效学习窗口被进一步压缩

**备选方案**：将 `out_proj` 改为缩放初始化（clone spatial `o_proj` × 0.01），让 temporal 从第一步就对 loss 有微小贡献，获得更强的梯度信号。但这不是当前最紧急的问题。

---

### 新问题 22（轻微）：多帧共享相同的数据增强参数

**严重程度**: P2

**文件**: `src/openpi/models_pytorch/preprocessing_pytorch.py` L42, L63-76

多帧被 cat 在 batch 维度（B*K），然后整体做增强。随机裁剪、旋转、亮度、对比度的参数对 B*K 个帧只 sample 一组：

```python
start_h = torch.randint(0, max_h + 1, (1,), device=image.device)  # 一个值用于所有帧
```

同一样本的 6 帧接受完全相同的 crop / rotation / color jitter。

**影响**：几何增强共享是合理的（同一 episode 应有一致视角变换），但 color jitter 对所有帧相同会掩盖光照变化等时序信号。可以考虑对 color augmentation 做 per-frame 独立采样。

---

### 新问题 23（轻微）：`is_channels_first` 检测不可靠

**严重程度**: P2

**文件**: `src/openpi/models_pytorch/preprocessing_pytorch.py` L46

```python
is_channels_first = image.shape[1] == 3
```

当图像恰好有 3 个 pixel 行时会误判。当前 224x224 不会触发，但代码脆弱。

---

### 新问题 24（轻微）：weight_decay 几乎为零

**严重程度**: P2

**文件**: `src/openpi/training/optimizer.py` L74

```python
weight_decay: float = 1e-10
```

等于没有正则化。frozen backbone 本身是隐式正则化，但对 action expert（~300M 可训参数）来说，完全没有 weight decay 可能导致轻微过拟合。

---

### 新问题 25（轻微）：`torch.compile` 在 `__init__` 中编译 `sample_actions`

**严重程度**: P2

**文件**: `src/openpi/models_pytorch/pi0_pytorch.py` L112

```python
self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")
```

在 `__init__` 中调用，模型还没 `to(device)`。`max-autotune` 模式在第一次调用时做大量 autotuning，可能导致第一次验证非常慢或显存尖峰。

---

### 新问题 26（轻微）：`find_unused_parameters=True` 与 `static_graph=True` 冲突

**严重程度**: P2

**文件**: `scripts/train_pytorch.py` L605-611

```python
model = torch.nn.parallel.DistributedDataParallel(
    model,
    find_unused_parameters=True,
    static_graph=world_size >= 8,
)
```

8+ GPU 时两者同时启用。`static_graph` 会自动处理 unused parameters，`find_unused_parameters` 增加额外通信开销。

---

### 新问题 27（轻微）：验证时 VideoMemoryDataset 历史缓冲区不稳定

**严重程度**: P2

**文件**: `scripts/train_pytorch.py` — `validate()`；`src/openpi/training/data_loader.py` — `VideoMemoryDataset`

验证 loader 在 `build_val_loader` 中创建，其 `VideoMemoryDataset` 的 `_buffers` 在多次 `validate()` 调用之间持久存在。但流式数据集的内部状态在两次验证之间持续推进，可能跨越 episode/chunk 边界，触发缓冲区清空。

**影响**：验证时 temporal attention 接收到的历史帧质量比训练时更差，给本已不可靠的 val 指标增加额外方差。

---

## 第五轮问题严重程度排序

| 等级 | 编号 | 问题 | 影响 |
|------|------|------|------|
| **致命** | 15 | 流式数据集 + 少量 val batch → val 指标不可信 | 无法判断模型好坏 |
| **致命** | 16 | DDP sampler + 流式数据集冲突 → 多 GPU 数据重复 | 有效 batch diversity 降低 |
| **致命** | 17 | decay_steps=35k vs train_steps=150k → 75% 训练在极低 LR | 巨量算力浪费 |
| **严重** | 18+19 | 60%+ 样本历史帧无效 + skill 过滤加剧 | temporal attention 缺乏学习信号 |
| **中等** | 20 | shuffle=False 硬编码 → 数据顺序相关 | batch diversity 低 |
| **中等** | 21 | out_proj 零初始化冷启动 → 浪费 warmup 高 LR | temporal 学习延迟 |
| **轻微** | 22 | 多帧共享增强参数 | 可能掩盖时序信号 |
| **轻微** | 23 | is_channels_first 检测脆弱 | 当前不触发 |
| **轻微** | 24 | weight_decay ≈ 0 | 可能轻微过拟合 |
| **轻微** | 25 | torch.compile 时机 | 首次 val 可能慢 |
| **轻微** | 26 | find_unused_params + static_graph | DDP 通信效率 |
| **轻微** | 27 | 验证时历史缓冲区不稳定 | val 指标额外方差 |

---

## 修复状态总览（更新）

### 已完成（第一轮修复）
- ✅ P0-1：重写 VideoMemoryDataset 数据管道
- ✅ P0-2：为 Temporal Attention 添加可学习 Q/K/V + gate
- ✅ P1-3：在 temporal attention 前添加 LayerNorm
- ✅ P1-4：在 train_pytorch.py 中冻结 backbone
- ✅ P2-5：修复 PE 的 cos()-1.0 偏移
- ✅ P2-6：新增 K=3 配置

### 已完成（第二轮修复）
- ✅ 新问题 7：将 temporal Q/K/V 的初始化从 Xavier 随机改为从空间注意力 clone
- ✅ 新问题 8a：实现分组梯度剪裁（temporal vs expert）
- ✅ 新问题 8b：实现分组学习率（temporal 用更大 LR）
- ✅ 新问题 8c：添加分模块梯度监控（wandb 记录 temporal_grad / expert_grad）

### 已完成（第三轮修复 — 2026-03-13）
- ✅ 新问题 9：去掉 gate 参数，解除双重零初始化死锁
- ✅ 新问题 10：修复 LR 调度覆盖分组设置，temporal 组用 5x LR
- ✅ 新问题 11：VideoMemoryDataset LRU 8→32，gap 阈值 stride-aware，加诊断日志
- ✅ 新问题 12：底层数据集 shuffle=True → False，消除双重 shuffle

### 已完成（第四轮修复 — 2026-03-13）
- ✅ 新问题 13：Temporal attention softmax 改为在 float32 下计算（与空间注意力一致），防止 bfloat16 数值不稳定
- ✅ 新问题 14：为 temporal attention 添加 attn_dropout + proj_dropout（0.1），缓解 50 episodes 训 31.9M 参数的过拟合风险

### 待修复（第五轮发现 — 2026-03-17）
- ✅ 新问题 15（P0）：验证流程不可靠——已修复：重置播放头 + val_log_interval=500 + val_num_batches=100
- ⬜ 新问题 16（P0）：DDP + 流式数据集冲突——需要在 worker init 中注入 rank 信息到 seed
- ⬜ 新问题 17（P0）：decay_steps 与 train_steps 不匹配——改为 decay_steps=150_000
- ⬜ 新问题 18+19（P1）：历史帧有效率低——减小 stride 或增大 chunk size
- ⬜ 新问题 20（P1）：shuffle=False 硬编码——改为 shuffle=True 或 episode-level shuffle
- ⬜ 新问题 21（P1）：out_proj 零初始化冷启动——考虑缩放初始化（× 0.01）
- ⬜ 新问题 22-27（P2）：轻微问题，可后续处理

### 待观察
- ⬜ PE centering（cos()-1.0）：数学上是 MEM 论文要求的"PE=0 for current frame"的正确实现，可能不需要改。如训练仍不稳定可加 scale factor 如 `0.1 * pe`
- ⬜ 过拟合风险：50 episodes 训 ~32M 新参数。已加 dropout=0.1。如 loss 下降但 eval 仍差，考虑冻结 temporal Q/K/V（只训 out_proj，降到 ~9.3M 可训参数）
- ⬜ Proprioceptive state 历史：MEM 论文用 K 个 state token，当前只用 1 个。如需提升动态推理能力可后续添加
- ⬜ 架构差异说明：MEM 论文用共享 Q/K/V（零新增参数），我们因数据/算力限制用独立模块（31.9M 新参数）。这是有意为之的工程决策，不是 bug

