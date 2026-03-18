# Temporal Attention 实现分析：我们的实现 vs MEM 论文原始设计

> 本文档对比我们当前 `TemporalCausalAttentionModule` 的实现与 MEM 论文（PI π0.6-MEM）的原始设计，
> 分析训练中观察到的问题（temporal 梯度消失、验证 loss 飙升）的根本原因。

---

## 1. MEM 论文的设计：零参数 temporal attention

### 1.1 论文原文关键表述

**来源：** [MEM: Multi-Scale Embodied Memory for Vision Language Action Models](https://www.pi.website/download/Mem.pdf)

#### 表述 1 — 不引入新参数（Section III-C）

> "A key property of our video encoder is that **it does not introduce new learnable parameters**
> compared to standard, single-image ViTs. Video encoding capabilities are added by
> **modifying the attention pattern** of the ViT and adding a **fixed sinusoidal temporal
> position encoding**."

#### 表述 2 — 加法式叠加（Section III-C）

> "Every 4-th layer **additively adds** attention over the time dimension by performing
> attention over the timesteps' representations for the same image patch using a causal
> attention mask."

#### 表述 3 — K=1 完全匹配原始 VLM（Section III-C）

> "To maximize feature transfer, we ensure that for K=1, i.e., single-image input,
> our encoder's initialization **exactly matches** that of the VLM, which we achieve with
> a sinusoidal temporal position embedding that has value 0 for t=0."

#### 表述 4 — 预训练权重可直接加载（Section III-C）

> "Thus, we can initialize the weights of our video encoder from the pre-trained ViT
> weights of any standard vision-language-models, just like in no-memory VLAs."

### 1.2 论文的实现方式（伪代码）

根据论文 Section III-C 和 Appendix C，temporal attention 的完整计算流程：

```python
def temporal_attention(hidden_states, num_frames, num_heads):
    """
    MEM 论文的 temporal attention — 纯计算操作，零新参数。
    
    输入: hidden_states [B*K, n, d] — spatial attention 输出后的特征
    输出: residual [B*K, n, d] — 直接加回 hidden_states
    """
    BK, n, d = hidden_states.shape
    B = BK // num_frames
    K = num_frames
    head_dim = d // num_heads

    # Reshape 为 [B, K, n, d]
    x = hidden_states.reshape(B, K, n, d)

    # ① Q = hidden_states + 固定 sinusoidal temporal PE
    #   （注意：不经过任何线性投影层）
    temporal_pe = sinusoidal_pe(K, d)           # [K, d]，固定，不可训练
    Q = x + temporal_pe[None, :, None, :]       # [B, K, n, d]

    # ② K = hidden_states + 固定 sinusoidal temporal PE
    K_tensor = Q  # 和 Q 一样

    # ③ V = hidden_states（不加 PE）
    V = x

    # ④ 转置：把 patch 维度并入 batch，在时间维度做 attention
    #    [B, K, n, d] → [B, n, K, d] → [B*n, K, num_heads, head_dim]
    Q = Q.permute(0, 2, 1, 3).reshape(B*n, K, num_heads, head_dim)
    K_tensor = K_tensor.permute(0, 2, 1, 3).reshape(B*n, K, num_heads, head_dim)
    V = V.permute(0, 2, 1, 3).reshape(B*n, K, num_heads, head_dim)

    # ⑤ 标准 scaled dot-product attention + causal mask
    attn_weights = (Q @ K_tensor.T) / sqrt(head_dim)
    attn_weights = apply_causal_mask(attn_weights)  # 下三角
    attn_weights = softmax(attn_weights)

    # ⑥ 加权求和
    out = attn_weights @ V

    # ⑦ Reshape 回 [B*K, n, d]
    out = out.reshape(B, n, K, d).permute(0, 2, 1, 3).reshape(BK, n, d)

    # ⑧ 直接返回（没有 out_proj、没有 LayerNorm、没有 Dropout）
    return out
```

### 1.3 在 ViT 中的集成方式

```python
# 每一层 ViT encoder layer 的 forward:
def encoder_layer_forward(hidden_states, layer_idx, num_frames):
    # ① 标准 spatial attention（不变）
    residual = hidden_states
    h = layer_norm1(hidden_states)
    h = spatial_self_attention(h)
    hidden_states = residual + h

    # ② 每 4 层加一次 temporal attention（纯计算）
    if num_frames > 1 and (layer_idx + 1) % 4 == 0:
        temporal_residual = temporal_attention(hidden_states, num_frames, num_heads)
        hidden_states = hidden_states + temporal_residual    # additive

    # ③ 标准 FFN（不变）
    residual = hidden_states
    h = layer_norm2(hidden_states)
    h = mlp(h)
    hidden_states = residual + h

    return hidden_states
```

### 1.4 关键特性总结

| 特性 | 说明 |
|------|------|
| **新参数数量** | **0**（零） |
| **Q/K 来源** | hidden_states 本身 + 固定 sinusoidal PE |
| **V 来源** | hidden_states 本身 |
| **线性投影** | 无（不经过 Q/K/V proj，不经过 out_proj） |
| **LayerNorm** | 无 |
| **Dropout** | 无 |
| **输出方式** | additive residual，直接加回 |
| **K=1 行为** | 不执行，完全退化为原始 ViT |

---

## 2. 我们的实现：带可学习参数的 temporal attention

### 2.1 当前代码

文件：`src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py`

```python
class TemporalCausalAttentionModule(nn.Module):
    def __init__(self, config, spatial_attn):
        super().__init__()
        embed_dim = config.hidden_size  # 1152

        # ❌ 新参数 1: LayerNorm（可训练，2 × 1152 = 2304 参数）
        self.layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        # 共享冻结的 spatial Q/K/V 线性投影
        self._shared_qkv = (spatial_attn.q_proj, spatial_attn.k_proj, spatial_attn.v_proj)

        # ❌ 新参数 2: out_proj（可训练，1152×1152 + 1152 ≈ 133 万参数）
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # ❌ 新参数 3: Dropout（不是参数但改变了前向行为）
        self.attn_dropout = nn.Dropout(p=0.1)
        self.proj_dropout = nn.Dropout(p=0.1)

        # out_proj 初始化为零
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x, num_frames, temporal_mask=None):
        if num_frames <= 1:
            return torch.zeros_like(x)

        BK, n, d = x.shape
        B = BK // num_frames
        K = num_frames

        # ❌ 差异 1: 先过 LayerNorm
        h = self.layer_norm(x)

        # ❌ 差异 2: 通过冻结的 Q/K/V 线性投影层
        shared_q, shared_k, shared_v = self._shared_qkv
        q = shared_q(h)   # h → 冻结的 Linear → q
        k = shared_k(h)   # h → 冻结的 Linear → k
        v = shared_v(h)   # h → 冻结的 Linear → v

        # 加 temporal PE（这部分和论文一致）
        pe = temporal_posemb_sincos(K, d, device=x.device)
        q = q + pe
        k = k + pe

        # attention 计算（这部分和论文一致）
        # ... scaled dot-product attention + causal mask ...

        # ❌ 差异 3: 输出经过 out_proj 线性层
        out = self.out_proj(out)

        # ❌ 差异 4: 输出经过 Dropout
        out = self.proj_dropout(out)

        return out
```

### 2.2 每层新增的参数统计

SigLIP So400m/14 的 `hidden_size = 1152`，temporal attention 每 4 层加一次，
27 层中有 **6 层**（layer 3, 7, 11, 15, 19, 23）带 temporal attention：

| 参数 | 每层 | 6 层总计 |
|------|------|----------|
| `layer_norm.weight` | 1,152 | 6,912 |
| `layer_norm.bias` | 1,152 | 6,912 |
| `out_proj.weight` | 1,152 × 1,152 = 1,327,104 | 7,962,624 |
| `out_proj.bias` | 1,152 | 6,912 |
| **合计** | **1,330,560** | **~800 万参数** |

---

## 3. 逐项对比

| 设计点 | MEM 论文 | 我们的实现 | 差异严重性 |
|--------|----------|------------|------------|
| **新可学习参数** | 0 | ~800 万 | 🔴 核心差异 |
| **Q/K 来源** | hidden_states 直接用 | hidden_states → LayerNorm → 冻结 Q/K proj | 🔴 多了两步变换 |
| **V 来源** | hidden_states 直接用 | hidden_states → LayerNorm → 冻结 V proj | 🔴 多了两步变换 |
| **LayerNorm** | 无 | 有（可训练） | 🟡 |
| **out_proj** | 无 | 有（可训练，初始化为零） | 🔴 梯度瓶颈 |
| **Dropout** | 无 | attn 0.1 + proj 0.1 | 🟡 |
| **temporal PE** | 固定 sinusoidal | 固定 sinusoidal | ✅ 一致 |
| **causal mask** | 下三角 | 上三角取反（等价） | ✅ 一致 |
| **additive residual** | 是 | 是 | ✅ 一致 |
| **K=1 退化** | 不执行，完全匹配原始 VLM | 返回 zeros_like | ✅ 一致 |
| **每 4 层一次** | 是 | 是 | ✅ 一致 |
| **最后丢弃历史帧** | 是 | 是 | ✅ 一致 |

---

## 4. 为什么差异导致了训练失败

### 4.1 梯度路径分析

要理解问题，先要搞清楚**前向传播**和**反向传播**的完整路径。

#### 前向传播（训练时数据流经哪些模块）

```
输入: K 帧图像 [B*K, C, H, W] + 语言指令 + 机器人状态 + 加噪动作

Step 1: SigLIP ViT 编码图像（冻结 + temporal attention）
  ├─ Layer 0:  spatial_attn → FFN                          (冻结)
  ├─ Layer 1:  spatial_attn → FFN                          (冻结)
  ├─ Layer 2:  spatial_attn → FFN                          (冻结)
  ├─ Layer 3:  spatial_attn → temporal_attn → FFN          (冻结 + ★temporal可训练)
  ├─ Layer 4-6: ...                                        (冻结)
  ├─ Layer 7:  spatial_attn → temporal_attn → FFN          (冻结 + ★temporal可训练)
  ├─ ...（每 4 层一个 temporal_attn）
  ├─ Layer 23: spatial_attn → temporal_attn → FFN          (冻结 + ★temporal可训练)
  ├─ Layer 24: spatial_attn → FFN                          (冻结)
  ├─ Layer 25: spatial_attn → FFN                          (冻结)
  ├─ Layer 26: spatial_attn → FFN                          (冻结)
  └─ 丢弃历史帧，只保留当前帧 → image_tokens [B, n, d]
这块冻结了21层，6层带temporal attention的不冻结，中间可训练的地方是 temporal_attn layer_norm, out_proj）


这块是为了让空间token（1152维度）和Gemma LLM的embedding（2048维）进行对齐。为什么之前要训练这块呢？是因为现在我们加了 temporal attention，SigLIP 输出的 image tokens 变了——它们不再是纯粹的单帧特征，而是融合了时序信息的特征。虽然变化不大，但特征的分布还是有微小偏移。如果冻结 multi_modal_projector，它还是按老方式"翻译"，可能无法把新增的时序信息正确传达给 LLM。让它可训练，就是让它有机会适应新的输入分布。

Step 2: multi_modal_projector（★可训练）
  └─ image_tokens → 映射到 LLM 的 embedding 空间 → prefix_embs


这里涉及冻结18层
冻结原因：涉及的参数量大，20亿个参数，其次是Google 用海量的图文数据预训练出来的。它已经学会了：理解图像内容理解自然语言把图像和语言关联起来。模型训练的本质是：在参数空间里找一组参数，让 loss 在训练数据上最小。参数空间维度 = 20 亿。约束条件（数据点）= 50 个 episode × 每个 episode ~几百步 ≈ 几万个数据点。20 亿个未知数，几万个方程 → 方程远少于未知数。有无穷多组解能让训练 loss = 0。优化器会找到一组"恰好。完美拟合这几万个数据点"的参数。但这组参数只是记住了训练数据，对新数据（验证集）毫无泛化能力
Step 3: PaliGemma LLM backbone（冻结）
  ├─ 输入 = [image_embs, language_embs, state_embs, noisy_action_embs]
  ├─ 经过 N 层 Transformer（冻结）
  └─ 输出 suffix_out（对应 action 部分）



18层。 3亿个参数，合适，大小和数据量匹配。可以用lora进行部分参数微调。
Step 4: Action Expert / gemma_expert（★可训练）
  └─ suffix_out → 经过 action expert 的 Transformer 层 → 预测去噪方向


把 Action Expert 输出的 hidden states 映射成最终的动作预测。
Step 5: action_out_proj（★可训练）
  └─ → 最终预测 v_t

Step 6: Loss = MSE(v_t, u_t)
```

#### 反向传播（梯度从 loss 往回流）

梯度就是"loss 对某个参数的偏导数"，如果每层的导数都小于1，经过这几十层之后，它的梯度就会消失

梯度从 loss 开始，**沿着前向路径反向**逐层传回：

```
Loss = MSE(v_t, u_t)           ← 梯度起点，信号很强
  ↑
action_out_proj                ← ★可训练，直接收到 loss 梯度，梯度大 ✅
  ↑
Action Expert (gemma_expert)   ← ★可训练，梯度稍衰减但仍然很强 ✅
  ↑
PaliGemma LLM backbone         ← 冻结，梯度只是"路过"，每穿过一层都会缩小
  ↑                               （这里有 ~18 层冻结 Transformer）
  ↑                               梯度到这里已经比起点小了很多
  ↑
multi_modal_projector          ← ★可训练，但梯度已经穿过了整个 LLM，变弱了 ⚠️
  ↑
┌───────────────── SigLIP ViT（冻结的 backbone）──────────────────┐
│ Layer 26 (冻结)  ← 梯度继续衰减                                 │
│ Layer 25 (冻结)  ← 继续衰减                                     │
│ Layer 24 (冻结)  ← 继续衰减                                     │
│ Layer 23: temporal_attn ← ★可训练，但梯度已经穿过了:             │
│                            · 整个 Action Expert (~十几层)        │
│                            · 整个 PaliGemma LLM (~18层)          │
│                            · multi_modal_projector               │
│                            · SigLIP Layer 24-26 (3层冻结)        │
│                            = 总共约 30+ 层，梯度极其微弱 ❌       │
│                                                                  │
│ ...更多冻结层...                                                  │
│                                                                  │
│ Layer 3:  temporal_attn ← ★可训练，梯度又多穿过了 20 层冻结层    │
│                            此时梯度基本为零 ❌❌                   │
└──────────────────────────────────────────────────────────────────┘
```

#### 对比 Action Expert 和 Temporal Attention 的梯度路径

```
                        ┌─ Action Expert (梯度路径短)
                        │
Loss ──→ action_out_proj ──→ gemma_expert
                        │     ↑
                        │     只穿过 1-2 层就到达可训练参数
                        │     梯度非常强 → expert_grad_norm ≈ 0.8 ✅
                        │
                        │
                        └─ Temporal Attention (梯度路径极长)
                              │
                              ↓ 穿过 Action Expert (~十几层)
                              ↓ 穿过 PaliGemma LLM (~18层)
                              ↓ 穿过 multi_modal_projector
                              ↓ 穿过 SigLIP 冻结层 (3~24层)
                              ↓ 总共 30~50+ 层
                              ↓
                              temporal_attn.out_proj
                              梯度几乎为零 → temporal_grad_norm ≈ 0.02 ❌
```


从 Loss 到 temporal attention 要穿过多少个 Jacobian？
粗略算一下：
模块	层数	每层的子操作数	Jacobian（雅可比矩阵） 数量
action_out_proj	1	1	~1
Action Expert (18层)	18	~5	~90
PaliGemma LLM (18层)	18	~5	~90
multi_modal_projector	1	1	~1
SigLIP 冻结层 (到Layer 23)	3	~5	~15
总计			~200 个 Jacobian
梯度到达 Layer 23 的 temporal attention 时，已经被 约 200 个 Jacobian 矩阵连乘过了。
到达 Layer 3 的 temporal attention？再多 20 层 × 5 = 100 个，总共 约 300 个 Jacobian 连乘。
























#### 为什么穿过冻结层梯度会衰减？

冻结层虽然不更新自己的权重，但梯度仍然要**穿过**它们（chain rule）。
每穿过一层 Transformer（LayerNorm + Attention + FFN），梯度会乘以该层的 Jacobian 矩阵。
冻结层的参数不是专门为"传递梯度"优化的，所以：

- 有些方向的梯度会被放大，有些会被缩小
- 总体效果是：**穿过越多层，梯度信号越弱**（类似深层网络的梯度消失问题）
- 虽然 ResNet 式的残差连接缓解了一部分，但穿过 30+ 层后衰减仍然非常严重

这就是为什么日志里看到 `grad_ratio` 从 4x 飙升到 42x —— expert 梯度正常，
temporal 梯度随着训练越来越小，两者的差距越来越大。

### 4.2 训练日志中的证据

从 wandb 日志观察到的现象：

| 指标 | 训练初期 | 训练后期 | 含义 |
|------|---------|---------|------|
| `expert_grad_norm` | ~0.8 | ~0.8 | Action Expert 梯度正常 |
| `temporal_grad_norm` | ~0.05 | ~0.02 | Temporal 梯度趋近于零 |
| `grad_ratio` | ~4x | ~42x | 比值快速增大 |
| `train_loss` | 下降 | 继续下降 | Expert 在拟合训练集 |
| `val_loss` | 下降 | 飙升 | Expert 过拟合 + temporal 模块引入噪声 |
| `val/action_cosine_sim` | ~0.3 | 变负 | 在验证集上预测方向完全错误 |

### 4.3 问题的因果链

```
out_proj 初始化为零
  → 初始时 temporal residual = 0，不影响前向
  → 但 out_proj 需要通过梯度来学习有用的变换
  → 梯度穿过 4~24 层冻结 backbone 后几乎为零
  → out_proj 基本保持在零附近，学不到东西
  → temporal attention 等于没有起作用
  → 模型退化为"只有 action expert 在学"
  → action expert 用 50 个 episode 训练 15 万步 → 严重过拟合
  → 同时 out_proj/layer_norm 里的微小随机更新成为噪声源
  → 验证集上 loss 飙升、cosine similarity 变负
```

### 4.4 论文为什么不存在这个问题

MEM 论文的 temporal attention 没有任何可训练参数：

- **没有 out_proj** → 不需要梯度来学习输出变换
- **没有 LayerNorm** → 不需要梯度来学习归一化参数
- **Q/K/V 直接用 hidden states** → 不需要经过任何投影层
- temporal attention 的效果完全由 **SigLIP 已经学好的特征空间** 决定

此外，PI 的训练策略也完全不同：

| | MEM 论文 | 我们 |
|---|---|---|
| **训练阶段** | 预训练阶段 | 后训练（SFT） |
| **SigLIP 状态** | 一起训练（有自己的 VL loss） | 完全冻结 |
| **数据量** | 海量混合数据 | 50 个 episode |
| **梯度流** | SigLIP 直接收到 VL loss 梯度 | 只有 action loss，穿过冻结层 |

---

## 5. 论文的零参数方案如何"理解"时序信息

可能的疑问：**没有可学习参数，temporal attention 怎么提取时序信息？**

### 5.1 信息已经存在于特征中

SigLIP 预训练后，hidden_states 对每个 patch 已经有了强语义表示。
不同帧在同一 patch 位置的特征差异，**天然地编码了时序变化**：

```
帧 t-2, patch(3,5): [0.8, -0.3, 0.5, ...] → "红色方块在这里"
帧 t-1, patch(3,5): [0.7, -0.2, 0.6, ...] → "红色方块稍微偏移了"
帧 t,   patch(3,5): [0.6, -0.1, 0.7, ...] → "红色方块继续移动"
```

### 5.2 Temporal attention 做的是信息聚合

temporal attention 把同一位置不同时间步的特征做**因果加权平均**：

```
当前帧 patch(3,5) 的新特征
  = 原特征 + α₁×帧t-2的特征 + α₂×帧t-1的特征 + α₃×帧t的特征
```

权重 α 由特征之间的相似度自动计算（dot-product attention），不需要学习。

聚合后的特征隐含了时序信息：
- 如果物体在移动 → 融合的特征和单帧不同，编码了"运动方向"
- 如果物体静止 → 融合的特征和单帧几乎一样

### 5.3 谁真正"理解"时序信息？

**Action Expert。** 它是可训练的，在 SFT 过程中学会：
- 识别融合后特征中的时序 pattern
- 利用这些 pattern 做出更好的 action 预测

temporal attention 本身不"理解"任何东西，它只是一个**信息汇聚管道**。

---

## 6. 修复方向

### 方案：回归论文原始设计（零参数 temporal attention）

**要改的：**

1. 删除 `TemporalCausalAttentionModule` 类
2. 替换为纯函数 `temporal_causal_attention(hidden_states, num_frames, num_heads)`
3. 函数内部：
   - Q = hidden_states + 固定 temporal PE
   - K = hidden_states + 固定 temporal PE
   - V = hidden_states
   - 做 causal attention，返回结果
   - 没有 LayerNorm、没有 out_proj、没有 Dropout

**不需要改的：**

- temporal_posemb_sincos 函数 — 已经正确
- SiglipEncoderLayer 的调用位置 — 已经正确（每 4 层一次）
- SiglipEncoder 最后丢弃历史帧 — 已经正确
- 数据管线 — 已经正确

**训练脚本的变化：**

- `_TRAINABLE_PATTERNS` 中去掉 `"temporal_attn"` — 不再有 temporal 参数
- 去掉 temporal 参数组、去掉 `_TEMPORAL_LR_MULTIPLIER` — 不需要了
- 去掉 `temporal_grad_norm` 相关日志 — 不再有 temporal 梯度
- 去掉 `_init_temporal_from_spatial()` 调用 — 没有 out_proj 需要初始化

---

## 7. 预期效果

| 指标 | 当前实现（有参数） | 修复后（零参数） | 原因 |
|------|-------------------|-----------------|------|
| temporal_grad | ~0.02（趋近零） | 不存在 | 没有 temporal 参数 |
| 过拟合风险 | 高（temporal 噪声 + expert 过拟合） | 降低 | temporal 不引入噪声 |
| val_loss | 飙升 | 预期稳定或缓慢下降 | 不再有学不好的参数干扰 |
| 时序信息利用 | 几乎没有（out_proj ≈ 0） | 自动融合 | 纯计算，不需要学习 |
| checkpoint 兼容性 | 有新 key（temporal_attn.*） | 完全兼容原始权重 | 零新参数 |
