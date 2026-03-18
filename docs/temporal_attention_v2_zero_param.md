# Temporal Attention v2：零参数实现（对齐 MEM 论文）

> 本文档记录从 v1（带可学习参数）到 v2（零参数）的所有改动。

---

## 1. 改动动机

v1 实现中 temporal attention 引入了 ~8M 新参数（LayerNorm + out_proj × 6 层），
导致以下训练问题：

- **梯度消失**：从 loss 到 temporal_attn 参数的梯度路径穿过 ~200 个 Jacobian，
  `temporal_grad_norm` 从 0.05 衰减到 0.02
- **参数无效学习**：out_proj 初始化为零且梯度极弱，始终停留在零附近
- **噪声注入**：LayerNorm / out_proj 的微小随机更新成为噪声源，导致 val_loss 飙升
- **模型退化**：temporal attention 等于没有工作，模型退化为单帧 + action expert 过拟合

MEM 论文的核心设计哲学：temporal attention 只是信息聚合管道，不需要可学习参数。
SigLIP 预训练特征已经足够好，时序信息由下游 Action Expert 学习利用。

---

## 2. v1 vs v2 对比

| 设计点 | v1（旧） | v2（新，对齐论文） |
|--------|---------|-----------------|
| 实现方式 | `TemporalCausalAttentionModule` 类（nn.Module） | `temporal_causal_attention()` 纯函数 |
| Q/K 来源 | hidden_states → LayerNorm → 冻结 spatial Q/K proj → + PE | hidden_states + PE（直接用，无投影） |
| V 来源 | hidden_states → LayerNorm → 冻结 spatial V proj | hidden_states（直接用，无投影） |
| LayerNorm | 有（可训练） | 无 |
| out_proj | 有（可训练，初始化 0.01×spatial） | 无 |
| Dropout | attn 0.1 + proj 0.1 | 无 |
| 新参数量 | ~8M（6 层 × 1.33M） | **0** |
| temporal PE | 固定 sinusoidal，当前帧=0 | 不变 |
| 因果 mask | causal（上三角 mask） | 不变 |
| 每 4 层一次 | 是 | 不变 |
| additive residual | 是 | 不变 |
| 最后丢弃历史帧 | 是 | 不变 |

---

## 3. 改动文件清单

### 3.1 `src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py`

**删除：**
- `TemporalCausalAttentionModule` 类（整个类）

**新增：**
- `temporal_causal_attention()` 纯函数：零参数，Q=x+PE, K=x+PE, V=x

**修改：**
- `SiglipEncoder.__init__`：`temporal_attns` 从 `nn.ModuleDict` 改为普通 `set`，
  只记录哪些层需要 temporal attention
- `SiglipEncoder._init_temporal_from_spatial()`：删除（不再需要）
- `SiglipEncoder.forward`：调用纯函数替代 Module
- `SiglipEncoderLayer.forward` 签名：`temporal_attn` 参数改为 `has_temporal_attn: bool`

### 3.2 `scripts/train_pytorch.py`

**删除：**
- `_TEMPORAL_LR_MULTIPLIER` 常量
- `_TRAINABLE_PATTERNS` 中的 `"temporal_attn"` 条目
- `_init_temporal_from_spatial()` 调用
- `temporal_params` / `expert_params` 分组逻辑
- temporal 组的独立 LR 设置
- temporal 组的独立梯度裁剪
- `temporal_grad_norm` / `grad_ratio` 的日志记录

**修改：**
- optimizer 从两组参数改为单组
- 梯度裁剪从分组改为统一

---

## 4. 向后兼容性

- v2 不引入任何新的 state_dict key
- 加载预训练权重时不会有 missing keys（v1 的 temporal_attn.* key 不存在了）
- K=1 时函数直接返回零向量，行为与原始 ViT 完全一致

---

## 5. 后续可选优化（暂不实施）

| 优化项 | 说明 | 优先级 |
|--------|------|--------|
| 上层 token 压缩 | 论文在高层逐步丢弃历史帧 token 以节省计算量 | 中 |
| 折中方案：冻结 Q/K/V proj | 用 spatial 的冻结投影但不引入新参数 | 低（先验证零参数效果） |
