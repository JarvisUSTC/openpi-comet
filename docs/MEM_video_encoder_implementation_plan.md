# MEM Short-Term Video Memory 实现方案

> 基于 Physical Intelligence 的 MEM (Multi-Scale Embodied Memory) 论文，在 pi0.5-comet 上实现 Short-Term Video Memory 部分。
>
> 论文链接：https://www.pi.website/download/Mem.pdf

---

## 1. 背景与目标

### 1.1 问题

当前 pi0.5-comet 每次推理只使用**单帧观测**，缺乏时序上下文，导致：
- 无法处理自遮挡（手臂挡住目标物体）
- 无法感知短期动态（物体运动方向、速度）
- 无法做 in-context adaptation（抓取失败后换策略）

### 1.2 MEM 的解决方案：Space-Time Separable Attention

MEM 将 SigLIP ViT 改造为**视频编码器**，核心设计：

1. **每隔 4 层**，在标准 spatial attention 之后加一个 **causal temporal attention**
2. Temporal attention 在**同一 patch 位置**跨 K 个时间步做因果注意力
3. 复杂度从 O(n²K²) 降到 O(Kn² + nK²)
4. 上层**丢弃历史帧 tokens**，只保留当前帧传给 VLA backbone → token 数不变
5. **不引入新参数**，复用同层 Q/K/V 权重 + 固定 sinusoidal temporal PE
6. K=1 时**完全退化**为原始单帧 ViT（temporal PE 在 t=0 时为 0）

### 1.3 本次实现范围

**只做 Short-Term Video Memory**（视频编码器 + 历史帧注入），不做 Long-Term Language Memory。

---

## 2. 架构概览

```
原始 pi0.5 流程:
  单帧图像 [B, H, W, 3]
    → SigLIP ViT (spatial attention only)
    → image tokens [B, n, d]
    → PaliGemma backbone + Action Expert
    → actions

改造后流程:
  K 帧图像 [B, K, H, W, 3]
    → SigLIP ViT (spatial + temporal attention)
    → 丢弃历史帧, 只保留当前帧 tokens [B, n, d]
    → PaliGemma backbone + Action Expert (完全不变)
    → actions
```

关键：**PaliGemma backbone 和 Action Expert 完全不需要改动**，所有变化封装在 SigLIP ViT 内部。

---

## 3. 需要改动的文件清单

### 阶段 1：模型架构（核心）

| 文件 | 改动内容 | 预估改动量 |
|------|---------|-----------|
| `src/openpi/models/pi0_config.py` | 新增 video memory 配置参数 | ~10 行 |
| `src/openpi/models/siglip.py` | JAX: ViT 加 temporal causal attention | ~150 行 |
| `src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py` | PyTorch: ViT 加 temporal causal attention | ~150 行 |
| `src/openpi/models/model.py` | Observation 数据结构扩展支持多帧 | ~20 行 |
| `src/openpi/models/pi0.py` | JAX: `embed_prefix` 处理多帧输入 | ~30 行 |
| `src/openpi/models_pytorch/pi0_pytorch.py` | PyTorch: `embed_prefix` 处理多帧输入 | ~30 行 |

### 阶段 2：数据管线

| 文件 | 改动内容 | 预估改动量 |
|------|---------|-----------|
| `src/behavior/learning/datas/dataset.py` | 数据集返回历史帧窗口 | ~50 行 |
| `src/openpi/training/config.py` | 新增 video memory 训练配置 | ~30 行 |
| `src/openpi/policies/b1k_policy.py` | 数据 transform 适配多帧 | ~20 行 |

### 阶段 3：推理

| 文件 | 改动内容 | 预估改动量 |
|------|---------|-----------|
| `src/openpi/shared/eval_b1k_wrapper.py` | 推理时维护帧缓冲区 | ~30 行 |
| `scripts/serve_b1k.py` | Policy server 适配 | ~10 行 |

---

## 4. 各文件详细改动

### 4.1 `src/openpi/models/pi0_config.py` — 配置参数

在 `Pi0Config` 中新增：

```python
@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    # ... 现有字段 ...

    # === Video Memory 配置 ===
    # 视频记忆帧数，K=1 时退化为原始单帧行为
    video_memory_frames: int = 1
    # 帧采样间隔（秒），预训练用 1.0s，后训练可扩展到 3.0s
    video_memory_stride_s: float = 1.0
```

同时修改 `inputs_spec` 方法，当 `video_memory_frames > 1` 时 image spec 增加时间维度。

---

### 4.2 `src/openpi/models/siglip.py` — JAX 侧 ViT 改造

#### 4.2.1 新增：Temporal Sinusoidal Position Encoding

```python
def temporal_posemb_sincos(K, width, dtype=jnp.float32):
    """生成 temporal position encoding，t=0（当前帧）时值为 0。

    Args:
        K: 帧数
        width: embedding 维度
    Returns:
        [K, width]，其中最后一个位置（当前帧）为全 0
    """
    # positions: [K-1, K-2, ..., 1, 0]，当前帧 = 0
    positions = jnp.arange(K - 1, -1, -1, dtype=jnp.float32)
    omega = jnp.arange(width // 2, dtype=jnp.float32) / (width // 2)
    omega = 1.0 / (10000.0 ** omega)
    out = jnp.einsum("t,d->td", positions, omega)
    pe = jnp.concatenate([jnp.sin(out), jnp.cos(out)], axis=-1)
    # 当 position=0 时，sin=0, cos=1，需要减去 cos 部分使其为 0
    # 更简单的做法：直接让 position=0 对应全 0
    pe = pe.at[-1].set(0.0)  # 当前帧 PE = 0
    return jnp.asarray(pe, dtype)
```

#### 4.2.2 新增：Temporal Causal Attention 函数

```python
def temporal_causal_attention(x, num_frames, num_heads):
    """对同一 patch 位置跨时间步做 causal self-attention。

    不引入新参数，复用同层的 Q/K/V 投影权重。

    Args:
        x: [B*K, n, d] — 所有帧的 patch embeddings
        num_frames: K — 帧数
        num_heads: attention head 数量
    Returns:
        [B*K, n, d] — temporal attention 输出（additive residual）
    """
    BK, n, d = x.shape
    B = BK // num_frames
    K = num_frames
    head_dim = d // num_heads

    # reshape: [B, K, n, d]
    x_4d = x.reshape(B, K, n, d)

    # 加 temporal PE
    temporal_pe = temporal_posemb_sincos(K, d)  # [K, d]
    x_with_pe = x_4d + temporal_pe[None, :, None, :]  # [B, K, n, d]

    # 转置为 [B, n, K, d]，把 patch 维度并入 batch → [B*n, K, d]
    q = x_with_pe.transpose(0, 2, 1, 3).reshape(B * n, K, d)
    k = q
    v = x_4d.transpose(0, 2, 1, 3).reshape(B * n, K, d)  # value 不加 PE

    # reshape for multi-head: [B*n, K, num_heads, head_dim]
    q = q.reshape(B * n, K, num_heads, head_dim)
    k = k.reshape(B * n, K, num_heads, head_dim)
    v = v.reshape(B * n, K, num_heads, head_dim)

    # attention scores: [B*n, num_heads, K, K]
    scale = head_dim ** -0.5
    attn = jnp.einsum("bknh,blnh->bnkl", q, k) * scale

    # causal mask: 下三角，每个时间步只能看到自己和之前的帧
    causal_mask = jnp.tril(jnp.ones((K, K), dtype=jnp.bool_))
    attn = jnp.where(causal_mask[None, None, :, :], attn, -2.3819763e38)
    attn = jax.nn.softmax(attn, axis=-1)

    # weighted sum: [B*n, num_heads, K, head_dim]
    out = jnp.einsum("bnkl,blnh->bknh", attn, v)

    # reshape 回 [B*n, K, d] → [B, n, K, d] → [B, K, n, d] → [B*K, n, d]
    out = out.reshape(B * n, K, d)
    out = out.reshape(B, n, K, d).transpose(0, 2, 1, 3).reshape(BK, n, d)

    return out
```

#### 4.2.3 修改：`Encoder1DBlock`

```python
class Encoder1DBlock(nn.Module):
    mlp_dim: int | None = None
    num_heads: int = 12
    dropout: float = 0.0
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, deterministic=True, *, num_frames=1, layer_idx=0):
        out = {}
        x = sharding.activation_sharding_constraint(x)

        # === Spatial Attention（不变）===
        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        y = out["sa"] = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=deterministic,
            dtype=self.dtype_mm,
        )(y, y)
        y = sharding.activation_sharding_constraint(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = out["+sa"] = x + y

        # === 新增：Temporal Causal Attention（每 4 层一次）===
        if num_frames > 1 and layer_idx % 4 == 3:
            temporal_out = temporal_causal_attention(
                x, num_frames=num_frames, num_heads=self.num_heads
            )
            x = x + temporal_out  # additive residual

        # === FFN（不变）===
        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        y = out["mlp"] = MlpBlock(
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            dtype_mm=self.dtype_mm,
        )(y, deterministic)
        y = sharding.activation_sharding_constraint(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = out["+mlp"] = x + y
        x = sharding.activation_sharding_constraint(x)
        return x, out
```

#### 4.2.4 修改：`Encoder`

在 `Encoder.__call__` 中传入 `num_frames` 和 `layer_idx`，并在最后丢弃历史帧：

```python
class Encoder(nn.Module):
    # ... 现有字段 ...

    @nn.compact
    def __call__(self, x, deterministic=True, *, num_frames=1):
        out = {}

        # 注意：scan 模式需要特殊处理 layer_idx，
        # 非 scan 模式直接传入
        if not self.scan:
            for lyr in range(self.depth):
                block_cur = Encoder1DBlock(...)
                x, out[f"block{lyr:02d}"] = block_cur(
                    x, deterministic,
                    num_frames=num_frames,
                    layer_idx=lyr,
                )

        x = nn.LayerNorm(name="encoder_norm", dtype=self.dtype_mm)(x)

        # === 新增：丢弃历史帧 tokens ===
        if num_frames > 1:
            BK, n, d = x.shape
            B = BK // num_frames
            x = x.reshape(B, num_frames, n, d)
            x = x[:, -1, :, :]  # 只保留当前帧 [B, n, d]

        return x, out
```

#### 4.2.5 修改：`_Module`（ViT 主类）

在 `__call__` 中接受 `num_frames` 参数：

```python
class _Module(nn.Module):
    @nn.compact
    def __call__(self, image, *, train=False, num_frames=1):
        # image: [B*K, H, W, 3] 或 [B, H, W, 3]（K=1 时）

        # Patch extraction（不变）
        x = nn.Conv(...)(image)
        n, h, w, c = x.shape
        x = jnp.reshape(x, [n, h * w, c])

        # Spatial position embedding（不变）
        x = x + get_posemb(self, self.posemb, (h, w), c, "pos_embedding", jnp.float32)

        # ... dropout, cast dtype ...

        # Encoder（传入 num_frames）
        x, out["encoder"] = Encoder(
            depth=self.depth, ...
        )(x, deterministic=not train, num_frames=num_frames)

        # x 现在是 [B, n, d]（已丢弃历史帧）
        # 后续 pooling 等不变
        ...
```

---

### 4.3 `src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py` — PyTorch 侧 ViT 改造

与 JAX 侧逻辑完全对称，改动点：

#### 4.3.1 新增：`temporal_posemb_sincos` 函数

```python
def temporal_posemb_sincos(K, width, device, dtype=torch.float32):
    """同 JAX 版本，当前帧 PE = 0"""
    positions = torch.arange(K - 1, -1, -1, dtype=torch.float32, device=device)
    omega = torch.arange(width // 2, dtype=torch.float32, device=device) / (width // 2)
    omega = 1.0 / (10000.0 ** omega)
    out = torch.einsum("t,d->td", positions, omega)
    pe = torch.cat([torch.sin(out), torch.cos(out)], dim=-1)
    pe[-1] = 0.0  # 当前帧 PE = 0
    return pe.to(dtype)
```

#### 4.3.2 新增：`temporal_causal_attention` 函数

```python
def temporal_causal_attention(x, num_frames, num_heads):
    """同 JAX 版本，PyTorch 实现"""
    BK, n, d = x.shape
    B = BK // num_frames
    K = num_frames
    head_dim = d // num_heads

    x_4d = x.reshape(B, K, n, d)

    temporal_pe = temporal_posemb_sincos(K, d, device=x.device, dtype=x.dtype)
    x_with_pe = x_4d + temporal_pe[None, :, None, :]

    q = x_with_pe.permute(0, 2, 1, 3).reshape(B * n, K, num_heads, head_dim)
    k = q
    v = x_4d.permute(0, 2, 1, 3).reshape(B * n, K, num_heads, head_dim)

    # [B*n, num_heads, K, K]
    scale = head_dim ** -0.5
    attn = torch.einsum("bknh,blnh->bnkl", q, k) * scale

    causal_mask = torch.tril(torch.ones(K, K, device=x.device, dtype=torch.bool))
    attn = attn.masked_fill(~causal_mask[None, None], float("-inf"))
    attn = torch.softmax(attn, dim=-1)

    out = torch.einsum("bnkl,blnh->bknh", attn, v)
    out = out.reshape(B * n, K, d)
    out = out.reshape(B, n, K, d).permute(0, 2, 1, 3).reshape(BK, n, d)

    return out
```

#### 4.3.3 修改：`SiglipEncoderLayer`

```python
class SiglipEncoderLayer(GradientCheckpointingLayer):
    def forward(self, hidden_states, attention_mask, output_attentions=False,
                *, num_frames=1, layer_idx=0):
        # === Spatial Attention（不变）===
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        # === 新增：Temporal Causal Attention ===
        if num_frames > 1 and layer_idx % 4 == 3:
            temporal_out = temporal_causal_attention(
                hidden_states, num_frames, self.self_attn.num_heads
            )
            hidden_states = hidden_states + temporal_out

        # === FFN（不变）===
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs
```

#### 4.3.4 修改：`SiglipEncoder`

```python
class SiglipEncoder(nn.Module):
    def forward(self, inputs_embeds, attention_mask=None, ..., *, num_frames=1):
        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            layer_outputs = encoder_layer(
                hidden_states, attention_mask, output_attentions,
                num_frames=num_frames, layer_idx=idx,
            )
            hidden_states = layer_outputs[0]

        # === 新增：丢弃历史帧 ===
        if num_frames > 1:
            BK, n, d = hidden_states.shape
            B = BK // num_frames
            hidden_states = hidden_states.reshape(B, num_frames, n, d)
            hidden_states = hidden_states[:, -1, :, :]  # [B, n, d]

        return BaseModelOutput(last_hidden_state=hidden_states, ...)
```

#### 4.3.5 修改：`SiglipVisionTransformer`

在 `forward` 中透传 `num_frames`：

```python
class SiglipVisionTransformer(nn.Module):
    def forward(self, pixel_values, ..., *, num_frames=1):
        hidden_states = self.embeddings(pixel_values)
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            ...,
            num_frames=num_frames,
        )
        ...
```

---

### 4.4 `src/openpi/models/model.py` — Observation 数据结构

在 `Observation` 中新增可选的历史帧字段：

```python
@struct.dataclass
class Observation(Generic[ArrayT]):
    images: dict[str, at.Float[ArrayT, "*b h w c"]]
    image_masks: dict[str, at.Bool[ArrayT, "*b"]]
    state: at.Float[ArrayT, "*b s"]

    # === 新增：历史帧 ===
    # 当 video_memory_frames > 1 时使用
    # 每个 key 对应 [B, K, H, W, C]，K 包含当前帧
    images_history: dict[str, ArrayT] | None = None

    # ... 其余字段不变 ...
```

---

### 4.5 `src/openpi/models/pi0.py` — JAX `embed_prefix`

```python
def embed_prefix(self, obs: _model.Observation):
    input_mask = []
    ar_mask = []
    tokens = []

    K = self.config.video_memory_frames if hasattr(self.config, 'video_memory_frames') else 1

    for name in obs.images:
        if K > 1 and obs.images_history is not None:
            # images_history[name]: [B, K, H, W, 3]
            B = obs.images_history[name].shape[0]
            img_input = obs.images_history[name].reshape(B * K, *obs.images_history[name].shape[2:])
        else:
            img_input = obs.images[name]

        # SigLIP 编码，传入 num_frames
        image_tokens, _ = self.PaliGemma.img(img_input, train=False, num_frames=K)
        # image_tokens: [B, n, d]（已丢弃历史帧）

        tokens.append(image_tokens)
        input_mask.append(
            einops.repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1])
        )
        ar_mask += [False] * image_tokens.shape[1]

    # language tokens（不变）
    if obs.tokenized_prompt is not None:
        tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
        tokens.append(tokenized_inputs)
        input_mask.append(obs.tokenized_prompt_mask)
        ar_mask += [False] * tokenized_inputs.shape[1]

    tokens = jnp.concatenate(tokens, axis=1)
    input_mask = jnp.concatenate(input_mask, axis=1)
    ar_mask = jnp.array(ar_mask)
    return tokens, input_mask, ar_mask
```

---

### 4.6 `src/openpi/models_pytorch/pi0_pytorch.py` — PyTorch `embed_prefix`

与 JAX 侧对称：

```python
def embed_prefix(self, images, img_masks, lang_tokens, lang_masks,
                 images_history=None, num_frames=1):
    embs = []
    pad_masks = []
    att_masks = []

    for (img, img_mask), name in zip(
        zip(images, img_masks), ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
    ):
        if num_frames > 1 and images_history is not None:
            # images_history: [B, K, C, H, W]
            B = images_history[name].shape[0]
            img_input = images_history[name].reshape(B * num_frames, *images_history[name].shape[2:])
        else:
            img_input = img

        def image_embed_func(img_input):
            return self.paligemma_with_expert.embed_image(img_input, num_frames=num_frames)

        img_emb = self._apply_checkpoint(image_embed_func, img_input)
        # img_emb: [B, n, d]（已丢弃历史帧）

        bsize, num_img_embs = img_emb.shape[:2]
        embs.append(img_emb)
        pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
        att_masks += [0] * num_img_embs

    # language tokens（不变）
    ...
```

---

### 4.7 `src/behavior/learning/datas/dataset.py` — 数据集返回历史帧

在 `BehaviorLeRobotDataset` 中新增历史帧窗口采样：

```python
class BehaviorLeRobotDataset(LeRobotDataset):
    def __init__(self, ..., video_memory_frames=1, video_memory_stride=1):
        super().__init__(...)
        self.video_memory_frames = video_memory_frames
        self.video_memory_stride = video_memory_stride  # 帧间隔（以数据集帧为单位）

    def __getitem__(self, idx):
        item = super().__getitem__(idx)

        if self.video_memory_frames > 1:
            # 获取当前帧所在 episode 的索引范围
            ep_idx = self._get_episode_index(idx)
            ep_start, ep_end = self.episode_data_index[ep_idx]

            # 计算历史帧索引
            history_indices = []
            for k in range(self.video_memory_frames - 1, -1, -1):
                hist_idx = idx - k * self.video_memory_stride
                # clamp 到 episode 边界（不足时用第一帧填充）
                hist_idx = max(hist_idx, ep_start)
                history_indices.append(hist_idx)

            # 对每个相机，取历史帧并堆叠
            for cam_key in self.camera_keys:
                frames = []
                for hist_idx in history_indices:
                    frame = self._load_frame(hist_idx, cam_key)
                    frames.append(frame)
                # [K, H, W, 3]
                item[f"{cam_key}_history"] = np.stack(frames, axis=0)

        return item
```

---

### 4.8 `src/openpi/training/config.py` — 训练配置

```python
TrainConfig(
    name="pi05_b1k_video_memory_K6",
    exp_name="openpi",
    project_name="B1K",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_horizon=32,
        video_memory_frames=6,       # 6 帧（5 历史 + 1 当前）
        video_memory_stride_s=1.0,   # 间隔 1 秒
    ),
    data=LeRobotB1KDataConfig(
        repo_id="behavior-1k/2025-challenge-demos",
        base_config=DataConfig(
            prompt_from_task=True,
            behavior_dataset_root="../DATASETS/behavior/2025-challenge-demos",
            fine_grained_level=0,
        ),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "sunshk/openpi_comet/pi05-b1kpt50-cs32"
    ),
    num_train_steps=30_000,
    lr_schedule=_optimizer.CosineDecaySchedule(
        peak_lr=2.5e-5,
        decay_steps=30_000,
    ),
    freeze_filter=pi0_config.Pi0Config(
        pi05=True, action_horizon=32
    ).get_freeze_filter(),
    ema_decay=None,
    checkpoint_base_dir=".",
    num_workers=8,
    batch_size=8 * 32,
)
```

---

### 4.9 `src/openpi/shared/eval_b1k_wrapper.py` — 推理帧缓冲区

```python
class B1KPolicyWrapper:
    def __init__(self, ..., video_memory_frames=1, video_memory_stride=1):
        # ... 现有初始化 ...
        self.video_memory_frames = video_memory_frames
        self.video_memory_stride = video_memory_stride
        # 每个相机维护一个帧缓冲区
        self.frame_buffers = {
            cam: deque(maxlen=video_memory_frames * video_memory_stride)
            for cam in ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
        }

    def reset(self):
        """Episode 开始时清空缓冲区"""
        for buf in self.frame_buffers.values():
            buf.clear()

    def process_obs(self, obs):
        # ... 现有的图像预处理 ...

        # 将当前帧加入缓冲区
        for cam in self.frame_buffers:
            self.frame_buffers[cam].append(current_images[cam])

        if self.video_memory_frames > 1:
            images_history = {}
            for cam in self.frame_buffers:
                buf = list(self.frame_buffers[cam])
                # 按 stride 采样 K 帧
                sampled = []
                for k in range(self.video_memory_frames - 1, -1, -1):
                    idx = len(buf) - 1 - k * self.video_memory_stride
                    if idx < 0:
                        sampled.append(buf[0])  # 不足时用最早的帧填充
                    else:
                        sampled.append(buf[idx])
                images_history[cam] = np.stack(sampled, axis=0)  # [K, H, W, 3]
            obs["images_history"] = images_history

        return obs
```

---

## 5. 关键设计要点

### 5.1 K=1 时完全退化

| 组件 | K=1 时行为 |
|------|-----------|
| Temporal PE | position=0 → PE=0，不影响任何计算 |
| Temporal Attention | `num_frames=1` 条件不满足，跳过 |
| 丢弃历史帧 | `num_frames=1` 条件不满足，跳过 |
| 数据管线 | 不加载历史帧，和原始行为完全一致 |

**结论：可以直接从现有预训练权重加载，不需要任何适配。**

### 5.2 不引入新参数

MEM 的 temporal attention 复用同层的 Q/K/V 投影权重，只新增：
- 固定的 sinusoidal temporal PE（不可训练）
- Causal mask（固定）

**结论：checkpoint 格式完全不变，权重可以无缝加载。**

### 5.3 计算量分析

以 SigLIP So400m/14（224×224 输入）为例：
- patch 数 n = (224/14)² = 256
- depth = 27 层，其中 temporal attention 层 = 27/4 ≈ 7 层
- K=6 时，temporal attention 额外计算 = 7 × O(n × K²) = 7 × 256 × 36 ≈ 64K FLOPs/token
- 相比 spatial attention 的 O(K × n²) = 6 × 65536 ≈ 393K FLOPs/token
- **总增量约 16%**，且上层丢弃历史帧后下游计算量不变

### 5.4 后训练策略建议

| 参数 | 建议值 | 说明 |
|------|--------|------|
| K（帧数） | 6 | 预训练用 6，后训练可扩展到 16 |
| stride | 1.0s | 覆盖约 5 秒历史 |
| 冻结策略 | 冻结 PaliGemma backbone，训练 SigLIP + Action Expert | SigLIP 需要学习 temporal attention |
| 学习率 | 2.5e-5 | 和现有后训练一致 |
| 训练步数 | 30K~50K | 视数据量调整 |

---

## 6. 逐步实施与验证方案

核心原则：**改一块，验证一块，确认没问题再往下走。**

---

### Step 0：建立基线（不改任何代码）

**做什么：**
- 用现有代码 + 现有权重，跑一次 forward pass，记录：
  - 单帧输入的 loss 值（固定一个 batch 的数据）
  - SigLIP 输出的 image tokens shape 和数值（取前几个 token 的值）
  - 完整 forward 的推理延迟
  - 一个固定 batch 的 action 输出

**验证标准：**
- 记录下来作为后续所有步骤的对比基线

**验证脚本示例：**
```python
# scripts/test_baseline.py
import torch
import numpy as np

# 加载模型和一个固定 batch
model = load_model(config)
batch = load_fixed_batch()  # 固定数据，后续每步都用同一个

# 记录基线
with torch.no_grad():
    loss = model(batch.observation, batch.actions)
    actions = model.sample_actions(device, batch.observation)

print(f"Baseline loss: {loss.mean().item()}")
print(f"Baseline actions[:3]: {actions[0, :3, :5]}")
torch.save({
    "loss": loss.mean().item(),
    "actions": actions.cpu(),
}, "baseline_checkpoint.pt")
```

---

### Step 1：只改配置（pi0_config.py）

**做什么：**
- 在 `Pi0Config` 中加 `video_memory_frames` 和 `video_memory_stride_s`
- 默认值 `video_memory_frames=1`

**验证标准：**
- 加载现有权重，跑同一个 batch
- loss 和 actions 与 Step 0 **完全一致**（bit-exact）
- 确认新字段不影响任何现有逻辑

**验证方法：**
```python
config = Pi0Config(pi05=True, action_horizon=32, video_memory_frames=1)
# 确认和原来的 config 行为完全一样
assert config.video_memory_frames == 1
# 跑 forward，对比 Step 0 的 loss
```

**通过标准：** loss diff == 0.0

---

### Step 2：改 SigLIP ViT，只加接口不加逻辑

**做什么：**
- 给 `SiglipEncoderLayer.forward()` 加 `num_frames=1, layer_idx=0` 参数
- 给 `SiglipEncoder.forward()` 加 `num_frames=1` 参数
- 给 `SiglipVisionTransformer.forward()` 加 `num_frames=1` 参数
- JAX 侧同理：`Encoder1DBlock`, `Encoder`, `_Module`
- **不加任何 temporal attention 逻辑**，只透传参数

**验证标准：**
- 加载现有权重，跑同一个 batch
- loss 和 actions 与 Step 0 **完全一致**
- 确认参数透传不破坏任何东西

**通过标准：** loss diff == 0.0

---

### Step 3：实现 temporal attention 函数（独立测试）

**做什么：**
- 实现 `temporal_posemb_sincos()` 函数
- 实现 `temporal_causal_attention()` 函数
- **不接入模型**，只写独立的单元测试

**验证标准：**

```python
# test_temporal_attention.py

def test_temporal_pe_zero_at_current():
    """当前帧（最后一帧）的 PE 必须为全 0"""
    pe = temporal_posemb_sincos(K=6, width=1152, device="cpu")
    assert pe.shape == (6, 1152)
    assert torch.allclose(pe[-1], torch.zeros(1152))  # 当前帧 = 0
    assert not torch.allclose(pe[0], torch.zeros(1152))  # 历史帧 ≠ 0

def test_temporal_attention_K1_is_zero():
    """K=1 时 temporal attention 输出应为 0（或不被调用）"""
    x = torch.randn(2, 256, 1152)  # [B*1, n, d]
    out = temporal_causal_attention(x, num_frames=1, num_heads=16)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)

def test_temporal_attention_shape():
    """输出 shape 必须和输入一致"""
    B, K, n, d = 2, 6, 256, 1152
    x = torch.randn(B * K, n, d)
    out = temporal_causal_attention(x, num_frames=K, num_heads=16)
    assert out.shape == (B * K, n, d)

def test_temporal_attention_causal():
    """因果性：修改未来帧不应影响过去帧的输出"""
    B, K, n, d = 1, 4, 256, 1152
    x = torch.randn(B * K, n, d)
    out1 = temporal_causal_attention(x, num_frames=K, num_heads=16)

    # 修改最后一帧（当前帧）
    x_modified = x.clone()
    x_modified[-1] = torch.randn(n, d)
    out2 = temporal_causal_attention(x_modified, num_frames=K, num_heads=16)

    # 前 3 帧的输出不应变化
    assert torch.allclose(out1[:3], out2[:3], atol=1e-5)
    # 最后一帧的输出应该变化
    assert not torch.allclose(out1[-1:], out2[-1:])
```

**通过标准：** 4 个测试全部通过

---

### Step 4：将 temporal attention 接入 ViT

**做什么：**
- 在 `SiglipEncoderLayer` 中，当 `num_frames > 1 and layer_idx % 4 == 3` 时调用 temporal attention
- 在 `SiglipEncoder` 最后，当 `num_frames > 1` 时丢弃历史帧 tokens
- JAX 侧同理

**验证标准 A — K=1 退化：**
```python
# 用 num_frames=1 跑 forward
# loss 和 actions 与 Step 0 完全一致
```

**验证标准 B — K>1 shape 正确：**
```python
# 构造假的多帧输入 [B*K, H, W, 3]
B, K = 2, 6
fake_images = torch.randn(B * K, 3, 224, 224)
output = siglip_model(fake_images, num_frames=K)
assert output.shape == (B, 256, 1152)  # 只有当前帧的 tokens
```

**验证标准 C — K>1 数值合理：**
```python
# 用同一张图复制 K 次作为输入
single_image = torch.randn(B, 3, 224, 224)
repeated = single_image.repeat(K, 1, 1, 1)  # [B*K, 3, 224, 224]

out_single = siglip_model(single_image, num_frames=1)  # [B, 256, 1152]
out_multi = siglip_model(repeated, num_frames=K)        # [B, 256, 1152]

# 同一张图重复 K 次，输出应该和单帧非常接近（因为 temporal PE 会引入微小差异）
diff = (out_single - out_multi).abs().mean()
print(f"Same-image diff: {diff.item()}")  # 应该很小，< 0.01
```

**通过标准：** A: loss diff == 0.0 | B: shape 正确 | C: same-image diff < 0.01

---

### Step 5：改 embed_prefix 支持多帧

**做什么：**
- 修改 `pi0_pytorch.py` 的 `embed_prefix`，接受 `images_history` 和 `num_frames`
- 当 `num_frames=1` 时走原始路径
- 修改 `model.py` 的 `Observation` 加 `images_history` 字段

**验证标准 A — K=1 退化：**
```python
# 不传 images_history，num_frames=1
# loss 和 actions 与 Step 0 完全一致
```

**验证标准 B — K>1 端到端 shape：**
```python
# 构造带 images_history 的 observation
# 完整 forward pass 不报错
# loss shape 正确: [B, action_horizon]
```

**验证标准 C — K>1 端到端数值：**
```python
# 用同一帧重复 K 次作为 history
# loss 应该和单帧 loss 非常接近（< 5% 相对差异）
loss_single = model(obs_single_frame, actions)
loss_multi = model(obs_repeated_frame, actions)
relative_diff = abs(loss_single - loss_multi) / loss_single
print(f"Relative loss diff: {relative_diff}")  # 应该 < 0.05
```

**通过标准：** A: loss diff == 0.0 | B: 不报错 | C: relative diff < 0.05

---

### Step 6：数据管线返回历史帧

**做什么：**
- 修改 `BehaviorLeRobotDataset.__getitem__` 返回历史帧窗口
- 修改 `b1k_policy.py` 的 transform 处理多帧

**验证标准 A — 数据格式：**
```python
dataset = BehaviorLeRobotDataset(..., video_memory_frames=6, video_memory_stride=30)
item = dataset[100]

# 检查历史帧 shape
for cam in ["observation/egocentric_camera", ...]:
    history = item[f"{cam}_history"]
    assert history.shape == (6, H, W, 3)

# 检查最后一帧 == 当前帧
assert np.allclose(history[-1], item[cam])
```

**验证标准 B — episode 边界处理：**
```python
# 取 episode 的第一帧
item = dataset[episode_start_idx]
history = item[f"{cam}_history"]

# 前面不够的帧应该用第一帧填充
assert np.allclose(history[0], history[1])  # 填充帧 == 第一帧
assert np.allclose(history[-1], item[cam])  # 最后一帧 == 当前帧
```

**验证标准 C — 完整训练 step：**
```python
# 用新的 dataloader 跑一个完整的 training step
batch = next(iter(dataloader))
loss = model(batch.observation, batch.actions)
loss.backward()
# 不报错，loss 是有限值
assert torch.isfinite(loss).all()
```

**通过标准：** A: shape 正确 | B: 边界填充正确 | C: 训练 step 不报错

---

### Step 7：小规模训练验证

**做什么：**
- 用少量数据（比如 1 个 task、100 个 episode）训练 1000 步
- K=6, stride=1s

**验证标准：**
```
□ loss 在 1000 步内持续下降
□ loss 收敛到和单帧模型相近的水平
□ 没有 NaN 或 Inf
□ GPU 显存在预期范围内（相比单帧增加 < 50%）
□ 单步训练时间在预期范围内（相比单帧增加 < 30%）
```

**对比实验：**
```python
# 同时跑两个训练：
# A: video_memory_frames=1（基线）
# B: video_memory_frames=6（新）
# 对比 loss 曲线，B 应该收敛更快或更低
```

**通过标准：** loss 下降 + 无异常 + 延迟可接受

---

### Step 8：推理验证

**做什么：**
- 修改 `eval_b1k_wrapper.py` 加帧缓冲区
- 用训练好的模型跑推理

**验证标准 A — 帧缓冲区逻辑：**
```python
wrapper = B1KPolicyWrapper(..., video_memory_frames=6)
wrapper.reset()

# 模拟 10 步推理
for t in range(10):
    obs = get_obs(t)
    processed = wrapper.process_obs(obs)

    if t < 6:
        # 前 6 步，缓冲区还没满，应该有填充
        assert processed["images_history"]["base_0_rgb"].shape == (6, 224, 224, 3)
    else:
        # 第 7 步开始，缓冲区满，滑动窗口
        assert processed["images_history"]["base_0_rgb"].shape == (6, 224, 224, 3)
```

**验证标准 B — 推理延迟：**
```python
# 单帧推理延迟 vs 6 帧推理延迟
# 6 帧应该 < 单帧 * 1.5（因为下游计算量不变）
# 绝对值应该 < 300ms（MEM 论文的 real-time barrier）
```

**验证标准 C — 定性检查：**
```
□ 在有自遮挡的场景中，video memory 模型表现更好
□ 在需要记住近期动作的场景中，video memory 模型表现更好
□ 在简单场景中，video memory 模型不比单帧差
```

**通过标准：** 缓冲区逻辑正确 + 延迟可接受 + 定性改善

---

### 验证流程总结

```
Step 0: 基线记录        → 得到参考数值
  ↓
Step 1: 加配置参数      → loss diff == 0 ✓ → 继续
  ↓                                    ✗ → 检查配置默认值
Step 2: 加接口不加逻辑  → loss diff == 0 ✓ → 继续
  ↓                                    ✗ → 检查参数透传
Step 3: 独立测试函数    → 4 个单元测试 ✓ → 继续
  ↓                                  ✗ → 修 temporal attention
Step 4: 接入 ViT       → K=1 退化 + K>1 shape + 数值 ✓ → 继续
  ↓                                                   ✗ → 检查 reshape/丢弃逻辑
Step 5: 改 embed_prefix → K=1 退化 + 端到端 ✓ → 继续
  ↓                                         ✗ → 检查 history 传递
Step 6: 数据管线        → shape + 边界 + 训练 step ✓ → 继续
  ↓                                                ✗ → 检查 dataset/transform
Step 7: 小规模训练      → loss 下降 + 无异常 ✓ → 继续
  ↓                                          ✗ → 检查梯度/学习率
Step 8: 推理验证        → 缓冲区 + 延迟 + 定性 ✓ → 完成！
                                               ✗ → 检查推理逻辑
```

每一步失败了就停下来修，不要跳步。

---

## 7. 参考

- MEM 论文: https://www.pi.website/download/Mem.pdf
- MEM 技术页面: https://www.pi.website/research/memory
- π0.5 论文: https://www.physicalintelligence.company/download/pi05_KI.pdf
- Space-Time Separable Attention: Bertasius et al., "Is Space-Time Attention All You Need for Video Understanding?"
