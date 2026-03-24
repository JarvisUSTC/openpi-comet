# memory0.1 -> JAX baseline 迁移记录

本文档记录本轮迁移中已经完成的代码修改、测试、排障和当前状态，便于后续继续推进。

## 目标

目标是把 `memory0.1` 中的视频记忆 / 时序注意力相关能力，按最小风险方式迁移到 `openpi-comet-baseline`，并优先保证：

- `K=1` 行为不退化
- `K>1` 的 shape / mask / 前向逻辑先打通
- 数据与训练链路可启动、可观测、可继续排障

## 已完成改动

### 1. 迁移计划文档补充

已更新：

- `/root/Training/memory0.1/MIGRATION_PLAN.md`

补充了迁移前期最关键的 5 类风险与测试建议：

- 帧顺序语义风险
- `temporal_mask` 对齐风险
- padding 历史帧污染风险
- `K=1` 回归风险
- 吞吐 / 显存压力风险

### 2. `Observation` 增加时序输入支持

已修改：

- `src/openpi/models/model.py`

完成内容：

- 在 `Observation` 中新增 `temporal_mask`
- 在 `Observation.from_dict()` 中支持 5D 图像输入
  - `[B, K, H, W, C] -> [B*K, H, W, C]`
- 在 `from_dict()` 中同步扩展 `image_mask`
  - `[B] -> [B*K]`
- 在 `preprocess_observation()` 中透传 `temporal_mask`

目的：

- 先让模型入口具备接受多帧图像 + 帧有效性 mask 的能力

### 3. SigLIP 视觉塔增加时序分支

已修改：

- `src/openpi/models/siglip.py`

完成内容：

- 新增 `temporal_posemb_sincos()`
  - 提供时序位置编码
- 新增 `temporal_causal_attention()`
  - 在时间维做 causal attention
  - 复用已有 spatial attention 投影
  - 支持 `temporal_mask`
- 在 `Encoder1DBlock.__call__()` 中增加时序分支参数
  - `num_frames`
  - `temporal_mask`
  - `has_temporal_attn`
- 时序分支按“spatial residual -> temporal residual -> MLP”顺序接入
- `K=1` 时完全不走 temporal 分支，保留 no-op 路径
- 在 `_Module.__call__()` 末尾恢复时序维并取最后一帧
  - `[B*K, N, D] -> [B, K, N, D] -> [B, N, D]`

目的：

- 在不破坏原始单帧路径的前提下，先接入最小可验证的 temporal 融合逻辑

### 4. Pi0 接入时序视觉特征

已修改：

- `src/openpi/models/pi0.py`

完成内容：

- 新增 `_align_image_mask_with_token_batch()`
  - 将展开后的 `[B*K]` image mask 对齐回 token batch 的 `[B]`
- 在 `Pi0.__init__()` 中保存 `self.video_memory_frames`
- 在 `embed_prefix()` 中将 `num_frames` 和 `temporal_mask` 传给视觉塔
- 在 lazy init 时也按 `video_memory_frames` 对齐初始化路径

目的：

- 让 `Pi0` 前缀嵌入路径真正支持 `K>1`
- 保证视觉 token 与 image mask 的 batch 维一致

### 5. 单元测试与联调测试

已新增：

- `scripts/test_temporal_siglip.py`
- `scripts/test_temporal_pi0.py`
- `scripts/test_temporal_pi0_integration.py`

覆盖内容：

- `K=1` no-op 等价性检查
- `K=3` 输出 shape 检查
- `temporal_mask` 含 `False` 时稳定性检查
- `Pi0` image mask 对齐检查
- `Pi0` prefix -> attn_mask -> positions -> LLM 前向联通检查

目的：

- 每改一块就能立刻验证，不把错误拖到训练阶段

### 6. JAX 训练脚本补上验证逻辑

已修改：

- `scripts/train.py`

完成内容：

- 增加验证是否启用的判断
- 增加验证配置覆盖逻辑
- 增加 `eval_step()`
- 在主训练循环中接入验证 dataloader
- 记录 `val_loss` / `val/flow_loss`
- WandB 日志打通验证指标

目的：

- 对齐组长 `openpi-comet` 的训练可观测性
- 后续做短训时可以直接看验证 loss

### 7. K=1 smoke 配置与启动脚本

已修改：

- `src/openpi/training/config.py`
- `scripts/run_k1_smoke.sh`

完成内容：

- 新增 `pi05_b1k-k1-smoke-step200` 配置（后续又调整成当前更长的 smoke 配置）
- `run_k1_smoke.sh` 做了以下兼容处理：
  - 优先使用项目 `.venv`
  - 修正 `wandb` 布尔参数传法
  - 支持 `WANDB_ENABLED`
  - 保留 `OVERWRITE`
  - 输出更清晰的启动日志

目的：

- 让云端可直接起一条短训练进行链路检查

### 8. `omnigibson` 依赖的最小 fallback

已修改：

- `src/openpi/policies/b1k_policy.py`
- `src/behavior/learning/datas/dataset.py`
- `src/openpi/training/data_loader.py`
- `scripts/compute_norm_stats.py`

完成内容：

- `b1k_policy.py`
  - 增加 `PROPRIOCEPTION_INDICES` 的本地 fallback
- `dataset.py`
  - 对若干 `omnigibson` 导入加 `try/except`
  - 缺 `omnigibson` 时退回到 `lerobot` 的统计 / 视频工具
  - 提供最小 `ROBOT_CAMERA_NAMES` / `TASK_NAMES_TO_INDICES`
- `data_loader.py`
  - 自动检测是否存在 `omnigibson`
  - 缺失时关闭 keyframe streaming，改走非流式 dataloader
  - 明确设置 `check_timestamp_sync=False`
  - 在 fallback 路径中把视频解码容差放宽到 `0.05`
- `compute_norm_stats.py`
  - 缺 `omnigibson` 时退回 dataloader 统计路径
  - 修正到 baseline 实际存在的 `create_behavior_dataset()` 接口

目的：

- 在未完整安装 `omnigibson` 的情况下，先把 B1K 离线训练链路跑通

### 9. 复用 `memory0.1` 的 norm stats

已处理：

- 将
  - `/root/Training/memory0.1/outputs/assets/train/pi05_b1k-all_skills_mem_K6/behavior-1k/2025-challenge-demos/norm_stats.json`
- 复制到
  - `/root/Training/openpi-comet-baseline/outputs/assets/train/pi05_b1k-k1-smoke-step200/behavior-1k/2025-challenge-demos/norm_stats.json`

说明：

- 这份统计目前用于 baseline 的 K=1 smoke
- 复用依据是：
  - 同一 `repo_id`
  - 相同的 `state/actions` 维度与语义
  - 归一化主要作用于 `state/actions`，不是时序图像本身

## 已遇到问题与处理

### A. 环境与依赖

- `uv: command not found`
  - 已通过 `.venv` 优先策略解决
- `ModuleNotFoundError: openpi`
  - 用 `PYTHONPATH=src` 解决
- `ModuleNotFoundError: omnigibson`
  - 在策略、数据集、统计脚本处分别补了 fallback

### B. CLI 与脚本

- `--wandb_enabled true` 参数格式错误
  - 已改为 `--wandb-enabled` / `--no-wandb-enabled`

### C. 模型联调

- `Pi0` 初始化缺 `config`
  - 已改为实例属性 `self.video_memory_frames`
- 集成测试中出现：
  - lazy init shape 问题
  - `nnx.Rngs` 用法问题
  - `jaxtyping` 运行时检查问题
  - temporal attention mask 广播问题
  - 均已通过最小修复和测试脚本验证

### D. 数据与训练链路

- 时间戳同步检查失败
  - 已关闭 `check_timestamp_sync`
- `norm_stats` 缺失
  - 已恢复“训练必须加载统计”的正式逻辑
  - 并把现有统计文件复制到 baseline 所需位置
- 非流式 fallback 视频解码时间戳容差过严
  - 已在 fallback 路径中将容差放宽到 `0.05`

## 当前状态

截至当前，已经确认：

- 时序模型改动已接入
- 单元测试 / 集成测试已具备
- baseline 已能读取复用的 `norm_stats`
- K=1 smoke 训练已通过：
  - GPU 识别
  - WandB 初始化
  - checkpoint 初始化
  - `norm_stats` 加载

当前最近一次训练排障聚焦在：

- 无 `omnigibson` fallback 时的视频解码容差问题

该问题已经做了代码修复，待继续重跑确认是否真正进入：

- `Initialized data loader`
- `Initialized train state`
- `Step ... loss=...`

## 剩余待做

### 1. 继续完成 K=1 smoke 训练闭环

目标：

- 确认能稳定进入训练 step
- 确认 `loss` 与 `val/flow_loss` 正常记录

### 2. 数据管道迁移（大头）

尚未完整迁移的核心部分：

- `VideoMemoryDataset`
- `B1kInputs._stack_history_frames`
- `_history` 相关字段透传
- `video_memory_frames` / `video_memory_stride_s` 真正接到 dataset
- `skill_resampling` 对齐

### 3. 正式的 K>1 配置

当前重点仍是 K=1 smoke。后续需要：

- 补正式 K=3 / K=6 配置
- 用真实数据管道验证 `temporal_mask` 和历史帧堆叠

### 4. 视情况恢复完整 OmniGibson 路径

如果后续需要：

- 仿真环境
- keyframe streaming
- segmentation instance mapping

则应安装完整 `omnigibson` 环境，而不是长期依赖 fallback。

## 备注

- 本文档记录的是本轮迁移会话中已经做过的修改，不等同于项目全部历史。
- 若后续继续修改，建议直接续写本文档，避免排障信息丢失。
