# MEM 数据管道 & 推理 Wrapper 改动文档

> 本文档记录 MEM（Multi-Scale Embodied Memory）短期视频记忆（Short-Term Video Memory）在 **数据管道、推理 wrapper、训练配置** 层面的全部改动。
>
> 模型架构层面的改动（SigLIP temporal attention、embed_prefix 等）见 `MEM_video_encoder_implementation_plan.md`。

---

## 目录

1. [整体数据流概览](#1-整体数据流概览)
2. [文件改动清单](#2-文件改动清单)
3. [改动 A：训练数据多帧采样 — `data_loader.py`](#改动-a训练数据多帧采样--data_loaderpy)
4. [改动 B：数据 Transform 多帧支持 — `b1k_policy.py`](#改动-b数据-transform-多帧支持--b1k_policypy)
5. [改动 C：Observation 多帧格式转换 — `model.py`](#改动-cobservation-多帧格式转换--modelpy)
6. [改动 D：推理帧缓冲区 — `eval_b1k_wrapper.py`](#改动-d推理帧缓冲区--eval_b1k_wrapperpy)
7. [改动 E：推理服务参数透传 — `serve_b1k.py`](#改动-e推理服务参数透传--serve_b1kpy)
8. [改动 F：训练配置 — `config.py`](#改动-f训练配置--configpy)
9. [向后兼容性保证](#7-向后兼容性保证)
10. [验证结果](#8-验证结果)

---

## 1. 整体数据流概览

### 训练时数据流

```
BehaviorLeRobotDataset.__getitem__()
  │  返回单帧 dict: {observation.images.rgb.head: [C,H,W], ...}
  ▼
PromptFromLeRobotItem()
  │  加 prompt 字段
  ▼
VideoMemoryDataset.__getitem__()          ← 新增
  │  维护 per-episode 帧缓冲区
  │  给每个 camera key 加 _history 列表
  │  例: observation.images.rgb.head_history = [frame_t-5, ..., frame_t-1]
  ▼
RepackTransform()
  │  observation.images.rgb.head → observation/egocentric_camera
  │  注意: _history key 不在映射表中，会原样保留
  ▼
B1kInputs.__call__()                      ← 修改
  │  K=1: 输出 image[H,W,C] (uint8)      ← 原始行为
  │  K>1: 查找 _history key，stack 成 image[K,H,W,C] (uint8)
  ▼
Normalize / ModelTransforms
  │  不改变图像维度
  ▼
_collate_fn()
  │  np.stack → [B,H,W,C] 或 [B,K,H,W,C]
  ▼
torch.as_tensor()
  │  numpy → torch tensor
  ▼
Observation.from_dict()                   ← 修改
  │  检测 torch.uint8:
  │    4D [B,H,W,C] → permute → [B,C,H,W] float32
  │    5D [B,K,H,W,C] → reshape → [B*K,H,W,C] → permute → [B*K,C,H,W] float32
  ▼
preprocess_observation_pytorch()
  │  检测 shape[1]==3 (CHW)，做 augmentation，保持 CHW 输出
  ▼
embed_prefix() → embed_image(num_frames=K)
  │  SigLIP 处理 [B*K,C,H,W]，temporal attention，丢弃历史帧
  │  输出 [B,n,d]
```

### 推理时数据流

```
OmniGibson env → obs dict
  ▼
B1KPolicyWrapper.process_obs()            ← 修改
  │  resize 三个相机图像
  │  K>1 时: 调用 _buffer_frame() 缓冲每帧，存入 _frame_history
  ▼
B1KPolicyWrapper._build_policy_batch()    ← 新增
  │  提取公共逻辑（原来分散在 act / act_receeding_temporal 中）
  │  K>1 时: 从 _frame_history 取出历史帧，加入 batch dict
  │  batch["observation/egocentric_camera_history"] = [frame_t-5, ..., frame_t-1]
  ▼
policy.infer(batch)
  │  → _input_transform (包含 B1kInputs)
  │  → B1kInputs._stack_history_frames()  ← 查找 _history key
  │  → torch.from_numpy()[None, ...] 加 batch 维度
  │  → model.sample_actions()
```

---

## 2. 文件改动清单

| 文件 | 改动类型 | 说明 |
|------|---------|------|
| `src/openpi/training/data_loader.py` | 新增类 + 修改函数 | `VideoMemoryDataset` 帧缓冲 wrapper |
| `src/openpi/policies/b1k_policy.py` | 修改类 + 新增函数 | `B1kInputs` 多帧 stack |
| `src/openpi/models/model.py` | 修改方法 | `Observation.from_dict` 5D 图像处理 |
| `src/openpi/shared/eval_b1k_wrapper.py` | 修改类 | 推理帧缓冲区 + 公共 batch 构建 |
| `scripts/serve_b1k.py` | 修改 dataclass + 函数 | CLI 参数透传 |
| `src/openpi/training/config.py` | 修改工厂 + 新增配置 | `LeRobotB1KDataConfig` + MEM 训练配置 |

---

## 改动 A：训练数据多帧采样 — `data_loader.py`

### A1. 新增 `VideoMemoryDataset` 类

**文件**: `src/openpi/training/data_loader.py`，第 128-196 行

```python
class VideoMemoryDataset(Dataset):
    _DEFAULT_CAMERA_KEYS = (
        "observation.images.rgb.head",
        "observation.images.rgb.left_wrist",
        "observation.images.rgb.right_wrist",
    )

    def __init__(self, dataset, num_frames, stride=1, camera_keys=None):
        ...
        self._max_buffer = (num_frames - 1) * self._stride + 1

    def __getitem__(self, index):
        item = self._dataset[index]
        # 维护 per-episode 缓冲区，返回 _history key
        ...
```

**详细解释**：

- **为什么需要这个类**：B1K 的 `BehaviorLeRobotDataset` 使用 chunk streaming 模式，每次 `__getitem__` 只返回一帧。MEM 需要 K 帧历史。`VideoMemoryDataset` 包装原始 dataset，在每次读取时把当前帧存入缓冲区，并从缓冲区中按 stride 采样 K-1 帧历史。

- **`_DEFAULT_CAMERA_KEYS`**：使用 B1K 原始 dataset 的 key（带 `.` 分隔符），因为 `VideoMemoryDataset` 在 repack transform 之前运行。repack 会把 `observation.images.rgb.head` 映射成 `observation/egocentric_camera`，但 `_history` 后缀的 key 不在映射表中，会原样保留在 data dict 里。

- **`_max_buffer`**：缓冲区最大长度 = `(K-1) * stride + 1`。例如 K=6, stride=30 时，需要保留最近 151 帧。这确保了按 stride 采样时有足够的历史帧。

- **`__getitem__` 逻辑**：
  1. 调用底层 dataset 获取当前帧 `item`
  2. 从 `item["episode_index"]` 获取 episode ID
  3. 如果 episode 切换了（`ep_idx not in self._buffers`），清空所有缓冲区（`self._buffers = {}`），因为跨 episode 的帧没有时间连续性
  4. 把当前帧 `copy` 后存入缓冲区（必须 copy，否则后续 transform 可能修改原始数据）
  5. 从缓冲区中按 stride 采样 K-1 帧历史（oldest first）
  6. 如果缓冲区不够（episode 开头），用最早可用帧填充
  7. 把历史帧列表存入 `item[f"{cam_key}_history"]`

- **stride 采样算法**：
  ```
  available = buf[:-1]  # 不含当前帧
  for i in range(K-1, 0, -1):  # 从最远到最近
      idx = len(available) - i * stride
      if idx < 0: 用最早帧填充
      else: sampled.append(available[idx])
  ```
  例如 K=4, stride=2, 缓冲区有 [f0, f1, f2, f3, f4, f5(当前)]：
  - available = [f0, f1, f2, f3, f4]
  - i=3: idx = 5 - 3*2 = -1 → 用 f0
  - i=2: idx = 5 - 2*2 = 1 → f1
  - i=1: idx = 5 - 1*2 = 3 → f3
  - 结果: history = [f0, f1, f3]，加上当前帧 f5 → [f0, f1, f3, f5]

### A2. 修改 `create_behavior_dataset` 函数签名

**文件**: `src/openpi/training/data_loader.py`，第 199 行

```python
def create_behavior_dataset(
    data_config, action_horizon,
    video_memory_frames=1,      # ← 新增
    video_memory_stride_s=1.0   # ← 新增
):
```

**详细解释**：

- 新增两个参数，默认值保证向后兼容
- `video_memory_frames`：K 值，1 表示不使用多帧（原始行为）
- `video_memory_stride_s`：历史帧间隔（秒），会在函数内转换为帧数：`stride_frames = int(video_memory_stride_s * 30)`（B1K 数据集 FPS=30）

### A3. 在函数末尾包装 `VideoMemoryDataset`

**文件**: `src/openpi/training/data_loader.py`，第 228-232 行

```python
    # MEM: wrap with video memory buffer if K > 1
    if video_memory_frames > 1:
        fps = 30
        stride_frames = max(1, int(video_memory_stride_s * fps))
        dataset = VideoMemoryDataset(dataset, num_frames=video_memory_frames, stride=stride_frames)
```

**详细解释**：

- 只在 K>1 时包装，K=1 时完全不影响原始管道
- 包装顺序：`BehaviorLeRobotDataset` → `TransformedDataset(PromptFromLeRobotItem)` → `VideoMemoryDataset`
- 必须在 `PromptFromLeRobotItem` 之后，因为 prompt 提取需要原始 dataset 的 key
- 必须在 `transform_dataset`（repack + B1kInputs + normalize）之前，因为 `VideoMemoryDataset` 用的是原始 key

### A4. 修改 `create_behavior_data_loader` 和 `create_torch_behavior_data_loader`

**文件**: `src/openpi/training/data_loader.py`，第 303 行和第 353 行

```python
    vm_frames = getattr(config.model, "video_memory_frames", 1)
    vm_stride = getattr(config.model, "video_memory_stride_s", 1.0)
    ...
    dataset = create_behavior_dataset(
        data_config,
        action_horizon=config.model.action_horizon,
        video_memory_frames=vm_frames,
        video_memory_stride_s=vm_stride,
    )
```

**详细解释**：

- 从 `config.model`（即 `Pi0Config`）读取 `video_memory_frames` 和 `video_memory_stride_s`
- 使用 `getattr` 带默认值，兼容没有这些字段的旧 config
- 两个 data loader 创建函数都做了同样的修改

---

## 改动 B：数据 Transform 多帧支持 — `b1k_policy.py`

### B1. `B1kInputs` 新增 `video_memory_frames` 字段

**文件**: `src/openpi/policies/b1k_policy.py`，第 157-158 行

```python
    # MEM: number of video memory frames. 1 = single-frame (original behavior).
    video_memory_frames: int = 1
```

**详细解释**：

- `B1kInputs` 是一个 frozen dataclass，作为数据 transform 在训练和推理时都会被调用
- 默认值 1 保证向后兼容
- 这个值从 `LeRobotB1KDataConfig.create()` 中传入，来源是 `Pi0Config.video_memory_frames`

### B2. `B1kInputs.__call__` 中多帧 stack 逻辑

**文件**: `src/openpi/policies/b1k_policy.py`，第 167-179 行

```python
        K = self.video_memory_frames

        base_image = _parse_image(data["observation/egocentric_camera"])
        wrist_image_left = _parse_image(data["observation/wrist_image_left"])
        wrist_image_right = _parse_image(data["observation/wrist_image_right"])

        # MEM: stack history frames if available
        if K > 1:
            base_image = _stack_history_frames(data, "observation/egocentric_camera", base_image, K)
            wrist_image_left = _stack_history_frames(data, "observation/wrist_image_left", wrist_image_left, K)
            wrist_image_right = _stack_history_frames(data, "observation/wrist_image_right", wrist_image_right, K)
```

**详细解释**：

- 先用 `_parse_image` 把当前帧转成 `uint8 [H,W,C]`（和原来一样）
- K>1 时，调用 `_stack_history_frames` 把 K-1 帧历史 + 当前帧 stack 成 `[K,H,W,C]`
- K=1 时，完全不执行多帧逻辑，`base_image` 保持 `[H,W,C]`
- stack 后的 `[K,H,W,C]` 会被放入 `inputs["image"]["base_0_rgb"]`，后续 collate 变成 `[B,K,H,W,C]`

### B3. 新增 `_REPACK_TO_RAW` 映射表

**文件**: `src/openpi/policies/b1k_policy.py`，第 225-230 行

```python
_REPACK_TO_RAW = {
    "observation/egocentric_camera": "observation.images.rgb.head",
    "observation/wrist_image_left": "observation.images.rgb.left_wrist",
    "observation/wrist_image_right": "observation.images.rgb.right_wrist",
}
```

**详细解释**：

- `B1kInputs` 在 repack 之后运行，所以它看到的 key 是 `observation/egocentric_camera`
- 但 `VideoMemoryDataset` 在 repack 之前运行，它创建的 history key 是 `observation.images.rgb.head_history`
- repack 不会映射 `_history` 后缀的 key，所以它们会以原始 key 保留在 data dict 中
- `_REPACK_TO_RAW` 让 `_stack_history_frames` 能从两种 key 命名空间中查找 history

### B4. 新增 `_stack_history_frames` 函数

**文件**: `src/openpi/policies/b1k_policy.py`，第 233-249 行

```python
def _stack_history_frames(data, key, current_frame, K):
    history = None
    for candidate in (f"{key}_history", f"{_REPACK_TO_RAW.get(key, key)}_history"):
        if candidate in data and len(data[candidate]) >= K - 1:
            history = [_parse_image(f) for f in data[candidate][-(K - 1):]]
            break

    if history is not None:
        return np.stack(history + [current_frame], axis=0)  # [K, H, W, C]
    return np.stack([current_frame] * K, axis=0)  # [K, H, W, C]
```

**详细解释**：

- **两种 key 查找**：先尝试 repacked key（`observation/egocentric_camera_history`，推理时使用），再尝试 raw key（`observation.images.rgb.head_history`，训练时使用）
- **`_parse_image`**：对每帧历史帧也做 uint8 HWC 转换，确保格式一致
- **fallback**：如果找不到 history（例如推理时 K=1 的旧 wrapper，或者训练时 `VideoMemoryDataset` 没启用），就把当前帧重复 K 次。这保证了即使数据管道没提供历史帧，模型也能正常运行（只是所有帧都一样）
- **输出格式**：`[K, H, W, C]` uint8 numpy array

---

## 改动 C：Observation 多帧格式转换 — `model.py`

### C1. `Observation.from_dict` 处理 5D 图像

**文件**: `src/openpi/models/model.py`，第 122-141 行

```python
        for key in data["image"]:
            img = data["image"][key]
            if img.dtype == np.uint8:
                if img.ndim == 5:
                    B, K, H, W, C = img.shape
                    img = img.reshape(B * K, H, W, C).astype(np.float32) / 255.0 * 2.0 - 1.0
                else:
                    img = img.astype(np.float32) / 255.0 * 2.0 - 1.0
                data["image"][key] = img
            elif hasattr(img, "dtype") and img.dtype == torch.uint8:
                img = img.to(torch.float32)
                if img.ndim == 5:
                    B, K, H, W, C = img.shape
                    img = img.reshape(B * K, H, W, C).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
                else:
                    img = img.permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
                data["image"][key] = img
```

**详细解释**：

- **原始逻辑**：`from_dict` 检测 `torch.uint8` 图像，做 `permute(0,3,1,2)` 把 `[B,H,W,C]` 转成 `[B,C,H,W]`，并归一化到 `[-1,1]`

- **问题**：当 K>1 时，`B1kInputs` 输出 `[K,H,W,C]`，collate 后变成 `[B,K,H,W,C]`（5D）。直接 `permute(0,3,1,2)` 会得到 `[B,H,K,C]`，这是错误的。

- **修改**：检测 `ndim == 5`，先 `reshape(B*K, H, W, C)` 展平成 4D，再做 permute。这样得到 `[B*K, C, H, W]`，正好是 SigLIP 期望的输入格式。

- **为什么 reshape 而不是 view**：`reshape` 在内存不连续时会自动 copy，更安全。

- **numpy 分支**：同样处理 5D，但不做 permute（numpy 分支是给 JAX 训练用的，JAX 使用 HWC 格式）。

- **K=1 时**：图像是 4D `[B,H,W,C]`，走 `else` 分支，行为和原来完全一致。

---

## 改动 D：推理帧缓冲区 — `eval_b1k_wrapper.py`

### D1. `__init__` 新增参数和帧缓冲区

**文件**: `src/openpi/shared/eval_b1k_wrapper.py`，第 50-52 行（参数）和第 97-101 行（初始化）

```python
        # MEM: video memory parameters
        video_memory_frames: int = 1,
        video_memory_stride: int = 1,
```

```python
        # MEM: video memory frame buffers
        self.video_memory_frames = video_memory_frames
        self.video_memory_stride = max(1, video_memory_stride)
        self._frame_buffers: dict[str, deque] = {}
        self._frame_buffer_maxlen = (video_memory_frames - 1) * self.video_memory_stride + 1
```

**详细解释**：

- `video_memory_frames`：K 值，和训练时一致
- `video_memory_stride`：推理时的帧间隔（以 env step 为单位，不是秒）。训练时用秒是因为数据集有固定 FPS，推理时 env step 频率可能不同
- `_frame_buffers`：per-camera 的帧缓冲区，key 是 camera 名（`"head"`, `"left_wrist"`, `"right_wrist"`）
- `_frame_buffer_maxlen`：和训练时一样的计算方式

### D2. `reset` 清空帧缓冲区

**文件**: `src/openpi/shared/eval_b1k_wrapper.py`，第 161-165 行

```python
    def reset(self):
        ...
        self._frame_buffers = {}
```

**详细解释**：

- episode 切换时必须清空帧缓冲区，否则新 episode 的第一帧会看到旧 episode 的历史帧

### D3. 新增 `_buffer_frame` 方法

**文件**: `src/openpi/shared/eval_b1k_wrapper.py`，第 258-276 行

```python
    def _buffer_frame(self, cam_name, frame):
        K = self.video_memory_frames
        if K <= 1:
            return []

        if cam_name not in self._frame_buffers:
            self._frame_buffers[cam_name] = deque(maxlen=self._frame_buffer_maxlen)
        buf = self._frame_buffers[cam_name]
        buf.append(frame.copy())

        available = list(buf)[:-1]
        sampled = []
        for i in range(K - 1, 0, -1):
            idx = len(available) - i * self.video_memory_stride
            if idx < 0:
                sampled.append(available[0] if available else frame)
            else:
                sampled.append(available[idx])
        return sampled
```

**详细解释**：

- 和 `VideoMemoryDataset` 的采样逻辑完全一致
- 返回 K-1 帧历史列表（不含当前帧），由 `_build_policy_batch` 附加到 batch dict
- 使用 `deque(maxlen=...)` 自动丢弃过旧的帧

### D4. `process_obs` 中缓冲帧

**文件**: `src/openpi/shared/eval_b1k_wrapper.py`，第 312-321 行

```python
        # MEM: buffer each camera frame for video memory
        if self.video_memory_frames > 1:
            head_hist = self._buffer_frame("head", img_obs[0, 0])
            left_hist = self._buffer_frame("left_wrist", img_obs[0, 1])
            right_hist = self._buffer_frame("right_wrist", img_obs[0, 2])
            processed_obs["_frame_history"] = {
                "head": head_hist,
                "left_wrist": left_hist,
                "right_wrist": right_hist,
            }
```

**详细解释**：

- **为什么在 `process_obs` 而不是 `_build_policy_batch` 中缓冲**：`process_obs` 在每次 `act()` 调用时都执行，而 `_build_policy_batch` 只在需要调用 policy 时执行。在 `receeding_horizon` 模式下，如果 action queue 不为空，会直接返回缓存的 action，不调用 policy。如果帧缓冲在 `_build_policy_batch` 中，这些跳过的帧就不会被缓冲，导致帧缓冲区不连续。
- 历史帧存入 `processed_obs["_frame_history"]`，一个临时的内部 key

### D5. 新增 `_build_policy_batch` 方法

**文件**: `src/openpi/shared/eval_b1k_wrapper.py`，第 336-367 行

```python
    def _build_policy_batch(self, nbatch):
        if nbatch["observation"].shape[-1] != 3:
            nbatch["observation"] = np.transpose(nbatch["observation"], (0, 1, 3, 4, 2))

        joint_positions = nbatch["proprio"][0]
        prompt = self._effective_task_prompt(nbatch)

        batch = {
            "observation/egocentric_camera": nbatch["observation"][0, 0],
            "observation/wrist_image_left": nbatch["observation"][0, 1],
            "observation/wrist_image_right": nbatch["observation"][0, 2],
            "observation/state": joint_positions,
            "prompt": prompt,
        }

        # MEM: attach frame history from process_obs
        if self.video_memory_frames > 1 and "_frame_history" in nbatch:
            fh = nbatch["_frame_history"]
            batch["observation/egocentric_camera_history"] = fh["head"]
            batch["observation/wrist_image_left_history"] = fh["left_wrist"]
            batch["observation/wrist_image_right_history"] = fh["right_wrist"]

        if self.wm_in_prompt and "working_memory" in nbatch:
            batch["working_memory"] = nbatch["working_memory"]
            ...

        if "observation/egocentric_depth" in nbatch:
            batch["observation/egocentric_depth"] = nbatch["observation/egocentric_depth"][0]

        return batch
```

**详细解释**：

- **提取公共逻辑**：原来 `act()` 和 `act_receeding_temporal()` 中有大量重复的 batch 构建代码。现在统一到 `_build_policy_batch` 中。
- **history key 命名**：使用 repacked key + `_history` 后缀（如 `observation/egocentric_camera_history`），和 `_stack_history_frames` 的第一优先查找路径一致。
- **WM 和 depth 处理**：也移到这里，保持逻辑集中。

### D6. `act` 和 `act_receeding_temporal` 简化

**文件**: `src/openpi/shared/eval_b1k_wrapper.py`

原来的重复代码：
```python
# 原来在 act() 和 act_receeding_temporal() 中各有一份
nbatch = copy.deepcopy(input_obs)
if nbatch["observation"].shape[-1] != 3:
    nbatch["observation"] = np.transpose(...)
joint_positions = nbatch["proprio"][0]
prompt = self._effective_task_prompt(nbatch)
batch = { ... }
```

现在简化为：
```python
nbatch = copy.deepcopy(input_obs)
batch = self._build_policy_batch(nbatch)
```

---

## 改动 E：推理服务参数透传 — `serve_b1k.py`

### E1. `Args` dataclass 新增参数

**文件**: `scripts/serve_b1k.py`，第 90-93 行

```python
    # MEM: number of video memory frames (1 = single-frame, no history).
    video_memory_frames: int = 1
    # MEM: stride in environment steps between historical frames.
    video_memory_stride: int = 1
```

### E2. 传递给 `B1KPolicyWrapper`

**文件**: `scripts/serve_b1k.py`，第 212-213 行

```python
    policy = B1KPolicyWrapper(
        ...
        video_memory_frames=args.video_memory_frames,
        video_memory_stride=args.video_memory_stride,
    )
```

**详细解释**：

- 使用 `tyro` CLI 框架，新增的 dataclass 字段自动变成命令行参数
- 启动推理服务时可以指定：`python scripts/serve_b1k.py --video-memory-frames 6 --video-memory-stride 5`

---

## 改动 F：训练配置 — `config.py`

### F1. `LeRobotB1KDataConfig.create` 传递 `video_memory_frames`

**文件**: `src/openpi/training/config.py`，第 306-308 行

```python
        vm_frames = getattr(model_config, "video_memory_frames", 1)
        data_transforms = _transforms.Group(
            inputs=[b1k_policy.B1kInputs(
                action_dim=model_config.action_dim,
                model_type=model_config.model_type,
                video_memory_frames=vm_frames,  # ← 新增
            )],
            ...
        )
```

**详细解释**：

- `model_config` 是 `Pi0Config` 实例，它已经有 `video_memory_frames` 字段（在之前的模型架构改动中加的）
- 使用 `getattr` 带默认值 1，兼容没有这个字段的旧 config
- 这样 `B1kInputs` 就知道要 stack 多少帧

### F2. 新增 MEM 训练配置

**文件**: `src/openpi/training/config.py`，第 748-773 行

```python
    TrainConfig(
        name="pi05_b1k-pt50_mem_K6_cs32_bs64_lr2.5e-5_step50k",
        ...
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=32,
            video_memory_frames=6,       # ← K=6
            video_memory_stride_s=1.0,   # ← 1秒间隔
        ),
        data=LeRobotB1KDataConfig(
            repo_id="behavior-1k/2025-challenge-demos",
            base_config=DataConfig(
                prompt_from_task=True,
                episodes_index=list(range(200)),
                behavior_dataset_root="../DATASETS/behavior/2025-challenge-demos",
                fine_grained_level=0,
            ),
        ),
        ...
        batch_size=8 * 32,
    ),
```

**详细解释**：

- 配置名 `pi05_b1k-pt50_mem_K6_cs32_bs64_lr2.5e-5_step50k`：50 task pretrain，MEM K=6
- `video_memory_frames=6`：使用 5 帧历史 + 1 帧当前 = 6 帧
- `video_memory_stride_s=1.0`：历史帧间隔 1 秒，即 30 帧（B1K FPS=30）
- 其他参数和 `pi05_b1k-pt50_cs32_bs64_lr2.5e-5_step50k` 完全一致
- 使用方式：`python scripts/train.py --config pi05_b1k-pt50_mem_K6_cs32_bs64_lr2.5e-5_step50k`

---

## 7. 向后兼容性保证

所有新增参数的默认值都设计为 **K=1（单帧）**，此时：

| 组件 | K=1 行为 |
|------|---------|
| `VideoMemoryDataset` | `__getitem__` 直接返回原始 item，不加 `_history` key |
| `B1kInputs` | 不调用 `_stack_history_frames`，image 保持 `[H,W,C]` |
| `Observation.from_dict` | 图像是 4D，走原始 `permute(0,3,1,2)` 分支 |
| `B1KPolicyWrapper` | `_buffer_frame` 返回空列表，不缓冲帧 |
| `_build_policy_batch` | 不附加 `_history` key |
| `create_behavior_dataset` | 不包装 `VideoMemoryDataset` |

**验证结果**：使用 `test_video_memory.py` Step 5 验证，K=1 时 loss 与基线完全一致（diff=0.0）。

---

## 8. 验证结果

### 单元测试

| 测试 | 结果 |
|------|------|
| `B1kInputs` K=1 输出 `[H,W,C]` | PASS |
| `B1kInputs` K=6 无 history → 重复当前帧 `[6,H,W,C]` | PASS |
| `B1kInputs` K=6 有 repacked key history | PASS |
| `B1kInputs` K=6 有 raw key history | PASS |
| `VideoMemoryDataset` K=4 stride=2 帧缓冲 | PASS |
| `B1KPolicyWrapper` 帧缓冲 + reset | PASS |
| 训练配置 `pi05_b1k-pt50_mem_K6` 加载 | PASS |

### 端到端测试（test_video_memory.py）

| Step | 测试内容 | 结果 |
|------|---------|------|
| Step 5 Test A | K=1 端到端退化（loss 与基线一致） | PASS (diff=0.0) |
| Step 5 Test B | K=6 端到端 forward（无报错） | PASS (loss=2.3690) |

### import 验证

| 模块 | 结果 |
|------|------|
| `openpi.policies.b1k_policy` | OK |
| `openpi.training.data_loader` | OK |
| `openpi.shared.eval_b1k_wrapper` | OK |
| `openpi.training.config` | OK |
