# MemoryVLA finetune 训练流程梳理（含 dataloader downloading 之后）

## 1. 脚本入口

- **`scripts/finetune_8gpu_pytorch.sh`**  
  - 调用：`torchrun --nproc_per_node=8 scripts/train_pytorch.py <CONFIG_NAME> --exp_name=...`  
  - 即 8 个进程（每卡一进程）各自执行 `train_pytorch.py`。

## 2. train_pytorch.py 中的整体顺序

1. **DDP 初始化** → `setup_ddp()`
2. **构建 DataLoader** → `build_datasets(config)`  
   - 打印：`[DEBUG] [1/6] Building datasets...`  
   - 内部会创建 **BehaviorLeRobotDataset** 并做 transform、TorchDataLoader 等。  
   - **所有「downloading」和「loading HF dataset」都发生在这步、且在每个 rank 的主进程里执行（尚未进 DataLoader 的 worker）。**
3. 只有 `build_datasets()` 返回后才会打印：`[DEBUG] [1/6] Datasets built OK.`
4. 之后才是：创建模型 → DDP wrap → 加载权重 → 优化器 → **进入训练循环** → `[DEBUG] [6/6] Entering training loop, fetching first batch...` → 第一次 `next(loader)`。

因此：**若在「downloading data」之后长时间没有新 log、且 GPU 利用率为 0，说明卡在 `build_datasets()` 内部、且多半是 CPU/IO 阶段（还没到训练循环）。**

---

## 3. Dataloader / Dataset 在「downloading data」之后具体做了什么

「downloading data」通常来自 **HuggingFace**（`snapshot_download` 或仓库拉取时的输出）。  
在代码里，**downloading 之后**的步骤全部在 **`BehaviorLeRobotDataset.__init__`**（`src/behavior/learning/datas/dataset.py`）中，顺序如下。

### 3.1 若本地缺文件：先 download，再 load HF dataset

- 先检查本地是否有所需 episode 文件（`get_episodes_file_paths()`）。
- 若缺文件或 `force_cache_sync`：
  - 调用 **`download_episodes(download_videos)`** → **`pull_from_repo()`** → **`snapshot_download(...)`**（这里会看到 HuggingFace 的 “downloading” 之类输出）。
  - 然后打印：`[DEBUG-DS] Downloaded, loading HF dataset...`（注意：这是 `BehaviorLeRobotDataset` 的 module logger，若未配置可能看不到）。
  - 接着执行 **`load_hf_dataset()`**（见下）。

### 3.2 `load_hf_dataset()` — 最可能长时间无 log、无 GPU 的步骤

- **位置**：`dataset.py` 第 479–489 行。
- **逻辑**：  
  - 若有 `self.episodes`：  
    `files = [self.root / self.meta.get_data_file_path(ep_idx) for ep_idx in self.episodes]`  
    → **`load_dataset("parquet", data_files=files, split="train")`**  
  - 若无：  
    `load_dataset("parquet", data_dir=path, split="train")`  
- **为何可能耗时很久且无 log**：  
  - HuggingFace `datasets` 会打开/内存映射**所有**列出的 parquet 文件。  
  - episode 很多时，`files` 很大，大量 I/O + 建索引，**且库内部几乎没有逐文件进度 log**。  
  - 纯 CPU/磁盘操作，**GPU 利用率为 0 是正常的**。  
- 只有执行完后才会打：`[DEBUG-DS] HF dataset loaded in X.Xs.`（同样依赖 module logger）。

### 3.3 之后依次执行的步骤（仍可能耗时）

| 步骤 | 代码位置 | 说明 |
|------|----------|------|
| **Building episode_data_index** | 369 行 | `get_episode_data_index(self.meta.episodes, self.episodes)`，建 episode 索引。 |
| **Checking timestamps sync** | 371–378 行 | 若 `check_timestamp_sync=True`：会 **`th.stack(self.hf_dataset["timestamp"])` / `th.stack(self.hf_dataset["episode_index"])`**，即**遍历整个 hf_dataset**，数据量大时可能很慢，且无进度 log。 |
| **prepare_task** | 386–387 行 | 按 `fine_grained_level` 准备 task，相对轻量。 |

其中 **check_timestamps_sync** 是另一处可能「静默」耗时的步骤（同样无 GPU）。

### 3.4 Dataset 初始化完成后

- `create_torch_behavior_data_loader` 还会：  
  - 用 `TransformedDataset` 包一层（含 prompt、normalize 等）；  
  - 若 `video_memory_frames > 1`，再包一层 **VideoMemoryDataset**；  
  - 最后用 **TorchDataLoader**（PyTorch `DataLoader` + DistributedSampler 等）。  
- **第一次用 GPU 的时机**：要等到训练循环里 **`for observation, actions in loader`** 第一次 `next(loader)` 之后，模型 forward 才开始用 GPU。  
- 若 `num_workers > 0`，DataLoader 的 **worker 子进程** 里会执行 **`__getitem__`**（读视频、解码等），但 **Dataset 的 `__init__`（含 download、load_hf_dataset、timestamp check）只在主进程 build_datasets 时执行一次（且每个 rank 各执行一次）。**

---

## 4. 多卡（8 进程）下的注意点

- **每个 rank 都会执行一次** `build_datasets()` → **每个 rank 都会跑一遍** `BehaviorLeRobotDataset.__init__`。  
- 也就是说：**8 个进程会各自**  
  - 做一次（或等待）`snapshot_download`（若触发），  
  - 各自执行 **`load_hf_dataset()`**（打开同一批 parquet 文件），  
  - 各自做 **episode_data_index** 和 **check_timestamps_sync**。  
- 因此：**load_hf_dataset + timestamp 检查 会被放大 8 倍（8 个进程同时做类似 I/O），且都无 GPU 占用。**

---

## 5. 小结：哪里可能「downloading 之后就没 log、也没 GPU」

1. **最可能：`load_hf_dataset()`**  
   - 大量 parquet 的打开/映射，无进度输出，纯 CPU/IO，耗时可很长。  
2. **其次：`check_timestamps_sync`**  
   - 全表遍历 `hf_dataset["timestamp"]` / `episode_index`，数据大时慢，且无进度。  
3. **若连「Downloaded, loading HF dataset...」都看不到**  
   - 可能是 **`snapshot_download` 尚未结束**（或 module logger 未打到控制台），看起来像「downloading 之后就没动静」。

建议在 **`load_hf_dataset()` 前后** 以及 **check_timestamps_sync 前后** 用 **根 logger（`logging.info`）** 打明确 log（并带耗时），这样无论 module logger 是否配置，控制台都能看到进度，便于确认卡在哪一阶段。
