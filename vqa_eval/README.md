# vqa_eval

用 pi05 KI VQA-joint 训练的 checkpoint 做纯 VQA（图像 + 问题 → 文本答案）推理。**不**测动作、**不**画注意力（注意力可视化留到下一阶段）。

目标 ckpt（默认）：

```
outputs/checkpoints/pi05_b1k-ki-vqa-joint-pick-up-from-no-task-planning/
  pi05_ki_joint_pick_up_from_vqa_no_task_planning_ga1/5000/
```

对应的训练 config：`pi05_b1k-knowledge_insulation-vqa-joint-skill-pick-up-from-no-task-planning`（见 `src/openpi/training/config.py`）。

## 目录

```
vqa_eval/
├── README.md
├── data/
│   ├── images/             # 5 张 sanity-check 用图，从 /dataset-vla/benchmark/final_box.zip 抽取
│   └── samples.jsonl       # 15 条 (image, question) 样本
├── outputs/                # 运行结果落到 <timestamp>/results.jsonl + report.md
└── scripts/
    └── eval_vqa.py         # 主脚本
```

## 设计要点

- **绕过 `create_trained_policy`**：训练 config 引用的 `assets_dir`、`behavior_dataset_root`（`/vepfs-C/...`）在推理机上不存在；直接用 `model.restore_params(...) + train_config.model.load(...)` 加载，bypass dataset/normstats 依赖。
- **输入 pack**：复用 `openpi.policies.vqa_policy._pack_images`，给单张图自动填到 `base_0_rgb`，另外两路置黑 + `image_mask=False`。和训练时的 `VQAInputs` 同一套逻辑。
- **prompt 模板**：直接拼 `Task: <question>, State: <discretized zero state>;\nAnswer:`（与 `FASTTokenizer._tokenize_prefix` + 空 answer 的格式一致），不附 EOS，让模型自己续写答案。
- **解码**：贪心自回归，复用 PaliGemma KV cache。第一遍前缀 forward 一次性走完图像 + prompt，之后每步只前向 1 个新 token。EOS 来自 SentencePiece。

## 运行

```bash
cd /b1k/Jiawei/openpi-comet-clean

OPENPI_DATA_HOME=/b1k/.cache/openpi \
    python vqa_eval/scripts/eval_vqa.py \
        --ckpt outputs/checkpoints/pi05_b1k-ki-vqa-joint-pick-up-from-no-task-planning/pi05_ki_joint_pick_up_from_vqa_no_task_planning_ga1/5000 \
        --samples vqa_eval/data/samples.jsonl \
        --out-dir vqa_eval/outputs
```

可选参数：

- `--max-new-tokens` 默认 64
- `--limit N` 只跑前 N 条（先用 `--limit 1` 做 smoke test 比较稳）
- `--tokenizer /path/to/paligemma_tokenizer.model` 显式指定 tokenizer

## 输出

每次运行生成一个 `outputs/<timestamp>/` 目录：

- `results.jsonl`：每行一个 JSON，字段 `id / image / question / answer / note / elapsed_s / prompt_token_len`
- `report.md`：按图分组的可读报告，里面会引用 `data/images/` 里的图

## 已知坑 / 注意

- 第一次运行会触发 JAX/Flax 编译，前几条样本特别慢；之后稳定。
- prompt 长度变了 → 重新编译一次。3 个固定问题 → 共编译 3 次。
- 解码循环本身是 eager 的，每步一次 forward，没有 KV cache pre-allocation；对 sanity check 够用，后面如果要批量跑会改成 `lax.scan` + 固定 cache 形状。
- VQA 训练时 state 是 `np.zeros(action_dim)`，因此推理时也喂 0 state，避免 prompt 分布漂移。
- 当前只用了 `base_0_rgb` 一路（其它两路置黑），跟训练时 `RoboInter-VQA` 走 `VQAInputs` 单图样本时的处理完全一致。

## 后续

- attention / feature heatmap：在 `src/openpi/models/gemma.py` 的 attention 层加 `return_attentions` 开关，导出权重后投回原图。**这一步暂不做**。
