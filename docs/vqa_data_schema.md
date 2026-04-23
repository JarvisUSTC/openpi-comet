# VQA Data Schema

为了让不同来源的 VQA 数据都能复用同一套训练入口，本仓库约定本地 VQA 预处理后的统一 sample schema 如下：

```json
{
  "sample_id": "unique-sample-id",
  "prompt": "Question text without raw <image> placeholders.",
  "answer": "Single normalized target string.",
  "images": [
    {
      "storage": "path",
      "path": "/abs/path/to/image.jpg"
    },
    {
      "storage": "zip",
      "archive_path": "/abs/path/to/images.zip",
      "member_path": "subdir/image.jpg"
    }
  ],
  "task_family": "understanding",
  "task_name": "contact_decide",
  "source": "RoboInter-VQA",
  "metadata": {
    "annotation_path": "Understanding/meta/val/contact_decide.json",
    "raw_task": "contact_decide"
  }
}
```

约束：

- `prompt` 必须已经清洗成最终训练文本，不再依赖原始数据集特定字段。
- `answer` 必须是单个字符串；如果原数据有多答案、多投票或结构化字段，需要在预处理阶段先归一化。
- `images` 允许单图或多图；训练侧会自动把多图打包成最多 3 个视觉槽位。
- `storage="zip"` 适合大数据集按需读图，避免全量解压。
- `metadata` 只放追踪信息，不参与模型监督。

当前实现：

- 原始 `RoboInter-VQA` 可直接通过 `robointer_vqa` dataset type 读取。
- 已经是统一 schema 的本地数据可通过 `local_vqa_schema` dataset type 直接训练。

推荐后续新数据接入流程：

1. 先把原始标注转换成上述 schema（JSON 或 JSONL）。
2. 统一把答案归一化成单字符串监督目标。
3. 若图片很多，优先保存为 zip 引用而不是解压散文件。
4. 在训练配置中使用 `LocalVQASchemaDataConfig` 接入。
