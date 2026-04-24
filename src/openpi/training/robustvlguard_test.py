import json
import sys
import types

import numpy as np

fake_eval_utils = types.ModuleType("omnigibson.learning.utils.eval_utils")
fake_eval_utils.PROPRIOCEPTION_INDICES = {"R1Pro": {}}
sys.modules.setdefault("omnigibson", types.ModuleType("omnigibson"))
sys.modules.setdefault("omnigibson.learning", types.ModuleType("omnigibson.learning"))
sys.modules.setdefault("omnigibson.learning.utils", types.ModuleType("omnigibson.learning.utils"))
sys.modules["omnigibson.learning.utils.eval_utils"] = fake_eval_utils

from PIL import Image

import openpi.training.config as _config
import openpi.training.data_loader as data_loader
import openpi.training.robustvlguard as robustvlguard


def test_convert_jsonl_to_local_vqa_schema_and_load(tmp_path):
    image_dir = tmp_path / "gqa" / "images"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "2336875.jpg"
    Image.fromarray(np.full((9, 7, 3), 180, dtype=np.uint8)).save(image_path)

    raw_path = tmp_path / "comprehensive_4k_sft_gpt_anno.jsonl"
    raw_path.write_text(
        json.dumps(
            {
                "id": 2336875,
                "image": "gqa/images/2336875.jpg",
                "conversations": [
                    {"from": "human", "value": "<image>\nAre there any grapes or bananas?"},
                    {"from": "gpt", "value": "Yes, there are bananas in the image."},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    schema_path = tmp_path / "comprehensive_4k_openpi_schema.jsonl"
    converted = robustvlguard.convert_jsonl(raw_path, schema_path, image_root=tmp_path)

    assert converted == 1

    config = _config.DataConfig(
        dataset_type="local_vqa_schema",
        local_vqa_schema_paths=(str(schema_path),),
        requires_norm_stats=False,
    )
    dataset = data_loader.LocalVQASchemaDataset(config)
    item = dataset[0]

    assert item["prompt"] == "Are there any grapes or bananas?"
    assert item["answer"] == "Yes, there are bananas in the image."
    assert item["sample_id"] == "2336875"
    assert item["task_family"] == robustvlguard.DEFAULT_TASK_FAMILY
    assert item["task_name"] == robustvlguard.DEFAULT_TASK_NAME
    assert item["source"] == robustvlguard.DEFAULT_SOURCE
    assert item["metadata"]["raw_image"] == "gqa/images/2336875.jpg"
    assert item["images"][0].shape == (9, 7, 3)
