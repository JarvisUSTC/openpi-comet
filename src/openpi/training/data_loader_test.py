import dataclasses
import json
import sys
import types
import zipfile

import numpy as np

fake_eval_utils = types.ModuleType("omnigibson.learning.utils.eval_utils")
fake_eval_utils.PROPRIOCEPTION_INDICES = {"R1Pro": {}}
sys.modules.setdefault("omnigibson", types.ModuleType("omnigibson"))
sys.modules.setdefault("omnigibson.learning", types.ModuleType("omnigibson.learning"))
sys.modules.setdefault("omnigibson.learning.utils", types.ModuleType("omnigibson.learning.utils"))
sys.modules["omnigibson.learning.utils.eval_utils"] = fake_eval_utils

import openpi.training.config as _config
import openpi.training.data_loader as data_loader
import openpi.transforms as _transforms
from openpi.models import pi0_config


@dataclasses.dataclass(frozen=True)
class _TagTransform:
    tag: str

    def __call__(self, data: dict) -> dict:
        return {**data, "tag": self.tag}


class _DummyDataset:
    def __init__(self, payload: dict):
        self._payload = payload

    def __getitem__(self, index):
        return dict(self._payload)

    def __len__(self):
        return 8


def test_huggingface_vqa_dataset_uses_config_columns(monkeypatch):
    def _fake_load_dataset(name, config_name, split):
        assert name == "dummy/vqa"
        assert config_name == "default"
        assert split == "train"
        return [
            {
                "image": "img0",
                "question": "What is shown?",
                "answers": ["cat", "cat", "dog"],
            }
        ]

    monkeypatch.setattr("datasets.load_dataset", _fake_load_dataset)
    config = _config.DataConfig(
        dataset_type="hf_vqa",
        hf_dataset_name="dummy/vqa",
        hf_dataset_config_name="default",
        hf_dataset_split="train",
        hf_image_column="image",
        hf_question_column="question",
        hf_answer_column="answers",
        requires_norm_stats=False,
    )
    dataset = data_loader.HuggingFaceVQADataset(config)
    item = dataset[0]
    assert item["image"] == "img0"
    assert item["prompt"] == "What is shown?"
    assert item["answer"] == ["cat", "cat", "dog"]


def test_create_mixed_dataset_keeps_per_dataset_transforms(monkeypatch):
    def _fake_create_dataset(data_config, model_config, *, num_samples):
        del model_config, num_samples
        return _DummyDataset({"source": data_config.repo_id})

    monkeypatch.setattr(data_loader, "create_dataset", _fake_create_dataset)
    model_config = pi0_config.Pi0Config()
    cfg_a = _config.DataConfig(
        repo_id="dataset_a",
        requires_norm_stats=False,
        data_transforms=_transforms.Group(inputs=[_TagTransform("A")]),
    )
    cfg_b = _config.DataConfig(
        repo_id="dataset_b",
        requires_norm_stats=False,
        data_transforms=_transforms.Group(inputs=[_TagTransform("B")]),
    )

    mixed = data_loader.create_mixed_dataset(
        [cfg_a, cfg_b],
        model_config,
        sample_weights=[0.5, 0.5],
        num_samples=4,
        skip_norm_stats=True,
    )

    assert isinstance(mixed, data_loader.WeightedMultiDataset)
    assert mixed.datasets[0][0]["tag"] == "A"
    assert mixed.datasets[1][0]["tag"] == "B"


def test_local_vqa_schema_dataset_reads_zip_images(tmp_path):
    image_path = tmp_path / "sample.png"
    archive_path = tmp_path / "images.zip"
    array = np.full((12, 12, 3), 127, dtype=np.uint8)

    from PIL import Image

    Image.fromarray(array).save(image_path)
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(image_path, arcname="frames/sample.png")

    schema_path = tmp_path / "schema.jsonl"
    schema_path.write_text(
        json.dumps(
            {
                "sample_id": "sample-0",
                "prompt": "What is the robot doing?",
                "answer": "moving",
                "images": [
                    {
                        "storage": "zip",
                        "archive_path": str(archive_path),
                        "member_path": "frames/sample.png",
                    }
                ],
                "task_family": "planning",
                "task_name": "toy",
                "source": "unit-test",
                "metadata": {},
            }
        )
        + "\n"
    )

    config = _config.DataConfig(
        dataset_type="local_vqa_schema",
        local_vqa_schema_paths=(str(schema_path),),
        requires_norm_stats=False,
    )
    dataset = data_loader.LocalVQASchemaDataset(config)
    item = dataset[0]

    assert item["prompt"] == "What is the robot doing?"
    assert item["answer"] == "moving"
    assert item["images"][0].shape == (12, 12, 3)


def _write_split_zip(zip_path, output_prefix, chunk_size):
    data = zip_path.read_bytes()
    for idx, start in enumerate(range(0, len(data), chunk_size)):
        (output_prefix.parent / f"{output_prefix.name}.{idx:02d}").write_bytes(data[start : start + chunk_size])


def test_robointer_vqa_dataset_reads_annotation_and_split_zip(tmp_path):
    root = tmp_path / "RoboInter-VQA"
    annotation_dir = root / "Task_planning" / "meta" / "train" / "manipvqa"
    image_dir = root / "Task_planning" / "image" / "train" / "manipvqa"
    annotation_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)

    png_path = tmp_path / "positive_image.png"
    from PIL import Image

    Image.fromarray(np.full((10, 8, 3), 200, dtype=np.uint8)).save(png_path)
    full_zip = tmp_path / "task_planning_full.zip"
    with zipfile.ZipFile(full_zip, "w") as archive:
        archive.write(png_path, arcname="task_planning/demo#0.jpg")
        archive.write(png_path, arcname="task_planning/demo#1.jpg")
    _write_split_zip(full_zip, image_dir / "task_planning.zip", chunk_size=120)

    annotation_path = annotation_dir / "task_planning.json"
    annotation_path.write_text(
        json.dumps(
            [
                {
                    "id": "demo-1",
                    "task": "Planning_Task",
                    "images": [
                        "Task_planning/image/train/manipvqa/task_planning/demo#0.jpg",
                        "Task_planning/image/train/manipvqa/task_planning/demo#1.jpg",
                    ],
                    "conversations": [
                        {
                            "from": "human",
                            "value": "<image>\n<image>\nWhat comes next to finish the task?",
                        },
                        {"from": "gpt", "value": "place the towel on the table"},
                    ],
                }
            ]
        )
    )

    config = _config.DataConfig(
        dataset_type="robointer_vqa",
        local_vqa_root=str(root),
        robointer_annotation_paths=("Task_planning/meta/train/manipvqa/task_planning.json",),
        requires_norm_stats=False,
    )
    dataset = data_loader.RoboInterVQADataset(config)
    item = dataset[0]

    assert item["prompt"] == "What comes next to finish the task?"
    assert item["answer"] == "place the towel on the table"
    assert len(item["images"]) == 2
    assert item["images"][0].shape == (10, 8, 3)
