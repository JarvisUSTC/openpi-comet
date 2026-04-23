import numpy as np

import openpi.policies.vqa_policy as vqa_policy


def test_vqa_inputs():
    transform = vqa_policy.VQAInputs(action_dim=8, action_horizon=4)
    image = np.random.randint(0, 255, size=(32, 32, 3), dtype=np.uint8)
    output = transform({"image": image, "prompt": "What color is the mug?", "answer": ["blue", "blue", "green"]})

    assert output["state"].shape == (8,)
    assert output["actions"].shape == (4, 8)
    assert output["image"]["base_0_rgb"].shape == (32, 32, 3)
    assert bool(output["image_mask"]["base_0_rgb"]) is True
    assert bool(output["image_mask"]["left_wrist_0_rgb"]) is False
    assert output["answer"] == "blue"


def test_vqa_inputs_multi_image():
    transform = vqa_policy.VQAInputs(action_dim=4, action_horizon=2)
    images = [
        np.full((16, 16, 3), fill_value=idx * 20, dtype=np.uint8)
        for idx in range(5)
    ]

    output = transform({"images": images, "prompt": "What happens next?", "answer": "pick up the cup"})

    assert output["image"]["base_0_rgb"].ndim == 3
    assert output["image"]["left_wrist_0_rgb"].ndim == 3
    assert output["image"]["right_wrist_0_rgb"].ndim == 3
    assert bool(output["image_mask"]["base_0_rgb"]) is True
    assert bool(output["image_mask"]["left_wrist_0_rgb"]) is True
    assert bool(output["image_mask"]["right_wrist_0_rgb"]) is True
    assert output["answer"] == "pick up the cup"
