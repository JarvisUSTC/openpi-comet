import behavior.learning.datas.dataset as dataset


def test_build_orchestrators_from_annotations_uses_assembled_subtask_for_level_1():
    skill_annotation = [
        {
            "skill_description": ["move to"],
            "object_id": [["radio_89"]],
            "frame_duration": [0, 10],
            "assembled_subtask": "move to the radio",
        }
    ]

    levels = dataset.build_orchestrator_levels_from_annotations(
        episode_key=1590,
        episode_len=10,
        skill_annotation=skill_annotation,
        level_0_task="turning on radio",
    )

    assert levels[1][0]["task"] == "move to the radio"
    assert levels[1][0]["skill_name"] == "move to"
    assert levels[2][0]["task"] == "move to radio"
    assert levels[2][0]["skill_name"] == "move to"


def test_skill_weight_supports_strict_whitelist_entries():
    assert dataset.skill_weight("move to", ["move to", "press"]) == 1.0
    assert dataset.skill_weight("pick up from", ["move to", "press"]) == 0.0
    assert dataset.skill_weight("move to", ["all", "move to:0.25"]) == 0.25
