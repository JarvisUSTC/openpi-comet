from __future__ import annotations

import re
from typing import Any

from benchmark.core.schemas import JudgeInput


def _sanitize_object_name(name: str) -> str:
    text = str(name).strip()
    parts = [part for part in text.split("_") if part]

    if (
        len(parts) >= 3
        and parts[-1].isdigit()
        and len(parts[-1]) == 1
        and re.fullmatch(r"[a-z0-9]{6,8}", parts[-2] or "")
    ):
        parts = parts[:-2]
    while len(parts) >= 2 and parts[-1].isdigit():
        parts = parts[:-1]
    return " ".join(parts).strip()


def _format_name_list(values: list[str] | None) -> str:
    if not values:
        return "-"
    cleaned = [_sanitize_object_name(v) for v in values if str(v).strip()]
    return ", ".join(cleaned) if cleaned else "-"


def build_system_prompt() -> str:
    return (
        "You are a strict evaluator for atomic robot skill execution videos. "
        "You will receive one atomic skill instruction, grounding information about the referenced objects, and several frames sampled from one rollout video. "
        "Each sampled image is a three-view montage: two wrist close-up views on the left and one larger scene/global view on the right. "
        "Use the wrist close-up views mainly to reason about grasp, release, continued contact, and target identity; use the larger scene/global view mainly to reason about the final spatial relation and destination context. "
        "You may also receive a relevant clause from the original full task prompt; if it clearly imposes stricter companion-count or relative-placement constraints on the same object category and destination, preserve those constraints. "
        "First identify the exact grounded target instance using the reference image and temporal continuity across the rollout, then judge the final state of that same instance. "
        "Do not switch the target identity to a different similar-looking object that is already nearby in the final frame. "
        "Judge only whether that single atomic skill has been successfully completed by the end of the rollout, not whether the overall long-horizon task is solved. "
        "Success requires that the grounded target object(s), destination, or state change requested by the atomic skill is satisfied in the final state. "
        "Be conservative: if the final pose is unstable, physically implausible, clearly toppled in a way inappropriate for that object type, or if the wrong object satisfies the relation, mark failure. "
        "If pose is ambiguous because of perspective, partial occlusion, or low image resolution, use the nearby final frames to disambiguate and do not call the object toppled unless side-lying or loss of upright support is clearly visible. "
        "Use earlier frames only to identify the grounded object(s) and destination, but decide success mainly from the final state. "
        "Return only valid JSON with keys: success (boolean), confidence (number between 0 and 1), reason (string), checklist (object). "
        "The checklist object must contain booleans for: grounded_target_tracked, final_relation_satisfied, gripper_state_consistent, object_state_stable, evidence_sufficient. "
        "Set success to true only if every checklist item is true. If any checklist item is false, or if the evidence is insufficient, set success to false. "
        "Base your judgment only on the visible evidence in the frames. If the evidence is insufficient, set success to false."
    )


def _normalize_instruction_text(text: str) -> str:
    normalized = str(text).strip()
    if normalized.lower().startswith("task:"):
        normalized = normalized.split(":", 1)[1].strip()
    return normalized


def _unique_nonempty(values: list[str] | None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _skill_family(item: JudgeInput) -> str:
    skill_desc = (item.skill_description or "").strip().lower()
    instruction = _normalize_instruction_text(item.prompt_used).lower()
    if skill_desc == "move to" or instruction.startswith("navigate to") or instruction.startswith("move to"):
        return "navigation"
    if skill_desc.startswith("place"):
        return "placement"
    if skill_desc == "turn to" or instruction.startswith("turn to"):
        return "orientation"
    if skill_desc.startswith(("turn on", "turn off", "open", "close")) or instruction.startswith(
        ("turn on", "turn off", "open ", "close ")
    ):
        return "state_change"
    return "generic"


def _normalize_word_token(token: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "", str(token).lower())
    if len(normalized) > 4 and normalized.endswith("ies"):
        return normalized[:-3] + "y"
    if len(normalized) > 3 and normalized.endswith("s") and not normalized.endswith("ss"):
        return normalized[:-1]
    return normalized


def _tokenize_words(text: str) -> list[str]:
    return [token for token in (_normalize_word_token(x) for x in re.findall(r"[a-z0-9]+", str(text).lower())) if token]


def _target_object_token_sets(item: JudgeInput) -> list[set[str]]:
    source_names = _unique_nonempty(item.manipulating_object_ids) or _unique_nonempty(item.object_ids)
    token_sets: list[set[str]] = []
    seen: set[tuple[str, ...]] = set()
    ignored = {"a", "an", "the", "of"}
    for name in source_names:
        tokens = tuple(token for token in _tokenize_words(_sanitize_object_name(name)) if token not in ignored)
        if not tokens or tokens in seen:
            continue
        seen.add(tokens)
        token_sets.append(set(tokens))
    return token_sets


def _relevant_task_prompt_clauses(item: JudgeInput) -> list[str]:
    task_prompt = str(item.task_prompt or "").strip()
    if not task_prompt:
        return []

    target_token_sets = _target_object_token_sets(item)
    if not target_token_sets:
        return []

    clauses = [clause.strip(" .") for clause in re.split(r"[;\n]+", task_prompt) if clause.strip()]
    matched: list[str] = []
    seen: set[str] = set()
    for clause in clauses:
        clause_tokens = set(_tokenize_words(clause))
        for target_tokens in target_token_sets:
            min_overlap = 1 if len(target_tokens) <= 1 else min(2, len(target_tokens))
            if len(clause_tokens & target_tokens) >= min_overlap:
                if clause not in seen:
                    seen.add(clause)
                    matched.append(clause)
                break
    return matched


def _task_context_lines(item: JudgeInput) -> list[str]:
    clauses = _relevant_task_prompt_clauses(item)
    if not clauses:
        return []
    lines = ["Relevant full-task clause(s) for this object category and destination:"]
    lines.extend(f"- {clause}" for clause in clauses)
    return lines


def _grounding_lines(item: JudgeInput) -> list[str]:
    skill_desc = (item.skill_description or "").strip().lower()
    family = _skill_family(item)
    manipulated = _unique_nonempty(item.manipulating_object_ids)
    referenced = _unique_nonempty(item.object_ids)
    manipulated_set = set(manipulated)
    remaining = [name for name in referenced if name not in manipulated_set]

    lines: list[str] = []
    if family == "navigation":
        destinations = remaining or referenced
        if destinations:
            lines.append(f"- Destination object(s): {_format_name_list(destinations)}")
        if manipulated:
            lines.append(f"- Moving target object(s): {_format_name_list(manipulated)}")
        return lines

    if family == "placement":
        if manipulated:
            lines.append(f"- Manipulated object(s): {_format_name_list(manipulated)}")
        if "next to" in skill_desc and len(remaining) >= 2:
            primary = remaining[:1]
            secondary = remaining[1:]
            if skill_desc.startswith("place on"):
                lines.append(f"- Support object(s): {_format_name_list(primary)}")
            elif skill_desc.startswith("place in"):
                lines.append(f"- Container object(s): {_format_name_list(primary)}")
            else:
                lines.append(f"- Primary reference object(s): {_format_name_list(primary)}")
            lines.append(f"- Proximity reference object(s): {_format_name_list(secondary)}")
            return lines
        if remaining:
            if skill_desc.startswith("place on"):
                lines.append(f"- Support object(s): {_format_name_list(remaining)}")
            elif skill_desc.startswith("place in"):
                lines.append(f"- Container object(s): {_format_name_list(remaining)}")
            else:
                lines.append(f"- Reference object(s): {_format_name_list(remaining)}")
        return lines

    if family == "orientation":
        if manipulated:
            lines.append(f"- Object to orient: {_format_name_list(manipulated)}")
        if remaining:
            lines.append(f"- Orientation reference object(s): {_format_name_list(remaining)}")
        elif referenced and not manipulated:
            lines.append(f"- Target/reference object(s): {_format_name_list(referenced)}")
        return lines

    if family == "state_change":
        state_targets = manipulated or referenced
        if state_targets:
            lines.append(f"- Object whose state should change: {_format_name_list(state_targets)}")
        if remaining and manipulated:
            lines.append(f"- Associated reference object(s): {_format_name_list(remaining)}")
        return lines

    if manipulated:
        lines.append(f"- Primary target object(s): {_format_name_list(manipulated)}")
    if remaining:
        lines.append(f"- Reference object(s): {_format_name_list(remaining)}")
    elif referenced and not manipulated:
        lines.append(f"- Referenced object(s): {_format_name_list(referenced)}")
    return lines


def _object_specific_rules(item: JudgeInput) -> list[str]:
    family = _skill_family(item)
    if family in {"navigation", "state_change"}:
        return []

    source_names = _unique_nonempty(item.manipulating_object_ids) or _unique_nonempty(item.object_ids)
    names = [_sanitize_object_name(x).lower() for x in source_names]
    if not names:
        return []
    joined = " ".join(names)
    rules = [
        "The target object should end in a plausible, stable resting pose appropriate for its object type.",
        "If the target object is still being held, visibly falling, floating, intersecting geometry, or half off the support surface, mark failure.",
    ]
    if any(keyword in joined for keyword in {"bottle", "cup", "can", "glass", "mug"}):
        rules.append(
            "For bottles, cups, cans, glasses, and mugs: success usually requires the object to be upright or naturally standing. "
            "If it is lying on its side or clearly toppled over, mark failure unless the instruction explicitly allows that pose."
        )
        rules.append(
            "Do not mark a bottle as toppled based only on perspective, a visible front label, or a small lean. "
            "Treat it as toppled only if the bottle body is clearly resting on its side, the base is not serving as the support contact, or the main body axis is clearly far from vertical."
        )
        rules.append(
            "If the bottle is resting on its base under gravity and appears roughly vertical in the final frames, count that as upright even if the camera view is oblique or the silhouette is slightly tilted."
        )
    if any(keyword in joined for keyword in {"monitor", "screen", "tv", "frame", "picture"}):
        rules.append(
            "For monitors, screens, TVs, picture frames, and similar upright objects: success requires a natural upright pose and the intended orientation if the task is about facing direction."
        )
    if any(keyword in joined for keyword in {"book", "notebook", "folder", "mouse", "keyboard", "pen", "remote"}):
        rules.append(
            "For books, notebooks, folders, mice, keyboards, pens, and remotes: lying flat can be acceptable if the final spatial relation is correct and stable."
        )
    return rules


def _skill_specific_rules(item: JudgeInput) -> list[str]:
    skill_desc = (item.skill_description or "").strip().lower()
    family = _skill_family(item)
    rules = [f"Skill family: {family}."]
    if skill_desc:
        rules.append(f"Skill type label: {skill_desc}.")

    rules.append("Judge only this atomic skill; do not score the whole task or require unrelated subtasks to be completed.")
    rules.append(
        "Use the grounding information only to identify the intended object(s) or destination. Do not replace them with a different similar-looking object just because the name is generic."
    )
    rules.append(
        "If multiple similar objects are visible, prefer the specifically grounded object(s) referenced below, not a different object in another area of the scene."
    )
    if item.manipulating_object_ids:
        rules.append(
            "When a manipulated target object is specified, track the object that the robot actually manipulates through the rollout and judge that same instance in the final state."
        )
        rules.append(
            "Do not reassign the target to a different similar object already sitting at the destination just because it is easier to see in the final frame."
        )
        rules.append(
            "If the final frame alone is visually ambiguous, use the last several sampled frames to maintain instance identity and infer whether the manipulated object ended upright, supported, and released."
        )

    if family == "placement" and skill_desc == "place on":
        rules.append(
            "For 'place on': the target object must be fully supported by the target surface in the final state, not merely touching, leaning precariously, or resting in an implausible pose."
        )
    elif family == "placement" and skill_desc == "place in":
        rules.append(
            "For 'place in': the target object must clearly end up inside the container or bounded region."
        )
    elif family == "placement" and skill_desc == "place under":
        rules.append(
            "For 'place under': the target object must clearly be beneath the reference object in the final state."
        )
    elif family == "placement" and skill_desc in {"place on next to", "place in next to"}:
        rules.append(
            "For 'place ... next to ...': both the placement relation and the 'next to' relation must hold in the final state."
        )
    elif family == "orientation":
        rules.append(
            "For 'turn to': success requires the target object to end with the correct orientation toward the reference object, not just nearby."
        )
    elif family == "navigation":
        if item.manipulating_object_ids:
            rules.append(
                "For navigation with manipulated target objects: success requires the manipulated target object to end up at the grounded destination area."
            )
        else:
            rules.append(
                "For navigation without manipulated target objects: judge whether the robot/agent reaches the grounded destination area."
            )
        rules.append(
            "Do not reinterpret navigation as moving unrelated scene objects into, onto, or near the destination unless the atomic instruction explicitly says so."
        )
        rules.append(
            "Do not invent an alternative room, cabinet, table, or destination if another similarly named object is visible; stay anchored to the grounded destination object."
        )
    elif family == "state_change":
        rules.append(
            "For state-change skills, judge only whether the grounded object's visible state has changed as requested by the atomic instruction in the final state."
        )
        rules.append(
            "Do not require the surrounding task outcome to be completed if the local state change itself is satisfied."
        )
    return rules


def _gripper_state_rules(item: JudgeInput) -> list[str]:
    instruction = _normalize_instruction_text(item.prompt_used).lower()
    skill_desc = (item.skill_description or "").strip().lower()
    family = _skill_family(item)
    combined = f"{skill_desc} || {instruction}"

    rules = [
        "Read each sampled image as a three-view montage: use the two wrist close-up views for grasp / release / contact evidence, and use the larger scene/global view for object placement, support, and destination relations.",
    ]
    if item.manipulating_object_ids:
        rules.append(
            "When a manipulated target object is specified, do not infer success just because a similar-looking object is visible at the destination in the global view; use wrist views and temporal continuity to verify the same manipulated instance."
        )

    if family == "placement":
        rules.append(
            "For placement skills, success requires both the correct final relation and a completed release: by the end, the manipulated target should no longer be supported, pinched, dragged, or constrained by the gripper, and should instead be stably supported by the environment."
        )
        rules.append(
            "If the target still appears attached to the gripper, moving together with the gripper, partially lifted, or not clearly released in the final frames, mark failure even if the global view makes the object look approximately near the destination."
        )
        rules.append(
            "If release is ambiguous because the object is tiny, partially occluded, or visually similar to another object already at the destination, be conservative and mark failure unless the final wrist views and nearby final frames clearly support release of the grounded manipulated instance."
        )
    elif any(phrase in combined for phrase in ("pick up", "pickup", "grasp", "lift ", "lift the", "pick ", "take ")):
        rules.append(
            "For pickup / grasp / lift skills, continued secure holding of the grounded target in the final frames can be positive evidence of success; do not require release unless the atomic instruction explicitly asks to place, put, set, leave, or deposit the object."
        )
    elif family == "navigation":
        rules.append(
            "For navigation skills, gripper state is supporting evidence for what object is being carried or interacted with. Do not require release at the end unless the atomic instruction explicitly includes a placement action."
        )
    else:
        rules.append(
            "Interpret gripper state according to the atomic instruction and skill family: some skills require release, some require continued holding, and some use gripper contact only as supporting evidence. Do not apply a blanket rule that holding always means failure."
        )

    return rules


def _checklist_rules(item: JudgeInput) -> list[str]:
    family = _skill_family(item)
    rules = [
        "You must reason through the following checklist before deciding success:",
        "grounded_target_tracked: whether you can follow the same grounded / manipulated target instance from earlier frames to the final frames without switching to a different similar-looking object.",
        "final_relation_satisfied: whether the final requested atomic relation or local state change is visibly satisfied for that same grounded instance.",
        "gripper_state_consistent: whether the final hand / gripper / contact evidence is consistent with this skill family.",
        "object_state_stable: whether the grounded target ends in a plausible, supported, stable final state appropriate for the instruction.",
        "evidence_sufficient: whether the wrist views, global view, and nearby final frames provide enough visible evidence to support the judgment.",
    ]
    if family == "placement":
        rules.append(
            "For placement skills, gripper_state_consistent should be true only if the grounded target has clearly been released and is supported by the environment rather than still by the gripper."
        )
    elif family == "navigation":
        rules.append(
            "For navigation skills, gripper_state_consistent means the observed hand / carrying state does not contradict the requested destination outcome; release is not automatically required unless the instruction explicitly includes placement."
        )
    elif family == "state_change":
        rules.append(
            "For state-change skills, gripper_state_consistent is supporting evidence only; focus mainly on whether the requested visible object state change is achieved."
        )
    else:
        rules.append(
            "For pickup / grasp / lift-like skills, gripper_state_consistent can remain true when the target is still securely held at the end, as long as continued holding matches the atomic instruction."
        )
    rules.append(
        "If you are uncertain about release, support, or target identity, set evidence_sufficient to false and success to false instead of guessing."
    )
    return rules


def _task_specific_rules(item: JudgeInput) -> list[str]:
    clauses = _relevant_task_prompt_clauses(item)
    if not clauses:
        return []

    combined = " ".join(clauses).lower()
    combined_tokens = set(_tokenize_words(combined))
    rules = [
        "Keep any stricter final-state constraints from the relevant full-task clause(s) above when they clearly refer to the same object category and destination as this atomic skill."
    ]
    if combined_tokens & {"both", "two", "pair", "all"}:
        rules.append(
            "If the relevant full-task clause requires multiple instances of the same object category, do not mark success unless the required companion object(s) also satisfy that shared final arrangement in the final state."
        )
        rules.append(
            "If a required companion object is already at the destination but is toppled, unstable, still being held, or otherwise violates the shared arrangement, mark failure even if the specifically grounded object alone looks correct."
        )
    if any(phrase in combined for phrase in ("next to each other", "side by side", "adjacent to each other", "together")):
        rules.append(
            "If the relevant full-task clause requires the objects to be next to each other, side by side, adjacent, or together, that companion relation must also hold in the final state."
        )
    return rules


def build_user_text(item: JudgeInput, sampled_frames: list[dict[str, Any]]) -> str:
    instruction = _normalize_instruction_text(item.prompt_used)
    context_lines = [
        f"Atomic skill instruction: {instruction}",
        f"Sampled frame count: {len(sampled_frames)}",
        "Each image is one sampled frame from the rollout video in chronological order.",
        "Each sampled image is a three-view montage: two wrist close-up views on the left and one larger scene/global view on the right.",
        "Use wrist close-up views mainly for grasp, release, and continued-contact evidence; use the larger scene/global view mainly for final spatial relation, support, and destination context.",
        "Focus on whether the final state satisfies this atomic skill instruction.",
        "Use earlier frames only to identify the grounded object(s) or destination, but judge success mainly from the final state.",
    ]

    task_context_lines = _task_context_lines(item)
    if task_context_lines:
        context_lines.extend(task_context_lines)

    grounding_lines = _grounding_lines(item)
    if grounding_lines:
        context_lines.append("Grounding information (use this only to identify the intended object(s) or destination):")
        context_lines.extend(grounding_lines)
    if item.before_image_path is not None and item.before_image_path.is_file():
        context_lines.append(
            "A reference image captured before the rollout will be provided first. Use it only to identify the specific grounded object instance when multiple similar objects are visible."
        )
        context_lines.append(
            "Do not judge success from the reference image itself; judge success from the rollout frames and their final state."
        )
        context_lines.append(
            "Anchor the target identity to the object shown in the reference image and keep following that same instance through the rollout."
        )

    if item.steps is not None and item.max_steps is not None:
        context_lines.append(f"Executed steps: {item.steps}/{item.max_steps}")
    context_lines.append("Judging rules:")
    for rule in _skill_specific_rules(item):
        context_lines.append(f"- {rule}")
    for rule in _gripper_state_rules(item):
        context_lines.append(f"- {rule}")
    for rule in _checklist_rules(item):
        context_lines.append(f"- {rule}")
    for rule in _task_specific_rules(item):
        context_lines.append(f"- {rule}")
    for rule in _object_specific_rules(item):
        context_lines.append(f"- {rule}")
    context_lines.append(
        'Return JSON only, for example: {"success": false, "confidence": 0.23, "reason": "The target object is not in the required final relation.", "checklist": {"grounded_target_tracked": true, "final_relation_satisfied": false, "gripper_state_consistent": false, "object_state_stable": false, "evidence_sufficient": true}}'
    )
    return "\n".join(context_lines)
