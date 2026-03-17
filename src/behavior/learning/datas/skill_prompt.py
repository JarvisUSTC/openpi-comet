from __future__ import annotations

import json
import re

# Skill -> task prompt formatting (aligned with scripts/check_skill_coverage.py)
_SKILL_PREP_PATTERNS = [
    (" from", 5),
    (" on", 3),
    (" in", 3),
    (" into", 5),
    (" onto", 5),
    (" under", 6),
    (" to", 3),
    (" off", 4),
    (" with", 5),
]


def _sanitize_object_name(obj_id: str) -> str:
    """
    Sanitize object identifiers from BEHAVIOR annotations into human-readable names.

    Typical patterns:
      - "coffee_table_koagbh_0" -> "coffee_table" (drops random suffix + instance id)
      - "food_processor_90" -> "food_processor" (drops trailing digits)
      - "half_log_176_0" -> "half_log" (drops numeric parts)
    """
    s = str(obj_id).strip()
    if not s:
        return s
    # Normalize separators used in some ingredient names.
    s = re.sub(r"_+", "_", s)
    parts = [p for p in s.split("_") if p]

    # Drop trailing random suffix + *single-digit* instance id patterns like "_koagbh_0".
    # This avoids stripping legitimate words like "camera" in "digital_camera_87".
    if (
        len(parts) >= 3
        and parts[-1].isdigit()
        and len(parts[-1]) == 1
        and re.fullmatch(r"[a-z0-9]{6,8}", parts[-2] or "")
    ):
        parts = parts[:-2]

    # Drop trailing numeric suffix(es) (instance ids), leaving the semantic base tokens.
    while len(parts) >= 2 and parts[-1].isdigit():
        parts = parts[:-1]
    # Also drop trailing tokens like "toaster_91" that become ["toaster", "91"] only if we didn't split
    # (handle cases where ids remain unsplit due to earlier normalization).
    if parts:
        parts[-1] = re.sub(r"\d+$", "", parts[-1]).strip()
        if not parts[-1]:
            parts = parts[:-1]

    out = " ".join(parts)
    out = re.sub(r"\s+", " ", out).strip()
    # If everything got stripped (e.g. "toaster_91"), fall back to removing trailing digits in-place.
    if not out:
        out = re.sub(r"\d+$", "", s).strip(" _")
        out = re.sub(r"_+", " ", out).strip()
    return out or s


def _flatten_objs(objs) -> list:
    """Flatten object list (handle nested lists and stringified lists)."""
    out = []
    if objs is None:
        return out
    if not isinstance(objs, list):
        return [objs]
    for o in objs:
        if isinstance(o, list):
            out.extend(_flatten_objs(o))
        elif isinstance(o, str):
            if o.startswith("[") and "]" in o:
                try:
                    parsed = json.loads(o.replace("'", '"'))
                    out.extend(_flatten_objs(parsed) if isinstance(parsed, list) else [parsed])
                except Exception:
                    out.append(o)
            else:
                out.append(o)
        else:
            out.append(o)
    return out


def _skill_desc_text(skill_item: dict) -> str:
    """Extract the canonical skill description string from an annotation item."""
    raw = skill_item.get("skill_description")
    if isinstance(raw, list):
        return str((raw or [""])[0] or "").strip()
    return str(raw or "").strip()


def format_skill_prompt(skill_item: dict) -> str:
    """Format a skill_annotation item into a short task prompt (object names sanitized)."""
    skill_desc = _skill_desc_text(skill_item)

    def _normalize_object_groups(v) -> list:
        if v is None:
            return []
        if isinstance(v, list):
            # Common: [[...]] where inner list holds role args.
            if v and isinstance(v[0], (list, tuple)):
                return v
            return [v]
        if isinstance(v, tuple):
            return [list(v)]
        return [[v]]

    def _clean_token(x) -> str:
        return _sanitize_object_name(str(x).strip())

    def _clean_maybe_list(x) -> list[str]:
        if x is None:
            return []
        if isinstance(x, (list, tuple)):
            return [_clean_token(i) for i in _flatten_objs(list(x)) if i is not None and str(i).strip()]
        return [_clean_token(x)]

    def _join(items: list[str]) -> str:
        # De-duplicate while preserving order (prompts read better than repeating identical tokens).
        seen = set()
        deduped = []
        for i in items:
            if not i:
                continue
            if i in seen:
                continue
            seen.add(i)
            deduped.append(i)
        items = deduped
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return ", ".join(items[:-1]) + f", and {items[-1]}"

    # NOTE: Some skills store objects across multiple groups (e.g. [[obj, container], [reference]]).
    # Flatten one level across groups to preserve nested "item lists" while capturing all role args.
    obj_groups = _normalize_object_groups(skill_item.get("object_id"))
    flat_role_args = []
    for g in obj_groups:
        if isinstance(g, (list, tuple)):
            flat_role_args.extend(list(g))
        else:
            flat_role_args.append(g)
    role_args = [_clean_maybe_list(x) for x in flat_role_args]

    def _arg(i: int) -> str:
        return _join(role_args[i]) if i < len(role_args) else ""

    if not role_args or not any(role_args):
        return skill_desc

    # Explicit per-skill rules for semantic correctness (34 skill types).
    if skill_desc == "move to":
        return f"move to {_arg(0)}".strip()
    if skill_desc == "pick up from":
        return f"pick up {_arg(0)} from {_arg(1)}".strip()
    if skill_desc == "place on":
        return f"place {_arg(0)} on {_arg(1)}".strip()
    if skill_desc == "place in":
        return f"place {_arg(0)} in {_arg(1)}".strip()
    if skill_desc == "place under":
        return f"place {_arg(0)} under {_arg(1)}".strip()
    if skill_desc == "push to":
        return f"push {_arg(0)} to {_arg(1)}".strip()

    if skill_desc == "place in next to":
        return f"place {_arg(0)} in {_arg(1)} next to {_arg(2)}".strip()
    if skill_desc == "place on next to":
        return f"place {_arg(0)} on {_arg(1)} next to {_arg(2)}".strip()

    if skill_desc == "attach":
        return f"attach {_arg(0)} to {_arg(1)}".strip()
    if skill_desc == "hang":
        return f"hang {_arg(0)} on {_arg(1)}".strip()
    if skill_desc == "insert":
        return f"insert {_arg(0)} into {_arg(1)}".strip()
    if skill_desc == "spray":
        return f"spray {_arg(0)} on {_arg(1)}".strip()

    if skill_desc == "chop":
        # Convention: [tool, target]
        tool, target = _arg(0), _arg(1)
        return f"chop {target} with {tool}".strip()
    if skill_desc == "ignite":
        # Convention: [tool, target]
        tool, target = _arg(0), _arg(1)
        return f"ignite {target} with {tool}".strip()
    if skill_desc == "sweep surface":
        # Convention: [tool, surface]
        tool, surface = _arg(0), _arg(1)
        return f"sweep {surface} with {tool}".strip()
    if skill_desc == "sweep off":
        # Convention: [[items...], surface]
        items, surface = _arg(0), _arg(1)
        return f"sweep {items} off {surface}".strip()
    if skill_desc == "pour":
        # Convention observed: [[items...], source, target]
        items = _arg(0)
        src = _arg(1)
        dst = _arg(2)
        if items and src and dst:
            return f"pour {items} from {src} into {dst}".strip()
        if items and dst:
            return f"pour {items} into {dst}".strip()
        return f"pour {items}".strip()

    if skill_desc == "hand over":
        obj = _arg(0)
        src = _arg(1)
        dst = _arg(2)
        if obj and src and dst:
            return f"hand over {obj} from {src} to {dst}".strip()
        if obj and dst:
            return f"hand over {obj} to {dst}".strip()
        return f"hand over {obj}".strip()

    if skill_desc in {"open door", "close door"}:
        verb = skill_desc.split(" ", 1)[0]
        obj = _arg(0)
        return f"{verb} {obj} door".strip()
    if skill_desc in {"open drawer", "close drawer"}:
        verb = skill_desc.split(" ", 1)[0]
        obj = _arg(0)
        return f"{verb} {obj} drawer".strip()
    if skill_desc in {"open lid", "close lid"}:
        verb = skill_desc.split(" ", 1)[0]
        obj = _arg(0)
        return f"{verb} {obj} lid".strip()

    if skill_desc == "pull tray":
        obj = _arg(0)
        return f"pull out {obj} tray".strip()
    if skill_desc == "push tray":
        obj = _arg(0)
        return f"push in {obj} tray".strip()

    if skill_desc == "turn on switch":
        obj = _arg(0)
        return f"turn on {obj} switch".strip()
    if skill_desc == "turn off switch":
        obj = _arg(0)
        return f"turn off {obj} switch".strip()

    if skill_desc == "turn to":
        return f"turn {_arg(0)} to {_arg(1)}".strip()
    if skill_desc == "tip over":
        return f"tip over {_arg(0)}".strip()
    if skill_desc == "press":
        return f"press {_arg(0)}".strip()
    if skill_desc == "hold":
        return f"hold {_arg(0)}".strip()
    if skill_desc == "release":
        return f"release {_arg(0)}".strip()
    if skill_desc == "wipe hard":
        # Convention: [tool, target]
        tool, target = _arg(0), _arg(1)
        return f"wipe {target} hard with {tool}".strip()

    # Fallback: best-effort generic prompt.
    objs_flat = [_clean_token(o) for o in _flatten_objs(obj_groups) if o is not None and str(o).strip()]
    return f"{skill_desc} {' '.join(objs_flat)}".strip()
