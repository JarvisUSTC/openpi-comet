#!/usr/bin/env python3
"""Check that format_skill_prompt covers all skill types in the dataset.

Prints one representative example per `skill_description` and flags likely prompt-shape issues.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import os

# Allow running as a standalone script from the repo root (src/ layout).
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

# Use formatter without importing heavy dataset dependencies.
from behavior.learning.datas.skill_prompt import _flatten_objs as _flatten_objs
from behavior.learning.datas.skill_prompt import format_skill_prompt as _format_skill_prompt

DEFAULT_ANNOTATIONS_ROOT = Path("/workspace/data/Behavior1K/2025-challenge-demos/annotations")
DEFAULT_ANNOTATIONS_ALL = Path("/workspace/data/Behavior1K/2025-challenge-demos/annotations_all")


def _skill_desc_text(skill_item: dict) -> str:
    raw = skill_item.get("skill_description")
    return str((raw or [""])[0] if isinstance(raw, list) else (raw or ""))


def _pick_better_sample(old: dict | None, new: dict) -> dict:
    """Pick a single representative sample per skill (prefer higher object count)."""
    if old is None:
        return new
    old_n = len(old.get("objs") or [])
    new_n = len(new.get("objs") or [])
    if new_n > old_n:
        return new
    return old


def _validate_prompt(skill_desc: str, objs_clean: list[str], prompt: str) -> list[str]:
    warnings: list[str] = []
    prompt_l = prompt.lower()

    # Only warn on obvious formatting issues; semantics are handled in format_skill_prompt().
    if " next to" in skill_desc and " next to " not in prompt_l:
        warnings.append("expected 'next to' in prompt")
    if skill_desc == "attach" and " to " not in prompt_l:
        warnings.append("attach expected 'to' in prompt")
    if skill_desc == "hang" and " on " not in prompt_l:
        warnings.append("hang expected 'on' in prompt")
    if skill_desc == "insert" and " into " not in prompt_l:
        warnings.append("insert expected 'into' in prompt")
    return warnings


def _count_role_args(object_id) -> int:
    """Count role arguments (treat nested item lists as 1 arg)."""
    if object_id is None:
        return 0
    if not isinstance(object_id, list):
        return 1
    # Common shape: [[arg0, arg1, arg2]]
    group = object_id[0] if (object_id and isinstance(object_id[0], list)) else object_id
    if not isinstance(group, list):
        return 1
    n = 0
    for x in group:
        # If the arg itself is a list of items, count as a single role arg.
        n += 1
    return n


def scan_dir(root: Path, skills: dict, issues: dict[str, list[str]]) -> None:
    if not root.exists():
        return
    for task_dir in sorted(root.iterdir()):
        if not task_dir.is_dir():
            continue
        for ann_file in task_dir.glob("episode_*.json"):
            try:
                with open(ann_file) as f:
                    d = json.load(f)
            except Exception as e:
                print(f"Skip {ann_file}: {e}")
                continue
            for s in d.get("skill_annotation", []):
                sd = _skill_desc_text(s)
                prompt = _format_skill_prompt(s)
                # Don't attempt to validate per-object coverage here: some skills use lists / role args.
                objs_clean = []
                n_role_args = _count_role_args(s.get("object_id"))

                skills[sd]["count"] += 1
                skills[sd]["obj_counts"].add(n_role_args)
                sample = {
                    "objs": s.get("object_id"),
                    "objs_flat": _flatten_objs(s.get("object_id") or []),
                    "prompt": prompt,
                    "file": str(ann_file),
                }
                skills[sd]["sample"] = _pick_better_sample(skills[sd].get("sample"), sample)

                warns = _validate_prompt(sd, objs_clean, prompt)
                if warns:
                    issues[sd].extend(warns)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-root", type=Path, default=DEFAULT_ANNOTATIONS_ROOT)
    parser.add_argument("--annotations-all", type=Path, default=DEFAULT_ANNOTATIONS_ALL)
    args = parser.parse_args()

    skills = defaultdict(lambda: {"count": 0, "obj_counts": set(), "sample": None})
    issues: dict[str, list[str]] = defaultdict(list)

    scan_dir(args.annotations_root, skills, issues)
    if args.annotations_all.exists():
        scan_dir(args.annotations_all, skills, issues)

    print("=" * 80)
    print("Skill prompt spot-check")
    print("=" * 80)
    if not skills:
        print(
            "No skills found. If you're not running inside the dataset container, pass paths via "
            "`--annotations-root` / `--annotations-all`."
        )
        print("=" * 80)
        return

    for skill_desc in sorted(skills.keys()):
        info = skills[skill_desc]
        print(f"\n{skill_desc!r}")
        print(f"  count: {info['count']}, obj_counts: {sorted(info['obj_counts'])}")
        sample = info.get("sample")
        if sample:
            print(f"  sample_object_id: {sample['objs']}")
            print(f"  sample_flat: {sample['objs_flat']}")
            print(f"  prompt: {sample['prompt']!r}")
            print(f"  source: {sample['file']}")
        uniq = sorted(set(issues.get(skill_desc, [])))
        if uniq:
            print(f"  WARN: {uniq}")

    print("\n" + "=" * 80)
    bad = {k: sorted(set(v)) for k, v in issues.items() if v}
    if bad:
        print(f"Found potential prompt-shape issues in {len(bad)}/{len(skills)} skill types.")
    else:
        print("No prompt-shape issues detected by heuristics.")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        # Allow piping into `head` / `tail` without a noisy traceback.
        try:
            sys.stdout.close()
        finally:
            os._exit(0)
