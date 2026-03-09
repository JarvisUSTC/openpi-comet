#!/usr/bin/env python3
"""Check that format_skill_prompt covers all skill types in the dataset."""
import json
from pathlib import Path
from collections import defaultdict

# Use shared formatter from dataset (single source of truth)
from behavior.learning.datas.dataset import format_skill_prompt as _format_skill_prompt

ANNOTATIONS_ROOT = Path("/workspace/data/Behavior1K/2025-challenge-demos/annotations")
# Also check annotations_all if exists (may have more tasks)
ANNOTATIONS_ALL = Path("/workspace/data/Behavior1K/2025-challenge-demos/annotations_all")


def scan_dir(root: Path, skills: dict) -> None:
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
                sd = (s.get("skill_description") or [""])[0]
                obj_groups = s.get("object_id") or [[]]
                objs = obj_groups[0] if obj_groups else []
                if not isinstance(objs, list):
                    objs = [objs]
                objs = [str(o) for o in objs if o is not None]
                skills[sd]["count"] += 1
                skills[sd]["obj_counts"].add(len(objs))
                if len(skills[sd]["samples"]) < 3:
                    prompt = _format_skill_prompt(s)
                    skills[sd]["samples"].append((objs, prompt))


def main():
    skills = defaultdict(lambda: {"count": 0, "obj_counts": set(), "samples": []})
    scan_dir(ANNOTATIONS_ROOT, skills)
    if ANNOTATIONS_ALL.exists():
        scan_dir(ANNOTATIONS_ALL, skills)

    print("=" * 80)
    print("Skill coverage check")
    print("=" * 80)
    for skill_desc in sorted(skills.keys()):
        info = skills[skill_desc]
        print(f"\n{skill_desc!r}")
        print(f"  count: {info['count']}, obj_counts: {info['obj_counts']}")
        for objs, prompt in info["samples"]:
            print(f"  sample: {objs} -> {prompt!r}")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
