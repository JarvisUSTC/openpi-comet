#!/usr/bin/env python3
"""
Generate weighted `--data.skill-list` entries from `scripts/skill_dis.txt`.

Rationale:
In BEHAVIOR-1K skill streaming mode, we sample skill segments with probability proportional to:
  keep_prob(skill) * segment_length

`DataConfig.skill_list` supports "skill:weight" entries which we treat as keep_prob/weight.
Weights are not clipped by the dataset code, so values > 1.0 effectively upsample a skill,
and values < 1.0 downsample it (after normalization).
"""

from __future__ import annotations

import argparse
import math
import re
import statistics
from pathlib import Path


_LINE_RE = re.compile(
    r"^(?P<skill>.+?)\s+(?P<freq>[0-9]+)\s+\(in\s+(?P<tasks>[0-9]+)\s+tasks\)\s*$"
)


def _parse(path: Path) -> list[tuple[str, int, int]]:
    rows: list[tuple[str, int, int]] = []
    for raw in path.read_text().splitlines():
        s = raw.strip()
        if not s or s.lower().startswith("skill name"):
            continue
        m = _LINE_RE.match(s)
        if not m:
            raise ValueError(f"Unrecognized line format in {path}: {raw!r}")
        skill = m.group("skill").strip()
        freq = int(m.group("freq"))
        tasks = int(m.group("tasks"))
        rows.append((skill, freq, tasks))
    if not rows:
        raise ValueError(f"No skill rows parsed from {path}")
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("scripts/skill_dis.txt"))
    p.add_argument("--tasks-threshold", type=int, default=10, help="Only reweight skills with tasks > threshold.")
    p.add_argument("--clip-min", type=float, default=0.1)
    p.add_argument("--clip-max", type=float, default=5.0)
    p.add_argument(
        "--include-all",
        action="store_true",
        default=True,
        help="If set (default), also emit a trailing 'all' entry so unknown skills default to weight=1.0.",
    )
    p.add_argument(
        "--no-include-all",
        dest="include_all",
        action="store_false",
        help="Do not emit the trailing 'all' entry (unknown skills default to weight=0.0).",
    )
    p.add_argument(
        "--ref",
        choices=["median", "mean", "sqrt_median"],
        default="median",
        help="Reference frequency for tasks>threshold skills.",
    )
    args = p.parse_args()

    rows = _parse(args.input)
    head = [(skill, freq, tasks) for (skill, freq, tasks) in rows if tasks > args.tasks_threshold]
    if not head:
        raise ValueError(f"No skills with tasks > {args.tasks_threshold} found in {args.input}")

    head_freqs = [freq for (_s, freq, _t) in head]
    if args.ref == "median":
        ref_freq = float(statistics.median(head_freqs))
    elif args.ref == "mean":
        ref_freq = float(statistics.mean(head_freqs))
    else:
        ref_freq = float(math.sqrt(statistics.median(head_freqs)))

    for skill, freq, tasks in rows:
        if tasks > args.tasks_threshold:
            w = ref_freq / float(freq)
            w = max(float(args.clip_min), min(float(args.clip_max), w))
        else:
            w = 1.0
        # Print one entry per line so bash can safely `mapfile -t` into an array.
        print(f"{skill}:{w:.6g}")

    if args.include_all:
        # Enable "weight mode" with default=1.0 for any skill not present in the stats file.
        print("all")


if __name__ == "__main__":
    main()
