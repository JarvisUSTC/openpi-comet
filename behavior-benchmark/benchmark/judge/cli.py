from __future__ import annotations

import sys

from benchmark.config.settings import apply_judge_cli_defaults, bootstrap_environment
from .pipeline import run


def main(argv: list[str] | None = None) -> int:
    bootstrap_environment()
    run(apply_judge_cli_defaults(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
