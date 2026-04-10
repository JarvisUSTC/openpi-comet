from __future__ import annotations

import sys

from benchmark.config.settings import bootstrap_environment
from .selector import main as selector_main


def main(argv: list[str] | None = None) -> int:
    bootstrap_environment()
    return selector_main(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
