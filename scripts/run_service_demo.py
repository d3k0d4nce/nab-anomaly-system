#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nab_system.service import ReplayService


def main() -> None:
    service = ReplayService()
    out = service.demo(write_log=True)
    print(f"\nactions: {out['counts']}")


if __name__ == "__main__":
    main()
