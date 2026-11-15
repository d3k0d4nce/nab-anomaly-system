#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nab_system.config import load_config
from nab_system.db import apply_schema, reset_schema


def main() -> None:
    reset = "--reset" in sys.argv
    cfg = load_config()
    if reset:
        reset_schema(cfg)
        print("Database reset and schema applied.")
    else:
        apply_schema(cfg)
        print("Schema applied.")


if __name__ == "__main__":
    main()
