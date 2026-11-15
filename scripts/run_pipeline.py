#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nab_system.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--no-reset", action="store_true")
    parser.add_argument("--only", default=None)
    parser.add_argument("--no-stress", action="store_true")
    args = parser.parse_args()
    metrics = run_pipeline(
        cfg_path=args.config,
        reset_db=not args.no_reset,
        only=args.only.split(",") if args.only else None,
        include_stress=not args.no_stress,
    )
    print(metrics["pilot_aggregate_combined"])


if __name__ == "__main__":
    main()
