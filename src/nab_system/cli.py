from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import run_audit
from .config import load_config
from .db import apply_schema, connect, reset_schema
from .pipeline import run_pipeline
from .reports import generate_reports
from .repository import fetch_alerts, fetch_quality_stats
from .service import ReplayService


def cmd_init_db(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    if args.reset:
        reset_schema(cfg)
        print("Database reset and schema applied.")
    else:
        apply_schema(cfg)
        print("Schema applied.")


def cmd_audit(args: argparse.Namespace) -> None:
    rows = run_audit(args.config)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


def cmd_pipeline(args: argparse.Namespace) -> None:
    only = args.only.split(",") if args.only else None
    metrics = run_pipeline(
        cfg_path=args.config,
        reset_db=not args.no_reset,
        only=only,
        include_stress=not args.no_stress,
    )
    print(json.dumps(metrics["pilot_aggregate_combined"], ensure_ascii=False, indent=2))


def cmd_demo(args: argparse.Namespace) -> None:
    service = ReplayService(args.config)
    service.demo(write_log=True)


def cmd_report(args: argparse.Namespace) -> None:
    path = generate_reports(args.config)
    print(f"Report written to {path}")


def cmd_alerts(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    with connect(cfg) as conn:
        print(json.dumps(fetch_alerts(conn, limit=args.limit), ensure_ascii=False, indent=2, default=str))


def cmd_stats(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    with connect(cfg) as conn:
        print(json.dumps(fetch_quality_stats(conn, limit=args.limit), ensure_ascii=False, indent=2, default=str))


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from .service import create_app

    app = create_app(args.config)
    uvicorn.run(app, host=args.host, port=args.port)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nab-anomaly", description="NAB anomaly detection system")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init-db", help="create the PostgreSQL schema")
    p.add_argument("--reset", action="store_true", help="drop existing tables first")
    p.set_defaults(func=cmd_init_db)

    p = sub.add_parser("audit", help="audit NAB series and write data_audit reports")
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("pipeline", help="run audit, training, evaluation and persistence")
    p.add_argument("--no-reset", action="store_true", help="keep existing database rows")
    p.add_argument("--only", default=None, help="comma-separated series names")
    p.add_argument("--no-stress", action="store_true", help="skip stress series")
    p.set_defaults(func=cmd_pipeline)

    p = sub.add_parser("demo", help="replay a normal -> failure -> recovery window")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("report", help="build figures and final_report.md")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("alerts", help="show recent alerts")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_alerts)

    p = sub.add_parser("stats", help="show quality statistics")
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("serve", help="run the FastAPI service")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
