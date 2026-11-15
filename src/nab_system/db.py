from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, Dict, Iterator

import psycopg2
from psycopg2 import sql

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def _conn_kwargs(cfg: Dict[str, Any], dbname: str | None = None) -> Dict[str, Any]:
    db = cfg["database"]
    return {
        "host": db["host"],
        "port": int(db["port"]),
        "dbname": dbname or db["dbname"],
        "user": db["user"],
        "password": db.get("password") or "",
    }


def ensure_database(cfg: Dict[str, Any]) -> None:
    """Create the target database if it does not exist."""
    target = cfg["database"]["dbname"]
    try:
        conn = psycopg2.connect(**_conn_kwargs(cfg))
        conn.close()
        return
    except psycopg2.OperationalError:
        pass

    admin = psycopg2.connect(**_conn_kwargs(cfg, dbname="postgres"))
    admin.autocommit = True
    try:
        with admin.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (target,))
            if cur.fetchone() is None:
                cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(target)))
    finally:
        admin.close()


@contextlib.contextmanager
def connect(cfg: Dict[str, Any]) -> Iterator[psycopg2.extensions.connection]:
    conn = psycopg2.connect(**_conn_kwargs(cfg))
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def apply_schema(cfg: Dict[str, Any]) -> None:
    ensure_database(cfg)
    with connect(cfg) as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
            # Lightweight migration for databases created before web auth.
            cur.execute("ALTER TABLE duty_engineers ADD COLUMN IF NOT EXISTS login TEXT")
            cur.execute("ALTER TABLE duty_engineers ADD COLUMN IF NOT EXISTS password_hash TEXT")
            cur.execute(
                "ALTER TABLE duty_engineers ADD COLUMN IF NOT EXISTS role TEXT "
                "NOT NULL DEFAULT 'operator'"
            )
            cur.execute(
                "ALTER TABLE duty_engineers ADD COLUMN IF NOT EXISTS is_active BOOLEAN "
                "NOT NULL DEFAULT TRUE"
            )
            cur.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_duty_engineers_login "
                "ON duty_engineers(login)"
            )
            cur.execute(
                "UPDATE duty_engineers SET role = 'operator' "
                "WHERE role IS NULL OR role NOT IN ('operator', 'admin')"
            )
            cur.execute(
                "ALTER TABLE duty_engineers "
                "DROP CONSTRAINT IF EXISTS duty_engineers_role_check"
            )
            cur.execute(
                "ALTER TABLE duty_engineers ADD CONSTRAINT duty_engineers_role_check "
                "CHECK (role IN ('operator', 'admin'))"
            )


def reset_schema(cfg: Dict[str, Any]) -> None:
    ensure_database(cfg)
    with connect(cfg) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DROP TABLE IF EXISTS
                    check_results, log_entries, alerts, decisions,
                    deviation_episodes, detection_params, label_windows,
                    quality_stats, observations, time_series, duty_engineers
                CASCADE;
                """
            )
    apply_schema(cfg)
