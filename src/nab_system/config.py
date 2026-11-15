from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    # Allow environment overrides for the database connection.
    db = cfg.setdefault("database", {})
    db["host"] = os.getenv("PGHOST", db.get("host", "localhost"))
    db["port"] = int(os.getenv("PGPORT", db.get("port", 5432)))
    db["dbname"] = os.getenv("PGDATABASE", db.get("dbname", "nab_anomaly"))
    db["user"] = os.getenv("PGUSER") or db.get("user") or os.getenv("USER") or "postgres"
    db["password"] = os.getenv("PGPASSWORD", db.get("password", ""))

    cfg["_root"] = str(ROOT)
    for key in ("nab_raw", "processed", "artifacts", "reports", "figures", "logs"):
        cfg["paths"][key] = str((ROOT / cfg["paths"][key]).resolve())
    return cfg


def benchmark_entries(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(cfg.get("benchmark", []))


def stress_entries(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(cfg.get("stress_series", []))


def all_series_entries(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    return benchmark_entries(cfg) + stress_entries(cfg)


def risk_thresholds(cfg: Dict[str, Any], criticality: int) -> Dict[str, float]:
    table = cfg["detection"].get("risk_thresholds", {})
    entry = table.get(int(criticality)) or table.get(str(int(criticality)))
    if entry:
        return {"warning": float(entry["warning"]), "incident": float(entry["incident"])}
    return {
        "warning": float(cfg["detection"]["warning_evidence"]),
        "incident": float(cfg["detection"]["incident_evidence"]),
    }
