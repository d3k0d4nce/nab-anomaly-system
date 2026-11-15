from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .config import all_series_entries, load_config
from .data_loader import build_series_frame, load_labels, load_raw_series


def audit_series(raw_dir: str, entry: Dict[str, Any], split_cfg: Dict[str, float]) -> Dict[str, Any]:
    raw = load_raw_series(raw_dir, entry["path"])
    frame, step, ranges = build_series_frame(raw_dir, entry["path"], split_cfg)
    combined, raw_labels = load_labels(raw_dir)

    diffs = raw["timestamp"].diff().dt.total_seconds().dropna()
    diffs = diffs[diffs > 0]
    expected = step
    gaps = diffs[diffs > expected * 1.5]
    duplicates = int(raw.duplicated(subset=["timestamp"]).sum())
    missing_raw = int(raw["value"].isna().sum())
    imputed = int(frame["is_imputed"].sum())
    missing_regular = int(frame["value"].isna().sum())

    period = max(1, int(round(86400 / step)))
    train_lo, train_hi = ranges["train"]
    train_values = frame["value"].values[train_lo:train_hi]
    if len(train_values) > period * 2 and np.isfinite(train_values).all():
        x = train_values - np.mean(train_values)
        denom = float(np.dot(x, x)) or 1.0
        acf_daily = float(np.dot(x[:-period], x[period:]) / denom)
    else:
        acf_daily = 0.0

    windows = combined.get(entry["path"], [])
    positive = 0
    for start, end in windows:
        positive += int(((frame.index >= pd.Timestamp(start)) & (frame.index <= pd.Timestamp(end))).sum())

    return {
        "series": entry["name"],
        "category": entry.get("category"),
        "criticality": int(entry.get("criticality", 3)),
        "raw_rows": int(len(raw)),
        "regular_rows": int(len(frame)),
        "step_seconds": step,
        "duplicates": duplicates,
        "missing_raw_values": missing_raw,
        "missing_regular_values": missing_regular,
        "imputed_values": imputed,
        "max_gap_seconds": float(gaps.max()) if len(gaps) else 0.0,
        "gap_count": int(len(gaps)),
        "daily_acf": acf_daily,
        "seasonal": bool(abs(acf_daily) >= 0.2),
        "train_rows": int(train_hi - train_lo),
        "validation_rows": int(ranges["validation"][1] - ranges["validation"][0]),
        "test_rows": int(ranges["test"][1] - ranges["test"][0]),
        "label_windows": int(len(windows)),
        "label_positive_points": positive,
        "label_positive_share": positive / max(len(frame), 1),
        "raw_onsets": int(len(raw_labels.get(entry["path"], []))),
    }


def run_audit(cfg_path: str | None = None) -> List[Dict[str, Any]]:
    cfg = load_config(cfg_path)
    rows = [
        audit_series(cfg["paths"]["nab_raw"], entry, cfg["split"])
        for entry in all_series_entries(cfg)
    ]
    out = Path(cfg["paths"]["reports"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "data_audit.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    header = (
        "| series | rows | step | gaps | max gap | imputed | daily ACF | windows | "
        "positive | train/val/test |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n"
    )
    lines = []
    for r in rows:
        lines.append(
            f"| {r['series']} | {r['regular_rows']} | {r['step_seconds']}s | "
            f"{r['gap_count']} | {r['max_gap_seconds']:.0f}s | {r['imputed_values']} | "
            f"{r['daily_acf']:.2f} | {r['label_windows']} | "
            f"{r['label_positive_points']} | "
            f"{r['train_rows']}/{r['validation_rows']}/{r['test_rows']} |"
        )
    (out / "data_audit.md").write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    return rows
