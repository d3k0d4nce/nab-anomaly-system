from __future__ import annotations

import glob
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

NAB_GIT = "https://github.com/numenta/NAB.git"


def ensure_nab(raw_dir: str | Path) -> Path:
    root = Path(raw_dir)
    if (root / "data").exists() and (root / "labels").exists():
        return root
    root.parent.mkdir(parents=True, exist_ok=True)
    if root.exists():
        raise RuntimeError(f"{root} exists but is not a valid NAB checkout")
    subprocess.run(["git", "clone", "--depth", "1", NAB_GIT, str(root)], check=True)
    return root


def load_raw_series(raw_dir: str | Path, rel_path: str) -> pd.DataFrame:
    path = Path(raw_dir) / "data" / rel_path
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if not {"timestamp", "value"}.issubset(df.columns):
        raise ValueError(f"Unexpected NAB columns in {path}: {df.columns.tolist()}")
    df = df.groupby("timestamp", as_index=False)["value"].mean()
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def infer_step_seconds(df: pd.DataFrame) -> int:
    diffs = df["timestamp"].diff().dt.total_seconds().dropna()
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return 300
    return int(round(float(np.median(diffs))))


def regularize(df: pd.DataFrame, step_seconds: int, max_interp: int = 6) -> pd.DataFrame:
    """Put a series on a regular grid, interpolating only short internal gaps."""
    idx = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq=f"{int(step_seconds)}s")
    s = df.set_index("timestamp")["value"].reindex(idx)
    raw_missing = s.isna()
    s = s.interpolate(method="linear", limit=max_interp, limit_area="inside")
    is_imputed = raw_missing & s.notna()
    out = pd.DataFrame({"value": s.values, "is_imputed": is_imputed.values}, index=idx)
    out.index.name = "timestamp"
    return out


def load_labels(raw_dir: str | Path) -> Tuple[Dict[str, List[List[str]]], Dict[str, List[str]]]:
    root = Path(raw_dir)
    combined = json.loads((root / "labels" / "combined_windows.json").read_text(encoding="utf-8"))
    raw: Dict[str, set] = {}
    for fn in glob.glob(str(root / "labels" / "raw" / "*.json")):
        data = json.loads(Path(fn).read_text(encoding="utf-8"))
        for key, values in data.items():
            raw.setdefault(key, set()).update(values)
    return combined, {k: sorted(v) for k, v in raw.items()}


def split_ranges(index: pd.DatetimeIndex, split_cfg: Dict[str, float]) -> Dict[str, Tuple[int, int]]:
    n = len(index)
    train_end = int(n * float(split_cfg["train"]))
    val_end = int(n * (float(split_cfg["train"]) + float(split_cfg["validation"])))
    train_end = max(1, min(n - 2, train_end))
    val_end = max(train_end + 1, min(n - 1, val_end))
    return {
        "train": (0, train_end),
        "validation": (train_end, val_end),
        "test": (val_end, n),
    }


def split_labels_for_row(i: int, ranges: Dict[str, Tuple[int, int]]) -> str:
    for name in ("train", "validation", "test"):
        lo, hi = ranges[name]
        if lo <= i < hi:
            return name
    return "unknown"


def build_series_frame(
    raw_dir: str | Path,
    rel_path: str,
    split_cfg: Dict[str, float],
) -> Tuple[pd.DataFrame, int, Dict[str, Tuple[int, int]]]:
    raw = load_raw_series(raw_dir, rel_path)
    step = infer_step_seconds(raw)
    frame = regularize(raw, step)
    ranges = split_ranges(frame.index, split_cfg)
    frame["split"] = [
        split_labels_for_row(i, ranges) for i in range(len(frame))
    ]
    return frame, step, ranges


def labels_in_range(
    index: pd.DatetimeIndex,
    combined: Dict[str, List[List[str]]],
    raw: Dict[str, List[str]],
    rel_path: str,
    lo: int,
    hi: int,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp | None, bool]]:
    """Return (start, end, onset, is_test) windows overlapping [lo, hi)."""
    out: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp | None, bool]] = []
    raw_onsets = [pd.Timestamp(t) for t in raw.get(rel_path, [])]
    for start, end in combined.get(rel_path, []):
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        if end_ts < index[lo] or start_ts > index[hi - 1]:
            continue
        cand = [o for o in raw_onsets if start_ts <= o <= end_ts]
        onset = min(cand) if cand else None
        out.append((start_ts, end_ts, onset, False))
    return out
