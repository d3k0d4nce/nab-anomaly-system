from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

WARNING = "WARNING"
INCIDENT = "INCIDENT"
SKIP = "SKIP"


@dataclass
class Detection:
    start: pd.Timestamp
    end: pd.Timestamp
    alert_ts: pd.Timestamp
    action: str
    latency_ms: int = 0


@dataclass
class Window:
    start: pd.Timestamp
    end: pd.Timestamp
    onset: Optional[pd.Timestamp] = None


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def match_detections(
    detections: Sequence[Detection],
    windows: Sequence[Window],
    step_seconds: int,
) -> Tuple[List[Tuple[Detection, Window]], List[Detection], List[Window], List[float], List[Detection]]:
    """Match whole episodes (not individual alerts) against labelled windows.

    A single long episode may legitimately cover several labelled windows.
    Repeated warnings inside one episode therefore do not create extra FP.
    """
    used: set[int] = set()
    matched: List[Tuple[Detection, Window]] = []
    false_detections: List[Detection] = []
    latencies: List[float] = []
    true_detections: List[Detection] = []

    for det in sorted(detections, key=lambda d: d.alert_ts):
        overlaps = [
            idx
            for idx, win in enumerate(windows)
            if det.start <= win.end and det.end >= win.start
        ]
        new = [idx for idx in overlaps if idx not in used]
        if not overlaps:
            false_detections.append(det)
            continue
        if new:
            true_detections.append(det)
        for idx in new:
            used.add(idx)
            win = windows[idx]
            reference = win.onset if win.onset is not None else win.start
            delta = max(0.0, (det.alert_ts - reference).total_seconds())
            latencies.append(delta / max(step_seconds, 1))
            matched.append((det, win))

    missed = [win for idx, win in enumerate(windows) if idx not in used]
    return matched, false_detections, missed, latencies, true_detections


def evaluate_series(
    series_name: str,
    index: pd.DatetimeIndex,
    labels: np.ndarray,
    decisions: Sequence[Dict[str, Any]],
    detections: Sequence[Detection],
    windows: Sequence[Window],
    step_seconds: int,
    split_range: Tuple[int, int],
    cost_cfg: Dict[str, Any],
    fp_budget_per_24h: float,
    latency_ms: Sequence[float],
) -> Dict[str, Any]:
    lo, hi = split_range
    test_index = index[lo:hi]
    test_labels = labels[lo:hi]
    test_decisions = [d for d in decisions if lo <= d["index"] < hi]

    windows_in_test = [
        Window(
            max(pd.Timestamp(w.start), test_index[0]),
            min(pd.Timestamp(w.end), test_index[-1]),
            pd.Timestamp(w.onset) if w.onset is not None else None,
        )
        for w in windows
        if pd.Timestamp(w.end) >= test_index[0] and pd.Timestamp(w.start) <= test_index[-1]
    ]
    clipped_detections = [
        Detection(
            max(pd.Timestamp(d.start), test_index[0]),
            min(pd.Timestamp(d.end), test_index[-1]),
            pd.Timestamp(d.alert_ts),
            d.action,
            d.latency_ms,
        )
        for d in detections
        if pd.Timestamp(d.end) >= test_index[0] and pd.Timestamp(d.start) <= test_index[-1]
    ]
    matched, false_detections, missed, latencies, true_detections = match_detections(
        clipped_detections, windows_in_test, step_seconds
    )

    tp = len(matched)
    fp = len(false_detections)
    fn = len(missed)
    tp_warn = sum(1 for d, _ in matched if d.action == WARNING)
    tp_inc = sum(1 for d, _ in matched if d.action == INCIDENT)
    fp_warn = sum(1 for d in false_detections if d.action == WARNING)
    fp_inc = sum(1 for d in false_detections if d.action == INCIDENT)

    recall = safe_div(tp, tp + fn)
    precision = safe_div(tp, tp + fp)
    f1 = safe_div(2 * precision * recall, precision + recall)

    duration_hours = max((test_index[-1] - test_index[0]).total_seconds() / 3600.0, 1e-6)
    fp_per_24h = fp / (duration_hours / 24.0)

    # Cost is charged per detection, so a single episode covering several
    # windows is not billed multiple times.
    cost = (
        sum(float(cost_cfg["cost_true_warning"]) for d in true_detections if d.action == WARNING)
        + sum(float(cost_cfg["cost_true_incident"]) for d in true_detections if d.action == INCIDENT)
        + fp_warn * float(cost_cfg["cost_false_warning"])
        + fp_inc * float(cost_cfg["cost_false_incident"])
        + fn * float(cost_cfg["cost_miss"])
    )

    pred = np.array([d["action"] in (WARNING, INCIDENT) for d in test_decisions], dtype=bool)
    truth = test_labels.astype(bool)
    pt_tp = int(np.sum(pred & truth))
    pt_fp = int(np.sum(pred & ~truth))
    pt_fn = int(np.sum(~pred & truth))
    pt_tn = int(np.sum(~pred & ~truth))

    lat_arr = np.asarray(latency_ms, dtype=float)
    p95 = float(np.percentile(lat_arr, 95)) if lat_arr.size else 0.0
    obs_lat = float(np.percentile(latencies, 95)) if latencies else 0.0

    return {
        "series": series_name,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": max(0, len(windows_in_test) - tp - fn),
        "tp_warning": tp_warn,
        "tp_incident": tp_inc,
        "fp_warning": fp_warn,
        "fp_incident": fp_inc,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "fp_per_24h": fp_per_24h,
        "fp_budget_per_24h": float(fp_budget_per_24h),
        "fp_budget_ok": bool(fp_per_24h <= fp_budget_per_24h),
        "latency_obs_p95": obs_lat,
        "latency_ms_p95": p95,
        "cost_units": cost,
        "point_tp": pt_tp,
        "point_fp": pt_fp,
        "point_fn": pt_fn,
        "point_tn": pt_tn,
        "windows_total": len(windows_in_test),
        "detections_total": len(clipped_detections),
        "test_start": str(test_index[0]) if len(test_index) else None,
        "test_end": str(test_index[-1]) if len(test_index) else None,
    }


def aggregate(results: Iterable[Dict[str, Any]], targets: Dict[str, Any]) -> Dict[str, Any]:
    rows = list(results)
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    recall = safe_div(tp, tp + fn)
    precision = safe_div(tp, tp + fp)
    f1 = safe_div(2 * precision * recall, precision + recall)
    worst_fp = max((r["fp_per_24h"] for r in rows), default=0.0)
    p95 = max((r["latency_ms_p95"] for r in rows), default=0.0)
    cost = sum(r["cost_units"] for r in rows)
    return {
        "series_count": len(rows),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "worst_fp_per_24h": worst_fp,
        "latency_ms_p95": p95,
        "cost_units": cost,
        "recall_ok": recall >= float(targets["recall_target"]),
        "precision_ok": precision >= float(targets["precision_target"]),
        "fp_ok": worst_fp <= float(targets["fp_budget_per_24h"]),
        "latency_ok": p95 <= float(targets["latency_p95_ms"]),
    }
