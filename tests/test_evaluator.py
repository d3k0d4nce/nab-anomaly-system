import pandas as pd

from nab_system.evaluator import Detection, Window, aggregate, match_detections


def test_episode_can_cover_multiple_windows():
    det = Detection(
        start=pd.Timestamp("2020-01-01 00:00"),
        end=pd.Timestamp("2020-01-03 00:00"),
        alert_ts=pd.Timestamp("2020-01-01 01:00"),
        action="WARNING",
    )
    windows = [
        Window(pd.Timestamp("2020-01-01 00:00"), pd.Timestamp("2020-01-01 06:00")),
        Window(pd.Timestamp("2020-01-02 00:00"), pd.Timestamp("2020-01-02 06:00")),
    ]
    matched, fp, fn, latencies, true_detections = match_detections([det], windows, 300)
    assert len(matched) == 2
    assert fp == []
    assert fn == []
    assert len(true_detections) == 1


def test_detection_outside_window_is_false_positive():
    det = Detection(
        start=pd.Timestamp("2020-01-05"),
        end=pd.Timestamp("2020-01-05 01:00"),
        alert_ts=pd.Timestamp("2020-01-05"),
        action="WARNING",
    )
    windows = [Window(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01 06:00"))]
    matched, fp, fn, latencies, _ = match_detections([det], windows, 300)
    assert matched == []
    assert len(fp) == 1
    assert len(fn) == 1


def test_aggregate_metrics():
    rows = [
        {"tp": 4, "fp": 1, "fn": 1, "fp_per_24h": 0.5, "latency_ms_p95": 10, "cost_units": 20},
        {"tp": 2, "fp": 0, "fn": 0, "fp_per_24h": 0.0, "latency_ms_p95": 5, "cost_units": 5},
    ]
    agg = aggregate(
        rows,
        {"recall_target": 0.8, "precision_target": 0.7, "fp_budget_per_24h": 2, "latency_p95_ms": 1000},
    )
    assert agg["tp"] == 6 and agg["fp"] == 1 and agg["fn"] == 1
    assert agg["recall_ok"] and agg["precision_ok"] and agg["fp_ok"] and agg["latency_ok"]
