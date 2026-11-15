from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import auth
from . import repository as repo
from .actions import ActionEngine, DecisionRecord
from .audit import run_audit
from .baseline import RollingZBaseline
from .config import all_series_entries, load_config
from .data_loader import build_series_frame, ensure_nab, load_labels
from .db import apply_schema, connect, reset_schema
from .detector import AnomalyDetector, DetectorParams
from .evaluator import Detection, Window, aggregate, evaluate_series

WARNING = "WARNING"
INCIDENT = "INCIDENT"


def _setup_logger(path: str) -> logging.Logger:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("nab_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def to_py(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: to_py(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_py(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def labels_from_windows(index: pd.DatetimeIndex, windows: Sequence[Sequence[str]]) -> np.ndarray:
    labels = np.zeros(len(index), dtype=int)
    for start, end in windows:
        labels[(index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))] = 1
    return labels


def run_engine(
    score_frame: pd.DataFrame,
    params: DetectorParams,
    response: Dict[str, Any],
    per_row_ms: float,
) -> ActionEngine:
    engine = ActionEngine(
        params,
        response_warning_min=int(response.get("response_warning_min", 5)),
        response_incident_min=int(response.get("response_incident_min", 2)),
    )
    for ts, row in score_frame.iterrows():
        engine.process(
            ts,
            float(row["evidence"]),
            float(row["probability"]),
            {k: float(row[k]) for k in ("z_short", "z_long", "z_diff", "z_seas", "z_level")},
        )
    engine.finish(pd.Timestamp(score_frame.index[-1]))
    latency = max(1, int(round(per_row_ms)))
    for rec in engine.decisions:
        rec.latency_ms = latency
    return engine


def build_detections(engine: ActionEngine) -> List[Detection]:
    detections: List[Detection] = []
    last_ts = engine.decisions[-1].ts if engine.decisions else None
    for ep in sorted(engine.episodes.values(), key=lambda e: e.started_at):
        alerts = [
            rec
            for rec in engine.decisions
            if rec.episode_key == ep.key and rec.action in (WARNING, INCIDENT)
        ]
        if not alerts:
            continue
        first = alerts[0]
        action = INCIDENT if ep.incident_sent else WARNING
        end = ep.ended_at if ep.ended_at is not None else (last_ts or ep.started_at)
        detections.append(
            Detection(
                start=pd.Timestamp(ep.started_at),
                end=pd.Timestamp(end),
                alert_ts=pd.Timestamp(first.ts),
                action=action,
                latency_ms=first.latency_ms,
            )
        )
    return detections


def score_with_timing(
    detector: AnomalyDetector, series: pd.Series, step: int
) -> Tuple[pd.DataFrame, float]:
    t0 = time.perf_counter()
    frame = detector.score(series, step)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return frame, max(elapsed_ms / max(len(series), 1), 1e-3)


def fit_and_evaluate(
    cfg: Dict[str, Any],
    entry: Dict[str, Any],
    frame: pd.DataFrame,
    step: int,
    ranges: Dict[str, Tuple[int, int]],
    labels: np.ndarray,
    windows: Sequence[Tuple[pd.Timestamp, pd.Timestamp, Optional[pd.Timestamp], bool]],
    logger: logging.Logger,
) -> Dict[str, Any]:
    series = frame["value"]
    criticality = int(entry.get("criticality", 3))
    detector = AnomalyDetector(cfg).fit(series, step, labels, ranges, criticality)
    score_frame, per_row_ms = score_with_timing(detector, series, step)
    combined_engine = run_engine(score_frame, detector.params, cfg["actions"], per_row_ms)

    train_lo, train_hi = ranges["train"]
    train_mask = np.zeros(len(series), dtype=bool)
    train_mask[train_lo:train_hi] = labels[train_lo:train_hi] == 0
    baseline = RollingZBaseline().fit(series, step, train_mask)
    baseline_score = baseline.score(series)
    baseline_engine = run_engine(baseline_score, detector.params, cfg["actions"], per_row_ms)

    windows_obj = [Window(start, end, onset) for start, end, onset, is_test in windows if is_test]
    targets = cfg["evaluation"]

    def decision_dicts(engine: ActionEngine) -> List[Dict[str, Any]]:
        return [
            {"index": i, "ts": rec.ts, "action": rec.action, "latency_ms": rec.latency_ms}
            for i, rec in enumerate(engine.decisions)
        ]

    combined_result = evaluate_series(
        entry["name"],
        frame.index,
        labels,
        decision_dicts(combined_engine),
        build_detections(combined_engine),
        windows_obj,
        step,
        ranges["test"],
        cfg["actions"],
        float(targets["fp_budget_per_24h"]),
        [r.latency_ms for r in combined_engine.decisions],
    )
    baseline_result = evaluate_series(
        entry["name"] + " [baseline]",
        frame.index,
        labels,
        decision_dicts(baseline_engine),
        build_detections(baseline_engine),
        windows_obj,
        step,
        ranges["test"],
        cfg["actions"],
        float(targets["fp_budget_per_24h"]),
        [r.latency_ms for r in baseline_engine.decisions],
    )

    detector_path = Path(cfg["paths"]["artifacts"]) / "detectors" / f"{entry['name']}.pkl"
    detector_path.parent.mkdir(parents=True, exist_ok=True)
    detector.save(str(detector_path))
    logger.info(
        "%s: recall=%.2f precision=%.2f fp/24h=%.2f cost=%.1f",
        entry["name"],
        combined_result["recall"],
        combined_result["precision"],
        combined_result["fp_per_24h"],
        combined_result["cost_units"],
    )
    return {
        "detector": detector,
        "score_frame": score_frame,
        "combined_engine": combined_engine,
        "baseline_engine": baseline_engine,
        "combined_result": combined_result,
        "baseline_result": baseline_result,
        "per_row_ms": per_row_ms,
        "detector_path": str(detector_path),
    }


def persist_series(
    cfg: Dict[str, Any],
    entry: Dict[str, Any],
    frame: pd.DataFrame,
    step: int,
    ranges: Dict[str, Tuple[int, int]],
    windows: Sequence[Tuple[pd.Timestamp, pd.Timestamp, Optional[pd.Timestamp], bool]],
    out: Dict[str, Any],
    logger: logging.Logger,
) -> int:
    with connect(cfg) as conn:
        series_id = repo.upsert_series(
            conn,
            {
                "name": entry["name"],
                "source": "NAB",
                "category": entry.get("category"),
                "criticality": int(entry.get("criticality", 3)),
                "step_seconds": step,
                "description": entry.get("description"),
            },
        )
        params_id = repo.insert_detection_params(conn, series_id, out["detector"].params.as_db_dict())
        repo.insert_observations(conn, series_id, frame, "train")
        with conn.cursor() as cur:
            for split_name, (lo, hi) in ranges.items():
                cur.execute(
                    "UPDATE observations SET split = %s WHERE series_id = %s AND ts >= %s AND ts <= %s",
                    (
                        split_name,
                        series_id,
                        frame.index[lo].to_pydatetime(),
                        frame.index[hi - 1].to_pydatetime(),
                    ),
                )
        repo.insert_label_windows(conn, series_id, windows)
        obs_ids = repo.fetch_observation_ids(conn, series_id)

        test_lo, test_hi = ranges["test"]
        test_start, test_end = frame.index[test_lo], frame.index[test_hi - 1]
        episode_rows = [
            ep
            for ep in out["combined_engine"].episode_rows()
            if (ep["ended_at"] is None or ep["ended_at"] >= test_start.to_pydatetime())
            and ep["started_at"] <= test_end.to_pydatetime()
        ]
        episode_ids: Dict[str, int] = {}
        if episode_rows:
            eids = repo.insert_episodes(conn, series_id, episode_rows)
            for ep, eid in zip(episode_rows, eids):
                episode_ids[pd.Timestamp(ep["started_at"]).isoformat()] = eid

        decision_rows: List[Tuple[Any, ...]] = []
        decision_meta: List[DecisionRecord] = []
        for rec in out["combined_engine"].decisions:
            if not (test_start <= rec.ts <= test_end):
                continue
            oid = obs_ids.get(pd.Timestamp(rec.ts))
            if oid is None:
                continue
            decision_rows.append(
                (
                    oid,
                    params_id,
                    episode_ids.get(rec.episode_key or ""),
                    rec.probability,
                    rec.evidence,
                    rec.action,
                    rec.reason,
                    rec.latency_ms,
                )
            )
            decision_meta.append(rec)
        dec_map = repo.insert_decisions_bulk(conn, decision_rows)

        user_ids = auth.seed_default_users(conn)
        engineer_ids = list(user_ids.values())[:1]
        alert_rows: List[Tuple[Any, ...]] = []
        alert_meta: List[Tuple[DecisionRecord, int]] = []
        for rec in decision_meta:
            if rec.action not in (WARNING, INCIDENT):
                continue
            did = dec_map.get(obs_ids.get(pd.Timestamp(rec.ts), -1))
            if did is None:
                continue
            engineer = engineer_ids[len(alert_rows) % len(engineer_ids)] if engineer_ids else None
            limit = (
                int(cfg["actions"]["response_warning_min"])
                if rec.action == WARNING
                else int(cfg["actions"]["response_incident_min"])
            )
            alert_rows.append((did, engineer, rec.action, rec.kind, limit))
            alert_meta.append((rec, did))
        alert_map = repo.insert_alerts_bulk(conn, alert_rows)
        for rec, did in alert_meta:
            aid = alert_map.get(did)
            if aid is not None:
                repo.insert_log(
                    conn,
                    obs_ids.get(pd.Timestamp(rec.ts)),
                    aid,
                    "ALERT",
                    f"{rec.action} ({rec.kind}), evidence={rec.evidence:.2f}, p={rec.probability:.3f}",
                )
        repo.insert_log(
            conn,
            None,
            None,
            "SERIES",
            f"{entry['name']}: step={step}s, test={out['combined_result']['test_start']}..{out['combined_result']['test_end']}",
        )
        repo.insert_quality_stat(
            conn,
            {
                "series_id": series_id,
                "params_id": params_id,
                "split": "test",
                "method": "combined_rules_model",
                **out["combined_result"],
            },
        )
        repo.insert_quality_stat(
            conn,
            {
                "series_id": series_id,
                "params_id": params_id,
                "split": "test",
                "method": "rolling_z_baseline",
                **out["baseline_result"],
            },
        )
    logger.info("persisted %s", entry["name"])
    return series_id


def populate_check_results(cfg: Dict[str, Any]) -> int:
    """Simulate first-line engineer checks for the generated alerts."""
    with connect(cfg) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO check_results(
                    alert_id, engineer_id, is_real_fault, verdict, comment,
                    started_at, finished_at
                )
                SELECT
                    a.alert_id,
                    (SELECT engineer_id FROM duty_engineers
                     WHERE role = 'operator' ORDER BY engineer_id LIMIT 1),
                    EXISTS (
                        SELECT 1
                        FROM observations o2
                        JOIN label_windows lw
                          ON lw.series_id = o2.series_id
                         AND o2.ts BETWEEN lw.start_ts AND lw.end_ts
                        WHERE o2.observation_id = d.observation_id
                    ),
                    CASE WHEN EXISTS (
                        SELECT 1
                        FROM observations o2
                        JOIN label_windows lw
                          ON lw.series_id = o2.series_id
                         AND o2.ts BETWEEN lw.start_ts AND lw.end_ts
                        WHERE o2.observation_id = d.observation_id
                    ) THEN 'confirmed' ELSE 'false_alarm' END,
                    'automatic first-line check',
                    a.created_at,
                    a.created_at + interval '5 minutes'
                FROM alerts a
                JOIN decisions d ON d.decision_id = a.decision_id
                ON CONFLICT (alert_id) DO NOTHING
                """
            )
            inserted = cur.rowcount
            cur.execute(
                """
                UPDATE alerts a
                SET status = CASE
                    WHEN c.is_real_fault THEN 'ACKNOWLEDGED'
                    ELSE 'CLOSED'
                END
                FROM check_results c
                WHERE c.alert_id = a.alert_id
                  AND c.comment = 'automatic first-line check'
                  AND a.status = 'OPEN'
                """
            )
            return inserted


def run_pipeline(
    cfg_path: Optional[str] = None,
    reset_db: bool = True,
    only: Optional[Sequence[str]] = None,
    include_stress: bool = True,
) -> Dict[str, Any]:
    cfg = load_config(cfg_path)
    logger = _setup_logger(str(Path(cfg["paths"]["logs"]) / "pipeline.log"))
    ensure_nab(cfg["paths"]["nab_raw"])
    run_audit(cfg_path)
    combined_labels, raw_labels = load_labels(cfg["paths"]["nab_raw"])
    if reset_db:
        reset_schema(cfg)
    else:
        apply_schema(cfg)

    entries = all_series_entries(cfg) if include_stress else list(cfg["benchmark"])
    benchmark_names = {e["name"] for e in cfg["benchmark"]}
    if only:
        wanted = set(only)
        entries = [e for e in entries if e["name"] in wanted]

    results: List[Dict[str, Any]] = []
    stress_results: List[Dict[str, Any]] = []
    for entry in entries:
        logger.info("processing %s", entry["name"])
        frame, step, ranges = build_series_frame(cfg["paths"]["nab_raw"], entry["path"], cfg["split"])
        labels = labels_from_windows(frame.index, combined_labels.get(entry["path"], []))
        raw_onsets = {pd.Timestamp(t) for t in raw_labels.get(entry["path"], [])}
        test_lo, test_hi = ranges["test"]
        windows: List[Tuple[pd.Timestamp, pd.Timestamp, Optional[pd.Timestamp], bool]] = []
        for start_s, end_s in combined_labels.get(entry["path"], []):
            start_ts, end_ts = pd.Timestamp(start_s), pd.Timestamp(end_s)
            cand = [o for o in raw_onsets if start_ts <= o <= end_ts]
            onset = min(cand) if cand else None
            reference = onset if onset is not None else start_ts
            is_test = frame.index[test_lo] <= reference <= frame.index[test_hi - 1]
            windows.append((start_ts, end_ts, onset, is_test))

        out = fit_and_evaluate(cfg, entry, frame, step, ranges, labels, windows, logger)
        persist_series(cfg, entry, frame, step, ranges, windows, out, logger)

        processed = frame.copy()
        processed["evidence"] = out["score_frame"]["evidence"].values
        processed["probability"] = out["score_frame"]["probability"].values
        processed["action"] = "SKIP"
        for rec in out["combined_engine"].decisions:
            processed.loc[rec.ts, "action"] = rec.action
        out_csv = Path(cfg["paths"]["processed"]) / f"{entry['name']}.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        processed.to_csv(out_csv)

        summary = {
            "series": entry["name"],
            "criticality": int(entry.get("criticality", 3)),
            "step_seconds": step,
            "rows": len(frame),
            "train_rows": ranges["train"][1] - ranges["train"][0],
            "validation_rows": ranges["validation"][1] - ranges["validation"][0],
            "test_rows": ranges["test"][1] - ranges["test"][0],
            "labels_positive": int(labels.sum()),
            "label_windows": len(windows),
            "detector_path": out["detector_path"],
            "params": out["detector"].params.as_db_dict(),
            "calibration": {
                "kind": out["detector"].calibrator.kind,
                "p_warning": out["detector"].params.p_warning,
                "p_incident": out["detector"].params.p_incident,
            },
            "combined": out["combined_result"],
            "baseline": out["baseline_result"],
        }
        if entry["name"] in benchmark_names:
            results.append(summary)
        else:
            stress_results.append(summary)

    checks = populate_check_results(cfg)
    logger.info("simulated engineer checks: %d", checks)

    targets = cfg["evaluation"]
    aggregate_combined = aggregate([r["combined"] for r in results], targets)
    aggregate_baseline = aggregate([r["baseline"] for r in results], targets)
    stress_aggregate = aggregate([r["combined"] for r in stress_results], targets) if stress_results else {}

    metrics = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "split": cfg["split"],
        "thresholds": cfg["detection"]["risk_thresholds"],
        "pilot": results,
        "pilot_aggregate_combined": aggregate_combined,
        "pilot_aggregate_baseline": aggregate_baseline,
        "stress": stress_results,
        "stress_aggregate_combined": stress_aggregate,
        "engineer_checks": checks,
    }
    metrics_path = Path(cfg["paths"]["reports"]) / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(to_py(metrics), ensure_ascii=False, indent=2), encoding="utf-8")

    with connect(cfg) as conn:
        for group, agg, method in (
            ("benchmark", aggregate_combined, "combined_rules_model"),
            ("benchmark", aggregate_baseline, "rolling_z_baseline"),
            ("stress", stress_aggregate, "combined_rules_model"),
        ):
            if not agg:
                continue
            repo.insert_quality_stat(
                conn,
                {
                    "series_id": None,
                    "params_id": None,
                    "split": "test",
                    "method": f"{method}_{group}_aggregate",
                    "tp": agg["tp"],
                    "fp": agg["fp"],
                    "fn": agg["fn"],
                    "recall": agg["recall"],
                    "precision": agg["precision"],
                    "f1": agg["f1"],
                    "fp_per_24h": agg["worst_fp_per_24h"],
                    "p95_latency_ms": int(agg["latency_ms_p95"]),
                    "cost_units": agg["cost_units"],
                },
            )
    logger.info("pilot combined: %s", aggregate_combined)
    logger.info("pilot baseline: %s", aggregate_baseline)
    return metrics
