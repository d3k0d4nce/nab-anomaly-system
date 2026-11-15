from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from psycopg2.extras import execute_values


def seed_engineers(conn, engineers: Sequence[Tuple[str, str]]) -> List[int]:
    ids: List[int] = []
    with conn.cursor() as cur:
        for name, contact in engineers:
            cur.execute(
                """
                INSERT INTO duty_engineers(full_name, contact)
                VALUES (%s, %s)
                ON CONFLICT DO NOTHING
                RETURNING engineer_id
                """,
                (name, contact),
            )
            row = cur.fetchone()
            if row is None:
                cur.execute(
                    "SELECT engineer_id FROM duty_engineers WHERE full_name = %s ORDER BY engineer_id LIMIT 1",
                    (name,),
                )
                row = cur.fetchone()
            ids.append(int(row[0]))
    return ids


def seed_users(conn, users: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """Insert/update dashboard users and return login -> engineer_id."""
    ids: Dict[str, int] = {}
    with conn.cursor() as cur:
        for user in users:
            cur.execute(
                "UPDATE duty_engineers SET login = %s "
                "WHERE full_name = %s AND login IS NULL",
                (user["login"], user["full_name"]),
            )
            cur.execute(
                """
                INSERT INTO duty_engineers(
                    full_name, contact, login, password_hash, role,
                    is_active, is_on_duty
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (login) DO UPDATE
                  SET full_name = EXCLUDED.full_name,
                      contact = EXCLUDED.contact,
                      role = EXCLUDED.role,
                      is_active = EXCLUDED.is_active,
                      is_on_duty = EXCLUDED.is_on_duty,
                      password_hash = EXCLUDED.password_hash
                RETURNING engineer_id
                """,
                (
                    user["full_name"],
                    user.get("contact"),
                    user["login"],
                    user["password_hash"],
                    user.get("role", "operator"),
                    bool(user.get("is_active", True)),
                    bool(user.get("is_on_duty", True)),
                ),
            )
            ids[user["login"]] = int(cur.fetchone()[0])
    return ids


def deactivate_non_duty_users(conn, duty_login: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE duty_engineers SET is_active = FALSE "
            "WHERE login IS DISTINCT FROM %s AND is_active = TRUE",
            (duty_login,),
        )
        return cur.rowcount


def fetch_user_by_login(conn, login: str) -> Optional[Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT engineer_id, full_name, contact, login, password_hash,
                   role, is_active, is_on_duty
            FROM duty_engineers WHERE login = %s
            """,
            (login,),
        )
        row = cur.fetchone()
        cols = [c.name for c in cur.description]
    return dict(zip(cols, row)) if row else None


def fetch_user_by_id(conn, engineer_id: int) -> Optional[Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT engineer_id, full_name, contact, login, password_hash,
                   role, is_active, is_on_duty
            FROM duty_engineers WHERE engineer_id = %s
            """,
            (engineer_id,),
        )
        row = cur.fetchone()
        cols = [c.name for c in cur.description]
    return dict(zip(cols, row)) if row else None


def upsert_series(conn, meta: Dict[str, Any]) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO time_series(name, source, category, criticality, step_seconds, description)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (name) DO UPDATE
              SET source = EXCLUDED.source,
                  category = EXCLUDED.category,
                  criticality = EXCLUDED.criticality,
                  step_seconds = EXCLUDED.step_seconds,
                  description = EXCLUDED.description
            RETURNING series_id
            """,
            (
                meta["name"],
                meta.get("source", "NAB"),
                meta.get("category"),
                int(meta.get("criticality", 3)),
                int(meta["step_seconds"]),
                meta.get("description"),
            ),
        )
        return int(cur.fetchone()[0])


def insert_observations(conn, series_id: int, frame: pd.DataFrame, split: str) -> None:
    rows = [
        (
            series_id,
            ts.to_pydatetime(),
            None if pd.isna(val) else float(val),
            bool(imp),
            split,
        )
        for ts, val, imp in zip(frame.index, frame["value"], frame.get("is_imputed", False))
    ]
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO observations(series_id, ts, value, is_imputed, split)
            VALUES %s
            ON CONFLICT (series_id, ts) DO UPDATE
              SET value = EXCLUDED.value,
                  is_imputed = EXCLUDED.is_imputed,
                  split = EXCLUDED.split
            """,
            rows,
            page_size=2000,
        )


def insert_label_windows(
    conn,
    series_id: int,
    windows: Iterable[Tuple[datetime, datetime, Optional[datetime], bool]],
) -> None:
    rows = [
        (series_id, start, end, onset, "NAB", bool(is_test))
        for start, end, onset, is_test in windows
    ]
    if not rows:
        return
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO label_windows(series_id, start_ts, end_ts, onset_ts, source, is_test)
            VALUES %s
            ON CONFLICT (series_id, start_ts, end_ts) DO UPDATE
              SET onset_ts = EXCLUDED.onset_ts,
                  is_test = EXCLUDED.is_test
            """,
            rows,
        )


def insert_detection_params(conn, series_id: int, params: Dict[str, Any]) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO detection_params(
                series_id, anomaly_threshold, critical_threshold,
                evidence_warning, evidence_incident,
                k_consecutive, min_history, window_size, end_after,
                exit_ratio, valid_from, version
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now(), %s)
            ON CONFLICT (series_id, version) DO UPDATE
              SET anomaly_threshold = EXCLUDED.anomaly_threshold,
                  critical_threshold = EXCLUDED.critical_threshold,
                  evidence_warning = EXCLUDED.evidence_warning,
                  evidence_incident = EXCLUDED.evidence_incident,
                  k_consecutive = EXCLUDED.k_consecutive,
                  min_history = EXCLUDED.min_history,
                  window_size = EXCLUDED.window_size,
                  end_after = EXCLUDED.end_after,
                  exit_ratio = EXCLUDED.exit_ratio
            RETURNING params_id
            """,
            (
                series_id,
                params.get("anomaly_threshold"),
                params.get("critical_threshold"),
                params.get("evidence_warning"),
                params.get("evidence_incident"),
                int(params.get("k_consecutive", 2)),
                int(params.get("min_history", 576)),
                int(params.get("window_size", 288)),
                int(params.get("end_after", 48)),
                float(params.get("exit_ratio", 0.6)),
                params.get("version", "v1"),
            ),
        )
        return int(cur.fetchone()[0])


def insert_episodes(conn, series_id: int, episodes: Sequence[Dict[str, Any]]) -> List[int]:
    ids: List[int] = []
    with conn.cursor() as cur:
        for ep in episodes:
            cur.execute(
                """
                INSERT INTO deviation_episodes(
                    series_id, started_at, ended_at, peak_evidence,
                    warning_sent, incident_sent, repeat_alerts, final_action
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (series_id, started_at) DO UPDATE
                  SET ended_at = EXCLUDED.ended_at,
                      peak_evidence = GREATEST(COALESCE(deviation_episodes.peak_evidence, 0),
                                               COALESCE(EXCLUDED.peak_evidence, 0)),
                      warning_sent = EXCLUDED.warning_sent,
                      incident_sent = EXCLUDED.incident_sent,
                      repeat_alerts = EXCLUDED.repeat_alerts,
                      final_action = EXCLUDED.final_action
                RETURNING episode_id
                """,
                (
                    series_id,
                    ep["started_at"],
                    ep.get("ended_at"),
                    ep.get("peak_evidence"),
                    bool(ep.get("warning_sent", False)),
                    bool(ep.get("incident_sent", False)),
                    int(ep.get("repeat_alerts", 0)),
                    ep.get("final_action"),
                ),
            )
            ids.append(int(cur.fetchone()[0]))
    return ids


def insert_decisions_bulk(conn, rows: Sequence[Tuple[Any, ...]]) -> Dict[int, int]:
    """Insert decisions and return mapping observation_id -> decision_id."""
    if not rows:
        return {}
    with conn.cursor() as cur:
        result = execute_values(
            cur,
            """
            INSERT INTO decisions(
                observation_id, params_id, episode_id, anomaly_score,
                evidence, action, reason, latency_ms
            ) VALUES %s
            ON CONFLICT (observation_id) DO UPDATE
              SET params_id = EXCLUDED.params_id,
                  episode_id = EXCLUDED.episode_id,
                  anomaly_score = EXCLUDED.anomaly_score,
                  evidence = EXCLUDED.evidence,
                  action = EXCLUDED.action,
                  reason = EXCLUDED.reason,
                  latency_ms = EXCLUDED.latency_ms
            RETURNING observation_id, decision_id
            """,
            rows,
            page_size=1000,
            fetch=True,
        )
    return {int(obs): int(dec) for obs, dec in result}


def insert_decision_once(conn, row: Tuple[Any, ...]) -> int:
    """Insert a streaming decision without overwriting historical decisions."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO decisions(
                observation_id, params_id, episode_id, anomaly_score,
                evidence, action, reason, latency_ms
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (observation_id) DO NOTHING
            RETURNING decision_id
            """,
            row,
        )
        result = cur.fetchone()
        if result is not None:
            return int(result[0])
        cur.execute(
            "SELECT decision_id FROM decisions WHERE observation_id = %s",
            (row[0],),
        )
        return int(cur.fetchone()[0])


def insert_alert(
    conn,
    decision_id: int,
    engineer_id: Optional[int],
    alert_type: str,
    deviation_kind: str,
    response_limit_min: int,
    status: str = "OPEN",
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO alerts(decision_id, engineer_id, alert_type, status,
                               deviation_kind, response_limit_min)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (decision_id) DO UPDATE
              SET alert_type = EXCLUDED.alert_type,
                  deviation_kind = EXCLUDED.deviation_kind,
                  response_limit_min = EXCLUDED.response_limit_min
            RETURNING alert_id
            """,
            (decision_id, engineer_id, alert_type, status, deviation_kind, response_limit_min),
        )
        return int(cur.fetchone()[0])


def insert_alerts_bulk(conn, rows: Sequence[Tuple[Any, ...]]) -> Dict[int, int]:
    """rows: (decision_id, engineer_id, alert_type, deviation_kind, response_limit_min)."""
    if not rows:
        return {}
    with conn.cursor() as cur:
        result = execute_values(
            cur,
            """
            INSERT INTO alerts(decision_id, engineer_id, alert_type, deviation_kind, response_limit_min)
            VALUES %s
            ON CONFLICT (decision_id) DO UPDATE
              SET alert_type = EXCLUDED.alert_type,
                  deviation_kind = EXCLUDED.deviation_kind,
                  response_limit_min = EXCLUDED.response_limit_min
            RETURNING decision_id, alert_id
            """,
            rows,
            page_size=1000,
            fetch=True,
        )
    return {int(dec): int(alert) for dec, alert in result}


def insert_log(conn, observation_id: Optional[int], alert_id: Optional[int], entry_type: str, message: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO log_entries(observation_id, alert_id, entry_type, message)
            VALUES (%s, %s, %s, %s)
            """,
            (observation_id, alert_id, entry_type, message),
        )


def insert_quality_stat(conn, stats: Dict[str, Any]) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO quality_stats(
                series_id, params_id, split, method, tp, fp, fn, tn,
                tp_warning, tp_incident, fp_warning, fp_incident,
                recall, precision, f1, fp_per_24h, p95_latency_ms, cost_units
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                      %s, %s, %s, %s, %s, %s)
            """,
            (
                stats.get("series_id"), stats.get("params_id"), stats["split"], stats["method"],
                int(stats.get("tp", 0)), int(stats.get("fp", 0)), int(stats.get("fn", 0)),
                int(stats.get("tn", 0)), int(stats.get("tp_warning", 0)),
                int(stats.get("tp_incident", 0)), int(stats.get("fp_warning", 0)),
                int(stats.get("fp_incident", 0)),
                stats.get("recall"), stats.get("precision"), stats.get("f1"),
                stats.get("fp_per_24h"), stats.get("p95_latency_ms"), stats.get("cost_units"),
            ),
        )


_ALERT_SELECT = """
    SELECT a.alert_id, a.decision_id, a.alert_type, a.status, a.deviation_kind,
           a.response_limit_min, a.created_at, a.engineer_id,
           ts.name, o.observation_id, o.ts, o.value,
           d.anomaly_score, d.evidence,
           c.check_id, c.is_real_fault, c.verdict, c.comment,
           c.started_at, c.finished_at,
           e.full_name AS engineer_name
    FROM alerts a
    JOIN decisions d ON d.decision_id = a.decision_id
    JOIN observations o ON o.observation_id = d.observation_id
    JOIN time_series ts ON ts.series_id = o.series_id
    LEFT JOIN check_results c ON c.alert_id = a.alert_id
    LEFT JOIN duty_engineers e ON e.engineer_id = COALESCE(c.engineer_id, a.engineer_id)
"""


ALERT_STATUSES = ("OPEN", "ACKNOWLEDGED", "CLOSED")
ALERT_TYPES = ("WARNING", "INCIDENT")


def _alert_filter_sql(
    status: Optional[str] = None,
    alert_type: Optional[str] = None,
    series: Optional[str] = None,
) -> Tuple[str, List[Any]]:
    clauses: List[str] = []
    params: List[Any] = []
    if status == "ACTIVE":
        clauses.append("a.status IN ('OPEN', 'ACKNOWLEDGED')")
    elif status in ALERT_STATUSES:
        clauses.append("a.status = %s")
        params.append(status)
    if alert_type in ALERT_TYPES:
        clauses.append("a.alert_type = %s")
        params.append(alert_type)
    if series:
        clauses.append("ts.name = %s")
        params.append(series)
    return (" WHERE " + " AND ".join(clauses) if clauses else ""), params


def fetch_alerts(conn, limit: int = 50) -> List[Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            _ALERT_SELECT
            + " ORDER BY o.ts DESC, a.alert_id DESC LIMIT %s",
            (int(limit),),
        )
        cols = [c.name for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def fetch_alerts_page(
    conn,
    limit: int = 10,
    offset: int = 0,
    status: Optional[str] = None,
    alert_type: Optional[str] = None,
    series: Optional[str] = None,
) -> List[Dict[str, Any]]:
    where, params = _alert_filter_sql(status, alert_type, series)
    with conn.cursor() as cur:
        cur.execute(
            _ALERT_SELECT
            + where
            + " ORDER BY o.ts DESC, a.alert_id DESC LIMIT %s OFFSET %s",
            (*params, int(limit), int(offset)),
        )
        cols = [c.name for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def count_alerts(
    conn,
    status: Optional[str] = None,
    alert_type: Optional[str] = None,
    series: Optional[str] = None,
) -> int:
    where, params = _alert_filter_sql(status, alert_type, series)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT count(*)
            FROM alerts a
            JOIN decisions d ON d.decision_id = a.decision_id
            JOIN observations o ON o.observation_id = d.observation_id
            JOIN time_series ts ON ts.series_id = o.series_id
            """
            + where,
            tuple(params),
        )
        return int(cur.fetchone()[0])


def fetch_alert(conn, alert_id: int) -> Optional[Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(_ALERT_SELECT + " WHERE a.alert_id = %s", (int(alert_id),))
        row = cur.fetchone()
        cols = [c.name for c in cur.description]
    return dict(zip(cols, row)) if row else None


def update_alert_status(
    conn,
    alert_id: int,
    status: str,
    engineer_id: Optional[int] = None,
) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE alerts
            SET status = %s,
                engineer_id = COALESCE(%s, engineer_id)
            WHERE alert_id = %s
            RETURNING alert_id
            """,
            (status, engineer_id, int(alert_id)),
        )
        return cur.fetchone() is not None


def upsert_check_result(
    conn,
    alert_id: int,
    engineer_id: int,
    is_real_fault: bool,
    verdict: str,
    comment: Optional[str] = None,
    started_at: Optional[datetime] = None,
    finished_at: Optional[datetime] = None,
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO check_results(
                alert_id, engineer_id, is_real_fault, verdict, comment,
                started_at, finished_at
            ) VALUES (%s, %s, %s, %s, %s, COALESCE(%s, now()), COALESCE(%s, now()))
            ON CONFLICT (alert_id) DO UPDATE
              SET engineer_id = EXCLUDED.engineer_id,
                  is_real_fault = EXCLUDED.is_real_fault,
                  verdict = EXCLUDED.verdict,
                  comment = EXCLUDED.comment,
                  started_at = COALESCE(check_results.started_at, EXCLUDED.started_at),
                  finished_at = EXCLUDED.finished_at
            RETURNING check_id
            """,
            (
                int(alert_id),
                int(engineer_id),
                bool(is_real_fault),
                verdict,
                comment,
                started_at,
                finished_at,
            ),
        )
        return int(cur.fetchone()[0])


def fetch_check_result(conn, alert_id: int) -> Optional[Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT check_id, alert_id, engineer_id, is_real_fault, verdict,
                   comment, started_at, finished_at
            FROM check_results WHERE alert_id = %s
            """,
            (int(alert_id),),
        )
        row = cur.fetchone()
        cols = [c.name for c in cur.description]
    return dict(zip(cols, row)) if row else None


def fetch_alert_logs(conn, alert_id: int, limit: int = 50) -> List[Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT log_id, observation_id, alert_id, entry_type, message, created_at
            FROM log_entries
            WHERE alert_id = %s
            ORDER BY log_id DESC
            LIMIT %s
            """,
            (int(alert_id), int(limit)),
        )
        cols = [c.name for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def fetch_decision_for_observation(conn, observation_id: int) -> Optional[Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT decision_id, action, evidence, anomaly_score, reason
            FROM decisions WHERE observation_id = %s
            """,
            (int(observation_id),),
        )
        row = cur.fetchone()
        cols = [c.name for c in cur.description]
    return dict(zip(cols, row)) if row else None


def fetch_operational_stats(conn) -> Dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
              count(*) AS alerts,
              count(*) FILTER (WHERE a.status = 'OPEN') AS open_alerts,
              count(*) FILTER (WHERE a.status = 'ACKNOWLEDGED') AS acknowledged_alerts,
              count(*) FILTER (WHERE a.status IN ('OPEN', 'ACKNOWLEDGED')) AS active_alerts,
              count(*) FILTER (WHERE a.status = 'CLOSED') AS closed_alerts,
              count(*) FILTER (WHERE c.check_id IS NOT NULL) AS checks,
              count(*) FILTER (WHERE c.is_real_fault IS TRUE) AS confirmed,
              count(*) FILTER (WHERE c.is_real_fault IS FALSE) AS false_alarms,
              count(*) FILTER (WHERE a.alert_type = 'WARNING') AS warnings,
              count(*) FILTER (WHERE a.alert_type = 'INCIDENT') AS incidents,
              count(*) FILTER (
                  WHERE c.is_real_fault IS TRUE AND a.alert_type = 'WARNING'
              ) AS confirmed_warnings,
              count(*) FILTER (
                  WHERE c.is_real_fault IS TRUE AND a.alert_type = 'INCIDENT'
              ) AS confirmed_incidents,
              avg(EXTRACT(EPOCH FROM (c.finished_at - a.created_at)) / 60.0)
                  FILTER (WHERE c.finished_at IS NOT NULL) AS avg_response_min
            FROM alerts a
            LEFT JOIN check_results c ON c.alert_id = a.alert_id
            """
        )
        row = cur.fetchone()
        cols = [c.name for c in cur.description]
    out = dict(zip(cols, row))
    for key in (
        "alerts",
        "open_alerts",
        "acknowledged_alerts",
        "active_alerts",
        "closed_alerts",
        "checks",
        "confirmed",
        "false_alarms",
        "warnings",
        "incidents",
        "confirmed_warnings",
        "confirmed_incidents",
    ):
        out[key] = int(out.get(key) or 0)
    out["avg_response_min"] = (
        round(float(out["avg_response_min"]), 1)
        if out.get("avg_response_min") is not None
        else None
    )
    return out


def fetch_series_names(conn) -> List[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT name FROM time_series ORDER BY name")
        return [row[0] for row in cur.fetchall()]


def fetch_dashboard_counts(conn) -> Dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM time_series")
        series = int(cur.fetchone()[0])
    counts = fetch_operational_stats(conn)
    counts["series"] = series
    return counts


def fetch_users(conn) -> List[Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT engineer_id, full_name, contact, login, role,
                   is_active, is_on_duty, created_at
            FROM duty_engineers
            ORDER BY engineer_id
            """
        )
        cols = [c.name for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def normalize_ts(ts: Any) -> pd.Timestamp:
    """Return a naive timestamp with the original wall-clock value.

    PostgreSQL returns TIMESTAMPTZ values in the session timezone.  NAB CSV
    timestamps are local wall-clock values, so dropping the offset preserves
    the original series timeline.
    """
    value = pd.Timestamp(ts)
    if value.tz is not None:
        value = value.tz_localize(None)
    return value


def fetch_observation_ids(conn, series_id: int) -> Dict[pd.Timestamp, int]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT ts, observation_id FROM observations WHERE series_id = %s ORDER BY ts",
            (series_id,),
        )
        return {normalize_ts(ts): int(oid) for ts, oid in cur.fetchall()}


def fetch_series_id(conn, name: str) -> Optional[int]:
    with conn.cursor() as cur:
        cur.execute("SELECT series_id FROM time_series WHERE name = %s", (name,))
        row = cur.fetchone()
        return int(row[0]) if row else None


def fetch_series_meta(conn, name: str) -> Optional[Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT series_id, step_seconds, criticality FROM time_series WHERE name = %s",
            (name,),
        )
        row = cur.fetchone()
    if not row:
        return None
    return {"series_id": int(row[0]), "step_seconds": int(row[1]), "criticality": int(row[2])}


def fetch_observation_id(conn, series_id: int, ts: pd.Timestamp) -> Optional[int]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT observation_id FROM observations WHERE series_id = %s AND ts = %s",
            (series_id, normalize_ts(ts).to_pydatetime()),
        )
        row = cur.fetchone()
        return int(row[0]) if row else None


def insert_observation(
    conn,
    series_id: int,
    ts: pd.Timestamp,
    value: float,
    split: str = "test",
    is_imputed: bool = False,
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO observations(series_id, ts, value, is_imputed, split)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (series_id, ts) DO UPDATE
              SET value = EXCLUDED.value,
                  is_imputed = EXCLUDED.is_imputed
            RETURNING observation_id
            """,
            (series_id, normalize_ts(ts).to_pydatetime(), float(value), bool(is_imputed), split),
        )
        return int(cur.fetchone()[0])


def fetch_recent_observations(
    conn, series_id: int, before_ts: pd.Timestamp, limit: int
) -> List[Tuple[pd.Timestamp, float]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT ts, value FROM observations
            WHERE series_id = %s AND ts < %s
            ORDER BY ts DESC
            LIMIT %s
            """,
            (series_id, normalize_ts(before_ts).to_pydatetime(), int(limit)),
        )
        rows = cur.fetchall()
    return [(normalize_ts(ts), float(value)) for ts, value in reversed(rows)]


def fetch_observations_between(
    conn, series_id: int, start_ts: pd.Timestamp, end_ts: pd.Timestamp
) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT ts, value FROM observations
            WHERE series_id = %s AND ts >= %s AND ts <= %s
            ORDER BY ts
            """,
            (
                series_id,
                normalize_ts(start_ts).to_pydatetime(),
                normalize_ts(end_ts).to_pydatetime(),
            ),
        )
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(columns=["value"], index=pd.DatetimeIndex([], name="ts"))
    index = pd.DatetimeIndex([normalize_ts(ts) for ts, _ in rows], name="ts")
    return pd.DataFrame({"value": [float(v) for _, v in rows]}, index=index)


def fetch_decision_window(conn, series_id: int, start_ts: Any, end_ts: Any) -> List[Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT o.ts, o.value, d.evidence, d.action
            FROM observations o
            JOIN decisions d ON d.observation_id = o.observation_id
            WHERE o.series_id = %s AND o.ts BETWEEN %s AND %s
            ORDER BY o.ts
            """,
            (series_id, start_ts, end_ts),
        )
        cols = [c.name for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def fetch_test_windows(conn, series_id: int) -> List[Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT start_ts, end_ts, onset_ts
            FROM label_windows
            WHERE series_id = %s AND is_test = TRUE
            ORDER BY start_ts
            """,
            (series_id,),
        )
        cols = [c.name for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def count_quality_stats(conn) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM quality_stats")
        return int(cur.fetchone()[0])


def fetch_quality_stats(
    conn,
    limit: int = 100,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT q.stats_id, ts.name, q.split, q.method, q.tp, q.fp, q.fn,
                   q.recall, q.precision, q.f1, q.fp_per_24h, q.cost_units
            FROM quality_stats q
            LEFT JOIN time_series ts USING(series_id)
            ORDER BY q.stats_id DESC
            LIMIT %s OFFSET %s
            """,
            (int(limit), int(offset)),
        )
        cols = [c.name for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
