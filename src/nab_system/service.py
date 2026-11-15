from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

from . import auth
from . import dashboard
from . import repository as repo
from .actions import SKIP, ActionEngine, DecisionRecord
from .config import load_config
from .db import apply_schema, connect
from .detector import AnomalyDetector
from .features import OnlineFeatureState

WARNING = "WARNING"
INCIDENT = "INCIDENT"
ALERT_PAGE_SIZE = 10
QUALITY_PAGE_SIZE = 10


def _alert_view(
    conn,
    names: List[str],
    page: int,
    status: str,
    alert_type: str,
    series: Optional[str],
) -> Dict[str, Any]:
    status = status if status in ("ACTIVE",) + repo.ALERT_STATUSES else "all"
    alert_type = alert_type if alert_type in repo.ALERT_TYPES else "all"
    series_filter = series if series in names else "all"
    filters = {
        "status": status,
        "alert_type": alert_type,
        "series": series_filter,
    }
    total = repo.count_alerts(
        conn,
        status=None if status == "all" else status,
        alert_type=None if alert_type == "all" else alert_type,
        series=None if series_filter == "all" else series_filter,
    )
    pages = max(1, (total + ALERT_PAGE_SIZE - 1) // ALERT_PAGE_SIZE)
    page = min(max(1, int(page)), pages)
    alerts = repo.fetch_alerts_page(
        conn,
        limit=ALERT_PAGE_SIZE,
        offset=(page - 1) * ALERT_PAGE_SIZE,
        status=None if status == "all" else status,
        alert_type=None if alert_type == "all" else alert_type,
        series=None if series_filter == "all" else series_filter,
    )
    return {
        "alerts": alerts,
        "filters": filters,
        "page": page,
        "pages": pages,
        "total": total,
        "page_size": ALERT_PAGE_SIZE,
    }


def _quality_view(conn, page: int) -> Dict[str, Any]:
    total = repo.count_quality_stats(conn)
    pages = max(1, (total + QUALITY_PAGE_SIZE - 1) // QUALITY_PAGE_SIZE)
    page = min(max(1, int(page)), pages)
    stats = repo.fetch_quality_stats(
        conn,
        limit=QUALITY_PAGE_SIZE,
        offset=(page - 1) * QUALITY_PAGE_SIZE,
    )
    return {
        "stats": stats,
        "page": page,
        "pages": pages,
        "total": total,
        "page_size": QUALITY_PAGE_SIZE,
    }


@dataclass
class Runtime:
    name: str
    series_id: int
    step_seconds: int
    criticality: int
    detector: AnomalyDetector
    params_id: int
    online: OnlineFeatureState
    action: ActionEngine
    engineer_ids: List[int] = field(default_factory=list)
    episode_ids: Dict[str, int] = field(default_factory=dict)
    episode_incident: Dict[str, bool] = field(default_factory=dict)
    last_episode_key: Optional[str] = None

    def warmup(self, observations: List[tuple[pd.Timestamp, float]], actions: bool = True) -> None:
        for ts, value in observations:
            features = self.online.update(pd.Timestamp(ts), float(value))
            if actions:
                evidence, probability = self.detector.score_features(features)
                self.action.process(pd.Timestamp(ts), evidence, probability, features)


class ReplayService:
    """Streaming replay service shared by the demo CLI and the FastAPI app."""

    def __init__(self, cfg_path: Optional[str] = None, logger: Optional[logging.Logger] = None):
        self.cfg = load_config(cfg_path)
        self.logger = logger or logging.getLogger("nab_service")
        self.runtimes: Dict[str, Runtime] = {}
        apply_schema(self.cfg)
        with connect(self.cfg) as conn:
            ids = auth.seed_default_users(conn)
        self.engineer_ids = list(ids.values())[:1]

    def load_runtime(self, name: str, before_ts: Optional[pd.Timestamp] = None) -> Runtime:
        if name in self.runtimes:
            return self.runtimes[name]
        detector_path = Path(self.cfg["paths"]["artifacts"]) / "detectors" / f"{name}.pkl"
        if not detector_path.exists():
            raise FileNotFoundError(f"Detector not found: {detector_path}. Run the pipeline first.")
        detector = AnomalyDetector.load(str(detector_path))
        with connect(self.cfg) as conn:
            meta = repo.fetch_series_meta(conn, name)
            if meta is None:
                raise ValueError(f"Unknown series: {name}")
            params_id = repo.insert_detection_params(conn, meta["series_id"], detector.params.as_db_dict())
            runtime = Runtime(
                name=name,
                series_id=meta["series_id"],
                step_seconds=meta["step_seconds"],
                criticality=meta["criticality"],
                detector=detector,
                params_id=params_id,
                online=OnlineFeatureState(detector.ctx),
                action=ActionEngine(
                    detector.params,
                    response_warning_min=int(self.cfg["actions"]["response_warning_min"]),
                    response_incident_min=int(self.cfg["actions"]["response_incident_min"]),
                ),
                engineer_ids=list(self.engineer_ids),
            )
            if before_ts is not None:
                history = repo.fetch_recent_observations(
                    conn,
                    meta["series_id"],
                    before_ts,
                    limit=max(detector.params.min_history, 2 * detector.ctx.period),
                )
                runtime.warmup(history)
        self.runtimes[name] = runtime
        return runtime

    def _persist(self, runtime: Runtime, rec: DecisionRecord, oid: int, episode_id: Optional[int]) -> Optional[int]:
        with connect(self.cfg) as conn:
            dec_map = repo.insert_decisions_bulk(
                conn,
                [
                    (
                        oid,
                        runtime.params_id,
                        episode_id,
                        rec.probability,
                        rec.evidence,
                        rec.action,
                        rec.reason,
                        rec.latency_ms,
                    )
                ],
            )
            decision_id = dec_map.get(oid)
            if decision_id is None:
                return None
            if rec.action not in (WARNING, INCIDENT):
                repo.insert_log(
                    conn,
                    oid,
                    None,
                    "DECISION",
                    f"{rec.action} ({rec.kind}), evidence={rec.evidence:.2f}, p={rec.probability:.3f}",
                )
                return None
            engineer = runtime.engineer_ids[0] if runtime.engineer_ids else None
            limit = (
                int(self.cfg["actions"]["response_warning_min"])
                if rec.action == WARNING
                else int(self.cfg["actions"]["response_incident_min"])
            )
            alert_id = repo.insert_alert(
                conn, decision_id, engineer, rec.action, rec.kind, limit
            )
            repo.insert_log(
                conn,
                oid,
                alert_id,
                "ALERT",
                f"{rec.action} ({rec.kind}), evidence={rec.evidence:.2f}, p={rec.probability:.3f}",
            )
            return alert_id

    def process_point(
        self,
        name: str,
        ts: pd.Timestamp,
        value: float,
        persist: bool = True,
        runtime: Optional[Runtime] = None,
        allow_existing: bool = False,
    ) -> Dict[str, Any]:
        ts = pd.Timestamp(ts)
        if pd.isna(ts):
            raise ValueError("Некорректная метка времени наблюдения")
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Некорректное значение наблюдения") from exc
        if not math.isfinite(value):
            raise ValueError("Значение наблюдения должно быть конечным числом")

        if persist and not allow_existing:
            with connect(self.cfg) as conn:
                series_id = repo.fetch_series_id(conn, name)
                if series_id is None:
                    raise ValueError(f"Unknown series: {name}")
                existing_oid = repo.fetch_observation_id(conn, series_id, ts)
                if existing_oid is not None:
                    existing = repo.fetch_decision_for_observation(conn, existing_oid)
                    if existing is not None:
                        raise ValueError("Наблюдение с такой меткой времени уже обработано")

        rt = runtime or self.load_runtime(name, before_ts=ts)
        history_count = len(rt.online.values)
        min_history = int(rt.detector.params.min_history)
        enough_history = history_count >= min_history
        t0 = time.perf_counter()
        features = rt.online.update(ts, value)
        if enough_history:
            evidence, probability = rt.detector.score_features(features)
            rec = rt.action.process(ts, evidence, probability, features)
        else:
            evidence = 0.0
            probability = 0.0
            rec = DecisionRecord(
                ts=ts,
                action=SKIP,
                reason=(
                    "недостаточно наблюдений для оценки "
                    f"({history_count}/{min_history})"
                ),
                evidence=0.0,
                probability=0.0,
                kind="insufficient_history",
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        rec.latency_ms = max(1, int(round(elapsed_ms)))

        episode_id = None
        if rec.episode_key:
            rt.last_episode_key = rec.episode_key
            episode = (
                rt.action.active
                if rt.action.active is not None and rt.action.active.key == rec.episode_key
                else None
            )
            warning_sent = bool(episode.warning_sent) if episode else rec.action == WARNING
            incident = bool(rt.episode_incident.get(rec.episode_key, False)) or rec.action == INCIDENT
            rt.episode_incident[rec.episode_key] = incident
            with connect(self.cfg) as conn:
                ids = repo.insert_episodes(
                    conn,
                    rt.series_id,
                    [
                        {
                            "started_at": pd.Timestamp(rec.episode_key).to_pydatetime(),
                            "ended_at": pd.Timestamp(ts).to_pydatetime()
                            if rec.action == INCIDENT
                            else None,
                            "peak_evidence": rec.evidence,
                            "warning_sent": warning_sent,
                            "incident_sent": incident,
                            "final_action": INCIDENT if incident else WARNING,
                        }
                    ],
                )
                episode_id = ids[0]
                rt.episode_ids[rec.episode_key] = episode_id
        elif rec.reason.startswith("эпизод завершён") and rt.last_episode_key:
            episode = rt.action.episodes.get(rt.last_episode_key)
            incident = bool(rt.episode_incident.get(rt.last_episode_key, False))
            warning_sent = bool(episode.warning_sent) if episode else True
            with connect(self.cfg) as conn:
                ids = repo.insert_episodes(
                    conn,
                    rt.series_id,
                    [
                        {
                            "started_at": pd.Timestamp(rt.last_episode_key).to_pydatetime(),
                            "ended_at": pd.Timestamp(ts).to_pydatetime(),
                            "peak_evidence": rec.evidence,
                            "warning_sent": warning_sent,
                            "incident_sent": incident,
                            "final_action": INCIDENT if incident else WARNING,
                        }
                    ],
                )
                episode_id = ids[0]

        alert_id = None
        if persist:
            with connect(self.cfg) as conn:
                oid = repo.fetch_observation_id(conn, rt.series_id, ts)
                if oid is None:
                    oid = repo.insert_observation(conn, rt.series_id, ts, value, split="test")
            alert_id = self._persist(rt, rec, oid, episode_id)

        return {
            "series": name,
            "timestamp": pd.Timestamp(ts).isoformat(),
            "value": float(value),
            "evidence": float(evidence),
            "probability": float(probability),
            "action": rec.action,
            "reason": rec.reason,
            "kind": rec.kind,
            "episode_key": rec.episode_key,
            "alert_id": alert_id,
            "latency_ms": rec.latency_ms,
            "features": features,
            "data_quality": {
                "value_finite": True,
                "timestamp_valid": True,
                "history_count": history_count,
                "min_history": min_history,
                "enough_history": enough_history,
            },
        }

    def demo(self, write_log: bool = True) -> Dict[str, Any]:
        name = self.cfg["demo"]["series"]
        runtime = self.load_runtime(name)
        with connect(self.cfg) as conn:
            windows = repo.fetch_test_windows(conn, runtime.series_id)
            if not windows:
                raise RuntimeError(f"No test label windows for {name}")
            window = windows[0]
            onset = repo.normalize_ts(window["onset_ts"] or window["start_ts"])
            start_ts = repo.normalize_ts(window["start_ts"])
            end_ts = repo.normalize_ts(window["end_ts"])
            radius_cfg = int(self.cfg["demo"].get("window_radius", 36))
            radius_obs = radius_cfg if radius_cfg <= 200 else max(6, int(radius_cfg / runtime.step_seconds))
            demo_start = onset - pd.Timedelta(seconds=runtime.step_seconds * radius_obs)
            demo_end = onset + pd.Timedelta(seconds=runtime.step_seconds * radius_obs)
            if demo_start < start_ts:
                demo_start = start_ts - pd.Timedelta(seconds=runtime.step_seconds * 6)
            if demo_end > end_ts:
                demo_end = end_ts
            history = repo.fetch_recent_observations(
                conn,
                runtime.series_id,
                demo_start,
                limit=max(runtime.detector.params.min_history, 2 * runtime.detector.ctx.period),
            )
            frame = repo.fetch_observations_between(conn, runtime.series_id, demo_start, demo_end)
        runtime.warmup(history)
        rows: List[Dict[str, Any]] = []
        for ts, row in frame.iterrows():
            rows.append(
                self.process_point(
                    name,
                    ts,
                    float(row["value"]),
                    persist=True,
                    runtime=runtime,
                    allow_existing=True,
                )
            )

        failure_start = next(
            (pd.Timestamp(r["timestamp"]) for r in rows if r["action"] in (WARNING, INCIDENT)),
            onset,
        )
        recovery_start = next(
            (
                pd.Timestamp(r["timestamp"])
                for r in rows
                if pd.Timestamp(r["timestamp"]) > failure_start
                and r["reason"].startswith("эпизод завершён")
            ),
            None,
        )
        lines = [
            f"DEMO {name}",
            f"window: {start_ts} .. {end_ts}",
            f"onset:  {onset}",
            f"replay: {demo_start} .. {demo_end}",
            f"failure start: {failure_start}",
            f"recovery start: {recovery_start}",
            "",
            f"{'timestamp':<20} {'value':>10} {'evidence':>9} {'p':>6}  {'action':<9} phase",
        ]
        for r in rows:
            ts = pd.Timestamp(r["timestamp"])
            if ts < failure_start:
                phase = "normal"
            elif recovery_start is None or ts < recovery_start:
                phase = "failure"
            else:
                phase = "recovery"
            lines.append(
                f"{ts.strftime('%Y-%m-%d %H:%M'):<20} {r['value']:>10.2f} "
                f"{r['evidence']:>9.2f} {r['probability']:>6.3f}  {r['action']:<9} {phase}"
            )
        counts = pd.Series([r["action"] for r in rows]).value_counts().to_dict()
        lines += ["", f"actions: {counts}"]
        text = "\n".join(lines)
        if write_log:
            path = Path(self.cfg["paths"]["reports"]) / "demo_log.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        print(text)
        return {"rows": rows, "counts": counts, "window": window, "onset": onset}


class ObserveRequest(BaseModel):
    series: str
    timestamp: str
    value: float


class AlertCheckRequest(BaseModel):
    is_real_fault: bool
    comment: str = ""
    diagnostics: str = ""


class AlertCloseRequest(BaseModel):
    comment: str = ""


def _safe_next(value: str) -> str:
    return value if value.startswith("/") and not value.startswith("//") else "/"


def create_app(cfg_path: Optional[str] = None):
    app = FastAPI(title="NAB Anomaly Detection Service", version="1.1.0")
    app.add_middleware(
        SessionMiddleware,
        secret_key=auth.session_secret(),
        max_age=8 * 3600,
        same_site="lax",
        https_only=auth.cookie_secure(),
    )
    service = ReplayService(cfg_path)

    def current_user(request: Request) -> Optional[Dict[str, Any]]:
        uid = request.session.get("uid")
        if not uid:
            return None
        with connect(service.cfg) as conn:
            user = repo.fetch_user_by_id(conn, int(uid))
        if not user or not user.get("is_active"):
            request.session.clear()
            return None
        user.pop("password_hash", None)
        return user

    def require_role(minimum: str):
        def dependency(request: Request) -> Dict[str, Any]:
            user = current_user(request)
            if user is None:
                raise HTTPException(status_code=401, detail="Authentication required")
            if not auth.role_allows(user["role"], minimum):
                raise HTTPException(status_code=403, detail="Insufficient role")
            return user

        return dependency

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "series_loaded": len(service.runtimes),
            "auth": "session",
        }

    @app.get("/login", response_class=HTMLResponse)
    def login_page(request: Request, next: str = "/"):
        if current_user(request) is not None:
            return RedirectResponse("/", status_code=303)
        return dashboard.login_html(next_url=_safe_next(next))

    @app.post("/login", response_class=HTMLResponse)
    def login_submit(
        request: Request,
        username: str = Form(...),
        password: str = Form(...),
        next: str = Form("/"),
    ):
        with connect(service.cfg) as conn:
            user = auth.authenticate(conn, username.strip(), password)
        if user is None:
            return HTMLResponse(
                dashboard.login_html("Неверный логин или пароль", _safe_next(next)),
                status_code=401,
            )
        request.session.clear()
        request.session.update(auth.login_to_session(user))
        return RedirectResponse(_safe_next(next), status_code=303)

    @app.get("/logout")
    def logout(request: Request):
        request.session.clear()
        return RedirectResponse("/login", status_code=303)

    @app.get("/", response_class=HTMLResponse)
    def index(
        request: Request,
        page: int = 1,
        qpage: int = 1,
        status: str = "all",
        alert_type: str = "all",
        series: Optional[str] = None,
    ):
        user = current_user(request)
        if user is None:
            return RedirectResponse("/login?next=/", status_code=303)
        with connect(service.cfg) as conn:
            names = repo.fetch_series_names(conn)
            selected = series if series in names else (names[0] if names else "")
            view = _alert_view(conn, names, page, status, alert_type, series)
            quality = _quality_view(conn, qpage)
            counts = repo.fetch_dashboard_counts(conn)
        return dashboard.dashboard_html(
            user,
            names,
            selected,
            view["alerts"],
            quality["stats"],
            counts,
            auth.role_allows(user["role"], "admin"),
            can_operate=auth.role_allows(user["role"], "operator"),
            filters=view["filters"],
            page=view["page"],
            total_alerts=view["total"],
            page_size=view["page_size"],
            total_stats=quality["total"],
            quality_page=quality["page"],
            quality_page_size=quality["page_size"],
        )

    @app.get("/dashboard/partials")
    def dashboard_partials(
        page: int = 1,
        qpage: int = 1,
        status: str = "all",
        alert_type: str = "all",
        series: Optional[str] = None,
        user: Dict[str, Any] = Depends(require_role("operator")),
    ) -> JSONResponse:
        with connect(service.cfg) as conn:
            names = repo.fetch_series_names(conn)
            view = _alert_view(conn, names, page, status, alert_type, series)
            quality = _quality_view(conn, qpage)
            counts = repo.fetch_dashboard_counts(conn)
        can_operate = auth.role_allows(user["role"], "operator")
        return JSONResponse(
            {
                "kpis": dashboard.kpis_html(counts),
                "alerts": dashboard.alert_rows_html(view["alerts"], can_operate),
                "stats": dashboard.stat_rows_html(quality["stats"]),
                "stat_count": dashboard.alert_range_text(
                    quality["page"], quality["total"], quality["page_size"]
                ),
                "stat_pagination": dashboard.quality_pagination_html(
                    quality["page"],
                    quality["total"],
                    view["filters"],
                    view["page"],
                    quality["page_size"],
                ),
                "alert_count": dashboard.alert_range_text(
                    view["page"], view["total"], view["page_size"]
                ),
                "pagination": dashboard.pagination_html(
                    view["page"],
                    view["total"],
                    view["filters"],
                    view["page_size"],
                    quality_page=quality["page"],
                ),
            }
        )

    @app.get("/chart/{series_name}.png")
    def chart(series_name: str, _user: Dict[str, Any] = Depends(require_role("operator"))):
        return Response(
            content=dashboard.render_chart(series_name),
            media_type="image/png",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/me")
    def me(user: Dict[str, Any] = Depends(require_role("operator"))) -> Dict[str, Any]:
        return user

    @app.get("/series")
    def series(_user: Dict[str, Any] = Depends(require_role("operator"))) -> List[str]:
        with connect(service.cfg) as conn:
            return repo.fetch_series_names(conn)

    @app.post("/observe")
    def observe(
        req: ObserveRequest,
        _user: Dict[str, Any] = Depends(require_role("admin")),
    ) -> Dict[str, Any]:
        try:
            return service.process_point(
                req.series, pd.Timestamp(req.timestamp), float(req.value), persist=True
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            status = 409 if "уже обработано" in str(exc) else 400
            raise HTTPException(status_code=status, detail=str(exc)) from exc

    @app.get("/alerts")
    def alerts(
        limit: int = 50,
        _user: Dict[str, Any] = Depends(require_role("operator")),
    ) -> List[Dict[str, Any]]:
        with connect(service.cfg) as conn:
            return repo.fetch_alerts(conn, limit=limit)

    @app.get("/alerts/{alert_id}")
    def alert_detail(
        alert_id: int,
        _user: Dict[str, Any] = Depends(require_role("operator")),
    ) -> Dict[str, Any]:
        with connect(service.cfg) as conn:
            alert = repo.fetch_alert(conn, alert_id)
            if alert is None:
                raise HTTPException(status_code=404, detail="Alert not found")
            return {"alert": alert, "logs": repo.fetch_alert_logs(conn, alert_id)}

    @app.post("/alerts/{alert_id}/ack")
    def ack_alert(
        alert_id: int,
        user: Dict[str, Any] = Depends(require_role("operator")),
    ) -> Dict[str, Any]:
        with connect(service.cfg) as conn:
            alert = repo.fetch_alert(conn, alert_id)
            if alert is None:
                raise HTTPException(status_code=404, detail="Alert not found")
            if alert["status"] == "CLOSED":
                raise HTTPException(status_code=409, detail="Alert already closed")
            repo.update_alert_status(
                conn, alert_id, "ACKNOWLEDGED", int(user["engineer_id"])
            )
            repo.insert_log(
                conn,
                alert.get("observation_id"),
                alert_id,
                "ACK",
                f"{user.get('full_name') or user.get('login')} взял алерт в работу",
            )
            return repo.fetch_alert(conn, alert_id) or {}

    @app.post("/alerts/{alert_id}/check")
    def check_alert(
        alert_id: int,
        req: AlertCheckRequest,
        user: Dict[str, Any] = Depends(require_role("operator")),
    ) -> Dict[str, Any]:
        with connect(service.cfg) as conn:
            alert = repo.fetch_alert(conn, alert_id)
            if alert is None:
                raise HTTPException(status_code=404, detail="Alert not found")
            if alert["status"] == "CLOSED":
                raise HTTPException(status_code=409, detail="Alert already closed")
            verdict = "confirmed" if req.is_real_fault else "false_alarm"
            repo.upsert_check_result(
                conn,
                alert_id,
                int(user["engineer_id"]),
                bool(req.is_real_fault),
                verdict,
                req.comment.strip() or None,
            )
            repo.update_alert_status(
                conn,
                alert_id,
                "ACKNOWLEDGED" if req.is_real_fault else "CLOSED",
                int(user["engineer_id"]),
            )
            who = user.get("full_name") or user.get("login")
            detail = f": {req.comment.strip()}" if req.comment.strip() else ""
            repo.insert_log(
                conn,
                alert.get("observation_id"),
                alert_id,
                "CHECK",
                f"{who} зафиксировал {verdict}{detail}",
            )
            if req.is_real_fault:
                diagnosis = req.diagnostics.strip() or "Требуется диагностика и устранение"
                repo.insert_log(
                    conn,
                    alert.get("observation_id"),
                    alert_id,
                    "DIAGNOSIS",
                    diagnosis,
                )
            return {
                "alert": repo.fetch_alert(conn, alert_id),
                "check": repo.fetch_check_result(conn, alert_id),
                "stats": repo.fetch_operational_stats(conn),
            }

    @app.post("/alerts/{alert_id}/close")
    def close_alert(
        alert_id: int,
        req: AlertCloseRequest,
        user: Dict[str, Any] = Depends(require_role("operator")),
    ) -> Dict[str, Any]:
        with connect(service.cfg) as conn:
            alert = repo.fetch_alert(conn, alert_id)
            if alert is None:
                raise HTTPException(status_code=404, detail="Alert not found")
            if alert["status"] != "CLOSED":
                repo.update_alert_status(
                    conn, alert_id, "CLOSED", int(user["engineer_id"])
                )
                who = user.get("full_name") or user.get("login")
                detail = f": {req.comment.strip()}" if req.comment.strip() else ""
                repo.insert_log(
                    conn,
                    alert.get("observation_id"),
                    alert_id,
                    "CLOSE",
                    f"{who} закрыл алерт{detail}",
                )
            return {
                "alert": repo.fetch_alert(conn, alert_id),
                "stats": repo.fetch_operational_stats(conn),
            }

    @app.get("/stats")
    def stats(
        limit: int = 100,
        _user: Dict[str, Any] = Depends(require_role("operator")),
    ) -> List[Dict[str, Any]]:
        with connect(service.cfg) as conn:
            return repo.fetch_quality_stats(conn, limit=limit)

    @app.get("/stats/operational")
    def operational_stats(
        _user: Dict[str, Any] = Depends(require_role("operator")),
    ) -> Dict[str, Any]:
        with connect(service.cfg) as conn:
            return repo.fetch_operational_stats(conn)

    return app
