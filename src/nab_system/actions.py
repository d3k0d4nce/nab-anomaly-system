from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .detector import DetectorParams

SKIP = "SKIP"
WARNING = "WARNING"
INCIDENT = "INCIDENT"


@dataclass
class Episode:
    key: str
    started_at: pd.Timestamp
    ended_at: Optional[pd.Timestamp] = None
    peak_evidence: float = 0.0
    warning_sent: bool = False
    incident_sent: bool = False
    repeat_alerts: int = 0
    final_action: Optional[str] = None
    deviations: List[str] = field(default_factory=list)


@dataclass
class DecisionRecord:
    ts: pd.Timestamp
    action: str
    reason: str
    evidence: float
    probability: float
    kind: str = "level_shift"
    latency_ms: int = 0
    episode_key: Optional[str] = None
    episode_started: bool = False


def classify_deviation(features: Dict[str, float]) -> str:
    candidates = {
        "spike": abs(features.get("z_diff", 0.0)),
        "level_shift": max(abs(features.get("z_short", 0.0)), abs(features.get("z_long", 0.0))),
        "seasonal": abs(features.get("z_seas", 0.0)),
        "trend": abs(features.get("z_level", 0.0)),
    }
    return max(candidates.items(), key=lambda kv: kv[1])[0]


class ActionEngine:
    """Episode state machine with hysteresis, dedup and incident escalation."""

    def __init__(self, params: DetectorParams, response_warning_min: int = 5, response_incident_min: int = 2):
        self.params = params
        self.response_warning_min = int(response_warning_min)
        self.response_incident_min = int(response_incident_min)
        self.active: Optional[Episode] = None
        self.high_run = 0
        self.low_run = 0
        self.candidate_start: Optional[pd.Timestamp] = None
        self.episodes: Dict[str, Episode] = {}
        self.decisions: List[DecisionRecord] = []

    def _high(self, evidence: float, probability: float) -> bool:
        return (
            evidence >= self.params.evidence_warning
            or probability >= self.params.p_warning
        )

    def _incident(self, evidence: float, probability: float) -> bool:
        return (
            evidence >= self.params.evidence_incident
            or probability >= self.params.p_incident
        )

    def _above_exit(self, evidence: float, probability: float) -> bool:
        return (
            evidence >= self.params.exit_ratio * self.params.evidence_warning
            or probability >= self.params.exit_ratio * self.params.p_warning
        )

    def process(
        self,
        ts: pd.Timestamp,
        evidence: float,
        probability: float,
        features: Dict[str, float],
    ) -> DecisionRecord:
        kind = classify_deviation(features)
        high = self._high(evidence, probability)
        incident_level = self._incident(evidence, probability)

        if self.active is not None and self._above_exit(evidence, probability):
            self.low_run = 0
            self.active.peak_evidence = max(self.active.peak_evidence, float(evidence))
            self.active.deviations.append(kind)
            if incident_level and not self.active.incident_sent:
                self.active.incident_sent = True
                self.active.final_action = INCIDENT
                return self._emit(ts, INCIDENT, "INCIDENT: устойчивое отклонение", evidence, probability, kind)
            self.active.repeat_alerts += 1
            return self._emit(ts, SKIP, "внутри активного эпизода", evidence, probability, kind)

        if self.active is not None:
            self.low_run += 1
            if self.low_run >= self.params.end_after:
                self.active.ended_at = ts
                self.episodes[self.active.key] = self.active
                self.active = None
                self.high_run = 0
                self.low_run = 0
                self.candidate_start = None
                return self._emit(ts, SKIP, "эпизод завершён", evidence, probability, kind)
            return self._emit(ts, SKIP, "затухание эпизода", evidence, probability, kind)

        if incident_level:
            start = ts
            key = pd.Timestamp(start).isoformat()
            episode = Episode(
                key=key,
                started_at=pd.Timestamp(start),
                peak_evidence=float(evidence),
            )
            episode.warning_sent = False
            episode.incident_sent = True
            episode.final_action = INCIDENT
            episode.deviations.append(kind)
            self.active = episode
            self.low_run = 0
            self.high_run = 0
            self.candidate_start = None
            return self._emit(
                ts,
                INCIDENT,
                "INCIDENT: критическое отклонение, WARNING не требуется",
                evidence,
                probability,
                kind,
            )

        if high:
            self.high_run += 1
            if self.high_run == 1:
                self.candidate_start = ts
            if self.high_run >= self.params.k_consecutive:
                start = self.candidate_start or ts
                key = pd.Timestamp(start).isoformat()
                episode = Episode(key=key, started_at=pd.Timestamp(start), peak_evidence=float(evidence))
                episode.warning_sent = True
                episode.final_action = WARNING
                episode.deviations.append(kind)
                self.active = episode
                self.low_run = 0
                self.high_run = 0
                self.candidate_start = None
                return self._emit(ts, WARNING, "WARNING: подтверждённое отклонение", evidence, probability, kind)
            return self._emit(ts, SKIP, "накопление подтверждения", evidence, probability, kind)

        self.high_run = 0
        self.candidate_start = None
        return self._emit(ts, SKIP, "норма", evidence, probability, kind)

    def _emit(
        self,
        ts: pd.Timestamp,
        action: str,
        reason: str,
        evidence: float,
        probability: float,
        kind: str,
    ) -> DecisionRecord:
        record = DecisionRecord(
            ts=pd.Timestamp(ts),
            action=action,
            reason=reason,
            evidence=float(evidence),
            probability=float(probability),
            kind=kind,
            episode_key=self.active.key if self.active else None,
            episode_started=action in (WARNING, INCIDENT) and self.active is not None,
        )
        self.decisions.append(record)
        return record

    def finish(self, ts: Optional[pd.Timestamp] = None) -> None:
        if self.active is not None:
            self.active.ended_at = ts or self.active.started_at
            self.episodes[self.active.key] = self.active
            self.active = None

    def episode_rows(self) -> List[Dict[str, Any]]:
        rows = []
        for ep in sorted(self.episodes.values(), key=lambda e: e.started_at):
            row = {
                "started_at": ep.started_at.to_pydatetime(),
                "ended_at": ep.ended_at.to_pydatetime() if ep.ended_at is not None else None,
                "peak_evidence": ep.peak_evidence,
                "warning_sent": ep.warning_sent,
                "incident_sent": ep.incident_sent,
                "repeat_alerts": ep.repeat_alerts,
                "final_action": ep.final_action,
            }
            rows.append(row)
        return rows
