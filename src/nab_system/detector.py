from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import genpareto
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .config import risk_thresholds
from .features import FEATURE_NAMES, FeatureContext, build_features, fit_context


class TailModel:
    """Peaks-over-threshold model for one scale-free feature."""

    def __init__(self, values: Iterable[float], q: float = 0.95):
        x = np.asarray(list(values), dtype=float)
        x = x[np.isfinite(x)]
        self.n = int(x.size)
        self.q = float(q)
        self.u = float(np.quantile(x, q)) if self.n else 0.0
        y = x[x > self.u] - self.u
        self.nex = int(y.size)
        self.c = 0.0
        self.loc = 0.0
        self.scale = max(float(np.mean(y)) if self.nex else 1.0, 1e-9)
        if self.nex >= 40:
            try:
                c, loc, scale = genpareto.fit(y, floc=0)
                if (
                    np.isfinite(c)
                    and np.isfinite(scale)
                    and scale > 0
                    and -1.0 < c < 1.5
                ):
                    self.c, self.loc, self.scale = float(c), float(loc), float(scale)
            except Exception:
                pass
        self.x_sorted = np.sort(x)

    def tail(self, values: np.ndarray) -> np.ndarray:
        v = np.asarray(values, dtype=float)
        ge = self.n - np.searchsorted(self.x_sorted, v, side="left")
        p = (1.0 + ge) / (self.n + 1.0)
        mask = v > self.u
        if np.any(mask):
            y = v[mask] - self.u
            sf = genpareto.sf(y, self.c, loc=self.loc, scale=self.scale)
            p[mask] = (self.nex / max(self.n, 1)) * sf
        return np.clip(p, 1e-12, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "q": self.q,
            "u": self.u,
            "nex": self.nex,
            "c": self.c,
            "loc": self.loc,
            "scale": self.scale,
            "x_sorted": self.x_sorted,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TailModel":
        obj = cls.__new__(cls)
        obj.n = int(data["n"])
        obj.q = float(data["q"])
        obj.u = float(data["u"])
        obj.nex = int(data["nex"])
        obj.c = float(data["c"])
        obj.loc = float(data["loc"])
        obj.scale = float(data["scale"])
        obj.x_sorted = np.asarray(data["x_sorted"], dtype=float)
        return obj


class EvidenceModel:
    def __init__(self, q: float = 0.95):
        self.q = q
        self.tails: Dict[str, TailModel] = {}

    def fit(self, features: pd.DataFrame, normal_mask: np.ndarray) -> "EvidenceModel":
        for name in FEATURE_NAMES:
            values = np.abs(features[name].values[normal_mask])
            self.tails[name] = TailModel(values, q=self.q)
        return self

    def score(self, features: pd.DataFrame) -> np.ndarray:
        columns = []
        for name in FEATURE_NAMES:
            tail = self.tails[name]
            columns.append(-np.log10(tail.tail(np.abs(features[name].values))))
        evidence = np.vstack(columns).T.max(axis=1)
        return np.nan_to_num(evidence, nan=0.0, posinf=12.0, neginf=0.0)


class Calibrator:
    """Monotone mapping evidence -> P(anomaly), fitted on validation labels."""

    def __init__(self, warning_evidence: float = 5.0):
        self.warning_evidence = float(warning_evidence)
        self.isotonic: Optional[IsotonicRegression] = None
        self.logistic: Optional[LogisticRegression] = None
        self.kind = "fallback"

    def fit(self, evidence: np.ndarray, labels: np.ndarray) -> "Calibrator":
        e = np.asarray(evidence, dtype=float)
        y = np.asarray(labels, dtype=int)
        ok = np.isfinite(e)
        e, y = e[ok], y[ok]
        if e.size and 0 < y.sum() < y.size:
            self.isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.isotonic.fit(e, y)
            self.kind = "isotonic"
        else:
            # No label variety in the validation slice: keep the smooth
            # logistic fallback centred on the operational warning evidence.
            self.kind = "fallback"
        return self

    def predict(self, evidence: np.ndarray | float) -> np.ndarray:
        e = np.atleast_1d(np.asarray(evidence, dtype=float))
        if self.isotonic is not None:
            return np.clip(self.isotonic.predict(e), 0.0, 1.0)
        if self.logistic is not None:
            return self.logistic.predict_proba(e.reshape(-1, 1))[:, 1]
        # Smooth fallback: logistic centred at the warning evidence.
        return 1.0 / (1.0 + np.exp(-(e - self.warning_evidence)))

    def predict_one(self, evidence: float) -> float:
        return float(self.predict([evidence])[0])


@dataclass
class DetectorParams:
    evidence_warning: float
    evidence_incident: float
    p_warning: float
    p_incident: float
    k_consecutive: int
    end_after: int
    exit_ratio: float
    min_history: int
    window_size: int
    criticality: int
    version: str = "v1"

    def as_db_dict(self) -> Dict[str, Any]:
        return {
            "anomaly_threshold": self.p_warning,
            "critical_threshold": self.p_incident,
            "evidence_warning": self.evidence_warning,
            "evidence_incident": self.evidence_incident,
            "k_consecutive": self.k_consecutive,
            "end_after": self.end_after,
            "exit_ratio": self.exit_ratio,
            "min_history": self.min_history,
            "window_size": self.window_size,
            "version": self.version,
        }


class AnomalyDetector:
    """Combined rule features + tail model + calibrated probability."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.ctx: Optional[FeatureContext] = None
        self.evidence_model: Optional[EvidenceModel] = None
        self.calibrator: Optional[Calibrator] = None
        self.params: Optional[DetectorParams] = None

    def fit(
        self,
        series: pd.Series,
        step_seconds: int,
        labels: np.ndarray,
        ranges: Dict[str, tuple[int, int]],
        criticality: int = 3,
    ) -> "AnomalyDetector":
        dcfg = self.cfg["detection"]
        train_lo, train_hi = ranges["train"]
        val_lo, val_hi = ranges["validation"]
        min_history = int(dcfg.get("min_history", 576))
        warmup = min(train_hi - train_lo - 1, max(min_history, 2 * int(round(86400 / step_seconds))))
        train_mask = np.zeros(len(series), dtype=bool)
        train_mask[train_lo:train_hi] = labels[train_lo:train_hi] == 0
        train_mask[: warmup] = False

        self.ctx = fit_context(
            series,
            step_seconds,
            train_mask,
            seasonal_days=int(dcfg.get("seasonal_days", 7)),
        )
        features = build_features(series, step_seconds, self.ctx)
        self.evidence_model = EvidenceModel(q=float(dcfg.get("tail_quantile", 0.95)))
        self.evidence_model.fit(features, train_mask)
        evidence = self.evidence_model.score(features)

        val_mask = np.zeros(len(series), dtype=bool)
        val_mask[val_lo:val_hi] = True
        thresholds = risk_thresholds(self.cfg, criticality)
        self.calibrator = Calibrator(warning_evidence=thresholds["warning"])
        self.calibrator.fit(evidence[val_mask], labels[val_mask])

        p_warn = self.calibrator.predict_one(thresholds["warning"])
        p_inc = self.calibrator.predict_one(thresholds["incident"])
        if p_inc <= p_warn:
            p_inc = min(1.0, p_warn + 0.1)
        self.params = DetectorParams(
            evidence_warning=float(thresholds["warning"]),
            evidence_incident=float(thresholds["incident"]),
            p_warning=float(p_warn),
            p_incident=float(p_inc),
            k_consecutive=int(dcfg.get("k_consecutive", 2)),
            end_after=int(dcfg.get("end_after", 48)),
            exit_ratio=float(dcfg.get("exit_ratio", 0.6)),
            min_history=max(min_history, self.ctx.min_history),
            window_size=int(self.ctx.period),
            criticality=int(criticality),
        )
        return self

    def score(self, series: pd.Series, step_seconds: int) -> pd.DataFrame:
        if self.ctx is None or self.evidence_model is None or self.calibrator is None:
            raise RuntimeError("Detector is not fitted")
        features = build_features(series, step_seconds, self.ctx)
        evidence = self.evidence_model.score(features)
        p = self.calibrator.predict(evidence)
        out = features.copy()
        out["evidence"] = evidence
        out["probability"] = p
        return out

    def score_features(self, features: Dict[str, float]) -> tuple[float, float]:
        if self.evidence_model is None or self.calibrator is None:
            raise RuntimeError("Detector is not fitted")
        frame = pd.DataFrame([features])
        evidence = float(self.evidence_model.score(frame)[0])
        p = float(self.calibrator.predict([evidence])[0])
        return evidence, p

    def save(self, path: str) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: str) -> "AnomalyDetector":
        with open(path, "rb") as fh:
            return pickle.load(fh)
