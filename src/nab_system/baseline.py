from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import norm

from .features import FEATURE_NAMES, FeatureContext, fit_context


class RollingZBaseline:
    """Stage-2 baseline: plain robust rolling z-score without a tail model."""

    name = "rolling_z"

    def __init__(self, sigma: float = 3.0):
        self.sigma = float(sigma)
        self.ctx: FeatureContext | None = None

    def fit(self, series: pd.Series, step_seconds: int, train_mask: np.ndarray) -> "RollingZBaseline":
        self.ctx = fit_context(series, step_seconds, train_mask)
        return self

    def score(self, series: pd.Series) -> pd.DataFrame:
        if self.ctx is None:
            raise RuntimeError("Baseline is not fitted")
        ctx = self.ctx
        med = series.rolling(ctx.period, min_periods=max(5, ctx.period // 4)).median()
        m = (series - med).abs().rolling(ctx.period, min_periods=max(5, ctx.period // 4)).median()
        scale = np.maximum(1.4826 * m, 0.75 * ctx.g)
        z = ((series - med.shift(1)) / scale.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        p = 2.0 * (1.0 - norm.cdf(np.abs(z.values)))
        evidence = -np.log10(np.clip(p, 1e-12, 1.0))
        out = pd.DataFrame(index=series.index)
        for name in FEATURE_NAMES:
            out[name] = 0.0
        out["z_short"] = z.values
        out["z_long"] = z.values
        out["evidence"] = evidence
        out["probability"] = np.clip(evidence / (evidence + 5.0), 0.0, 1.0)
        return out


class FixedThresholdBaseline:
    """Stage-2 baseline: one global threshold learned on the training split."""

    name = "fixed_threshold"

    def __init__(self, quantile: float = 0.999):
        self.quantile = float(quantile)
        self.median = 0.0
        self.high = 1.0
        self.low = -1.0

    def fit(self, series: pd.Series, train_mask: np.ndarray) -> "FixedThresholdBaseline":
        values = series.values[train_mask]
        values = values[np.isfinite(values)]
        self.median = float(np.median(values))
        self.high = float(np.quantile(values, self.quantile))
        self.low = float(np.quantile(values, 1.0 - self.quantile))
        return self

    def score(self, series: pd.Series) -> pd.DataFrame:
        upper = max(self.high - self.median, 1e-9)
        lower = max(self.median - self.low, 1e-9)
        deviation = np.where(
            series.values >= self.median,
            (series.values - self.median) / upper,
            (self.median - series.values) / lower,
        )
        evidence = np.clip(np.abs(deviation), 0.0, None) * 5.0
        out = pd.DataFrame(index=series.index)
        for name in FEATURE_NAMES:
            out[name] = 0.0
        out["z_short"] = deviation
        out["evidence"] = evidence
        out["probability"] = np.clip(evidence / (evidence + 5.0), 0.0, 1.0)
        return out
