from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

FEATURE_NAMES = ["z_short", "z_long", "z_diff", "z_seas", "z_level"]


@dataclass
class FeatureContext:
    period: int
    short: int
    g: float
    gd: float
    seasonal_days: int = 7

    @property
    def min_history(self) -> int:
        return max(2 * self.period, 2 * self.short, self.seasonal_days * self.period)


def mad(a: Sequence[float]) -> float:
    arr = np.asarray(a, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(np.median(np.abs(arr - np.median(arr))) * 1.4826)


def _rolling_robust_z(x: pd.Series, window: int, floor: float) -> pd.Series:
    med = x.rolling(window, min_periods=max(5, window // 4)).median()
    m = (x - med).abs().rolling(window, min_periods=max(5, window // 4)).median()
    scale = np.maximum(1.4826 * m, floor)
    return (x - med.shift(1)) / scale.shift(1)


def _seasonal_component(
    x: pd.Series,
    period: int,
    floor: float,
    days: int,
) -> tuple[pd.Series, pd.Series, np.ndarray]:
    n = len(x)
    if n < period * 2:
        return (
            pd.Series(np.zeros(n), index=x.index),
            pd.Series(np.full(n, np.nan), index=x.index),
            np.full(n, floor),
        )
    nd = int(np.ceil(n / period))
    arr = np.concatenate([x.values.astype(float), np.full(nd * period - n, np.nan)])
    mat = arr.reshape(nd, period)
    base = np.full_like(mat, np.nan)
    m = np.full_like(mat, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for d in range(1, nd):
            block = mat[max(0, d - days):d]
            base[d] = np.nanmedian(block, axis=0)
            m[d] = np.nanmedian(np.abs(block - base[d]), axis=0)
    base[0] = base[1] if nd > 1 else 0.0
    m[0] = m[1] if nd > 1 else 0.0
    fb = np.nanmedian(base) if np.isfinite(np.nanmedian(base)) else 0.0
    fm = np.nanmedian(m) if np.isfinite(np.nanmedian(m)) else 0.0
    base = median_filter(np.nan_to_num(base, nan=fb), size=(1, 5), mode="nearest")
    m = median_filter(np.nan_to_num(m, nan=fm), size=(1, 5), mode="nearest")
    b = base.reshape(-1)[:n]
    scale = np.maximum(1.4826 * m.reshape(-1)[:n], floor)
    z = (x.values - b) / scale
    return pd.Series(z, index=x.index), pd.Series(b, index=x.index), scale


def fit_context(
    series: pd.Series,
    step_seconds: int,
    train_mask: np.ndarray,
    seasonal_days: int = 7,
) -> FeatureContext:
    period = max(1, int(round(86400 / step_seconds)))
    short = max(12, period // 24)
    train_values = series.values[: len(train_mask)][train_mask]
    g = max(mad(train_values), 1e-9)
    gd = max(mad(np.diff(train_values)), 1e-9)
    return FeatureContext(period=period, short=short, g=g, gd=gd, seasonal_days=seasonal_days)


def build_features(series: pd.Series, step_seconds: int, ctx: FeatureContext) -> pd.DataFrame:
    z_short = _rolling_robust_z(series, ctx.short, 0.5 * ctx.g)
    z_long = _rolling_robust_z(series, ctx.period, 0.75 * ctx.g)
    z_diff = _rolling_robust_z(series.diff(), ctx.short, 0.5 * ctx.gd)
    smoothed = series.rolling(5, min_periods=1).median()
    z_seas, base, scale = _seasonal_component(
        smoothed, ctx.period, 0.3 * ctx.g, ctx.seasonal_days
    )
    lvl = series.rolling(ctx.short, min_periods=3).median()
    lvl_std = series.rolling(ctx.period, min_periods=max(5, ctx.period // 4)).std()
    z_level = (lvl - base) / np.maximum(1.4826 * lvl_std, 0.3 * ctx.g)
    frame = pd.DataFrame(
        {
            "z_short": z_short,
            "z_long": z_long,
            "z_diff": z_diff,
            "z_seas": z_seas,
            "z_level": z_level,
        },
        index=series.index,
    )
    return frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)


class OnlineFeatureState:
    """Causal incremental feature extractor used by the streaming service."""

    def __init__(self, ctx: FeatureContext):
        self.ctx = ctx
        self.values: List[float] = []
        self.timestamps: List[pd.Timestamp] = []

    def update(self, ts: pd.Timestamp, value: float) -> Dict[str, float]:
        self.timestamps.append(pd.Timestamp(ts))
        self.values.append(float(value))
        i = len(self.values) - 1
        x = float(value)
        ctx = self.ctx
        zeros = {name: 0.0 for name in FEATURE_NAMES}
        if i < max(ctx.period, ctx.short, 2):
            return zeros

        prev_short = np.asarray(self.values[i - ctx.short:i], dtype=float)
        med_short = float(np.median(prev_short))
        mad_short = float(np.median(np.abs(prev_short - med_short)) * 1.4826)
        z_short = (x - med_short) / max(mad_short, 0.5 * ctx.g)

        prev_long = np.asarray(self.values[i - ctx.period:i], dtype=float)
        med_long = float(np.median(prev_long))
        mad_long = float(np.median(np.abs(prev_long - med_long)) * 1.4826)
        z_long = (x - med_long) / max(mad_long, 0.75 * ctx.g)

        prev_point = self.values[i - 1]
        diffs = np.diff(np.asarray(self.values[i - ctx.short - 1:i], dtype=float))
        med_dx = float(np.median(diffs)) if len(diffs) else 0.0
        mad_dx = float(np.median(np.abs(diffs - med_dx)) * 1.4826) if len(diffs) else 0.0
        z_diff = ((x - prev_point) - med_dx) / max(mad_dx, 0.5 * ctx.gd)

        # Seasonal baseline from the same and neighbouring slots in prior days.
        slot = i % ctx.period
        samples: List[float] = []
        for d in range(1, ctx.seasonal_days + 1):
            for delta in (-2, -1, 0, 1, 2):
                j = i - d * ctx.period + delta
                if 0 <= j < i:
                    s = j % ctx.period
                    if min(abs(s - slot), ctx.period - abs(s - slot)) <= 2:
                        samples.append(self.values[j])
        if samples:
            base = float(np.median(samples))
            seas_mad = float(np.median(np.abs(np.asarray(samples) - base)) * 1.4826)
            z_seas = (x - base) / max(seas_mad, 0.3 * ctx.g)
        else:
            base = float(np.median(prev_long))
            z_seas = 0.0

        level = float(np.median(np.asarray(self.values[i - ctx.short + 1:i + 1], dtype=float)))
        level_std = float(np.std(prev_long))
        z_level = (level - base) / max(1.4826 * level_std, 0.3 * ctx.g)

        return {
            "z_short": float(np.clip(z_short, -1e6, 1e6)),
            "z_long": float(np.clip(z_long, -1e6, 1e6)),
            "z_diff": float(np.clip(z_diff, -1e6, 1e6)),
            "z_seas": float(np.clip(z_seas, -1e6, 1e6)),
            "z_level": float(np.clip(z_level, -1e6, 1e6)),
        }
