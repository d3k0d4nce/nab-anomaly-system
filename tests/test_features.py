import numpy as np
import pandas as pd

from nab_system.features import (
    FEATURE_NAMES,
    OnlineFeatureState,
    build_features,
    fit_context,
)


def synthetic_series(n=2016, step=300):
    idx = pd.date_range("2020-01-01", periods=n, freq=f"{step}s")
    t = np.arange(n)
    daily = 0.5 * np.sin(2 * np.pi * t / 288)
    rng = np.random.default_rng(7)
    values = 10 + daily + rng.normal(0, 0.1, n)
    return pd.Series(values, index=idx)


def test_build_features_columns_and_finiteness():
    series = synthetic_series()
    mask = np.zeros(len(series), dtype=bool)
    mask[:1000] = True
    ctx = fit_context(series, 300, mask)
    frame = build_features(series, 300, ctx)
    assert list(frame.columns) == FEATURE_NAMES
    assert np.isfinite(frame.values).all()


def test_online_state_matches_shape_and_warmup():
    series = synthetic_series()
    mask = np.zeros(len(series), dtype=bool)
    mask[:1000] = True
    ctx = fit_context(series, 300, mask)
    state = OnlineFeatureState(ctx)
    first = state.update(series.index[0], float(series.iloc[0]))
    assert all(v == 0.0 for v in first.values())
    for i in range(1, 700):
        last = state.update(series.index[i], float(series.iloc[i]))
    assert set(last) == set(FEATURE_NAMES)
    assert all(np.isfinite(list(last.values())))
