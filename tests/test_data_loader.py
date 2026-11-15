import numpy as np
import pandas as pd

from nab_system.data_loader import regularize, split_ranges
from nab_system.repository import normalize_ts


def test_regularize_marks_short_gaps():
    ts = pd.date_range("2020-01-01", periods=6, freq="5min")
    df = pd.DataFrame({"timestamp": ts, "value": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0]})
    out = regularize(df, 300, max_interp=6)
    assert len(out) == 6
    assert out["is_imputed"].sum() == 1
    assert out["value"].notna().all()


def test_split_ranges_cover_series():
    idx = pd.date_range("2020-01-01", periods=100, freq="5min")
    ranges = split_ranges(idx, {"train": 0.3, "validation": 0.3, "test": 0.4})
    assert ranges["train"] == (0, 30)
    assert ranges["validation"] == (30, 60)
    assert ranges["test"] == (60, 100)


def test_normalize_ts_preserves_wall_clock():
    ts = normalize_ts(pd.Timestamp("2014-03-18 17:06+04:00"))
    assert ts.tz is None
    assert ts.hour == 17
