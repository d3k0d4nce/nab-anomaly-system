from types import SimpleNamespace

import pandas as pd
import pytest

from nab_system.actions import SKIP
from nab_system.features import FEATURE_NAMES
from nab_system.service import ReplayService


class FakeOnline:
    def __init__(self, n: int):
        self.values = [1.0] * n

    def update(self, ts, value):
        self.values.append(value)
        return {name: 0.0 for name in FEATURE_NAMES}


class FakeDetector:
    def __init__(self, min_history: int = 10):
        self.params = SimpleNamespace(min_history=min_history)
        self.called = False

    def score_features(self, features):
        self.called = True
        return 99.0, 0.99


def runtime_with_history(n: int):
    detector = FakeDetector(min_history=10)
    return SimpleNamespace(online=FakeOnline(n), detector=detector), detector


def service_without_db():
    return ReplayService.__new__(ReplayService)


def test_insufficient_history_skips_without_scoring():
    service = service_without_db()
    runtime, detector = runtime_with_history(5)
    result = service.process_point(
        "series",
        pd.Timestamp("2020-01-01 00:00:00"),
        1.0,
        persist=False,
        runtime=runtime,
    )
    assert result["action"] == SKIP
    assert "недостаточно наблюдений" in result["reason"]
    assert result["data_quality"]["enough_history"] is False
    assert result["data_quality"]["history_count"] == 5
    assert detector.called is False


def test_invalid_value_is_rejected_before_runtime():
    service = service_without_db()
    with pytest.raises(ValueError, match="конечным"):
        service.process_point(
            "series", pd.Timestamp("2020-01-01"), float("nan"), persist=False
        )


def test_invalid_timestamp_is_rejected():
    service = service_without_db()
    with pytest.raises(ValueError):
        service.process_point("series", "not-a-date", 1.0, persist=False)
