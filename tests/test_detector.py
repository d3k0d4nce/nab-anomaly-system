import numpy as np
import pandas as pd

from nab_system.config import load_config
from nab_system.detector import AnomalyDetector


def synthetic_with_spike(n=3000, step=300):
    idx = pd.date_range("2020-01-01", periods=n, freq=f"{step}s")
    t = np.arange(n)
    rng = np.random.default_rng(11)
    values = 50 + 5 * np.sin(2 * np.pi * t / 288) + rng.normal(0, 0.3, n)
    spike = slice(2200, 2208)
    values[spike] += 30
    return pd.Series(values, index=idx)


def test_detector_flags_injected_spike(tmp_path):
    cfg = load_config()
    series = synthetic_with_spike()
    labels = np.zeros(len(series), dtype=int)
    labels[2200:2208] = 1
    ranges = {"train": (0, 1200), "validation": (1200, 2000), "test": (2000, 3000)}
    det = AnomalyDetector(cfg).fit(series, 300, labels, ranges, criticality=3)
    scored = det.score(series, 300)

    assert scored["evidence"].iloc[2200:2208].max() >= 5.0
    assert scored["probability"].iloc[2200:2208].max() >= det.params.p_warning
    assert scored["evidence"].iloc[2100:2190].max() < 5.0

    path = tmp_path / "det.pkl"
    det.save(str(path))
    loaded = AnomalyDetector.load(str(path))
    assert loaded.params.evidence_warning == det.params.evidence_warning
