from nab_system.config import all_series_entries, load_config, risk_thresholds


def test_load_config_paths_are_absolute():
    cfg = load_config()
    assert cfg["paths"]["nab_raw"].startswith("/")
    assert cfg["split"]["train"] == 0.30
    assert cfg["split"]["validation"] == 0.30
    assert cfg["split"]["test"] == 0.40


def test_risk_thresholds_follow_criticality():
    cfg = load_config()
    assert risk_thresholds(cfg, 1) == {"warning": 7.0, "incident": 9.0}
    assert risk_thresholds(cfg, 4) == {"warning": 4.5, "incident": 6.5}


def test_series_entries():
    cfg = load_config()
    names = {e["name"] for e in all_series_entries(cfg)}
    assert "ec2_request_latency_system_failure" in names
    assert "nyc_taxi" in names
