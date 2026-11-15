import pandas as pd

from nab_system.actions import INCIDENT, SKIP, WARNING, ActionEngine
from nab_system.detector import DetectorParams


def make_params():
    return DetectorParams(
        evidence_warning=5.0,
        evidence_incident=7.0,
        p_warning=0.5,
        p_incident=0.8,
        k_consecutive=2,
        end_after=3,
        exit_ratio=0.6,
        min_history=10,
        window_size=288,
        criticality=3,
    )


def features(value):
    return {
        "z_short": value,
        "z_long": 0.0,
        "z_diff": 0.0,
        "z_seas": 0.0,
        "z_level": 0.0,
    }


def test_warning_incident_and_episode_close():
    engine = ActionEngine(make_params())
    ts = pd.date_range("2020-01-01", periods=9, freq="5min")
    evidence = [1.0, 0.0, 6.0, 6.0, 8.0, 0.0, 0.0, 0.0, 0.0]
    actions = [engine.process(t, e, 0.0, features(0.0)).action for t, e in zip(ts, evidence)]
    assert actions[:5] == [SKIP, SKIP, SKIP, WARNING, INCIDENT]
    assert actions[5] == SKIP
    engine.finish(ts[-1])
    assert len(engine.episodes) == 1
    episode = next(iter(engine.episodes.values()))
    assert episode.warning_sent and episode.incident_sent
    assert episode.final_action == INCIDENT
    assert episode.ended_at == ts[7]


def test_immediate_incident_skips_warning_confirmation():
    engine = ActionEngine(make_params())
    ts = pd.date_range("2020-01-01", periods=3, freq="5min")
    evidence = [8.0, 8.0, 0.0]
    actions = [engine.process(t, e, 0.0, features(0.0)).action for t, e in zip(ts, evidence)]
    assert actions[0] == INCIDENT
    assert actions[1] == SKIP
    assert WARNING not in actions
    assert engine.active is not None
    assert engine.active.warning_sent is False
    assert engine.active.incident_sent is True
    assert engine.active.final_action == INCIDENT


def test_no_duplicate_warning_inside_episode():
    engine = ActionEngine(make_params())
    ts = pd.date_range("2020-01-01", periods=6, freq="5min")
    evidence = [6.0, 6.0, 6.0, 6.0, 6.0, 6.0]
    actions = [engine.process(t, e, 0.0, features(0.0)).action for t, e in zip(ts, evidence)]
    assert actions.count(WARNING) == 1
    assert actions.count(INCIDENT) == 0
    assert actions[0] == SKIP
    assert actions[1] == WARNING
