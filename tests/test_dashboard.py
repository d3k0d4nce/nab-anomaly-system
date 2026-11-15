from nab_system.dashboard import (
    _alert_rows,
    _kpis,
    alert_filters_html,
    alert_range_text,
    dashboard_html,
    pagination_html,
    quality_pagination_html,
)


def alert(**overrides):
    row = {
        "alert_id": 7,
        "alert_type": "WARNING",
        "status": "OPEN",
        "deviation_kind": "level_shift",
        "evidence": 6.5,
        "response_limit_min": 5,
        "ts": "2020-01-01 00:00:00",
        "name": "series",
        "check_id": None,
        "is_real_fault": None,
        "comment": None,
    }
    row.update(overrides)
    return row


def test_operator_gets_alert_actions():
    html = _alert_rows([alert()], can_operate=True)
    assert "alertAck(7)" in html
    assert "alertCheck(7, true)" in html
    assert "alertCheck(7, false)" in html


def test_read_only_user_has_no_alert_actions():
    html = _alert_rows([alert()], can_operate=False)
    assert "alertAck" not in html
    assert "alertCheck" not in html
    assert "только" not in html


def test_confirmed_alert_can_be_closed():
    html = _alert_rows(
        [
            alert(
                status="ACKNOWLEDGED",
                check_id=1,
                is_real_fault=True,
                comment="диск перегрет",
            )
        ],
        can_operate=True,
    )
    assert "confirmed" in html
    assert "alertClose(7)" in html
    assert "alertCheck" not in html


def test_false_alarm_is_closed_and_has_no_more_actions():
    html = _alert_rows(
        [alert(status="CLOSED", check_id=2, is_real_fault=False)],
        can_operate=True,
    )
    assert "false alarm" in html
    assert "alertCheck" not in html
    assert "alertClose" not in html


def test_kpis_include_operational_stats():
    html = _kpis(
        {
            "alerts": 10,
            "open_alerts": 3,
            "acknowledged_alerts": 2,
            "confirmed": 4,
            "confirmed_warnings": 3,
            "confirmed_incidents": 1,
            "false_alarms": 2,
            "checks": 6,
            "avg_response_min": 12.34,
        }
    )
    assert "12.3 мин" in html
    assert "W: 3 / I: 1" in html
    assert "ложное срабатывание" in html


def test_alert_range_text_uses_page_size():
    assert alert_range_text(1, 37) == "1–10 из 37"
    assert alert_range_text(2, 37) == "11–20 из 37"
    assert alert_range_text(1, 0) == "0 из 0"


def test_pagination_preserves_filters():
    html = pagination_html(
        2,
        37,
        {"status": "ACTIVE", "alert_type": "INCIDENT", "series": "nyc_taxi"},
    )
    assert "Страница 2 из 4" in html
    assert "status=ACTIVE" in html
    assert "alert_type=INCIDENT" in html
    assert "series=nyc_taxi" in html
    assert "page=3" in html


def test_alert_pagination_preserves_quality_page():
    html = pagination_html(
        1,
        37,
        {"status": "all", "alert_type": "all", "series": "all"},
        quality_page=2,
    )
    assert "page=2" in html
    assert "qpage=2" in html


def test_quality_pagination_preserves_alert_page_and_filters():
    html = quality_pagination_html(
        2,
        25,
        {"status": "ACTIVE", "alert_type": "WARNING", "series": "nyc_taxi"},
        alert_page=3,
    )
    assert "Страница 2 из 3" in html
    assert "qpage=3" in html
    assert "page=3" in html
    assert "status=ACTIVE" in html
    assert "alert_type=WARNING" in html
    assert "series=nyc_taxi" in html


def test_alert_filters_mark_current_values():
    html = alert_filters_html(
        {"status": "ACTIVE", "alert_type": "INCIDENT", "series": "nyc_taxi"},
        ["nyc_taxi", "other"],
    )
    assert '<option value="ACTIVE" selected>' in html
    assert '<option value="INCIDENT" selected>' in html
    assert '<option value="nyc_taxi" selected>' in html


def test_manual_point_and_footer_visibility():
    kwargs = {
        "user": {"full_name": "Пользователь", "login": "user", "role": "operator"},
        "series_names": ["series"],
        "selected": "series",
        "alerts": [],
        "stats": [],
        "counts": {},
    }
    operator_html = dashboard_html(can_observe=False, can_operate=True, **kwargs)
    assert "Ручная точка" not in operator_html
    assert "Данные: NAB / PostgreSQL" not in operator_html

    admin_html = dashboard_html(can_observe=True, can_operate=True, **kwargs)
    assert "Ручная точка" in admin_html
