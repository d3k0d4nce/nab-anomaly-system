from __future__ import annotations

import html
import io
import threading
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urlencode

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .config import load_config, risk_thresholds
from .db import connect
from .repository import fetch_decision_window, fetch_test_windows, normalize_ts

_CHART_LOCK = threading.Lock()
ALERT_PAGE_SIZE = 10
QUALITY_PAGE_SIZE = 10

CSS = """
:root {
  --bg: #0b1020;
  --panel: #131a2e;
  --panel2: #1a2340;
  --text: #e8ecf8;
  --muted: #93a0bf;
  --accent: #5b8cff;
  --green: #35d07f;
  --orange: #ffb020;
  --red: #ff5c6c;
  --border: #273252;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: radial-gradient(1200px 800px at 20% -10%, #1b2a55 0%, var(--bg) 55%);
  color: var(--text); font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
}
a { color: var(--accent); text-decoration: none; }
.topbar {
  display: flex; align-items: center; justify-content: space-between; gap: 16px;
  padding: 18px 28px; border-bottom: 1px solid var(--border);
  background: rgba(10, 15, 30, 0.75); backdrop-filter: blur(8px); position: sticky; top: 0; z-index: 5;
}
.brand { font-weight: 700; letter-spacing: .3px; font-size: 18px; }
.brand span { color: var(--accent); }
.user { display: flex; align-items: center; gap: 12px; color: var(--muted); font-size: 14px; }
.badge { padding: 4px 9px; border-radius: 999px; background: #22305a; color: #c8d6ff; font-size: 12px; }
.wrap { max-width: 1400px; margin: 0 auto; padding: 22px 28px 60px; }
.grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }
.card {
  background: linear-gradient(180deg, var(--panel2), var(--panel));
  border: 1px solid var(--border); border-radius: 16px; padding: 16px 18px;
  box-shadow: 0 8px 30px rgba(0,0,0,.22);
}
.kpi .label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }
.kpi .value { font-size: 28px; font-weight: 700; margin-top: 6px; }
.kpi .hint { color: var(--muted); font-size: 12px; margin-top: 4px; }
.section { margin-top: 18px; }
.section h2 { font-size: 16px; margin: 0 0 10px; color: #dfe6ff; }
.toolbar { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-bottom: 12px; }
select, input, button {
  background: #0f162b; color: var(--text); border: 1px solid var(--border);
  border-radius: 10px; padding: 9px 11px; font: inherit;
}
button { cursor: pointer; background: linear-gradient(180deg, #3f6fff, #2f55d6); border: 0; font-weight: 600; }
button:hover { filter: brightness(1.08); }
button.ghost { background: #1a2340; border: 1px solid var(--border); }
.actions { display: flex; gap: 4px; flex-wrap: wrap; }
.actions button { padding: 4px 7px; font-size: 11px; border-radius: 8px; }
.chart { width: 100%; border-radius: 12px; display: block; background: #0f162b; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { text-align: left; padding: 9px 8px; border-bottom: 1px solid #202b4a; vertical-align: top; }
th { color: var(--muted); font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: .06em; }
tr:hover td { background: rgba(91,140,255,.06); }
.pill { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 700; }
.warn { background: rgba(255,176,32,.16); color: var(--orange); }
.inc { background: rgba(255,92,108,.16); color: var(--red); }
.ok { background: rgba(53,208,127,.14); color: var(--green); }
.muted { color: var(--muted); }
.observe { display: grid; grid-template-columns: 2fr 1.2fr auto; gap: 10px; align-items: end; }
.observe label { display: block; color: var(--muted); font-size: 12px; margin-bottom: 5px; }
.filters {
  display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-end;
  margin: 0 0 14px; padding: 12px 14px;
  background: rgba(15, 22, 43, .55); border: 1px solid var(--border);
  border-radius: 14px;
}
.filters .filter-field { display: flex; flex-direction: column; gap: 5px; min-width: 160px; }
.filters .filter-field label {
  color: var(--muted); font-size: 10px; font-weight: 600; letter-spacing: .09em;
  text-transform: uppercase; margin: 0;
}
.filters select {
  min-width: 170px; padding: 9px 32px 9px 11px; background-color: #0f162b;
  border: 1px solid var(--border); transition: border-color .15s, box-shadow .15s;
}
.filters select:hover { border-color: #3f5a9e; }
.filters select:focus {
  outline: none; border-color: var(--accent);
  box-shadow: 0 0 0 3px rgba(91, 140, 255, .16);
}
.filter-actions { display: flex; gap: 8px; align-items: center; margin-left: auto; }
.filter-apply { padding: 10px 16px; box-shadow: 0 6px 18px rgba(47, 85, 214, .25); }
.filter-reset {
  display: inline-flex; align-items: center; justify-content: center;
  padding: 9px 14px; border: 1px solid var(--border); border-radius: 10px;
  color: var(--muted); background: transparent; font-size: 13px; font-weight: 600;
  transition: color .15s, border-color .15s, background .15s;
}
.filter-reset:hover { color: var(--text); border-color: #3f5a9e; background: #1a2340; }
.pagination { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin-top: 12px; }
.pagination a, .pagination span {
  display: inline-block; padding: 6px 10px; border: 1px solid var(--border);
  border-radius: 8px; color: var(--text); background: #1a2340; font-size: 12px;
}
.pagination .active { background: #2f55d6; border-color: #2f55d6; color: #fff; }
.pagination .disabled { color: var(--muted); background: #141b30; }
pre { background: #0f162b; border: 1px solid var(--border); border-radius: 10px; padding: 10px; overflow: auto; white-space: pre-wrap; }
.login-wrap { min-height: 100vh; display: grid; place-items: center; padding: 24px; }
.login { width: min(420px, 100%); }
.login h1 { margin: 0 0 4px; font-size: 24px; }
.login p { color: var(--muted); margin-top: 0; }
.login label { display: block; color: var(--muted); font-size: 13px; margin: 14px 0 6px; }
.login input { width: 100%; }
.login button { width: 100%; margin-top: 18px; padding: 11px; }
.error { color: var(--red); background: rgba(255,92,108,.1); border: 1px solid rgba(255,92,108,.3); padding: 9px 11px; border-radius: 10px; margin-top: 12px; }
.footer { color: var(--muted); font-size: 12px; margin-top: 28px; }
@media (max-width: 900px) {
  .grid { grid-template-columns: repeat(2, minmax(0,1fr)); }
  .observe { grid-template-columns: 1fr; }
  .filters .filter-field, .filters select { width: 100%; }
  .filter-actions { width: 100%; margin-left: 0; }
  .filter-actions button, .filter-actions a { flex: 1; }
}
"""

DASHBOARD = Template(
    """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NAB Anomaly Monitor</title>
<style>$css</style>
</head>
<body>
<div class="topbar">
  <div class="brand">NAB <span>Anomaly Monitor</span></div>
  <div class="user">
    <span>$user_name</span>
    <span class="badge">$role</span>
    <a class="ghost" href="/logout">Выйти</a>
  </div>
</div>
<div class="wrap">
  <div class="grid" id="kpis">
    $kpis
  </div>

  <div class="section card">
    <div class="toolbar">
      <div>
        <div class="muted" style="font-size:12px;margin-bottom:5px;">Ряд</div>
        <select id="series">$series_options</select>
      </div>
      <div style="flex:1"></div>
      <span id="chart-updated" class="muted" style="font-size:12px;"></span>
      <button id="refresh-chart" class="ghost" type="button">Обновить график</button>
      <a class="ghost" href="/docs" style="padding:9px 11px;border:1px solid var(--border);border-radius:10px;">API docs</a>
    </div>
    <img id="chart" class="chart" src="$chart_src" alt="series chart">
  </div>

  $observe_block

  <div class="section card">
    <h2>Последние алерты <span id="alert-count" class="muted" style="font-size:12px;font-weight:400;">$alert_count</span></h2>
    $alert_filters
    <table>
      <thead><tr><th>Время</th><th>Ряд</th><th>Тип</th><th>Статус</th><th>Характер</th><th>Evidence</th><th>Лимит</th><th>Проверка</th><th>Действия</th></tr></thead>
      <tbody id="alert-rows">$alert_rows</tbody>
    </table>
    <div id="pagination">$pagination</div>
  </div>

  <div class="section card">
    <h2>Качество детекции <span id="stat-count" class="muted" style="font-size:12px;font-weight:400;">$stat_count</span></h2>
    <table>
      <thead><tr><th>Метод</th><th>Ряд</th><th>TP</th><th>FP</th><th>FN</th><th>Recall</th><th>Precision</th><th>FP/24h</th><th>Cost</th></tr></thead>
      <tbody id="stat-rows">$stat_rows</tbody>
    </table>
    <div id="stat-pagination">$stat_pagination</div>
  </div>

</div>
<script>
function refreshChart() {
  var sel = document.getElementById('series');
  var img = document.getElementById('chart');
  var status = document.getElementById('chart-updated');
  if (!sel || !img) return;
  if (status) status.textContent = 'обновление...';
  img.onload = function () {
    if (status) status.textContent = 'обновлено ' + new Date().toLocaleTimeString();
  };
  img.onerror = function () {
    if (status) status.textContent = 'ошибка загрузки графика';
  };
  img.src = '/chart/' + encodeURIComponent(sel.value) + '.png?t=' + Date.now();
}
function applyDashboard(data) {
  var kpis = document.getElementById('kpis');
  var alerts = document.getElementById('alert-rows');
  var stats = document.getElementById('stat-rows');
  var alertCount = document.getElementById('alert-count');
  var statCount = document.getElementById('stat-count');
  var pagination = document.getElementById('pagination');
  var statPagination = document.getElementById('stat-pagination');
  if (kpis && data.kpis) kpis.innerHTML = data.kpis;
  if (alerts && data.alerts) alerts.innerHTML = data.alerts;
  if (stats && data.stats) stats.innerHTML = data.stats;
  if (alertCount && data.alert_count) alertCount.textContent = data.alert_count;
  if (statCount && data.stat_count) statCount.textContent = data.stat_count;
  if (pagination && data.pagination !== undefined) pagination.innerHTML = data.pagination;
  if (statPagination && data.stat_pagination !== undefined) {
    statPagination.innerHTML = data.stat_pagination;
  }
}
async function refreshDashboard(showErrors) {
  try {
    var response = await fetch('/dashboard/partials' + window.location.search, {
      headers: {'Accept': 'application/json'},
      cache: 'no-store'
    });
    if (response.status === 401) {
      window.location.href = '/login';
      return;
    }
    if (!response.ok) throw new Error('HTTP ' + response.status);
    applyDashboard(await response.json());
  } catch (err) {
    if (showErrors) {
      var result = document.getElementById('observe-result');
      if (result) result.textContent = 'Ошибка обновления: ' + err;
    }
  }
}
function alertsUrlFromForm(form) {
  var params = new URLSearchParams();
  ['status', 'alert_type', 'series'].forEach(function (name) {
    var field = form.elements[name];
    if (field && field.value && field.value !== 'all') params.set(name, field.value);
  });
  var query = params.toString();
  return query ? '/?' + query : '/';
}
function partialsUrl(href) {
  var query = href.indexOf('?');
  return '/dashboard/partials' + (query >= 0 ? href.slice(query) : '');
}
function syncFiltersFromUrl() {
  var form = document.getElementById('alert-filters');
  if (!form) return;
  var params = new URLSearchParams(window.location.search);
  ['status', 'alert_type', 'series'].forEach(function (name) {
    var field = form.elements[name];
    if (field) field.value = params.get(name) || 'all';
  });
}
function syncChartFromUrl() {
  var series = new URLSearchParams(window.location.search).get('series');
  var chartSeries = document.getElementById('series');
  if (!series || series === 'all' || !chartSeries) return;
  if (chartSeries.value !== series) {
    chartSeries.value = series;
    refreshChart();
  }
}
async function navigateAlerts(href, push) {
  try {
    var response = await fetch(partialsUrl(href), {
      headers: {'Accept': 'application/json'},
      cache: 'no-store'
    });
    if (!response.ok) throw new Error('HTTP ' + response.status);
    applyDashboard(await response.json());
    if (push) history.pushState({}, '', href);
    syncFiltersFromUrl();
    syncChartFromUrl();
  } catch (err) {
    alert('Ошибка обновления: ' + err);
  }
}
function startLiveRefresh() {
  setInterval(function () {
    if (!document.hidden) refreshDashboard(false);
  }, 10000);
  setInterval(function () {
    if (!document.hidden) refreshChart();
  }, 30000);
}
document.addEventListener('DOMContentLoaded', function () {
  var sel = document.getElementById('series');
  var button = document.getElementById('refresh-chart');
  var filters = document.getElementById('alert-filters');
  if (sel) sel.addEventListener('change', refreshChart);
  if (button) button.addEventListener('click', refreshChart);
  if (filters) {
    filters.addEventListener('submit', function (event) {
      event.preventDefault();
      navigateAlerts(alertsUrlFromForm(filters), true);
    });
  }
  document.addEventListener('click', function (event) {
    var link = event.target && event.target.closest
      ? event.target.closest('#pagination a, #stat-pagination a')
      : null;
    if (!link) return;
    event.preventDefault();
    navigateAlerts(link.getAttribute('href'), true);
  });
  window.addEventListener('popstate', function () {
    refreshDashboard(false);
    syncFiltersFromUrl();
    syncChartFromUrl();
  });
  startLiveRefresh();
});
async function sendPoint(event) {
  event.preventDefault();
  var result = document.getElementById('observe-result');
  var payload = {
    series: document.getElementById('observe-series').value,
    timestamp: document.getElementById('observe-ts').value,
    value: parseFloat(document.getElementById('observe-value').value)
  };
  result.textContent = 'Отправка...';
  try {
    var response = await fetch('/observe', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    var raw = await response.text();
    var data = null;
    try { data = JSON.parse(raw); } catch (parseErr) { data = null; }
    if (!response.ok) {
      var detail = data && data.detail ? data.detail : raw;
      result.textContent = 'Ошибка ' + response.status + ': ' + detail;
      result.classList.remove('muted');
      result.scrollIntoView({behavior: 'smooth', block: 'nearest'});
      return;
    }
    var summary = [
      'Действие: ' + data.action,
      'Причина: ' + data.reason,
      'Evidence: ' + Number(data.evidence).toFixed(3) +
        ', вероятность: ' + Number(data.probability).toFixed(3),
      data.alert_id
        ? 'Создан алерт #' + data.alert_id + ' — он появился в таблице ниже.'
        : (data.reason && data.reason.indexOf('активного эпизода') >= 0
            ? 'Новый алерт не создан: точка добавлена в активный эпизод ' +
              data.episode_key + '.'
            : 'Новый алерт не создан: ' + data.reason + '.')
    ].join('\\n');
    result.textContent = summary + '\\n\\n' + JSON.stringify(data, null, 2);
    result.classList.remove('muted');
    result.scrollIntoView({behavior: 'smooth', block: 'nearest'});
    var chartSeries = document.getElementById('series');
    if (chartSeries) chartSeries.value = payload.series;
    await refreshDashboard(false);
    refreshChart();
  } catch (err) {
    result.textContent = 'Ошибка: ' + err;
    result.classList.remove('muted');
    result.scrollIntoView({behavior: 'smooth', block: 'nearest'});
  }
}
async function alertRequest(url, payload) {
  try {
    var response = await fetch(url, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload || {})
    });
    if (!response.ok) {
      var text = await response.text();
      alert('Ошибка: ' + text);
      return;
    }
    await refreshDashboard(false);
  } catch (err) {
    alert('Ошибка: ' + err);
  }
}
function alertAck(id) {
  alertRequest('/alerts/' + id + '/ack', {});
}
function alertCheck(id, isReal) {
  var comment = prompt(isReal ? 'Комментарий к подтверждению' : 'Причина ложного срабатывания');
  if (comment === null) return;
  var diagnostics = '';
  if (isReal) {
    diagnostics = prompt('Диагностика и устранение (можно оставить пустым)') || '';
  }
  alertRequest('/alerts/' + id + '/check', {
    is_real_fault: isReal,
    comment: comment,
    diagnostics: diagnostics
  });
}
function alertClose(id) {
  var comment = prompt('Комментарий к закрытию');
  if (comment === null) return;
  alertRequest('/alerts/' + id + '/close', {comment: comment});
}
</script>
</body>
</html>"""
)

LOGIN = Template(
    """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Вход — NAB Anomaly Monitor</title>
<style>$css</style>
</head>
<body>
<div class="login-wrap">
  <div class="card login">
    <h1>NAB Anomaly Monitor</h1>
    <p>Вход в панель мониторинга</p>
    $error
    <form method="post" action="/login">
      <input type="hidden" name="next" value="$next_url">
      <label for="username">Логин</label>
      <input id="username" name="username" autocomplete="username" required autofocus>
      <label for="password">Пароль</label>
      <input id="password" name="password" type="password" autocomplete="current-password" required>
      <button type="submit">Войти</button>
    </form>
    <div class="footer">Демо: operator / demo-operator (дежурный инженер),
      admin / demo-admin (администратор).</div>
  </div>
</div>
</body>
</html>"""
)


def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def _fmt_ts(value: Any) -> str:
    if value is None:
        return ""
    ts = normalize_ts(value)
    return ts.strftime("%Y-%m-%d %H:%M")


def _series_options(names: Sequence[str], selected: str) -> str:
    out = []
    for name in names:
        attr = " selected" if name == selected else ""
        out.append(f'<option value="{_esc(name)}"{attr}>{_esc(name)}</option>')
    return "".join(out)


def _kpis(counts: Dict[str, Any]) -> str:
    avg_response = counts.get("avg_response_min")
    avg_text = f"{float(avg_response):.1f} мин" if avg_response is not None else "—"
    cards = [
        (
            "Активных",
            counts.get("active_alerts", counts.get("open_alerts", 0)),
            f"OPEN: {counts.get('open_alerts', 0)} · ACK: {counts.get('acknowledged_alerts', 0)}",
        ),
        (
            "Подтверждено",
            counts.get("confirmed", 0),
            f"W: {counts.get('confirmed_warnings', 0)} / I: {counts.get('confirmed_incidents', 0)}",
        ),
        ("Ложных", counts.get("false_alarms", 0), "ложное срабатывание"),
        ("Средняя реакция", avg_text, f"проверок: {counts.get('checks', 0)}"),
    ]
    return "".join(
        f'<div class="card kpi"><div class="label">{_esc(label)}</div>'
        f'<div class="value">{_esc(value)}</div><div class="hint">{_esc(hint)}</div></div>'
        for label, value, hint in cards
    )


def _check_cell(a: Dict[str, Any]) -> str:
    if a.get("check_id") is None:
        return '<span class="muted">—</span>'
    if a.get("is_real_fault"):
        verdict = '<span class="pill inc">confirmed</span>'
    else:
        verdict = '<span class="pill ok">false alarm</span>'
    comment = a.get("comment")
    tail = f'<div class="muted">{_esc(comment)}</div>' if comment else ""
    return verdict + tail


def _action_cell(a: Dict[str, Any], can_operate: bool) -> str:
    if not can_operate:
        return '<span class="muted">—</span>'
    alert_id = int(a.get("alert_id") or 0)
    status = str(a.get("status") or "")
    checked = a.get("check_id") is not None
    buttons: List[str] = []
    if status == "OPEN":
        buttons.append(
            f'<button class="ghost" type="button" onclick="alertAck({alert_id})">В работу</button>'
        )
    if not checked and status != "CLOSED":
        buttons.append(
            f'<button class="ghost" type="button" onclick="alertCheck({alert_id}, true)">Реальный сбой</button>'
        )
        buttons.append(
            f'<button class="ghost" type="button" onclick="alertCheck({alert_id}, false)">Ложная тревога</button>'
        )
    if checked and a.get("is_real_fault") and status != "CLOSED":
        buttons.append(
            f'<button class="ghost" type="button" onclick="alertClose({alert_id})">Закрыть</button>'
        )
    return '<div class="actions">' + "".join(buttons) + "</div>" if buttons else '<span class="muted">—</span>'


def _alert_rows(alerts: Sequence[Dict[str, Any]], can_operate: bool) -> str:
    rows = []
    for a in alerts:
        kind = "inc" if a.get("alert_type") == "INCIDENT" else "warn"
        status = str(a.get("status") or "")
        status_cls = "ok" if status == "CLOSED" else "warn"
        rows.append(
            "<tr>"
            f"<td>{_esc(_fmt_ts(a.get('ts')))}</td>"
            f"<td>{_esc(a.get('name'))}</td>"
            f'<td><span class="pill {kind}">{_esc(a.get("alert_type"))}</span></td>'
            f'<td><span class="pill {status_cls}">{_esc(status)}</span></td>'
            f"<td>{_esc(a.get('deviation_kind'))}</td>"
            f"<td>{float(a.get('evidence') or 0):.2f}</td>"
            f"<td>{_esc(a.get('response_limit_min'))} мин</td>"
            f"<td>{_check_cell(a)}</td>"
            f"<td>{_action_cell(a, can_operate)}</td>"
            "</tr>"
        )
    return "".join(rows) or '<tr><td colspan="9" class="muted">Пока нет алертов</td></tr>'


def _stat_rows(stats: Sequence[Dict[str, Any]]) -> str:
    rows = []
    for s in stats:
        rows.append(
            "<tr>"
            f"<td>{_esc(s.get('method'))}</td>"
            f"<td>{_esc(s.get('name') or 'aggregate')}</td>"
            f"<td>{_esc(s.get('tp'))}</td>"
            f"<td>{_esc(s.get('fp'))}</td>"
            f"<td>{_esc(s.get('fn'))}</td>"
            f"<td>{_esc(s.get('recall'))}</td>"
            f"<td>{_esc(s.get('precision'))}</td>"
            f"<td>{_esc(s.get('fp_per_24h'))}</td>"
            f"<td>{_esc(s.get('cost_units'))}</td>"
            "</tr>"
        )
    return "".join(rows) or '<tr><td colspan="9" class="muted">Нет статистики</td></tr>'


def alert_range_text(page: int, total: int, page_size: int = ALERT_PAGE_SIZE) -> str:
    if total <= 0:
        return "0 из 0"
    start = (max(1, page) - 1) * page_size + 1
    end = min(total, start + page_size - 1)
    return f"{start}–{end} из {total}"


def _alert_filter_params(filters: Dict[str, str]) -> List[tuple[str, str]]:
    params = []
    for key in ("status", "alert_type", "series"):
        value = filters.get(key)
        if value and value != "all":
            params.append((key, value))
    return params


def _alerts_url(
    page: int,
    filters: Dict[str, str],
    quality_page: int = 1,
) -> str:
    params = _alert_filter_params(filters)
    if page > 1:
        params.append(("page", str(page)))
    if quality_page > 1:
        params.append(("qpage", str(quality_page)))
    query = urlencode(params)
    return "/?" + query if query else "/"


def _quality_url(
    page: int,
    filters: Dict[str, str],
    alert_page: int = 1,
) -> str:
    params = _alert_filter_params(filters)
    if alert_page > 1:
        params.append(("page", str(alert_page)))
    if page > 1:
        params.append(("qpage", str(page)))
    query = urlencode(params)
    return "/?" + query if query else "/"


def _pagination_block(page: int, total: int, page_size: int, url_for) -> str:
    pages = max(1, (total + page_size - 1) // page_size)
    if pages <= 1:
        return ""
    page = min(max(1, page), pages)

    def item(label: str, target: Optional[int], active: bool = False) -> str:
        if active:
            return f'<span class="active">{_esc(label)}</span>'
        if target is None:
            return f'<span class="disabled">{_esc(label)}</span>'
        return f'<a href="{_esc(url_for(target))}">{_esc(label)}</a>'

    items = [item("←", None if page <= 1 else page - 1)]
    start = max(1, page - 2)
    end = min(pages, start + 4)
    start = max(1, end - 4)
    for number in range(start, end + 1):
        items.append(item(str(number), number, active=number == page))
    items.append(item("→", None if page >= pages else page + 1))
    items.append(f'<span class="muted">Страница {page} из {pages}</span>')
    return '<div class="pagination">' + "".join(items) + "</div>"


def pagination_html(
    page: int,
    total: int,
    filters: Dict[str, str],
    page_size: int = ALERT_PAGE_SIZE,
    quality_page: int = 1,
) -> str:
    return _pagination_block(
        page,
        total,
        page_size,
        lambda target: _alerts_url(target, filters, quality_page),
    )


def quality_pagination_html(
    page: int,
    total: int,
    filters: Dict[str, str],
    alert_page: int = 1,
    page_size: int = QUALITY_PAGE_SIZE,
) -> str:
    return _pagination_block(
        page,
        total,
        page_size,
        lambda target: _quality_url(target, filters, alert_page),
    )


def _select_options(
    options: Sequence[tuple[str, str]],
    current: str,
) -> str:
    return "".join(
        f'<option value="{_esc(value)}"'
        + (" selected" if value == current else "")
        + f">{_esc(label)}</option>"
        for value, label in options
    )


def alert_filters_html(filters: Dict[str, str], series_names: Sequence[str]) -> str:
    status_options = [
        ("all", "Все статусы"),
        ("ACTIVE", "Активные (OPEN + ACK)"),
        ("OPEN", "OPEN"),
        ("ACKNOWLEDGED", "ACKNOWLEDGED"),
        ("CLOSED", "CLOSED"),
    ]
    type_options = [
        ("all", "Все типы"),
        ("WARNING", "WARNING"),
        ("INCIDENT", "INCIDENT"),
    ]
    series_options = [("all", "Все ряды")] + [(name, name) for name in series_names]
    return (
        '<form id="alert-filters" class="filters" method="get" action="/">'
        + '<div class="filter-field"><label for="filter-status">Статус</label>'
        + '<select id="filter-status" name="status">'
        + _select_options(status_options, filters.get("status", "all"))
        + "</select></div>"
        + '<div class="filter-field"><label for="filter-type">Тип</label>'
        + '<select id="filter-type" name="alert_type">'
        + _select_options(type_options, filters.get("alert_type", "all"))
        + "</select></div>"
        + '<div class="filter-field"><label for="filter-series">Ряд</label>'
        + '<select id="filter-series" name="series">'
        + _select_options(series_options, filters.get("series", "all"))
        + "</select></div>"
        + '<div class="filter-actions">'
        + '<button type="submit" class="filter-apply">Применить</button>'
        + '<a class="filter-reset" href="/">Сбросить</a>'
        + "</div>"
        + "</form>"
    )


def kpis_html(counts: Dict[str, Any]) -> str:
    return _kpis(counts)


def alert_rows_html(alerts: Sequence[Dict[str, Any]], can_operate: bool) -> str:
    return _alert_rows(alerts, can_operate)


def stat_rows_html(stats: Sequence[Dict[str, Any]]) -> str:
    return _stat_rows(stats)


def dashboard_html(
    user: Dict[str, Any],
    series_names: Sequence[str],
    selected: str,
    alerts: Sequence[Dict[str, Any]],
    stats: Sequence[Dict[str, Any]],
    counts: Dict[str, Any],
    can_observe: bool,
    can_operate: Optional[bool] = None,
    filters: Optional[Dict[str, str]] = None,
    page: int = 1,
    total_alerts: Optional[int] = None,
    page_size: int = ALERT_PAGE_SIZE,
    total_stats: Optional[int] = None,
    quality_page: int = 1,
    quality_page_size: int = QUALITY_PAGE_SIZE,
) -> str:
    filters = filters or {"status": "all", "alert_type": "all", "series": "all"}
    total_alerts = len(alerts) if total_alerts is None else total_alerts
    total_stats = len(stats) if total_stats is None else total_stats
    can_operate = can_observe if can_operate is None else can_operate
    observe_block = ""
    if can_observe:
        observe_block = (
            '<div class="section card"><h2>Ручная точка</h2>'
            '<form class="observe" onsubmit="sendPoint(event)">'
            '<div><label>Ряд</label><select id="observe-series">'
            + _series_options(series_names, selected)
            + "</select></div>"
            '<div><label>Время (ISO)</label><input id="observe-ts" placeholder="2014-03-18 22:41:00" required></div>'
            '<div><label>Значение</label><input id="observe-value" type="number" step="any" required></div>'
            '<div style="grid-column: 1 / -1;"><button type="submit">Отправить</button></div>'
            "</form>"
            '<pre id="observe-result" class="muted">Результат появится здесь.</pre></div>'
        )
    return DASHBOARD.substitute(
        css=CSS,
        user_name=_esc(user.get("full_name") or user.get("login")),
        role=_esc(
            {"operator": "дежурный", "admin": "администратор"}.get(
                str(user.get("role")), user.get("role")
            )
        ),
        kpis=_kpis(counts),
        series_options=_series_options(series_names, selected),
        chart_src="/chart/" + html.escape(selected, quote=True) + ".png",
        observe_block=observe_block,
        alert_count=alert_range_text(page, total_alerts, page_size),
        alert_filters=alert_filters_html(filters, series_names),
        alert_rows=_alert_rows(alerts, can_operate),
        pagination=pagination_html(
            page, total_alerts, filters, page_size, quality_page=quality_page
        ),
        stat_count=alert_range_text(quality_page, total_stats, quality_page_size),
        stat_pagination=quality_pagination_html(
            quality_page, total_stats, filters, page, quality_page_size
        ),
        stat_rows=_stat_rows(stats),
    )


def login_html(error: Optional[str] = None, next_url: str = "/") -> str:
    error_block = f'<div class="error">{_esc(error)}</div>' if error else ""
    return LOGIN.substitute(css=CSS, error=error_block, next_url=_esc(next_url))


def render_chart(series_name: str) -> bytes:
    cfg = load_config()
    path = Path(cfg["paths"]["processed"]) / f"{series_name}.csv"
    with _CHART_LOCK:
        fig = Figure(figsize=(11, 4.8), facecolor="#0f162b")
        canvas = FigureCanvasAgg(fig)
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
        for ax in (ax1, ax2):
            ax.set_facecolor("#0f162b")
            ax.tick_params(colors="#93a0bf", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#273252")
            ax.grid(color="#202b4a", linewidth=0.6)
        if not path.exists():
            ax1.text(0.5, 0.5, "Нет данных. Запустите pipeline.", ha="center", va="center",
                     color="#e8ecf8", transform=ax1.transAxes, fontsize=14)
            ax1.set_xticks([])
            ax1.set_yticks([])
            buf = io.BytesIO()
            canvas.print_png(buf)
            return buf.getvalue()

        df = pd.read_csv(path, index_col=0, parse_dates=True)
        n = len(df)
        start = int(n * (cfg["split"]["train"] + cfg["split"]["validation"]))
        with connect(cfg) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT series_id, criticality, step_seconds FROM time_series WHERE name = %s",
                    (series_name,),
                )
                row = cur.fetchone()
            series_id = int(row[0]) if row else None
            criticality = int(row[1]) if row else 3
            step = int(row[2]) if row else 300
            windows = fetch_test_windows(conn, series_id) if series_id else []
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT evidence_warning, evidence_incident FROM detection_params "
                    "WHERE series_id = %s ORDER BY params_id DESC LIMIT 1",
                    (series_id,),
                )
                prow = cur.fetchone()
            thresholds = risk_thresholds(cfg, criticality)
            e_warn = float(prow[0]) if prow else thresholds["warning"]
            e_inc = float(prow[1]) if prow else thresholds["incident"]

            view = df.iloc[max(0, start - int(86400 * 2 / max(step, 1))):]
            live = (
                fetch_decision_window(
                    conn,
                    series_id,
                    view.index.min().to_pydatetime(),
                    view.index.max().to_pydatetime(),
                )
                if series_id and len(view)
                else []
            )

        if live:
            live_df = pd.DataFrame(live)
            live_df["ts"] = live_df["ts"].map(normalize_ts)
            live_df = live_df.set_index("ts").sort_index()
            live_cols = [c for c in ("value", "evidence", "action") if c in live_df]
            base = view.drop(columns=[c for c in live_cols if c in view.columns])
            view = base.join(live_df[live_cols], how="outer").sort_index()

        ax1.plot(view.index, view["value"], color="#61a8ff", linewidth=0.9)
        for w in windows:
            ax1.axvspan(normalize_ts(w["start_ts"]), normalize_ts(w["end_ts"]),
                        color="#ff5c6c", alpha=0.12)
            ax2.axvspan(normalize_ts(w["start_ts"]), normalize_ts(w["end_ts"]),
                        color="#ff5c6c", alpha=0.12)
        alerts = view[view["action"].isin(["WARNING", "INCIDENT"])]
        colors = alerts["action"].map({"WARNING": "#ffb020", "INCIDENT": "#ff5c6c"})
        ax1.scatter(alerts.index, alerts["value"], c=colors, s=34, zorder=4)
        ax1.set_title(f"{series_name}  ·  criticality {criticality}", color="#dfe6ff",
                      fontsize=10, loc="left")

        ax2.plot(view.index, view["evidence"], color="#9aa7c7", linewidth=0.8)
        ax2.axhline(e_warn, color="#ffb020", linestyle="--", linewidth=1)
        ax2.axhline(e_inc, color="#ff5c6c", linestyle=":", linewidth=1)
        ax2.set_ylabel("-log10 p", color="#93a0bf", fontsize=8)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        fig.autofmt_xdate(rotation=0, ha="center")
        fig.tight_layout(pad=1.2)
        buf = io.BytesIO()
        canvas.print_png(buf)
        return buf.getvalue()
