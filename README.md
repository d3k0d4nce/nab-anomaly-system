# NAB Anomaly Detection System

Course project for stages 2–4 of the anomaly-monitoring assignment. The system
uses the [Numenta Anomaly Benchmark](https://github.com/numenta/NAB) telemetry,
stores all metadata and decisions in PostgreSQL, evaluates a stage-2 baseline
against a combined rules+model detector, applies a criticality-aware action
policy, and demonstrates a normal → failure → recovery replay.

## What is implemented

- **Stage 2 — data and baseline**
  - dataset download/validation, regular grid, missing-value audit, gap report,
    daily seasonality estimate, raw and combined label windows;
  - chronological 30% / 30% / 40% train / validation / test split;
  - rolling robust-z baseline with the same action/episode policy.
- **Stage 3 — combined detector and actions**
  - five causal robust-z features: short level, long level, first difference,
    seasonal residual and level deviation;
  - per-feature peaks-over-threshold GPD tail models and `evidence = max(-log10 p)`;
  - probability calibration on the validation split (isotonic, with a smooth
    fallback for control series without validation anomalies);
  - `SKIP` / `WARNING` / `INCIDENT` action engine with `k=2` confirmation,
    48-point episode tail, 0.6 hysteresis exit, at most one warning and one
    incident per episode, immediate `INCIDENT` for critical evidence, and
    criticality-aware thresholds;
  - PostgreSQL persistence of series, observations, parameters, episodes,
    decisions, alerts, engineers, check results, logs, quality stats and labels.
- **Stage 4 — evaluation, reports and service**
  - event-level TP/FP/FN, recall, precision, F1, false positives per 24 h,
    latency from the raw onset, and cost in `у.е.`;
  - pilot benchmark + separate stress report;
  - figures, `reports/final_report.md`, `reports/metrics.json`,
    `reports/data_audit.*`, `reports/demo_log.txt`;
  - replay demo and FastAPI streaming endpoint with data-quality/history gates;
  - web dashboard with charts, alerts, workflow buttons, operational KPIs,
    quality stats, manual point entry and live auto-refresh;
  - interactive alert workflow: acknowledge, confirm/false alarm, diagnostics,
    close, journal and live operational statistics;
  - session authentication for the duty engineer (one role);
  - Docker Compose deployment: PostgreSQL + FastAPI + nginx.

## Quick start

The easiest option is Docker Compose (requires Docker Desktop/Engine with the
Compose plugin; Python and PostgreSQL are not needed on the host):

```bash
git clone <your-repo-url> nab-anomaly-system
cd nab-anomaly-system
cp .env.example .env          # then change NAB_SECRET_KEY and the demo passwords
docker compose up --build -d  # first start downloads NAB and trains detectors
# dashboard: http://127.0.0.1:8081
```

The first build/start can take several minutes: the container clones the NAB
dataset (~350 MB), runs the audit, trains the detectors and fills PostgreSQL.
Follow progress with `docker compose logs -f app`.

For a local Python run, see the step-by-step guide in
[`ИНСТРУКЦИЯ.md`](ИНСТРУКЦИЯ.md). In short:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

# PostgreSQL must be running; the database is created automatically.
# Set PGUSER/PGPASSWORD if the defaults do not match your installation.
python -m nab_system.cli pipeline   # clones NAB on first run, trains, evaluates
python -m nab_system.cli demo
python -m nab_system.cli report
pytest -q

# Dashboard + API (opens at http://127.0.0.1:8000)
python -m nab_system.cli serve --host 127.0.0.1 --port 8000
```

The same commands are available as scripts:

```bash
python scripts/init_db.py --reset
python scripts/run_audit.py
python scripts/run_pipeline.py
python scripts/run_service_demo.py
```

The package is installed in editable mode (`pip install -e .`) and exposes the
`nab-anomaly` console script as well.

## Environment

- Python 3.9+ (3.11/3.12 recommended) or Docker
- PostgreSQL 16 (local service or the Compose `db` container)
- Default connection: `localhost:5432`, database `nab_anomaly`,
  user = `PGUSER` or the current OS user, empty password. The database is
  created automatically; the PostgreSQL role must already exist.
- Override with `PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`, `PGPASSWORD`.
- Web/auth env: `NAB_SECRET_KEY` (a stable random secret),
  `NAB_COOKIE_SECURE`, `NAB_OPERATOR_LOGIN/PASSWORD`,
  `NAB_ADMIN_LOGIN/PASSWORD`, `NAB_RUN_PIPELINE` (Compose only).

Docker Compose reads `.env`; a local Python run reads only real environment
variables. Export them in the shell before `serve` if needed.

Python dependencies: see `requirements.txt` and `pyproject.toml`.

## Project layout

```
config.yaml                  thresholds, split, benchmark and stress series, demo
src/nab_system/
  config.py                  config loading and risk map
  db.py, schema.sql          connection, schema, reset/init
  repository.py              all SQL operations
  data_loader.py             NAB download, CSV loading, regular grid, labels
  audit.py                   data-quality and seasonality audit
  features.py                batch and online causal features
  detector.py                GPD tail model, evidence, calibration
  actions.py                 episode state machine and dedup
  baseline.py                stage-2 baselines
  evaluator.py               event-level and point-level metrics
  pipeline.py                train/evaluate/persist orchestration
  service.py                 replay demo and FastAPI app, auth routes
  auth.py                    PBKDF2 password hashing, users and role checks
  dashboard.py               server-rendered dashboard and chart PNGs
  reports.py                 figures and final report
  cli.py                     command line interface
scripts/                     thin entry points
docker/                      nginx config and container entrypoint
Dockerfile, docker-compose.yml
tests/                       unit tests
reports/                     generated reports and figures
artifacts/detectors/         serialized detectors
data/processed/              scored series for inspection
logs/                        pipeline log
```

## Database model

The schema follows the assignment passport. 11 tables are used:

| table | purpose |
|---|---|
| `time_series` | series metadata, category, criticality, step |
| `observations` | regular-grid values, imputation flag, split |
| `detection_params` | per-series calibrated thresholds and episode settings |
| `deviation_episodes` | episode start/end, peak evidence, warning/incident flags |
| `decisions` | per-observation evidence, calibrated probability, action |
| `alerts` | WARNING/INCIDENT alerts, assigned engineer, response limit |
| `duty_engineers` | dashboard users: login, PBKDF2 hash, role, on-duty |
| `check_results` | engineer checks of alerts |
| `log_entries` | series and alert audit log |
| `quality_stats` | TP/FP/FN, recall, precision, F1, FP/24h, cost |
| `label_windows` | NAB combined windows and raw onsets |

Two references in the original passport are corrected: `log_entries.alert_id`
references `alerts` (not `duty_engineers`), and `check_results.engineer_id`
references `duty_engineers` (not `engineers`). Two audit fields are added:
`time_series.criticality` / `category`, and `observations.split` / `is_imputed`.
Auth columns (`login`, `password_hash`, `role`, `is_active`) are added to
`duty_engineers`, so the original 11-table passport is preserved.

## Detector and action policy

For every observation the detector computes five scale-free features. Tail
models are fitted only on normal training points, excluding labelled windows
and a warm-up prefix. The maximum tail surprise over the five features is the
evidence. Validation data calibrates evidence to `P(anomaly)`.

Operational evidence thresholds by criticality:

| criticality | WARNING | INCIDENT |
|---|---:|---:|
| 1 | 7.0 | 9.0 |
| 2 | 6.0 | 8.0 |
| 3 | 5.0 | 7.0 |
| 4 | 4.5 | 6.5 |
| 5 | 4.0 | 6.0 |

The passport's theoretical cost thresholds (`p = 0.14` and `p = 0.50`) are
converted into these evidence thresholds and then into per-series calibrated
probabilities; both are stored in `detection_params`.

An episode starts after `k=2` consecutive points above the warning threshold.
An episode normally starts with `WARNING` after `k=2` points above the warning
threshold. If a point immediately exceeds the incident threshold, the system
creates `INCIDENT` at once without waiting for `WARNING` or operator
confirmation; otherwise the active episode is escalated to one `INCIDENT`.
The episode ends after 48 points below `0.6 × warning_threshold`. Repeat alerts
inside the same episode are suppressed.

## Results

Latest pilot run (`reports/metrics.json`):

| metric | combined | rolling-z baseline |
|---|---:|---:|
| recall | 0.833 | 1.000 |
| precision | 1.000 | 0.231 |
| worst FP / 24h | 0.000 | 1.371 |
| cost, у.е. | 18.0 | 72.0 |
| p95 latency | 1 ms | 1 ms |

All four success criteria pass: recall ≥ 0.80, precision ≥ 0.70,
FP ≤ 2 per series per 24 h, and p95 latency < 1 s. One pilot window
(`ec2_network_in_5abac7`) is not detected because its evidence remains below
the warning threshold.

The stress set (`machine_temperature`, `nyc_taxi`, `speed_t4013`,
`Twitter_volume_AMZN`, `exchange-4_cpm_results`, `ec2_cpu_utilization_24ae8d`)
is reported separately; its aggregate recall/precision is 0.786 / 0.379 and
cost is 98.0 у.е. These series are non-stationary or have very wide contextual
labels and are not part of the pilot acceptance criteria.

## Demo

```bash
python -m nab_system.cli demo
```

The replay uses `ec2_request_latency_system_failure`, around the first test
failure. It shows normal operation, immediate `INCIDENT` at 22:36 when evidence
crosses the critical threshold, and recovery after the episode closes at 02:46.
Full output is written to `reports/demo_log.txt`.

## HTTP service

```bash
python -m nab_system.cli serve --port 8000
# or: python scripts/serve.py --port 8000
```

Open `http://127.0.0.1:8000/` for the dashboard. Two demo accounts are seeded
(override with `NAB_OPERATOR_LOGIN/PASSWORD` and `NAB_ADMIN_LOGIN/PASSWORD`):

| login | password | role | access |
|---|---|---|---|
| `operator` | `demo-operator` | дежурный инженер | dashboard, API, alert workflow |
| `admin` | `demo-admin` | администратор | dashboard, API, `/observe`, alert workflow |

Endpoints:

- `GET /` — HTML dashboard (login required, auto-refreshes every 10 s);
  query: `page`, `qpage`, `status`, `alert_type`, `series`;
- `GET /dashboard/partials` — KPI/alert/stat/pagination HTML fragments for
  live updates; same query parameters as `/`;
- `GET /chart/{series}.png` — chart with test labels, actions and evidence;
- `GET /login`, `POST /login`, `GET /logout` — session auth;
- `GET /health` — public service status;
- `GET /me` — current user;
- `GET /series` — known series;
- `POST /observe` — `{"series": "...", "timestamp": "...", "value": 12.3}`,
  administrator only;
- `GET /alerts?limit=50` — recent alerts with check results;
- `GET /alerts/{id}` — alert details and its journal entries;
- `POST /alerts/{id}/ack` — take an alert into work (duty engineer);
- `POST /alerts/{id}/check` — duty-engineer result
  `{"is_real_fault": true|false, "comment": "...", "diagnostics": "..."}`;
  a real fault moves the alert to `ACKNOWLEDGED`, a false alarm to `CLOSED`;
- `POST /alerts/{id}/close` — close an acknowledged alert after diagnostics;
- `GET /stats?limit=100` — quality statistics;
- `GET /stats/operational` — live alert/check counters.

Sessions are signed with `itsdangerous`, stored in an HttpOnly SameSite=Lax
cookie. Passwords use PBKDF2-HMAC-SHA256 (260k iterations, per-user salt). The
assignment has one role: `operator` (duty engineer); other accounts are disabled
on startup.

### Manual point input and alert response

The duty engineer can submit one point from the dashboard or via `POST /observe`.
The service loads the detector, warms the online state from recent observations,
validates timestamp/value, checks that enough history exists, computes
features/evidence/probability, applies the action policy, and persists the
observation, decision, episode and alert. A duplicate timestamp is rejected
with `409`; a non-finite value is rejected with `400`; insufficient history
returns `SKIP` with `data_quality.enough_history=false`. Timestamps should be
sequential; a single point may return `SKIP` because an episode needs `k=2`
confirmed points.

Processing follows the supplied scheme:

1. new observation (series, time, value) and a window of previous values;
2. completeness/correctness check and `min_history` gate;
3. anomaly score; below threshold → `SKIP`;
4. immediate exceedance of the incident threshold → `INCIDENT` at once;
5. otherwise sustained deviation (`k=2`) above warning threshold → `WARNING`,
   then one `INCIDENT` escalation if the episode reaches the incident threshold;
6. warning/incident is logged, the episode is marked `warning_sent`/`incident_sent`,
   and an incident is stored with status `OPEN`;
7. duty engineer takes the alert into work and submits a check result;
8. real fault → `confirmed`, diagnostics/mitigation, then close; false alarm →
   `false_alarm` and close;
9. operational statistics and journal are updated, then the window is passed to
   the next observation.

The scheme has one contradictory branch: both “выраженное отклонение” and
“устойчивое отклонение” point to `INCIDENT`, while a separate `WARNING` node
exists. The implementation resolves it as follows: critical evidence creates
`INCIDENT` immediately (no operator confirmation is required for creation);
non-critical sustained evidence creates `WARNING`, with one later `INCIDENT`
escalation if the episode reaches the critical threshold. The web workflow is
independent of this policy choice.

The dashboard shows the alert queue, newest by event time (`observations.ts`)
first, 10 alerts per page with pagination and filters by status
(`ACTIVE`/`OPEN`/`ACKNOWLEDGED`/`CLOSED`), type (`WARNING`/`INCIDENT`) and
series. The heading shows the visible range and total. The quality table is
paginated separately, 10 rows per page (`qpage`), and preserves the alert
filters and page. Actions are `В работу`, `Реальный сбой`, `Ложная тревога`
and `Закрыть` (дежурный инженер).
Automatic pilot checks are
kept as historical `check_results`: confirmed alerts are marked
`ACKNOWLEDGED`, false alarms `CLOSED`. New manual alerts start as `OPEN`.
The dashboard refreshes KPIs, alerts and statistics every 10 s and the chart
every 30 s; a manual point updates the chart and tables immediately.

A continuing failure is one episode: while it is active, new high values return
`SKIP` with `reason="внутри активного эпизода"` and do not create duplicate
alerts. The episode ends after `end_after` low points; then a new anomaly can
create a new alert. This is intentional deduplication, not lost data: every new
point still updates the chart, decisions journal and counters.

The first call for a series loads its detector and warms the online state from
recent database observations. Use sequential timestamps for meaningful
streaming decisions.

## Docker + nginx

```bash
cp .env.example .env
# edit .env: change NAB_SECRET_KEY and all demo passwords
docker compose up --build
# dashboard: http://127.0.0.1:8081
```

The `app` container waits for PostgreSQL, then starts uvicorn behind nginx.
With `NAB_RUN_PIPELINE=auto` (default) it runs the pipeline only when the
database or detectors are missing. Set `NAB_RUN_PIPELINE=1` to force a reset
and re-run, or `0` to skip. Data, artifacts, reports and
logs are stored in named volumes; PostgreSQL data is in `pgdata`. To copy
generated reports out of the volume:

```bash
docker compose cp app:/app/reports/final_report.md ./final_report.md
docker compose exec app ls /app/reports
```

For a public deployment, terminate TLS at nginx (or a load balancer), set
`NAB_COOKIE_SECURE=1`, use a strong random `NAB_SECRET_KEY`, and change all
default passwords before exposing port 8081.

## Reproducibility

`seed` is fixed in `config.yaml`. The pipeline resets the schema by default;
use `--no-reset` to append. Every run rewrites `reports/metrics.json`,
`reports/final_report.md`, the audit files, figure files, processed series and
serialized detectors.

## Limitations

- The GPD tail model assumes the training split is representative; strong
  regime changes produce false positives (see the stress report).
- NAB raw labels are used for onsets, while combined windows are used for
  masking and matching. Contextual anomalies can begin before the test split;
  such windows are attributed to the split containing their onset.
- The service is single-process and keeps runtime state in memory. A production
  deployment would use a shared state store and a queue.
- Auth is intentionally small: no registration, password reset, CSRF token,
  rate limiting or MFA. The dashboard and nginx config are HTTP-only by default;
  add TLS and `NAB_COOKIE_SECURE=1` before public exposure.
- Demo passwords are re-applied from env at startup, so change them together
  with `NAB_SECRET_KEY` and restart the app.
