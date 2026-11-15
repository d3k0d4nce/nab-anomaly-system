from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import load_config
from .db import connect
from .repository import fetch_test_windows, normalize_ts


def _md_table(rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> str:
    head = "| " + " | ".join(columns) + " |\n"
    sep = "|" + "|".join(["---"] * len(columns)) + "|\n"
    body = []
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                cells.append(f"{value:.3f}")
            else:
                cells.append(str(value))
        body.append("| " + " | ".join(cells) + " |")
    return head + sep + "\n".join(body) + "\n"


def _short_result(r: Dict[str, Any]) -> Dict[str, Any]:
    c = r["combined"]
    b = r["baseline"]
    return {
        "series": r["series"],
        "crit": r["criticality"],
        "windows": c["windows_total"],
        "TP": c["tp"],
        "FP": c["fp"],
        "FN": c["fn"],
        "recall": round(c["recall"], 3),
        "precision": round(c["precision"], 3),
        "fp/24h": round(c["fp_per_24h"], 3),
        "latency_obs": round(c["latency_obs_p95"], 2),
        "cost": round(c["cost_units"], 1),
        "baseline_recall": round(b["recall"], 3),
        "baseline_precision": round(b["precision"], 3),
        "baseline_cost": round(b["cost_units"], 1),
    }


def _plot_series(cfg: Dict[str, Any], r: Dict[str, Any], out_dir: Path) -> None:
    name = r["series"]
    path = Path(cfg["paths"]["processed"]) / f"{name}.csv"
    if not path.exists():
        return
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    params = r["params"]
    with connect(cfg) as conn:
        series_id = None
        with conn.cursor() as cur:
            cur.execute("SELECT series_id FROM time_series WHERE name = %s", (name,))
            row = cur.fetchone()
            series_id = int(row[0]) if row else None
        windows = fetch_test_windows(conn, series_id) if series_id else []

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    ax1.plot(df.index, df["value"], color="#1f77b4", linewidth=0.8, label="value")
    for w in windows:
        ax1.axvspan(normalize_ts(w["start_ts"]), normalize_ts(w["end_ts"]), color="#ffcccc", alpha=0.5)
    alerts = df[df["action"] != "SKIP"]
    colors = alerts["action"].map({"WARNING": "#ff9900", "INCIDENT": "#cc0000"})
    ax1.scatter(alerts.index, alerts["value"], c=colors, s=28, zorder=5, label="alert")
    ax1.set_ylabel("value")
    ax1.set_title(f"{name} — test split")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.25)

    ax2.plot(df.index, df["evidence"], color="#444444", linewidth=0.7, label="evidence")
    ax2.axhline(params["evidence_warning"], color="#ff9900", linestyle="--", linewidth=1, label="warning")
    ax2.axhline(params["evidence_incident"], color="#cc0000", linestyle=":", linewidth=1, label="incident")
    for w in windows:
        ax2.axvspan(pd.Timestamp(w["start_ts"]), pd.Timestamp(w["end_ts"]), color="#ffcccc", alpha=0.5)
    ax2.set_ylabel("-log10 p")
    ax2.legend(loc="upper left")
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=130)
    plt.close(fig)


def _plot_summary(metrics: Dict[str, Any], out_dir: Path) -> None:
    for key, filename, title in (
        ("pilot", "pilot_summary.png", "Pilot benchmark"),
        ("stress", "stress_summary.png", "Stress series"),
    ):
        rows = metrics.get(key, [])
        if not rows:
            continue
        names = [r["series"] for r in rows]
        recall = [r["combined"]["recall"] for r in rows]
        precision = [r["combined"]["precision"] for r in rows]
        cost = [r["combined"]["cost_units"] for r in rows]
        x = np.arange(len(names))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        ax1.bar(x - 0.2, recall, width=0.4, label="recall", color="#2ca02c")
        ax1.bar(x + 0.2, precision, width=0.4, label="precision", color="#1f77b4")
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
        ax1.set_ylim(0, 1.05)
        ax1.set_title(f"{title}: event metrics")
        ax1.legend()
        ax1.grid(axis="y", alpha=0.25)
        ax2.bar(x, cost, color="#d62728")
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
        ax2.set_title(f"{title}: cost, у.е.")
        ax2.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=130)
        plt.close(fig)


def _plot_calibration(cfg: Dict[str, Any], metrics: Dict[str, Any], out_dir: Path) -> None:
    from .detector import AnomalyDetector

    grid = np.linspace(0, 12, 200)
    fig, ax = plt.subplots(figsize=(7, 4))
    for r in metrics.get("pilot", []):
        path = Path(cfg["paths"]["artifacts"]) / "detectors" / f"{r['series']}.pkl"
        if not path.exists():
            continue
        det = AnomalyDetector.load(str(path))
        p = det.calibrator.predict(grid)
        ax.plot(grid, p, linewidth=1.2, label=r["series"])
    ax.set_xlabel("evidence = -log10 tail p")
    ax.set_ylabel("calibrated P(anomaly)")
    ax.set_title("Validation calibration")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "calibration.png", dpi=130)
    plt.close(fig)


def generate_reports(cfg_path: str | None = None) -> Path:
    cfg = load_config(cfg_path)
    reports = Path(cfg["paths"]["reports"])
    figures = Path(cfg["paths"]["figures"])
    figures.mkdir(parents=True, exist_ok=True)
    metrics_path = reports / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError("Run the pipeline before generating reports")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    for r in metrics.get("pilot", []):
        _plot_series(cfg, r, figures)
    _plot_summary(metrics, figures)
    _plot_calibration(cfg, metrics, figures)

    audit_rows: List[Dict[str, Any]] = []
    audit_path = reports / "data_audit.json"
    if audit_path.exists():
        audit_rows = json.loads(audit_path.read_text(encoding="utf-8"))

    pilot_rows = [_short_result(r) for r in metrics.get("pilot", [])]
    stress_rows = [_short_result(r) for r in metrics.get("stress", [])]
    agg = metrics["pilot_aggregate_combined"]
    base = metrics["pilot_aggregate_baseline"]
    stress_agg = metrics.get("stress_aggregate_combined", {})
    criteria = [
        ("Recall ≥ 0.80", agg["recall_ok"], f"{agg['recall']:.3f}"),
        ("Precision ≥ 0.70", agg["precision_ok"], f"{agg['precision']:.3f}"),
        ("FP ≤ 2 / series / 24h", agg["fp_ok"], f"worst {agg['worst_fp_per_24h']:.3f}"),
        ("p95 decision latency < 1 s", agg["latency_ok"], f"{agg['latency_ms_p95']:.1f} ms"),
    ]

    demo_text = ""
    demo_path = reports / "demo_log.txt"
    if demo_path.exists():
        lines = demo_path.read_text(encoding="utf-8").splitlines()
        selected = [
            line
            for line in lines
            if line.startswith("failure start")
            or line.startswith("recovery start")
            or line.startswith("actions:")
            or "WARNING" in line
            or "INCIDENT" in line
        ]
        demo_text = "\n".join(selected)

    md = []
    md.append("# NAB Anomaly Detection System — Final Report\n")
    md.append(
        "Stages 2–4: data audit, split, baseline, combined detector, action policy, "
        "quality evaluation and service demonstration.\n"
    )
    md.append("## 1. Data audit\n")
    if audit_rows:
        md.append(_md_table(
            audit_rows,
            ["series", "regular_rows", "step_seconds", "gap_count", "max_gap_seconds",
             "imputed_values", "daily_acf", "label_windows", "label_positive_points"],
        ))
        md.append(
            "Full audit: `reports/data_audit.md`, `reports/data_audit.json`. "
            "Missing short gaps are linearly interpolated (≤ 6 points); longer gaps are kept as NaN "
            "and are not used for fitting.\n"
        )
    else:
        md.append("Audit file not found; run `python -m nab_system.cli audit`.\n")

    md.append("## 2. Split and detector\n")
    md.append(
        f"Split: train/validation/test = {cfg['split']['train']:.0%}/"
        f"{cfg['split']['validation']:.0%}/{cfg['split']['test']:.0%}. "
        "The train split excludes labelled windows and a warm-up prefix. "
        "The validation split is used only for probability calibration.\n\n"
        "Detector: five causal robust-z features (`z_short`, `z_long`, `z_diff`, `z_seas`, "
        "`z_level`) with global MAD floors; per-feature peaks-over-threshold GPD tail models "
        "(q=0.95); evidence = max(-log10 p). `WARNING` requires `k=2` consecutive points "
        "above the warning threshold; episodes end after 48 quiet points and use a 0.6 "
        "hysteresis exit. Critical evidence creates `INCIDENT` immediately; otherwise "
        "`WARNING` is sent first and one later `INCIDENT` escalation is allowed. Each "
        "alert type is sent at most once per episode.\n"
    )
    md.append("### Criticality-aware evidence thresholds\n")
    md.append(_md_table(
        [{"criticality": k, "warning": v["warning"], "incident": v["incident"]}
         for k, v in cfg["detection"]["risk_thresholds"].items()],
        ["criticality", "warning", "incident"],
    ))
    md.append(
        "The passport's theoretical cost thresholds (p=0.14 / 0.50) are converted to "
        "operational evidence thresholds and then to calibrated probabilities per series. "
        "The probability columns below show the resulting per-series operating points.\n"
    )
    cal_rows = []
    for r in metrics.get("pilot", []):
        cal_rows.append({
            "series": r["series"],
            "calibration": r["calibration"]["kind"],
            "e_warning": round(r["params"]["evidence_warning"], 2),
            "e_incident": round(r["params"]["evidence_incident"], 2),
            "p_warning": round(r["calibration"]["p_warning"], 3),
            "p_incident": round(r["calibration"]["p_incident"], 3),
        })
    md.append(_md_table(cal_rows, ["series", "calibration", "e_warning", "e_incident", "p_warning", "p_incident"]))

    md.append("## 3. Stage-2 baseline vs combined detector\n")
    md.append(
        f"Rolling-z baseline: recall={base['recall']:.3f}, precision={base['precision']:.3f}, "
        f"cost={base['cost_units']:.1f} у.е. Combined detector: recall={agg['recall']:.3f}, "
        f"precision={agg['precision']:.3f}, cost={agg['cost_units']:.1f} у.е.\n"
    )
    md.append(_md_table(pilot_rows, list(pilot_rows[0].keys()) if pilot_rows else []))

    check_total = check_confirmed = 0
    with connect(cfg) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*), coalesce(sum(CASE WHEN is_real_fault THEN 1 ELSE 0 END), 0) FROM check_results")
            row = cur.fetchone()
            if row:
                check_total, check_confirmed = int(row[0]), int(row[1])
    if check_total:
        md.append(
            f"First-line checks (`check_results`): {check_total} alerts checked automatically "
            f"for the pilot history, {check_confirmed} confirmed as real faults, "
            f"{check_total - check_confirmed} false alarms. Historical confirmed alerts are "
            "marked `ACKNOWLEDGED` (diagnostics/closure pending), false alarms are `CLOSED`; "
            "new manual alerts start as `OPEN` and use the interactive duty-engineer workflow.\n"
        )

    md.append("## 4. Success criteria\n")
    md.append(_md_table(
        [{"criterion": c, "status": "PASS" if ok else "FAIL", "value": v} for c, ok, v in criteria],
        ["criterion", "status", "value"],
    ))
    md.append(
        "Event-level metrics use whole episodes: one long episode may cover several label "
        "windows, so warning+incident of the same episode are not counted as separate FP. "
        "Latency is measured from the raw NAB onset to the first alert of the episode.\n"
    )
    md.append("![Pilot summary](figures/pilot_summary.png)\n")

    md.append("## 5. Stress series (separate report)\n")
    md.append(
        "These series are deliberately non-stationary or have very wide contextual labels. "
        "They are not part of the pilot success criteria; the table documents the degradation.\n"
    )
    md.append(_md_table(stress_rows, list(stress_rows[0].keys()) if stress_rows else []))
    if stress_agg:
        md.append(
            f"Stress aggregate: recall={stress_agg['recall']:.3f}, "
            f"precision={stress_agg['precision']:.3f}, cost={stress_agg['cost_units']:.1f} у.е.\n"
        )
    md.append("![Stress summary](figures/stress_summary.png)\n")

    md.append("## 6. Service demo\n")
    md.append(
        f"Replay around the first test failure of `{cfg['demo']['series']}`. "
        "The demo shows normal operation, immediate `INCIDENT` at the first critical "
        "point, and recovery after the episode closes. Full log: `reports/demo_log.txt`.\n"
    )
    if demo_text:
        md.append("```\n" + demo_text + "\n```\n")
    md.append("![Calibration](figures/calibration.png)\n")

    md.append("## 7. Web dashboard and deployment\n")
    md.append(
        "The FastAPI app also serves a server-rendered dashboard (`/`) with a chart "
        "per series, latest alerts, operational KPIs, quality statistics and manual "
        "point entry. Sessions use signed HttpOnly cookies; passwords use "
        "PBKDF2-HMAC-SHA256. The assignment defines one actor: the duty engineer "
        "(`operator`); extra demo roles are disabled on startup. `/observe` and the "
        "alert workflow require the duty engineer. The dashboard refreshes KPIs, "
        "alerts and statistics every 10 s, and the chart every 30 s; manual points "
        "update it immediately. Docker Compose runs PostgreSQL, the app and nginx; "
        "the app waits for the database and can run the full pipeline on first "
        "start.\n"
    )
    md.append(
        "The alert workflow implements the supplied processing scheme: data-quality "
        "and `min_history` gates, anomaly score, immediate `INCIDENT` for critical "
        "evidence, otherwise `WARNING` after `k=2` with one later `INCIDENT` "
        "escalation, status `OPEN` with episode context, duty-engineer check "
        "(`is_real_fault`), confirmation/diagnostics or false-alarm closure, "
        "operational statistics, and journal entries for decisions/checks. Operators "
        "can use `POST /alerts/{id}/ack`, `/check` and `/close` or the dashboard "
        "buttons. The scheme’s contradictory arrows (both “выраженное” and "
        "“устойчивое” deviation point to `INCIDENT`) are resolved by the critical-"
        "evidence rule above; the web workflow is independent of that policy choice.\n"
    )

    md.append("## 8. Reproduce\n")
    md.append(
        "```bash\n"
        "source .venv/bin/activate\n"
        "python -m nab_system.cli init-db\n"
        "python -m nab_system.cli audit\n"
        "python -m nab_system.cli pipeline\n"
        "python -m nab_system.cli demo\n"
        "python -m nab_system.cli report\n"
        "python -m nab_system.cli serve --port 8000\n"
        "pytest -q\n"
        "docker compose up --build  # dashboard at http://127.0.0.1:8081\n"
        "```\n"
    )

    report_path = reports / "final_report.md"
    report_path.write_text("\n".join(md), encoding="utf-8")

    pd.DataFrame(pilot_rows).to_csv(reports / "pilot_metrics.csv", index=False)
    pd.DataFrame(stress_rows).to_csv(reports / "stress_metrics.csv", index=False)
    return report_path
