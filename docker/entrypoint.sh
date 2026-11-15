#!/usr/bin/env bash
set -euo pipefail

: "${PGHOST:=db}"
: "${PGPORT:=5432}"
: "${PGDATABASE:=nab_anomaly}"
: "${PGUSER:=nab}"
: "${PGPASSWORD:=nab}"
: "${NAB_RUN_PIPELINE:=auto}"

export PGHOST PGPORT PGDATABASE PGUSER PGPASSWORD

echo "waiting for postgres at ${PGHOST}:${PGPORT}/${PGDATABASE}"
python - <<'PY'
import os
import sys
import time

import psycopg2

for attempt in range(1, 61):
    try:
        conn = psycopg2.connect(
            host=os.environ["PGHOST"],
            port=int(os.environ["PGPORT"]),
            dbname=os.environ["PGDATABASE"],
            user=os.environ["PGUSER"],
            password=os.environ.get("PGPASSWORD", ""),
        )
        conn.close()
        print("postgres is ready")
        break
    except Exception as exc:
        print(f"postgres not ready ({attempt}/60): {exc}")
        time.sleep(2)
else:
    sys.exit("postgres did not become ready")
PY

case "${NAB_RUN_PIPELINE}" in
    1|true|yes)
        run_pipeline=1
        ;;
    0|false|no)
        run_pipeline=0
        ;;
    auto|*)
        if python - <<'PY'
import glob
import os
import sys

import psycopg2

count = 0
try:
    conn = psycopg2.connect(
        host=os.environ["PGHOST"],
        port=int(os.environ["PGPORT"]),
        dbname=os.environ["PGDATABASE"],
        user=os.environ["PGUSER"],
        password=os.environ.get("PGPASSWORD", ""),
    )
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM time_series")
        count = int(cur.fetchone()[0])
    conn.close()
except Exception:
    count = 0
detectors = glob.glob("/app/artifacts/detectors/*.pkl")
sys.exit(0 if count == 0 or not detectors else 1)
PY
        then
            run_pipeline=1
        else
            run_pipeline=0
        fi
        ;;
esac

if [ "${run_pipeline}" = "1" ]; then
    echo "running NAB audit/training/evaluation/persistence pipeline"
    python -m nab_system.cli pipeline
else
    echo "skipping pipeline (database and detectors already present)"
fi

exec uvicorn nab_system.service:create_app \
    --factory \
    --host 0.0.0.0 \
    --port 8000 \
    --proxy-headers \
    --forwarded-allow-ips='*'
