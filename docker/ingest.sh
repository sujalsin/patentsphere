#!/usr/bin/env bash
set -euo pipefail

echo "[ingest] Waiting for Postgres/Qdrant..."
sleep 2

run_cmd() {
  echo "[ingest] $*"
  eval "$*"
}

# Initialize DB schema
run_cmd "python /app/db/init_db.py"

# Litigation ingestion
if [ -f /app/data/stanford_litigation.jsonl ]; then
  run_cmd "python /app/scripts/ingest_litigation.py /app/data/stanford_litigation.jsonl stanford"
fi
if [ -f /app/data/uspto_litigation.jsonl ]; then
  run_cmd "python /app/scripts/ingest_litigation.py /app/data/uspto_litigation.jsonl uspto_oce"
fi

# Patent ingestion
if [ -f /app/data/patents_2015.jsonl ]; then
  run_cmd "python /app/scripts/ingest_patents.py /app/data/patents_2015.jsonl"
fi
if [ -f /app/data/patents_2016.jsonl ]; then
  run_cmd "python /app/scripts/ingest_patents.py /app/data/patents_2016.jsonl"
fi
if [ -f /app/data/patents_bigquery.jsonl ]; then
  run_cmd "python /app/scripts/ingest_patents.py /app/data/patents_bigquery.jsonl"
fi

echo "[ingest] Done."
