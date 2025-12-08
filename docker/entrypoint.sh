#!/usr/bin/env bash
set -euo pipefail

echo "[entrypoint] Waiting for dependencies..."

wait_for_service() {
  local name="$1"
  local host="$2"
  local port="$3"
  for i in $(seq 1 60); do
    if nc -z "$host" "$port" >/dev/null 2>&1; then
      echo "[entrypoint] $name is up at $host:$port"
      return 0
    fi
    echo "[entrypoint] Waiting for $name at $host:$port (attempt $i)..."
    sleep 2
  done
  echo "[entrypoint] $name not reachable; exiting."
  exit 1
}

wait_for_service "postgres" "${POSTGRES_HOST:-postgres}" "${POSTGRES_PORT:-5432}"
wait_for_service "qdrant" "${QDRANT_HOST:-qdrant}" "${QDRANT_PORT:-6333}"
wait_for_service "ollama" "${OLLAMA_HOST:-127.0.0.1}" "${OLLAMA_PORT:-11435}"

echo "[entrypoint] Running DB init..."
python /app/db/init_db.py || true

if [ "${RUN_INGEST:-false}" = "true" ]; then
  echo "[entrypoint] Running ingestion..."
  bash /app/docker/ingest.sh || true
fi

echo "[entrypoint] Starting Chainlit..."
exec chainlit run /app/app.py -h 0.0.0.0 -p 8000
