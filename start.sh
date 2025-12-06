#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

log_step() {
    printf "\n[patentsphere:start] %s\n" "$1"
}

# --- Docker services ---------------------------------------------------------
if command -v systemctl >/dev/null 2>&1; then
    log_step "Ensuring Docker daemon is running"
    sudo systemctl enable docker >/dev/null 2>&1 || true
    sudo systemctl start docker
fi

log_step "Starting Postgres + Qdrant containers"
docker compose up -d postgres qdrant

wait_for_service() {
    local service="$1"
    local container_id
    container_id="$(docker compose ps -q "$service")"
    if [ -z "$container_id" ]; then
        echo "Unable to find container for service '$service'"
        exit 1
    fi

    echo "Waiting for $service to become healthy..."
    while true; do
        status="$(docker inspect --format='{{.State.Health.Status}}' "$container_id" 2>/dev/null || echo "starting")"
        if [ "$status" = "healthy" ]; then
            break
        fi
        sleep 2
    done
}

wait_for_service postgres
wait_for_service qdrant

# --- Python environment ------------------------------------------------------
VENV_PATH="$ROOT_DIR/.venv"
if [ ! -d "$VENV_PATH" ]; then
    log_step "Creating virtual environment (.venv)"
    python3 -m venv "$VENV_PATH"
fi

log_step "Activating virtual environment"
# shellcheck disable=SC1090
source "$VENV_PATH/bin/activate"

log_step "Installing Python dependencies"
pip install --upgrade pip >/dev/null
pip install -q -r requirements.txt

# --- Data ingestion (optional) ----------------------------------------------
# Set SKIP_INGEST=1 to bypass this step if the DB/Qdrant are already seeded.
if [ "${SKIP_INGEST:-0}" -ne 1 ]; then
    CHUNKS_FILE="data/processed/chunks.jsonl"
    EMB_FILE="data/processed/embeddings.pt"
    CITATIONS_FILE="data/citations.jsonl"
    LIT_FILE="data/litigation_data.jsonl"

    if [ -f "$CHUNKS_FILE" ] && [ -f "$EMB_FILE" ]; then
        log_step "Running ingestion pipeline (Postgres + Qdrant)"
        python scripts/ingest_local.py \
            --chunks "$CHUNKS_FILE" \
            --embeddings "$EMB_FILE" \
            ${CITATIONS_FILE:+--citations "$CITATIONS_FILE"} \
            ${LIT_FILE:+--litigation "$LIT_FILE"}
    else
        log_step "Skipping ingestion (chunk/embedding files not found)"
    fi
else
    log_step "SKIP_INGEST=1 -> Skipping ingestion step"
fi