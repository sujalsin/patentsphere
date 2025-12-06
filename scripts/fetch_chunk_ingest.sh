#!/bin/bash
# Fetch patents from BigQuery, chunk them, and ingest into Qdrant

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
KEY_DIR="/home/sujals2144/project/key"

echo "=========================================="
echo "PatentSphere Data Pipeline"
echo "=========================================="

# Find credentials file
CREDENTIALS_FILE=""
if [ -d "$KEY_DIR" ]; then
    CREDENTIALS_FILE=$(find "$KEY_DIR" -name "*.json" -type f | head -1)
fi

if [ -z "$CREDENTIALS_FILE" ]; then
    echo "Error: No credentials JSON file found in $KEY_DIR"
    echo "Please place your GCP service account JSON file in that directory"
    exit 1
fi

echo "Using credentials: $CREDENTIALS_FILE"
export GOOGLE_APPLICATION_CREDENTIALS="$CREDENTIALS_FILE"
export GCP_ALLOW_BIGQUERY_EXPORTS=true

# Set project ID
export GCP_PROJECT_ID="ancient-courage-478809-p0"

cd "$PROJECT_ROOT"

# Step 1: Fetch patents from BigQuery
echo ""
echo "Step 1: Fetching patents from BigQuery..."
python scripts/fetch_bigquery_subset.py \
    --limit 10000 \
    --output data/patents_bigquery.jsonl \
    --execute

if [ ! -f "data/patents_bigquery.jsonl" ]; then
    echo "Error: Failed to fetch patents"
    exit 1
fi

PATENT_COUNT=$(wc -l < data/patents_bigquery.jsonl)
echo "✓ Fetched $PATENT_COUNT patents"

# Step 2: Chunk patents
echo ""
echo "Step 2: Chunking patents..."
python scripts/process_patents.py \
    --input data/patents_bigquery.jsonl \
    --output-dir data/processed \
    --chunking-strategy hybrid \
    --max-patents 10000 \
    --generate-embeddings

if [ ! -f "data/processed/chunks.jsonl" ]; then
    echo "Error: Failed to chunk patents"
    exit 1
fi

CHUNK_COUNT=$(wc -l < data/processed/chunks.jsonl)
echo "✓ Generated $CHUNK_COUNT chunks"

# Step 3: Ingest into databases
echo ""
echo "Step 3: Ingesting into Postgres and Qdrant..."
python scripts/ingest_local.py \
    --chunks data/processed/chunks.jsonl \
    --embeddings data/processed/embeddings.pt

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo "Patents: $PATENT_COUNT"
echo "Chunks: $CHUNK_COUNT"
echo ""
echo "Data is now available in:"
echo "  - Postgres: patent_chunks table"
echo "  - Qdrant: patents collection"
