#!/bin/bash
# Download and ingest patents from 2015-2018
# 2015: 50K, 2016: 50K, 2017: All, 2018: All

set -e

echo "============================================================"
echo "📥 Downloading and Ingesting Patents 2015-2018"
echo "============================================================"
echo ""

PROJECT_ID="ancient-courage-478809-p0"
DATA_DIR="data"

# Check authentication
echo "🔑 Checking BigQuery authentication..."
python3 -c "from google.cloud import bigquery; bigquery.Client(project='$PROJECT_ID'); print('✅ Authenticated')" 2>&1 || {
    echo "❌ Not authenticated. Run:"
    echo "   gcloud auth application-default login --project=$PROJECT_ID"
    exit 1
}

# 2015: 50K patents
# echo ""
# echo "============================================================"
# echo "📥 Step 1: Downloading 2015 patents (50K limit)"
# echo "============================================================"
# python3 scripts/download_patents_bigquery.py \
#     --start-year 2015 \
#     --end-year 2015 \
#     --limit 50000 \
#     --output ${DATA_DIR}/patents_2015.jsonl

# echo ""
# echo "🔄 Ingesting 2015 patents..."
# python3 scripts/ingest_patents_gpu_optimized.py \
#     ${DATA_DIR}/patents_2015.jsonl \
#     --keep-existing \
#     --workers 3 \
#     --embedding-batch-size 200

# 2016: 50K patents
# echo ""
# echo "============================================================"
# echo "📥 Step 2: Downloading 2016 patents (50K limit)"
# echo "============================================================"
# python3 scripts/download_patents_bigquery.py \
#     --start-year 2016 \
#     --end-year 2016 \
#     --limit 50000 \
#     --output ${DATA_DIR}/patents_2016.jsonl

echo ""
echo "🔄 Ingesting 2016 patents..."
python3 scripts/ingest_patents_gpu_optimized.py \
    ${DATA_DIR}/patents_2016.jsonl \
    --keep-existing \
    --workers 3 \
    --embedding-batch-size 200

# 2017: All patents
echo ""
echo "============================================================"
echo "📥 Step 3: Downloading 2017 patents (ALL)"
echo "============================================================"
python3 scripts/download_patents_bigquery.py \
    --start-year 2017 \
    --end-year 2017 \
    --limit 50000 \
    --output ${DATA_DIR}/patents_2017.jsonl

echo ""
echo "🔄 Ingesting 2017 patents..."
python3 scripts/ingest_patents_gpu_optimized.py \
    ${DATA_DIR}/patents_2017.jsonl \
    --keep-existing \
    --workers 3 \
    --embedding-batch-size 200

# 2018: All patents
echo ""
echo "============================================================"
echo "📥 Step 4: Downloading 2018 patents (ALL)"
echo "============================================================"
python3 scripts/download_patents_bigquery.py \
    --start-year 2018 \
    --end-year 2018 \
    --limit 50000 \
    --output ${DATA_DIR}/patents_2018.jsonl

echo ""
echo "🔄 Ingesting 2018 patents..."
python3 scripts/ingest_patents_gpu_optimized.py \
    ${DATA_DIR}/patents_2018.jsonl \
    --keep-existing \
    --workers 3 \
    --embedding-batch-size 200

echo ""
echo "============================================================"
echo "✅ ALL DOWNLOADS AND INGESTION COMPLETE!"
echo "============================================================"
echo ""
echo "📊 Final Status:"
python3 -c "
import asyncio
from db.qdrant_client import qdrant_client

async def check():
    info = await qdrant_client.client.get_collection('patents')
    print(f'   Total patent chunks in Qdrant: {info.points_count:,}')
    await qdrant_client.close()

asyncio.run(check())
"

