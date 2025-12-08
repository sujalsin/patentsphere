#!/bin/bash
# Parallel ingestion script for all data sources

set -e

echo "🚀 Starting Parallel Ingestion Pipeline"
echo "========================================"
echo ""

# Configuration
PATENT_FILE="data/patents_bigquery.jsonl"
STANFORD_FILE="data/stanford_litigation.jsonl"
USPTO_FILE="data/uspto_litigation.jsonl"

# Patent ingestion (GPU-optimized)
echo "📦 Step 1: Ingesting Patents (GPU-Optimized)"
echo "--------------------------------------------"
python scripts/ingest_patents_gpu_optimized.py "$PATENT_FILE"

echo ""
echo "✅ Patent ingestion complete!"
echo ""

# Litigation ingestion - Stanford (multiprocessing)
echo "⚖️  Step 2: Ingesting Stanford Litigation (Multiprocessing)"
echo "-----------------------------------------------------------"
python scripts/ingest_litigation_multiprocessing.py "$STANFORD_FILE" stanford

echo ""
echo "✅ Stanford litigation ingestion complete!"
echo ""

# Litigation ingestion - USPTO (multiprocessing)
echo "⚖️  Step 3: Ingesting USPTO Litigation (Multiprocessing)"
echo "---------------------------------------------------------"
python scripts/ingest_litigation_multiprocessing.py "$USPTO_FILE" uspto

echo ""
echo "✅ USPTO litigation ingestion complete!"
echo ""

echo "========================================"
echo "🎉 ALL INGESTION COMPLETE!"
echo "========================================"

