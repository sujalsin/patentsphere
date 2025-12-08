#!/bin/bash
# Setup BigQuery authentication and download patents

set -e

PROJECT_ID="ancient-courage-478809-p0"

echo "============================================================"
echo "🔑 BigQuery Authentication Setup"
echo "============================================================"
echo ""

# Check if already authenticated
echo "Checking authentication..."
if python3 -c "from google.cloud import bigquery; client = bigquery.Client(project='$PROJECT_ID'); client.query('SELECT 1').result()" 2>/dev/null; then
    echo "✅ Already authenticated!"
else
    echo "❌ Not authenticated. Setting up..."
    echo ""
    echo "You need to authenticate with your Google account."
    echo "This will open a browser window."
    echo ""
    read -p "Press Enter to continue with authentication..."
    
    gcloud auth application-default login --project=$PROJECT_ID
    
    echo ""
    echo "✅ Authentication complete!"
fi

echo ""
echo "============================================================"
echo "📥 Starting Download and Ingestion"
echo "============================================================"
echo ""

# Run the download and ingestion script
bash scripts/download_and_ingest_2015_2018.sh

