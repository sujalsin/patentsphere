#!/bin/bash
# Helper script to set up GCP credentials

KEY_DIR="/home/sujals2144/project/key"

echo "GCP Credentials Setup"
echo "===================="
echo ""
echo "Please place your GCP service account JSON file in:"
echo "  $KEY_DIR"
echo ""
echo "The file should be named something like:"
echo "  - service-account-key.json"
echo "  - gcp-credentials.json"
echo "  - ancient-courage-478809-p0.json"
echo ""
echo "Once the file is in place, run:"
echo "  python scripts/fetch_chunk_ingest.py"
echo ""

if [ -d "$KEY_DIR" ]; then
    echo "Current contents of $KEY_DIR:"
    ls -la "$KEY_DIR"
else
    echo "Creating directory: $KEY_DIR"
    mkdir -p "$KEY_DIR"
fi
