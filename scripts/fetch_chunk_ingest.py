#!/usr/bin/env python3
"""
Fetch patents from BigQuery, chunk them, and ingest into Qdrant.

This script:
1. Finds GCP credentials in ../key/ directory
2. Fetches patents from BigQuery
3. Chunks patents using hybrid strategy
4. Generates embeddings
5. Ingests into Postgres and Qdrant
"""

import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

KEY_DIR = Path("/home/sujals2144/project/key")


def find_credentials() -> Path:
    """Find GCP credentials JSON file in key directory."""
    if not KEY_DIR.exists():
        print(f"Error: Key directory not found: {KEY_DIR}")
        print("Please create the directory and place your GCP service account JSON file there.")
        sys.exit(1)
    
    json_files = list(KEY_DIR.glob("*.json"))
    if not json_files:
        print(f"Error: No JSON credentials file found in {KEY_DIR}")
        print("Please place your GCP service account JSON file in that directory.")
        sys.exit(1)
    
    cred_file = json_files[0]
    print(f"Found credentials: {cred_file}")
    return cred_file


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, cwd=ROOT)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with return code {e.returncode}")
        return False
    except Exception as e:
        print(f"✗ {description} failed: {e}")
        return False


def main():
    print("\n" + "="*60)
    print("PatentSphere Data Pipeline")
    print("="*60)
    
    # Find credentials
    cred_file = find_credentials()
    
    # Set environment variables
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(cred_file)
    os.environ["GCP_ALLOW_BIGQUERY_EXPORTS"] = "true"
    os.environ["GCP_PROJECT_ID"] = "ancient-courage-478809-p0"
    
    print(f"\nEnvironment configured:")
    print(f"  GOOGLE_APPLICATION_CREDENTIALS: {cred_file}")
    print(f"  GCP_PROJECT_ID: ancient-courage-478809-p0")
    
    # Step 1: Fetch patents
    success = run_command(
        [
            sys.executable,
            "scripts/fetch_bigquery_subset.py",
            "--limit", "10000",
            "--output", "data/patents_bigquery.jsonl",
            "--execute",
        ],
        "Fetch patents from BigQuery",
    )
    
    if not success:
        print("\nFailed to fetch patents. Exiting.")
        sys.exit(1)
    
    # Check if file was created
    patents_file = ROOT / "data/patents_bigquery.jsonl"
    if not patents_file.exists():
        print("\nError: Patent file was not created")
        sys.exit(1)
    
    patent_count = sum(1 for _ in patents_file.open()) if patents_file.exists() else 0
    print(f"\n✓ Fetched {patent_count} patents")
    
    # Step 2: Chunk patents
    success = run_command(
        [
            sys.executable,
            "scripts/process_patents.py",
            "--input", "data/patents_bigquery.jsonl",
            "--output-dir", "data/processed",
            "--chunking-strategy", "hybrid",
            "--max-patents", "10000",
            "--generate-embeddings",
        ],
        "Chunk patents and generate embeddings",
    )
    
    if not success:
        print("\nFailed to chunk patents. Exiting.")
        sys.exit(1)
    
    # Check if chunks were created
    chunks_file = ROOT / "data/processed/chunks.jsonl"
    embeddings_file = ROOT / "data/processed/embeddings.pt"
    
    if not chunks_file.exists():
        print("\nError: Chunks file was not created")
        sys.exit(1)
    
    if not embeddings_file.exists():
        print("\nWarning: Embeddings file was not created")
        print("Continuing without embeddings...")
    
    chunk_count = sum(1 for _ in chunks_file.open()) if chunks_file.exists() else 0
    print(f"\n✓ Generated {chunk_count} chunks")
    
    # Step 3: Ingest into databases
    ingest_cmd = [
        sys.executable,
        "scripts/ingest_local.py",
        "--chunks", "data/processed/chunks.jsonl",
    ]
    
    if embeddings_file.exists():
        ingest_cmd.extend(["--embeddings", "data/processed/embeddings.pt"])
    
    success = run_command(
        ingest_cmd,
        "Ingest chunks into Postgres and Qdrant",
    )
    
    if not success:
        print("\nFailed to ingest data. Exiting.")
        sys.exit(1)
    
    # Summary
    print("\n" + "="*60)
    print("Pipeline Complete!")
    print("="*60)
    print(f"Patents fetched: {patent_count}")
    print(f"Chunks generated: {chunk_count}")
    print("\nData is now available in:")
    print("  - Postgres: patent_chunks table")
    if embeddings_file.exists():
        print("  - Qdrant: patents collection (with embeddings)")
    print("="*60)


if __name__ == "__main__":
    main()
