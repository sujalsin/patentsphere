#!/usr/bin/env python3
"""
Master script to fetch patent data from BigQuery and process litigation data,
then chunk and ingest everything into Postgres/Qdrant.

This script orchestrates the full data pipeline:
1. Fetch USPTO patents from BigQuery
2. Process Stanford NPE litigation CSV
3. Chunk patents using hybrid strategy
4. Generate embeddings
5. Ingest into Postgres and Qdrant

Usage:
    # Dry run (check costs)
    python scripts/fetch_and_ingest_all.py --dry-run
    
    # Execute full pipeline
    python scripts/fetch_and_ingest_all.py --execute --patent-limit 10000
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_command(cmd: list, description: str, check: bool = True) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=check, capture_output=False)
        if result.returncode == 0:
            print(f"✓ {description} completed successfully")
            return True
        else:
            print(f"✗ {description} failed with return code {result.returncode}")
            return False
    except Exception as e:
        print(f"✗ {description} failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Fetch and ingest all patent data")
    parser.add_argument(
        "--patent-limit",
        type=int,
        default=10000,
        help="Number of patents to fetch from BigQuery",
    )
    parser.add_argument(
        "--patents-output",
        type=str,
        default="data/patents_bigquery.jsonl",
        help="Output path for patent JSONL",
    )
    parser.add_argument(
        "--stanford-csv",
        type=str,
        default="data/cases-2025-12-05PST01-24-57.csv",
        help="Path to Stanford NPE litigation CSV",
    )
    parser.add_argument(
        "--stanford-output",
        type=str,
        default="data/stanford_litigation.jsonl",
        help="Output path for Stanford litigation JSONL",
    )
    parser.add_argument(
        "--uspto-zip",
        type=str,
        default="data/csv.zip",
        help="Path to USPTO litigation CSV zip file",
    )
    parser.add_argument(
        "--uspto-output",
        type=str,
        default="data/uspto_litigation.jsonl",
        help="Output path for USPTO litigation JSONL",
    )
    parser.add_argument(
        "--chunks-output-dir",
        type=str,
        default="data/processed",
        help="Output directory for chunks and embeddings",
    )
    parser.add_argument(
        "--generate-embeddings",
        action="store_true",
        help="Generate embeddings (requires torch and sentence-transformers)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute queries (otherwise dry run)",
    )
    parser.add_argument(
        "--skip-bigquery",
        action="store_true",
        help="Skip BigQuery fetch (use existing patent file)",
    )
    parser.add_argument(
        "--skip-litigation",
        action="store_true",
        help="Skip litigation processing",
    )
    parser.add_argument(
        "--skip-chunking",
        action="store_true",
        help="Skip chunking (use existing chunks)",
    )
    parser.add_argument(
        "--skip-ingestion",
        action="store_true",
        help="Skip database ingestion",
    )
    args = parser.parse_args()
    
    dry_run = not args.execute
    
    if dry_run:
        print("\n" + "="*60)
        print("DRY RUN MODE - No data will be fetched or written")
        print("="*60)
    
    # Step 1: Fetch patents from BigQuery
    if not args.skip_bigquery:
        if dry_run:
            print("\n[DRY RUN] Would fetch patents from BigQuery...")
        else:
            success = run_command(
                [
                    sys.executable,
                    "scripts/fetch_bigquery_subset.py",
                    "--limit", str(args.patent_limit),
                    "--output", args.patents_output,
                    "--execute",
                ],
                "Fetch patents from BigQuery",
                check=False,
            )
            if not success:
                print("Warning: BigQuery fetch failed. Continuing with existing data if available...")
    
    # Step 2: Process Stanford litigation CSV
    if not args.skip_litigation:
        stanford_csv = Path(args.stanford_csv)
        if stanford_csv.exists():
            if dry_run:
                print("\n[DRY RUN] Would process Stanford litigation CSV...")
            else:
                run_command(
                    [
                        sys.executable,
                        "scripts/process_stanford_litigation.py",
                        "--input", str(stanford_csv),
                        "--output", args.stanford_output,
                    ],
                    "Process Stanford litigation CSV",
                )
        else:
            print(f"\nWarning: Stanford CSV not found: {stanford_csv}")
            print("Skipping litigation processing...")
    
    # Step 3: Chunk patents
    if not args.skip_chunking:
        patents_file = Path(args.patents_output)
        if patents_file.exists():
            if dry_run:
                print("\n[DRY RUN] Would chunk patents...")
            else:
                chunk_cmd = [
                    sys.executable,
                    "scripts/process_patents.py",
                    "--input", str(patents_file),
                    "--output-dir", args.chunks_output_dir,
                    "--chunking-strategy", "hybrid",
                    "--max-patents", str(args.patent_limit),
                ]
                
                if args.generate_embeddings:
                    chunk_cmd.append("--generate-embeddings")
                
                run_command(
                    chunk_cmd,
                    "Chunk patents and generate embeddings",
                )
        else:
            print(f"\nWarning: Patent file not found: {patents_file}")
            print("Skipping chunking...")
    
    # Step 4: Ingest into databases
    if not args.skip_ingestion:
        chunks_file = Path(args.chunks_output_dir) / "chunks.jsonl"
        embeddings_file = Path(args.chunks_output_dir) / "embeddings.pt"
        
        if chunks_file.exists():
            if dry_run:
                print("\n[DRY RUN] Would ingest chunks into Postgres/Qdrant...")
            else:
                ingest_cmd = [
                    sys.executable,
                    "scripts/ingest_local.py",
                    "--chunks", str(chunks_file),
                    "--embeddings", str(embeddings_file),
                ]
                
                # Add litigation files if available
                litigation_files = []
                stanford_litigation = Path(args.stanford_output)
                if stanford_litigation.exists():
                    litigation_files.append(str(stanford_litigation))
                
                uspto_litigation = Path(args.uspto_output)
                if uspto_litigation.exists():
                    litigation_files.append(str(uspto_litigation))
                
                # Note: ingest_local.py currently only accepts one --litigation file
                # We'll need to merge them or update the script to accept multiple
                if litigation_files:
                    # Use the first available file (or merge them)
                    ingest_cmd.extend(["--litigation", litigation_files[0]])
                    if len(litigation_files) > 1:
                        print(f"\nNote: Multiple litigation files found. Using: {litigation_files[0]}")
                        print(f"      Consider merging {litigation_files} before ingestion.")
                
                run_command(
                    ingest_cmd,
                    "Ingest chunks and embeddings into databases",
                )
        else:
            print(f"\nWarning: Chunks file not found: {chunks_file}")
            print("Skipping ingestion...")
    
    print("\n" + "="*60)
    if dry_run:
        print("DRY RUN complete. Use --execute to run the pipeline.")
    else:
        print("Pipeline complete!")
    print("="*60)


if __name__ == "__main__":
    main()
