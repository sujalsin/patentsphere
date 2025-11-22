#!/usr/bin/env python3
"""
Research USPTO OCE Litigation dataset schema in BigQuery.

This script explores the structure of the USPTO Office of Chief Economist
litigation dataset to understand available tables and fields.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import get_settings
from config.runtime_profile import get_runtime_profile


def list_datasets(client, project_id: str):
    """List all datasets in the project."""
    print(f"\n=== Datasets in project '{project_id}' ===")
    datasets = list(client.list_datasets())
    for dataset in datasets:
        print(f"  - {dataset.dataset_id}")
    return datasets


def list_tables(client, dataset_id: str):
    """List all tables in a dataset."""
    print(f"\n=== Tables in dataset '{dataset_id}' ===")
    try:
        dataset_ref = client.dataset(dataset_id)
        tables = list(client.list_tables(dataset_ref))
        for table in tables:
            print(f"  - {table.table_id}")
        return tables
    except Exception as e:
        print(f"  Error: {e}")
        return []


def get_table_schema(client, dataset_id: str, table_id: str):
    """Get schema for a specific table."""
    print(f"\n=== Schema for '{dataset_id}.{table_id}' ===")
    try:
        table_ref = client.dataset(dataset_id).table(table_id)
        table = client.get_table(table_ref)
        
        print(f"  Rows: {table.num_rows:,}")
        print(f"  Size: {table.num_bytes / (1024**3):.2f} GB")
        print(f"\n  Fields:")
        for field in table.schema:
            field_type = field.field_type
            if field.mode == "REPEATED":
                field_type = f"ARRAY<{field_type}>"
            print(f"    - {field.name}: {field_type} ({field.mode})")
            if field.description:
                print(f"      Description: {field.description}")
        
        return table.schema
    except Exception as e:
        print(f"  Error: {e}")
        return None


def estimate_query_cost(client, query: str):
    """Estimate cost of a query."""
    print(f"\n=== Query Cost Estimation ===")
    try:
        job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        job = client.query(query, job_config=job_config)
        
        bytes_processed = job.total_bytes_processed
        cost_usd = (bytes_processed / (1024**4)) * 5  # $5 per TB
        
        print(f"  Bytes to process: {bytes_processed:,}")
        print(f"  Estimated cost: ${cost_usd:.4f}")
        return bytes_processed, cost_usd
    except Exception as e:
        print(f"  Error: {e}")
        return None, None


def main():
    parser = argparse.ArgumentParser(description="Research USPTO OCE Litigation dataset schema")
    parser.add_argument(
        "--dataset",
        type=str,
        default="uspto_oce_litigation",
        help="BigQuery dataset ID to explore",
    )
    parser.add_argument(
        "--table",
        type=str,
        default=None,
        help="Specific table to examine (if not provided, lists all tables)",
    )
    parser.add_argument(
        "--sample-query",
        action="store_true",
        help="Show sample query for fetching litigation data",
    )
    args = parser.parse_args()
    
    settings = get_settings()
    profile = get_runtime_profile()
    
    if not settings.gcp.project_id:
        print("Error: GCP_PROJECT_ID not set in config")
        sys.exit(1)
    
    if settings.gcp.credentials_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.gcp.credentials_path
    
    from google.cloud import bigquery
    
    # Use user's project but access public datasets
    # For listing, we can use the public project
    try:
        client = bigquery.Client(project="patents-public-data")
    except:
        # Fallback to user's project
        client = bigquery.Client(project=settings.gcp.project_id)
    
    print(f"Project: patents-public-data")
    print(f"Dataset: {args.dataset}")
    
    # List tables in the dataset
    tables = list_tables(client, args.dataset)
    
    if not tables:
        print(f"\nDataset '{args.dataset}' not found or empty.")
        print("Available public datasets:")
        print("  - Try: patents-public-data")
        print("  - Or check: https://console.cloud.google.com/bigquery")
        return
    
    # If specific table requested, show its schema
    if args.table:
        schema = get_table_schema(client, args.dataset, args.table)
    else:
        # Show schema for all tables
        for table in tables:
            get_table_schema(client, args.dataset, table.table_id)
    
    # Show sample query if requested
    if args.sample_query and tables:
        sample_table = args.table or tables[0].table_id
        query = f"""
        SELECT *
        FROM `patents-public-data.{args.dataset}.{sample_table}`
        LIMIT 10
        """
        print(f"\n=== Sample Query ===")
        print(query)
        estimate_query_cost(client, query)


if __name__ == "__main__":
    main()

