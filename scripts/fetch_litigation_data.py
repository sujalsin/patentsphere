#!/usr/bin/env python3
"""
Fetch USPTO OCE Litigation data from BigQuery.

This script queries the USPTO Office of Chief Economist litigation dataset
and extracts litigation cases related to patents in our database.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Set

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import get_settings
from config.runtime_profile import get_runtime_profile


def normalize_patent_id(patent_id: str) -> str:
    """Normalize patent ID to match our format."""
    if not patent_id:
        return ""
    # Remove whitespace and convert to uppercase
    patent_id = patent_id.strip().upper()
    # Handle common formats: US123456, US-123456-A1, etc.
    if patent_id.startswith("US") and len(patent_id) > 2:
        # Keep the format but normalize separators
        return patent_id.replace(" ", "-")
    return patent_id


def get_patent_ids_from_database(settings) -> Set[str]:
    """Get list of patent IDs from our database."""
    import psycopg
    
    pg_cfg = settings.database
    conn_str = f"postgresql://{pg_cfg.user}:{pg_cfg.password}@{pg_cfg.host}:{pg_cfg.port}/{pg_cfg.database}"
    
    try:
        with psycopg.connect(conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT patent_id FROM patent_chunks WHERE patent_id IS NOT NULL")
                patent_ids = {row[0] for row in cur.fetchall()}
                return patent_ids
    except Exception as e:
        print(f"Warning: Could not fetch patent IDs from database: {e}")
        return set()


def build_litigation_query(
    project_id: str,
    dataset_id: str,
    patent_ids: Set[str] | None = None,
    limit: int | None = None,
) -> str:
    """
    Build BigQuery SQL to fetch litigation data from USPTO OCE dataset.
    
    We join cases + party names + the patents table so every row emits an
    explicit patent_id when USPTO provides it. If no patent exists, the row
    still includes the case (patent_id NULL) so we can backfill later.
    """
    # Optional filter clause
    patent_filter = ""
    if patent_ids:
        normalized_ids = [normalize_patent_id(pid) for pid in patent_ids if pid]
        if normalized_ids:
            # Use IN clause on normalized patent numbers
            # The patents table stores patent numbers without separators (e.g. US1234567B2)
            normalized_ids = [pid.replace("-", "").replace(" ", "") for pid in normalized_ids]
            patent_values = ",".join(f"'{pid}'" for pid in normalized_ids[:1000])
            patent_filter = f"WHERE REGEXP_REPLACE(pat.patent_number, r'[- ]', '') IN ({patent_values})"
    
    limit_clause = f"LIMIT {limit}" if limit else ""
    
    query = f"""
    WITH case_parties AS (
        SELECT 
            c.case_row_id,
            c.case_number,
            c.case_name,
            c.court_name,
            c.date_filed,
            c.date_closed,
            c.settlement,
            STRING_AGG(DISTINCT CASE WHEN n.party_type = 'Plaintiff' THEN n.name END, ', ') AS plaintiff_name,
            STRING_AGG(DISTINCT CASE WHEN n.party_type = 'Defendant' THEN n.name END, ', ') AS defendant_name
        FROM `patents-public-data.{dataset_id}.cases` c
        LEFT JOIN `patents-public-data.{dataset_id}.names` n
            ON c.case_row_id = n.case_row_id
        WHERE c.date_filed IS NOT NULL
        GROUP BY c.case_row_id, c.case_number, c.case_name, c.court_name, c.date_filed, c.date_closed, c.settlement
    ),
    case_patents AS (
        SELECT
            cp.case_number,
            CONCAT(
                UPPER(COALESCE(pat.country_code, 'US')),
                '-',
                REGEXP_REPLACE(UPPER(COALESCE(pat.patent_number, '')), r'[^A-Z0-9]', '')
            ) AS normalized_patent_id
        FROM `patents-public-data.{dataset_id}.case_patents` cp
        JOIN `patents-public-data.{dataset_id}.patents` pat
            ON cp.patent_row_id = pat.patent_row_id
        {patent_filter}
    )
    SELECT 
        parties.case_number,
        parties.case_name,
        parties.court_name,
        parties.date_filed AS filing_date,
        parties.date_closed,
        CASE 
            WHEN parties.date_closed IS NOT NULL AND parties.date_closed != '' THEN 'closed'
            WHEN parties.settlement IS NOT NULL AND parties.settlement != '' THEN 'settled'
            ELSE 'active'
        END AS case_status,
        parties.settlement AS outcome,
        parties.plaintiff_name,
        parties.defendant_name,
        cp.normalized_patent_id AS patent_id
    FROM case_parties parties
    LEFT JOIN case_patents cp
        ON parties.case_number = cp.case_number
    {limit_clause}
    """
    
    return query


def fetch_litigation_data(
    client,
    query: str,
    output_path: Path,
    execute: bool = False,
) -> int:
    """Fetch litigation data from BigQuery and write to JSONL."""
    from google.cloud import bigquery
    
    job_config = bigquery.QueryJobConfig(dry_run=not execute)
    job = client.query(query, job_config=job_config)
    
    if job_config.dry_run:
        bytes_processed = job.total_bytes_processed
        cost_usd = (bytes_processed / (1024**4)) * 5
        print(f"Dry run successful.")
        print(f"  Bytes to process: {bytes_processed:,}")
        print(f"  Estimated cost: ${cost_usd:.4f}")
        return 0
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    
    print("Fetching litigation data from BigQuery...")
    with output_path.open("w", encoding="utf-8") as f:
        for row in job:
            record = dict(row)
            # Extract patent IDs from potential_patents array or case_name
            patent_ids = []
            if "potential_patents" in record and record["potential_patents"]:
                patent_ids = [normalize_patent_id(p) for p in record["potential_patents"] if p]
            elif "case_name" in record:
                # Try to extract from case name as fallback
                import re
                case_name = record.get("case_name") or ""
                if case_name and isinstance(case_name, str):
                    found_patents = re.findall(r'\b(US|EP|WO|JP|CN|KR|GB|DE|FR)[- ]?\d+[A-Z]?\d*\b', case_name, re.IGNORECASE)
                    patent_ids = [normalize_patent_id(p) for p in found_patents]
            
            # Create one record per patent ID found (or one record if no patents found)
            if patent_ids:
                for patent_id in patent_ids:
                    record_copy = record.copy()
                    record_copy["patent_id"] = patent_id
                    # Remove the array field
                    if "potential_patents" in record_copy:
                        del record_copy["potential_patents"]
                    f.write(json.dumps(record_copy, default=str) + "\n")
                    count += 1
            else:
                # No patent ID found, still save the case
                record["patent_id"] = None
                if "potential_patents" in record:
                    del record["potential_patents"]
                f.write(json.dumps(record, default=str) + "\n")
                count += 1
            
            if count % 1000 == 0:
                print(f"  Fetched {count} records...")
    
    print(f"✓ Wrote {count} litigation records to {output_path}")
    return count


def main():
    parser = argparse.ArgumentParser(description="Fetch USPTO litigation data from BigQuery")
    parser.add_argument(
        "--output",
        type=str,
        default="data/litigation_data.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of records to fetch (for testing)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute the query (otherwise dry run)",
    )
    parser.add_argument(
        "--use-db-patents",
        action="store_true",
        help="Filter to patents in our database",
    )
    args = parser.parse_args()
    
    settings = get_settings()
    profile = get_runtime_profile()
    
    if args.execute:
        if not profile.allow_bigquery_exports():
            raise RuntimeError(
                "BigQuery exports disabled. Set gcp.usage_policy.allow_bigquery_exports=true and "
                "export GCP_ALLOW_BIGQUERY_EXPORTS=true to proceed."
            )
    else:
        print("Running in DRY RUN mode. Use --execute to fetch data.")
    
    if not settings.gcp.project_id:
        print("Error: GCP_PROJECT_ID not set in config")
        sys.exit(1)
    
    if settings.gcp.credentials_path:
        creds_path = Path(settings.gcp.credentials_path)
        if creds_path.exists():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_path)
        else:
            print(f"Warning: Credentials file not found: {creds_path}")
            print("Attempting to use default credentials...")
    
    from google.cloud import bigquery
    
    # Use user's project for query execution, but reference public datasets
    # The query will reference patents-public-data.uspto_oce_litigation
    client = bigquery.Client(project=settings.gcp.project_id)
    
    # Get patent IDs if filtering
    patent_ids = None
    if args.use_db_patents:
        print("Fetching patent IDs from database...")
        patent_ids = get_patent_ids_from_database(settings)
        print(f"  Found {len(patent_ids)} unique patents in database")
    
    # Build query
    dataset_id = getattr(settings.gcp.bigquery, 'litigation_dataset', "uspto_oce_litigation")
    
    query = build_litigation_query(
        project_id="patents-public-data",
        dataset_id=dataset_id,
        patent_ids=patent_ids,
        limit=args.limit,
    )
    
    print(f"\nQuery:")
    print(query[:500] + "..." if len(query) > 500 else query)
    
    # Fetch data
    output_path = Path(args.output)
    count = fetch_litigation_data(client, query, output_path, execute=args.execute)
    
    if args.execute and count > 0:
        print(f"\n✓ Successfully fetched {count} litigation records")
        print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()

