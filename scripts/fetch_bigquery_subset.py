#!/usr/bin/env python3
"""
Fetch a limited subset of patents from Google BigQuery into a local JSONL file.

This script honors the cost guardrails defined in config/runtime_profile.py:
- Requires `resource_policy.require_manual_cloud_enable` passes
- Requires `gcp.usage_policy.allow_bigquery_exports` plus env flag `GCP_ALLOW_BIGQUERY_EXPORTS=true`

Usage:
  python scripts/fetch_bigquery_subset.py \
      --limit 10000 \
      --output data/patents_10k.jsonl \
      --execute

By default it runs in DRY-RUN mode (no charges). Pass --execute to actually export.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.runtime_profile import get_runtime_profile  # type: ignore  # noqa
from config.settings import get_settings  # type: ignore  # noqa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch patent subset from BigQuery.")
    parser.add_argument("--limit", type=int, default=10000, help="Number of rows to fetch.")
    parser.add_argument(
        "--output",
        type=str,
        default="data/patents_10k.jsonl",
        help="Destination JSONL file.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run the query (otherwise DRY RUN). Requires cloud guard flag.",
    )
    parser.add_argument(
        "--fields",
        type=str,
        default="publication_number,title_localized,abstract_localized,claims_localized,cpc,citation,application_number,filing_date",
        help="Comma-separated columns to select. Script handles array extraction for *_localized fields.",
    )
    return parser.parse_args()


def ensure_gcp_access(profile) -> None:
    if not profile.allow_bigquery_exports():
        raise RuntimeError(
            "BigQuery exports disabled. Set gcp.usage_policy.allow_bigquery_exports=true and "
            "export GCP_ALLOW_BIGQUERY_EXPORTS=true to proceed."
        )


def build_query(dataset: str, table: str, fields: str, limit: int) -> str:
    # Construct full table reference: dataset.table
    qualified = f"`{dataset}.{table}`"
    
    # Handle fields - if they're simple, use as-is; if they need extraction, handle specially
    field_list = [f.strip() for f in fields.split(",")]
    
    # Build SELECT clause with proper handling for array fields
    select_parts = []
    for field in field_list:
        if field in ["title_localized", "abstract_localized", "claims_localized"]:
            # Extract first element from array, or empty string if array is empty
            select_parts.append(f"IF(ARRAY_LENGTH({field}) > 0, {field}[OFFSET(0)].text, '') AS {field.replace('_localized', '')}")
        elif field == "cpc":
            # CPC is an array, convert to JSON string for storage
            select_parts.append(f"TO_JSON_STRING({field}) AS cpc_codes")
        elif field == "citation":
            # Citations is an array, convert to JSON string
            select_parts.append(f"TO_JSON_STRING({field}) AS citations")
        else:
            # Use field as-is
            select_parts.append(field)
    
    columns = ", ".join(select_parts)
    return f"SELECT {columns} FROM {qualified} LIMIT {limit}"


def main() -> None:
    args = parse_args()
    settings = get_settings()
    profile = get_runtime_profile()

    if args.execute:
        ensure_gcp_access(profile)
    else:
        print("Running in DRY RUN mode (no data exported). Use --execute to run.")

    if settings.gcp.credentials_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.gcp.credentials_path

    from google.cloud import bigquery

    client = bigquery.Client(project=settings.gcp.project_id or None)
    query = build_query(
        dataset=settings.gcp.bigquery.dataset_id,
        table=settings.gcp.bigquery.table_id,
        fields=args.fields,
        limit=args.limit,
    )

    job_config = bigquery.QueryJobConfig(dry_run=not args.execute)
    job = client.query(query, job_config=job_config)

    if job_config.dry_run:
        print(f"Dry run successful. Estimated bytes processed: {job.total_bytes_processed}")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for row in job:
            record = dict(row)
            f.write(json.dumps(record, default=str) + "\n")

    print(f"Wrote {args.limit} rows to {output_path}")


if __name__ == "__main__":
    main()

