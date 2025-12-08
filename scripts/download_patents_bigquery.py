"""Download patents from Google BigQuery for years 2015-2018.

This script queries the Google Patents Public Datasets on BigQuery
and downloads patent data for the specified date range.
"""
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from google.cloud import bigquery
    from google.oauth2 import service_account
except ImportError:
    print("❌ Error: google-cloud-bigquery not installed")
    print("   Install with: pip install google-cloud-bigquery")
    sys.exit(1)


def download_patents_bigquery(
    start_year: int = 2015,
    end_year: int = 2018,
    output_file: str = "data/patents_bigquery_2015_2018.jsonl",
    limit: Optional[int] = None,
    credentials_path: Optional[str] = None,
):
    """Download patents from BigQuery for specified years.
    
    Args:
        start_year: Start year (inclusive)
        end_year: End year (inclusive)
        output_file: Output JSONL file path
        limit: Maximum number of patents to download (None for all)
        credentials_path: Path to Google Cloud credentials JSON file
    """
    print("="*60)
    print("📥 Downloading Patents from Google BigQuery")
    print("="*60)
    print(f"   Date Range: {start_year}-{end_year}")
    print(f"   Output: {output_file}")
    if limit:
        print(f"   Limit: {limit:,} patents")
    print("="*60)
    
    # Initialize BigQuery client
    project_id = "ancient-courage-478809-p0"  # Default project ID
    
    if credentials_path and os.path.exists(credentials_path):
        print(f"\n🔑 Using credentials from: {credentials_path}")
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=["https://www.googleapis.com/auth/bigquery"]
        )
        client = bigquery.Client(credentials=credentials, project=project_id)
    else:
        print(f"\n🔑 Using default Google Cloud credentials")
        print(f"   Project ID: {project_id}")
        print("   (Set GOOGLE_APPLICATION_CREDENTIALS or provide credentials_path)")
        client = bigquery.Client(project=project_id)
    
    # BigQuery query for patents
    # Using the public patents dataset: patents-public-data.patents.publications
    # Note: Fields are REPEATED RECORDs - need to UNNEST and extract text
    # publication_date is stored as INT64 (YYYYMMDD format), need to convert to DATE
    # Handle invalid dates (0 or NULL) by checking length and valid date range
    # Note: ORDER BY is removed when limit is specified to avoid memory issues
    query = f"""
    SELECT
        publication_number,
        (SELECT text FROM UNNEST(title_localized) WHERE language = 'en' LIMIT 1) AS title,
        (SELECT text FROM UNNEST(abstract_localized) WHERE language = 'en' LIMIT 1) AS abstract,
        (SELECT text FROM UNNEST(claims_localized) WHERE language = 'en' LIMIT 1) AS claims,
        (SELECT text FROM UNNEST(description_localized) WHERE language = 'en' LIMIT 1) AS description,
        filing_date,
        publication_date,
        ARRAY(SELECT code FROM UNNEST(cpc)) AS cpc_codes,
        ARRAY(SELECT name FROM UNNEST(inventor_harmonized)) AS inventor_name,
        ARRAY(SELECT name FROM UNNEST(assignee_harmonized)) AS assignee_name
    FROM
        `patents-public-data.patents.publications`
    WHERE
        publication_date IS NOT NULL
        AND publication_date > 0
        AND LENGTH(CAST(publication_date AS STRING)) = 8
        AND EXTRACT(YEAR FROM PARSE_DATE('%Y%m%d', CAST(publication_date AS STRING))) >= {start_year}
        AND EXTRACT(YEAR FROM PARSE_DATE('%Y%m%d', CAST(publication_date AS STRING))) <= {end_year}
        AND publication_number IS NOT NULL
        AND ARRAY_LENGTH(title_localized) > 0
    """
    
    # Skip ORDER BY to avoid memory issues with large datasets
    # Results will be processed in whatever order BigQuery returns them
    # If ordering is needed, it can be done post-download or in smaller batches
    
    if limit:
        query += f"\n    LIMIT {limit}"
    
    print(f"\n🔍 Executing BigQuery query...")
    print(f"   This may take a few minutes depending on data size...")
    
    try:
        query_job = client.query(query)
        results = query_job.result()
        
        print(f"✅ Query completed")
        print(f"\n📝 Writing patents to {output_file}...")
        
        # Create output directory if it doesn't exist
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        count = 0
        with open(output_file, "w", encoding="utf-8") as f:
            for row in results:
                # Convert row to dictionary
                # Handle arrays: cpc_codes, inventor_name, assignee_name are arrays from BigQuery
                inventor_names = row.inventor_name if row.inventor_name else []
                assignee_names = row.assignee_name if row.assignee_name else []
                
                patent_data = {
                    "publication_number": row.publication_number,
                    "title": row.title or "",
                    "abstract": row.abstract or "",
                    "claims": row.claims or "",
                    "description": row.description or "",
                    "filing_date": str(row.filing_date) if row.filing_date else None,
                    "publication_date": str(row.publication_date) if row.publication_date else None,
                    "cpc_codes": json.dumps(row.cpc_codes) if row.cpc_codes else "[]",
                    "inventor_name": ", ".join(inventor_names) if inventor_names else "",
                    "assignee_name": ", ".join(assignee_names) if assignee_names else "",
                }
                
                f.write(json.dumps(patent_data, ensure_ascii=False) + "\n")
                count += 1
                
                if count % 1000 == 0:
                    print(f"   Downloaded {count:,} patents...", end='\r')
        
        print(f"\n✅ Download complete!")
        print(f"   Total patents: {count:,}")
        print(f"   Saved to: {output_file}")
        print(f"   File size: {os.path.getsize(output_file) / 1024 / 1024:.1f} MB")
        
    except Exception as e:
        print(f"\n❌ Error downloading patents: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure you have Google Cloud credentials set up")
        print("2. Check that you have access to BigQuery")
        print("3. Verify the project has BigQuery API enabled")
        print("4. For public data, you may need to set up billing (free tier available)")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download patents from BigQuery")
    parser.add_argument("--start-year", type=int, default=2015, help="Start year (default: 2015)")
    parser.add_argument("--end-year", type=int, default=2018, help="End year (default: 2018)")
    parser.add_argument("--output", type=str, default="data/patents_bigquery_2015_2018.jsonl", 
                       help="Output file path")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of patents (default: all)")
    parser.add_argument("--credentials", type=str, default=None, 
                       help="Path to Google Cloud credentials JSON file")
    
    args = parser.parse_args()
    
    download_patents_bigquery(
        start_year=args.start_year,
        end_year=args.end_year,
        output_file=args.output,
        limit=args.limit,
        credentials_path=args.credentials,
    )

