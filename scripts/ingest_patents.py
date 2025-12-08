"""Ingest patent data from JSONL into Qdrant and PostgreSQL."""
import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from db.qdrant_client import qdrant_client
from db.postgres_client import postgres_client
from config import settings


def parse_filing_date(filing_date: int) -> str:
    """Convert filing date integer to YYYY-MM-DD format."""
    if not filing_date or filing_date == 0:
        return None
    date_str = str(filing_date)
    if len(date_str) == 8:
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    return None


def generate_patent_url(publication_number: str) -> str:
    """Generate Google Patents URL from publication number."""
    # Clean up publication number (remove dashes)
    clean_number = publication_number.replace("-", "")
    return f"https://patents.google.com/patent/{clean_number}"


async def ingest_patents(jsonl_path: str, batch_size: int = 100):
    """Ingest patents from JSONL file."""
    print(f"Starting patent ingestion from {jsonl_path}...")
    
    # Initialize clients
    # snowflake-arctic-embed-m produces 384-dimensional embeddings
    await qdrant_client.create_collection_if_not_exists(vector_size=384)
    await postgres_client.connect()
    
    batch = []
    total_processed = 0
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                publication_number = data.get("publication_number", "")
                if not publication_number:
                    continue
                
                title = data.get("title", "")
                abstract = data.get("abstract", "")
                cpc_codes = data.get("cpc_codes", "[]")
                
                # Parse filing date
                filing_date = parse_filing_date(data.get("filing_date"))
                
                # Generate URL
                url = generate_patent_url(publication_number)
                
                # Prepare metadata for Qdrant
                metadata = {
                    "publication_number": publication_number,
                    "filing_date": filing_date,
                }
                
                # Upsert to Qdrant
                await qdrant_client.upsert_patent(
                    patent_id=publication_number,
                    title=title,
                    abstract=abstract,
                    metadata=metadata,
                )
                
                # Prepare for PostgreSQL batch insert
                batch.append({
                    "publication_number": publication_number,
                    "title": title,
                    "publication_date": filing_date,
                    "cpc_codes": cpc_codes,
                    "url": url,
                })
                
                if len(batch) >= batch_size:
                    await insert_batch_postgres(batch)
                    total_processed += len(batch)
                    print(f"Processed {total_processed} patents...")
                    batch = []
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    # Insert remaining batch
    if batch:
        await insert_batch_postgres(batch)
        total_processed += len(batch)
    
    print(f"Ingestion complete! Processed {total_processed} patents.")
    
    # Close connections
    await postgres_client.close()
    await qdrant_client.close()


async def insert_batch_postgres(batch: list):
    """Insert batch of patents into PostgreSQL."""
    query = """
        INSERT INTO patents_metadata 
        (publication_number, title, publication_date, cpc_codes, url)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (publication_number) 
        DO UPDATE SET
            title = EXCLUDED.title,
            publication_date = EXCLUDED.publication_date,
            cpc_codes = EXCLUDED.cpc_codes,
            url = EXCLUDED.url,
            updated_at = CURRENT_TIMESTAMP
    """
    
    async with postgres_client.pool.acquire() as conn:
        for item in batch:
            await conn.execute(
                query,
                item["publication_number"],
                item["title"],
                item["publication_date"],
                item["cpc_codes"],
                item["url"],
            )


if __name__ == "__main__":
    jsonl_path = sys.argv[1] if len(sys.argv) > 1 else "data/patents_bigquery.jsonl"
    asyncio.run(ingest_patents(jsonl_path))

