"""Ingest litigation data from JSONL files into PostgreSQL."""
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from db.postgres_client import postgres_client


async def ingest_litigation(jsonl_path: str, source: str, batch_size: int = 100):
    """Ingest litigation cases from JSONL file."""
    print(f"Starting litigation ingestion from {jsonl_path} (source: {source})...")
    
    await postgres_client.connect()
    
    batch = []
    total_processed = 0
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                case_number = data.get("case_number", "")
                if not case_number:
                    continue
                
                # Handle patent_id - could be null or string
                patent_id = data.get("patent_id")
                if patent_id and patent_id.strip() and patent_id != "null" and patent_id != "US-NA":
                    patent_id = patent_id.strip()
                else:
                    patent_id = None
                
                batch.append({
                    "case_number": case_number,
                    "case_name": data.get("case_name"),
                    "court_name": data.get("court_name"),
                    "filing_date": data.get("filing_date"),
                    "case_status": data.get("case_status"),
                    "plaintiff_name": data.get("plaintiff_name"),
                    "defendant_name": data.get("defendant_name"),
                    "patent_id": patent_id,
                    "outcome": data.get("outcome"),
                    "source": source,
                })
                
                if len(batch) >= batch_size:
                    await insert_batch_postgres(batch)
                    total_processed += len(batch)
                    print(f"Processed {total_processed} cases...")
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
    
    print(f"Ingestion complete! Processed {total_processed} cases from {source}.")
    
    await postgres_client.close()


async def insert_batch_postgres(batch: list):
    """Insert batch of litigation cases into PostgreSQL."""
    query = """
        INSERT INTO litigation_cases 
        (case_number, case_name, court_name, filing_date, case_status, 
         plaintiff_name, defendant_name, patent_id, outcome, source)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        ON CONFLICT (case_number, patent_id) 
        DO UPDATE SET
            case_name = EXCLUDED.case_name,
            court_name = EXCLUDED.court_name,
            filing_date = EXCLUDED.filing_date,
            case_status = EXCLUDED.case_status,
            plaintiff_name = EXCLUDED.plaintiff_name,
            defendant_name = EXCLUDED.defendant_name,
            outcome = EXCLUDED.outcome,
            source = EXCLUDED.source,
            updated_at = CURRENT_TIMESTAMP
    """
    
    async with postgres_client.pool.acquire() as conn:
        for item in batch:
            await conn.execute(
                query,
                item["case_number"],
                item["case_name"],
                item["court_name"],
                item["filing_date"],
                item["case_status"],
                item["plaintiff_name"],
                item["defendant_name"],
                item["patent_id"],
                item["outcome"],
                item["source"],
            )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python ingest_litigation.py <jsonl_path> <source>")
        print("Example: python ingest_litigation.py data/stanford_litigation.jsonl stanford_npe")
        sys.exit(1)
    
    jsonl_path = sys.argv[1]
    source = sys.argv[2]
    asyncio.run(ingest_litigation(jsonl_path, source))


