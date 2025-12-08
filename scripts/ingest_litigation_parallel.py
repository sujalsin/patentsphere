"""Parallel litigation ingestion with progress logging."""
import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from db.postgres_client import postgres_client


def parse_filing_date(filing_date_str: str) -> datetime.date:
    """Parse filing date from string."""
    if not filing_date_str:
        return None
    try:
        # Try YYYY-MM-DD format
        if len(filing_date_str) == 10 and '-' in filing_date_str:
            return datetime.strptime(filing_date_str, "%Y-%m-%d").date()
        # Try YYYYMMDD format
        elif len(filing_date_str) == 8:
            return datetime.strptime(filing_date_str, "%Y%m%d").date()
    except:
        pass
    return None


async def insert_litigation_batch(cases_batch: List[Dict[str, Any]], source: str):
    """Insert a batch of litigation cases into PostgreSQL."""
    if not postgres_client.pool:
        return 0
    
    try:
        async with postgres_client.pool.acquire() as conn:
            inserted = 0
            for case_data in cases_batch:
                case_number = case_data.get("case_number")
                patent_id = case_data.get("patent_id")
                
                if not case_number or not patent_id:
                    continue
                
                filing_date = parse_filing_date(case_data.get("filing_date"))
                
                try:
                    await conn.execute(
                        """
                        INSERT INTO litigation_cases (
                            case_number, case_name, court_name, filing_date, case_status,
                            plaintiff_name, defendant_name, outcome, patent_id, source
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                        ON CONFLICT (case_number, patent_id) DO UPDATE SET
                            case_name = EXCLUDED.case_name,
                            court_name = EXCLUDED.court_name,
                            filing_date = EXCLUDED.filing_date,
                            case_status = EXCLUDED.case_status,
                            plaintiff_name = EXCLUDED.plaintiff_name,
                            defendant_name = EXCLUDED.defendant_name,
                            outcome = EXCLUDED.outcome,
                            source = EXCLUDED.source;
                        """,
                        case_number,
                        case_data.get("case_name"),
                        case_data.get("court_name"),
                        filing_date,
                        case_data.get("case_status"),
                        case_data.get("plaintiff_name"),
                        case_data.get("defendant_name"),
                        case_data.get("outcome"),
                        patent_id,
                        source,
                    )
                    inserted += 1
                except Exception as e:
                    print(f"Warning: Could not insert case {case_number}: {e}")
                    continue
            return inserted
    except Exception as e:
        print(f"Error inserting batch: {e}")
        return 0


async def ingest_litigation_parallel(jsonl_path: str, source: str, batch_size: int = 1000, parallel_batches: int = 5):
    """Ingest litigation data with parallel processing and progress logging.
    
    Args:
        jsonl_path: Path to JSONL file
        source: Source name (e.g., "stanford", "uspto")
        batch_size: Number of cases to process before logging
        parallel_batches: Number of batches to process in parallel
    """
    print(f"\n{'='*60}")
    print(f"Starting PARALLEL litigation ingestion: {source.upper()}")
    print(f"File: {jsonl_path}")
    print(f"Batch size: {batch_size}, Parallel batches: {parallel_batches}")
    print(f"{'='*60}\n")
    
    # Connect to PostgreSQL
    try:
        await postgres_client.connect()
        print("✅ Connected to PostgreSQL")
    except Exception as e:
        print(f"❌ Could not connect to PostgreSQL: {e}")
        return
    
    # Load all cases into memory
    print("📖 Loading litigation cases from file...")
    all_cases = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                # Only require case_number - patent_id can be None (many cases don't have patent_id)
                if data.get("case_number"):
                    all_cases.append(data)
            except json.JSONDecodeError:
                continue
    
    total_cases = len(all_cases)
    print(f"✅ Loaded {total_cases:,} cases")
    
    if total_cases == 0:
        print("⚠️  No cases to ingest")
        return
    
    # Process in parallel batches
    total_inserted = 0
    start_time = time.time()
    batch_list = []
    
    # Create batches
    for i in range(0, total_cases, batch_size):
        batch = all_cases[i:min(i + batch_size, total_cases)]
        batch_list.append(batch)
    
    # Process batches in parallel
    for batch_idx in range(0, len(batch_list), parallel_batches):
        batch_group = batch_list[batch_idx:batch_idx + parallel_batches]
        
        # Process this group of batches in parallel
        tasks = [insert_litigation_batch(batch, source) for batch in batch_group]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count inserted
        for result in results:
            if isinstance(result, int):
                total_inserted += result
            elif isinstance(result, Exception):
                print(f"Error in batch: {result}")
        
        # Progress logging
        processed = min((batch_idx + parallel_batches) * batch_size, total_cases)
        if processed % (batch_size * 10) == 0 or processed >= total_cases:
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = total_cases - processed
            eta = remaining / rate if rate > 0 else 0
            
            print(f"\n📊 Progress: {processed:,}/{total_cases:,} cases ({processed*100//total_cases}%)")
            print(f"   Inserted: {total_inserted:,} cases")
            print(f"   Rate: {rate:.1f} cases/sec")
            if eta > 0:
                print(f"   ETA: {eta/60:.1f} minutes")
    
    elapsed_total = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"✅ INGESTION COMPLETE: {source.upper()}")
    print(f"{'='*60}")
    print(f"   Total cases: {total_cases:,}")
    print(f"   Inserted: {total_inserted:,} cases")
    print(f"   Total time: {elapsed_total/60:.1f} minutes ({elapsed_total:.1f} seconds)")
    print(f"   Throughput: {total_cases/elapsed_total:.1f} cases/sec")
    print(f"{'='*60}\n")
    
    await postgres_client.close()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python ingest_litigation_parallel.py <jsonl_path> <source>")
        print("Example: python ingest_litigation_parallel.py data/stanford_litigation.jsonl stanford")
        sys.exit(1)
    
    jsonl_path = sys.argv[1]
    source = sys.argv[2]
    asyncio.run(ingest_litigation_parallel(jsonl_path, source))


