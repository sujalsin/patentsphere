"""Multiprocessing-based parallel litigation ingestion.

This script uses multiprocessing to insert litigation cases into PostgreSQL
in parallel, significantly speeding up ingestion.

Note: Litigation data doesn't require embeddings, so we focus on parallel
database inserts for maximum throughput.
"""
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from multiprocessing import Pool, cpu_count
from functools import partial
import asyncio
import asyncpg

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings


def parse_filing_date(filing_date_str: str) -> str:
    """Parse filing date from string."""
    if not filing_date_str:
        return None
    try:
        # Try YYYY-MM-DD format
        if len(filing_date_str) == 10 and '-' in filing_date_str:
            return filing_date_str
        # Try YYYYMMDD format
        elif len(filing_date_str) == 8:
            return f"{filing_date_str[:4]}-{filing_date_str[4:6]}-{filing_date_str[6:8]}"
    except:
        pass
    return None


def insert_litigation_batch_worker(cases_batch: List[Dict[str, Any]], source: str) -> int:
    """Worker function to insert a batch of litigation cases (runs in separate process)."""
    inserted = 0
    
    # Create connection in this process
    async def insert_batch():
        nonlocal inserted
        conn = None
        try:
            conn = await asyncpg.connect(
                host=settings.postgres_host,
                port=settings.postgres_port,
                user=settings.postgres_user,
                password=settings.postgres_password,
                database=settings.postgres_database,
            )
            
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
                            source = EXCLUDED.source,
                            updated_at = CURRENT_TIMESTAMP
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
                    # Skip invalid cases
                    continue
        except Exception as e:
            print(f"Error in worker: {e}")
        finally:
            if conn:
                await conn.close()
        return inserted
    
    # Run async function in worker process
    return asyncio.run(insert_batch())


def ingest_litigation_multiprocessing(
    jsonl_path: str,
    source: str,
    num_workers: int = None,
    batch_size: int = 1000,
):
    """Ingest litigation data using multiprocessing for parallel database inserts.
    
    Args:
        jsonl_path: Path to JSONL file
        source: Source name (e.g., "stanford", "uspto")
        num_workers: Number of worker processes (default: CPU count)
        batch_size: Number of cases per batch
    """
    print("="*60)
    print(f"🚀 Multiprocessing Parallel Litigation Ingestion: {source.upper()}")
    print("="*60)
    
    # Determine number of workers
    if num_workers is None:
        num_workers = cpu_count()
    
    print(f"\n📊 Using {num_workers} worker processes (CPU cores: {cpu_count()})")
    print(f"   Batch size: {batch_size} cases per batch")
    
    # Load all cases
    print(f"\n📖 Loading litigation cases from {jsonl_path}...")
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
    
    # Split into batches
    case_batches = []
    for i in range(0, total_cases, batch_size):
        batch = all_cases[i:min(i + batch_size, total_cases)]
        case_batches.append(batch)
    
    print(f"\n⚙️  Processing {len(case_batches):,} batches in parallel...")
    processing_start = datetime.now()
    
    # Process batches in parallel
    process_func = partial(insert_litigation_batch_worker, source=source)
    
    with Pool(processes=num_workers) as pool:
        results = pool.imap_unordered(process_func, case_batches)
        
        total_inserted = 0
        completed = 0
        
        for inserted_count in results:
            total_inserted += inserted_count
            completed += 1
            
            if completed % 10 == 0 or completed == len(case_batches):
                elapsed = (datetime.now() - processing_start).total_seconds()
                processed = min(completed * batch_size, total_cases)
                rate = processed / elapsed if elapsed > 0 else 0
                remaining = total_cases - processed
                eta = remaining / rate if rate > 0 else 0
                
                print(f"   Processed {completed:,}/{len(case_batches):,} batches ({completed*100//len(case_batches)}%)...", end='\r')
                if completed % 50 == 0:
                    print(f"\n   Inserted: {total_inserted:,} cases | Rate: {rate:.1f} cases/sec | ETA: {eta/60:.1f} min")
    
    processing_time = (datetime.now() - processing_start).total_seconds()
    
    print("\n" + "="*60)
    print(f"✅ INGESTION COMPLETE: {source.upper()}")
    print("="*60)
    print(f"   Total cases: {total_cases:,}")
    print(f"   Inserted: {total_inserted:,} cases")
    print(f"   Total time: {processing_time/60:.1f} minutes ({processing_time:.1f} seconds)")
    if processing_time > 0:
        print(f"   Throughput: {total_cases/processing_time:.1f} cases/sec")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest litigation using multiprocessing")
    parser.add_argument("jsonl_path", help="Path to JSONL file")
    parser.add_argument("source", help="Source name (e.g., stanford, uspto)")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes (default: CPU count)")
    parser.add_argument("--batch-size", type=int, default=1000, help="Cases per batch")
    
    args = parser.parse_args()
    
    ingest_litigation_multiprocessing(
        jsonl_path=args.jsonl_path,
        source=args.source,
        num_workers=args.workers,
        batch_size=args.batch_size,
    )

