"""Ingest patents with structural chunking + metadata into Qdrant and PostgreSQL.

Structural Chunking Strategy:
- Title (1 chunk)
- Abstract (1 chunk)
- Claims (chunked by claim number, ~500 chars each)
- Description (chunked by paragraphs, ~500 chars each)

Each chunk includes:
- patent_id
- chunk_type (title, abstract, claim_N, description_N)
- chunk_text
- chunk_order
- metadata (publication_date, cpc_codes, url, etc.)
"""
import asyncio
import json
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from qdrant_client.models import PointStruct, Distance, VectorParams

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
    clean_number = publication_number.replace("-", "")
    return f"https://patents.google.com/patent/{clean_number}"


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into chunks with overlap."""
    if not text or len(text) <= chunk_size:
        return [text] if text else []
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence endings
            last_period = chunk.rfind('. ')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)
            if break_point > chunk_size * 0.7:  # Only break if we're past 70% of chunk size
                chunk = text[start:start + break_point + 1]
                end = start + break_point + 1
        chunks.append(chunk.strip())
        start = end - overlap  # Overlap for context
    
    return chunks


def chunk_claims(claims_text: str) -> List[Dict[str, Any]]:
    """Chunk claims by claim number."""
    if not claims_text:
        return []
    
    chunks = []
    # Split by claim numbers (e.g., "1.", "2.", "Claim 1", etc.)
    claim_pattern = r'(?:^|\n)\s*(?:Claim\s+)?(\d+)\.\s*'
    claim_matches = list(re.finditer(claim_pattern, claims_text, re.MULTILINE))
    
    if not claim_matches:
        # No clear claim structure, chunk by size
        text_chunks = chunk_text(claims_text, chunk_size=500)
        for i, chunk in enumerate(text_chunks):
            chunks.append({
                "chunk_type": f"claim_part_{i+1}",
                "chunk_text": chunk,
                "chunk_order": i + 1,
            })
    else:
        # Chunk by claim number
        for i, match in enumerate(claim_matches):
            claim_num = match.group(1)
            start_pos = match.start()
            end_pos = claim_matches[i + 1].start() if i + 1 < len(claim_matches) else len(claims_text)
            claim_text = claims_text[start_pos:end_pos].strip()
            
            # If claim is too long, split it further
            if len(claim_text) > 1000:
                sub_chunks = chunk_text(claim_text, chunk_size=500)
                for j, sub_chunk in enumerate(sub_chunks):
                    chunks.append({
                        "chunk_type": f"claim_{claim_num}_part_{j+1}",
                        "chunk_text": sub_chunk,
                        "chunk_order": int(claim_num) * 100 + j,  # Order by claim number
                    })
            else:
                chunks.append({
                    "chunk_type": f"claim_{claim_num}",
                    "chunk_text": claim_text,
                    "chunk_order": int(claim_num) * 100,
                })
    
    return chunks


def chunk_description(description_text: str) -> List[Dict[str, Any]]:
    """Chunk description by paragraphs."""
    if not description_text:
        return []
    
    chunks = []
    # Split by paragraphs (double newlines or single newline after sentence)
    paragraphs = re.split(r'\n\s*\n', description_text)
    
    current_chunk = ""
    chunk_order = 1
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # If adding this paragraph would exceed chunk size, save current chunk
        if len(current_chunk) + len(para) > 500 and current_chunk:
            chunks.append({
                "chunk_type": f"description_part_{chunk_order}",
                "chunk_text": current_chunk.strip(),
                "chunk_order": chunk_order,
            })
            current_chunk = para
            chunk_order += 1
        else:
            current_chunk += "\n\n" + para if current_chunk else para
    
    # Add final chunk
    if current_chunk:
        chunks.append({
            "chunk_type": f"description_part_{chunk_order}",
            "chunk_text": current_chunk.strip(),
            "chunk_order": chunk_order,
        })
    
    return chunks


async def process_patent_batch(patent_batch: List[Dict[str, Any]], batch_num: int) -> Tuple[int, int]:
    """Process a batch of patents in parallel."""
    total_chunks = 0
    processed = 0
    
    # Process patents in parallel within batch
    tasks = []
    for patent_data in patent_batch:
        tasks.append(process_single_patent(patent_data))
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Collect points from all patents
    all_points = []
    for result in results:
        if isinstance(result, Exception):
            print(f"Error processing patent: {result}")
            continue
        if result:
            points, chunks_count = result
            all_points.extend(points)
            total_chunks += chunks_count
            processed += 1
    
    # Batch upsert all points
    if all_points:
        try:
            await qdrant_client.client.upsert(
                collection_name=qdrant_client.collection_name,
                points=all_points,
            )
        except Exception as e:
            print(f"Error upserting batch {batch_num}: {e}")
    
    return processed, total_chunks


async def process_single_patent(data: Dict[str, Any]) -> Optional[Tuple[List, int]]:
    """Process a single patent and return points + chunk count."""
    try:
        publication_number = data.get("publication_number", "")
        if not publication_number:
            return None
        
        title = data.get("title", "")
        abstract = data.get("abstract", "")
        claims = data.get("claims", "")
        description = data.get("description", "")
        cpc_codes = data.get("cpc_codes", "[]")
        filing_date = parse_filing_date(data.get("filing_date"))
        url = generate_patent_url(publication_number)
        
        # Parse CPC codes
        try:
            if isinstance(cpc_codes, str):
                cpc_codes = json.loads(cpc_codes)
        except:
            cpc_codes = []
        
        # Create structural chunks
        chunks = []
        if title:
            chunks.append({"chunk_type": "title", "chunk_text": title, "chunk_order": 0})
        if abstract:
            chunks.append({"chunk_type": "abstract", "chunk_text": abstract, "chunk_order": 1})
        if claims:
            chunks.extend(chunk_claims(claims))
        if description:
            chunks.extend(chunk_description(description))
        if not chunks and abstract:
            chunks.append({"chunk_type": "abstract", "chunk_text": abstract, "chunk_order": 0})
        
        # Generate embeddings for all chunks in parallel
        embedding_tasks = []
        for chunk in chunks:
            embedding_tasks.append(generate_chunk_embedding(chunk["chunk_text"]))
        
        embeddings = await asyncio.gather(*embedding_tasks)
        
        # Create points
        points = []
        for chunk_idx, (chunk, dense_embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{publication_number}:{chunk['chunk_type']}:{chunk_idx}"
            point_id = hash(chunk_id) % (2**63)
            
            point = PointStruct(
                id=point_id,
                vector=dense_embedding,
                payload={
                    "patent_id": publication_number,
                    "chunk_id": chunk_id,
                    "chunk_type": chunk["chunk_type"],
                    "chunk_text": chunk["chunk_text"],
                    "chunk_order": chunk["chunk_order"],
                    "title": title,
                    "abstract": abstract,
                    "publication_date": str(filing_date) if filing_date else None,
                    "cpc_codes": cpc_codes,
                    "url": url,
                },
            )
            points.append(point)
        
        return points, len(points)
    except Exception as e:
        print(f"Error processing patent {data.get('publication_number', 'unknown')}: {e}")
        return None


async def generate_chunk_embedding(chunk_text: str) -> List[float]:
    """Generate embedding for a chunk."""
    return await qdrant_client.generate_dense_embedding(chunk_text)


async def ingest_patents_structural(jsonl_path: str, batch_size: int = 100, parallel_patents: int = 10):
    """Ingest patents with structural chunking - PARALLELIZED.
    
    Args:
        jsonl_path: Path to JSONL file
        batch_size: Number of patents to process before logging progress
        parallel_patents: Number of patents to process in parallel
    """
    print(f"Starting PARALLEL structural patent ingestion from {jsonl_path}...")
    print(f"Batch size: {batch_size}, Parallel patents: {parallel_patents}")
    
    # Recreate collection with proper vector size (384 for all-MiniLM-L6-v2)
    # Note: This will DELETE existing data - use with caution!
    print("\n" + "="*60)
    print("⚠️  WARNING: This will DELETE the existing collection!")
    print("   All current patent data will be lost.")
    print("   Press Ctrl+C to cancel, or wait 10 seconds to continue...")
    print("="*60)
    await asyncio.sleep(10)
    
    # Delete existing collection if it exists
    try:
        collections = await qdrant_client.client.get_collections()
        collection_names = [col.name for col in collections.collections]
        if qdrant_client.collection_name in collection_names:
            print(f"\n🗑️  Deleting existing collection: {qdrant_client.collection_name}")
            await qdrant_client.client.delete_collection(qdrant_client.collection_name)
            print(f"✅ Deleted existing collection")
        else:
            print(f"\nℹ️  Collection '{qdrant_client.collection_name}' does not exist, will create new one")
    except Exception as e:
        print(f"❌ Error deleting collection: {e}")
        raise
    
    # Create new collection with proper configuration
    print(f"\n📦 Creating new collection: {qdrant_client.collection_name}")
    from qdrant_client.models import VectorParams, Distance, SparseVectorParams
    
    await qdrant_client.client.create_collection(
        collection_name=qdrant_client.collection_name,
        vectors_config=VectorParams(
            size=384,  # all-MiniLM-L6-v2 dimension
            distance=Distance.COSINE,
        ),
    )
    print(f"✅ Created new collection with 384-dimensional vectors")
    
    # Try to connect to PostgreSQL (optional)
    try:
        await postgres_client.connect()
    except Exception as e:
        print(f"Warning: Could not connect to PostgreSQL: {e}")
        print("Continuing with Qdrant only...")
    
    # Load all patents into memory (for parallel processing)
    print("\n📖 Loading patents from file...")
    all_patents = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                if data.get("publication_number"):
                    all_patents.append(data)
            except json.JSONDecodeError:
                continue
    
    total_patents = len(all_patents)
    print(f"✅ Loaded {total_patents:,} patents")
    
    total_processed = 0
    total_chunks = 0
    start_time = asyncio.get_event_loop().time()
    
    # Process in batches with parallelization
    for batch_start in range(0, total_patents, parallel_patents):
        batch_end = min(batch_start + parallel_patents, total_patents)
        patent_batch = all_patents[batch_start:batch_end]
        batch_num = (batch_start // parallel_patents) + 1
        
        # Process batch in parallel
        processed, chunks = await process_patent_batch(patent_batch, batch_num)
        total_processed += processed
        total_chunks += chunks
        
        # Progress logging
        if total_processed % batch_size == 0 or batch_end == total_patents:
            elapsed = asyncio.get_event_loop().time() - start_time
            rate = total_processed / elapsed if elapsed > 0 else 0
            remaining = total_patents - total_processed
            eta = remaining / rate if rate > 0 else 0
            
            print(f"\n📊 Progress: {total_processed:,}/{total_patents:,} patents ({total_processed*100//total_patents if total_patents > 0 else 0}%)")
            print(f"   Chunks created: {total_chunks:,}")
            print(f"   Rate: {rate:.1f} patents/sec")
            if eta > 0:
                print(f"   ETA: {eta/60:.1f} minutes")
        
        # Insert into PostgreSQL in batches (non-blocking)
        if postgres_client.pool and patent_batch:
            asyncio.create_task(insert_patents_postgres_batch(patent_batch))
    
    elapsed_total = asyncio.get_event_loop().time() - start_time
    
    print(f"\n" + "="*60)
    print(f"✅ INGESTION COMPLETE!")
    print(f"="*60)
    print(f"   Processed: {total_processed:,} patents")
    print(f"   Created: {total_chunks:,} chunks")
    print(f"   Average: {total_chunks/total_processed:.1f} chunks per patent")
    print(f"   Total time: {elapsed_total/60:.1f} minutes ({elapsed_total:.1f} seconds)")
    print(f"   Throughput: {total_processed/elapsed_total:.1f} patents/sec")
    print(f"="*60)
    
    # Close connections
    if postgres_client.pool:
        await postgres_client.close()
    await qdrant_client.close()


async def insert_patents_postgres_batch(patent_batch: List[Dict[str, Any]]):
    """Insert a batch of patents into PostgreSQL."""
    if not postgres_client.pool:
        return
    
    try:
        async with postgres_client.pool.acquire() as conn:
            for data in patent_batch:
                publication_number = data.get("publication_number", "")
                if not publication_number:
                    continue
                
                title = data.get("title", "")
                filing_date = parse_filing_date(data.get("filing_date"))
                cpc_codes = data.get("cpc_codes", "[]")
                url = generate_patent_url(publication_number)
                
                try:
                    if isinstance(cpc_codes, str):
                        cpc_codes = json.loads(cpc_codes)
                except:
                    cpc_codes = []
                
                await conn.execute(
                    """
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
                    """,
                    publication_number,
                    title,
                    filing_date,
                    json.dumps(cpc_codes) if cpc_codes else None,
                    url,
                )
    except Exception as e:
        # Silent fail - PostgreSQL is optional
        pass


if __name__ == "__main__":
    jsonl_path = sys.argv[1] if len(sys.argv) > 1 else "data/patents_bigquery.jsonl"
    asyncio.run(ingest_patents_structural(jsonl_path))

