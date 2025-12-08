"""Multiprocessing-based parallel patent ingestion with structural chunking.

This script uses Python's multiprocessing to process patents in parallel across
multiple CPU cores, with optional GPU acceleration for embeddings.

Key optimizations:
- Parallel processing across multiple CPU cores
- GPU acceleration for embeddings (if available)
- Batch embedding generation per process
- Efficient memory management
- Progress tracking
- No Java/PySpark dependency required
"""
import sys
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from multiprocessing import Pool, cpu_count, Manager, Queue, Process
from functools import partial
import asyncio
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client.models import PointStruct, Distance, VectorParams
from qdrant_client import QdrantClient

# Import chunking functions from original script
from scripts.ingest_patents_structural import (
    parse_filing_date,
    generate_patent_url,
    chunk_claims,
    chunk_description,
)

# Import config and clients
from config import settings
from db.postgres_client import postgres_client


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into chunks with overlap."""
    if not text or len(text) <= chunk_size:
        return [text] if text else []
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if end < len(text):
            last_period = chunk.rfind('. ')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)
            if break_point > chunk_size * 0.7:
                chunk = text[start:start + break_point + 1]
                end = start + break_point + 1
        chunks.append(chunk.strip())
        start = end - overlap
    
    return chunks


def create_chunks_for_patent(patent_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create structural chunks for a single patent."""
    publication_number = patent_data.get("publication_number", "")
    if not publication_number:
        return []
    
    title = patent_data.get("title", "")
    abstract = patent_data.get("abstract", "")
    claims = patent_data.get("claims", "")
    description = patent_data.get("description", "")
    filing_date = parse_filing_date(patent_data.get("filing_date"))
    url = generate_patent_url(publication_number)
    
    # Parse CPC codes
    cpc_codes = patent_data.get("cpc_codes", "[]")
    try:
        if isinstance(cpc_codes, str):
            cpc_codes = json.loads(cpc_codes)
    except:
        cpc_codes = []
    
    # Create structural chunks
    chunks = []
    if title:
        chunks.append({
            "chunk_type": "title",
            "chunk_text": title,
            "chunk_order": 0,
            "patent_id": publication_number,
            "title": title,
            "abstract": abstract,
            "publication_date": str(filing_date) if filing_date else None,
            "cpc_codes": cpc_codes,
            "url": url,
        })
    if abstract:
        chunks.append({
            "chunk_type": "abstract",
            "chunk_text": abstract,
            "chunk_order": 1,
            "patent_id": publication_number,
            "title": title,
            "abstract": abstract,
            "publication_date": str(filing_date) if filing_date else None,
            "cpc_codes": cpc_codes,
            "url": url,
        })
    if claims:
        claim_chunks = chunk_claims(claims)
        for chunk in claim_chunks:
            chunk["patent_id"] = publication_number
            chunk["title"] = title
            chunk["abstract"] = abstract
            chunk["publication_date"] = str(filing_date) if filing_date else None
            chunk["cpc_codes"] = cpc_codes
            chunk["url"] = url
            chunks.append(chunk)
    if description:
        desc_chunks = chunk_description(description)
        for chunk in desc_chunks:
            chunk["patent_id"] = publication_number
            chunk["title"] = title
            chunk["abstract"] = abstract
            chunk["publication_date"] = str(filing_date) if filing_date else None
            chunk["cpc_codes"] = cpc_codes
            chunk["url"] = url
            chunks.append(chunk)
    if not chunks and abstract:
        chunks.append({
            "chunk_type": "abstract",
            "chunk_text": abstract,
            "chunk_order": 0,
            "patent_id": publication_number,
            "title": title,
            "abstract": abstract,
            "publication_date": str(filing_date) if filing_date else None,
            "cpc_codes": cpc_codes,
            "url": url,
        })
    
    return chunks


def get_device():
    """Detect and return the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def process_patent_batch(patent_batch: List[Dict[str, Any]], use_gpu: bool = False) -> List[Dict[str, Any]]:
    """Process a batch of patents and return chunks with embeddings."""
    from langchain_huggingface import HuggingFaceEmbeddings
    
    # Determine device
    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    
    # Initialize embedding model once per process
    embedder = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
    
    all_chunks = []
    for patent_data in patent_batch:
        chunks = create_chunks_for_patent(patent_data)
        all_chunks.extend(chunks)
    
    # Generate embeddings for all chunks in this batch
    chunk_texts = [chunk["chunk_text"] for chunk in all_chunks]
    if chunk_texts:
        # GPU is much faster for batch embeddings
        embeddings = embedder.embed_documents(chunk_texts)
        
        # Attach embeddings to chunks
        for chunk_idx, (chunk, embedding) in enumerate(zip(all_chunks, embeddings)):
            chunk["embedding"] = embedding
            chunk["chunk_id"] = f"{chunk['patent_id']}:{chunk['chunk_type']}:{chunk_idx}"
            chunk["point_id"] = hash(chunk["chunk_id"]) % (2**63)
    
    return all_chunks


def ingest_patents_multiprocessing(
    jsonl_path: str,
    num_workers: int = None,
    batch_size: int = 10,
    qdrant_batch_size: int = 1000,
    delete_existing: bool = True,
    use_gpu: bool = True,
):
    """Ingest patents using multiprocessing for parallel processing.
    
    Args:
        jsonl_path: Path to JSONL file
        num_workers: Number of worker processes (default: CPU count)
        batch_size: Number of patents per batch per worker
        qdrant_batch_size: Batch size for Qdrant uploads
        delete_existing: Whether to delete existing collection
        use_gpu: Whether to use GPU for embeddings (if available)
    """
    print("="*60)
    print("🚀 Multiprocessing Parallel Patent Ingestion")
    print("="*60)
    
    # Check GPU availability
    gpu_available = torch.cuda.is_available()
    device = get_device()
    
    if use_gpu and gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n🎮 GPU Acceleration: ENABLED")
        print(f"   Device: {gpu_name}")
        print(f"   Memory: {gpu_memory:.1f} GB")
        print(f"   ⚡ Embeddings will be 5-10x faster on GPU!")
    else:
        if use_gpu:
            print(f"\n⚠️  GPU requested but not available, using CPU")
        else:
            print(f"\n💻 Using CPU for embeddings")
        device = "cpu"
    
    # Determine number of workers
    if num_workers is None:
        num_workers = cpu_count()
    
    print(f"\n📊 Using {num_workers} worker processes (CPU cores: {cpu_count()})")
    print(f"   Batch size per worker: {batch_size} patents")
    print(f"   Device: {device.upper()}")
    
    # Load all patents
    print(f"\n📖 Loading patents from {jsonl_path}...")
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
    
    # Initialize Qdrant client
    print("\n🔌 Connecting to Qdrant...")
    qdrant_client_sync = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )
    
    # Delete existing collection if requested
    if delete_existing:
        print("\n" + "="*60)
        print("⚠️  WARNING: This will DELETE the existing collection!")
        print("   All current patent data will be lost.")
        print("   Press Ctrl+C to cancel, or wait 10 seconds to continue...")
        print("="*60)
        import time
        time.sleep(10)
        
        try:
            collections = qdrant_client_sync.get_collections()
            collection_names = [col.name for col in collections.collections]
            if settings.qdrant_collection in collection_names:
                print(f"\n🗑️  Deleting existing collection: {settings.qdrant_collection}")
                qdrant_client_sync.delete_collection(settings.qdrant_collection)
                print(f"✅ Deleted existing collection")
        except Exception as e:
            print(f"❌ Error deleting collection: {e}")
            raise
    
    # Create new collection
    print(f"\n📦 Creating new collection: {settings.qdrant_collection}")
    qdrant_client_sync.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(
            size=384,  # all-MiniLM-L6-v2 dimension
            distance=Distance.COSINE,
        ),
    )
    print(f"✅ Created new collection with 384-dimensional vectors")
    
    # Split patents into batches for parallel processing
    patent_batches = []
    for i in range(0, total_patents, batch_size):
        batch = all_patents[i:i+batch_size]
        patent_batches.append(batch)
    
    print(f"\n⚙️  Processing {len(patent_batches):,} batches in parallel...")
    print(f"   This may take a while - embeddings are being generated...")
    
    processing_start = datetime.now()
    
    # Process batches in parallel
    # For GPU, we need to use 'spawn' method for CUDA compatibility
    if use_gpu and gpu_available:
        print(f"\n⚠️  Note: Using 'spawn' method for CUDA compatibility (slower startup).")
        print(f"   For optimal GPU performance, consider: python scripts/ingest_patents_gpu_optimized.py")
        print(f"   Continuing with multiprocessing...\n")
        
        # Use spawn context for CUDA compatibility
        from multiprocessing import get_context
        ctx = get_context('spawn')
        pool_class = ctx.Pool
    else:
        # Use default fork for CPU (faster)
        pool_class = Pool
    
    # Create partial function with GPU setting
    process_func = partial(process_patent_batch, use_gpu=(use_gpu and gpu_available))
    
    with pool_class(processes=num_workers) as pool:
        # Use imap_unordered for better progress tracking
        results = pool.imap_unordered(process_func, patent_batches)
        
        all_chunks = []
        completed = 0
        for result in results:
            all_chunks.extend(result)
            completed += 1
            if completed % 10 == 0 or completed == len(patent_batches):
                print(f"   Processed {completed:,}/{len(patent_batches):,} batches ({completed*100//len(patent_batches)}%)...", end='\r')
    
    processing_time = (datetime.now() - processing_start).total_seconds()
    print(f"\n✅ Processed {len(all_chunks):,} chunks in {processing_time:.1f} seconds")
    if total_patents > 0:
        print(f"   Average: {len(all_chunks)/total_patents:.1f} chunks per patent")
    
    # Upload to Qdrant in batches
    print(f"\n📤 Uploading to Qdrant in batches of {qdrant_batch_size}...")
    upload_start = datetime.now()
    
    points = []
    for i, chunk in enumerate(all_chunks):
        point = PointStruct(
            id=chunk["point_id"],
            vector=chunk["embedding"],
            payload={
                "patent_id": chunk["patent_id"],
                "chunk_id": chunk["chunk_id"],
                "chunk_type": chunk["chunk_type"],
                "chunk_text": chunk["chunk_text"],
                "chunk_order": chunk["chunk_order"],
                "title": chunk["title"],
                "abstract": chunk["abstract"],
                "publication_date": chunk["publication_date"],
                "cpc_codes": chunk["cpc_codes"],
                "url": chunk["url"],
            },
        )
        points.append(point)
        
        # Upload in batches
        if len(points) >= qdrant_batch_size:
            qdrant_client_sync.upsert(
                collection_name=settings.qdrant_collection,
                points=points,
            )
            print(f"   Uploaded {i+1:,}/{len(all_chunks):,} chunks...", end='\r')
            points = []
    
    # Upload remaining points
    if points:
        qdrant_client_sync.upsert(
            collection_name=settings.qdrant_collection,
            points=points,
        )
    
    upload_time = (datetime.now() - upload_start).total_seconds()
    print(f"\n✅ Uploaded {len(all_chunks):,} chunks to Qdrant in {upload_time:.1f} seconds")
    
    # Upload to PostgreSQL in parallel
    print(f"\n📤 Uploading metadata to PostgreSQL...")
    asyncio.run(upload_to_postgres_parallel(all_chunks))
    
    total_time = (datetime.now() - processing_start).total_seconds()
    
    print("\n" + "="*60)
    print("✅ INGESTION COMPLETE!")
    print("="*60)
    print(f"   Processed: {total_patents:,} patents")
    print(f"   Created: {len(all_chunks):,} chunks")
    if total_patents > 0:
        print(f"   Average: {len(all_chunks)/total_patents:.1f} chunks per patent")
    print(f"   Processing time: {processing_time/60:.1f} minutes")
    print(f"   Upload time: {upload_time/60:.1f} minutes")
    print(f"   Total time: {total_time/60:.1f} minutes ({total_time:.1f} seconds)")
    if total_time > 0:
        print(f"   Throughput: {total_patents/total_time:.1f} patents/sec")
    print("="*60)


async def upload_to_postgres_parallel(chunks: List[Dict[str, Any]]):
    """Upload patent metadata to PostgreSQL in parallel."""
    try:
        await postgres_client.connect()
    except Exception as e:
        print(f"Warning: Could not connect to PostgreSQL: {e}")
        return
    
    # Get unique patents from chunks
    patents = {}
    for chunk in chunks:
        patent_id = chunk["patent_id"]
        if patent_id not in patents:
            patents[patent_id] = {
                "publication_number": patent_id,
                "title": chunk.get("title", ""),
                "publication_date": chunk.get("publication_date"),
                "cpc_codes": chunk.get("cpc_codes", []),
                "url": chunk.get("url", ""),
            }
    
    # Batch insert
    batch_size = 1000
    patent_list = list(patents.values())
    
    for i in range(0, len(patent_list), batch_size):
        batch = patent_list[i:i+batch_size]
        await insert_patents_postgres_batch(batch)
        print(f"   Inserted {min(i+batch_size, len(patent_list)):,}/{len(patent_list):,} patents...", end='\r')
    
    print(f"\n✅ Inserted {len(patent_list):,} patents into PostgreSQL")
    await postgres_client.close()


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
                filing_date = data.get("publication_date")
                cpc_codes = data.get("cpc_codes", [])
                url = data.get("url", "")
                
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
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest patents using multiprocessing")
    parser.add_argument("jsonl_path", help="Path to JSONL file")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes (default: CPU count)")
    parser.add_argument("--batch-size", type=int, default=10, help="Patents per batch per worker")
    parser.add_argument("--qdrant-batch-size", type=int, default=1000, help="Qdrant batch size")
    parser.add_argument("--keep-existing", action="store_true", help="Keep existing collection (don't delete)")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration (use CPU only)")
    
    args = parser.parse_args()
    
    ingest_patents_multiprocessing(
        jsonl_path=args.jsonl_path,
        num_workers=args.workers,
        batch_size=args.batch_size,
        qdrant_batch_size=args.qdrant_batch_size,
        delete_existing=not args.keep_existing,
        use_gpu=not args.no_gpu,
    )

