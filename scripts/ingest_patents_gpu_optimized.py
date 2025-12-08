"""GPU-optimized parallel patent ingestion with hybrid CPU/GPU processing.

This script uses a hybrid approach:
- CPU workers: Handle chunking in parallel
- GPU worker: Handles all embeddings (single process to avoid GPU memory conflicts)

This provides optimal performance by leveraging both CPU and GPU efficiently.
"""
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from multiprocessing import Pool, cpu_count, Queue, Process, Manager, get_context
from functools import partial
import asyncio
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client.models import PointStruct, Distance, VectorParams
from qdrant_client import QdrantClient

# Import chunking functions
from scripts.ingest_patents_structural import (
    parse_filing_date,
    generate_patent_url,
    chunk_claims,
    chunk_description,
)

from config import settings
from db.postgres_client import postgres_client


def create_chunks_for_patent(patent_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create structural chunks for a single patent (CPU-bound)."""
    publication_number = patent_data.get("publication_number", "")
    if not publication_number:
        return []
    
    title = patent_data.get("title", "")
    abstract = patent_data.get("abstract", "")
    claims = patent_data.get("claims", "")
    description = patent_data.get("description", "")
    filing_date = parse_filing_date(patent_data.get("filing_date"))
    url = generate_patent_url(publication_number)
    
    cpc_codes = patent_data.get("cpc_codes", "[]")
    try:
        if isinstance(cpc_codes, str):
            cpc_codes = json.loads(cpc_codes)
    except:
        cpc_codes = []
    
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


def chunk_worker(patent_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """CPU worker: Creates chunks from patents."""
    all_chunks = []
    for patent_data in patent_batch:
        chunks = create_chunks_for_patent(patent_data)
        all_chunks.extend(chunks)
    return all_chunks


def gpu_embedding_worker(chunk_queue: Queue, result_queue: Queue, use_gpu: bool):
    """GPU worker: Generates embeddings for chunks."""
    from langchain_huggingface import HuggingFaceEmbeddings
    
    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    
    # Initialize embedding model once
    embedder = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
    
    processed_count = 0
    while True:
        batch = chunk_queue.get()
        if batch is None:  # Sentinel value to stop
            break
        
        chunk_texts = [chunk["chunk_text"] for chunk in batch]
        if chunk_texts:
            # Generate embeddings on GPU
            embeddings = embedder.embed_documents(chunk_texts)
            
            # Attach embeddings
            for chunk_idx, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                chunk["embedding"] = embedding
                chunk["chunk_id"] = f"{chunk['patent_id']}:{chunk['chunk_type']}:{chunk_idx}"
                chunk["point_id"] = hash(chunk["chunk_id"]) % (2**63)
        
        result_queue.put(batch)
        processed_count += len(batch)
    
    return processed_count


def ingest_patents_gpu_optimized(
    jsonl_path: str,
    num_workers: int = None,
    batch_size: int = 10,
    embedding_batch_size: int = 100,
    qdrant_batch_size: int = 1000,
    delete_existing: bool = True,
    use_gpu: bool = True,
):
    """Ingest patents using hybrid CPU/GPU processing.
    
    Args:
        jsonl_path: Path to JSONL file
        num_workers: Number of CPU workers for chunking (default: CPU count - 1)
        batch_size: Number of patents per batch per worker
        embedding_batch_size: Chunks per embedding batch
        qdrant_batch_size: Batch size for Qdrant uploads
        delete_existing: Whether to delete existing collection
        use_gpu: Whether to use GPU for embeddings
    """
    print("="*60)
    print("🚀 GPU-Optimized Parallel Patent Ingestion")
    print("="*60)
    
    # Check GPU availability
    gpu_available = torch.cuda.is_available()
    device = "cuda" if (use_gpu and gpu_available) else "cpu"
    
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
    
    # Determine number of workers (reserve 1 for GPU worker if using GPU)
    if num_workers is None:
        num_workers = max(cpu_count() - 1, 1) if (use_gpu and gpu_available) else cpu_count()
    
    print(f"\n📊 Using {num_workers} CPU workers for chunking (CPU cores: {cpu_count()})")
    print(f"   Batch size per worker: {batch_size} patents")
    print(f"   Embedding batch size: {embedding_batch_size} chunks")
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
    
    # Initialize Qdrant
    print("\n🔌 Connecting to Qdrant...")
    qdrant_client_sync = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )
    
    # Check if collection exists
    collections = qdrant_client_sync.get_collections()
    collection_names = [col.name for col in collections.collections]
    collection_exists = settings.qdrant_collection in collection_names
    
    # Delete existing collection if requested
    if delete_existing and collection_exists:
        print("\n" + "="*60)
        print("⚠️  WARNING: This will DELETE the existing collection!")
        print("   All current patent data will be lost.")
        print("   Press Ctrl+C to cancel, or wait 10 seconds to continue...")
        print("="*60)
        import time
        time.sleep(10)
        
        try:
            print(f"\n🗑️  Deleting existing collection: {settings.qdrant_collection}")
            qdrant_client_sync.delete_collection(settings.qdrant_collection)
            print(f"✅ Deleted existing collection")
            collection_exists = False
        except Exception as e:
            print(f"❌ Error deleting collection: {e}")
            raise
    
    # Create collection if it doesn't exist
    if not collection_exists:
        print(f"\n📦 Creating new collection: {settings.qdrant_collection}")
        try:
            qdrant_client_sync.create_collection(
                collection_name=settings.qdrant_collection,
                vectors_config=VectorParams(
                    size=384,
                    distance=Distance.COSINE,
                ),
            )
            print(f"✅ Created new collection with 384-dimensional vectors")
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"ℹ️  Collection already exists, will append to it")
            else:
                raise
    else:
        print(f"\n📦 Using existing collection: {settings.qdrant_collection}")
        info = qdrant_client_sync.get_collection(settings.qdrant_collection)
        print(f"   Current points: {info.points_count:,}")
        print(f"   Will append new patents to existing collection")
    
    # Split patents into batches
    patent_batches = []
    for i in range(0, total_patents, batch_size):
        batch = all_patents[i:i+batch_size]
        patent_batches.append(batch)
    
    print(f"\n⚙️  Processing {len(patent_batches):,} batches...")
    processing_start = datetime.now()
    
    # Create queues for GPU worker
    # Use spawn context for Manager when GPU is enabled (required for CUDA)
    if use_gpu and gpu_available:
        ctx = get_context('spawn')
        manager = ctx.Manager()
        chunk_queue = manager.Queue()
        result_queue = manager.Queue()
        
        # Start GPU embedding worker using spawn context for CUDA compatibility
        gpu_process = ctx.Process(
            target=gpu_embedding_worker,
            args=(chunk_queue, result_queue, True)
        )
    else:
        # Use default fork for CPU (faster)
        manager = Manager()
        chunk_queue = manager.Queue()
        result_queue = manager.Queue()
        
        gpu_process = Process(
            target=gpu_embedding_worker,
            args=(chunk_queue, result_queue, False)
        )
    gpu_process.start()
    
    # Process chunks in parallel on CPU
    # Use fork for CPU workers (faster than spawn)
    all_chunks = []
    with Pool(processes=num_workers) as pool:
        chunk_results = pool.imap_unordered(chunk_worker, patent_batches)
        
        # Collect chunks and send to GPU worker in batches
        current_batch = []
        completed = 0
        
        for chunks in chunk_results:
            all_chunks.extend(chunks)
            current_batch.extend(chunks)
            
            # Send batch to GPU worker when it reaches embedding_batch_size
            if len(current_batch) >= embedding_batch_size:
                chunk_queue.put(current_batch)
                current_batch = []
            
            completed += 1
            if completed % 10 == 0 or completed == len(patent_batches):
                print(f"   Chunked {completed:,}/{len(patent_batches):,} batches ({completed*100//len(patent_batches)}%)...", end='\r')
        
        # Send remaining chunks
        if current_batch:
            chunk_queue.put(current_batch)
    
    # Signal GPU worker to stop
    chunk_queue.put(None)
    
    # Collect embedded chunks from GPU worker
    print(f"\n   Generating embeddings on {device.upper()}...")
    embedded_chunks = []
    total_embedded = 0
    
    while total_embedded < len(all_chunks):
        batch = result_queue.get()
        embedded_chunks.extend(batch)
        total_embedded += len(batch)
        print(f"   Embedded {total_embedded:,}/{len(all_chunks):,} chunks...", end='\r')
    
    gpu_process.join()
    
    processing_time = (datetime.now() - processing_start).total_seconds()
    print(f"\n✅ Processed {len(embedded_chunks):,} chunks in {processing_time:.1f} seconds")
    if total_patents > 0:
        print(f"   Average: {len(embedded_chunks)/total_patents:.1f} chunks per patent")
    
    # Upload to Qdrant
    print(f"\n📤 Uploading to Qdrant in batches of {qdrant_batch_size}...")
    upload_start = datetime.now()
    
    points = []
    for i, chunk in enumerate(embedded_chunks):
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
        
        if len(points) >= qdrant_batch_size:
            qdrant_client_sync.upsert(
                collection_name=settings.qdrant_collection,
                points=points,
            )
            print(f"   Uploaded {i+1:,}/{len(embedded_chunks):,} chunks...", end='\r')
            points = []
    
    if points:
        qdrant_client_sync.upsert(
            collection_name=settings.qdrant_collection,
            points=points,
        )
    
    upload_time = (datetime.now() - upload_start).total_seconds()
    print(f"\n✅ Uploaded {len(embedded_chunks):,} chunks to Qdrant in {upload_time:.1f} seconds")
    
    # Upload to PostgreSQL
    print(f"\n📤 Uploading metadata to PostgreSQL...")
    asyncio.run(upload_to_postgres_parallel(embedded_chunks))
    
    total_time = (datetime.now() - processing_start).total_seconds()
    
    print("\n" + "="*60)
    print("✅ INGESTION COMPLETE!")
    print("="*60)
    print(f"   Processed: {total_patents:,} patents")
    print(f"   Created: {len(embedded_chunks):,} chunks")
    if total_patents > 0:
        print(f"   Average: {len(embedded_chunks)/total_patents:.1f} chunks per patent")
    print(f"   Processing time: {processing_time/60:.1f} minutes")
    print(f"   Upload time: {upload_time/60:.1f} minutes")
    print(f"   Total time: {total_time/60:.1f} minutes ({total_time:.1f} seconds)")
    if total_time > 0:
        print(f"   Throughput: {total_patents/total_time:.1f} patents/sec")
    print("="*60)


async def upload_to_postgres_parallel(chunks: List[Dict[str, Any]]):
    """Upload patent metadata to PostgreSQL."""
    try:
        await postgres_client.connect()
    except Exception as e:
        print(f"Warning: Could not connect to PostgreSQL: {e}")
        return
    
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
        pass


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest patents using GPU-optimized processing")
    parser.add_argument("jsonl_path", help="Path to JSONL file")
    parser.add_argument("--workers", type=int, default=None, help="Number of CPU workers (default: CPU count - 1)")
    parser.add_argument("--batch-size", type=int, default=10, help="Patents per batch per worker")
    parser.add_argument("--embedding-batch-size", type=int, default=100, help="Chunks per embedding batch")
    parser.add_argument("--qdrant-batch-size", type=int, default=1000, help="Qdrant batch size")
    parser.add_argument("--keep-existing", action="store_true", help="Keep existing collection")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    
    args = parser.parse_args()
    
    ingest_patents_gpu_optimized(
        jsonl_path=args.jsonl_path,
        num_workers=args.workers,
        batch_size=args.batch_size,
        embedding_batch_size=args.embedding_batch_size,
        qdrant_batch_size=args.qdrant_batch_size,
        delete_existing=not args.keep_existing,
        use_gpu=not args.no_gpu,
    )

