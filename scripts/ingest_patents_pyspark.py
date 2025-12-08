"""PySpark-based parallel patent ingestion with structural chunking.

This script uses PySpark to process patents in parallel across multiple cores,
significantly speeding up chunking and embedding generation.

Key optimizations:
- Parallel processing across Spark partitions
- Batch embedding generation
- Efficient memory management
- Progress tracking
"""
import sys
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from functools import partial

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, explode, lit
from pyspark.sql.types import (
    StructType, StructField, StringType, ArrayType, IntegerType, MapType
)
import numpy as np
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
import asyncio


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


def process_partition_chunks(partition):
    """Process a partition of patents and return chunks."""
    from langchain_huggingface import HuggingFaceEmbeddings
    
    # Initialize embedding model once per partition
    embedder = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    
    all_chunks = []
    for row in partition:
        # Handle both dict and Row objects
        if hasattr(row, 'asDict'):
            patent_data = row.asDict()
        elif isinstance(row, dict):
            patent_data = row
        else:
            patent_data = dict(row)
        
        chunks = create_chunks_for_patent(patent_data)
        all_chunks.extend(chunks)
    
    # Generate embeddings for all chunks in this partition
    chunk_texts = [chunk["chunk_text"] for chunk in all_chunks]
    if chunk_texts:
        embeddings = embedder.embed_documents(chunk_texts)
        
        # Attach embeddings to chunks
        for chunk_idx, (chunk, embedding) in enumerate(zip(all_chunks, embeddings)):
            chunk["embedding"] = embedding
            chunk["chunk_id"] = f"{chunk['patent_id']}:{chunk['chunk_type']}:{chunk_idx}"
            chunk["point_id"] = hash(chunk["chunk_id"]) % (2**63)
    
    return all_chunks


def ingest_patents_pyspark(
    jsonl_path: str,
    num_partitions: int = None,
    qdrant_batch_size: int = 1000,
    delete_existing: bool = True,
):
    """Ingest patents using PySpark for parallel processing.
    
    Args:
        jsonl_path: Path to JSONL file
        num_partitions: Number of Spark partitions (default: auto-detect)
        qdrant_batch_size: Batch size for Qdrant uploads
        delete_existing: Whether to delete existing collection
    """
    print("="*60)
    print("🚀 PySpark Parallel Patent Ingestion")
    print("="*60)
    
    # Initialize Spark
    try:
        spark = SparkSession.builder \
            .appName("PatentIngestion") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .getOrCreate()
    except Exception as e:
        if "JAVA_HOME" in str(e) or "Java" in str(e) or "java" in str(e).lower():
            print("\n" + "="*60)
            print("❌ ERROR: Java is required for PySpark but not found!")
            print("="*60)
            print("\nOptions:")
            print("1. Install Java and set JAVA_HOME:")
            print("   sudo apt-get install openjdk-11-jdk")
            print("   export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64")
            print("\n2. Use multiprocessing version instead (no Java required):")
            print("   python scripts/ingest_patents_multiprocessing.py data/patents_bigquery.jsonl")
            print("="*60)
        raise
    
    # Set log level to reduce noise
    spark.sparkContext.setLogLevel("WARN")
    
    print(f"\n📖 Loading patents from {jsonl_path}...")
    
    # Read JSONL file
    # PySpark doesn't have native JSONL support, so we'll read as text and parse
    lines_rdd = spark.sparkContext.textFile(jsonl_path)
    
    # Parse JSON lines
    def parse_json_line(line):
        try:
            return json.loads(line.strip())
        except:
            return None
    
    patents_rdd = lines_rdd.map(parse_json_line).filter(lambda x: x is not None)
    
    # Get total count
    total_patents = patents_rdd.count()
    print(f"✅ Loaded {total_patents:,} patents")
    
    # Determine number of partitions
    if num_partitions is None:
        # Use 2-4 partitions per CPU core for optimal parallelism
        num_cores = spark.sparkContext.defaultParallelism
        num_partitions = max(num_cores * 2, 8)  # At least 8 partitions
    
    print(f"📊 Using {num_partitions} partitions for parallel processing")
    
    # Repartition for parallel processing
    patents_rdd = patents_rdd.repartition(num_partitions)
    
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
    
    # Process patents in parallel and generate chunks with embeddings
    print(f"\n⚙️  Processing patents in parallel...")
    print(f"   This may take a while - embeddings are being generated...")
    
    processing_start = datetime.now()
    
    # Process partitions in parallel
    chunks_rdd = patents_rdd.mapPartitions(process_partition_chunks)
    
    # Collect all chunks (this triggers the parallel processing)
    # For very large datasets, we might want to process in batches
    # but for now, we'll collect all chunks
    print("   Collecting processed chunks...")
    all_chunks = chunks_rdd.collect()
    
    processing_time = (datetime.now() - processing_start).total_seconds()
    print(f"✅ Processed {len(all_chunks):,} chunks in {processing_time:.1f} seconds")
    if total_patents > 0:
        print(f"   Average: {len(all_chunks)/total_patents:.1f} chunks per patent")
    
    # Upload to Qdrant in batches
    print(f"\n📤 Uploading to Qdrant in batches of {qdrant_batch_size}...")
    start_time = datetime.now()
    
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
    
    upload_time = (datetime.now() - start_time).total_seconds()
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
    
    spark.stop()


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
    
    parser = argparse.ArgumentParser(description="Ingest patents using PySpark")
    parser.add_argument("jsonl_path", help="Path to JSONL file")
    parser.add_argument("--partitions", type=int, default=None, help="Number of Spark partitions (auto if not specified)")
    parser.add_argument("--batch-size", type=int, default=1000, help="Qdrant batch size")
    parser.add_argument("--keep-existing", action="store_true", help="Keep existing collection (don't delete)")
    
    args = parser.parse_args()
    
    ingest_patents_pyspark(
        jsonl_path=args.jsonl_path,
        num_partitions=args.partitions,
        qdrant_batch_size=args.batch_size,
        delete_existing=not args.keep_existing,
    )

