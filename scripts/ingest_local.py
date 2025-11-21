#!/usr/bin/env python3
"""
Ingest locally processed chunks + embeddings into Postgres and Qdrant.

This script is intended for the Phase 1 setup where everything runs on the
developer laptop with dockerized Postgres & Qdrant. It refuses to run unless
the runtime profile is `local_dev`, preventing accidental cloud charges.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from config.runtime_profile import get_runtime_profile  # type: ignore  # noqa
from config.settings import get_settings  # type: ignore  # noqa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest chunks + embeddings locally.")
    parser.add_argument(
        "--chunks",
        type=str,
        default="data/processed/chunks.jsonl",
        help="Path to chunk metadata JSONL.",
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        default="data/processed/embeddings.pt",
        help="Path to embeddings tensor saved via torch.save.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for Qdrant upserts.",
    )
    return parser.parse_args()


def load_chunks(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def ensure_postgres_tables(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS patent_chunks (
                chunk_id TEXT PRIMARY KEY,
                patent_id TEXT,
                chunk_type TEXT,
                chunk_text TEXT,
                chunk_order INT,
                publication_date DATE,
                cpc_codes JSONB
            )
            """
        )
        conn.commit()


def convert_date(date_val):
    """Convert integer date (YYYYMMDD) or string to date format."""
    if date_val is None:
        return None
    if isinstance(date_val, int):
        # Convert YYYYMMDD integer to YYYY-MM-DD string
        date_str = str(date_val)
        if len(date_str) == 8:
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        elif len(date_str) == 6:
            # YYYYMM format - use first day of month
            return f"{date_str[:4]}-{date_str[4:6]}-01"
        else:
            # Invalid format, return None
            return None
    if isinstance(date_val, str):
        # Already a string, return as-is if it looks like a date
        if len(date_val) >= 10 and "-" in date_val:
            return date_val
        # Try to parse as integer string
        try:
            return convert_date(int(date_val))
        except ValueError:
            return None
    return None


def upsert_postgres(conn, chunks: List[Dict]) -> None:
    # Preprocess chunks to convert dates and JSON fields
    processed_chunks = []
    for chunk in chunks:
        processed = chunk.copy()
        processed["publication_date"] = convert_date(chunk.get("publication_date"))
        # Convert cpc_codes to Jsonb type for psycopg
        cpc = chunk.get("cpc_codes")
        if cpc is None:
            processed["cpc_codes"] = None
        elif isinstance(cpc, str):
            # Parse JSON string to dict/list, then wrap in Jsonb
            try:
                cpc_obj = json.loads(cpc)
                processed["cpc_codes"] = Jsonb(cpc_obj)
            except json.JSONDecodeError:
                processed["cpc_codes"] = None
        else:
            # Already a dict/list, wrap in Jsonb
            processed["cpc_codes"] = Jsonb(cpc)
        processed_chunks.append(processed)
    
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO patent_chunks (
                chunk_id, patent_id, chunk_type, chunk_text,
                chunk_order, publication_date, cpc_codes
            )
            VALUES (%(chunk_id)s, %(patent_id)s, %(chunk_type)s, %(text)s,
                    %(order)s, %(publication_date)s, %(cpc_codes)s)
            ON CONFLICT (chunk_id) DO UPDATE SET
                chunk_text = EXCLUDED.chunk_text,
                cpc_codes = EXCLUDED.cpc_codes
            """,
            processed_chunks,
        )
        conn.commit()


def ensure_qdrant_collection(client: QdrantClient, vector_size: int, collection: str) -> None:
    collections = [c.name for c in client.get_collections().collections]
    if collection in collections:
        return

    client.create_collection(
        collection_name=collection,
        vectors_config=qmodels.VectorParams(
            size=vector_size,
            distance=qmodels.Distance.COSINE,
        ),
    )


def chunk_id_to_int(chunk_id: str) -> int:
    """Convert chunk_id string to integer for Qdrant (using hash)."""
    # Use hash to convert string to integer, ensure positive
    return abs(hash(chunk_id)) % (2**63 - 1)  # Max int64


def upsert_qdrant(
    client: QdrantClient,
    collection: str,
    chunks: List[Dict],
    embeddings: torch.Tensor,
    batch_size: int,
) -> None:
    if embeddings.shape[0] != len(chunks):
        raise ValueError(
            f"Embedding count {embeddings.shape[0]} does not match chunks {len(chunks)}"
        )

    for start in range(0, len(chunks), batch_size):
        batch_chunks = chunks[start : start + batch_size]
        batch_vectors = embeddings[start : start + batch_size].cpu()
        points = []
        for idx, chunk in enumerate(batch_chunks):
            # Convert chunk_id to integer for Qdrant
            point_id = chunk_id_to_int(chunk["chunk_id"])
            points.append(
                qmodels.PointStruct(
                    id=point_id,
                    vector=batch_vectors[idx].tolist(),
                    payload={
                        "chunk_id": chunk["chunk_id"],  # Keep original in payload
                        "patent_id": chunk["patent_id"],
                        "chunk_type": chunk["chunk_type"],
                        "chunk_text": chunk["text"],
                        "chunk_order": chunk["order"],
                        "publication_date": chunk.get("publication_date"),
                        "cpc_codes": chunk.get("cpc_codes"),
                    },
                )
            )
        client.upsert(collection_name=collection, points=points)


def main() -> None:
    args = parse_args()
    rp = get_runtime_profile()
    if not rp.is_local():
        raise RuntimeError("Ingestion is limited to local_dev profile to avoid cloud costs.")

    settings = get_settings()
    chunks_path = Path(args.chunks)
    embeddings_path = Path(args.embeddings)

    chunks = load_chunks(chunks_path)
    embeddings = torch.load(embeddings_path, map_location="cpu")

    print(f"Loaded {len(chunks)} chunks and embeddings tensor {tuple(embeddings.shape)}")

    # Postgres - use connection string format
    pg_cfg = settings.database
    conn_str = f"postgresql://{pg_cfg.user}:{pg_cfg.password}@{pg_cfg.host}:{pg_cfg.port}/{pg_cfg.database}"
    conn = psycopg.connect(conn_str, row_factory=dict_row)
    ensure_postgres_tables(conn)
    upsert_postgres(conn, chunks)
    conn.close()
    print("Postgres ingestion complete.")

    # Qdrant - use HTTP for local connections
    q_cfg = settings.qdrant
    client = QdrantClient(
        host=q_cfg.host,
        port=q_cfg.port,
        api_key=q_cfg.api_key if q_cfg.api_key else None,
        https=False,  # Local Qdrant uses HTTP
    )
    ensure_qdrant_collection(client, vector_size=q_cfg.vector_size, collection=q_cfg.collection_name)
    upsert_qdrant(
        client,
        q_cfg.collection_name,
        chunks,
        embeddings,
        batch_size=args.batch_size,
    )
    print("Qdrant ingestion complete.")


if __name__ == "__main__":
    main()

