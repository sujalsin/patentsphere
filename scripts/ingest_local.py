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
    parser.add_argument(
        "--citations",
        type=str,
        default=None,
        help="Path to citations JSONL file (optional).",
    )
    parser.add_argument(
        "--litigation",
        type=str,
        default=None,
        help="Path to litigation JSONL file (optional).",
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
        # Create patent_citations table
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS patent_citations (
                id SERIAL PRIMARY KEY,
                citing_patent_id TEXT NOT NULL,
                cited_patent_id TEXT NOT NULL,
                citation_type TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE(citing_patent_id, cited_patent_id)
            )
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_citing_patent ON patent_citations(citing_patent_id)
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_cited_patent ON patent_citations(cited_patent_id)
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


def load_citation_graph(citations_path: Path) -> List[Tuple[str, str, str]]:
    """Load citation pairs from JSONL file."""
    citations = []
    with citations_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                citing = record.get("citing_patent")
                cited = record.get("cited_patent")
                ctype = record.get("type", "UNKNOWN")
                if citing and cited:
                    citations.append((citing, cited, ctype))
            except json.JSONDecodeError:
                continue
    return citations


def upsert_citations(conn, citations: List[Tuple[str, str, str]]) -> None:
    """Bulk insert citation pairs into patent_citations table."""
    if not citations:
        return
    
    with conn.cursor() as cur:
        # Use executemany for bulk insert
        cur.executemany(
            """
            INSERT INTO patent_citations (citing_patent_id, cited_patent_id, citation_type)
            VALUES (%s, %s, %s)
            ON CONFLICT (citing_patent_id, cited_patent_id) DO NOTHING
            """,
            citations,
        )
        conn.commit()


def ensure_rl_tables(conn) -> None:
    """Create RL experiences table for logging rewards."""
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS rl_experiences (
                id SERIAL PRIMARY KEY,
                query_text TEXT NOT NULL,
                query_type TEXT,
                retrieved_patent_ids TEXT[],
                retrieved_chunks JSONB,
                total_reward DOUBLE PRECISION,
                reward_components JSONB,
                agent_outputs JSONB,
                run_id UUID,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS adaptive_retrieval_events (
                id SERIAL PRIMARY KEY,
                run_id UUID NOT NULL,
                query_text TEXT NOT NULL,
                query_type TEXT,
                iteration INT NOT NULL,
                action TEXT NOT NULL,
                state JSONB,
                chunk_ids TEXT[],
                total_chunks INT,
                chunk_quality DOUBLE PRECISION,
                exploration_rate DOUBLE PRECISION,
                metadata JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
            """
        )
        cur.execute(
            """
            ALTER TABLE rl_experiences
            ADD COLUMN IF NOT EXISTS retrieved_chunks JSONB
            """
        )
        cur.execute(
            """
            ALTER TABLE rl_experiences
            ADD COLUMN IF NOT EXISTS run_id UUID
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_query_type ON rl_experiences(query_type)
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_created_at ON rl_experiences(created_at)
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_adaptive_events_run
            ON adaptive_retrieval_events(run_id)
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_adaptive_events_created
            ON adaptive_retrieval_events(created_at)
            """
        )
        conn.commit()


def ensure_litigation_tables(conn) -> None:
    """Create patent_litigation table for storing litigation data."""
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS patent_litigation (
                id SERIAL PRIMARY KEY,
                patent_id TEXT,
                case_number TEXT NOT NULL,
                court_name TEXT,
                filing_date DATE,
                case_status TEXT,
                outcome TEXT,
                plaintiff_name TEXT,
                defendant_name TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE(patent_id, case_number)
            )
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_patent_litigation_patent ON patent_litigation(patent_id)
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_patent_litigation_case ON patent_litigation(case_number)
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_patent_litigation_date ON patent_litigation(filing_date)
            """
        )
        conn.commit()


def load_litigation_data(litigation_path: Path) -> List[Dict]:
    """Load litigation records from JSONL file."""
    records = []
    with litigation_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError:
                continue
    return records


def upsert_litigation(conn, litigation_records: List[Dict]) -> None:
    """Bulk insert litigation records into patent_litigation table."""
    if not litigation_records:
        return
    
    processed_records = []
    for record in litigation_records:
        # Normalize patent_id - try multiple sources
        patent_id = record.get("patent_id") or record.get("patent_number", "")
        
        # If no patent_id, try to extract from case_name
        if not patent_id or patent_id == "None":
            case_name = record.get("case_name", "")
            if case_name:
                import re
                # Try to find patent numbers in case name
                found_patents = re.findall(
                    r'\b(US|EP|WO|JP|CN|KR|GB|DE|FR)[- ]?(\d+)[A-Z]?(\d*)\b',
                    case_name,
                    re.IGNORECASE
                )
                if found_patents:
                    # Use first found patent
                    country, num1, num2 = found_patents[0]
                    patent_id = f"{country.upper()}-{num1}{num2}"
        
        # Allow NULL patent_id - we can still store cases and match them later
        # Set to None if not found
        if not patent_id or patent_id == "None":
            patent_id = None
        
        # Convert filing_date if needed
        filing_date = record.get("filing_date")
        if filing_date and isinstance(filing_date, str):
            # Try to parse date string
            try:
                from datetime import datetime
                filing_date = datetime.strptime(filing_date[:10], "%Y-%m-%d").date()
            except (ValueError, TypeError):
                filing_date = None
        
        processed_records.append({
            "patent_id": patent_id,
            "case_number": record.get("case_number", ""),
            "court_name": record.get("court_name"),
            "filing_date": filing_date,
            "case_status": record.get("case_status"),
            "outcome": record.get("outcome"),
            "plaintiff_name": record.get("plaintiff_name"),
            "defendant_name": record.get("defendant_name"),
        })
    
    if not processed_records:
        return
    
    with conn.cursor() as cur:
            # Use case_number as unique identifier (patent_id can be NULL)
            cur.executemany(
                """
                INSERT INTO patent_litigation (
                    patent_id, case_number, court_name, filing_date,
                    case_status, outcome, plaintiff_name, defendant_name
                )
                VALUES (%(patent_id)s, %(case_number)s, %(court_name)s, %(filing_date)s,
                        %(case_status)s, %(outcome)s, %(plaintiff_name)s, %(defendant_name)s)
                ON CONFLICT (case_number) DO UPDATE SET
                    patent_id = EXCLUDED.patent_id,
                    court_name = EXCLUDED.court_name,
                    filing_date = EXCLUDED.filing_date,
                    case_status = EXCLUDED.case_status,
                    outcome = EXCLUDED.outcome,
                    plaintiff_name = EXCLUDED.plaintiff_name,
                    defendant_name = EXCLUDED.defendant_name
                """,
                processed_records,
            )
            conn.commit()


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
    ensure_rl_tables(conn)
    ensure_litigation_tables(conn)
    upsert_postgres(conn, chunks)
    
    # Load citations if provided
    if args.citations:
        citations_path = Path(args.citations)
        if citations_path.exists():
            print(f"Loading citations from: {citations_path}")
            citations = load_citation_graph(citations_path)
            if citations:
                upsert_citations(conn, citations)
                print(f"✓ Loaded {len(citations)} citation pairs")
            else:
                print("No citations found in file")
        else:
            print(f"Warning: Citations file not found: {citations_path}")
    
    # Load litigation if provided
    if args.litigation:
        litigation_path = Path(args.litigation)
        if litigation_path.exists():
            print(f"Loading litigation data from: {litigation_path}")
            litigation_records = load_litigation_data(litigation_path)
            if litigation_records:
                upsert_litigation(conn, litigation_records)
                print(f"✓ Loaded {len(litigation_records)} litigation records")
            else:
                print("No litigation records found in file")
        else:
            print(f"Warning: Litigation file not found: {litigation_path}")
    
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

