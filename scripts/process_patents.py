#!/usr/bin/env python3
"""
PatentSphere preprocessing script.

Goals:
- Chunk raw Google Patents JSONL data into fixed units (title, abstract, claims).
- (Optionally) generate sentence-transformer embeddings for each chunk.
- Persist outputs locally so we can stay within the free tier until GCP is explicitly enabled.

Usage Examples:
    python scripts/process_patents.py \
        --input data/patents_50k.jsonl \
        --output-dir data/processed \
        --max-patents 50000

    # Generate embeddings (requires numpy + sentence-transformers installed)
    python scripts/process_patents.py --generate-embeddings

The script intentionally refuses to touch remote (gs://) paths unless you flip
the relevant resource-policy flags and env vars, keeping your $50 GCP credit safe.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.runtime_profile import get_runtime_profile  # type: ignore  # noqa
from config.settings import get_settings  # type: ignore  # noqa


DEFAULT_OUTPUT_DIR = Path("data/processed")


@dataclass
class Chunk:
    """Represents a single chunk ready for embedding + Qdrant ingestion."""

    chunk_id: str
    patent_id: str
    chunk_type: str  # title | abstract | claim
    text: str
    order: int
    publication_date: Optional[str] = None
    cpc_codes: Optional[List[str]] = None


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    rp = get_runtime_profile()

    parser = argparse.ArgumentParser(description="Process patent JSONL into chunks.")
    parser.add_argument(
        "--input",
        type=str,
        default=rp.dataset_path(),
        help="Path to the raw patents JSONL file (defaults to profile dataset).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to store chunked outputs and embeddings.",
    )
    parser.add_argument(
        "--max-patents",
        type=int,
        default=settings.evaluation.baseline_test_size * 100,
        help="Hard cap on processed patents (prevents accidental full-corpus runs).",
    )
    parser.add_argument(
        "--chunks-per-patent",
        type=int,
        default=settings.data.chunking.chunks_per_patent
        if hasattr(settings, "data")
        else 5,
        help="Maximum chunks per patent.",
    )
    parser.add_argument(
        "--generate-embeddings",
        action="store_true",
        help="Generate sentence-transformer embeddings (requires extra deps).",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=settings.embeddings.model_name,
        help="Model name for sentence-transformers.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for PyTorch embeddings (auto selects CUDA if available).",
    )
    parser.add_argument(
        "--embedding-num-workers",
        type=int,
        default=0,
        help="SentenceTransformer encode num_workers for CPU parallelism.",
    )
    parser.add_argument(
        "--use-pyspark",
        action="store_true",
        help="Use PySpark for distributed parallel embedding generation (faster for large datasets).",
    )
    parser.add_argument(
        "--spark-partitions",
        type=int,
        default=None,
        help="Number of Spark partitions (default: auto based on CPU count).",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=settings.embeddings.batch_size,
        help="Batch size for embedding generation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate without writing outputs (for quick sanity checks).",
    )

    return parser.parse_args()


def ensure_local_input(path_str: str) -> Path:
    path = Path(path_str)
    if str(path).startswith("gs://"):
        raise RuntimeError(
            "Remote gs:// paths are disabled for this script. "
            "Download the subset locally or explicitly enable cloud mode."
        )
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return path


def iter_patents(path: Path, max_patents: int) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_patents and idx >= max_patents:
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[WARN] Skipping malformed line {idx}: {exc}", file=sys.stderr)


def chunk_patent(
    record: Dict, chunks_per_patent: int, max_claims: int = 3
) -> List[Chunk]:
    publication_number = record.get("publication_number") or record.get("id")
    if not publication_number:
        return []

    chunks: List[Chunk] = []
    order = 0

    def add_chunk(text: Optional[str], chunk_type: str):
        nonlocal order
        if not text:
            return
        chunk_id = f"{publication_number}:{chunk_type}:{order}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                patent_id=publication_number,
                chunk_type=chunk_type,
                text=text.strip(),
                order=order,
                publication_date=record.get("filing_date"),
                cpc_codes=record.get("cpc_codes"),
            )
        )
        order += 1

    add_chunk(record.get("title"), "title")
    add_chunk(record.get("abstract"), "abstract")

    claims = record.get("claims") or []
    for idx, claim in enumerate(claims[: max_claims]):
        add_chunk(claim, f"claim_{idx+1}")

    if len(chunks) > chunks_per_patent:
        chunks = chunks[:chunks_per_patent]
    return chunks


def write_chunks(chunks: List[Chunk], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def _resolve_device(device_flag: str):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - handled upstream
        raise RuntimeError(
            "PyTorch is required for embedding generation. "
            "Install with `pip install torch sentence-transformers`."
        ) from exc

    if device_flag == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available on this machine.")
        return torch.device("cuda")

    if device_flag == "cpu":
        return torch.device("cpu")

    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_embeddings_pyspark(
    chunks: List[Chunk],
    model_name: str,
    batch_size: int,
    output_path: Path,
    device_flag: str,
    num_partitions: Optional[int],
) -> None:
    """Generate embeddings using PySpark for parallel processing."""
    import os
    import multiprocessing
    
    # Set Java options BEFORE importing Spark to avoid security manager issues
    # This works around Java 17+ security restrictions
    if "_JAVA_OPTIONS" not in os.environ:
        os.environ["_JAVA_OPTIONS"] = "-Djava.security.manager=allow"
    
    try:
        from pyspark.sql import SparkSession
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "PySpark mode requires 'pyspark'. Install with `pip install pyspark`. "
            "Also ensure 'torch' and 'sentence-transformers' are installed."
        ) from exc

    if num_partitions is None:
        num_partitions = max(1, multiprocessing.cpu_count())

    device = _resolve_device(device_flag)
    
    try:
        # Initialize Spark session with Java 17+ compatibility settings
        spark = SparkSession.builder \
            .appName("PatentEmbeddings") \
            .master("local[*]") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "2g") \
            .config("spark.driver.host", "localhost") \
            .config("spark.driver.bindAddress", "127.0.0.1") \
            .config("spark.sql.adaptive.enabled", "false") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "false") \
            .config("spark.ui.enabled", "false") \
            .getOrCreate()
        
        # Set log level to reduce warnings
        spark.sparkContext.setLogLevel("ERROR")
    except Exception as spark_error:
        # Fallback to standard multiprocessing if Spark fails
        error_msg = str(spark_error)
        if "UnsupportedOperationException" in error_msg or "security manager" in error_msg.lower():
            print(f"[WARN] PySpark failed due to Java security manager issue: {spark_error}")
            print("[INFO] Falling back to standard multiprocessing mode...")
            # Use the standard embedding function instead
            generate_embeddings(
                chunks, model_name, batch_size, output_path, device_flag,
                num_workers=num_partitions
            )
            return
        raise

    try:
        # Convert chunks to RDD of (chunk_id, text) tuples
        chunk_rdd = spark.sparkContext.parallelize(
            [(chunk.chunk_id, chunk.text) for chunk in chunks],
            numSlices=num_partitions
        )
        
        # Broadcast the model initialization function
        def encode_partition(partition):
            """Encode texts in this partition using a local model instance."""
            import torch
            from sentence_transformers import SentenceTransformer
            
            # Convert partition iterator to list
            partition_list = list(partition)
            if not partition_list:
                return
            
            texts = [text for _, text in partition_list]
            chunk_ids = [chunk_id for chunk_id, _ in partition_list]
            
            model = SentenceTransformer(model_name, device=device.type)
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                convert_to_tensor=False,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            
            # Yield (chunk_id, embedding) pairs as iterator
            embeddings_list = embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings
            for chunk_id, emb in zip(chunk_ids, embeddings_list):
                yield (chunk_id, emb)

        # Process each partition in parallel
        results_rdd = chunk_rdd.mapPartitions(encode_partition)
        results = results_rdd.collect()
        
        if not results:
            raise RuntimeError("No embeddings were generated. Check if chunks have valid text.")
        
        # Sort by chunk_id to maintain order and extract embeddings
        results_sorted = sorted(results, key=lambda x: x[0])
        embedding_list = [emb for _, emb in results_sorted]
        
        # Convert to tensor and save
        import numpy as np
        embedding_array = np.array(embedding_list, dtype=np.float32)
        embedding_tensor = torch.from_numpy(embedding_array)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(embedding_tensor, output_path)
        
    finally:
        spark.stop()


def generate_embeddings(
    chunks: List[Chunk],
    model_name: str,
    batch_size: int,
    output_path: Path,
    device_flag: str,
    num_workers: int,
) -> None:
    try:
        import torch
        # Test torch functionality - this will catch internal import errors
        _ = torch.zeros(1)
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Embedding generation requires 'torch' and 'sentence-transformers'. "
            "Install with `pip install torch sentence-transformers`."
        ) from exc
    except (ModuleNotFoundError, AttributeError, RuntimeError) as exc:
        error_str = str(exc).lower()
        if "torchgen" in error_str or ("torch" in error_str and "module" in error_str):
            raise RuntimeError(
                f"PyTorch installation issue detected: {exc}\n"
                "This usually means torch is partially installed. Try:\n"
                "  pip install --upgrade --force-reinstall torch\n"
                "Or if using conda:\n"
                "  conda install pytorch -c pytorch"
            ) from exc
        raise

    device = _resolve_device(device_flag)
    model = SentenceTransformer(model_name, device=device.type)

    texts = [chunk.text for chunk in chunks]
    if not texts:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        empty = torch.zeros(
            (0, model.get_sentence_embedding_dimension()), dtype=torch.float32
        )
        torch.save(empty, output_path)
        return

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
        device=device,
        normalize_embeddings=True,
        num_workers=num_workers,
    )

    array = embeddings.detach().cpu()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(array, output_path)


def write_stats(chunks: List[Chunk], output_path: Path, patents_processed: int) -> None:
    stats = {
        "patents_processed": patents_processed,
        "chunks_generated": len(chunks),
        "timestamp_utc": datetime.utcnow().isoformat(),
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def main() -> None:
    args = parse_args()

    input_path = ensure_local_input(args.input)
    output_dir = Path(args.output_dir)
    chunks_path = output_dir / "chunks.jsonl"
    embeddings_path = output_dir / "embeddings.pt"
    stats_path = output_dir / "stats.json"

    rp = get_runtime_profile()
    if not rp.is_local():
        # Fail fast unless user explicitly enables cloud mode via env vars.
        rp.ensure_bigquery_export_allowed()

    output_dir.mkdir(parents=True, exist_ok=True)

    chunks: List[Chunk] = []
    patents_processed = 0

    for record in iter_patents(input_path, args.max_patents):
        patent_chunks = chunk_patent(record, args.chunks_per_patent)
        if not patent_chunks:
            continue
        chunks.extend(patent_chunks)
        patents_processed += 1

        if patents_processed % 1000 == 0:
            print(f"...processed {patents_processed} patents ({len(chunks)} chunks)")

    if args.dry_run:
        print(
            f"[DRY RUN] Would write {len(chunks)} chunks for {patents_processed} patents."
        )
        return

    write_chunks(chunks, chunks_path)
    print(f"Wrote chunks to {chunks_path}")

    if args.generate_embeddings:
        if args.use_pyspark:
            print(f"Using PySpark with {args.spark_partitions or 'auto'} partitions for parallel processing...")
            generate_embeddings_pyspark(
                chunks,
                model_name=args.embedding_model,
                batch_size=args.embedding_batch_size,
                output_path=embeddings_path,
                device_flag=args.device,
                num_partitions=args.spark_partitions,
            )
        else:
            generate_embeddings(
                chunks,
                model_name=args.embedding_model,
                batch_size=args.embedding_batch_size,
                output_path=embeddings_path,
                device_flag=args.device,
                num_workers=args.embedding_num_workers,
            )
        print(f"Wrote embeddings to {embeddings_path}")

    write_stats(chunks, stats_path, patents_processed)
    print(f"Wrote stats to {stats_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

