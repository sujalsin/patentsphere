#!/usr/bin/env python3
"""
PatentSphere preprocessing script.

Goals:
- Chunk raw Google Patents JSONL data into token-aware sections covering the
  entire patent (title, abstract, description, all claims, etc.).
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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.runtime_profile import get_runtime_profile  # type: ignore  # noqa
from config.settings import get_settings  # type: ignore  # noqa


DEFAULT_OUTPUT_DIR = Path("data/processed")


TEXTUAL_FIELD_CANDIDATES: Tuple[str, ...] = (
    "title",
    "abstract",
    "description",
    "background",
    "summary",
    "detailed_description",
    "claims",
    "drawings",
)

NON_TEXTUAL_FIELDS = {
    "publication_number",
    "filing_date",
    "application_number",
    "cpc_codes",
    "citations",
}


@dataclass
class Chunk:
    """Represents a single chunk ready for embedding + Qdrant ingestion."""

    chunk_id: str
    patent_id: str
    chunk_type: str
    text: str
    order: int
    publication_date: Optional[str] = None
    cpc_codes: Optional[List[str]] = None


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    rp = get_runtime_profile()
    chunking_cfg = getattr(getattr(settings, "data", None), "chunking", None)
    default_chunk_limit = getattr(chunking_cfg, "max_chunks_per_patent", 0) or 0
    default_chunk_size = getattr(chunking_cfg, "chunk_size", 2000)
    default_chunk_overlap = getattr(chunking_cfg, "overlap", 200)
    default_chunk_fields: Sequence[str] = getattr(
        chunking_cfg, "fields", TEXTUAL_FIELD_CANDIDATES
    )

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
        default=default_chunk_limit,
        help=(
            "Hard cap on chunks per patent (0 for unlimited). "
            "Use to guard against runaway long claims."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=default_chunk_size,
        help="Target number of tokens (word splits) per chunk.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=default_chunk_overlap,
        help="Token overlap between consecutive chunks.",
    )
    parser.add_argument(
        "--chunk-fields",
        type=str,
        default=",".join(default_chunk_fields),
        help=(
            "Comma-separated list of JSON fields to include (order matters). "
            "Available defaults: title, abstract, description, summary, claims, drawings."
        ),
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


def _normalize_text(value: str) -> str:
    """Collapse whitespace for more stable chunk sizing."""
    return " ".join(value.split())


def _split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into overlapping chunks measured in words."""
    words = text.split()
    if not words:
        return []

    chunk_size = max(chunk_size, 1)
    overlap = max(overlap, 0)
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than chunk_overlap.")

    chunks: List[str] = []
    start = 0
    total = len(words)
    while start < total:
        end = min(total, start + chunk_size)
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end >= total:
            break
        start = end - overlap
    return chunks


def _append_section(
    sections: List[Tuple[str, str]],
    field_name: str,
    value: object,
) -> None:
    """Recursively append textual content with normalized names."""

    if value is None:
        return

    if isinstance(value, str):
        normalized = _normalize_text(value)
        if normalized:
            sections.append((field_name, normalized))
        return

    if isinstance(value, list):
        for idx, item in enumerate(value, start=1):
            next_name = field_name if field_name.startswith("claim_") else f"{field_name}_{idx}"
            _append_section(sections, next_name, item)
        return

    if isinstance(value, dict):
        for sub_key, sub_val in value.items():
            next_name = f"{field_name}_{sub_key}".strip("_")
            _append_section(sections, next_name, sub_val)
        return

    # Fallback: coerce to string
    text_value = _normalize_text(str(value))
    if text_value:
        sections.append((field_name, text_value))


def _extract_sections(record: Dict, fields: Optional[Sequence[str]]) -> List[Tuple[str, str]]:
    """Return (section_name, text) tuples for the requested fields."""
    sections: List[Tuple[str, str]] = []
    prioritized_fields = list(fields or TEXTUAL_FIELD_CANDIDATES)

    for field in prioritized_fields:
        raw_value = record.get(field)
        if not raw_value:
            continue

        if field == "claims":
            if isinstance(raw_value, list):
                for idx, claim in enumerate(raw_value, start=1):
                    if not claim:
                        continue
                    _append_section(sections, f"claim_{idx}", claim)
            else:
                _append_section(sections, "claims", raw_value)
            continue

        _append_section(sections, field, raw_value)

    # Include any other textual fields we didn't explicitly request
    for key, value in record.items():
        if key in NON_TEXTUAL_FIELDS or key in prioritized_fields:
            continue
        _append_section(sections, key, value)

    return sections


def chunk_patent(
    record: Dict,
    chunk_size: int,
    chunk_overlap: int,
    max_chunks: Optional[int] = None,
    fields: Optional[Sequence[str]] = None,
) -> List[Chunk]:
    publication_number = record.get("publication_number") or record.get("id")
    if not publication_number:
        return []

    sections = _extract_sections(record, fields)
    if not sections:
        return []

    chunks: List[Chunk] = []
    order = 0

    for section_name, text in sections:
        split_sections = _split_text(text, chunk_size, chunk_overlap)
        for idx, chunk_text in enumerate(split_sections):
            if not chunk_text:
                continue
            chunk_type = (
                f"{section_name}_part{idx + 1}" if idx > 0 else section_name
            )
            chunk_id = f"{publication_number}:{chunk_type}:{order}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    patent_id=publication_number,
                    chunk_type=chunk_type,
                    text=chunk_text,
                    order=order,
                    publication_date=record.get("filing_date"),
                    cpc_codes=record.get("cpc_codes"),
                )
            )
            order += 1
            if max_chunks and order >= max_chunks:
                return chunks

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

    if args.chunk_size <= args.chunk_overlap:
        raise ValueError("chunk-size must be greater than chunk-overlap.")

    if isinstance(args.chunk_fields, str):
        chunk_fields = [f.strip() for f in args.chunk_fields.split(",") if f.strip()]
    else:
        chunk_fields = list(args.chunk_fields)

    max_chunks = args.chunks_per_patent or None

    chunks: List[Chunk] = []
    patents_processed = 0

    for record in iter_patents(input_path, args.max_patents):
        patent_chunks = chunk_patent(
            record,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            max_chunks=max_chunks,
            fields=chunk_fields or None,
        )
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

