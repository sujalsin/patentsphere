import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import process_patents as proc  # type: ignore  # noqa


DATA_DIR = Path(__file__).parent / "data"


def test_chunk_patent_builds_expected_sections():
    record = {
        "publication_number": "US123",
        "title": "Title text",
        "abstract": "Abstract text",
        "claims": ["Claim 1", "Claim 2", "Claim 3"],
        "filing_date": "2020-01-01",
        "cpc_codes": ["G06F"],
    }
    chunks = proc.chunk_patent(
        record,
        chunk_size=50,
        chunk_overlap=10,
        max_chunks=None,
        fields=["title", "abstract", "claims"],
    )
    assert [c.chunk_type for c in chunks] == [
        "title",
        "abstract",
        "claim_1",
        "claim_2",
        "claim_3",
    ]


def test_chunk_patent_splits_long_sections():
    record = {
        "publication_number": "US456",
        "title": "Short title",
        "abstract": " ".join(["abstract"] * 120),
        "claims": [" ".join(["claim"] * 60)],
    }
    chunks = proc.chunk_patent(
        record,
        chunk_size=40,
        chunk_overlap=10,
        max_chunks=None,
        fields=["abstract", "claims"],
    )
    abstract_chunks = [c for c in chunks if c.chunk_type.startswith("abstract")]
    assert len(abstract_chunks) == 4
    assert abstract_chunks[1].chunk_type == "abstract_part2"
    claim_chunks = [c for c in chunks if c.chunk_type.startswith("claim_1")]
    assert len(claim_chunks) == 2


def test_chunk_patent_handles_nested_fields():
    record = {
        "publication_number": "US789",
        "description": ["Paragraph one.", {"subsection": "Paragraph two."}],
        "summary": {"overview": "High level summary."},
        "extras": {"notes": ["Extra note A", "Extra note B"]},
    }
    chunks = proc.chunk_patent(
        record,
        chunk_size=200,
        chunk_overlap=20,
        max_chunks=None,
        fields=["description", "summary", "extras"],
    )
    chunk_types = {c.chunk_type for c in chunks}
    assert {"description_1", "description_2_subsection", "summary_overview"}.issubset(
        chunk_types
    )
    assert "extras_notes_1" in chunk_types


def test_iter_patents_limits_count(tmp_path):
    sample = DATA_DIR / "sample_patents.jsonl"
    records = list(proc.iter_patents(sample, max_patents=1))
    assert len(records) == 1


def test_cli_dry_run_executes():
    sample = DATA_DIR / "sample_patents.jsonl"
    result = subprocess.run(
        [
            "python",
            "scripts/process_patents.py",
            "--input",
            str(sample),
            "--output-dir",
            "data/test_cli",
            "--max-patents",
            "1",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Would write" in result.stdout

