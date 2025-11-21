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
    chunks = proc.chunk_patent(record, chunks_per_patent=5)
    assert [c.chunk_type for c in chunks] == ["title", "abstract", "claim_1", "claim_2", "claim_3"]


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

