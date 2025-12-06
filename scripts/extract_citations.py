#!/usr/bin/env python3
"""
Extract citation relationships from patent JSONL files.

This script parses patent JSONL files and extracts citation relationships,
normalizing patent IDs and writing citation pairs to a JSONL output file.

Usage:
    python scripts/extract_citations.py \
        --input data/patents_10k.jsonl \
        --output data/citations.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def normalize_patent_id(patent_id: str) -> str | None:
    """
    Normalize patent ID to a consistent format.
    
    Handles formats like:
    - US-123456-A1
    - US-123456-B2
    - US123456A1
    - US123456
    """
    if not patent_id or not isinstance(patent_id, str):
        return None
    
    # Remove whitespace
    patent_id = patent_id.strip()
    if not patent_id:
        return None
    
    # Remove common prefixes/suffixes
    patent_id = patent_id.upper()
    
    # Handle formats with dashes: US-123456-A1
    if "-" in patent_id:
        parts = patent_id.split("-")
        if len(parts) >= 2:
            country = parts[0]
            number = parts[1]
            # Keep the format as-is for now
            return patent_id
    
    # Handle formats without dashes: US123456A1
    # Try to extract country code (2-3 letters) and number
    if len(patent_id) >= 2:
        # Assume first 2-3 chars are country code
        if patent_id[:2].isalpha():
            return patent_id
    
    return patent_id


def extract_citations_from_patent(patent_record: Dict) -> List[Tuple[str, str, str]]:
    """
    Extract citation pairs from a single patent record.
    
    Returns list of (citing_patent, cited_patent, citation_type) tuples.
    """
    citations = []
    
    # Get the citing patent ID
    citing_patent = patent_record.get("publication_number") or patent_record.get("id")
    if not citing_patent:
        return citations
    
    citing_patent = normalize_patent_id(citing_patent)
    if not citing_patent:
        return citations
    
    # Get citations array
    citations_data = patent_record.get("citations", [])
    if not citations_data:
        return citations
    
    # Handle both string (JSON) and list formats
    if isinstance(citations_data, str):
        try:
            citations_data = json.loads(citations_data)
        except json.JSONDecodeError:
            return citations
    
    if not isinstance(citations_data, list):
        return citations
    
    # Extract each citation
    for citation in citations_data:
        if not isinstance(citation, dict):
            continue
        
        # Get cited patent ID
        cited_patent = citation.get("publication_number")
        if not cited_patent:
            continue
        
        cited_patent = normalize_patent_id(cited_patent)
        if not cited_patent:
            continue
        
        # Skip self-citations
        if citing_patent == cited_patent:
            continue
        
        # Get citation type (category field)
        citation_type = citation.get("category") or citation.get("type") or "UNKNOWN"
        
        citations.append((citing_patent, cited_patent, citation_type))
    
    return citations


def extract_citations(input_path: Path, output_path: Path) -> None:
    """
    Extract all citations from input JSONL and write to output JSONL.
    """
    all_citations = []
    total_patents = 0
    patents_with_citations = 0
    
    print(f"Reading patents from: {input_path}")
    
    with input_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                patent = json.loads(line)
                total_patents += 1
                
                citations = extract_citations_from_patent(patent)
                if citations:
                    patents_with_citations += 1
                    all_citations.extend(citations)
                
                if line_num % 1000 == 0:
                    print(f"  Processed {line_num} patents, found {len(all_citations)} citations...")
            
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed line {line_num}: {e}", file=sys.stderr)
                continue
            except Exception as e:
                print(f"Warning: Error processing line {line_num}: {e}", file=sys.stderr)
                continue
    
    print(f"\nExtracted {len(all_citations)} citations from {patents_with_citations}/{total_patents} patents")
    
    # Write citations to output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing citations to: {output_path}")
    
    with output_path.open("w", encoding="utf-8") as f:
        for citing, cited, ctype in all_citations:
            citation_record = {
                "citing_patent": citing,
                "cited_patent": cited,
                "type": ctype,
            }
            f.write(json.dumps(citation_record, ensure_ascii=False) + "\n")
    
    print(f"✓ Wrote {len(all_citations)} citation pairs to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract citation relationships from patent JSONL files")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input patent JSONL file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output citations JSONL file",
    )
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    extract_citations(input_path, output_path)


if __name__ == "__main__":
    main()

