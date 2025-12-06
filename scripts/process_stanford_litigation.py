#!/usr/bin/env python3
"""
Process Stanford NPE Litigation Dataset CSV into JSONL format.

This script converts the Stanford CSV file into JSONL format compatible
with our litigation ingestion pipeline.

Usage:
    python scripts/process_stanford_litigation.py \
        --input data/cases-2025-12-05PST01-24-57.csv \
        --output data/stanford_litigation.jsonl
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def normalize_patent_id(patent_str: str) -> Optional[str]:
    """Extract and normalize patent ID from string."""
    if not patent_str or patent_str == "None" or patent_str.strip() == "":
        return None
    
    # Try to extract patent number (could be just a number or US format)
    patent_str = str(patent_str).strip()
    
    # If it's just a number, assume US patent
    if patent_str.isdigit():
        return f"US{patent_str}"
    
    # Try to find patent pattern
    match = re.search(r'\b(US|EP|WO|JP|CN|KR|GB|DE|FR)?[- ]?(\d+[A-Z]?\d*)\b', patent_str, re.IGNORECASE)
    if match:
        country = match.group(1) or "US"
        number = match.group(2)
        return f"{country.upper()}-{number}"
    
    return None


def parse_patents_field(patents_str: str) -> List[str]:
    """Parse patents field which may contain multiple patent numbers."""
    if not patents_str or patents_str.strip() == "":
        return []
    
    # Split by common delimiters
    parts = re.split(r'[,;]', patents_str)
    patent_ids = []
    
    for part in parts:
        patent_id = normalize_patent_id(part.strip())
        if patent_id:
            patent_ids.append(patent_id)
    
    return patent_ids


def parse_date(date_str: str) -> Optional[str]:
    """Parse date string to YYYY-MM-DD format."""
    if not date_str or date_str.strip() == "":
        return None
    
    # Try common date formats
    date_str = date_str.strip()
    
    # YYYY-MM-DD format
    if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        return date_str
    
    # Try other formats
    try:
        from datetime import datetime
        # Try parsing various formats
        for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d']:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                continue
    except Exception:
        pass
    
    return None


def determine_case_status(filing_date: Optional[str], closed_date: Optional[str]) -> str:
    """Determine case status from dates."""
    if closed_date:
        return "closed"
    if filing_date:
        return "active"
    return "unknown"


def process_csv_row(row: Dict[str, str]) -> List[Dict]:
    """Process a CSV row and return list of litigation records (one per patent)."""
    case_number = row.get("Civil Action #", "").strip()
    case_name = row.get("Case Title", "").strip()
    venue = row.get("Venue", "").strip()
    filing_date_str = row.get("Filing Date", "").strip()
    patents_str = row.get("patents", "").strip()
    plaintiff = row.get("Patent Asserter", "").strip()
    defendant = row.get("Alleged Infringer", "").strip()
    
    # Parse dates
    filing_date = parse_date(filing_date_str)
    
    # Parse patents
    patent_ids = parse_patents_field(patents_str)
    
    # If no patents found, still create one record with None patent_id
    if not patent_ids:
        patent_ids = [None]
    
    # Create one record per patent
    records = []
    for patent_id in patent_ids:
        record = {
            "case_number": case_number,
            "case_name": case_name,
            "court_name": venue,
            "filing_date": filing_date,
            "case_status": determine_case_status(filing_date, None),
            "plaintiff_name": plaintiff,
            "defendant_name": defendant,
            "patent_id": patent_id,
            "source": "stanford_npe",
        }
        records.append(record)
    
    return records


def main():
    parser = argparse.ArgumentParser(description="Process Stanford NPE Litigation CSV")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input CSV file path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/stanford_litigation.jsonl",
        help="Output JSONL file path",
    )
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    total_records = 0
    
    print(f"Processing Stanford litigation CSV: {input_path}")
    
    with input_path.open("r", encoding="utf-8") as csvfile, \
         output_path.open("w", encoding="utf-8") as jsonlfile:
        
        # Try to detect delimiter
        sample = csvfile.read(1024)
        csvfile.seek(0)
        sniffer = csv.Sniffer()
        delimiter = sniffer.sniff(sample).delimiter
        
        reader = csv.DictReader(csvfile, delimiter=delimiter)
        
        for row_idx, row in enumerate(reader, start=1):
            try:
                records = process_csv_row(row)
                for record in records:
                    jsonlfile.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_records += 1
                
                if row_idx % 1000 == 0:
                    print(f"  Processed {row_idx} rows, {total_records} records...")
            except Exception as e:
                print(f"Warning: Error processing row {row_idx}: {e}")
                continue
    
    print(f"\n✓ Processed {total_records} litigation records")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
