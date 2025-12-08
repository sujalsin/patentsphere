#!/usr/bin/env python3
"""Convert USPTO litigation CSV to JSONL format."""
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from collections import defaultdict

def convert_csv_to_jsonl(cases_csv_path: str, patents_csv_path: str, output_path: str):
    """Convert USPTO cases and patents CSV to JSONL format."""
    print(f"Converting USPTO litigation data to {output_path}...")
    print(f"  Cases CSV: {cases_csv_path}")
    print(f"  Patents CSV: {patents_csv_path}\n")
    
    # First, load patents by case_number
    print("📖 Loading patents data...")
    patents_by_case = defaultdict(list)
    with open(patents_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_number = row.get("case_number", "").strip()
            patent = row.get("patent", "").strip()
            if case_number and patent and patent.isdigit():
                # Format as US patent ID
                patent_id = f"US{patent}"
                patents_by_case[case_number].append(patent_id)
    
    print(f"  ✅ Loaded patents for {len(patents_by_case):,} cases")
    
    # Now process cases
    print("\n📖 Processing cases...")
    total_rows = 0
    converted = 0
    
    with open(cases_csv_path, 'r', encoding='utf-8') as csvfile, \
         open(output_path, 'w', encoding='utf-8') as jsonlfile:
        
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            total_rows += 1
            
            case_number = row.get("case_number", "").strip()
            if not case_number:
                continue
            
            # Get patent IDs for this case
            patent_ids = patents_by_case.get(case_number, [])
            
            # Determine case status
            date_closed = row.get("date_closed", "").strip()
            case_status = "closed" if date_closed else "open"
            
            # Create JSON object matching the expected schema
            case_data = {
                "case_number": case_number,
                "case_name": row.get("case_name", "").strip() or None,
                "court_name": row.get("court_name", "").strip() or None,
                "filing_date": row.get("date_filed", "").strip() or None,
                "case_status": case_status,
                "plaintiff_name": None,  # Not in cases.csv, would need names.csv join
                "defendant_name": None,  # Not in cases.csv, would need names.csv join
                "patent_id": patent_ids[0] if patent_ids else None,
                "outcome": None,  # CSV doesn't have outcome
                "source": "uspto_oce"
            }
            
            # Write as JSONL - if no patents, write one entry with patent_id=None
            # If patents exist, write one entry per patent
            if patent_ids:
                for patent_id in patent_ids:
                    case_data_with_patent = case_data.copy()
                    case_data_with_patent["patent_id"] = patent_id
                    jsonlfile.write(json.dumps(case_data_with_patent, ensure_ascii=False) + "\n")
                    converted += 1
            else:
                # Case without patents - still include it
                jsonlfile.write(json.dumps(case_data, ensure_ascii=False) + "\n")
                converted += 1
            
            # Progress logging
            if total_rows % 10000 == 0:
                print(f"  Processed {total_rows:,} cases, converted {converted:,} entries...")
    
    print(f"\n✅ Conversion complete!")
    print(f"   Total cases processed: {total_rows:,}")
    print(f"   Entries converted: {converted:,}")
    print(f"   Cases with patents: {len([c for c in patents_by_case.values() if c]):,}")
    print(f"   Output file: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_uspto_csv_to_jsonl.py <cases_csv> <patents_csv> [output_path]")
        print("Example: python convert_uspto_csv_to_jsonl.py data/cases.csv data/patents.csv data/uspto_litigation.jsonl")
        sys.exit(1)
    
    cases_csv = sys.argv[1]
    patents_csv = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else "data/uspto_litigation.jsonl"
    
    if not Path(cases_csv).exists():
        print(f"Error: Cases CSV file not found: {cases_csv}")
        sys.exit(1)
    
    if not Path(patents_csv).exists():
        print(f"Error: Patents CSV file not found: {patents_csv}")
        sys.exit(1)
    
    convert_csv_to_jsonl(cases_csv, patents_csv, output_path)

