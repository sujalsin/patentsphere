#!/usr/bin/env python3
"""
Process USPTO OCE Litigation Dataset from CSV zip file.

This script extracts and processes the USPTO litigation CSV files from csv.zip
and converts them into JSONL format compatible with our ingestion pipeline.

The zip file contains:
- cases.csv: Case information
- patents.csv: Patent information linked to cases
- names.csv: Party names (plaintiffs/defendants)

Usage:
    python scripts/process_uspto_litigation.py \
        --zip data/csv.zip \
        --output data/uspto_litigation.jsonl
"""

import argparse
import csv
import io
import json
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def normalize_patent_id(patent_str: str) -> Optional[str]:
    """Normalize patent ID to standard format."""
    if not patent_str or patent_str.strip() == "" or patent_str == "None":
        return None
    
    patent_str = str(patent_str).strip().upper()
    
    # Remove common separators
    patent_str = patent_str.replace(" ", "").replace("-", "")
    
    # If it starts with country code, keep it; otherwise assume US
    if re.match(r'^(US|EP|WO|JP|CN|KR|GB|DE|FR)', patent_str):
        country = patent_str[:2]
        number = patent_str[2:]
    else:
        # Assume US if no country code
        country = "US"
        number = patent_str
    
    # Format as US-1234567
    if number:
        return f"{country}-{number}"
    
    return None


def parse_date(date_str: str) -> Optional[str]:
    """Parse date string to YYYY-MM-DD format."""
    if not date_str or date_str.strip() == "":
        return None
    
    date_str = date_str.strip()
    
    # Try YYYY-MM-DD format first
    if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        return date_str
    
    # Try other common formats
    try:
        from datetime import datetime
        for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d', '%Y%m%d']:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                continue
    except Exception:
        pass
    
    return None


def determine_case_status(filing_date: Optional[str], closed_date: Optional[str], settlement: Optional[str]) -> str:
    """Determine case status from dates and settlement info."""
    if closed_date:
        return "closed"
    if settlement:
        return "settled"
    if filing_date:
        return "active"
    return "unknown"


def load_cases(zip_path: Path) -> Dict[str, Dict]:
    """Load cases from cases.csv in zip file."""
    cases = {}
    
    print("Loading cases from cases.csv...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        with zf.open('cases.csv', 'r') as f:
            # Wrap in TextIOWrapper for proper text handling
            text_file = io.TextIOWrapper(f, encoding='utf-8')
            reader = csv.DictReader(text_file)
            for row in reader:
                case_row_id = row.get('case_row_id', '').strip()
                if not case_row_id:
                    continue
                
                cases[case_row_id] = {
                    'case_row_id': case_row_id,
                    'case_number': row.get('case_number', '').strip(),
                    'case_name': row.get('case_name', '').strip(),
                    'court_name': row.get('court_name', '').strip(),
                    'date_filed': parse_date(row.get('date_filed', '')),
                    'date_closed': parse_date(row.get('date_closed', '')),
                    'settlement': row.get('settlement', '').strip() or None,
                }
    
    print(f"  Loaded {len(cases)} cases")
    return cases


def load_patents(zip_path: Path) -> Dict[str, List[str]]:
    """Load patent IDs grouped by case_row_id from patents.csv."""
    case_patents = defaultdict(list)
    
    print("Loading patents from patents.csv...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        with zf.open('patents.csv', 'r') as f:
            # Wrap in TextIOWrapper for proper text handling
            text_file = io.TextIOWrapper(f, encoding='utf-8')
            reader = csv.DictReader(text_file)
            for row in reader:
                case_row_id = row.get('case_row_id', '').strip()
                # Try different column names for patent number
                patent_number = (
                    row.get('patent_number', '').strip() or 
                    row.get('patent', '').strip()
                )
                country_code = row.get('country_code', 'US').strip().upper() or 'US'
                
                if not case_row_id or not patent_number:
                    continue
                
                # Normalize patent ID
                patent_id = normalize_patent_id(f"{country_code}{patent_number}")
                if patent_id:
                    case_patents[case_row_id].append(patent_id)
    
    print(f"  Loaded patents for {len(case_patents)} cases")
    return dict(case_patents)


def load_names(zip_path: Path) -> Dict[str, Dict[str, List[str]]]:
    """Load party names grouped by case_row_id from names.csv."""
    case_names = defaultdict(lambda: {'plaintiffs': [], 'defendants': []})
    
    print("Loading party names from names.csv...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        with zf.open('names.csv', 'r') as f:
            # Wrap in TextIOWrapper for proper text handling
            text_file = io.TextIOWrapper(f, encoding='utf-8')
            reader = csv.DictReader(text_file)
            for row in reader:
                case_row_id = row.get('case_row_id', '').strip() if row.get('case_row_id') else ''
                party_type = (row.get('party_type') or '').strip().lower()
                name = (row.get('name') or '').strip()
                
                if not case_row_id or not name:
                    continue
                
                if 'plaintiff' in party_type:
                    case_names[case_row_id]['plaintiffs'].append(name)
                elif 'defendant' in party_type:
                    case_names[case_row_id]['defendants'].append(name)
    
    print(f"  Loaded names for {len(case_names)} cases")
    return dict(case_names)


def process_litigation_data(
    cases: Dict[str, Dict],
    case_patents: Dict[str, List[str]],
    case_names: Dict[str, Dict[str, List[str]]],
    output_path: Path,
    limit: Optional[int] = None,
) -> int:
    """Process all litigation data and write to JSONL."""
    total_records = 0
    
    print("\nProcessing litigation records...")
    
    with output_path.open('w', encoding='utf-8') as f:
        for idx, (case_row_id, case) in enumerate(cases.items(), start=1):
            if limit and idx > limit:
                break
            
            # Get patents for this case
            patent_ids = case_patents.get(case_row_id, [])
            
            # Get party names
            names = case_names.get(case_row_id, {'plaintiffs': [], 'defendants': []})
            plaintiff_name = '; '.join(names['plaintiffs']) if names['plaintiffs'] else None
            defendant_name = '; '.join(names['defendants']) if names['defendants'] else None
            
            # Create one record per patent (or one record if no patents)
            if patent_ids:
                for patent_id in patent_ids:
                    record = {
                        'case_number': case['case_number'],
                        'case_name': case['case_name'],
                        'court_name': case['court_name'],
                        'filing_date': case['date_filed'],
                        'case_status': determine_case_status(
                            case['date_filed'],
                            case['date_closed'],
                            case['settlement']
                        ),
                        'outcome': case['settlement'],
                        'plaintiff_name': plaintiff_name,
                        'defendant_name': defendant_name,
                        'patent_id': patent_id,
                        'source': 'uspto_oce',
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
                    total_records += 1
            else:
                # No patent ID, still create record
                record = {
                    'case_number': case['case_number'],
                    'case_name': case['case_name'],
                    'court_name': case['court_name'],
                    'filing_date': case['date_filed'],
                    'case_status': determine_case_status(
                        case['date_filed'],
                        case['date_closed'],
                        case['settlement']
                    ),
                    'outcome': case['settlement'],
                    'plaintiff_name': plaintiff_name,
                    'defendant_name': defendant_name,
                    'patent_id': None,
                    'source': 'uspto_oce',
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                total_records += 1
            
            if idx % 1000 == 0:
                print(f"  Processed {idx} cases, {total_records} records...")
    
    return total_records


def main():
    parser = argparse.ArgumentParser(description="Process USPTO litigation CSV zip file")
    parser.add_argument(
        '--zip',
        type=str,
        default='data/csv.zip',
        help='Path to csv.zip file',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/uspto_litigation.jsonl',
        help='Output JSONL file path',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of cases to process (for testing)',
    )
    args = parser.parse_args()
    
    zip_path = Path(args.zip)
    if not zip_path.exists():
        print(f"Error: Zip file not found: {zip_path}")
        sys.exit(1)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing USPTO litigation data from: {zip_path}")
    print(f"Output: {output_path}")
    
    # Load all data
    cases = load_cases(zip_path)
    case_patents = load_patents(zip_path)
    case_names = load_names(zip_path)
    
    # Process and write
    total_records = process_litigation_data(
        cases,
        case_patents,
        case_names,
        output_path,
        limit=args.limit,
    )
    
    print(f"\n✓ Processed {total_records} litigation records")
    print(f"  Output: {output_path}")


if __name__ == '__main__':
    main()
