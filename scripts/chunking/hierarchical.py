"""Hierarchical chunking that preserves patent document structure.

This module extracts structured sections from patent documents and creates
parent chunks that represent major document components (title, abstract,
claims, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class SectionType(str, Enum):
    """Types of patent document sections."""
    TITLE = "title"
    ABSTRACT = "abstract"
    DESCRIPTION = "description"
    BACKGROUND = "background"
    SUMMARY = "summary"
    DETAILED_DESCRIPTION = "detailed_description"
    CLAIMS = "claims"
    CLAIM = "claim"  # Individual claim
    DRAWINGS = "drawings"
    OTHER = "other"


@dataclass
class PatentSection:
    """Represents a section of a patent document."""
    
    section_type: SectionType
    text: str
    order: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Hierarchical information
    parent_section: Optional[str] = None  # For claims under CLAIMS section
    section_number: Optional[int] = None  # For numbered claims
    
    @property
    def is_claim(self) -> bool:
        return self.section_type in (SectionType.CLAIMS, SectionType.CLAIM)
    
    @property
    def word_count(self) -> int:
        return len(self.text.split())


# Section importance weights for retrieval
SECTION_WEIGHTS = {
    SectionType.TITLE: 1.5,
    SectionType.ABSTRACT: 1.4,
    SectionType.CLAIMS: 1.3,
    SectionType.CLAIM: 1.3,
    SectionType.SUMMARY: 1.2,
    SectionType.DESCRIPTION: 1.0,
    SectionType.DETAILED_DESCRIPTION: 1.0,
    SectionType.BACKGROUND: 0.9,
    SectionType.DRAWINGS: 0.7,
    SectionType.OTHER: 0.8,
}


class HierarchicalChunker:
    """
    Extracts hierarchical structure from patent documents.
    
    Creates parent-level chunks that represent major document sections,
    preserving the semantic structure of patents.
    """
    
    # Field mappings from raw patent data to section types
    FIELD_TO_SECTION = {
        "title": SectionType.TITLE,
        "abstract": SectionType.ABSTRACT,
        "description": SectionType.DESCRIPTION,
        "background": SectionType.BACKGROUND,
        "summary": SectionType.SUMMARY,
        "detailed_description": SectionType.DETAILED_DESCRIPTION,
        "claims": SectionType.CLAIMS,
        "drawings": SectionType.DRAWINGS,
    }
    
    # Section extraction priority (order matters)
    SECTION_PRIORITY = [
        "title",
        "abstract",
        "summary",
        "background",
        "description",
        "detailed_description",
        "claims",
        "drawings",
    ]
    
    def __init__(
        self,
        min_section_words: int = 10,
        max_section_words: int = 5000,
        include_metadata: bool = True,
    ):
        """
        Initialize the hierarchical chunker.
        
        Args:
            min_section_words: Minimum words for a section to be included
            max_section_words: Maximum words before a section needs splitting
            include_metadata: Whether to include patent metadata in sections
        """
        self.min_section_words = min_section_words
        self.max_section_words = max_section_words
        self.include_metadata = include_metadata
    
    def extract_sections(self, patent_record: Dict[str, Any]) -> List[PatentSection]:
        """
        Extract hierarchical sections from a patent record.
        
        Args:
            patent_record: Raw patent data from JSONL
        
        Returns:
            List of PatentSection objects
        """
        sections: List[PatentSection] = []
        order = 0
        
        patent_id = patent_record.get("publication_number") or patent_record.get("id", "")
        base_metadata = {
            "patent_id": patent_id,
            "filing_date": patent_record.get("filing_date"),
            "cpc_codes": patent_record.get("cpc_codes", []),
        }
        
        # Extract sections in priority order
        for field_name in self.SECTION_PRIORITY:
            raw_value = patent_record.get(field_name)
            if not raw_value:
                continue
            
            section_type = self.FIELD_TO_SECTION.get(field_name, SectionType.OTHER)
            
            # Handle claims specially (they're often a list)
            if field_name == "claims":
                claim_sections = self._extract_claims(raw_value, order, base_metadata)
                sections.extend(claim_sections)
                order += len(claim_sections)
            else:
                text = self._normalize_text(raw_value)
                if text and len(text.split()) >= self.min_section_words:
                    sections.append(PatentSection(
                        section_type=section_type,
                        text=text,
                        order=order,
                        metadata=base_metadata.copy() if self.include_metadata else {},
                    ))
                    order += 1
        
        # Extract any other textual fields
        for key, value in patent_record.items():
            if key in self.FIELD_TO_SECTION:
                continue
            if key in ("publication_number", "id", "filing_date", "cpc_codes", "citations"):
                continue
            
            text = self._extract_text(value)
            if text and len(text.split()) >= self.min_section_words:
                sections.append(PatentSection(
                    section_type=SectionType.OTHER,
                    text=text,
                    order=order,
                    metadata={**base_metadata, "field_name": key} if self.include_metadata else {},
                ))
                order += 1
        
        return sections
    
    def _extract_claims(
        self,
        claims_data: Any,
        start_order: int,
        base_metadata: Dict[str, Any],
    ) -> List[PatentSection]:
        """
        Extract individual claims as separate sections.
        
        Args:
            claims_data: Raw claims data (can be list, dict, or string)
            start_order: Starting order number
            base_metadata: Base metadata to include
        
        Returns:
            List of PatentSection objects for claims
        """
        sections: List[PatentSection] = []
        order = start_order
        
        if isinstance(claims_data, list):
            for idx, claim in enumerate(claims_data, start=1):
                text = self._extract_text(claim)
                if text and len(text.split()) >= 5:  # Lower threshold for claims
                    sections.append(PatentSection(
                        section_type=SectionType.CLAIM,
                        text=text,
                        order=order,
                        metadata={**base_metadata, "claim_number": idx} if self.include_metadata else {},
                        parent_section="claims",
                        section_number=idx,
                    ))
                    order += 1
        elif isinstance(claims_data, str):
            # Try to split by claim numbers (1., 2., etc.)
            claims_text = self._normalize_text(claims_data)
            individual_claims = self._split_claims_text(claims_text)
            
            for idx, claim_text in enumerate(individual_claims, start=1):
                if claim_text and len(claim_text.split()) >= 5:
                    sections.append(PatentSection(
                        section_type=SectionType.CLAIM,
                        text=claim_text,
                        order=order,
                        metadata={**base_metadata, "claim_number": idx} if self.include_metadata else {},
                        parent_section="claims",
                        section_number=idx,
                    ))
                    order += 1
        elif isinstance(claims_data, dict):
            text = self._extract_text(claims_data)
            if text:
                sections.append(PatentSection(
                    section_type=SectionType.CLAIMS,
                    text=text,
                    order=order,
                    metadata=base_metadata.copy() if self.include_metadata else {},
                ))
        
        return sections
    
    def _split_claims_text(self, text: str) -> List[str]:
        """Split claims text by claim numbers."""
        import re
        
        # Pattern to match claim numbers: "1.", "1:", "Claim 1", etc.
        pattern = r'(?:^|\n)\s*(?:Claim\s+)?(\d+)[.\):\s]'
        
        # Find all claim starts
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        
        if not matches:
            return [text]  # Return as single claim if no pattern found
        
        claims = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            claim_text = text[start:end].strip()
            if claim_text:
                claims.append(claim_text)
        
        return claims
    
    def _normalize_text(self, value: Any) -> str:
        """Normalize text by collapsing whitespace."""
        if isinstance(value, str):
            return " ".join(value.split())
        return ""
    
    def _extract_text(self, value: Any) -> str:
        """Extract text from various data types."""
        if value is None:
            return ""
        
        if isinstance(value, str):
            return self._normalize_text(value)
        
        if isinstance(value, list):
            texts = [self._extract_text(item) for item in value]
            return " ".join(t for t in texts if t)
        
        if isinstance(value, dict):
            # Try common text keys
            for key in ("text", "content", "value", "description"):
                if key in value:
                    return self._extract_text(value[key])
            # Concatenate all string values
            texts = [self._extract_text(v) for v in value.values()]
            return " ".join(t for t in texts if t)
        
        return str(value) if value else ""
    
    def get_section_weight(self, section: PatentSection) -> float:
        """Get the importance weight for a section type."""
        return SECTION_WEIGHTS.get(section.section_type, 1.0)

