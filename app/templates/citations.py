"""Citation tracking and URL generation for PatentSphere.

This module provides:
1. URL builders for patent and litigation sources
2. Citation indexing and management
3. Source aggregation for final output
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field


# =============================================================================
# URL Builders
# =============================================================================

class PatentURLBuilder:
    """Generate URLs for patent documents from various sources."""
    
    # URL templates for different patent offices
    GOOGLE_PATENTS_TEMPLATE = "https://patents.google.com/patent/{patent_id}"
    USPTO_TEMPLATE = "https://patft.uspto.gov/netacgi/nph-Parser?Sect1=PTO1&Sect2=HITOFF&p=1&u=/netahtml/PTO/srchnum.html&r=1&f=G&l=50&d=PALL&s1={patent_number}.PN."
    USPTO_APPFT_TEMPLATE = "https://appft.uspto.gov/netacgi/nph-Parser?Sect1=PTO1&Sect2=HITOFF&p=1&u=/netahtml/PTO/srchnum.html&r=1&f=G&l=50&d=PG01&s1={app_number}.PGNR."
    EPO_TEMPLATE = "https://worldwide.espacenet.com/patent/search?q=pn%3D{patent_id}"
    WIPO_TEMPLATE = "https://patentscope.wipo.int/search/en/detail.jsf?docId={patent_id}"
    
    @classmethod
    def normalize_patent_id(cls, patent_id: str) -> str:
        """Normalize patent ID format (remove spaces, standardize)."""
        # Remove spaces and convert to uppercase
        normalized = patent_id.replace(" ", "").replace("-", "").upper()
        return normalized
    
    @classmethod
    def extract_patent_number(cls, patent_id: str) -> str:
        """Extract just the numeric portion for USPTO search."""
        # Remove country code and kind code
        match = re.search(r'(\d{6,})', patent_id)
        if match:
            return match.group(1)
        return patent_id
    
    @classmethod
    def get_google_patents_url(cls, patent_id: str) -> str:
        """Generate Google Patents URL."""
        normalized = cls.normalize_patent_id(patent_id)
        return cls.GOOGLE_PATENTS_TEMPLATE.format(patent_id=normalized)
    
    @classmethod
    def get_uspto_url(cls, patent_id: str) -> Optional[str]:
        """Generate USPTO URL for US patents."""
        normalized = cls.normalize_patent_id(patent_id)
        
        # Only generate for US patents
        if not normalized.startswith("US"):
            return None
        
        # Check if it's an application or granted patent
        patent_number = cls.extract_patent_number(normalized)
        
        # Application numbers typically start with year (e.g., 20230123456)
        if len(patent_number) >= 11 and patent_number.startswith("20"):
            return cls.USPTO_APPFT_TEMPLATE.format(app_number=patent_number)
        
        return cls.USPTO_TEMPLATE.format(patent_number=patent_number)
    
    @classmethod
    def get_all_urls(cls, patent_id: str) -> Dict[str, str]:
        """Get all available URLs for a patent."""
        urls = {
            "google_patents": cls.get_google_patents_url(patent_id),
        }
        
        uspto_url = cls.get_uspto_url(patent_id)
        if uspto_url:
            urls["uspto"] = uspto_url
        
        normalized = cls.normalize_patent_id(patent_id)
        
        # Add EPO for European patents
        if normalized.startswith("EP"):
            urls["espacenet"] = cls.EPO_TEMPLATE.format(patent_id=normalized)
        
        # Add WIPO for PCT applications
        if normalized.startswith("WO"):
            urls["wipo"] = cls.WIPO_TEMPLATE.format(patent_id=normalized)
        
        return urls


class LitigationURLBuilder:
    """Generate URLs for litigation case documents."""
    
    # URL templates for court dockets
    JUSTIA_TEMPLATE = "https://dockets.justia.com/docket/{court}/{case_number}"
    PACER_TEMPLATE = "https://ecf.{court}.uscourts.gov/cgi-bin/DktRpt.pl?{case_number}"
    COURTLISTENER_TEMPLATE = "https://www.courtlistener.com/docket/{docket_id}/"
    
    @classmethod
    def parse_case_number(cls, case_string: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse case number to extract court and case number.
        
        Examples:
            "1:16-cv-0023" -> ("", "1:16-cv-0023")
            "Case 1:16-cv-0023 (D. Del.)" -> ("ded", "1:16-cv-0023")
        """
        # Try to extract court code
        court_match = re.search(r'\(([^)]+)\)', case_string)
        court = None
        if court_match:
            court_text = court_match.group(1).lower()
            # Map common court names to PACER codes
            court_map = {
                "d. del.": "ded",
                "n.d. cal.": "cand",
                "s.d.n.y.": "nysd",
                "e.d. tex.": "txed",
                "d.n.j.": "njd",
                "c.d. cal.": "cacd",
            }
            court = court_map.get(court_text)
        
        # Extract case number
        case_match = re.search(r'(\d+:\d+-\w+-\d+)', case_string)
        case_number = case_match.group(1) if case_match else None
        
        return court, case_number
    
    @classmethod
    def get_justia_url(cls, case_string: str, court_code: Optional[str] = None) -> Optional[str]:
        """Generate Justia docket URL."""
        court, case_number = cls.parse_case_number(case_string)
        court = court or court_code
        
        if not case_number:
            return None
        
        if court:
            return cls.JUSTIA_TEMPLATE.format(court=court, case_number=case_number)
        
        # Fallback to search URL
        return f"https://dockets.justia.com/search?query={case_number}"
    
    @classmethod
    def get_all_urls(cls, case_string: str, docket_id: Optional[str] = None) -> Dict[str, str]:
        """Get all available URLs for a litigation case."""
        urls = {}
        
        justia_url = cls.get_justia_url(case_string)
        if justia_url:
            urls["justia"] = justia_url
        
        if docket_id:
            urls["courtlistener"] = cls.COURTLISTENER_TEMPLATE.format(docket_id=docket_id)
        
        return urls


# =============================================================================
# Citation Source Models
# =============================================================================

class CitationSource(BaseModel):
    """A source citation with full metadata and URLs."""
    
    index: int = Field(
        ...,
        description="Citation index (1-based) for inline references [1], [2]",
    )
    type: Literal["patent", "litigation"] = Field(
        ...,
        description="Type of source",
    )
    id: str = Field(
        ...,
        description="Unique identifier (patent ID or case number)",
    )
    title: str = Field(
        ...,
        description="Title of the patent or case name",
    )
    # Patent-specific fields
    assignee: Optional[str] = Field(
        default=None,
        description="Patent assignee/owner (for patents)",
    )
    filing_date: Optional[str] = Field(
        default=None,
        description="Filing date (for patents)",
    )
    # Litigation-specific fields
    outcome: Optional[str] = Field(
        default=None,
        description="Case outcome (for litigation)",
    )
    parties: Optional[str] = Field(
        default=None,
        description="Parties involved (for litigation)",
    )
    risk_level: Optional[Literal["high", "medium", "low"]] = Field(
        default=None,
        description="Litigation risk level",
    )
    # Common fields
    urls: Dict[str, str] = Field(
        default_factory=dict,
        description="URLs to source documents",
    )
    snippet: str = Field(
        default="",
        description="Key passage or excerpt from the source",
    )
    section_type: Optional[str] = Field(
        default=None,
        description="Section type (claim, abstract, etc.) for patents",
    )
    score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Relevance score",
    )
    chunk_id: Optional[str] = Field(
        default=None,
        description="Reference to the exact retrieved chunk",
    )
    
    def format_header_entry(self) -> str:
        """Format this source for the sources header line."""
        if self.type == "patent":
            assignee_str = f" ({self.assignee})" if self.assignee else ""
            return f"[{self.index}] {self.id}{assignee_str}"
        else:
            return f"[{self.index}] {self.title} (Litigation)"


class AnswerSection(BaseModel):
    """Structured answer section with technical and legal analysis."""
    
    technical_summary: str = Field(
        ...,
        description="Technical landscape and prior art analysis with inline [1], [2] citations",
    )
    legal_summary: Optional[str] = Field(
        default=None,
        description="Litigation and legal risk analysis with citations",
    )
    novelty_assessment: Optional[str] = Field(
        default=None,
        description="Assessment of novelty/patentability gaps",
    )


class EnhancedFinalResponse(BaseModel):
    """Enhanced final response with indexed, clickable citations."""
    
    response_id: str = Field(
        ...,
        description="Unique response identifier",
    )
    query: str = Field(
        ...,
        description="Original user query",
    )
    sources_header: str = Field(
        ...,
        description="Formatted sources line: [1] US-123 (IBM) · [2] US-456 (Siemens)",
    )
    answer_section: AnswerSection = Field(
        ...,
        description="Main answer with technical and legal analysis",
    )
    sources: List[CitationSource] = Field(
        default_factory=list,
        description="All source citations with full metadata",
    )
    risk_score: int = Field(
        default=50,
        ge=0,
        le=100,
        description="Overall risk assessment (0-100)",
    )
    quality_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Response quality score from critic",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "response_id": "resp_12345",
                "query": "Analyze the patentability of a blockchain-based energy trading system",
                "sources_header": "[1] US-1123456 (IBM) · [2] US-9876543 (Siemens) · [3] SolarCity v. SunPower (Litigation)",
                "answer_section": {
                    "technical_summary": "The concept of distributed energy trading on a blockchain has significant prior art coverage. Specifically, US-1123456 (IBM) discloses a method for peer-to-peer energy transactions [1].",
                    "legal_summary": "You face a Medium-High litigation risk. The case of SolarCity v. SunPower [3] resulted in a finding of infringement."
                },
                "sources": [
                    {
                        "index": 1,
                        "type": "patent",
                        "id": "US-1123456",
                        "title": "Method for Decentralized Grid Management",
                        "assignee": "IBM",
                        "urls": {
                            "google_patents": "https://patents.google.com/patent/US1123456",
                            "uspto": "https://patft.uspto.gov/..."
                        },
                        "snippet": "A plurality of nodes negotiating energy transfer rates...",
                        "score": 0.92
                    }
                ],
                "risk_score": 65,
                "quality_score": 0.78
            }
        }


# =============================================================================
# Citation Index Manager
# =============================================================================

class CitationIndexManager:
    """Manages citation indices and deduplication."""
    
    def __init__(self):
        self._sources: List[CitationSource] = []
        self._patent_index: Dict[str, int] = {}  # patent_id -> index
        self._litigation_index: Dict[str, int] = {}  # case_id -> index
        self._current_index = 1
    
    def add_patent(
        self,
        patent_id: str,
        title: str = "",
        assignee: Optional[str] = None,
        snippet: str = "",
        score: Optional[float] = None,
        section_type: Optional[str] = None,
        chunk_id: Optional[str] = None,
        filing_date: Optional[str] = None,
    ) -> int:
        """
        Add a patent source and return its index.
        Returns existing index if patent already added.
        """
        normalized_id = PatentURLBuilder.normalize_patent_id(patent_id)
        
        # Check if already indexed
        if normalized_id in self._patent_index:
            return self._patent_index[normalized_id]
        
        # Create new citation
        index = self._current_index
        self._current_index += 1
        
        urls = PatentURLBuilder.get_all_urls(patent_id)
        
        source = CitationSource(
            index=index,
            type="patent",
            id=patent_id,
            title=title or f"Patent {patent_id}",
            assignee=assignee,
            filing_date=filing_date,
            urls=urls,
            snippet=snippet,
            section_type=section_type,
            score=score,
            chunk_id=chunk_id,
        )
        
        self._sources.append(source)
        self._patent_index[normalized_id] = index
        
        return index
    
    def add_litigation(
        self,
        case_id: str,
        title: str,
        outcome: Optional[str] = None,
        parties: Optional[str] = None,
        risk_level: Optional[str] = None,
        snippet: str = "",
        docket_id: Optional[str] = None,
    ) -> int:
        """
        Add a litigation source and return its index.
        Returns existing index if case already added.
        """
        # Check if already indexed
        if case_id in self._litigation_index:
            return self._litigation_index[case_id]
        
        # Create new citation
        index = self._current_index
        self._current_index += 1
        
        urls = LitigationURLBuilder.get_all_urls(case_id, docket_id)
        
        source = CitationSource(
            index=index,
            type="litigation",
            id=case_id,
            title=title,
            outcome=outcome,
            parties=parties,
            risk_level=risk_level,
            urls=urls,
            snippet=snippet,
        )
        
        self._sources.append(source)
        self._litigation_index[case_id] = index
        
        return index
    
    def get_index(self, source_id: str) -> Optional[int]:
        """Get the index for a source ID (patent or litigation)."""
        normalized = PatentURLBuilder.normalize_patent_id(source_id)
        if normalized in self._patent_index:
            return self._patent_index[normalized]
        if source_id in self._litigation_index:
            return self._litigation_index[source_id]
        return None
    
    def get_sources(self) -> List[CitationSource]:
        """Get all sources sorted by index."""
        return sorted(self._sources, key=lambda s: s.index)
    
    def generate_sources_header(self) -> str:
        """Generate the formatted sources header line."""
        sources = self.get_sources()
        entries = [s.format_header_entry() for s in sources]
        return " · ".join(entries)
    
    def format_inline_citation(self, source_id: str) -> str:
        """Format an inline citation reference like [1]."""
        index = self.get_index(source_id)
        if index:
            return f"[{index}]"
        return ""


def extract_citations_from_chunks(
    chunks: List[Dict[str, Any]],
    manager: Optional[CitationIndexManager] = None,
) -> CitationIndexManager:
    """
    Extract citations from retrieved chunks and build citation index.
    
    Args:
        chunks: List of retrieved patent chunks
        manager: Optional existing manager to add to
    
    Returns:
        CitationIndexManager with all citations indexed
    """
    if manager is None:
        manager = CitationIndexManager()
    
    seen_patents = set()
    
    for chunk in chunks:
        patent_id = chunk.get("patent_id")
        if not patent_id or patent_id in seen_patents:
            continue
        
        seen_patents.add(patent_id)
        
        # Extract metadata
        title = chunk.get("title", "")
        assignee = chunk.get("assignee")
        snippet = chunk.get("chunk_text", chunk.get("text", ""))[:300]
        score = chunk.get("score")
        section_type = chunk.get("chunk_type", chunk.get("section_type"))
        chunk_id = chunk.get("chunk_id")
        filing_date = chunk.get("publication_date", chunk.get("filing_date"))
        
        manager.add_patent(
            patent_id=patent_id,
            title=title,
            assignee=assignee,
            snippet=snippet,
            score=score,
            section_type=section_type,
            chunk_id=chunk_id,
            filing_date=filing_date,
        )
    
    return manager


def extract_citations_from_litigation(
    litigation_data: Dict[str, Any],
    manager: Optional[CitationIndexManager] = None,
) -> CitationIndexManager:
    """
    Extract citations from litigation scout data.
    
    Args:
        litigation_data: Litigation scout output
        manager: Optional existing manager to add to
    
    Returns:
        CitationIndexManager with litigation citations indexed
    """
    if manager is None:
        manager = CitationIndexManager()
    
    cases = litigation_data.get("cases", litigation_data.get("results", []))
    
    for case in cases:
        case_id = case.get("case_id", case.get("case_number", ""))
        if not case_id:
            continue
        
        title = case.get("case_name", case.get("title", f"Case {case_id}"))
        outcome = case.get("outcome", case.get("status"))
        parties = case.get("parties")
        risk_level = case.get("risk_level", case.get("risk"))
        snippet = case.get("summary", case.get("snippet", ""))[:300]
        docket_id = case.get("docket_id")
        
        manager.add_litigation(
            case_id=case_id,
            title=title,
            outcome=outcome,
            parties=parties,
            risk_level=risk_level,
            snippet=snippet,
            docket_id=docket_id,
        )
    
    return manager

