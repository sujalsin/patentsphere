"""Pydantic output models for PatentSphere agents.

These models define the structured output format for all LLM-using agents,
ensuring consistent, validated responses with proper type checking.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, validator


# =============================================================================
# Claims Analyzer Output Models
# =============================================================================

class FeatureItem(BaseModel):
    """A technical feature extracted from the query."""
    
    name: str = Field(
        ...,
        description="Short name for the technical feature",
        min_length=1,
        max_length=100,
    )
    insight: str = Field(
        ...,
        description="Brief insight about this feature's patent relevance",
        min_length=1,
    )
    evidence: str = Field(
        default="",
        description="Supporting evidence or keywords from the query",
    )


class CPCCode(BaseModel):
    """A CPC classification code with confidence and justification."""
    
    code: str = Field(
        ...,
        description="CPC classification code (e.g., G06N3/063)",
        min_length=1,
        max_length=20,
    )
    title: str = Field(
        default="",
        description="Human-readable title for the CPC code",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1",
    )
    justification: str = Field(
        default="",
        description="Reasoning for this CPC code assignment",
    )


class RiskEntities(BaseModel):
    """Risk entities extracted from the query for litigation and retrieval."""
    
    assignees: List[str] = Field(
        default_factory=list,
        description="List of company names or assignees mentioned in the query",
    )
    topics: List[str] = Field(
        default_factory=list,
        description="List of technical topics or keywords for litigation search",
    )
    search_query: str = Field(
        default="",
        description="Optimized query string for retrieval (technical terms, keywords)",
    )


class MetadataFilters(BaseModel):
    """Metadata filters for retrieval (date range, CPC codes)."""
    
    date_range: Optional[List[int]] = Field(
        default=None,
        description="Date range [start_year, end_year] for filtering patents",
    )
    cpc_codes: List[str] = Field(
        default_factory=list,
        description="CPC codes to use as filters in retrieval",
    )


class ClaimsOutput(BaseModel):
    """Structured output from the Claims Analyzer agent."""
    
    summary: str = Field(
        ...,
        description="Concise summary of the query in patent-focused terms",
        min_length=10,
    )
    query_type: Literal["emergence", "litigation", "portfolio", "research", "other"] = Field(
        ...,
        description="Classified intent category of the query",
    )
    features: List[FeatureItem] = Field(
        default_factory=list,
        description="List of 2-4 critical technical features",
        min_items=0,
        max_items=6,
    )
    cpc_codes: List[CPCCode] = Field(
        default_factory=list,
        description="Likely CPC codes with confidence scores",
        min_items=0,
        max_items=5,
    )
    assumptions: List[str] = Field(
        default_factory=list,
        description="Explicit assumptions made during analysis",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Overall confidence in the analysis",
    )
    risk_entities: Optional[RiskEntities] = Field(
        default=None,
        description="Risk entities (assignees, topics) extracted for litigation and retrieval",
    )
    metadata_filters: Optional[MetadataFilters] = Field(
        default=None,
        description="Metadata filters (date range, CPC codes) for retrieval",
    )
    
    @validator("confidence", pre=True, always=True)
    def clamp_confidence(cls, v):
        if v is None:
            return 0.5
        return max(0.0, min(1.0, float(v)))

    class Config:
        json_schema_extra = {
            "example": {
                "summary": "Query focuses on neural network optimization techniques for edge devices",
                "query_type": "research",
                "features": [
                    {
                        "name": "Model Compression",
                        "insight": "Reducing neural network size for deployment",
                        "evidence": "edge devices, optimization"
                    }
                ],
                "cpc_codes": [
                    {
                        "code": "G06N3/08",
                        "title": "Learning methods",
                        "confidence": 0.85,
                        "justification": "Neural network training optimization"
                    }
                ],
                "assumptions": ["Focus is on inference optimization, not training"],
                "confidence": 0.8
            }
        }


# =============================================================================
# Synthesis Agent Output Models
# =============================================================================

class BulletPoint(BaseModel):
    """A single bullet point within an insight section."""
    
    headline: str = Field(
        ...,
        description="Entity or signal with concise descriptor",
        min_length=1,
    )
    details: List[str] = Field(
        default_factory=list,
        description="Short bullet strings with supporting details",
    )
    citations: List[str] = Field(
        default_factory=list,
        description="Patent IDs referenced (e.g., US1234567A)",
    )


class InsightSection(BaseModel):
    """A section of insights in the synthesis output."""
    
    title: str = Field(
        ...,
        description="Dynamic section title matching query focus",
        min_length=1,
    )
    bullets: List[BulletPoint] = Field(
        default_factory=list,
        description="List of bullet points with headlines and details",
    )


class NextStep(BaseModel):
    """A recommended next step with priority and rationale."""
    
    priority: Literal["high", "medium", "low"] = Field(
        ...,
        description="Priority level of this recommendation",
    )
    recommendation: str = Field(
        ...,
        description="Actionable step to take",
        min_length=1,
    )
    rationale: str = Field(
        default="",
        description="Why this step is recommended now",
    )


class CitationReference(BaseModel):
    """A citation reference with reason for inclusion."""
    
    patent_id: str = Field(
        ...,
        description="Patent ID (e.g., US1234567A, WO2023123456)",
    )
    reason: str = Field(
        default="",
        description="Why this patent is referenced",
    )


class SynthesisOutput(BaseModel):
    """Structured output from the Synthesis agent."""
    
    executive_summary: str = Field(
        ...,
        description="2-4 sentences synthesizing the most important insight with inline patent citations",
        min_length=20,
    )
    insight_sections: List[InsightSection] = Field(
        default_factory=list,
        description="2-3 sections with 1-3 bullets each",
    )
    next_steps: List[NextStep] = Field(
        default_factory=list,
        description="Actionable recommendations with priorities",
    )
    citations: List[CitationReference] = Field(
        default_factory=list,
        description="All patents referenced in the response",
    )
    risk_score: int = Field(
        default=50,
        ge=0,
        le=100,
        description="Overall threat/uncertainty score (0-100)",
    )
    
    @validator("risk_score", pre=True, always=True)
    def clamp_risk_score(cls, v):
        if v is None:
            return 50
        return max(0, min(100, int(v)))

    class Config:
        json_schema_extra = {
            "example": {
                "executive_summary": "Analysis reveals emerging patent activity in neural architecture search, with US12345678A introducing novel efficiency metrics.",
                "insight_sections": [
                    {
                        "title": "Emerging Threats",
                        "bullets": [
                            {
                                "headline": "AutoML Patent Cluster",
                                "details": ["3 patents filed in Q4 2023", "Focus on edge deployment"],
                                "citations": ["US12345678A"]
                            }
                        ]
                    }
                ],
                "next_steps": [
                    {
                        "priority": "high",
                        "recommendation": "Review US12345678A claims for overlap",
                        "rationale": "Directly relevant to current R&D direction"
                    }
                ],
                "citations": [
                    {"patent_id": "US12345678A", "reason": "Core reference for neural architecture search"}
                ],
                "risk_score": 65
            }
        }


# =============================================================================
# Critic Agent Output Models
# =============================================================================

class RewardComponents(BaseModel):
    """Individual reward component scores from the critic."""
    
    # Heuristic signals (30% of total reward)
    citation_overlap: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Score for citation network connectivity (heuristic)",
    )
    cpc_relevance: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Score for CPC code alignment (heuristic)",
    )
    temporal_diversity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Score for publication date spread (heuristic)",
    )
    
    # AI Feedback signals - TRUE RLAIF (70% of total reward)
    llm_fluency: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="LLM-rated response quality and coherence (RLAIF)",
    )
    llm_relevance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="LLM-rated relevance to user query (RLAIF)",
    )
    llm_completeness: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="LLM-rated response completeness (RLAIF)",
    )


class CriticOutput(BaseModel):
    """Structured output from the Critic agent."""
    
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Weighted total reward score",
    )
    components: RewardComponents = Field(
        default_factory=RewardComponents,
        description="Individual component scores",
    )
    weights: Dict[str, float] = Field(
        default_factory=dict,
        description="Weights used for each component",
    )
    feedback: str = Field(
        default="",
        description="Optional qualitative feedback",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "score": 0.72,
                "components": {
                    "citation_overlap": 0.8,
                    "cpc_relevance": 0.7,
                    "temporal_diversity": 0.6,
                    "llm_fluency": 0.85
                },
                "weights": {
                    "citation_overlap": 0.4,
                    "cpc_relevance": 0.3,
                    "temporal_diversity": 0.2,
                    "llm_fluency": 0.1
                },
                "feedback": "Good citation coverage with strong CPC alignment"
            }
        }


# =============================================================================
# Final Output Model (for the synthesis layer)
# =============================================================================

class FinalResponse(BaseModel):
    """Final structured response combining all agent outputs."""
    
    query: str = Field(
        ...,
        description="Original user query",
    )
    executive_summary: str = Field(
        ...,
        description="Executive summary from synthesis",
    )
    query_analysis: ClaimsOutput = Field(
        ...,
        description="Claims analysis output",
    )
    synthesis: SynthesisOutput = Field(
        ...,
        description="Full synthesis output",
    )
    quality_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Quality score from critic agent",
    )
    retrieved_patents: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of retrieved patent chunks",
    )
    litigation_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Litigation scout findings if available",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (latencies, agent info, etc.)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the emerging patents in neural architecture search?",
                "executive_summary": "Analysis reveals emerging patent activity...",
                "query_analysis": {},
                "synthesis": {},
                "quality_score": 0.72,
                "retrieved_patents": [],
                "litigation_data": None,
                "metadata": {"total_latency_ms": 2500}
            }
        }


# =============================================================================
# Enhanced API Response Models (with clickable citations)
# =============================================================================

class SourceItem(BaseModel):
    """A source citation with full metadata for API response."""
    
    index: int = Field(
        ...,
        description="Citation index (1-based) for inline references [1], [2]",
    )
    type: Literal["patent", "litigation"] = Field(
        ...,
        description="Type of source",
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
    url: Optional[str] = Field(
        default=None,
        description="Primary URL to the source document",
    )
    snippet: str = Field(
        default="",
        description="Key passage or excerpt from the source",
    )
    score: Optional[float] = Field(
        default=None,
        description="Relevance score (0-1)",
    )
    # Litigation-specific fields
    outcome: Optional[str] = Field(
        default=None,
        description="Case outcome (for litigation)",
    )
    risk_level: Optional[Literal["high", "medium", "low"]] = Field(
        default=None,
        description="Litigation risk level",
    )
    ui_state: Optional[Dict[str, Any]] = Field(
        default=None,
        description="UI state for expandable citation drawer (expandable, chunk_text, etc.)",
    )


class AnswerSectionModel(BaseModel):
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


class APIResponse(BaseModel):
    """
    Enhanced API response with indexed, clickable citations.
    
    This is the primary response format for the PatentSphere API,
    designed for frontend consumption with expandable source details.
    """
    
    response_id: str = Field(
        ...,
        description="Unique response identifier",
    )
    query: str = Field(
        ...,
        description="Original user query",
    )
    sources_header: str = Field(
        default="",
        description="Formatted sources line: [1] US-123 (IBM) · [2] US-456 (Siemens)",
    )
    answer_section: AnswerSectionModel = Field(
        ...,
        description="Main answer with technical and legal analysis",
    )
    sources: List[SourceItem] = Field(
        default_factory=list,
        description="All source citations with full metadata and URLs",
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
        description="Additional metadata (latencies, iterations, etc.)",
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "response_id": "resp_12345",
                "query": "Analyze the patentability of a blockchain-based energy trading system",
                "sources_header": "[1] US-1123456 (IBM) · [2] US-9876543 (Siemens) · [3] SolarCity v. SunPower (Litigation)",
                "answer_section": {
                    "technical_summary": "The concept of distributed energy trading on a blockchain has significant prior art coverage. Specifically, US-1123456 (IBM) discloses a method for peer-to-peer energy transactions [1]. Furthermore, US-9876543 (Siemens) expands on this by integrating IoT smart meters [2].",
                    "legal_summary": "You face a Medium-High litigation risk. The case of SolarCity v. SunPower [3] resulted in a finding of infringement for similar distributed grid management algorithms.",
                    "novelty_assessment": "Your proposed feature regarding 'off-chain zero-knowledge proofs for privacy' appears less crowded."
                },
                "sources": [
                    {
                        "index": 1,
                        "type": "patent",
                        "title": "US-1123456: Method for Decentralized Grid Management",
                        "assignee": "IBM",
                        "url": "https://patents.google.com/patent/US1123456",
                        "snippet": "A plurality of nodes negotiating energy transfer rates via a shared immutable ledger...",
                        "score": 0.92
                    },
                    {
                        "index": 2,
                        "type": "patent",
                        "title": "US-9876543: IoT Integration for Smart Grids",
                        "assignee": "Siemens",
                        "url": "https://patents.google.com/patent/US9876543",
                        "snippet": "Integrating smart meters to validate energy production...",
                        "score": 0.88
                    },
                    {
                        "index": 3,
                        "type": "litigation",
                        "title": "SolarCity Corp v. SunPower Corp",
                        "outcome": "Infringement Found",
                        "url": "https://dockets.justia.com/docket/...",
                        "snippet": "Defendant's use of a peer-to-peer matching algorithm infringes...",
                        "risk_level": "high"
                    }
                ],
                "risk_score": 65,
                "quality_score": 0.78,
                "metadata": {
                    "total_latency_ms": 2500,
                    "iterations": 1,
                    "agents_executed": ["claims_analyzer", "citation_mapper", "litigation_scout", "synthesis", "critic"]
                }
            }
        }

