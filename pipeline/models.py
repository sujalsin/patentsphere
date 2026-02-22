"""
Pydantic v2 models for the PatentSphere Disclosure-to-Draft Pipeline.

Every model is immutable (frozen=True where sensible) and includes
full JSON serialisation support for audit logging and API responses.

ENTERPRISE BRIDGE: tenant_id is optional on top-level models.
  Future: use tenant_id to route to per-tenant Qdrant collections
  and isolate audit logs per client.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SectionName(str, Enum):
    """The 7 canonical atomic sections of a patent disclosure."""
    BACKGROUND = "background"
    PROBLEM_STATEMENT = "problem_statement"
    CORE_INVENTION = "core_invention"
    EMBODIMENTS = "embodiments"
    ADVANTAGES = "advantages"
    USE_CASES = "use_cases"
    FIGURES_DESCRIPTION = "figures_description"


class ClaimType(str, Enum):
    INDEPENDENT = "independent"
    DEPENDENT = "dependent"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class FlagType(str, Enum):
    CITATION_NEEDED = "citation_needed"
    NOVELTY_RISK = "novelty_risk"
    OBVIOUSNESS_RISK = "obviousness_risk"
    ENABLEMENT_GAP = "enablement_gap"
    LITIGATION_RISK = "litigation_risk"
    INCOMPLETE_SECTION = "incomplete_section"
    ATTORNEY_REVIEW = "attorney_review"


# ---------------------------------------------------------------------------
# Step 1 – Decomposition
# ---------------------------------------------------------------------------

class ProvenanceSpan(BaseModel):
    """Maps a section back to its exact origin in the raw input."""
    char_start: int = Field(..., description="Start character index in raw_input")
    char_end: int = Field(..., description="End character index in raw_input")
    excerpt: str = Field(..., description="Short verbatim excerpt from original text")

    model_config = {"frozen": True}


class DisclosureSection(BaseModel):
    """One atomic, professionally cleaned section of an inventor disclosure."""
    section_name: SectionName
    text: str = Field(..., description="Cleaned, professional rewrite of the section")
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="LLM confidence that this section was correctly extracted (0–1)"
    )
    provenance: ProvenanceSpan = Field(
        ..., description="Traceable link back to the original raw input"
    )
    word_count: int = Field(default=0)

    @model_validator(mode="after")
    def compute_word_count(self) -> "DisclosureSection":
        object.__setattr__(self, "word_count", len(self.text.split()))
        return self


class DecomposedDisclosure(BaseModel):
    """Full decomposition of a raw inventor disclosure into canonical sections."""
    # ENTERPRISE BRIDGE: tenant_id links this disclosure to a specific client workspace.
    tenant_id: Optional[str] = Field(None, description="Tenant identifier for enterprise isolation")
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    raw_input_hash: str = Field(..., description="SHA-256 hex digest of raw_input for audit")
    raw_input_length: int = Field(..., description="Character count of raw input")
    sections: List[DisclosureSection] = Field(default_factory=list)
    decomposition_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    # Map section_name -> section for fast lookup
    # populated automatically by validator
    _section_map: Dict[str, DisclosureSection] = {}

    @model_validator(mode="after")
    def build_section_map(self) -> "DecomposedDisclosure":
        self._section_map = {s.section_name.value: s for s in self.sections}
        return self

    def get_section(self, name: SectionName) -> Optional[DisclosureSection]:
        return self._section_map.get(name.value)


# ---------------------------------------------------------------------------
# Step 2 – Parallel Analysis
# ---------------------------------------------------------------------------

class PriorArtReference(BaseModel):
    """A single prior-art hit with traceable citation."""
    patent_id: str
    title: Optional[str] = None
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    relevant_excerpt: str = Field(default="")
    citation_format: str = Field(default="", description="e.g. (Patent: US_12345, Claim 1)")


class SectionAnalysis(BaseModel):
    """Prior-art + risk analysis for one disclosure section."""
    section_name: SectionName
    # Scores (0–1, higher = worse except novelty_score where higher = better)
    novelty_score: float = Field(..., ge=0.0, le=1.0, description="Higher = more novel")
    obviousness_risk: float = Field(..., ge=0.0, le=1.0, description="Higher = more risk")
    enablement_score: float = Field(..., ge=0.0, le=1.0, description="Higher = better enablement")
    litigation_risk: float = Field(..., ge=0.0, le=1.0, description="Higher = more risk")
    prior_art_hits: List[PriorArtReference] = Field(default_factory=list)
    analysis_summary: str = Field(default="", description="Human-readable summary from RAG agents")
    raw_rag_output: str = Field(default="", description="Raw synthesizer output for audit")
    analysis_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ---------------------------------------------------------------------------
# Step 3 – Strategic Claims
# ---------------------------------------------------------------------------

class PatentClaim(BaseModel):
    """One patent claim with strategic rationale."""
    claim_number: int
    claim_type: ClaimType
    # If dependent, which claim number it depends on
    depends_on: Optional[int] = None
    text: str = Field(..., description="Formal claim language")
    strategy_reasoning: str = Field(
        ..., description="Plain-English explanation of why this claim is written this way"
    )
    breadth_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Estimated claim breadth (1=very broad/market-protecting, 0=very narrow)"
    )


class ClaimSet(BaseModel):
    """Full set of claims: 1 independent + 3-5 dependent."""
    claims: List[PatentClaim] = Field(default_factory=list)
    independent_claim_count: int = Field(default=0)
    dependent_claim_count: int = Field(default=0)
    drafting_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @model_validator(mode="after")
    def count_claims(self) -> "ClaimSet":
        ind = sum(1 for c in self.claims if c.claim_type == ClaimType.INDEPENDENT)
        dep = sum(1 for c in self.claims if c.claim_type == ClaimType.DEPENDENT)
        object.__setattr__(self, "independent_claim_count", ind)
        object.__setattr__(self, "dependent_claim_count", dep)
        return self


# ---------------------------------------------------------------------------
# Step 4 – Full Specification
# ---------------------------------------------------------------------------

class DraftSpecification(BaseModel):
    """Complete attorney-ready patent specification."""
    title: str
    abstract: str = Field(..., description="≤150 words, USPTO-compliant")
    background: str = Field(..., description="Lean background – what existed before")
    summary_of_invention: str
    detailed_description: str = Field(
        ..., description="Full embodiment description with figure references"
    )
    claims: List[PatentClaim] = Field(default_factory=list)
    # Provenance: maps each top-level field to originating section analyses
    provenance_map: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Maps spec section names → list of source section_name values"
    )
    generation_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def word_count(self) -> int:
        full_text = " ".join([
            self.title, self.abstract, self.background,
            self.summary_of_invention, self.detailed_description
        ])
        return len(full_text.split())


# ---------------------------------------------------------------------------
# Step 5 – Human-Wall Review
# ---------------------------------------------------------------------------

class CriticFlag(BaseModel):
    """A specific issue flagged by the multi-critic reviewer."""
    section: str = Field(..., description="Which part of the draft (e.g. 'abstract', 'claim_1')")
    flag_type: FlagType
    severity: Severity
    issue: str = Field(..., description="Concise description of the problem")
    suggestion: str = Field(..., description="Actionable improvement recommendation")

    model_config = {"frozen": True}


class SectionScore(BaseModel):
    """Per-section confidence score in the final draft."""
    section: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    word_count: int = Field(default=0)

    model_config = {"frozen": True}


class HumanWallChecklist(BaseModel):
    """
    The Human-Wall checkpoint gate.

    export_allowed starts as False and is set to True only when
    the user explicitly approves in the Chainlit wizard.
    """
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    section_scores: List[SectionScore] = Field(default_factory=list)
    flags: List[CriticFlag] = Field(default_factory=list)
    high_severity_count: int = Field(default=0)
    medium_severity_count: int = Field(default=0)
    low_severity_count: int = Field(default=0)
    # HUMAN-WALL GATE: export is blocked until user explicitly approves
    export_allowed: bool = Field(
        default=False,
        description="Set to True only after user approves the Human-Wall checkpoint"
    )
    user_approved_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp of user approval (None = not yet approved)"
    )
    reviewer_notes: str = Field(default="", description="Overall reviewer summary")
    review_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @model_validator(mode="after")
    def count_severity(self) -> "HumanWallChecklist":
        h = sum(1 for f in self.flags if f.severity == Severity.HIGH)
        m = sum(1 for f in self.flags if f.severity == Severity.MEDIUM)
        l = sum(1 for f in self.flags if f.severity == Severity.LOW)
        object.__setattr__(self, "high_severity_count", h)
        object.__setattr__(self, "medium_severity_count", m)
        object.__setattr__(self, "low_severity_count", l)
        return self

    def approve(self) -> "HumanWallChecklist":
        """Called when the human explicitly approves the draft at the checkpoint."""
        return self.model_copy(update={
            "export_allowed": True,
            "user_approved_at": datetime.now(timezone.utc),
        })


# ---------------------------------------------------------------------------
# Top-level Pipeline State
# ---------------------------------------------------------------------------

class AuditLogEntry(BaseModel):
    """One immutable entry in the pipeline audit trail."""
    step: str
    status: str  # "started" | "completed" | "failed"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    detail: Optional[str] = None

    model_config = {"frozen": True}


class PipelineState(BaseModel):
    """
    Master state object for one complete Disclosure-to-Draft pipeline run.

    ENTERPRISE BRIDGE: tenant_id enables future multi-tenant isolation.
      - Qdrant collection routing: f"{tenant_id}_patents" (future)
      - Audit log partitioning per tenant (future)
      - Role-based access control per tenant (future)
    """
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: Optional[str] = Field(
        None,
        description="ENTERPRISE BRIDGE: identifies the client tenant for isolation"
    )
    raw_input: str = Field(..., description="Original unmodified inventor input")

    # Step outputs – populated progressively
    decomposition: Optional[DecomposedDisclosure] = None
    analyses: Optional[List[SectionAnalysis]] = None
    claim_set: Optional[ClaimSet] = None
    specification: Optional[DraftSpecification] = None
    checklist: Optional[HumanWallChecklist] = None

    # Audit trail – append-only log of every pipeline event
    audit_log: List[AuditLogEntry] = Field(default_factory=list)

    # Pipeline-level metadata
    pipeline_version: str = Field(default="1.0.0")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    failed_at_step: Optional[str] = Field(
        default=None,
        description="Name of the step that failed, if any"
    )

    def log(self, step: str, status: str, detail: Optional[str] = None) -> "PipelineState":
        """Return a new state with an audit entry appended (immutable-style)."""
        entry = AuditLogEntry(step=step, status=status, detail=detail)
        return self.model_copy(update={"audit_log": [*self.audit_log, entry]})

    def is_complete(self) -> bool:
        return all([
            self.decomposition is not None,
            self.analyses is not None,
            self.claim_set is not None,
            self.specification is not None,
            self.checklist is not None,
        ])

    def export_ready(self) -> bool:
        return self.is_complete() and (
            self.checklist is not None and self.checklist.export_allowed
        )
