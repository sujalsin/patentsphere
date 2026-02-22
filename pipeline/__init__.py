"""
PatentSphere Disclosure-to-Strategic-Patent-Draft Pipeline.

This package implements a 5-step end-to-end flow:
  1. Decompose  – raw input → atomic sections with provenance
  2. Analyze    – per-section RAG + RLAIF novelty / risk scores
  3. Draft      – strategic independent + dependent patent claims
  4. Generate   – full attorney-ready specification
  5. Review     – multi-critic Human-Wall checklist + export gating

Enterprise Bridge: tenant_id is threaded through every step.
Future per-tenant Qdrant collection isolation is scaffolded via
  # ENTERPRISE BRIDGE comments throughout the package.
"""

from pipeline.models import (
    DisclosureSection,
    DecomposedDisclosure,
    SectionAnalysis,
    PatentClaim,
    DraftSpecification,
    CriticFlag,
    HumanWallChecklist,
    PipelineState,
)
from pipeline.orchestrator import run_pipeline, run_pipeline_step_by_step

__all__ = [
    "DisclosureSection",
    "DecomposedDisclosure",
    "SectionAnalysis",
    "PatentClaim",
    "DraftSpecification",
    "CriticFlag",
    "HumanWallChecklist",
    "PipelineState",
    "run_pipeline",
    "run_pipeline_step_by_step",
]
