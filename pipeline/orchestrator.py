"""
Central Orchestrator for the Disclosure-to-Draft Pipeline.

This is the single entry-point that strings together all 5 steps:
  Step 1: decompose_disclosure  (decomposer.py)
  Step 2: analyze_sections      (analyzer.py)
  Step 3: draft_claims          (claims_drafter.py)
  Step 4: generate_specification (spec_generator.py)
  Step 5: review_draft          (reviewer.py)

Two call styles are provided:
  • run_pipeline()            – full sequential run → PipelineState
  • run_pipeline_step_by_step() – async generator yielding PipelineState
    after each step so callers (Chainlit wizard) can stream progress.

Error handling:
  On any step failure, the error is recorded in audit_log,
  failed_at_step is set, HumanWallChecklist.export_allowed remains False,
  and the partial PipelineState is returned so the UI can surface the issue.
"""

from __future__ import annotations

import traceback
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Optional

from pipeline.decomposer import decompose_disclosure
from pipeline.analyzer import analyze_sections
from pipeline.claims_drafter import draft_claims
from pipeline.spec_generator import generate_specification
from pipeline.reviewer import review_draft
from pipeline.models import (
    HumanWallChecklist,
    PipelineState,
    SectionScore,
    Severity,
    CriticFlag,
    FlagType,
)


# ---------------------------------------------------------------------------
# Full sequential run
# ---------------------------------------------------------------------------

async def run_pipeline(
    raw_input: str,
    llm: Any,
    workflow: Any,
    score_llm: Optional[Any] = None,
    tenant_id: Optional[str] = None,
) -> PipelineState:
    """
    Run the complete 5-step pipeline in sequence.

    Args:
        raw_input:  Plain-text inventor disclosure (already extracted from file).
        llm:        Primary LLM for decomposition, claims drafting, and spec generation.
        workflow:   Compiled LangGraph workflow (existing PatentSphere graph).
        score_llm:  Small/fast LLM for score extraction. Defaults to llm if None.
        tenant_id:  ENTERPRISE BRIDGE – propagated into all steps for future isolation.

    Returns:
        Completed PipelineState. Check state.failed_at_step for errors.
    """
    if score_llm is None:
        score_llm = llm

    state = PipelineState(raw_input=raw_input, tenant_id=tenant_id)

    # Consume step-by-step generator to get final state
    async for step_state in run_pipeline_step_by_step(
        raw_input=raw_input,
        llm=llm,
        workflow=workflow,
        score_llm=score_llm,
        tenant_id=tenant_id,
        _state=state,
    ):
        state = step_state

    return state


# ---------------------------------------------------------------------------
# Step-by-step async generator (for Chainlit streaming)
# ---------------------------------------------------------------------------

async def run_pipeline_step_by_step(
    raw_input: str,
    llm: Any,
    workflow: Any,
    score_llm: Optional[Any] = None,
    tenant_id: Optional[str] = None,
    _state: Optional[PipelineState] = None,
) -> AsyncGenerator[PipelineState, None]:
    """
    Async generator that yields PipelineState after every completed step.
    Callers can await each yield to show incremental progress in the UI.

    Yields:
        PipelineState – updated after each of the 5 pipeline steps.
    """
    if score_llm is None:
        score_llm = llm

    state = _state or PipelineState(raw_input=raw_input, tenant_id=tenant_id)

    # ── Step 1: Decompose ─────────────────────────────────────────────────
    state = state.log("decompose", "started")
    try:
        decomposition = await decompose_disclosure(
            raw_text=raw_input,
            llm=llm,
            tenant_id=tenant_id,
        )
        state = state.model_copy(update={"decomposition": decomposition})
        state = state.log("decompose", "completed",
                         f"{len(decomposition.sections)} sections extracted")
    except Exception as exc:
        state = state.log("decompose", "failed", _fmt_exc(exc))
        state = state.model_copy(update={
            "failed_at_step": "decompose",
            "checklist": _error_checklist("decompose", exc),
        })
        yield state
        return  # Cannot continue without decomposition

    yield state  # UI shows Step 1 complete

    # ── Step 2: Analyze ───────────────────────────────────────────────────
    state = state.log("analyze", "started")
    try:
        analyses = await analyze_sections(
            disclosure=state.decomposition,
            workflow=workflow,
            score_llm=score_llm,
            tenant_id=tenant_id,
        )
        state = state.model_copy(update={"analyses": analyses})
        state = state.log("analyze", "completed",
                         f"{len(analyses)} sections analyzed")
    except Exception as exc:
        state = state.log("analyze", "failed", _fmt_exc(exc))
        state = state.model_copy(update={
            "failed_at_step": "analyze",
            "analyses": [],
            "checklist": _error_checklist("analyze", exc),
        })
        yield state
        return

    yield state  # UI shows Step 2 complete

    # ── Step 3: Draft Claims ──────────────────────────────────────────────
    state = state.log("draft_claims", "started")
    try:
        claim_set = await draft_claims(
            disclosure=state.decomposition,
            analyses=state.analyses,
            llm=llm,
        )
        state = state.model_copy(update={"claim_set": claim_set})
        state = state.log("draft_claims", "completed",
                         f"{len(claim_set.claims)} claims drafted "
                         f"({claim_set.independent_claim_count} independent, "
                         f"{claim_set.dependent_claim_count} dependent)")
    except Exception as exc:
        state = state.log("draft_claims", "failed", _fmt_exc(exc))
        state = state.model_copy(update={
            "failed_at_step": "draft_claims",
            "checklist": _error_checklist("draft_claims", exc),
        })
        yield state
        return

    yield state  # UI shows Step 3 complete

    # ── Step 4: Generate Specification ───────────────────────────────────
    state = state.log("generate_spec", "started")
    try:
        specification = await generate_specification(
            disclosure=state.decomposition,
            analyses=state.analyses,
            claim_set=state.claim_set,
            llm=llm,
        )
        state = state.model_copy(update={"specification": specification})
        state = state.log("generate_spec", "completed",
                         f"Specification: ~{specification.word_count()} words")
    except Exception as exc:
        state = state.log("generate_spec", "failed", _fmt_exc(exc))
        state = state.model_copy(update={
            "failed_at_step": "generate_spec",
            "checklist": _error_checklist("generate_spec", exc),
        })
        yield state
        return

    yield state  # UI shows Step 4 complete

    # ── Step 5: Human-Wall Review ─────────────────────────────────────────
    state = state.log("review", "started")
    try:
        checklist = await review_draft(
            spec=state.specification,
            analyses=state.analyses,
            llm=llm,
        )
        state = state.model_copy(update={
            "checklist": checklist,
            "completed_at": datetime.now(timezone.utc),
        })
        state = state.log("review", "completed",
                         f"overall_confidence={checklist.overall_confidence:.2f}, "
                         f"flags={len(checklist.flags)}, "
                         f"export_allowed={checklist.export_allowed}")
    except Exception as exc:
        state = state.log("review", "failed", _fmt_exc(exc))
        state = state.model_copy(update={
            "failed_at_step": "review",
            "checklist": _error_checklist("review", exc),
        })

    yield state  # Final state – pipeline complete (or failed at review)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_exc(exc: Exception) -> str:
    return f"{type(exc).__name__}: {str(exc)[:300]}\n{traceback.format_exc()[-400:]}"


def _error_checklist(step: str, exc: Exception) -> HumanWallChecklist:
    """Return a checklist that blocks export and explains the failure."""
    return HumanWallChecklist(
        overall_confidence=0.0,
        reviewer_notes=(
            f"Pipeline failed at step '{step}': {type(exc).__name__}: {str(exc)[:200]}. "
            "Export is blocked. Please check the audit log for details."
        ),
        section_scores=[],
        flags=[
            CriticFlag(
                section=step,
                flag_type=FlagType.ATTORNEY_REVIEW,
                severity=Severity.HIGH,
                issue=f"Pipeline step '{step}' failed: {str(exc)[:150]}",
                suggestion="Review the error in the audit log and retry or contact support.",
            )
        ],
        export_allowed=False,
    )
