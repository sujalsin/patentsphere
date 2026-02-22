"""
Step 2 – Parallel Section Analyzer.

For each decomposed disclosure section we formulate a focused patent query
and run it through the existing PatentSphere LangGraph workflow to get:
  - Prior-art hits with traceable citations
  - Novelty, obviousness, enablement, and litigation risk scores

All sections are processed in parallel via asyncio.gather for speed.

ENTERPRISE BRIDGE: tenant_id is propagated into the workflow state so that
  future per-tenant Qdrant collection routing can be added with a one-line
  change to graph/tools.py (see # ENTERPRISE BRIDGE comment in tools.py).
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from pipeline.models import (
    DecomposedDisclosure,
    DisclosureSection,
    PriorArtReference,
    SectionAnalysis,
    SectionName,
)

# ---------------------------------------------------------------------------
# Score extraction prompt
# ---------------------------------------------------------------------------

_SCORE_SYSTEM = """\
You are a senior patent analyst. Given a patent analysis text produced by a RAG
system, extract structured risk and quality scores.

Return ONLY valid JSON:
{
  "novelty_score": <0.0–1.0, higher = more novel>,
  "obviousness_risk": <0.0–1.0, higher = more obvious / riskier>,
  "enablement_score": <0.0–1.0, higher = better enablement>,
  "litigation_risk": <0.0–1.0, higher = more litigation risk>,
  "prior_art_ids": ["US_XXXXXX", ...],
  "summary": "<2-3 sentence plain-English summary of the key findings>"
}

Base your scores on what the analysis text says. If the text does not mention a
dimension, default to 0.5. Do not wrap output in markdown fences.
"""

_QUERY_TEMPLATE = (
    "Analyze the following patent concept for prior art, novelty, "
    "obviousness, enablement quality, and litigation risk:\n\n{text}"
)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

async def analyze_sections(
    disclosure: DecomposedDisclosure,
    workflow: Any,            # The existing LangGraph compiled workflow
    score_llm: Any,           # LLM used only for score extraction (small/fast)
    tenant_id: Optional[str] = None,
) -> List[SectionAnalysis]:
    """
    Run parallel RAG + RLAIF analysis over all disclosure sections.

    Args:
        disclosure: The decomposed disclosure from Step 1.
        workflow:   The existing compiled PatentSphere LangGraph workflow.
        score_llm:  A fast LLM for extracting structured scores from RAG output.
        tenant_id:  ENTERPRISE BRIDGE – forwarded into workflow state.

    Returns:
        A list of SectionAnalysis objects, one per section.
    """
    tasks = [
        _analyze_single_section(section, workflow, score_llm, tenant_id)
        for section in disclosure.sections
    ]
    # Run all sections in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)

    analyses: List[SectionAnalysis] = []
    for section, result in zip(disclosure.sections, results):
        if isinstance(result, Exception):
            # On failure: return a stub with neutral scores rather than crashing
            print(f"[analyzer] Section '{section.section_name.value}' failed: {result}")
            analyses.append(_make_stub_analysis(section.section_name))
        else:
            analyses.append(result)

    return analyses


# ---------------------------------------------------------------------------
# Per-section analysis
# ---------------------------------------------------------------------------

async def _analyze_single_section(
    section: DisclosureSection,
    workflow: Any,
    score_llm: Any,
    tenant_id: Optional[str],
) -> SectionAnalysis:
    """Run one section through the existing RAG workflow + score extraction."""
    # Build a targeted query for this section
    query = _QUERY_TEMPLATE.format(text=section.text[:3000])

    # Build initial state compatible with existing AgentState schema
    initial_state: Dict[str, Any] = {
        "query": query,
        "intent": None,
        "keywords": None,
        "date_range": None,
        "documents": [],
        "litigation_context": [],
        "draft": None,
        "critique": None,
        "retry_count": 0,
        "final_response": None,
        # ENTERPRISE BRIDGE: tenant_id in state → future Qdrant collection routing
        "tenant_id": tenant_id,
    }
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # Run the full existing workflow (Router→Extractor→Retrieval→Synthesizer→Critic)
    try:
        final_state: Dict[str, Any] = {}
        async for state in workflow.astream(initial_state, config=config, stream_mode="values"):
            final_state = state
    except Exception as e:
        print(f"[analyzer] Workflow failed for section {section.section_name.value}: {e}")
        return _make_stub_analysis(section.section_name)

    raw_output = (
        final_state.get("final_response")
        or final_state.get("draft")
        or ""
    )
    raw_documents: List[Dict] = final_state.get("documents") or []

    # Extract scores from the RAG output using the score LLM
    scores = await _extract_scores(raw_output, score_llm)

    # Build prior-art references from retrieved documents
    prior_art = _extract_prior_art(raw_documents, raw_output)

    return SectionAnalysis(
        section_name=section.section_name,
        novelty_score=scores.get("novelty_score", 0.5),
        obviousness_risk=scores.get("obviousness_risk", 0.5),
        enablement_score=scores.get("enablement_score", 0.5),
        litigation_risk=scores.get("litigation_risk", 0.5),
        prior_art_hits=prior_art,
        analysis_summary=scores.get("summary", "Analysis completed."),
        raw_rag_output=raw_output[:4000],  # Cap for audit storage
    )


async def _extract_scores(rag_output: str, score_llm: Any) -> Dict[str, Any]:
    """Ask a fast LLM to extract structured scores from unstructured RAG text."""
    if not rag_output.strip():
        return {}

    messages = [
        SystemMessage(content=_SCORE_SYSTEM),
        HumanMessage(content=f"Analysis text:\n\n{rag_output[:4000]}"),
    ]
    try:
        response = await score_llm.ainvoke(messages)
        content = response.content.strip()
        # Strip markdown fences
        cleaned = re.sub(r"```(?:json)?\s*", "", content).strip().rstrip("`").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                return json.loads(match.group())
    except Exception as e:
        print(f"[analyzer] Score extraction failed: {e}")
    return {}


def _extract_prior_art(
    documents: List[Dict], raw_output: str
) -> List[PriorArtReference]:
    """Build PriorArtReference objects from retrieved documents + cited IDs in text."""
    refs: List[PriorArtReference] = []
    seen_ids: set = set()

    for doc in documents:
        pid = doc.get("patent_id", "")
        if not pid or pid in seen_ids:
            continue
        seen_ids.add(pid)

        # Find a short relevant excerpt from the raw output
        excerpt = ""
        if pid.replace("-", "_") in raw_output:
            # Extract surrounding context
            idx = raw_output.find(pid.replace("-", "_"))
            excerpt = raw_output[max(0, idx - 20): idx + 120].strip()

        refs.append(PriorArtReference(
            patent_id=pid,
            title=doc.get("title"),
            relevance_score=min(1.0, doc.get("score", 0.5)),
            relevant_excerpt=excerpt[:300],
            citation_format=f"(Patent: {pid.replace('-', '_')}, Abstract)",
        ))

    return refs[:10]  # Cap for readability


def _make_stub_analysis(section_name: SectionName) -> SectionAnalysis:
    """Return a neutral stub when analysis fails for a section."""
    return SectionAnalysis(
        section_name=section_name,
        novelty_score=0.5,
        obviousness_risk=0.5,
        enablement_score=0.5,
        litigation_risk=0.5,
        prior_art_hits=[],
        analysis_summary="Analysis could not be completed for this section.",
        raw_rag_output="",
    )
