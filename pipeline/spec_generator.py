"""
Step 4 – Full Specification Generator.

Assembles the complete, attorney-ready patent specification from:
  - Decomposed disclosure sections   (Step 1)
  - Section analysis results         (Step 2)
  - Strategic claim set              (Step 3)

Produces a DraftSpecification with:
  • Title
  • Abstract (≤150 words, USPTO-compliant)
  • Lean Background
  • Summary of the Invention
  • Detailed Description with inline figure references
  • Full Claims section (verbatim from Step 3)
  • Provenance map tracing every spec section to source disclosure sections
"""

from __future__ import annotations

import re
from typing import Any, List

from langchain_core.messages import HumanMessage, SystemMessage

from pipeline.models import (
    ClaimSet,
    DecomposedDisclosure,
    DraftSpecification,
    PatentClaim,
    SectionAnalysis,
    SectionName,
)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SPEC_SYSTEM = """\
You are the lead specification writer at a premier USPTO patent prosecution firm.
Write complete patent specifications that read like filed, granted patents.

STRUCTURE REQUIREMENTS:
title            — Noun-first, max 12 words. "System and Method for X" not "A System..."
abstract         — Exactly 1 paragraph, ≤150 words. Opens "A system/method for X is disclosed."
                   Ends with primary distinguishing advantage. NO claim numbers cited.
background       — 2 paragraphs. ¶1: field sentence. ¶2: 3-4 prior art deficiencies.
                   NEVER mention this invention or its solution in the background.
summary_of_invention — 2 paragraphs. ¶1: "In one aspect, the invention provides..."
                       ¶2: "The invention achieves one or more of..." (list advantages).
detailed_description — 5-7 paragraphs. MANDATORY:
                       ¶1 starts "Referring now to FIG. 1..."
                       ¶4 starts "In an alternative embodiment..."
                       Every element of Claim 1 must appear here.
                       POSITA enablement standard — sufficient detail to reproduce.
prior_art_navigation — A "Prior Art Navigation" sidebar (in markdown). Map specific claim limitations to prior art gaps.
                       Example: "While medication adherence monitoring is crowded (10 prior art hits), Claim 1's 'local encrypted data store' limitation distinguishes from cloud-based competitors (e.g., AiCure)."
competitive_positioning_map — A "Competitive Positioning Map" — a 2x2 matrix (markdown table).
                              Show X/Y axes (e.g. Privacy vs. Clinical Integration). Point out your invention vs competitors.

Return ONLY valid JSON — no markdown fences, no explanation.
"""


_SPEC_HUMAN = """\
Study this complete example. Then write the same quality specification for INVENTION 2.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLE 1 — INVENTION: On-device pharmaceutical ingestion verification via camera + ML

COMPLETE SPECIFICATION OUTPUT:
{{
  "title": "On-Device Pharmaceutical Ingestion Verification System and Method",

  "abstract": "A system and method for verifying pharmaceutical ingestion using an on-device machine-learning inference module are disclosed. An optical sensor of a mobile computing device captures a video stream of a user's oropharyngeal region during a scheduled medication administration event. A machine-learning inference module, executing entirely on the device processor, analyzes successive image frames to detect a swallowing-motion event and generates an ingestion verification record comprising a confidence score and timestamp. The ingestion verification record is persisted in an on-device local encrypted data store without transmission to any external network or server, preserving patient biometric privacy while providing objective, passive medication adherence verification operable on standard consumer hardware.",

  "background": "The present invention relates generally to digital health monitoring and, more specifically, to computer-implemented systems and methods for verifying pharmaceutical ingestion using on-device machine-learning inference applied to video streams captured by an integrated optical sensor of a mobile computing device.\\n\\nExisting approaches to medication adherence monitoring suffer from well-documented limitations that collectively prevent reliable, scalable objective adherence tracking in standard outpatient care settings. Manual self-reporting and paper-based medication diaries are inherently subjective and susceptible to patient misrepresentation, producing adherence estimates that are systematically biased toward over-reporting. Smart-pill technologies that embed radio-frequency or electrochemical sensors in pharmaceutical capsules impose significant per-dose hardware costs and require pharmaceutical manufacturing process modifications that are prohibitive for widespread adoption across diverse medication classes. Remote video observation platforms that stream patient video to third-party servers for pharmacist or caregiver review introduce patient privacy risks and are subject to complex multi-jurisdictional health data sovereignty regulations. Accordingly, there is a need in the art for a passive, objective, on-device system for verifying pharmaceutical ingestion that operates without dedicated hardware peripherals and without biometric data transmission to external parties.",

  "summary_of_invention": "In one aspect, the invention provides a system for verifying pharmaceutical ingestion comprising: a mobile computing device having an optical sensor and a processor; a machine-learning inference module stored in non-transitory memory and configured to receive image frames from the optical sensor, detect swallowing-motion events, and generate ingestion verification records comprising confidence scores and timestamps; and a local encrypted data store configured to persist the ingestion verification records exclusively on the mobile computing device without transmission to external networks.\\n\\nThe invention achieves one or more of the following advantages over prior art systems: (i) passive, objective ingestion verification based on optical biometric analysis of the oropharyngeal region eliminates reliance on patient self-reporting and reduces systematic adherence over-reporting bias; (ii) on-device inference architecture eliminates transmission of patient biometric video data to external networks, enabling HIPAA-compliant deployment without cloud data processing agreements; (iii) operation on standard consumer mobile computing devices eliminates dedicated hardware peripheral requirements, enabling population-scale deployment at negligible per-patient infrastructure cost; and (iv) the optional FHIR-compliant structured reporting interface enables integration with electronic health record systems for clinical workflow consumption without mandating continuous data transmission.",

  "detailed_description": "Referring now to FIG. 1, a system architecture block diagram illustrates the principal hardware components... [abbreviated for example]",

  "prior_art_navigation": "### Navigation Strategy\\nWhile medication adherence monitoring is crowded (over 10 retrieved hits in smart pills and video observation), our primary independent claims specifically navigate these risks.\\n\\n- **Privacy and Data Sovereignty**: Unlike prior art systems relying on cloud-based telemedicine video storage (e.g., US_202029342_A1), Claim 1’s limitation to \"persist the ingestion verification records exclusively on the mobile computing device without transmission to external networks\" avoids the crowded cloud-based adherence field completely.\\n- **Hardware Cost Gap**: While sensor-embedded pill patent US_2019409893 requires specialized ingested hardware, Claim 5 establishes an alternative vector entirely through standard mobile optical sensors.",

  "competitive_positioning_map": "| Quadrant | Enterprise / Clinical Integration | Consumer / Low Integration |\\n|---|---|---|\\n| **High Privacy (Local Device)** | **Our Invention (Top Right)** | Offline Habit Trackers |\\n| **Low Privacy (Cloud-Based)** | AiCure, Proteus Health | Regular Pill Reminder Apps |"
}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INVENTION 2 — Write the complete specification JSON now using the same quality.

Decomposed Disclosure Sections:

Background Context:
{background}

Problem Being Solved:
{problem}

Core Invention (the mechanism):
{core}

Embodiments and Implementation Variants:
{embodiments}

Advantages Over Prior Art:
{advantages}

Target Use Cases and Markets:
{use_cases}

Figure Descriptions:
{figures}

Prior Art Retrieved (Step 2):
{prior_art}

Independent Claim (Claim 1) — Every element must appear in detailed_description:
{claim_1}

RULES FOR THIS SPECIFICATION:
- background MUST NOT mention this invention or its solution
- detailed_description MUST start "Referring now to FIG. 1..."
- detailed_description MUST include "In an alternative embodiment..."
- Every element of Claim 1 above MUST appear explicitly in the detailed_description
- Return ONLY the JSON object — no markdown, no preamble
"""


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

async def generate_specification(
    disclosure: DecomposedDisclosure,
    analyses: List[SectionAnalysis],
    claim_set: ClaimSet,
    llm: Any,
) -> DraftSpecification:
    """
    Assemble a complete patent specification from prior pipeline steps.

    Args:
        disclosure: Decomposed sections from Step 1.
        analyses:   Analysis results from Step 2.
        claim_set:  Strategic claims from Step 3.
        llm:        LLM for spec generation (capable writing model).

    Returns:
        A DraftSpecification ready for attorney review.
    """
    def _sec(name: SectionName) -> str:
        s = disclosure.get_section(name)
        return s.text[:2000] if s else "[Not provided]"

    prior_art_str = _format_prior_art(analyses)
    independent_claim = next(
        (c for c in claim_set.claims if c.claim_type.value == "independent"), None
    )
    claim_1_text = independent_claim.text if independent_claim else "[No independent claim generated]"

    messages = [
        SystemMessage(content=_SPEC_SYSTEM),
        HumanMessage(content=_SPEC_HUMAN.format(
            background=_sec(SectionName.BACKGROUND),
            problem=_sec(SectionName.PROBLEM_STATEMENT),
            core=_sec(SectionName.CORE_INVENTION),
            embodiments=_sec(SectionName.EMBODIMENTS),
            advantages=_sec(SectionName.ADVANTAGES),
            use_cases=_sec(SectionName.USE_CASES),
            figures=_sec(SectionName.FIGURES_DESCRIPTION),
            prior_art=prior_art_str[:1500],
            claim_1=claim_1_text[:600],
        )),
    ]

    response = await llm.ainvoke(messages)
    content = response.content.strip()

    # Parse JSON
    import json
    cleaned = re.sub(r"```(?:json)?\s*", "", content).strip().rstrip("`").strip()
    parsed = None
    try:
        parsed = json.loads(cleaned)
    except Exception:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
            except Exception:
                pass

    if not parsed:
        # Graceful fallback: build a minimal spec from disclosure sections
        parsed = _fallback_spec(_sec(SectionName.CORE_INVENTION), claim_1_text)

    # Build provenance map
    provenance_map = {
        "title": ["core_invention"],
        "abstract": ["core_invention", "advantages"],
        "background": ["background", "problem_statement"],
        "summary_of_invention": ["core_invention", "advantages", "use_cases"],
        "detailed_description": ["embodiments", "figures_description", "core_invention"],
        "claims": ["core_invention", "embodiments"],
    }

    return DraftSpecification(
        title=_clean_title(parsed.get("title", "Invention Title")),
        abstract=_truncate_abstract(parsed.get("abstract", "")),
        background=parsed.get("background", _sec(SectionName.BACKGROUND)),
        summary_of_invention=parsed.get("summary_of_invention", ""),
        detailed_description=parsed.get("detailed_description", ""),
        prior_art_navigation=parsed.get("prior_art_navigation", ""),
        competitive_positioning_map=parsed.get("competitive_positioning_map", ""),
        claims=claim_set.claims,
        provenance_map=provenance_map,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_prior_art(analyses: List[SectionAnalysis]) -> str:
    lines = []
    seen_ids: set = set()
    for a in analyses:
        for ref in a.prior_art_hits[:3]:
            if ref.patent_id not in seen_ids:
                seen_ids.add(ref.patent_id)
                lines.append(
                    f"- {ref.patent_id}: {ref.title or 'N/A'} "
                    f"(relevance: {ref.relevance_score:.2f})"
                )
    return "\n".join(lines) if lines else "No prior art retrieved."


def _clean_title(title: str) -> str:
    # Remove leading "A method for..." if present
    title = (title or "").strip().strip('"').strip("'")
    if len(title) > 120:
        title = title[:120].rsplit(" ", 1)[0]
    return title or "System and Method of the Invention"


def _truncate_abstract(abstract: str) -> str:
    words = abstract.split()
    if len(words) > 150:
        abstract = " ".join(words[:150]) + "..."
    return abstract


def _fallback_spec(core_text: str, claim_1: str) -> dict:
    return {
        "title": "System and Method of the Invention",
        "abstract": (
            f"The present invention relates to {core_text[:200].rstrip('.')}. "
            "The invention provides improvements over the prior art through novel "
            "technical approaches described herein."
        ),
        "background": (
            "Prior art solutions in this field suffer from known limitations. "
            "There remains a need for improved approaches that address the problems "
            "identified in the disclosure."
        ),
        "summary_of_invention": (
            f"The present invention provides {core_text[:300].rstrip('.')}. "
            "The invention achieves this through the mechanisms described in the "
            "detailed description below."
        ),
        "detailed_description": (
            f"Referring now to FIG. 1, the preferred embodiment of the invention "
            f"comprises the following elements: {core_text[:500]}. "
            "In an alternative embodiment, the system may be configured differently "
            "as described in the claims."
        ),
        "prior_art_navigation": "### Navigation Strategy\nThis invention employs specific claim limitations to effectively separate from prior art constraints.",
        "competitive_positioning_map": "| Positioning | Primary | Alternative |\n|---|---|---|\n| Advantage | Current Invention | Prior Art |"
    }
