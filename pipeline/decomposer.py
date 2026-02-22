"""
Step 1 – Disclosure Decomposer.

Takes raw inventor input (already extracted to plain text) and uses an LLM
to split it into the 7 canonical patent sections, each with a confidence score
and a provenance span back to the original text.

Design notes:
- The LLM returns structured JSON; we validate it with Pydantic.
- Provenance spans are estimated via substring search (best-effort). If the LLM
  paraphrases heavily the span may be approximate—the excerpt is authoritative.
- Audit logging is the caller's responsibility (orchestrator).
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from pipeline.models import (
    DecomposedDisclosure,
    DisclosureSection,
    ProvenanceSpan,
    SectionName,
)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_DECOMPOSE_SYSTEM = """\
You are a senior patent prosecution attorney. Transform raw inventor disclosures into
seven canonical patent specification sections in formal, third-person USPTO prose.

ABSOLUTE RULES:
1. NEVER copy-paste or minimally rephrase inventor language. FULLY REWRITE every section.
2. Each section is DISTINCT — zero repeated content between sections:
   background        = field + prior art problems ONLY (never mention this invention)
   problem_statement = specific technical gap in prior art ONLY (no solution)
   core_invention    = novel mechanism ONLY (no background recap, no problem recap)
   embodiments       = 2-3 variants using "In a first embodiment..." structure
   advantages        = numbered benefits vs. prior art (i, ii, iii...)
   use_cases         = named markets, user classes, regulatory contexts
   figures_description = FIG. 1, FIG. 2 descriptions inferred from disclosure
3. MINIMUM TEXT LENGTH: each "text" field must be at least 3 complete sentences.
4. Return ONLY valid JSON — no markdown, no explanation.
"""

_DECOMPOSE_HUMAN = """\
EXAMPLE — Study this transformation, then do the same for INPUT below.

RAW INVENTOR INPUT:
"My app uses the phone's camera to figure out if someone took their pill.
The camera watches your throat and sees if you swallow. Right now patients
just lie to their doctors about taking meds. My system logs everything
locally so privacy is protected. Doctors can optionally see a report."

CORRECT OUTPUT — 7 formal patent sections:
{{
  "sections": [
    {{
      "section_name": "background",
      "text": "The field of digital health monitoring faces a persistent challenge in medication non-adherence, wherein patients across chronic disease populations fail to take prescribed medications at the correct dose and schedule. Existing monitoring approaches — including self-reported patient diaries, manual pill-count reconciliation, smart-pill blister packs, and telepharmacy video consultations — uniformly rely on active patient participation or dedicated hardware peripherals. These approaches impose significant logistical burdens, require centralized data infrastructure, and are systematically susceptible to patient misrepresentation, rendering accurate population-level adherence tracking infeasible in standard outpatient care settings.",
      "confidence": 0.82,
      "excerpt": "patients just lie to their doctors"
    }},
    {{
      "section_name": "problem_statement",
      "text": "A critical unresolved technical problem in medication adherence monitoring is the absence of an objective, passive verification mechanism that can confirm the physical act of pharmaceutical ingestion without specialized hardware, without transmitting sensitive patient biometrics to remote servers, and without requiring active behavioral change from the patient. Prior art solutions that rely on patient self-reporting introduce systematic bias, while hardware-dependent solutions such as embedded sensor capsules impose prohibitive per-dose costs and are unsuitable for long-term chronic disease management. No existing system provides passive, on-device, real-time ingestion verification operable on standard consumer mobile hardware.",
      "confidence": 0.90,
      "excerpt": "just lie to their doctors about taking"
    }},
    {{
      "section_name": "core_invention",
      "text": "The present invention provides a computer-implemented system and method for verifying pharmaceutical ingestion using an optical sensor integrated into a standard consumer mobile computing device. During a scheduled medication administration window, the system activates the device front-facing optical sensor to capture a real-time video stream depicting the oropharyngeal region of the user. A machine-learning inference module, executing entirely within the device processor and operating on locally stored non-transitory memory, analyzes successive video frames to detect characteristic swallowing-motion patterns indicative of pill ingestion and generates a confidence-scored binary ingestion classification. All inference computations and resulting ingestion verification records are retained exclusively on the device, precluding transmission of biometric data or protected health information to any external network or server.",
      "confidence": 0.95,
      "excerpt": "camera watches your throat and sees"
    }},
    {{
      "section_name": "embodiments",
      "text": "In a first embodiment, the machine-learning inference module comprises a two-stage convolutional neural network: a first stage configured to detect anatomical landmarks of the oropharyngeal region within each video frame, and a second stage configured to classify temporal motion sequences across consecutive frames as indicative of a swallowing event with a classification confidence score exceeding a configurable threshold. In a second embodiment, the system further comprises a medication schedule module that retrieves the user-specific dosing schedule from local encrypted storage and automatically activates the optical sensor within a configurable time window centered on each scheduled administration event, without requiring manual initiation by the user. In a third embodiment, the on-device ingestion verification records are structured according to a FHIR-compliant data schema and may be selectively exported to an authorized clinician-facing dashboard upon receipt of an explicit, revocable user consent signal.",
      "confidence": 0.78,
      "excerpt": "logs everything locally so privacy is protected"
    }},
    {{
      "section_name": "advantages",
      "text": "The present invention confers the following advantages over prior art medication adherence systems: (i) passive, objective ingestion verification based on optical biometric analysis eliminates the systematic inaccuracy inherent in patient self-reporting; (ii) on-device inference architecture eliminates transmission of patient biometric video data to external servers, preserving compliance with HIPAA data minimization requirements; (iii) operation on standard consumer smartphones eliminates the need for dedicated monitoring hardware peripherals, enabling population-scale deployment at negligible per-patient cost; and (iv) the optional FHIR-compliant clinician reporting module enables integration with existing electronic health record systems without mandating continuous data transmission.",
      "confidence": 0.88,
      "excerpt": "Doctors can optionally see a report"
    }},
    {{
      "section_name": "use_cases",
      "text": "The system is applicable across chronic disease management programs wherein medication adherence represents a primary determinant of clinical outcome, including oncology chemotherapy regimens, solid-organ transplant immunosuppression protocols where missed doses carry acute rejection risk, HIV antiretroviral therapy programs, and psychiatric pharmacotherapy management for conditions including major depressive disorder. The system further serves managed care organizations and pharmacy benefit managers requiring objective adherence metrics for outcomes-based drug reimbursement contracts, and contract research organizations conducting Phase III-IV clinical trials that mandate independently verified adherence data.",
      "confidence": 0.70,
      "excerpt": "Doctors can optionally see a report"
    }},
    {{
      "section_name": "figures_description",
      "text": "FIG. 1 is a system architecture block diagram illustrating the principal components of the mobile computing device embodiment, comprising: the front-facing optical sensor module, the on-device machine-learning inference module, the medication schedule module, the local encrypted data store, and the optional FHIR-compliant clinician reporting interface with user consent gate. FIG. 2 is a process flow diagram of the ingestion verification method, depicting the sequential steps of: (a) receiving a scheduled administration trigger; (b) activating the optical sensor and initiating video frame capture; (c) executing the two-stage swallowing-motion classification pipeline; (d) generating a confidence-scored ingestion record; and (e) persisting the timestamped record to local encrypted storage.",
      "confidence": 0.55,
      "excerpt": "My app uses the phone's camera"
    }}
  ]
}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUT — Raw inventor disclosure to transform:
{raw_input}

OUTPUT — Produce the same quality JSON for the input above.
All 7 sections. Each "text" field: formal third-person patent prose, minimum 3 sentences.
NEVER copy-paste inventor words into "text". Return ONLY JSON.
"""


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------



async def decompose_disclosure(
    raw_text: str,
    llm: Any,
    tenant_id: Optional[str] = None,
) -> DecomposedDisclosure:
    """
    Decompose raw inventor text into 7 canonical patent sections.

    Args:
        raw_text:  The plain text of the inventor's disclosure.
        llm:       A LangChain chat model instance.
        tenant_id: ENTERPRISE BRIDGE – tenant identifier passed through for audit.

    Returns:
        A validated DecomposedDisclosure with all 7 sections.
    """
    raw_hash = hashlib.sha256(raw_text.encode()).hexdigest()

    messages = [
        SystemMessage(content=_DECOMPOSE_SYSTEM),
        HumanMessage(content=_DECOMPOSE_HUMAN.format(raw_input=raw_text[:12000])),
    ]

    response = await llm.ainvoke(messages)
    content = response.content.strip()

    # Parse JSON – strip markdown fences if model wraps output
    parsed = _parse_json(content)

    # Build validated section objects
    sections: list[DisclosureSection] = []
    seen_names: set[str] = set()

    raw_sections: list[dict] = parsed.get("sections", []) if parsed else []

    for raw_sec in raw_sections:
        name_str = raw_sec.get("section_name", "").strip().lower()
        try:
            section_name = SectionName(name_str)
        except ValueError:
            # Map near-matches
            section_name = _fuzzy_section_name(name_str)
            if section_name is None:
                continue

        if section_name.value in seen_names:
            continue
        seen_names.add(section_name.value)

        text = raw_sec.get("text", "").strip()
        confidence = float(raw_sec.get("confidence", 0.3))
        confidence = max(0.0, min(1.0, confidence))
        excerpt = raw_sec.get("excerpt", text[:60])

        # Find provenance span via substring search
        span = _find_span(raw_text, excerpt)

        section = DisclosureSection(
            section_name=section_name,
            text=text if text else f"[{section_name.value} not described in disclosure]",
            confidence=confidence,
            provenance=span,
        )
        sections.append(section)

    # Ensure all 7 sections are present (pad with stubs for missing ones)
    for sn in SectionName:
        if sn.value not in seen_names:
            sections.append(
                DisclosureSection(
                    section_name=sn,
                    text=f"[{sn.value} not described in the disclosure]",
                    confidence=0.1,
                    provenance=ProvenanceSpan(
                        char_start=0, char_end=min(80, len(raw_text)),
                        excerpt=raw_text[:80]
                    ),
                )
            )

    # Sort in canonical order
    order = [s.value for s in SectionName]
    sections.sort(key=lambda s: order.index(s.section_name.value))

    return DecomposedDisclosure(
        tenant_id=tenant_id,
        raw_input_hash=raw_hash,
        raw_input_length=len(raw_text),
        sections=sections,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json(content: str) -> Optional[Dict]:
    """Robustly parse JSON from LLM output, stripping markdown fences."""
    # Remove markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", content).strip().rstrip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find first {...} block
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


_FUZZY_MAP = {
    "background_of_invention": SectionName.BACKGROUND,
    "prior art": SectionName.BACKGROUND,
    "field": SectionName.BACKGROUND,
    "problem": SectionName.PROBLEM_STATEMENT,
    "technical problem": SectionName.PROBLEM_STATEMENT,
    "invention": SectionName.CORE_INVENTION,
    "summary": SectionName.CORE_INVENTION,
    "embodiment": SectionName.EMBODIMENTS,
    "example": SectionName.EMBODIMENTS,
    "advantage": SectionName.ADVANTAGES,
    "benefit": SectionName.ADVANTAGES,
    "use case": SectionName.USE_CASES,
    "application": SectionName.USE_CASES,
    "figure": SectionName.FIGURES_DESCRIPTION,
    "drawing": SectionName.FIGURES_DESCRIPTION,
}


def _fuzzy_section_name(name: str) -> Optional[SectionName]:
    for key, value in _FUZZY_MAP.items():
        if key in name:
            return value
    return None


def _find_span(raw_text: str, excerpt: str) -> ProvenanceSpan:
    """Find the character span of an excerpt in the raw text (best-effort)."""
    if not excerpt:
        return ProvenanceSpan(char_start=0, char_end=min(80, len(raw_text)), excerpt=raw_text[:80])

    idx = raw_text.find(excerpt)
    if idx >= 0:
        return ProvenanceSpan(
            char_start=idx,
            char_end=idx + len(excerpt),
            excerpt=excerpt,
        )

    # Case-insensitive fallback
    idx_lower = raw_text.lower().find(excerpt.lower())
    if idx_lower >= 0:
        return ProvenanceSpan(
            char_start=idx_lower,
            char_end=idx_lower + len(excerpt),
            excerpt=raw_text[idx_lower: idx_lower + len(excerpt)],
        )

    # Last resort – return beginning of document
    return ProvenanceSpan(
        char_start=0,
        char_end=min(len(excerpt), len(raw_text)),
        excerpt=raw_text[:len(excerpt)] if raw_text else "",
    )
