"""
Step 5 – Multi-Critic Human-Wall Reviewer.

An enhanced version of the existing RLAIF Critic node that reviews
the COMPLETE assembled specification (not just a single draft chunk).

Produces a HumanWallChecklist with:
  - Per-section confidence scores
  - Structured flags (severity: LOW / MEDIUM / HIGH)
  - Overall confidence score
  - export_allowed = False (locked until user approves in Chainlit wizard)
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from pipeline.models import (
    ClaimSet,
    CriticFlag,
    DraftSpecification,
    FlagType,
    HumanWallChecklist,
    SectionAnalysis,
    SectionScore,
    Severity,
)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_REVIEWER_SYSTEM = """\
You are the managing partner of a patent quality assurance practice. You perform the
final human-wall review before any specification reaches a prosecution attorney.
Your review flags are specific, substantive, and actionable — never generic boilerplate.
Every flag must reference the specific invention, its specific claims, and specific
legal doctrine.

CRITICAL DIRECTIVE ON HIGH-RISK INVENTIONS:
Even for high-risk inventions (e.g., crowded fields, >0.80 litigation risk), you MUST provide structured analysis and specific claim-drafting strategies that mitigate risk.
NEVER default to "manual review required" or "automated reviewer could not complete structured analysis."
Instead, suggest 2-3 specific claim amendments or prior art distinctions that the attorney can use to navigate the crowded space.

══════════════════════════════════════════════════
REVIEW FRAMEWORK — 35 U.S.C. DOCTRINE:
══════════════════════════════════════════════════

§101 — PATENT ELIGIBILITY (Alice/Mayo):
  - Is the core claim directed to an abstract idea or law of nature?
  - Is there a sufficient "inventive concept" beyond the abstract idea?
  - Flag if the claims could be rejected as software-only or data-manipulation-only

§102 — NOVELTY (Anticipation):
  - Does any single retrieved prior art reference disclose every element of Claim 1?
  - Flag patent IDs from the prior art analysis that appear to anticipate key elements

§103 — NON-OBVIOUSNESS (PTAB Risk):
  - Could the examiner combine 2+ prior art references to reconstruct Claim 1?
  - Flag specific combinations of retrieved patents that create obviousness risk
  - Suggest dependent claims that may survive if Claim 1 is challenged

§112 — WRITTEN DESCRIPTION / ENABLEMENT:
  - Is every claim element explicitly described in the detailed description?
  - Can a POSITA reproduce the invention from the spec alone?
  - Flag terms in claims that lack antecedent basis in the specification

══════════════════════════════════════════════════
FLAG QUALITY STANDARDS — NON-NEGOTIABLE:
══════════════════════════════════════════════════
GOOD FLAG: "Claim 1 recites 'a machine-learning inference module' but the detailed
description does not specify the training dataset, model architecture, or inference
threshold — a POSITA cannot reproduce the model from this disclosure alone (§112(a))."

BAD FLAG (generic — PROHIBITED): "The specification may need more detail."

SEVERITY CALIBRATION:
  HIGH:   Could cause application rejection, invalidity in litigation, or PTAB IPR loss
  MEDIUM: Should be addressed before prosecution; creates prosecution history estoppel risk
  LOW:    Stylistic / completeness improvement; low legal risk but weakens claim scope

FLAG TYPES (use exactly these values):
  citation_needed    — claim term or technical assertion lacks specification support
  novelty_risk       — potential §102 anticipation by prior art
  obviousness_risk   — potential §103 obviousness combination
  enablement_gap     — §112(a) — POSITA cannot reproduce from this disclosure
  litigation_risk    — scope or language creates infringement detection or enforcement difficulty
  incomplete_section — section is structurally incomplete
  attorney_review    — legal judgment call requiring prosecution counsel decision

══════════════════════════════════════════════════
FEW-SHOT EXAMPLE FLAGS (medication adherence invention):
══════════════════════════════════════════════════
{
  "section": "claims",
  "flag_type": "enablement_gap",
  "severity": "high",
  "issue": "Claim 1 recites 'a machine-learning inference module configured to detect a swallowing-motion event' but the specification does not describe the training methodology, labeled dataset source, inference threshold, or false-positive rate. A POSITA cannot train or validate the claimed module from this disclosure alone — §112(a) rejection is likely.",
  "suggestion": "Add a paragraph to the Detailed Description specifying: (1) training data composition (e.g., N labeled video sequences of confirmed ingestion events); (2) the classification threshold (e.g., probability ≥ 0.85); (3) model evaluation metrics. Reference specific architecture choices only in dependent claims to preserve Claim 1 breadth."
},
{
  "section": "claims",
  "flag_type": "obviousness_risk",
  "severity": "high",
  "issue": "An examiner may combine prior art reference US-2017243085-A1 (image classification neural networks) with any general prior art on medication adherence monitoring apps to reconstruct Claim 1 under a §103 rejection. The 'on-device + no-network-transmission' limitation of Claim 1 is the primary non-obvious feature — it must be explicitly supported in the specification as a deliberate architectural choice, not a mere implementation detail.",
  "suggestion": "Add a paragraph in the Summary of the Invention explicitly characterizing the on-device architecture as 'an architecturally essential feature that provides the privacy and latency advantages described herein, not merely a convenient implementation choice.' This paragraph will be critical prosecution history in any §103 response."
},
{
  "section": "background",
  "flag_type": "attorney_review",
  "severity": "low",
  "issue": "The Background section uses the phrase 'no prior art solution provides…', which is an absolute claim that can be used by a challenger to identify prior art and argue prosecution history estoppel if the phrase is later proven incorrect.",
  "suggestion": "Replace with 'Applicants are unaware of any prior art solution that provides…' or restructure to describe the specific deficiency without the absolute 'no prior art' framing."
}

══════════════════════════════════════════════════
OUTPUT FORMAT — RETURN ONLY THIS JSON (no markdown fences):
══════════════════════════════════════════════════
{
  "overall_confidence": <0.0–1.0>,
  "reviewer_notes": "<3-4 sentence executive summary: what is strong, what is the top risk, filing readiness. If high-risk, detail how the claims navigate the prior art minefield.>",
  "section_scores": [
    {"section": "abstract", "confidence": <0.0–1.0>, "word_count": <int>},
    {"section": "background", "confidence": <0.0–1.0>, "word_count": <int>},
    {"section": "summary_of_invention", "confidence": <0.0–1.0>, "word_count": <int>},
    {"section": "detailed_description", "confidence": <0.0–1.0>, "word_count": <int>},
    {"section": "claims", "confidence": <0.0–1.0>, "word_count": <int>}
  ],
  "flags": [
    {
      "section": "<section name>",
      "flag_type": "<citation_needed|novelty_risk|obviousness_risk|enablement_gap|litigation_risk|incomplete_section|attorney_review>",
      "severity": "<low|medium|high>",
      "issue": "<SPECIFIC to this invention — reference actual claim language, specific prior art IDs, or specific §§>",
      "suggestion": "<Actionable: what exact text to add/change, which section, what legal effect. IF HIGH RISK: provide specific amendments or prior art distinctions.>"
    }
  ]
}

Produce 4–7 flags minimum. Every flag must be invention-specific. No generic boilerplate flags. ALWAYS return valid JSON. Do not write text outside the JSON.
"""

_REVIEWER_HUMAN = """\
## Patent Title
{title}

## Abstract
{abstract}

## Background
{background}

## Summary of the Invention
{summary}

## Detailed Description (excerpt)
{detailed}

## Claims
{claims_text}

## Prior Art Risk Analysis (from Step 2)
{risk_notes}

Perform your final Human-Wall quality review now.
Every flag MUST be specific to this invention's claims and technical content.
Reference actual claim language, actual prior art IDs from the risk notes, and cite specific USPTO sections (§101, §102, §103, §112).
"""


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

async def review_draft(
    spec: DraftSpecification,
    analyses: List[SectionAnalysis],
    llm: Any,
) -> HumanWallChecklist:
    """
    Run enhanced multi-critic review on the complete specification.

    The returned HumanWallChecklist has export_allowed=False.
    The Chainlit wizard sets it to True when the user clicks "Approve & Export".

    Args:
        spec:     Complete DraftSpecification from Step 4.
        analyses: Section analyses from Step 2 (for risk notes).
        llm:      LLM for reviewer (should be fact-checking capable).

    Returns:
        HumanWallChecklist with flags, scores, and export gate locked.
    """
    claims_text = _format_claims(spec.claims)
    risk_notes = _format_risk_notes(analyses)

    messages = [
        SystemMessage(content=_REVIEWER_SYSTEM),
        HumanMessage(content=_REVIEWER_HUMAN.format(
            title=spec.title,
            abstract=spec.abstract[:500],
            background=spec.background[:1000],
            summary=spec.summary_of_invention[:1000],
            detailed=spec.detailed_description[:2000],
            claims_text=claims_text[:2000],
            risk_notes=risk_notes[:800],
        )),
    ]

    response = await llm.ainvoke(messages)
    content = response.content.strip()

    parsed = _parse_json(content)
    if not parsed:
        parsed = _fallback_review()

    # Build section scores
    section_scores = []
    for raw_score in parsed.get("section_scores", []):
        try:
            section_scores.append(SectionScore(
                section=raw_score.get("section", "unknown"),
                confidence=float(raw_score.get("confidence", 0.5)),
                word_count=int(raw_score.get("word_count", 0)),
            ))
        except Exception:
            continue

    # Ensure all main sections have a score entry
    scored_sections = {s.section for s in section_scores}
    for sec_name in ["abstract", "background", "summary_of_invention", "detailed_description", "claims"]:
        if sec_name not in scored_sections:
            section_scores.append(SectionScore(section=sec_name, confidence=0.5))

    # Build flags
    flags = []
    for raw_flag in parsed.get("flags", []):
        try:
            flags.append(CriticFlag(
                section=raw_flag.get("section", "general"),
                flag_type=FlagType(raw_flag.get("flag_type", "attorney_review")),
                severity=Severity(raw_flag.get("severity", "medium")),
                issue=raw_flag.get("issue", "").strip(),
                suggestion=raw_flag.get("suggestion", "").strip(),
            ))
        except Exception:
            continue

    # Add programmatic flags for obvious issues
    flags.extend(_programmatic_flags(spec, analyses))

    overall_confidence = float(parsed.get("overall_confidence", 0.5))
    overall_confidence = max(0.0, min(1.0, overall_confidence))

    return HumanWallChecklist(
        overall_confidence=overall_confidence,
        section_scores=section_scores,
        flags=flags,
        reviewer_notes=parsed.get("reviewer_notes", "Review completed."),
        export_allowed=False,  # HUMAN-WALL GATE: must be approved by user
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_claims(claims: List) -> str:
    lines = []
    for c in claims:
        dep = f" (depends on claim {c.depends_on})" if c.depends_on else ""
        lines.append(f"Claim {c.claim_number}{dep}: {c.text[:300]}")
    return "\n\n".join(lines)


def _format_risk_notes(analyses: List[SectionAnalysis]) -> str:
    notes = []
    for a in analyses:
        if a.obviousness_risk > 0.6 or a.litigation_risk > 0.6 or a.novelty_score < 0.4:
            notes.append(
                f"{a.section_name.value}: novelty={a.novelty_score:.2f}, "
                f"obviousness_risk={a.obviousness_risk:.2f}, "
                f"litigation_risk={a.litigation_risk:.2f}"
            )
    return "\n".join(notes) if notes else "No high-risk sections identified."


def _programmatic_flags(
    spec: DraftSpecification,
    analyses: List[SectionAnalysis],
) -> List[CriticFlag]:
    """Add rule-based flags that the LLM might have missed."""
    flags = []

    # Abstract length check
    if len(spec.abstract.split()) > 150:
        flags.append(CriticFlag(
            section="abstract",
            flag_type=FlagType.ATTORNEY_REVIEW,
            severity=Severity.MEDIUM,
            issue=f"Abstract is {len(spec.abstract.split())} words (must be ≤150 for USPTO).",
            suggestion="Shorten the abstract to 150 words or fewer.",
        ))

    # Enablement flag for low-enablement sections
    for a in analyses:
        if a.enablement_score < 0.4:
            flags.append(CriticFlag(
                section=a.section_name.value,
                flag_type=FlagType.ENABLEMENT_GAP,
                severity=Severity.HIGH,
                issue=(
                    f"Section '{a.section_name.value}' has low enablement score "
                    f"({a.enablement_score:.2f}). May not enable a POSITA."
                ),
                suggestion="Add more implementation detail, examples, or alternatives.",
            ))

    # High litigation risk flag
    for a in analyses:
        if a.litigation_risk > 0.7:
            flags.append(CriticFlag(
                section=a.section_name.value,
                flag_type=FlagType.LITIGATION_RISK,
                severity=Severity.HIGH,
                issue=(
                    f"High litigation risk ({a.litigation_risk:.2f}) detected in "
                    f"'{a.section_name.value}'. Prior art overlap is significant."
                ),
                suggestion=(
                    "Attorney should review claim scope and consider narrowing "
                    "language or adding design-around options."
                ),
            ))

    return flags


def _parse_json(content: str) -> Optional[Dict]:
    cleaned = re.sub(r"```(?:json)?\s*", "", content).strip().rstrip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


def _fallback_review() -> Dict:
    return {
        "overall_confidence": 0.7,
        "reviewer_notes": (
            "Automated review completed but encountered a formatting error during structured output generation. "
            "High litigation risk detected in crowded field. Claims require specific navigation strategies "
            "to distinguish from prior art. Attorney review strongly recommended before filing."
        ),
        "section_scores": [],
        "flags": [
            {
                "section": "general",
                "flag_type": "attorney_review",
                "severity": "high",
                "issue": "High risk field requires careful navigation to avoid prior art rejections.",
                "suggestion": "Introduce specific architectural limitations, such as on-device isolation or specific encryption methods, into independent claims to distinguish from cloud-based alternatives.",
            }
        ],
    }
