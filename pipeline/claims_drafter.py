"""
Step 3 – Strategic Claims Drafter.

Generates a market-protecting independent claim and 3–5 layered dependent
claims, each accompanied by plain-English strategy reasoning.

The drafter takes the decomposed disclosure + section analyses as context
so it can tailor claim breadth based on novelty + risk scores.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from pipeline.models import (
    ClaimSet,
    ClaimType,
    DecomposedDisclosure,
    PatentClaim,
    SectionAnalysis,
    SectionName,
)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_CLAIMS_SYSTEM = """\
You are a principal patent claims attorney. Write strategic, market-protecting patent
claims that survive IPR and generate licensing revenue.

MANDATORY RULES:
1. Claim 1 MUST be SYSTEM or METHOD — never "a system and method".
   Opens: "A system comprising:" or "A method comprising:" — not both.
2. Claim 1 captures MAXIMUM MARKET SCOPE — omit every detail not essential to novelty.
   Use PTAB-resilient language: "mobile computing device" not "smartphone";
   "machine-learning model" not "neural network"; "optical sensor" not "camera".
3. Write EXACTLY 6 claims: 1 independent + 5 dependent.
4. Each dependent claim protects a DISTINCT commercial embodiment or PTAB fallback.
5. strategy_reasoning MUST explain: (a) what market/use case this captures,
   (b) what invalidity or design-around risk it avoids, (c) licensing leverage.
6. Return ONLY valid JSON — no markdown fences.
"""

_CLAIMS_HUMAN = """\
Study these two examples carefully. Then draft 6 claims for INVENTION 3.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLE 1 — INVENTION: On-device medication ingestion verification via camera + ML

CLAIMS OUTPUT:
{{
  "claims": [
    {{
      "claim_number": 1,
      "claim_type": "independent",
      "depends_on": null,
      "text": "A system for verifying ingestion of a pharmaceutical agent, the system comprising: a mobile computing device having an optical sensor and a processor; a machine-learning inference module stored in non-transitory memory and executable by the processor, the module configured to: receive a sequence of image frames captured by the optical sensor depicting an oropharyngeal region of a user during a medication administration event; analyze the image frames to detect a swallowing-motion event indicative of pharmaceutical ingestion; and generate an ingestion verification record comprising a timestamp and a confidence score; and a local encrypted data store configured to persist the ingestion verification record exclusively on the mobile computing device without transmission to an external network.",
      "strategy_reasoning": "Drafted as a system claim using 'mobile computing device' rather than 'smartphone' to capture tablets, wearables, AR headsets, and future form factors — a Samsung Galaxy smartphone, an Apple Watch, and a Meta Ray-Ban frame all read on this claim. The three-element structure (sensor → inference module → local store) is the minimum necessary to capture the novelty and is deliberately silent on CNN architecture, training methodology, and UI — leaving those as dependent claim territory. The 'exclusively on the mobile computing device without transmission' limitation creates a privacy-moat that competitors cannot design around without fundamentally changing their product architecture, and creates a licensing necessity for any medication adherence app that processes biometric video locally.",
      "breadth_score": 0.93
    }},
    {{
      "claim_number": 2,
      "claim_type": "dependent",
      "depends_on": 1,
      "text": "The system of claim 1, wherein the machine-learning inference module comprises a first neural network stage configured to detect anatomical landmarks of the oropharyngeal region within each image frame, and a second neural network stage configured to classify temporal motion sequences across consecutive image frames as indicative of a swallowing event when the resulting classification probability exceeds a configurable detection threshold.",
      "strategy_reasoning": "Narrows to two-stage CNN architecture — inventive enough to survive §103 if Claim 1 falls to a combination attack pairing generic ML prior art with any adherence monitoring prior art. Broad enough to cover ResNet, EfficientNet, Vision Transformer, and any future attention-based architecture, because 'neural network stage' is agnostic to topology.",
      "breadth_score": 0.72
    }},
    {{
      "claim_number": 3,
      "claim_type": "dependent",
      "depends_on": 1,
      "text": "The system of claim 1, further comprising a medication schedule module configured to retrieve a user-specific dosing schedule from the local encrypted data store and to automatically activate the optical sensor within a configurable time window surrounding each scheduled pharmaceutical administration event without requiring manual initiation by the user.",
      "strategy_reasoning": "The scheduling-triggered proactive capture is the dominant commercial UX workflow — this claim ensures any pharma partner deployment that links adherence monitoring to a prescription schedule requires a license. The 'without requiring manual initiation' limitation specifically captures passive deployments, which represent the highest commercial value and is designed around by apps that require the user to tap 'start capturing'.",
      "breadth_score": 0.68
    }},
    {{
      "claim_number": 4,
      "claim_type": "dependent",
      "depends_on": 1,
      "text": "A method for verifying pharmaceutical ingestion using a mobile computing device, the method comprising: activating an optical sensor of the mobile computing device in response to a scheduled medication administration event; acquiring a sequence of image frames from the optical sensor depicting an oropharyngeal region of a user; executing, by a processor of the mobile computing device, a machine-learning inference model against the image frames to produce a swallowing-event classification and an associated confidence score; and storing an ingestion verification record comprising the swallowing-event classification, the confidence score, and a timestamp in a local encrypted data store of the mobile computing device.",
      "strategy_reasoning": "Method claim is essential for licensing assertions against app developers and care management platform operators who 'use the method' but do not 'make or sell a system'. Courts have repeatedly held that system and method claims cover different infringers — this claim captures the software-as-a-service delivery model where no hardware changes hands.",
      "breadth_score": 0.89
    }},
    {{
      "claim_number": 5,
      "claim_type": "dependent",
      "depends_on": 1,
      "text": "The system of claim 1, wherein the ingestion verification record is structured according to a FHIR Release 4 compliant data schema, and wherein the local encrypted data store is further configured to transmit the ingestion verification record to a remote clinician-authorized endpoint upon receipt of a digitally-signed, revocable user consent token.",
      "strategy_reasoning": "FHIR R4 compliance targets the healthcare interoperability market and creates licensing leverage over EHR integration vendors (Epic MyChart, Oracle Cerner) integrating adherence data into clinical workflows. The consent token requirement is an FDA Digital Health Software criterion and creates a compliance driver that makes licensing the path of least resistance for regulated medical app developers.",
      "breadth_score": 0.58
    }},
    {{
      "claim_number": 6,
      "claim_type": "dependent",
      "depends_on": 1,
      "text": "The system of claim 1, wherein the local encrypted data store implements AES-256 symmetric encryption at rest using a device-specific encryption key derived from the secure enclave of the mobile computing device, and wherein each ingestion verification record is cryptographically signed prior to storage to provide tamper-evident audit-chain integrity.",
      "strategy_reasoning": "The cryptographic audit-chain claim targets FDA 21 CFR Part 11-compliant electronic records contexts — clinical trial data, regulated DSAW submissions, and pharmaceutical SaaS platforms — where unsigned electronic records are categorically unacceptable. Creates a defensible claim position in the clinical trials software market independent of the commercial consumer market targeted by Claims 1–5.",
      "breadth_score": 0.47
    }}
  ]
}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLE 2 — INVENTION: Non-invasive wrist-worn blood glucose monitor using NIR spectroscopy

CLAIMS OUTPUT:
{{
  "claims": [
    {{
      "claim_number": 1,
      "claim_type": "independent",
      "depends_on": null,
      "text": "A system for non-invasive blood glucose monitoring, the system comprising: a wearable sensor device configured for transcutaneous attachment to a body surface of a user, the sensor device comprising: an illumination module configured to emit electromagnetic radiation at a plurality of wavelengths toward the body surface; a detection module configured to capture a reflectance spectrum of electromagnetic radiation returning from the body surface; and a signal processing module configured to derive a blood glucose concentration estimate from the captured reflectance spectrum using a personalized calibration model specific to the user; and a wireless communication module configured to transmit the blood glucose concentration estimate to a companion computing device.",
      "strategy_reasoning": "Drafted to capture any wearable form factor (wristband, patch, ring, earbud) by using 'wearable sensor device configured for transcutaneous attachment' rather than 'wristband'. 'Electromagnetic radiation at a plurality of wavelengths' covers NIR, MIR, and Raman spectroscopy without locking to specific wavelengths. 'Personalized calibration model' is the core novelty — individual tissue correction — and is deliberately broad enough to cover regression, neural network, and Gaussian process models. This claim creates a blocking position on any transcutaneous non-invasive glucose monitor that uses individual calibration.",
      "breadth_score": 0.91
    }},
    {{
      "claim_number": 2,
      "claim_type": "dependent",
      "depends_on": 1,
      "text": "The system of claim 1, wherein the illumination module comprises a plurality of near-infrared light-emitting diodes operating at wavelengths in the range of 1400 nm to 1800 nm, and wherein the detection module comprises a photodetector having a spectral response spanning the same wavelength range, the illumination module and detection module arranged in a reflectance geometry with a source-detector separation of between 2 mm and 10 mm.",
      "strategy_reasoning": "Narrows to NIR glucose absorption window — the technically validated spectroscopic approach — providing a fallback over any obviousness attack combining generic NIR spectroscopy prior art with non-invasive monitoring prior art. The source-detector separation range covers both single-point and multi-distance measurement geometries without specifying our preferred 4 mm value.",
      "breadth_score": 0.70
    }},
    {{
      "claim_number": 3,
      "claim_type": "dependent",
      "depends_on": 1,
      "text": "The system of claim 1, further comprising a motion artifact rejection module configured to receive output from an inertial sensor co-located with the sensor device and to exclude reflectance spectrum measurements acquired during detected body-motion events from input to the signal processing module.",
      "strategy_reasoning": "Motion rejection is a mandatory commercial feature for ambulatory accuracy — any wearable glucose monitor that is clinically validated in ambulatory settings must solve this problem. This dependent claim covers the dominant technical approach (IMU-gated sampling) and creates licensing necessity for any ambulatory-validated non-invasive glucose product.",
      "breadth_score": 0.65
    }},
    {{
      "claim_number": 4,
      "claim_type": "dependent",
      "depends_on": 1,
      "text": "A method for non-invasive blood glucose monitoring, the method comprising: emitting electromagnetic radiation at a plurality of wavelengths from a wearable sensor device toward a body surface of a user; capturing a reflectance spectrum of electromagnetic radiation returning from the body surface using a detection module of the wearable sensor device; applying a personalized calibration model, calibrated using at least one reference blood glucose measurement obtained from the user, to derive a blood glucose concentration estimate from the captured reflectance spectrum; and wirelessly transmitting the blood glucose concentration estimate to a companion computing device for display.",
      "strategy_reasoning": "Method claim independently covers the measurement process itself, enabling licensing assertions against service providers who operate non-invasive glucose monitoring services using wearable hardware made by a third party. Captures the SaaS clinical monitoring model where the platform provides the algorithm and calibration as a service independent of the hardware manufacturer.",
      "breadth_score": 0.87
    }},
    {{
      "claim_number": 5,
      "claim_type": "dependent",
      "depends_on": 1,
      "text": "The system of claim 1, wherein the personalized calibration model is updated periodically using a Bayesian parameter update procedure that incorporates a reference blood glucose measurement obtained via capillary sampling, such that the calibration model adapts to changes in the user's physiological tissue parameters over time.",
      "strategy_reasoning": "The Bayesian adaptive recalibration is the key commercial differentiator — it enables long-term accuracy without full recalibration sessions. Targets the continuous monitoring market where accuracy drift over weeks or months is the primary barrier to clinical adoption. Creates licensing leverage over any non-invasive glucose competitor using adaptive machine learning calibration.",
      "breadth_score": 0.60
    }},
    {{
      "claim_number": 6,
      "claim_type": "dependent",
      "depends_on": 1,
      "text": "The system of claim 1, wherein the companion computing device is further configured to: compare the derived blood glucose concentration estimate against user-specific hypoglycemia and hyperglycemia threshold values; and generate an alert notification transmitted to the user and, upon user consent, to a designated clinician device, when the blood glucose concentration estimate falls outside a target glycemic range.",
      "strategy_reasoning": "Alert generation with clinician notification targets the regulated medical device market, where automated glycemic alerts are a Class II FDA De Novo device feature. Creates a licensing position in the remote patient monitoring reimbursement space under CPT codes 99453–99458, which represents a multi-billion dollar annual billing market in the US alone.",
      "breadth_score": 0.52
    }}
  ]
}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INVENTION 3 — Draft 6 claims now using the same quality as Examples 1 and 2.

Core Invention:
{core_invention}

Problem Being Solved:
{problem_statement}

Embodiments and Variants:
{embodiments}

Prior Art Risk (from analysis):
{risk_summary}

RETURN ONLY THE JSON. Claim 1: broadest defensible scope, system OR method (not both).
Claims 2–6: each a distinct market protection layer or PTAB fallback.
strategy_reasoning: specific market, specific invalidity risk avoided, specific licensing leverage.
"""


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

async def draft_claims(
    disclosure: DecomposedDisclosure,
    analyses: List[SectionAnalysis],
    llm: Any,
) -> ClaimSet:
    """
    Generate strategic claims from the decomposed disclosure + analysis results.

    Args:
        disclosure: Decomposed disclosure sections from Step 1.
        analyses:   Section analysis results from Step 2.
        llm:        LLM for claims drafting (should be a capable model).

    Returns:
        A ClaimSet with 1 independent + 3-5 dependent claims.
    """
    core_text = _get_section_text(disclosure, SectionName.CORE_INVENTION)
    problem_text = _get_section_text(disclosure, SectionName.PROBLEM_STATEMENT)
    embodiment_text = _get_section_text(disclosure, SectionName.EMBODIMENTS)
    risk_summary = _build_risk_summary(analyses)

    messages = [
        SystemMessage(content=_CLAIMS_SYSTEM),
        HumanMessage(content=_CLAIMS_HUMAN.format(
            core_invention=core_text[:2500],
            problem_statement=problem_text[:1200],
            embodiments=embodiment_text[:2000],
            risk_summary=risk_summary[:1200],
        )),
    ]

    response = await llm.ainvoke(messages)
    content = response.content.strip()

    # Parse JSON
    parsed = _parse_json(content)
    claims = _build_claims(parsed)

    # Guarantee at least 1 independent + 4 dependent if LLM under-delivered
    if not any(c.claim_type == ClaimType.INDEPENDENT for c in claims):
        claims.insert(0, _fallback_independent_claim(core_text))
    if len(claims) < 5:
        claims.extend(_fallback_dependent_claims(core_text, len(claims)))

    return ClaimSet(claims=claims)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_section_text(disclosure: DecomposedDisclosure, name: SectionName) -> str:
    section = disclosure.get_section(name)
    return section.text if section else f"[{name.value} not available]"


def _build_risk_summary(analyses: List[SectionAnalysis]) -> str:
    lines = []
    for a in analyses:
        if a.novelty_score < 0.4 or a.obviousness_risk > 0.6:
            lines.append(
                f"- {a.section_name.value}: novelty={a.novelty_score:.2f}, "
                f"obviousness_risk={a.obviousness_risk:.2f}, "
                f"litigation_risk={a.litigation_risk:.2f}"
            )
    if not lines:
        lines.append("No high-risk sections identified – proceed with broad claims.")
    return "\n".join(lines)


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


def _build_claims(parsed: Optional[Dict]) -> List[PatentClaim]:
    if not parsed or "claims" not in parsed:
        return []
    claims = []
    for raw in parsed["claims"]:
        try:
            claim = PatentClaim(
                claim_number=int(raw.get("claim_number", 1)),
                claim_type=ClaimType(raw.get("claim_type", "independent")),
                depends_on=raw.get("depends_on"),
                text=raw.get("text", "").strip(),
                strategy_reasoning=raw.get("strategy_reasoning", "").strip(),
                breadth_score=float(raw.get("breadth_score", 0.5)),
            )
            claims.append(claim)
        except Exception:
            continue
    return sorted(claims, key=lambda c: c.claim_number)


def _fallback_independent_claim(core_text: str) -> PatentClaim:
    """Produce a properly-structured independent claim as a safety fallback."""
    # Truncate core text to a single coherent sentence for claim preamble
    core_snippet = core_text[:300].rstrip('.')
    return PatentClaim(
        claim_number=1,
        claim_type=ClaimType.INDEPENDENT,
        depends_on=None,
        text=(
            f"A system comprising: a processor; and a non-transitory computer-readable "
            f"medium storing instructions that, when executed by the processor, cause the "
            f"system to perform operations comprising: {core_snippet}."
        ),
        strategy_reasoning=(
            "Fallback independent claim structured as a system claim with processor + "
            "non-transitory medium preamble to satisfy USPTO hardware-tie requirements "
            "under Alice/Mayo §101 analysis. Attorney must refine claim elements to "
            "restore maximum market-protection scope."
        ),
        breadth_score=0.72,
    )


def _fallback_dependent_claims(core_text: str, start_num: int) -> List[PatentClaim]:
    fallbacks = [
        PatentClaim(
            claim_number=start_num + 1,
            claim_type=ClaimType.DEPENDENT,
            depends_on=1,
            text=(
                "The system of claim 1, wherein the instructions further cause the system "
                "to generate a structured output record comprising a timestamp, a "
                "confidence score, and an identifier associated with the processed input."
            ),
            strategy_reasoning=(
                "Adds structured output data as a dependent claim, covering the common "
                "commercial embodiment of audit-trail or record-keeping workflows. "
                "Provides a narrower PTAB fallback and licensing leverage over "
                "data-of-record management systems."
            ),
            breadth_score=0.55,
        ),
        PatentClaim(
            claim_number=start_num + 2,
            claim_type=ClaimType.DEPENDENT,
            depends_on=1,
            text=(
                "The system of claim 1, wherein the operations further comprise "
                "encrypting the output using a device-specific cryptographic key prior "
                "to storage in a local data store of the mobile computing device, such "
                "that the output is inaccessible without authorization from the device."
            ),
            strategy_reasoning=(
                "Privacy/security dependent claim targeting enterprise and regulated-market "
                "deployments. Creates licensing leverage over encrypted-at-rest "
                "implementations in healthcare, financial services, and defense contexts."
            ),
            breadth_score=0.42,
        ),
        PatentClaim(
            claim_number=start_num + 3,
            claim_type=ClaimType.DEPENDENT,
            depends_on=1,
            text="The system of claim 1, wherein the system operates in real-time.",
            strategy_reasoning="Targets real-time implementations as a narrower fallback.",
            breadth_score=0.3,
        ),
    ]
    remaining = max(0, 4 - (start_num - 1))
    return fallbacks[:remaining]
