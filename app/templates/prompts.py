"""Prompt templates with embedded JSON schemas for PatentSphere agents.

These templates include the JSON schema from Pydantic models directly in the prompts
to ensure the LLM understands the exact output format required.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any, Dict

from app.templates.models import ClaimsOutput, SynthesisOutput


def json_serializer(obj):
    """Custom JSON serializer for date/datetime objects."""
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


# =============================================================================
# Schema Generation Helpers
# =============================================================================

def _get_json_schema(model_class) -> str:
    """Extract JSON schema from a Pydantic model and format it for prompts."""
    schema = model_class.model_json_schema()
    # Remove unnecessary fields for cleaner prompt
    schema.pop("$defs", None)
    schema.pop("title", None)
    return json.dumps(schema, indent=2)


def _get_simple_schema(model_class) -> str:
    """Get a simplified schema representation for prompts."""
    schema = model_class.model_json_schema()
    
    def simplify_property(prop: Dict[str, Any]) -> str:
        prop_type = prop.get("type", "any")
        if prop_type == "array":
            items = prop.get("items", {})
            item_type = items.get("type", "object")
            return f"[{item_type}]"
        return prop_type
    
    properties = schema.get("properties", {})
    simplified = {}
    for key, value in properties.items():
        desc = value.get("description", "")
        simplified[key] = f"{simplify_property(value)} - {desc}" if desc else simplify_property(value)
    
    return json.dumps(simplified, indent=2)


# =============================================================================
# Claims Analyzer Prompts (optimized for qwen2.5:1.5b-instruct)
# =============================================================================

CLAIMS_SYSTEM_PROMPT = """You are a patent classification expert. You MUST output valid JSON matching the exact schema.
CRITICAL: Return ONLY the JSON object. No markdown, no explanations, no code blocks."""


CLAIMS_USER_TEMPLATE = """### Task
Analyze this patent query and return EXACT JSON matching the schema below.

### Query
{query}

### Required JSON Schema (MUST MATCH EXACTLY)
{{
  "summary": "string (required)",
  "query_type": "research" OR "litigation" OR "portfolio" OR "emergence" OR "other" (required, pick ONE),
  "features": [
    {{"name": "string", "insight": "string", "evidence": "string"}}
  ],
  "cpc_codes": [
    {{"code": "string like G06N3/08", "title": "string", "confidence": 0.0-1.0, "justification": "string"}}
  ],
  "assumptions": ["string"],
  "confidence": 0.0-1.0,
  "risk_entities": {{
    "assignees": ["string - company names mentioned"],
    "topics": ["string - technical topics/keywords"],
    "search_query": "string - optimized query for retrieval"
  }},
  "metadata_filters": {{
    "date_range": [start_year, end_year] OR null,
    "cpc_codes": ["string - CPC codes for filtering"]
  }}
}}

### Example for "Samsung solid-state battery electrolyte dendrite suppression"
{{
  "summary": "Query seeks patents on solid-state battery electrolyte dendrite suppression, with focus on Samsung",
  "query_type": "research",
  "features": [
    {{"name": "solid-state battery", "insight": "Core technology focus", "evidence": "solid-state battery"}},
    {{"name": "dendrite suppression", "insight": "Specific technical problem", "evidence": "dendrite suppression"}}
  ],
  "cpc_codes": [
    {{"code": "H01M10/0562", "title": "Solid electrolytes", "confidence": 0.9, "justification": "Direct match for solid-state batteries"}}
  ],
  "assumptions": ["Focus on recent developments (2015-2024)"],
  "confidence": 0.85,
  "risk_entities": {{
    "assignees": ["Samsung"],
    "topics": ["Electrolyte", "Dendrite", "Solid-state battery"],
    "search_query": "Solid-state battery electrolyte dendrite suppression"
  }},
  "metadata_filters": {{
    "date_range": [2015, 2024],
    "cpc_codes": ["H01M10/0562", "H01M"]
  }}
}}

### Rules
1. query_type MUST be one of: research, litigation, portfolio, emergence, other
2. Extract assignees: company names, organizations mentioned (e.g., "Samsung", "Apple", "IBM")
3. Extract topics: technical keywords for litigation search (e.g., "Electrolyte", "Dendrite", "OLED")
4. search_query: Optimize the query for retrieval - use technical terms, remove filler words
5. date_range: Infer from query context (e.g., "recent patents" = [2020, 2024], "historical" = [2000, 2010])
6. Return ONLY valid JSON, no markdown, no code blocks
7. Use double quotes for all strings

### Your Response (JSON only):"""


def get_claims_prompt(query: str) -> str:
    """Generate the full claims analyzer prompt for a query."""
    return CLAIMS_USER_TEMPLATE.format(query=query)


# =============================================================================
# Synthesis Agent Prompts
# =============================================================================

SYNTHESIS_SYSTEM_PROMPT = """You synthesize patent data into executive briefings. Output JSON only."""


SYNTHESIS_USER_TEMPLATE = """### Task
Create an executive patent briefing from the data below.

### Query
{query}

### Intent
{intent_explanation}

### Claims Data
{claims_data}

### Patent Data
{citation_data}

### Litigation Data
{litigation_data}

### Output Format
{{
  "executive_summary": "2-3 sentences with patent IDs (e.g., US-12345). Answer the query directly.",
  "insight_sections": [
    {{
      "title": "Section Title",
      "bullets": [
        {{"headline": "Key point", "details": ["detail 1"], "citations": ["US-12345"]}}
      ]
    }}
  ],
  "next_steps": [
    {{"priority": "high", "recommendation": "action", "rationale": "why"}}
  ],
  "citations": [{{"patent_id": "US-12345", "reason": "why cited"}}],
  "risk_score": 50
}}

### Rules
- 2-3 insight sections, 1-3 bullets each
- Include specific patent IDs from the data
- risk_score: 0-30 low, 30-60 medium, 60-100 high

### Response (JSON only)"""


def get_synthesis_prompt(
    query: str,
    intent_explanation: str,
    claims_data: Dict[str, Any],
    citation_data: Dict[str, Any],
    litigation_data: Dict[str, Any],
) -> str:
    """Generate the full synthesis prompt with all agent data."""
    return SYNTHESIS_USER_TEMPLATE.format(
        query=query,
        intent_explanation=intent_explanation,
        claims_data=json.dumps(claims_data, indent=2, ensure_ascii=False, default=json_serializer),
        citation_data=json.dumps(citation_data, indent=2, ensure_ascii=False, default=json_serializer),
        litigation_data=json.dumps(litigation_data, indent=2, ensure_ascii=False, default=json_serializer),
    )


# =============================================================================
# Synthesis with Source Index Template (for [1], [2] citations)
# =============================================================================

SYNTHESIS_WITH_SOURCES_TEMPLATE = """Synthesize the following agent outputs into an executive briefing with indexed citations.

## User Query
"{query}"

## Detected Intent
{intent_explanation}

## Source Index (USE THESE INDICES FOR INLINE CITATIONS)
{source_index}

## Claims Analysis Data
```json
{claims_data}
```

## Citation/Retrieval Data
```json
{citation_data}
```

## Litigation Scout Data
```json
{litigation_data}
```

## Required Output Schema
Return ONLY valid JSON. Use [1], [2], [3] etc. to reference sources inline.

```json
{{
  "executive_summary": "string - 2-4 sentences with inline citations like [1], [2]",
  "technical_summary": "string - Technical analysis. Reference sources: 'US1234567A [1] discloses...'",
  "legal_summary": "string or null - Litigation analysis with source references like [3]",
  "insight_sections": [
    {{
      "title": "string - Section title",
      "bullets": [
        {{
          "headline": "string",
          "details": ["string with [N] citations"],
          "citations": ["patent_id"]
        }}
      ]
    }}
  ],
  "next_steps": [
    {{
      "priority": "high|medium|low",
      "recommendation": "string",
      "rationale": "string"
    }}
  ],
  "citations": [
    {{
      "patent_id": "string",
      "reason": "string"
    }}
  ],
  "risk_score": 0-100
}}
```

## Guidelines
- Use [N] inline citations to reference the source index provided above
- Include technical_summary for prior art analysis
- Include legal_summary only if litigation data is available
- risk_score: Higher (70-100) for crowded spaces with litigation history, Lower (0-30) for novel areas

Return ONLY the JSON object, no additional text."""


def get_synthesis_prompt_with_sources(
    query: str,
    intent_explanation: str,
    claims_data: Dict[str, Any],
    citation_data: Dict[str, Any],
    litigation_data: Dict[str, Any],
    source_index: str,
) -> str:
    """Generate synthesis prompt with pre-indexed sources for [1], [2] style citations."""
    return SYNTHESIS_WITH_SOURCES_TEMPLATE.format(
        query=query,
        intent_explanation=intent_explanation,
        source_index=source_index,
        claims_data=json.dumps(claims_data, indent=2, ensure_ascii=False, default=json_serializer),
        citation_data=json.dumps(citation_data, indent=2, ensure_ascii=False, default=json_serializer),
        litigation_data=json.dumps(litigation_data, indent=2, ensure_ascii=False, default=json_serializer),
    )


# =============================================================================
# Critic Agent Prompts
# =============================================================================

CRITIC_SYSTEM_PROMPT = """You evaluate patent analysis quality. Output only a number 0.0-1.0."""


CRITIC_FLUENCY_TEMPLATE = """### Task
Rate this patent analysis quality from 0.0 to 1.0.

### Response
{response}

### Scale
- 0.0-0.3: Poor/incoherent
- 0.4-0.6: Average
- 0.7-0.9: Good/comprehensive
- 0.9-1.0: Excellent

### Output
Return ONLY a number like 0.75"""


# =============================================================================
# Final Output Template (for final synthesis layer)
# =============================================================================

FINAL_OUTPUT_SYSTEM_PROMPT = """You are a patent expert. Create polished analysis responses. Output JSON only."""


FINAL_OUTPUT_USER_TEMPLATE = """### Task
Create a polished patent analysis response.

### Query
{query}

### Sources (use [N] citations)
{source_index}

### Data
{synthesis_output}

### Litigation
{litigation_data}

### Output Format
{{
  "technical_summary": "2-3 paragraphs answering the query. Include patent IDs like [1] US-12345. Cover: (1) landscape overview, (2) key patents, (3) major players.",
  "legal_summary": "Litigation risks if data exists, else null",
  "novelty_assessment": "Innovation gaps and opportunities",
  "risk_score": 50
}}

### Rules
- Answer the user's question directly
- Cite patents using [N] format
- risk_score: 0-30 low, 30-60 moderate, 60-100 high

### Response (JSON only)"""


def get_final_output_prompt(
    query: str,
    synthesis_output: Dict[str, Any],
    quality_score: float,
    retrieved_patents: list,
) -> str:
    """Generate the final output formatting prompt (legacy)."""
    return FINAL_OUTPUT_USER_TEMPLATE.format(
        query=query,
        source_index="",
        synthesis_output=json.dumps(synthesis_output, indent=2, ensure_ascii=False),
        litigation_data="{}",
    )


def get_enhanced_final_prompt(
    query: str,
    source_index: str,
    synthesis_output: Dict[str, Any],
    litigation_data: Dict[str, Any],
) -> str:
    """Generate the enhanced final output prompt with source indexing."""
    return FINAL_OUTPUT_USER_TEMPLATE.format(
        query=query,
        source_index=source_index,
        synthesis_output=json.dumps(synthesis_output, indent=2, ensure_ascii=False),
        litigation_data=json.dumps(litigation_data or {}, indent=2, ensure_ascii=False, default=json_serializer),
    )


# =============================================================================
# JSON Repair Prompt
# =============================================================================

JSON_REPAIR_SYSTEM_PROMPT = """You are a JSON repair assistant. Your only job is to fix malformed JSON and return valid JSON.

Rules:
1. Fix syntax errors (missing quotes, brackets, commas)
2. Preserve all original information
3. Return ONLY valid JSON
4. Do not add commentary or explanation"""


JSON_REPAIR_TEMPLATE = """The following JSON is malformed. Fix it and return valid JSON matching this schema:

## Expected Schema
```json
{schema}
```

## Malformed JSON
```
{malformed_json}
```

Return ONLY the corrected JSON object, no additional text."""


def get_json_repair_prompt(malformed_json: str, schema: Dict[str, Any]) -> str:
    """Generate a JSON repair prompt."""
    return JSON_REPAIR_TEMPLATE.format(
        schema=json.dumps(schema, indent=2),
        malformed_json=malformed_json,
    )

