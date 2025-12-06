"""Output templates and Pydantic models for PatentSphere agents."""

from app.templates.models import (
    ClaimsOutput,
    CPCCode,
    FeatureItem,
    SynthesisOutput,
    InsightSection,
    BulletPoint,
    NextStep,
    CitationReference,
    CriticOutput,
    RewardComponents,
    FinalResponse,
    # Enhanced API models
    SourceItem,
    AnswerSectionModel,
    APIResponse,
)
from app.templates.prompts import (
    CLAIMS_SYSTEM_PROMPT,
    CLAIMS_USER_TEMPLATE,
    SYNTHESIS_SYSTEM_PROMPT,
    SYNTHESIS_USER_TEMPLATE,
    SYNTHESIS_WITH_SOURCES_TEMPLATE,
    CRITIC_SYSTEM_PROMPT,
    CRITIC_FLUENCY_TEMPLATE,
    FINAL_OUTPUT_SYSTEM_PROMPT,
    FINAL_OUTPUT_USER_TEMPLATE,
    get_claims_prompt,
    get_synthesis_prompt,
    get_synthesis_prompt_with_sources,
    get_final_output_prompt,
    get_enhanced_final_prompt,
)
from app.templates.citations import (
    CitationSource,
    EnhancedFinalResponse,
    CitationIndexManager,
    PatentURLBuilder,
    LitigationURLBuilder,
    extract_citations_from_chunks,
    extract_citations_from_litigation,
)

__all__ = [
    # Models
    "ClaimsOutput",
    "CPCCode",
    "FeatureItem",
    "SynthesisOutput",
    "InsightSection",
    "BulletPoint",
    "NextStep",
    "CitationReference",
    "CriticOutput",
    "RewardComponents",
    "FinalResponse",
    # Enhanced API models
    "SourceItem",
    "AnswerSectionModel",
    "APIResponse",
    # Prompts
    "CLAIMS_SYSTEM_PROMPT",
    "CLAIMS_USER_TEMPLATE",
    "SYNTHESIS_SYSTEM_PROMPT",
    "SYNTHESIS_USER_TEMPLATE",
    "SYNTHESIS_WITH_SOURCES_TEMPLATE",
    "CRITIC_SYSTEM_PROMPT",
    "CRITIC_FLUENCY_TEMPLATE",
    "FINAL_OUTPUT_SYSTEM_PROMPT",
    "FINAL_OUTPUT_USER_TEMPLATE",
    "get_claims_prompt",
    "get_synthesis_prompt",
    "get_synthesis_prompt_with_sources",
    "get_final_output_prompt",
    "get_enhanced_final_prompt",
    # Citations
    "CitationSource",
    "EnhancedFinalResponse",
    "CitationIndexManager",
    "PatentURLBuilder",
    "LitigationURLBuilder",
    "extract_citations_from_chunks",
    "extract_citations_from_litigation",
]

