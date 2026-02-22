"""
Smoke tests for the Disclosure-to-Draft Pipeline models and decomposer helpers.

These tests are fast (no LLM calls, no DB connections) and verify:
  1. All Pydantic models instantiate correctly
  2. Field validation (confidence bounds, word counts, etc.)
  3. JSON round-trip serialisation
  4. HumanWallChecklist approval workflow
  5. PipelineState audit log append behaviour
  6. Decomposer helper functions (_parse_json, _find_span, _fuzzy_section_name)

Run with: pytest tests/test_pipeline_models.py -v
"""

import hashlib
import json
import pytest
from datetime import datetime, timezone

# Pipeline models
from pipeline.models import (
    AuditLogEntry,
    ClaimSet,
    ClaimType,
    CriticFlag,
    DecomposedDisclosure,
    DisclosureSection,
    DraftSpecification,
    FlagType,
    HumanWallChecklist,
    PatentClaim,
    PipelineState,
    PriorArtReference,
    ProvenanceSpan,
    SectionAnalysis,
    SectionName,
    SectionScore,
    Severity,
)

# Decomposer helpers
from pipeline.decomposer import _find_span, _fuzzy_section_name, _parse_json


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_provenance():
    return ProvenanceSpan(char_start=0, char_end=50, excerpt="This is the sample excerpt text from the inventor")


@pytest.fixture
def sample_section(sample_provenance):
    return DisclosureSection(
        section_name=SectionName.CORE_INVENTION,
        text="A neural network system for predicting patent novelty.",
        confidence=0.85,
        provenance=sample_provenance,
    )


@pytest.fixture
def sample_disclosure(sample_section):
    raw = "This is the sample raw inventor disclosure text."
    return DecomposedDisclosure(
        raw_input_hash=hashlib.sha256(raw.encode()).hexdigest(),
        raw_input_length=len(raw),
        sections=[sample_section],
    )


@pytest.fixture
def sample_claim():
    return PatentClaim(
        claim_number=1,
        claim_type=ClaimType.INDEPENDENT,
        depends_on=None,
        text="A system comprising a neural network configured to predict patent novelty.",
        strategy_reasoning="Broad independent claim covering all neural network architectures.",
        breadth_score=0.9,
    )


@pytest.fixture
def sample_claim_set(sample_claim):
    dependent = PatentClaim(
        claim_number=2,
        claim_type=ClaimType.DEPENDENT,
        depends_on=1,
        text="The system of claim 1, wherein the neural network is a transformer.",
        strategy_reasoning="Narrows to transformer architecture as first fallback.",
        breadth_score=0.5,
    )
    return ClaimSet(claims=[sample_claim, dependent])


@pytest.fixture
def sample_spec(sample_claim):
    return DraftSpecification(
        title="System for Predicting Patent Novelty",
        abstract="A neural network based system for predicting patent novelty scores.",
        background="Prior systems relied on keyword matching with limited accuracy.",
        summary_of_invention="The present invention provides a transformer-based system.",
        detailed_description=(
            "Referring to FIG. 1, the system comprises a transformer neural network. "
            "The network is trained on a corpus of patent documents."
        ),
        claims=[sample_claim],
        provenance_map={"title": ["core_invention"], "abstract": ["core_invention", "advantages"]},
    )


@pytest.fixture
def sample_checklist():
    flag = CriticFlag(
        section="abstract",
        flag_type=FlagType.ATTORNEY_REVIEW,
        severity=Severity.MEDIUM,
        issue="Abstract may exceed 150 words after attorney revision.",
        suggestion="Trim to 150 words.",
    )
    score = SectionScore(section="abstract", confidence=0.75, word_count=42)
    return HumanWallChecklist(
        overall_confidence=0.78,
        section_scores=[score],
        flags=[flag],
        reviewer_notes="Draft is generally well-structured.",
        export_allowed=False,
    )


# ---------------------------------------------------------------------------
# 1.  Model instantiation
# ---------------------------------------------------------------------------

class TestModelInstantiation:

    def test_provenance_span(self):
        span = ProvenanceSpan(char_start=10, char_end=50, excerpt="test excerpt")
        assert span.char_start == 10
        assert span.char_end == 50

    def test_disclosure_section_word_count(self, sample_section):
        # word_count is computed automatically by validator
        assert sample_section.word_count > 0

    def test_decoded_disclosure_section_map(self, sample_disclosure):
        sec = sample_disclosure.get_section(SectionName.CORE_INVENTION)
        assert sec is not None
        assert sec.section_name == SectionName.CORE_INVENTION

    def test_decoded_disclosure_missing_section(self, sample_disclosure):
        sec = sample_disclosure.get_section(SectionName.BACKGROUND)
        assert sec is None

    def test_claim_set_counters(self, sample_claim_set):
        assert sample_claim_set.independent_claim_count == 1
        assert sample_claim_set.dependent_claim_count == 1

    def test_draft_specification_word_count(self, sample_spec):
        wc = sample_spec.word_count()
        assert wc > 10

    def test_critic_flag_frozen(self, sample_checklist):
        flag = sample_checklist.flags[0]
        with pytest.raises(Exception):
            flag.section = "claims"  # Should be immutable (frozen=True)

    def test_checklist_severity_counts(self, sample_checklist):
        assert sample_checklist.medium_severity_count == 1
        assert sample_checklist.high_severity_count == 0


# ---------------------------------------------------------------------------
# 2.  Field validation
# ---------------------------------------------------------------------------

class TestFieldValidation:

    def test_confidence_out_of_range_raises(self, sample_provenance):
        with pytest.raises(Exception):
            DisclosureSection(
                section_name=SectionName.BACKGROUND,
                text="test",
                confidence=1.5,  # > 1.0, should fail
                provenance=sample_provenance,
            )

    def test_novelty_score_bounds(self, sample_section):
        with pytest.raises(Exception):
            SectionAnalysis(
                section_name=SectionName.BACKGROUND,
                novelty_score=-0.1,  # < 0.0
                obviousness_risk=0.5,
                enablement_score=0.5,
                litigation_risk=0.5,
            )

    def test_abstract_max_length_ok(self, sample_spec):
        # 150 word abstract should not raise
        words = ["word"] * 149
        spec = sample_spec.model_copy(update={"abstract": " ".join(words)})
        assert spec.abstract  # Just verify it's set


# ---------------------------------------------------------------------------
# 3.  JSON round-trip serialisation
# ---------------------------------------------------------------------------

class TestJsonSerialisation:

    def test_pipeline_state_roundtrip(self, sample_disclosure, sample_checklist):
        state = PipelineState(
            raw_input="inventor raw text",
            decomposition=sample_disclosure,
            checklist=sample_checklist,
        )
        json_str = state.model_dump_json()
        restored = PipelineState.model_validate_json(json_str)
        assert restored.session_id == state.session_id
        assert restored.decomposition is not None

    def test_claim_set_roundtrip(self, sample_claim_set):
        json_str = sample_claim_set.model_dump_json()
        restored = ClaimSet.model_validate_json(json_str)
        assert len(restored.claims) == 2
        assert restored.independent_claim_count == 1


# ---------------------------------------------------------------------------
# 4.  HumanWall approval workflow
# ---------------------------------------------------------------------------

class TestHumanWallApproval:

    def test_export_blocked_by_default(self, sample_checklist):
        assert sample_checklist.export_allowed is False
        assert sample_checklist.user_approved_at is None

    def test_approve_unlocks_export(self, sample_checklist):
        approved = sample_checklist.approve()
        assert approved.export_allowed is True
        assert approved.user_approved_at is not None
        assert isinstance(approved.user_approved_at, datetime)

    def test_original_not_mutated(self, sample_checklist):
        approved = sample_checklist.approve()
        assert sample_checklist.export_allowed is False  # Original unchanged


# ---------------------------------------------------------------------------
# 5.  PipelineState audit log
# ---------------------------------------------------------------------------

class TestAuditLog:

    def test_log_appends_immutably(self):
        state = PipelineState(raw_input="test text")
        assert len(state.audit_log) == 0

        state2 = state.log("decompose", "started", "beginning")
        assert len(state.audit_log) == 0   # Original untouched
        assert len(state2.audit_log) == 1

        state3 = state2.log("decompose", "completed")
        assert len(state3.audit_log) == 2

    def test_log_entry_fields(self):
        state = PipelineState(raw_input="test")
        state2 = state.log("analyze", "failed", "timeout")
        entry = state2.audit_log[0]
        assert entry.step == "analyze"
        assert entry.status == "failed"
        assert entry.detail == "timeout"

    def test_is_complete_false_when_partial(self, sample_disclosure):
        state = PipelineState(raw_input="test", decomposition=sample_disclosure)
        assert not state.is_complete()

    def test_export_ready_false_without_approval(
        self, sample_disclosure, sample_spec, sample_claim_set, sample_checklist
    ):
        from pipeline.models import SectionAnalysis, SectionName
        state = PipelineState(
            raw_input="test",
            decomposition=sample_disclosure,
            analyses=[SectionAnalysis(
                section_name=SectionName.CORE_INVENTION,
                novelty_score=0.7, obviousness_risk=0.3,
                enablement_score=0.8, litigation_risk=0.2,
            )],
            claim_set=sample_claim_set,
            specification=sample_spec,
            checklist=sample_checklist,  # export_allowed=False
        )
        assert state.is_complete()
        assert not state.export_ready()


# ---------------------------------------------------------------------------
# 6.  Decomposer helper functions
# ---------------------------------------------------------------------------

class TestDecomposerHelpers:

    def test_parse_json_valid(self):
        result = _parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_json_with_markdown_fence(self):
        result = _parse_json('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_parse_json_invalid_returns_none(self):
        result = _parse_json("This is not JSON at all.")
        assert result is None

    def test_find_span_exact_match(self):
        raw = "The invention relates to a neural network for patents."
        excerpt = "neural network"
        span = _find_span(raw, excerpt)
        assert span.char_start == raw.index("neural network")
        assert span.excerpt == "neural network"

    def test_find_span_case_insensitive(self):
        raw = "The invention relates to a Neural Network for patents."
        excerpt = "neural network"
        span = _find_span(raw, excerpt)
        # Should find case-insensitively
        assert span.char_start >= 0

    def test_find_span_not_found_returns_stub(self):
        raw = "Hello world"
        excerpt = "zzz not present zzz"
        span = _find_span(raw, excerpt)
        # Should return a valid stub, not raise
        assert isinstance(span, ProvenanceSpan)

    def test_fuzzy_section_name_invention(self):
        result = _fuzzy_section_name("summary of the invention")
        assert result == SectionName.CORE_INVENTION

    def test_fuzzy_section_name_prior_art(self):
        result = _fuzzy_section_name("prior art and field")
        assert result == SectionName.BACKGROUND

    def test_fuzzy_section_name_unknown_returns_none(self):
        result = _fuzzy_section_name("xyzzy impossible section")
        assert result is None


# ---------------------------------------------------------------------------
# 7.  Integration: pipeline models chain
# ---------------------------------------------------------------------------

class TestPipelineChain:

    def test_full_pipeline_state_structure(
        self, sample_disclosure, sample_spec, sample_claim_set, sample_checklist
    ):
        """Build a representative pipeline state and verify structure."""
        from pipeline.models import SectionAnalysis, SectionName

        analyses = [
            SectionAnalysis(
                section_name=SectionName.CORE_INVENTION,
                novelty_score=0.8,
                obviousness_risk=0.2,
                enablement_score=0.9,
                litigation_risk=0.1,
                prior_art_hits=[
                    PriorArtReference(
                        patent_id="US_12345678_A1",
                        title="Prior neural network patent",
                        relevance_score=0.75,
                    )
                ],
                analysis_summary="High novelty, low litigation risk.",
            )
        ]

        state = PipelineState(
            raw_input="Test inventor disclosure",
            tenant_id="demo_tenant_001",
            decomposition=sample_disclosure,
            analyses=analyses,
            claim_set=sample_claim_set,
            specification=sample_spec,
            checklist=sample_checklist,
        )

        # Verify complete
        assert state.is_complete()
        # Not export ready (not approved yet)
        assert not state.export_ready()

        # Approve and check
        approved_checklist = sample_checklist.approve()
        state2 = state.model_copy(update={"checklist": approved_checklist})
        assert state2.export_ready()

        # Tenant ID preserved
        assert state2.tenant_id == "demo_tenant_001"
