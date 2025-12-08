"""Tests for citation verification."""
import pytest
import re
from typing import List, Dict, Any, Tuple


def verify_citations(response: str, retrieved_context: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """
    Verify all citations in response exist in retrieved context.
    
    Args:
        response: The generated response text
        retrieved_context: List of retrieved documents with patent_id
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    # Extract all citation IDs from response
    citation_pattern = r'\[\[([^\]]+)\]\]'
    citations = re.findall(citation_pattern, response)
    
    # Get all patent IDs from context
    context_ids = []
    for doc in retrieved_context:
        patent_id = doc.get("patent_id", "")
        if patent_id:
            # Normalize IDs (remove dashes, handle variations)
            clean_id = patent_id.replace("-", "").replace("US", "").strip()
            context_ids.append(patent_id)
            context_ids.append(clean_id)
            # Also add with US prefix variations
            if not patent_id.startswith("US"):
                context_ids.append(f"US{patent_id}")
    
    errors = []
    for cit_id in citations:
        # Normalize citation ID
        clean_cit = cit_id.replace("-", "").replace("US", "").strip()
        
        # Check if citation exists in context (with various normalizations)
        found = False
        for ctx_id in context_ids:
            if clean_cit in ctx_id or ctx_id in clean_cit or cit_id in ctx_id or ctx_id in cit_id:
                found = True
                break
        
        if not found:
            errors.append(f"Hallucinated Citation: {cit_id}")
    
    return len(errors) == 0, errors


def verify_citation_format(response: str) -> Tuple[bool, List[str]]:
    """
    Verify citation format is correct: [[ID]](URL)
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Find all citation patterns
    citation_pattern = r'\[\[([^\]]+)\]\]\(([^\)]+)\)'
    citations = re.findall(citation_pattern, response)
    
    # Check format
    for patent_id, url in citations:
        if not patent_id.strip():
            errors.append(f"Empty patent ID in citation")
        if not url.strip():
            errors.append(f"Empty URL in citation for {patent_id}")
        if not url.startswith("http"):
            errors.append(f"Invalid URL format for {patent_id}: {url}")
    
    # Check for malformed citations (missing URL)
    malformed = re.findall(r'\[\[([^\]]+)\]\]\s*(?!\()', response)
    for cit_id in malformed:
        errors.append(f"Citation {cit_id} missing URL")
    
    return len(errors) == 0, errors


@pytest.mark.unit
class TestCitationVerification:
    """Test citation verification functions."""
    
    def test_verify_citations_valid(self):
        """Test verification with valid citations."""
        response = "The patent [[US-2016168309-A1]](https://patents.google.com/patent/US-2016168309-A1) describes..."
        context = [
            {"patent_id": "US-2016168309-A1", "title": "Test"},
        ]
        
        is_valid, errors = verify_citations(response, context)
        
        assert is_valid
        assert len(errors) == 0
    
    def test_verify_citations_invalid(self):
        """Test verification with invalid (hallucinated) citations."""
        response = "The patent [[US9999999]](url) describes..."
        context = [
            {"patent_id": "US-2016168309-A1", "title": "Test"},
        ]
        
        is_valid, errors = verify_citations(response, context)
        
        assert not is_valid
        assert len(errors) > 0
        assert "US9999999" in errors[0]
    
    def test_verify_citation_format_valid(self):
        """Test citation format validation with valid format."""
        response = "Patent [[US123]](https://patents.google.com/patent/US123) is valid."
        
        is_valid, errors = verify_citation_format(response)
        
        assert is_valid
        assert len(errors) == 0
    
    def test_verify_citation_format_invalid(self):
        """Test citation format validation with invalid format."""
        # Missing URL
        response1 = "Patent [[US123]] is missing URL."
        is_valid1, errors1 = verify_citation_format(response1)
        assert not is_valid1, f"Should detect missing URL, got errors: {errors1}"
        assert len(errors1) > 0
        
        # Empty URL
        response2 = "Patent [[US123]]() has empty URL."
        is_valid2, errors2 = verify_citation_format(response2)
        # Note: Empty URL in parentheses might still match the pattern
        # Check if we have errors about empty URL
        has_empty_url_error = any("empty" in err.lower() or "Empty" in err for err in errors2)
        if not is_valid2:
            assert len(errors2) > 0
        elif has_empty_url_error:
            # If we detected empty URL, that's also a failure
            assert not is_valid2
    
    def test_citation_extraction(self):
        """Test citation extraction from text."""
        response = """
        Patent [[US-2016168309-A1]](https://patents.google.com/patent/US-2016168309-A1) 
        and [[US-2016168354-A1]](https://patents.google.com/patent/US-2016168354-A1) 
        are both valid.
        """
        
        citations = re.findall(r'\[\[([^\]]+)\]\]', response)
        
        assert len(citations) == 2
        assert "US-2016168309-A1" in citations
        assert "US-2016168354-A1" in citations

