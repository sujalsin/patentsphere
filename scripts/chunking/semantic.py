"""Semantic chunking that splits at sentence boundaries.

This module provides sentence-aware text splitting that preserves
semantic coherence while keeping chunks within size limits.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class SemanticChunk:
    """A semantically coherent text chunk."""
    
    text: str
    start_sentence: int
    end_sentence: int
    word_count: int
    
    @property
    def sentence_count(self) -> int:
        return self.end_sentence - self.start_sentence


class SemanticChunker:
    """
    Splits text into chunks at sentence boundaries.
    
    Unlike simple word-based splitting, this chunker ensures that:
    1. Chunks end at sentence boundaries
    2. Semantic coherence is preserved
    3. Overlap is handled at sentence level
    """
    
    # Sentence boundary patterns
    SENTENCE_END_PATTERN = re.compile(
        r'(?<=[.!?])\s+(?=[A-Z0-9])|'  # Standard sentence end
        r'(?<=[.!?])\s*\n+|'           # Sentence end followed by newline
        r'(?<=\))\s+(?=[A-Z])|'        # End of parenthetical
        r'\n\n+'                        # Paragraph breaks
    )
    
    # Patterns that should NOT break sentences
    ABBREVIATIONS = {
        "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.",
        "vs.", "etc.", "i.e.", "e.g.", "fig.", "no.", "nos.",
        "u.s.", "u.s.a.", "inc.", "ltd.", "corp.", "co.",
        "jan.", "feb.", "mar.", "apr.", "jun.", "jul.",
        "aug.", "sep.", "sept.", "oct.", "nov.", "dec.",
    }
    
    def __init__(
        self,
        target_chunk_size: int = 400,
        min_chunk_size: int = 100,
        max_chunk_size: int = 800,
        overlap_sentences: int = 1,
    ):
        """
        Initialize the semantic chunker.
        
        Args:
            target_chunk_size: Target words per chunk
            min_chunk_size: Minimum words per chunk
            max_chunk_size: Maximum words per chunk
            overlap_sentences: Number of sentences to overlap
        """
        self.target_chunk_size = target_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_sentences = overlap_sentences
    
    def split_text(self, text: str) -> List[SemanticChunk]:
        """
        Split text into semantic chunks.
        
        Args:
            text: Text to split
        
        Returns:
            List of SemanticChunk objects
        """
        if not text or not text.strip():
            return []
        
        # First, split into sentences
        sentences = self._split_sentences(text)
        
        if not sentences:
            return []
        
        if len(sentences) == 1:
            word_count = len(sentences[0].split())
            return [SemanticChunk(
                text=sentences[0],
                start_sentence=0,
                end_sentence=1,
                word_count=word_count,
            )]
        
        # Build chunks from sentences
        chunks = self._build_chunks(sentences)
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences, handling edge cases."""
        # Normalize whitespace
        text = " ".join(text.split())
        
        # Protect abbreviations by temporarily replacing periods
        protected_text = text
        for abbr in self.ABBREVIATIONS:
            # Case-insensitive replacement
            pattern = re.compile(re.escape(abbr), re.IGNORECASE)
            protected_text = pattern.sub(abbr.replace(".", "<DOT>"), protected_text)
        
        # Protect decimal numbers (e.g., 3.14)
        protected_text = re.sub(r'(\d)\.(\d)', r'\1<DOT>\2', protected_text)
        
        # Protect patent numbers (e.g., US7,123,456)
        protected_text = re.sub(r'(US|WO|EP|JP|CN)(\d)', r'\1<PAT>\2', protected_text)
        
        # Split on sentence boundaries
        sentences = self.SENTENCE_END_PATTERN.split(protected_text)
        
        # Restore protected characters
        sentences = [
            s.replace("<DOT>", ".").replace("<PAT>", "").strip()
            for s in sentences
            if s and s.strip()
        ]
        
        # Filter out very short "sentences" (likely noise)
        sentences = [s for s in sentences if len(s.split()) >= 3]
        
        return sentences
    
    def _build_chunks(self, sentences: List[str]) -> List[SemanticChunk]:
        """Build chunks from sentences with overlap."""
        chunks: List[SemanticChunk] = []
        
        i = 0
        while i < len(sentences):
            # Find the best chunk boundary
            chunk_sentences, end_idx = self._find_chunk_boundary(sentences, i)
            
            if chunk_sentences:
                chunk_text = " ".join(chunk_sentences)
                chunks.append(SemanticChunk(
                    text=chunk_text,
                    start_sentence=i,
                    end_sentence=end_idx,
                    word_count=len(chunk_text.split()),
                ))
            
            # Move to next chunk with overlap
            if end_idx <= i:
                # Prevent infinite loop
                i += 1
            else:
                # Apply overlap
                overlap_start = max(0, end_idx - self.overlap_sentences)
                i = max(i + 1, overlap_start)
        
        return chunks
    
    def _find_chunk_boundary(
        self, sentences: List[str], start_idx: int
    ) -> Tuple[List[str], int]:
        """
        Find the best boundary for a chunk starting at start_idx.
        
        Returns:
            Tuple of (sentences_in_chunk, end_index)
        """
        chunk_sentences: List[str] = []
        word_count = 0
        end_idx = start_idx
        
        for i in range(start_idx, len(sentences)):
            sentence = sentences[i]
            sentence_words = len(sentence.split())
            
            # Check if adding this sentence would exceed max
            if word_count + sentence_words > self.max_chunk_size and chunk_sentences:
                break
            
            chunk_sentences.append(sentence)
            word_count += sentence_words
            end_idx = i + 1
            
            # Check if we've reached target size
            if word_count >= self.target_chunk_size:
                # Look ahead for a better break point
                if i + 1 < len(sentences):
                    next_words = len(sentences[i + 1].split())
                    # If next sentence is small and won't exceed max, include it
                    if word_count + next_words <= self.max_chunk_size and next_words < 50:
                        continue
                break
        
        return chunk_sentences, end_idx
    
    def split_with_context(
        self, text: str, context_sentences: int = 1
    ) -> List[Tuple[str, str, str]]:
        """
        Split text and return chunks with leading/trailing context.
        
        Args:
            text: Text to split
            context_sentences: Number of context sentences
        
        Returns:
            List of tuples (leading_context, chunk, trailing_context)
        """
        sentences = self._split_sentences(text)
        chunks = self._build_chunks(sentences)
        
        results = []
        for chunk in chunks:
            # Get leading context
            lead_start = max(0, chunk.start_sentence - context_sentences)
            leading = " ".join(sentences[lead_start:chunk.start_sentence])
            
            # Get trailing context
            trail_end = min(len(sentences), chunk.end_sentence + context_sentences)
            trailing = " ".join(sentences[chunk.end_sentence:trail_end])
            
            results.append((leading, chunk.text, trailing))
        
        return results


def estimate_sentence_count(text: str) -> int:
    """Estimate the number of sentences in text."""
    # Quick estimate based on sentence-ending punctuation
    return len(re.findall(r'[.!?]+', text))

