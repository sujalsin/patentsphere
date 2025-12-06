"""Hybrid chunking combining hierarchical and semantic strategies.

This module provides the main chunking interface that:
1. Preserves patent document structure (hierarchical)
2. Splits large sections semantically (sentence-aware)
3. Maintains parent-child relationships for retrieval expansion
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from scripts.chunking.hierarchical import (
    HierarchicalChunker,
    PatentSection,
    SectionType,
    SECTION_WEIGHTS,
)
from scripts.chunking.semantic import SemanticChunker


@dataclass
class HybridChunk:
    """
    A chunk with hierarchical context and semantic boundaries.
    
    Includes parent-child relationship information for retrieval expansion.
    """
    
    chunk_id: str
    patent_id: str
    text: str
    
    # Hierarchical information
    chunk_type: str  # Section type (title, abstract, claim_1, etc.)
    chunk_level: int  # 0 = parent, 1 = child
    parent_chunk_id: Optional[str] = None
    section_type: str = "other"
    
    # Position and ordering
    order: int = 0
    chunk_index: int = 0  # Index within parent section
    total_chunks_in_section: int = 1
    
    # Metadata
    word_count: int = 0
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Context for retrieval expansion
    leading_context: str = ""
    trailing_context: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "chunk_id": self.chunk_id,
            "patent_id": self.patent_id,
            "text": self.text,
            "chunk_type": self.chunk_type,
            "chunk_level": self.chunk_level,
            "parent_chunk_id": self.parent_chunk_id,
            "section_type": self.section_type,
            "order": self.order,
            "chunk_index": self.chunk_index,
            "total_chunks_in_section": self.total_chunks_in_section,
            "word_count": self.word_count,
            "weight": self.weight,
            "metadata": self.metadata,
            "leading_context": self.leading_context,
            "trailing_context": self.trailing_context,
        }


class HybridChunker:
    """
    Hybrid chunking strategy combining hierarchical and semantic approaches.
    
    Strategy:
    1. Extract hierarchical sections from patent (preserves structure)
    2. For large sections, apply semantic chunking (sentence-aware)
    3. Maintain parent-child relationships for context expansion
    4. Apply section-specific weights for retrieval scoring
    
    Optimized for patent RAG accuracy with smaller, precise chunks.
    """
    
    # Default section weights optimized for patent retrieval
    DEFAULT_SECTION_WEIGHTS = {
        "title": 1.5,
        "abstract": 1.4,
        "claims": 1.3,
        "claim": 1.3,
        "summary": 1.2,
        "description": 1.0,
        "detailed_description": 1.0,
        "background": 0.9,
        "drawings": 0.7,
        "other": 0.8,
    }
    
    def __init__(
        self,
        # Hierarchical settings
        min_section_words: int = 10,
        max_section_words: int = 5000,
        
        # Semantic settings - OPTIMIZED FOR RETRIEVAL ACCURACY
        target_chunk_size: int = 350,  # Smaller chunks for precision
        min_chunk_size: int = 100,
        max_chunk_size: int = 500,     # Avoid overly large chunks
        overlap_sentences: int = 2,     # More overlap for context
        
        # Hybrid settings
        split_threshold: int = 400,    # Split earlier for smaller chunks
        include_context: bool = True,
        context_sentences: int = 1,
        max_chunks_per_patent: Optional[int] = 80,
        
        # Section-specific weights
        section_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the hybrid chunker.
        
        Args:
            min_section_words: Minimum words for hierarchical section
            max_section_words: Maximum words before forcing split
            target_chunk_size: Target words per semantic chunk
            min_chunk_size: Minimum words per semantic chunk
            max_chunk_size: Maximum words per semantic chunk
            overlap_sentences: Sentence overlap for semantic chunks
            split_threshold: Word count threshold to trigger semantic splitting
            include_context: Whether to include context in chunks
            context_sentences: Number of context sentences
            max_chunks_per_patent: Maximum total chunks per patent
        """
        self.hierarchical = HierarchicalChunker(
            min_section_words=min_section_words,
            max_section_words=max_section_words,
            include_metadata=True,
        )
        
        self.semantic = SemanticChunker(
            target_chunk_size=target_chunk_size,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
            overlap_sentences=overlap_sentences,
        )
        
        self.split_threshold = split_threshold
        self.include_context = include_context
        self.context_sentences = context_sentences
        self.max_chunks_per_patent = max_chunks_per_patent
        self.section_weights = section_weights or self.DEFAULT_SECTION_WEIGHTS
    
    def get_section_weight(self, section_type: str) -> float:
        """Get the retrieval weight for a section type."""
        return self.section_weights.get(section_type.lower(), 1.0)
    
    def chunk_patent(self, patent_record: Dict[str, Any]) -> List[HybridChunk]:
        """
        Chunk a patent record using hybrid strategy.
        
        Args:
            patent_record: Raw patent data
        
        Returns:
            List of HybridChunk objects
        """
        patent_id = patent_record.get("publication_number") or patent_record.get("id", "")
        
        if not patent_id:
            return []
        
        # Step 1: Extract hierarchical sections
        sections = self.hierarchical.extract_sections(patent_record)
        
        if not sections:
            return []
        
        # Step 2: Process each section
        chunks: List[HybridChunk] = []
        global_order = 0
        
        for section in sections:
            section_chunks = self._process_section(
                section=section,
                patent_id=patent_id,
                start_order=global_order,
            )
            chunks.extend(section_chunks)
            global_order += len(section_chunks)
        
        # Step 3: Apply max chunks limit if set
        if self.max_chunks_per_patent and len(chunks) > self.max_chunks_per_patent:
            chunks = self._prioritize_chunks(chunks, self.max_chunks_per_patent)
        
        return chunks
    
    def _process_section(
        self,
        section: PatentSection,
        patent_id: str,
        start_order: int,
    ) -> List[HybridChunk]:
        """
        Process a single section, splitting if needed.
        
        Args:
            section: PatentSection to process
            patent_id: Patent identifier
            start_order: Starting order number
        
        Returns:
            List of chunks for this section
        """
        word_count = section.word_count
        section_type_str = section.section_type.value
        weight = self.get_section_weight(section_type_str)
        
        # Create chunk type string
        if section.section_number:
            chunk_type = f"{section_type_str}_{section.section_number}"
        else:
            chunk_type = section_type_str
        
        # If section is small enough, create single chunk
        if word_count <= self.split_threshold:
            chunk_id = f"{patent_id}:{chunk_type}:{start_order}"
            
            return [HybridChunk(
                chunk_id=chunk_id,
                patent_id=patent_id,
                text=section.text,
                chunk_type=chunk_type,
                chunk_level=0,  # Parent level
                parent_chunk_id=None,
                section_type=section_type_str,
                order=start_order,
                chunk_index=0,
                total_chunks_in_section=1,
                word_count=word_count,
                weight=weight,
                metadata=section.metadata,
            )]
        
        # Section is large, apply semantic chunking
        parent_id = f"{patent_id}:{chunk_type}:parent"
        
        if self.include_context:
            context_chunks = self.semantic.split_with_context(
                section.text,
                context_sentences=self.context_sentences,
            )
            
            chunks = []
            for idx, (leading, text, trailing) in enumerate(context_chunks):
                chunk_id = f"{patent_id}:{chunk_type}:{start_order + idx}"
                chunks.append(HybridChunk(
                    chunk_id=chunk_id,
                    patent_id=patent_id,
                    text=text,
                    chunk_type=f"{chunk_type}_part{idx + 1}",
                    chunk_level=1,  # Child level
                    parent_chunk_id=parent_id,
                    section_type=section_type_str,
                    order=start_order + idx,
                    chunk_index=idx,
                    total_chunks_in_section=len(context_chunks),
                    word_count=len(text.split()),
                    weight=weight,
                    metadata=section.metadata,
                    leading_context=leading,
                    trailing_context=trailing,
                ))
            
            return chunks
        else:
            semantic_chunks = self.semantic.split_text(section.text)
            
            chunks = []
            for idx, sem_chunk in enumerate(semantic_chunks):
                chunk_id = f"{patent_id}:{chunk_type}:{start_order + idx}"
                chunks.append(HybridChunk(
                    chunk_id=chunk_id,
                    patent_id=patent_id,
                    text=sem_chunk.text,
                    chunk_type=f"{chunk_type}_part{idx + 1}",
                    chunk_level=1,
                    parent_chunk_id=parent_id,
                    section_type=section_type_str,
                    order=start_order + idx,
                    chunk_index=idx,
                    total_chunks_in_section=len(semantic_chunks),
                    word_count=sem_chunk.word_count,
                    weight=weight,
                    metadata=section.metadata,
                ))
            
            return chunks
    
    def _prioritize_chunks(
        self, chunks: List[HybridChunk], max_count: int
    ) -> List[HybridChunk]:
        """
        Prioritize chunks when over limit.
        
        Keeps high-weight chunks and ensures coverage of all section types.
        """
        if len(chunks) <= max_count:
            return chunks
        
        # Sort by weight (descending), then by order (ascending)
        sorted_chunks = sorted(
            chunks,
            key=lambda c: (-c.weight, c.order)
        )
        
        # Ensure we keep at least one chunk from each section type
        section_types_seen = set()
        prioritized = []
        remaining = []
        
        for chunk in sorted_chunks:
            if chunk.section_type not in section_types_seen:
                prioritized.append(chunk)
                section_types_seen.add(chunk.section_type)
            else:
                remaining.append(chunk)
        
        # Fill remaining slots
        slots_left = max_count - len(prioritized)
        prioritized.extend(remaining[:slots_left])
        
        # Re-sort by order for consistency
        prioritized.sort(key=lambda c: c.order)
        
        return prioritized
    
    def get_parent_chunks(self, chunks: List[HybridChunk]) -> Dict[str, List[HybridChunk]]:
        """
        Group child chunks by their parent ID.
        
        Useful for retrieval expansion.
        """
        parent_map: Dict[str, List[HybridChunk]] = {}
        
        for chunk in chunks:
            if chunk.parent_chunk_id:
                if chunk.parent_chunk_id not in parent_map:
                    parent_map[chunk.parent_chunk_id] = []
                parent_map[chunk.parent_chunk_id].append(chunk)
        
        return parent_map
    
    def expand_with_siblings(
        self,
        chunk: HybridChunk,
        all_chunks: List[HybridChunk],
        include_context: bool = True,
    ) -> str:
        """
        Expand a chunk with its sibling chunks for better context.
        
        Args:
            chunk: The retrieved chunk
            all_chunks: All chunks from the same patent
            include_context: Whether to include leading/trailing context
        
        Returns:
            Expanded text with siblings
        """
        if not chunk.parent_chunk_id:
            # No parent, return chunk with context
            if include_context and chunk.leading_context:
                return f"{chunk.leading_context} {chunk.text} {chunk.trailing_context}".strip()
            return chunk.text
        
        # Find sibling chunks
        siblings = [
            c for c in all_chunks
            if c.parent_chunk_id == chunk.parent_chunk_id
        ]
        siblings.sort(key=lambda c: c.chunk_index)
        
        # Concatenate sibling texts
        texts = [s.text for s in siblings]
        return " ".join(texts)


def chunk_patents_batch(
    patents: List[Dict[str, Any]],
    chunker: Optional[HybridChunker] = None,
    **chunker_kwargs,
) -> List[HybridChunk]:
    """
    Chunk a batch of patents.
    
    Args:
        patents: List of patent records
        chunker: Optional pre-configured chunker
        **chunker_kwargs: Arguments to pass to HybridChunker
    
    Returns:
        List of all chunks from all patents
    """
    if chunker is None:
        chunker = HybridChunker(**chunker_kwargs)
    
    all_chunks = []
    for patent in patents:
        chunks = chunker.chunk_patent(patent)
        all_chunks.extend(chunks)
    
    return all_chunks

