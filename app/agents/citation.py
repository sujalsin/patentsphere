from __future__ import annotations

import re
from typing import List, Dict, Any, Optional
from collections import Counter

from app.agents.base import AgentResult, BaseAgent
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny
from sentence_transformers import SentenceTransformer

# Try to import rank_bm25, fall back to simple keyword matching if not available
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False


class CitationMapperAgent(BaseAgent):
    name = "citation_mapper"
    
    def __init__(self, settings=None):
        super().__init__(settings)
        if settings:
            qdrant_cfg = settings.qdrant
            # Use HTTP for local connections (no SSL)
            self.qdrant_client = QdrantClient(
                host=qdrant_cfg.host,
                port=qdrant_cfg.port,
                api_key=qdrant_cfg.api_key if qdrant_cfg.api_key else None,
                https=False,  # Local Qdrant uses HTTP
            )
            self.collection_name = qdrant_cfg.collection_name
            # Load embedding model for query encoding
            # Auto-detect device if "auto" is specified
            device = settings.embeddings.device
            if device == "auto":
                try:
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device = "cpu"
            
            self.embedding_model = SentenceTransformer(
                settings.embeddings.model_name,
                device=device,
            )
            self.top_k = settings.citation_mapper.top_k
            # Hybrid search settings
            self.hybrid_alpha = getattr(settings.citation_mapper, 'hybrid_search_alpha', 0.7)
            self.bm25_enabled = getattr(settings.citation_mapper, 'bm25_enabled', True)
        else:
            self.qdrant_client = None
            self.hybrid_alpha = 0.7
            self.bm25_enabled = True

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for keyword search."""
        # Remove punctuation and split
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return [w for w in text.split() if len(w) > 2]  # Filter short words

    def _simple_keyword_score(self, query_tokens: List[str], text: str) -> float:
        """Simple keyword matching score (fallback when BM25 not available)."""
        text_tokens = self._tokenize(text)
        if not text_tokens:
            return 0.0
        
        # Count matches
        matches = sum(1 for token in query_tokens if token in text_tokens)
        # Normalize by query length
        return matches / len(query_tokens) if query_tokens else 0.0

    async def _vector_search(self, query_vector: List[float], limit: int) -> List[Dict[str, Any]]:
        """Perform vector search in Qdrant."""
        try:
            search_response = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit * 2,  # Get more results for hybrid combination
            )
            search_results = search_response.points
        except (AttributeError, TypeError):
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit * 2,
            )
        
        results = []
        for hit in search_results:
            if hasattr(hit, 'payload'):
                payload = hit.payload
                point_id = hit.id
                score = getattr(hit, 'score', 0.0)
            else:
                payload = hit.get('payload', {}) if isinstance(hit, dict) else {}
                point_id = hit.get('id', '')
                score = hit.get('score', 0.0)
            
            results.append({
                "patent_id": payload.get("patent_id", "") if isinstance(payload, dict) else "",
                "chunk_id": str(point_id),
                "score": float(score),
                "chunk_type": payload.get("chunk_type", "") if isinstance(payload, dict) else "",
                "chunk_text": payload.get("chunk_text", "") if isinstance(payload, dict) else "",
                "cpc_code": payload.get("cpc_code", "") if isinstance(payload, dict) else "",
                "filing_date": payload.get("filing_date", "") if isinstance(payload, dict) else "",
            })
        
        return results

    async def _bm25_search(
        self, 
        query: str, 
        vector_results: List[Dict[str, Any]],
        limit: int
    ) -> List[Dict[str, Any]]:
        """Perform BM25 keyword search on vector results."""
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return vector_results[:limit]
        
        # Build corpus from vector results
        corpus = []
        for result in vector_results:
            chunk_text = result.get("chunk_text", "")
            corpus.append(self._tokenize(chunk_text))
        
        if not corpus:
            return vector_results[:limit]
        
        # Use BM25 if available, otherwise simple keyword matching
        if BM25_AVAILABLE:
            bm25 = BM25Okapi(corpus)
            scores = bm25.get_scores(query_tokens)
        else:
            # Simple keyword matching
            scores = [
                self._simple_keyword_score(query_tokens, result.get("chunk_text", ""))
                for result in vector_results
            ]
        
        # Combine with vector results
        combined_results = []
        for i, result in enumerate(vector_results):
            bm25_score = scores[i] if i < len(scores) else 0.0
            vector_score = result.get("score", 0.0)
            # Normalize BM25 score to [0, 1] range
            normalized_bm25 = min(bm25_score / (len(query_tokens) + 1), 1.0)
            # Hybrid combination
            hybrid_score = self.hybrid_alpha * vector_score + (1 - self.hybrid_alpha) * normalized_bm25
            result["hybrid_score"] = hybrid_score
            result["bm25_score"] = bm25_score
            combined_results.append(result)
        
        # Sort by hybrid score
        combined_results.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
        return combined_results[:limit]

    def _apply_filters(
        self, 
        results: List[Dict[str, Any]], 
        metadata_filters: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply metadata filters (CPC codes, date range) to results."""
        if not metadata_filters:
            return results
        
        filtered = []
        cpc_codes = metadata_filters.get("cpc_codes", [])
        date_range = metadata_filters.get("date_range")
        
        for result in results:
            # CPC filter
            if cpc_codes:
                result_cpc = result.get("cpc_code", "")
                # Check if result CPC matches any filter CPC (prefix match)
                if result_cpc:
                    matches = any(
                        result_cpc.startswith(cpc) or cpc.startswith(result_cpc.split("/")[0])
                        for cpc in cpc_codes
                    )
                    if not matches:
                        continue
            
            # Date filter
            if date_range and len(date_range) == 2:
                filing_date = result.get("filing_date", "")
                if filing_date:
                    try:
                        # Try to extract year from date string
                        year_match = re.search(r'\d{4}', str(filing_date))
                        if year_match:
                            year = int(year_match.group())
                            if not (date_range[0] <= year <= date_range[1]):
                                continue
                    except (ValueError, AttributeError):
                        pass  # Skip date filtering if parsing fails
            
            filtered.append(result)
        
        return filtered if filtered else results  # Return original if all filtered out

    async def run(
        self, 
        query: str, 
        claims_analysis: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        if not self.qdrant_client:
            return AgentResult(
                agent=self.name,
                success=False,
                data={},
                error="Qdrant client not initialized",
            )
        
        if not self.qdrant_client:
            return AgentResult(
                agent=self.name,
                success=False,
                data={},
                error="Qdrant client not initialized",
            )
        
        try:
            # Extract search query and metadata filters from claims_analysis
            search_query = query
            metadata_filters = None
            if claims_analysis:
                risk_entities = claims_analysis.get("risk_entities")
                if risk_entities and isinstance(risk_entities, dict):
                    search_query = risk_entities.get("search_query", query)
                    if hasattr(risk_entities, "search_query"):
                        search_query = risk_entities.search_query
                
                metadata_filters_dict = claims_analysis.get("metadata_filters")
                if metadata_filters_dict:
                    if hasattr(metadata_filters_dict, "model_dump"):
                        metadata_filters = metadata_filters_dict.model_dump()
                    else:
                        metadata_filters = metadata_filters_dict
            
            # Step 1: Vector search
            query_vector = self.embedding_model.encode(
                search_query,
                convert_to_tensor=False,
                normalize_embeddings=True,
            ).tolist()
            
            vector_results = await self._vector_search(query_vector, self.top_k * 2)
            
            # Step 2: BM25 search (if enabled)
            if self.bm25_enabled:
                results = await self._bm25_search(search_query, vector_results, self.top_k * 2)
            else:
                results = vector_results[:self.top_k * 2]
            
            # Step 3: Apply metadata filters (CPC codes, date range)
            if metadata_filters:
                results = self._apply_filters(results, metadata_filters)
            
            # Step 4: Return top_k results
            final_results = results[:self.top_k]
            
            # Format for output (truncate chunk_text for preview)
            formatted_results = []
            for result in final_results:
                formatted_results.append({
                    "patent_id": result.get("patent_id", ""),
                    "chunk_id": result.get("chunk_id", ""),
                    "score": result.get("hybrid_score", result.get("score", 0.0)),
                    "vector_score": result.get("score", 0.0),
                    "bm25_score": result.get("bm25_score", 0.0),
                    "chunk_type": result.get("chunk_type", ""),
                    "chunk_text": result.get("chunk_text", "")[:200],  # Preview
                    "cpc_code": result.get("cpc_code", ""),
                })
            
            return AgentResult(agent=self.name, success=True, data={"results": formatted_results})
        except Exception as exc:
            return AgentResult(
                agent=self.name,
                success=False,
                data={},
                error=str(exc),
            )

