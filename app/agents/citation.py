from __future__ import annotations

from typing import List, Dict, Any

from app.agents.base import AgentResult, BaseAgent
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


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
        else:
            self.qdrant_client = None

    async def run(self, query: str) -> AgentResult:
        if not self.qdrant_client:
            return AgentResult(
                agent=self.name,
                success=False,
                data={},
                error="Qdrant client not initialized",
            )
        
        try:
            # Encode query
            query_vector = self.embedding_model.encode(
                query,
                convert_to_tensor=False,
                normalize_embeddings=True,
            ).tolist()
            
            # Search Qdrant (using query_points for newer API)
            try:
                # Try newer API first (qdrant-client >= 1.7.0)
                search_response = self.qdrant_client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=self.top_k,
                )
                search_results = search_response.points
            except (AttributeError, TypeError):
                # Fallback to older API (qdrant-client < 1.7.0)
                search_results = self.qdrant_client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=self.top_k,
                )
            
            # Format results
            results: List[Dict[str, Any]] = []
            for hit in search_results:
                # Handle both new API (ScoredPoint) and old API (Point) formats
                if hasattr(hit, 'payload'):
                    payload = hit.payload
                    point_id = hit.id
                    score = getattr(hit, 'score', 0.0)
                else:
                    # Old API format
                    payload = hit.get('payload', {}) if isinstance(hit, dict) else {}
                    point_id = hit.get('id', '')
                    score = hit.get('score', 0.0)
                
                results.append({
                    "patent_id": payload.get("patent_id", "") if isinstance(payload, dict) else "",
                    "chunk_id": str(point_id),
                    "score": float(score),
                    "chunk_type": payload.get("chunk_type", "") if isinstance(payload, dict) else "",
                    "chunk_text": (payload.get("chunk_text", "")[:200] if isinstance(payload, dict) else "")[:200],  # Preview
                })
            
            return AgentResult(agent=self.name, success=True, data={"results": results})
        except Exception as exc:
            return AgentResult(
                agent=self.name,
                success=False,
                data={},
                error=str(exc),
            )

