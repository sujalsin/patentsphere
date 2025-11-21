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
            self.embedding_model = SentenceTransformer(
                settings.embeddings.model_name,
                device=settings.embeddings.device,
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
            
            # Search Qdrant
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=self.top_k,
            )
            
            # Format results
            results: List[Dict[str, Any]] = [
                {
                    "patent_id": hit.payload.get("patent_id", ""),
                    "chunk_id": str(hit.id),
                    "score": float(hit.score),
                    "chunk_type": hit.payload.get("chunk_type", ""),
                    "chunk_text": hit.payload.get("chunk_text", "")[:200],  # Preview
                }
                for hit in search_results
            ]
            
            return AgentResult(agent=self.name, success=True, data={"results": results})
        except Exception as exc:
            return AgentResult(
                agent=self.name,
                success=False,
                data={},
                error=str(exc),
            )

