"""Async Qdrant client wrapper with Hybrid Search (Dense + Sparse vectors)."""
import asyncio
from typing import List, Dict, Any, Optional
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SparseVectorParams,
    NamedSparseVector,
    NamedVector,
    Query,
)
try:
    from qdrant_client.models import SparseIndices
except ImportError:
    # Fallback for older versions
    from typing import Dict, List
    SparseIndices = Dict[str, List]  # Use dict representation
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    # Fallback to deprecated import
    from langchain_community.embeddings import HuggingFaceEmbeddings
try:
    from fastembed import TextEmbedding
except ImportError:
    # Fallback if fastembed is not available
    TextEmbedding = None
import numpy as np
from config import settings


class QdrantHybridClient:
    """Qdrant client with hybrid search support (dense + sparse vectors)."""
    
    def __init__(self):
        self.client = AsyncQdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        self.collection_name = settings.qdrant_collection
        
        # Initialize dense embedding model
        self.dense_embedder = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        
        # Initialize sparse embedding model (SPLADE)
        if TextEmbedding:
            try:
                # Try to use a supported SPLADE model
                # Check available models first
                try:
                    supported = TextEmbedding.list_supported_models()
                    # Handle different return types (list of dicts or list of strings)
                    if isinstance(supported, list) and len(supported) > 0:
                        if isinstance(supported[0], dict):
                            model_names = [m.get('model', m.get('name', '')) for m in supported]
                        else:
                            model_names = [str(m) for m in supported]
                    else:
                        model_names = []
                    
                    # Note: fastembed TextEmbedding doesn't include SPLADE models
                    # Sparse vectors require separate SPLADE implementation
                    # For now, we'll use dense-only search (can be enhanced later)
                    self.sparse_embedder = None
                    print("Info: Sparse vectors not available in fastembed. Using dense-only search.")
                    print("      Hybrid search will use dense vectors only. This is still effective for semantic search.")
                except Exception as e:
                    # Fallback: try common model names directly
                    try:
                        self.sparse_embedder = TextEmbedding(model_name="prithivida/Splade_PP_en_v1")
                    except:
                        try:
                            self.sparse_embedder = TextEmbedding(model_name="sentence-transformers/splade-v3")
                        except:
                            self.sparse_embedder = None
                            print(f"Warning: Could not initialize sparse embedder: {e}, sparse vectors disabled")
            except Exception as e:
                self.sparse_embedder = None
                print(f"Warning: fastembed initialization failed: {e}, sparse vectors disabled")
        else:
            self.sparse_embedder = None
            print("Warning: fastembed not available, sparse vectors will be disabled")
    
    async def create_collection_if_not_exists(self, vector_size: int = 768):
        """Create collection with both dense and sparse vector support."""
        collections = await self.client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if self.collection_name not in collection_names:
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE,
                    ),
                    "sparse": SparseVectorParams(),
                },
            )
    
    async def generate_dense_embedding(self, text: str) -> List[float]:
        """Generate dense embedding using snowflake-arctic-embed-m."""
        embedding = await asyncio.to_thread(
            self.dense_embedder.embed_query, text
        )
        return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
    
    async def generate_sparse_embedding(self, text: str) -> Dict[str, Any]:
        """Generate sparse embedding using SPLADE."""
        if not self.sparse_embedder:
            # Return empty sparse vector if fastembed is not available
            return {"indices": [], "values": []}
        
        # FastEmbed SPLADE returns sparse vectors
        embeddings = list(self.sparse_embedder.embed([text]))
        sparse_vec = embeddings[0]
        
        # Convert to Qdrant format: {indices: [int], values: [float]}
        indices = []
        values = []
        for idx, val in enumerate(sparse_vec):
            if abs(val) > 1e-6:  # Only store non-zero values
                indices.append(idx)
                values.append(float(val))
        
        return {"indices": indices, "values": values}
    
    async def upsert_patent(
        self,
        patent_id: str,
        title: str,
        abstract: str,
        metadata: Dict[str, Any],
    ):
        """Upsert a patent with both dense and sparse embeddings."""
        # Generate embeddings
        dense_embedding = await self.generate_dense_embedding(f"{title} {abstract}")
        sparse_embedding = await self.generate_sparse_embedding(f"{title} {abstract}")
        
        # Handle sparse vector format
        if isinstance(SparseIndices, type) and hasattr(SparseIndices, '__call__'):
            sparse_vec = SparseIndices(
                indices=sparse_embedding["indices"],
                values=sparse_embedding["values"],
            )
        else:
            # Use dict format for older Qdrant versions
            sparse_vec = {
                "indices": sparse_embedding["indices"],
                "values": sparse_embedding["values"],
            }
        
        point = PointStruct(
            id=hash(patent_id) % (2**63),  # Convert to int64
            vector={
                "dense": dense_embedding,
                "sparse": sparse_vec,
            },
            payload={
                "patent_id": patent_id,
                "title": title,
                "abstract": abstract,
                **metadata,
            },
        )
        
        await self.client.upsert(
            collection_name=self.collection_name,
            points=[point],
        )
    
    async def hybrid_search(
        self,
        query: str,
        limit: int = 10,
        score_threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search using both dense and sparse vectors."""
        # Generate query embeddings
        dense_query = await self.generate_dense_embedding(query)
        sparse_query = await self.generate_sparse_embedding(query)
        
        # Perform separate searches for dense and sparse
        # Collection uses default vector (not named), so use direct vector
        try:
            from qdrant_client.models import Query
            
            # Use default vector (collection was created without named vectors)
            dense_query_result = await self.client.query(
                collection_name=self.collection_name,
                query=Query(
                    vector=dense_query,  # Default vector (no name)
                    limit=limit * 2,
                    score_threshold=score_threshold,
                    with_payload=True,
                ),
            )
            # Extract results - query() returns QueryResponse with points
            if hasattr(dense_query_result, 'points'):
                dense_results = dense_query_result.points
            elif isinstance(dense_query_result, list):
                dense_results = dense_query_result
            else:
                dense_results = []
        except Exception as e1:
            # Fallback: try query_points with default vector (no using parameter)
            try:
                dense_query_points = await self.client.query_points(
                    collection_name=self.collection_name,
                    query=dense_query,  # Direct vector, no name
                    query_filter=None,
                    limit=limit * 2,
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vectors=False,
                    # No 'using' parameter for default vector
                )
                if hasattr(dense_query_points, 'points'):
                    dense_results = dense_query_points.points
                elif isinstance(dense_query_points, list):
                    dense_results = dense_query_points
                else:
                    dense_results = []
            except Exception as e2:
                print(f"Warning: Dense search failed: {e1}, {e2}")
                dense_results = []
        
        # Handle sparse vector format for search
        if isinstance(SparseIndices, type) and hasattr(SparseIndices, '__call__'):
            sparse_vec = SparseIndices(
                indices=sparse_query["indices"],
                values=sparse_query["values"],
            )
        else:
            sparse_vec = {
                "indices": sparse_query["indices"],
                "values": sparse_query["values"],
            }
        
        # Only do sparse search if sparse embedder is available and we have sparse data
        if self.sparse_embedder and sparse_query.get("indices"):
            try:
                # Use query() method with named sparse vector
                
                sparse_query_result = await self.client.query(
                    collection_name=self.collection_name,
                    query=Query(
                        vector=NamedSparseVector(
                            name="sparse",
                            vector=sparse_vec
                        ),
                        limit=limit * 2,
                        score_threshold=score_threshold,
                        with_payload=True,
                    ),
                )
                if hasattr(sparse_query_result, 'points'):
                    sparse_results = sparse_query_result.points
                elif isinstance(sparse_query_result, list):
                    sparse_results = sparse_query_result
                else:
                    sparse_results = []
            except Exception as e1:
                # Fallback: try query_points
                try:
                    sparse_query_points = await self.client.query_points(
                        collection_name=self.collection_name,
                        query=sparse_vec,
                        query_filter=None,
                        limit=limit * 2,
                        score_threshold=score_threshold,
                        with_payload=True,
                        with_vectors=False,
                        using="sparse",
                    )
                    if hasattr(sparse_query_points, 'points'):
                        sparse_results = sparse_query_points.points
                    elif isinstance(sparse_query_points, list):
                        sparse_results = sparse_query_points
                    else:
                        sparse_results = []
                except Exception as e2:
                    print(f"Warning: Sparse search failed: {e1}, {e2}")
                    sparse_results = []
        else:
            # No sparse vectors available, skip sparse search
            sparse_results = []
        
        # Combine and deduplicate results
        # Handle both point objects and result objects
        combined = {}
        
        for result in dense_results:
            # Handle different result formats
            if hasattr(result, 'payload'):
                payload = result.payload
                score = getattr(result, 'score', 0.0)
            elif isinstance(result, dict):
                payload = result.get('payload', result)
                score = result.get('score', 0.0)
            else:
                continue
                
            patent_id = payload.get("patent_id") if isinstance(payload, dict) else getattr(payload, 'patent_id', None)
            chunk_id = payload.get("chunk_id", "") if isinstance(payload, dict) else getattr(payload, 'chunk_id', '')
            chunk_type = payload.get("chunk_type", "") if isinstance(payload, dict) else getattr(payload, 'chunk_type', '')
            
            if patent_id:
                # Handle both chunked data (chunk_text) and full patent data (title/abstract)
                title = payload.get("title", "") if isinstance(payload, dict) else getattr(payload, 'title', '')
                abstract = payload.get("abstract", "") if isinstance(payload, dict) else getattr(payload, 'abstract', '')
                chunk_text = payload.get("chunk_text", "") if isinstance(payload, dict) else getattr(payload, 'chunk_text', '')
                
                # For structural chunks: prefer title/abstract chunks, use chunk_text as fallback
                if chunk_type == "title" and chunk_text:
                    title = chunk_text
                elif chunk_type == "abstract" and chunk_text:
                    abstract = chunk_text
                elif chunk_text and not abstract:
                    # Use chunk text as abstract if no abstract available
                    abstract = chunk_text[:500]  # Limit length
                
                # Fallback title if missing
                if not title:
                    title = f"Patent {patent_id}"
                
                # If we already have this patent, keep the one with higher score
                if patent_id in combined:
                    if score > combined[patent_id]["score"]:
                        # Update with better scoring chunk
                        combined[patent_id].update({
                            "title": title or combined[patent_id]["title"],
                            "abstract": abstract or combined[patent_id]["abstract"],
                            "score": score,
                            "chunk_id": chunk_id,
                            "chunk_type": chunk_type,
                        })
                else:
                    combined[patent_id] = {
                        "patent_id": patent_id,
                        "chunk_id": chunk_id,
                        "chunk_type": chunk_type,
                        "title": title,
                        "abstract": abstract,
                        "score": score,
                        "source": "dense",
                        "url": payload.get("url", "") if isinstance(payload, dict) else getattr(payload, 'url', ''),
                        "publication_date": payload.get("publication_date", "") if isinstance(payload, dict) else getattr(payload, 'publication_date', ''),
                        "cpc_codes": payload.get("cpc_codes", []) if isinstance(payload, dict) else getattr(payload, 'cpc_codes', []),
                    }
        
        for result in sparse_results:
            # Handle different result formats
            if hasattr(result, 'payload'):
                payload = result.payload
                score = getattr(result, 'score', 0.0)
            elif isinstance(result, dict):
                payload = result.get('payload', result)
                score = result.get('score', 0.0)
            else:
                continue
                
            patent_id = payload.get("patent_id") if isinstance(payload, dict) else getattr(payload, 'patent_id', None)
            chunk_id = payload.get("chunk_id", "") if isinstance(payload, dict) else getattr(payload, 'chunk_id', '')
            chunk_type = payload.get("chunk_type", "") if isinstance(payload, dict) else getattr(payload, 'chunk_type', '')
            
            if patent_id:
                # Handle both chunked data and full patent data
                title = payload.get("title", "") if isinstance(payload, dict) else getattr(payload, 'title', '')
                abstract = payload.get("abstract", "") if isinstance(payload, dict) else getattr(payload, 'abstract', '')
                chunk_text = payload.get("chunk_text", "") if isinstance(payload, dict) else getattr(payload, 'chunk_text', '')
                
                # For structural chunks: prefer title/abstract chunks
                if chunk_type == "title" and chunk_text:
                    title = chunk_text
                elif chunk_type == "abstract" and chunk_text:
                    abstract = chunk_text
                elif chunk_text and not abstract:
                    abstract = chunk_text[:500]
                
                if not title:
                    title = f"Patent {patent_id}"
                
                if patent_id in combined:
                    # Boost score if found in both (weighted average)
                    combined[patent_id]["score"] = (combined[patent_id]["score"] * 0.6 + score * 0.4)
                    combined[patent_id]["source"] = "hybrid"
                    # Update title/abstract if this chunk has better info
                    if chunk_type == "title" and chunk_text:
                        combined[patent_id]["title"] = chunk_text
                    elif chunk_type == "abstract" and chunk_text:
                        combined[patent_id]["abstract"] = chunk_text
                else:
                    combined[patent_id] = {
                        "patent_id": patent_id,
                        "chunk_id": chunk_id,
                        "chunk_type": chunk_type,
                        "title": title,
                        "abstract": abstract,
                        "score": score,
                        "source": "sparse",
                        "url": payload.get("url", "") if isinstance(payload, dict) else getattr(payload, 'url', ''),
                        "publication_date": payload.get("publication_date", "") if isinstance(payload, dict) else getattr(payload, 'publication_date', ''),
                        "cpc_codes": payload.get("cpc_codes", []) if isinstance(payload, dict) else getattr(payload, 'cpc_codes', []),
                    }
        
        # Sort by score and return top results
        sorted_results = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
        return sorted_results[:limit]
    
    async def close(self):
        """Close the client connection."""
        await self.client.close()


# Global instance
qdrant_client = QdrantHybridClient()

