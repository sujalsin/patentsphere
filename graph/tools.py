"""Retrieval tools with parallel execution for vector search and metadata lookup."""
import asyncio
from typing import List, Dict, Any
from db.qdrant_client import qdrant_client
from db.postgres_client import postgres_client


async def retrieval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieve documents using parallel execution of vector search and metadata lookup."""
    technical_keywords = state.get("technical_keywords") or state.get("keywords") or []
    legal_entities = state.get("legal_entities") or []
    patent_ids_from_extractor = state.get("patent_ids") or []
    intent = state.get("intent", "BOTH")
    
    # Extract patent IDs from technical keywords (if any look like patent numbers)
    potential_patent_ids = []
    for pid in patent_ids_from_extractor:
        if pid:
            potential_patent_ids.append(pid)
    for keyword in technical_keywords:
        if keyword.startswith("US") or keyword.replace("-", "").replace("US", "").isdigit():
            clean_id = keyword.replace("CPC:", "").strip()
            if clean_id and not clean_id.startswith("CPC"):
                potential_patent_ids.append(clean_id)
    
    # Vector search only on technical keywords
    documents = await vector_search(technical_keywords, limit=10)
    
    # Extract patent IDs from retrieved documents
    retrieved_patent_ids = [doc.get("patent_id", "") for doc in documents if doc.get("patent_id")]
    all_patent_ids = list(set(potential_patent_ids + retrieved_patent_ids))
    
    # Parallel execution: enrich documents and get litigation data simultaneously
    enrich_task = enrich_documents_with_metadata(documents)
    metadata_task = metadata_lookup(technical_keywords, legal_entities, intent, all_patent_ids)
    
    enriched_documents, litigation_context = await asyncio.gather(
        enrich_task,
        metadata_task,
    )
    
    return {
        "documents": enriched_documents,
        "litigation_context": litigation_context,
    }


async def vector_search(keywords: List[str], limit: int = 10) -> List[Dict[str, Any]]:
    """Perform hybrid vector search using expanded query variations.
    
    Returns unique patents (not chunks). If multiple chunks from same patent are found,
    keeps the one with highest score.
    """
    all_chunks = []  # Collect all chunks first
    seen_chunk_ids = set()
    
    # Search for each keyword variation
    for keyword in keywords:
        # Skip CPC codes for vector search (they're metadata filters)
        if keyword.startswith("CPC:"):
            continue
        
        try:
            # Search for chunks - get more chunks to ensure we have enough unique patents
            chunk_results = await qdrant_client.hybrid_search(
                query=keyword,
                limit=limit * 3,  # Get more chunks to ensure unique patents
                score_threshold=0.2,  # Lower threshold for better recall
            )
            
            # Collect all chunks (avoid duplicate chunks)
            for chunk in chunk_results:
                chunk_id = chunk.get("chunk_id") or chunk.get("patent_id", "")
                if chunk_id and chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk_id)
                    all_chunks.append(chunk)
        except Exception as e:
            print(f"Error in vector search for '{keyword}': {e}")
            continue
    
    # Group chunks by patent_id and keep best chunk per patent
    patents_dict = {}  # patent_id -> best_chunk
    for chunk in all_chunks:
        patent_id = chunk.get("patent_id", "")
        if not patent_id:
            continue
        
        score = chunk.get("score", 0.0)
        
        # If we haven't seen this patent, or this chunk has a higher score
        if patent_id not in patents_dict or score > patents_dict[patent_id].get("score", 0.0):
            patents_dict[patent_id] = chunk
    
    # Convert to list and sort by score
    unique_patents = list(patents_dict.values())
    unique_patents.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    # Return top N unique patents
    return unique_patents[:limit]


async def metadata_lookup(
    technical_keywords: List[str],
    legal_entities: List[str],
    intent: str,
    potential_patent_ids: List[str],
) -> List[Dict[str, Any]]:
    """Lookup litigation history and patent metadata from PostgreSQL."""
    litigation_context = []
    
    if intent in ["LEGAL", "BOTH"]:
        try:
            if not postgres_client.pool:
                await postgres_client.connect()
            
            patent_ids_to_check = potential_patent_ids.copy()
            
            # Also check for patent-like patterns in technical keywords
            for keyword in technical_keywords:
                if "US" in keyword.upper() and any(c.isdigit() for c in keyword):
                    clean = keyword.replace("CPC:", "").strip()
                    if clean not in patent_ids_to_check:
                        patent_ids_to_check.append(clean)
            
            if patent_ids_to_check:
                litigation_context = await postgres_client.get_litigation_by_patents(
                    patent_ids_to_check
                )
            
            # If no litigation found by patent ID, try legal-entity keyword search
            if not litigation_context:
                print("DEBUG: No litigation found by patent ID, trying legal entity keyword search...")
                
                litigation_keywords = []
                for ent in legal_entities:
                    if ent:
                        litigation_keywords.append(ent.strip())
                
                # Fallback to technical keywords only if we have no legal entities
                if not litigation_keywords:
                    for kw in technical_keywords:
                        if kw and not kw.startswith("CPC:"):
                            litigation_keywords.append(kw.strip())
                
                base_hints = ["infringement", "patent", "case", "lawsuit", "defendant", "plaintiff"]
                for hint in base_hints:
                    if hint not in " ".join(litigation_keywords).lower():
                        litigation_keywords.append(hint)
                
                keyword_litigation = await postgres_client.search_litigation_by_keywords(
                    litigation_keywords, limit=10
                )
                if keyword_litigation:
                    print(f"DEBUG: Found {len(keyword_litigation)} litigation case(s) by keyword search")
                    litigation_context = keyword_litigation
                    
        except Exception as e:
            print(f"Warning: Could not fetch litigation data: {e}")
            import traceback
            traceback.print_exc()
            litigation_context = []
    
    return litigation_context


async def enrich_documents_with_metadata(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Enrich document results with full metadata from PostgreSQL."""
    if not documents:
        return documents
    
    # Extract patent IDs
    patent_ids = [doc.get("patent_id", "") for doc in documents if doc.get("patent_id")]
    
    if not patent_ids:
        return documents
    
    # Ensure PostgreSQL connection is established
    # If connection fails, return documents as-is (graceful degradation)
    try:
        if not postgres_client.pool:
            await postgres_client.connect()
        # Check if pool is still valid
        if postgres_client.pool and postgres_client.pool.is_closing():
            await postgres_client.connect()
    except Exception as e:
        # PostgreSQL unavailable - return documents with available data
        # This is acceptable as Qdrant already has basic metadata
        return documents
    
    # Fetch full metadata
    try:
        # Check pool again before using
        if not postgres_client.pool or (hasattr(postgres_client.pool, 'is_closing') and postgres_client.pool.is_closing()):
            return documents
        
        metadata_records = await postgres_client.get_patents_by_ids(patent_ids)
        
        # Create lookup dict
        metadata_lookup = {
            record["publication_number"]: record
            for record in metadata_records
        }
        
        # Enrich documents
        enriched = []
        for doc in documents:
            patent_id = doc.get("patent_id", "")
            if patent_id in metadata_lookup:
                meta = metadata_lookup[patent_id]
                # Update title and abstract if available from PostgreSQL
                if meta.get("title"):
                    doc["title"] = meta.get("title")
                if meta.get("abstract"):  # If abstract is stored in PostgreSQL
                    doc["abstract"] = meta.get("abstract")
                doc.update({
                    "url": meta.get("url", doc.get("url", "")),
                    "publication_date": meta.get("publication_date"),
                    "cpc_codes": meta.get("cpc_codes"),
                })
            enriched.append(doc)
        
        return enriched
    except Exception as e:
        print(f"Error enriching documents: {e}")
        return documents

