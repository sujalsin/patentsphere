"""Async PostgreSQL client wrapper using asyncpg."""
import asyncpg
from typing import List, Dict, Any, Optional
from config import settings


class PostgresClient:
    """Async PostgreSQL client for patent metadata and litigation data."""
    
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
    
    async def connect(self):
        """Create connection pool."""
        self.pool = await asyncpg.create_pool(
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password,
            database=settings.postgres_database,
            min_size=5,
            max_size=20,
        )
    
    async def close(self):
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
    
    async def execute(self, query: str, *args):
        """Execute a query."""
        async with self.pool.acquire() as conn:
            return await conn.execute(query, *args)
    
    async def fetch(self, query: str, *args) -> List[Dict[str, Any]]:
        """Fetch rows as dictionaries."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *args)
            return [dict(row) for row in rows]
    
    async def fetch_one(self, query: str, *args) -> Optional[Dict[str, Any]]:
        """Fetch a single row as dictionary."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, *args)
            return dict(row) if row else None
    
    async def get_patent_metadata(self, patent_id: str) -> Optional[Dict[str, Any]]:
        """Get patent metadata by patent ID."""
        query = """
            SELECT 
                id,
                publication_number,
                title,
                publication_date,
                cpc_codes,
                url
            FROM patents_metadata
            WHERE publication_number = $1
        """
        return await self.fetch_one(query, patent_id)
    
    async def get_patents_by_ids(self, patent_ids: List[str]) -> List[Dict[str, Any]]:
        """Get multiple patents by their IDs."""
        if not patent_ids:
            return []
        
        query = """
            SELECT 
                id,
                publication_number,
                title,
                publication_date,
                cpc_codes,
                url
            FROM patents_metadata
            WHERE publication_number = ANY($1::text[])
        """
        return await self.fetch(query, patent_ids)
    
    async def get_litigation_by_patent(self, patent_id: str) -> List[Dict[str, Any]]:
        """Get litigation cases for a specific patent."""
        query = """
            SELECT 
                case_number,
                case_name,
                court_name,
                filing_date,
                case_status,
                plaintiff_name,
                defendant_name,
                patent_id,
                outcome
            FROM litigation_cases
            WHERE patent_id = $1
            ORDER BY filing_date DESC
        """
        return await self.fetch(query, patent_id)
    
    async def get_litigation_by_patents(self, patent_ids: List[str]) -> List[Dict[str, Any]]:
        """Get litigation cases for multiple patents.
        
        Handles patent ID format variations by normalizing both the search terms
        and database values for comparison.
        """
        if not patent_ids:
            return []
        
        # Normalize patent IDs for matching (remove dashes, slashes, spaces, case-insensitive)
        # Create a normalized version for each patent ID to match against
        normalized_ids = []
        for pid in patent_ids:
            if pid:
                # Add original
                normalized_ids.append(pid.upper())
                # Add without dashes
                normalized_ids.append(pid.upper().replace("-", ""))
                # Add without slashes
                normalized_ids.append(pid.upper().replace("/", ""))
                # Add without both
                normalized_ids.append(pid.upper().replace("-", "").replace("/", ""))
        
        # Remove duplicates
        unique_ids = list(set([pid for pid in normalized_ids if pid]))
        # Fuzzy patterns to allow kind-code suffixes (e.g., US12345 -> US12345A)
        like_patterns = [f"{pid}%" for pid in unique_ids]
        
        # Query with case-insensitive matching and format normalization
        # Use UPPER and REPLACE to normalize both sides for comparison
        query = """
            SELECT 
                case_number,
                case_name,
                court_name,
                filing_date,
                case_status,
                plaintiff_name,
                defendant_name,
                patent_id,
                outcome
            FROM litigation_cases
            WHERE patent_id IS NOT NULL
            AND (
                UPPER(REPLACE(REPLACE(patent_id, '-', ''), '/', '')) = ANY($1::text[])
                OR UPPER(patent_id) = ANY($1::text[])
                OR UPPER(REPLACE(patent_id, '-', '')) = ANY($1::text[])
                OR UPPER(REPLACE(patent_id, '/', '')) = ANY($1::text[])
                OR patent_id ILIKE ANY($2::text[])
            )
            ORDER BY filing_date DESC
        """
        return await self.fetch(query, unique_ids, like_patterns)
    
    async def search_litigation_by_keywords(self, keywords: List[str], limit: int = 10) -> List[Dict[str, Any]]:
        """Search litigation cases by keywords in case name, plaintiff, or defendant.
        
        Useful when query asks about litigation but retrieved patents don't have cases.
        """
        if not keywords:
            return []
        
        # Build search terms from keywords (focus on relevant terms)
        search_terms = []
        for keyword in keywords:
            if keyword and not keyword.startswith("CPC:"):
                # Clean keyword - remove common stop words
                clean_keyword = keyword.strip().lower()
                # Skip very short or common words
                if len(clean_keyword) > 3 and clean_keyword not in ["patent", "infringement", "case", "cases"]:
                    search_terms.append(clean_keyword)
        
        if not search_terms:
            return []
        
        # Use the first few most relevant terms
        search_terms = search_terms[:3]
        
        # Build query with ILIKE for case-insensitive matching
        # Search in case_name, plaintiff_name, or defendant_name
        conditions = []
        params = []
        param_idx = 1
        
        for term in search_terms:
            pattern = f"%{term}%"
            conditions.append(
                f"(LOWER(case_name) LIKE ${param_idx} OR LOWER(plaintiff_name) LIKE ${param_idx} OR LOWER(defendant_name) LIKE ${param_idx})"
            )
            params.append(pattern)
            param_idx += 1
        
        if not conditions:
            return []
        
        and_query = f"""
            SELECT 
                case_number,
                case_name,
                court_name,
                filing_date,
                case_status,
                plaintiff_name,
                defendant_name,
                patent_id,
                outcome
            FROM litigation_cases
            WHERE ({' AND '.join(conditions)})
            ORDER BY filing_date DESC
            LIMIT ${param_idx}
        """
        params.append(limit)
        
        results = await self.fetch(and_query, *params)
        if results:
            return results
        
        # Fallback: OR search (limited to 5 results to avoid flooding)
        or_query = and_query.replace(' AND '.join(conditions), ' OR '.join(conditions)).replace(f"LIMIT ${param_idx}", "LIMIT 5")
        return await self.fetch(or_query, *params[:-1])


# Global instance
postgres_client = PostgresClient()


