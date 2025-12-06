from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List

from app.agents.base import AgentResult, BaseAgent
import psycopg
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)


class LitigationScoutAgent(BaseAgent):
    name = "litigation_scout"
    
    def __init__(self, settings=None):
        super().__init__(settings)
        self.settings = settings
        # Get config values with defaults
        self.risk_threshold = 3  # High risk if 3+ cases
        self.recent_years = 2    # Recent = last 2 years
        if settings:
            try:
                litigation_config = settings.litigation_scout
                if hasattr(litigation_config, 'risk_threshold'):
                    self.risk_threshold = litigation_config.risk_threshold
                if hasattr(litigation_config, 'recent_years'):
                    self.recent_years = litigation_config.recent_years
            except AttributeError:
                pass  # Use defaults

    async def run(self, query: str, retrieved_patent_ids: List[str] | None = None, **kwargs) -> AgentResult:
        """
        Analyze litigation risk for specific patents (legacy mode).
        
        Use run_general_mode() for parallel execution with Risk Entities.
        """
        """
        Analyze litigation risk for patents.
        
        Args:
            query: User query (may contain patent numbers)
            retrieved_patent_ids: List of patent IDs from CitationMapperAgent
        """
        if not self.settings:
            return AgentResult(
                agent=self.name,
                success=False,
                data={},
                error="Settings not initialized",
            )
        
        try:
            pg_cfg = self.settings.database
            conn_str = f"postgresql://{pg_cfg.user}:{pg_cfg.password}@{pg_cfg.host}:{pg_cfg.port}/{pg_cfg.database}"
            
            with psycopg.connect(conn_str, row_factory=dict_row) as conn:
                with conn.cursor() as cur:
                    # Get patent IDs to check
                    patent_ids_to_check = []
                    
                    # Use retrieved patent IDs if provided
                    if retrieved_patent_ids:
                        patent_ids_to_check = [pid for pid in retrieved_patent_ids if pid]
                    
                    # Also try to extract patent numbers from query
                    import re
                    query_patents = re.findall(r'\b(US|EP|WO|JP|CN|KR|GB|DE|FR)-\d+[A-Z]?\d*\b', query, re.IGNORECASE)
                    if query_patents:
                        patent_ids_to_check.extend(query_patents)
                    
                    # Remove duplicates
                    patent_ids_to_check = list(set(patent_ids_to_check))
                    
                    if not patent_ids_to_check:
                        # No specific patents to check, return general stats
                        return self._get_general_stats(conn, cur)
                    
                    # Query litigation data for these patents
                    cur.execute(
                        """
                        SELECT 
                            patent_id,
                            COUNT(*) as case_count,
                            COUNT(CASE WHEN case_status IN ('active', 'pending', 'Active', 'Pending') THEN 1 END) as active_count,
                            COUNT(CASE WHEN filing_date >= CURRENT_DATE - INTERVAL '%s years' THEN 1 END) as recent_count,
                            COUNT(CASE WHEN case_status IN ('resolved', 'closed', 'Resolved', 'Closed', 'settled', 'Settled') THEN 1 END) as resolved_count
                        FROM patent_litigation
                        WHERE patent_id = ANY(%s)
                        GROUP BY patent_id
                        """,
                        (self.recent_years, patent_ids_to_check),
                    )
                    
                    patent_stats = {row["patent_id"]: dict(row) for row in cur.fetchall()}
                    
                    # Get detailed case information for high-risk patents
                    high_risk_patents = [
                        pid for pid, stats in patent_stats.items()
                        if stats["case_count"] >= self.risk_threshold
                    ]
                    
                    case_details = []
                    if high_risk_patents:
                        cur.execute(
                            """
                            SELECT 
                                patent_id,
                                case_number,
                                court_name,
                                filing_date,
                                case_status,
                                outcome,
                                plaintiff_name,
                                defendant_name
                            FROM patent_litigation
                            WHERE patent_id = ANY(%s)
                            ORDER BY filing_date DESC
                            LIMIT 20
                            """,
                            (high_risk_patents,),
                        )
                        case_details = [dict(row) for row in cur.fetchall()]
                    
                    # Calculate aggregate metrics
                    total_cases = sum(stats["case_count"] for stats in patent_stats.values())
                    active_cases = sum(stats["active_count"] for stats in patent_stats.values())
                    recent_cases = sum(stats["recent_count"] for stats in patent_stats.values())
                    resolved_cases = sum(stats["resolved_count"] for stats in patent_stats.values())
                    
                    # Calculate risk score (0-100)
                    risk_score = self._calculate_risk_score(
                        total_cases=total_cases,
                        active_cases=active_cases,
                        recent_cases=recent_cases,
                        high_risk_count=len(high_risk_patents),
                        patents_checked=len(patent_ids_to_check),
                    )
                    
                    data: Dict[str, Any] = {
                        "total_cases": total_cases,
                        "active_cases": active_cases,
                        "resolved_cases": resolved_cases,
                        "recent_cases": recent_cases,
                        "high_risk_patents": high_risk_patents,
                        "high_risk_count": len(high_risk_patents),
                        "patents_checked": len(patent_ids_to_check),
                        "patents_with_cases": len(patent_stats),
                        "case_details": case_details[:10],  # Top 10 most recent
                        "risk_score": risk_score,
                        "risk_level": self._get_risk_level(risk_score),
                    }
            
            return AgentResult(agent=self.name, success=True, data=data)
        except Exception as exc:
            logger.error("LitigationScoutAgent error: %s", exc, exc_info=True)
            return AgentResult(
                agent=self.name,
                success=False,
                data={},
                error=str(exc),
            )
    
    def _get_general_stats(self, conn, cur) -> AgentResult:
        """Get general litigation statistics when no specific patents are queried."""
        try:
            cur.execute("""
                SELECT 
                    COUNT(*) as total_cases,
                    COUNT(DISTINCT patent_id) as patents_with_cases,
                    COUNT(CASE WHEN case_status IN ('active', 'pending', 'Active', 'Pending') THEN 1 END) as active_cases,
                    COUNT(CASE WHEN filing_date >= CURRENT_DATE - INTERVAL '%s years' THEN 1 END) as recent_cases
                FROM patent_litigation
            """, (self.recent_years,))
            
            row = cur.fetchone()
            data = {
                "total_cases": row["total_cases"] if row else 0,
                "patents_with_cases": row["patents_with_cases"] if row else 0,
                "active_cases": row["active_cases"] if row else 0,
                "recent_cases": row["recent_cases"] if row else 0,
                "message": "General statistics (no specific patents queried)",
            }
            return AgentResult(agent=self.name, success=True, data=data)
        except Exception as exc:
            logger.error("Error getting general stats: %s", exc)
            return AgentResult(
                agent=self.name,
                success=False,
                data={},
                error=str(exc),
            )
    
    def _calculate_risk_score(
        self,
        total_cases: int,
        active_cases: int,
        recent_cases: int,
        high_risk_count: int,
        patents_checked: int,
    ) -> int:
        """
        Calculate litigation risk score (0-100).
        
        Higher score = higher risk
        """
        if patents_checked == 0:
            return 0
        
        # Base score from case count
        case_score = min(total_cases * 10, 50)  # Max 50 points for cases
        
        # Active cases multiplier
        active_score = min(active_cases * 15, 30)  # Max 30 points for active cases
        
        # Recent cases multiplier
        recent_score = min(recent_cases * 10, 20)  # Max 20 points for recent cases
        
        # High-risk patents (3+ cases each)
        high_risk_score = min(high_risk_count * 5, 20)  # Max 20 points
        
        total_score = case_score + active_score + recent_score + high_risk_score
        
        # Normalize to 0-100
        return min(int(total_score), 100)
    
    def _get_risk_level(self, risk_score: int) -> str:
        """Convert risk score to level."""
        if risk_score >= 70:
            return "high"
        elif risk_score >= 40:
            return "medium"
        elif risk_score >= 10:
            return "low"
        else:
            return "minimal"
    
    async def run_general_mode(
        self, 
        query: str, 
        risk_entities: Dict[str, Any] | None = None,
        **kwargs
    ) -> AgentResult:
        """
        General mode: Search for litigation using Risk Entities (assignees, topics).
        
        This mode runs in parallel with retrieval and doesn't require specific patent IDs.
        It searches for general risk patterns like "Has {assignee} been sued regarding {topics}?"
        
        Args:
            query: User query (for context)
            risk_entities: Dict with 'assignees', 'topics', 'search_query' from ClaimsAnalyzer
        """
        if not self.settings:
            return AgentResult(
                agent=self.name,
                success=False,
                data={},
                error="Settings not initialized",
            )
        
        try:
            pg_cfg = self.settings.database
            conn_str = f"postgresql://{pg_cfg.user}:{pg_cfg.password}@{pg_cfg.host}:{pg_cfg.port}/{pg_cfg.database}"
            
            # Extract risk entities
            assignees = []
            topics = []
            if risk_entities:
                assignees = risk_entities.get("assignees", [])
                topics = risk_entities.get("topics", [])
            
            # If no risk entities, fall back to query parsing
            if not assignees and not topics:
                # Try to extract from query
                import re
                # Simple heuristic: look for company-like words (capitalized, common company names)
                query_words = query.split()
                potential_assignees = [w for w in query_words if w[0].isupper() and len(w) > 2]
                assignees = potential_assignees[:3]  # Limit to 3
                topics = [w.lower() for w in query_words if len(w) > 4][:5]  # Longer words as topics
            
            with psycopg.connect(conn_str, row_factory=dict_row) as conn:
                with conn.cursor() as cur:
                    # Build SQL query for general risk search
                    # Search for cases where assignees are involved OR topics appear in case descriptions
                    conditions = []
                    params = []
                    
                    # Search by assignees (plaintiff or defendant names)
                    if assignees:
                        assignee_patterns = [f"%{a}%" for a in assignees]
                        conditions.append(
                            "(plaintiff_name ILIKE ANY(%s) OR defendant_name ILIKE ANY(%s))"
                        )
                        params.extend([assignee_patterns, assignee_patterns])
                    
                    # Search by topics (in party names - companies often have topic-related names)
                    # Note: We can't search patent titles directly without a patents table,
                    # so we search in party names which often contain company names related to topics
                    if topics:
                        topic_patterns = [f"%{t}%" for t in topics]
                        conditions.append(
                            "(plaintiff_name ILIKE ANY(%s) OR defendant_name ILIKE ANY(%s))"
                        )
                        params.extend([topic_patterns, topic_patterns])
                    
                    if not conditions:
                        # No search criteria, return general stats
                        return self._get_general_stats(conn, cur)
                    
                    # Combine conditions with OR
                    where_clause = " OR ".join(conditions)
                    
                    # Execute query
                    query_sql = f"""
                        SELECT 
                            patent_id,
                            case_number,
                            court_name,
                            filing_date,
                            case_status,
                            outcome,
                            plaintiff_name,
                            defendant_name,
                            COUNT(*) OVER (PARTITION BY patent_id) as case_count
                        FROM patent_litigation
                        WHERE {where_clause}
                        ORDER BY filing_date DESC
                        LIMIT 50
                    """
                    
                    cur.execute(query_sql, params)
                    case_details = [dict(row) for row in cur.fetchall()]
                    
                    # Calculate aggregate metrics
                    total_cases = len(case_details)
                    unique_patents = len(set(c.get("patent_id") for c in case_details if c.get("patent_id")))
                    active_cases = len([c for c in case_details if c.get("case_status", "").lower() in ["active", "pending"]])
                    recent_cases = len([
                        c for c in case_details 
                        if c.get("filing_date")
                    ])
                    # Filter by date if needed (simplified - just count all cases for now)
                    # The SQL query already filters by date if needed
                    
                    # Calculate risk score based on general patterns
                    risk_score = self._calculate_general_risk_score(
                        total_cases=total_cases,
                        active_cases=active_cases,
                        recent_cases=recent_cases,
                        unique_patents=unique_patents,
                        assignees_found=len([c for c in case_details if any(a.lower() in str(c.get("plaintiff_name", "")).lower() or a.lower() in str(c.get("defendant_name", "")).lower() for a in assignees)]),
                    )
                    
                    data: Dict[str, Any] = {
                        "total_cases": total_cases,
                        "active_cases": active_cases,
                        "recent_cases": recent_cases,
                        "unique_patents": unique_patents,
                        "case_details": case_details[:20],  # Top 20 most recent
                        "risk_score": risk_score,
                        "risk_level": self._get_risk_level(risk_score),
                        "search_mode": "general",
                        "assignees_searched": assignees,
                        "topics_searched": topics,
                    }
            
            return AgentResult(agent=self.name, success=True, data=data)
        except Exception as exc:
            logger.error("LitigationScoutAgent general mode error: %s", exc, exc_info=True)
            return AgentResult(
                agent=self.name,
                success=False,
                data={},
                error=str(exc),
            )
    
    def _calculate_general_risk_score(
        self,
        total_cases: int,
        active_cases: int,
        recent_cases: int,
        unique_patents: int,
        assignees_found: int,
    ) -> int:
        """
        Calculate litigation risk score for general mode (0-100).
        
        Higher score = higher risk based on general patterns.
        """
        # Base score from case count
        case_score = min(total_cases * 5, 40)  # Max 40 points
        
        # Active cases multiplier
        active_score = min(active_cases * 10, 25)  # Max 25 points
        
        # Recent cases multiplier
        recent_score = min(recent_cases * 8, 20)  # Max 20 points
        
        # Assignee involvement (if searched assignees appear in cases)
        assignee_score = min(assignees_found * 5, 15)  # Max 15 points
        
        total_score = case_score + active_score + recent_score + assignee_score
        
        # Normalize to 0-100
        return min(int(total_score), 100)
