from __future__ import annotations

import asyncio
import importlib.util
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import streamlit as st

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ensure the streamlit_app directory does not shadow the actual app package
if str(SCRIPT_DIR) in sys.path:
    sys.path = [p for p in sys.path if p != str(SCRIPT_DIR)]
    sys.path.append(str(SCRIPT_DIR))

# Explicitly load the backend package to avoid Streamlit's module name clash
if "app" not in sys.modules or not getattr(sys.modules["app"], "__path__", None):
    spec = importlib.util.spec_from_file_location(
        "app", ROOT / "app" / "__init__.py"
    )
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules["app"] = module
        spec.loader.exec_module(module)

from app.agents.base import AgentResult
from app.orchestrator import Orchestrator


# =============================================================================
# Custom CSS for Enhanced Styling
# =============================================================================

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Global styles */
.stApp {
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Header styling */
.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

.main-header h1 {
    color: #e94560;
    font-size: 2.5rem;
    font-weight: 700;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.5px;
}

.main-header p {
    color: #94a3b8;
    font-size: 1.1rem;
    margin: 0;
}

/* Status badge */
.status-badge {
    display: inline-flex;
    align-items: center;
    padding: 0.4rem 0.9rem;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 500;
    gap: 6px;
}

.status-success {
    background: rgba(34, 197, 94, 0.15);
    color: #22c55e;
    border: 1px solid rgba(34, 197, 94, 0.3);
}

.status-error {
    background: rgba(239, 68, 68, 0.15);
    color: #ef4444;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

.status-pending {
    background: rgba(251, 191, 36, 0.15);
    color: #fbbf24;
    border: 1px solid rgba(251, 191, 36, 0.3);
}

/* Agent card */
.agent-card {
    background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 0.8rem;
    transition: transform 0.2s, box-shadow 0.2s;
}

.agent-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.2);
}

.agent-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}

.agent-name {
    font-weight: 600;
    font-size: 1rem;
    color: #f1f5f9;
}

.agent-latency {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #64748b;
}

/* Synthesis panel */
.synthesis-panel {
    background: linear-gradient(145deg, #1e3a5f 0%, #0d2137 100%);
    border: 1px solid rgba(233, 69, 96, 0.2);
    border-radius: 16px;
    padding: 2rem;
    margin: 1.5rem 0;
}

.synthesis-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 1.2rem;
}

.synthesis-header h3 {
    color: #e94560;
    font-size: 1.4rem;
    font-weight: 600;
    margin: 0;
}

.executive-summary {
    background: rgba(0,0,0,0.3);
    border-left: 4px solid #e94560;
    padding: 1.2rem 1.5rem;
    border-radius: 0 12px 12px 0;
    margin: 1rem 0;
    color: #e2e8f0;
    font-size: 1.05rem;
    line-height: 1.7;
}

/* Risk meter */
.risk-meter {
    background: rgba(0,0,0,0.3);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    margin: 1rem 0;
}

.risk-label {
    color: #94a3b8;
    font-size: 0.85rem;
    margin-bottom: 0.5rem;
}

.risk-bar {
    height: 8px;
    background: #1e293b;
    border-radius: 4px;
    overflow: hidden;
}

.risk-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.5s ease;
}

.risk-low { background: linear-gradient(90deg, #22c55e, #4ade80); }
.risk-medium { background: linear-gradient(90deg, #fbbf24, #f59e0b); }
.risk-high { background: linear-gradient(90deg, #f97316, #ef4444); }

/* Patent card */
.patent-card {
    background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 0.8rem;
}

.patent-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 0.8rem;
}

.patent-id {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.95rem;
    color: #60a5fa;
    font-weight: 500;
}

.patent-score {
    background: rgba(96, 165, 250, 0.15);
    color: #60a5fa;
    padding: 0.3rem 0.7rem;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
}

.patent-text {
    color: #cbd5e1;
    font-size: 0.9rem;
    line-height: 1.6;
}

.patent-link {
    color: #60a5fa;
    text-decoration: none;
    font-size: 0.85rem;
    display: inline-flex;
    align-items: center;
    gap: 4px;
    margin-top: 0.8rem;
}

.patent-link:hover {
    text-decoration: underline;
}

/* Section headers */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.1);
}

.section-header h2 {
    color: #f1f5f9;
    font-size: 1.3rem;
    font-weight: 600;
    margin: 0;
}

.section-count {
    background: rgba(96, 165, 250, 0.15);
    color: #60a5fa;
    padding: 0.2rem 0.6rem;
    border-radius: 12px;
    font-size: 0.8rem;
    font-weight: 500;
}

/* Insight section */
.insight-section {
    background: rgba(0,0,0,0.2);
    border-radius: 12px;
    padding: 1.2rem;
    margin: 1rem 0;
}

.insight-title {
    color: #fbbf24;
    font-weight: 600;
    font-size: 1rem;
    margin-bottom: 0.8rem;
}

.insight-bullet {
    background: rgba(0,0,0,0.2);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
}

.bullet-headline {
    color: #e2e8f0;
    font-weight: 500;
    margin-bottom: 0.3rem;
}

.bullet-detail {
    color: #94a3b8;
    font-size: 0.9rem;
    padding-left: 1rem;
}

.bullet-citations {
    color: #60a5fa;
    font-size: 0.8rem;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 0.5rem;
}

/* Citation badges - styled references */
.citation-badge {
    color: #60a5fa;
    font-weight: 600;
    background: rgba(96, 165, 250, 0.2);
    padding: 0.15rem 0.4rem;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    cursor: help;
    position: relative;
    border: 1px solid rgba(96, 165, 250, 0.4);
}

.citation-badge:hover {
    color: #93c5fd;
    background: rgba(96, 165, 250, 0.35);
    border-color: rgba(96, 165, 250, 0.6);
}

/* Citation link style for legend */
.citation-link {
    color: #60a5fa;
    text-decoration: none;
    font-weight: 600;
    background: rgba(96, 165, 250, 0.15);
    padding: 0.1rem 0.3rem;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
}

/* Source card for detailed citation display */
.source-card {
    background: rgba(0, 0, 0, 0.4);
    border: 1px solid rgba(96, 165, 250, 0.2);
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
}

.source-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}

.source-card-id {
    color: #60a5fa;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    font-size: 0.9rem;
}

.source-card-num {
    background: rgba(96, 165, 250, 0.2);
    color: #60a5fa;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    font-size: 0.8rem;
    font-weight: 600;
}

.source-card-text {
    color: #cbd5e1;
    font-size: 0.85rem;
    line-height: 1.6;
    max-height: 150px;
    overflow-y: auto;
    padding: 0.5rem;
    background: rgba(0, 0, 0, 0.2);
    border-radius: 6px;
}

/* Citation expander styling */
.stExpander {
    border: 1px solid rgba(96, 165, 250, 0.2) !important;
    border-radius: 8px !important;
}

/* Sources container styling */
.sources-container {
    background: rgba(0, 0, 0, 0.2);
    border-radius: 12px;
    padding: 1rem;
}

.sources-title {
    color: #60a5fa;
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(96, 165, 250, 0.2);
}

.source-item {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(71, 85, 105, 0.3);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.75rem;
}

.source-item:last-child {
    margin-bottom: 0;
}

.source-item-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.5rem;
}

.source-number {
    color: #60a5fa;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    background: rgba(96, 165, 250, 0.15);
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    font-size: 0.85rem;
}

.source-patent-id {
    color: #22c55e;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
    font-weight: 500;
}

.source-excerpt {
    color: #94a3b8;
    font-size: 0.85rem;
    line-height: 1.5;
    font-style: italic;
    padding-left: 0.5rem;
    border-left: 2px solid rgba(96, 165, 250, 0.3);
}

/* Citation legend */
.citation-legend {
    margin-top: 1.5rem;
    padding: 1rem;
    background: rgba(0, 0, 0, 0.3);
    border-radius: 10px;
    border: 1px solid rgba(96, 165, 250, 0.2);
}

.sources-header {
    background: rgba(96, 165, 250, 0.1);
    padding: 0.8rem 1rem;
    border-radius: 8px;
    margin: 1rem 0;
    color: #cbd5e1;
    font-size: 0.95rem;
    border-left: 3px solid #60a5fa;
}

.legend-title {
    color: #60a5fa;
    font-size: 0.9rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}

.legend-items {
    color: #94a3b8;
    font-size: 0.85rem;
    line-height: 1.8;
}

/* Next steps */
.next-step {
    display: flex;
    gap: 12px;
    padding: 1rem;
    background: rgba(0,0,0,0.2);
    border-radius: 10px;
    margin: 0.5rem 0;
}

.priority-badge {
    padding: 0.3rem 0.7rem;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    flex-shrink: 0;
}

.priority-high { background: rgba(239, 68, 68, 0.2); color: #ef4444; }
.priority-medium { background: rgba(251, 191, 36, 0.2); color: #fbbf24; }
.priority-low { background: rgba(34, 197, 94, 0.2); color: #22c55e; }

.step-content {
    flex: 1;
}

.step-recommendation {
    color: #e2e8f0;
    font-weight: 500;
    margin-bottom: 0.3rem;
}

.step-rationale {
    color: #94a3b8;
    font-size: 0.85rem;
}

/* Query input area */
.query-container {
    background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 2rem;
}

/* Sidebar styling */
.sidebar-header {
    color: #e94560;
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.1);
}

.history-item {
    background: rgba(0,0,0,0.2);
    border-radius: 8px;
    padding: 0.8rem;
    margin-bottom: 0.5rem;
    cursor: pointer;
    transition: background 0.2s;
}

.history-item:hover {
    background: rgba(233, 69, 96, 0.1);
}

.history-time {
    color: #64748b;
    font-size: 0.75rem;
    margin-bottom: 0.3rem;
}

.history-query {
    color: #e2e8f0;
    font-size: 0.85rem;
    line-height: 1.4;
}

/* Metrics row */
.metrics-row {
    display: flex;
    gap: 1rem;
    margin: 1.5rem 0;
}

.metric-card {
    flex: 1;
    background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #60a5fa;
    font-family: 'JetBrains Mono', monospace;
}

.metric-label {
    color: #94a3b8;
    font-size: 0.8rem;
    margin-top: 0.3rem;
}

/* Loading animation */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.loading-indicator {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #60a5fa;
    font-size: 0.9rem;
}

.loading-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #60a5fa;
    animation: pulse 1.5s infinite;
}

.loading-dot:nth-child(2) { animation-delay: 0.2s; }
.loading-dot:nth-child(3) { animation-delay: 0.4s; }

/* Empty state */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: #64748b;
}

.empty-state-icon {
    font-size: 4rem;
    margin-bottom: 1rem;
    opacity: 0.5;
}

.empty-state-title {
    color: #94a3b8;
    font-size: 1.2rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}

.empty-state-desc {
    color: #64748b;
    font-size: 0.95rem;
}

/* Hide default Streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* RLAIF Panel */
.rlaif-panel {
    background: linear-gradient(145deg, #1a2332 0%, #0d1520 100%);
    border: 1px solid rgba(96, 165, 250, 0.2);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1.5rem 0;
}

.rlaif-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1rem;
    color: #60a5fa;
    font-weight: 600;
    font-size: 1.1rem;
}

.rlaif-breakdown {
    display: flex;
    gap: 1.5rem;
    margin: 1rem 0;
}

.rlaif-section {
    flex: 1;
    background: rgba(0,0,0,0.3);
    border-radius: 12px;
    padding: 1rem;
}

.rlaif-section-title {
    font-size: 0.85rem;
    color: #94a3b8;
    margin-bottom: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.ai-feedback-section {
    border: 1px solid rgba(34, 197, 94, 0.3);
}

.heuristic-section {
    border: 1px solid rgba(251, 191, 36, 0.3);
}

.reward-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}

.reward-item:last-child {
    border-bottom: none;
}

.reward-name {
    color: #e2e8f0;
    font-size: 0.9rem;
}

.reward-score {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    padding: 0.2rem 0.5rem;
    border-radius: 6px;
    font-size: 0.85rem;
}

.score-high {
    background: rgba(34, 197, 94, 0.2);
    color: #22c55e;
}

.score-medium {
    background: rgba(251, 191, 36, 0.2);
    color: #fbbf24;
}

.score-low {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
}

.rlaif-total {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 1rem;
    padding: 1rem;
    background: rgba(96, 165, 250, 0.1);
    border-radius: 10px;
    border: 1px solid rgba(96, 165, 250, 0.2);
}

.rlaif-total-label {
    color: #60a5fa;
    font-weight: 600;
}

.rlaif-total-score {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: #60a5fa;
}

/* LangGraph Flow */
.langgraph-panel {
    background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid rgba(168, 85, 247, 0.2);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1.5rem 0;
}

.langgraph-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1rem;
    color: #a855f7;
    font-weight: 600;
    font-size: 1.1rem;
}

.langgraph-flow {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    flex-wrap: wrap;
    padding: 1rem;
    background: rgba(0,0,0,0.3);
    border-radius: 12px;
}

.flow-node {
    padding: 0.6rem 1rem;
    border-radius: 8px;
    font-size: 0.85rem;
    font-weight: 500;
}

.flow-node-active {
    background: rgba(168, 85, 247, 0.2);
    color: #a855f7;
    border: 1px solid rgba(168, 85, 247, 0.4);
}

.flow-node-complete {
    background: rgba(34, 197, 94, 0.2);
    color: #22c55e;
    border: 1px solid rgba(34, 197, 94, 0.4);
}

.flow-arrow {
    color: #64748b;
    font-size: 1.2rem;
}

.langgraph-stats {
    display: flex;
    gap: 2rem;
    margin-top: 1rem;
}

.langgraph-stat {
    text-align: center;
}

.langgraph-stat-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.3rem;
    font-weight: 600;
    color: #a855f7;
}

.langgraph-stat-label {
    font-size: 0.8rem;
    color: #94a3b8;
}

/* Q-Learning Policy */
.policy-panel {
    background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid rgba(34, 197, 94, 0.2);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1.5rem 0;
}

.policy-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1rem;
    color: #22c55e;
    font-weight: 600;
    font-size: 1.1rem;
}

.policy-stats {
    display: flex;
    gap: 1.5rem;
}

.policy-stat {
    flex: 1;
    background: rgba(0,0,0,0.3);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}

.policy-stat-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem;
    font-weight: 600;
    color: #22c55e;
}

.policy-stat-label {
    font-size: 0.8rem;
    color: #94a3b8;
    margin-top: 0.3rem;
}
</style>
"""


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="PatentSphere",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
# Cached Resources
# =============================================================================

@st.cache_resource(show_spinner=False)
def get_orchestrator() -> Orchestrator:
    """Cache a single orchestrator instance across reruns."""
    return Orchestrator()


def run_orchestrator(query: str) -> Mapping[str, AgentResult]:
    """Execute all agents synchronously via the orchestrator."""
    orchestrator = get_orchestrator()
    return asyncio.run(orchestrator.run_all(query))


def serialize_results(results: Mapping[str, AgentResult]) -> Dict[str, Dict[str, Any]]:
    """Convert AgentResult objects into JSON-friendly dicts for session storage."""
    serialized: Dict[str, Dict[str, Any]] = {}
    for name, result in results.items():
        serialized[name] = {
            "success": result.success,
            "data": result.data or {},
            "error": result.error,
        }
    return serialized


def extract_retrieval_rows(serialized: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Find the first agent that exposes retrieval rows."""
    retrieval_agents = ("adaptive_retrieval", "citation", "citation_mapper")
    candidates = ("results", "chunks", "retrieved_chunks", "retrieved")

    for agent_name in retrieval_agents:
        agent_payload = serialized.get(agent_name)
        if not agent_payload:
            continue
        data = agent_payload.get("data") or {}
        for key in candidates:
            rows = data.get(key)
            if rows:
                return rows
    return []


# =============================================================================
# UI Components
# =============================================================================

def render_header():
    """Render the main header."""
    st.markdown("""
    <div class="main-header">
        <h1>🔬 PatentSphere</h1>
        <p>RLAIF-Powered Patent Intelligence • LangGraph Multi-Agent Pipeline • AI Feedback Loop</p>
    </div>
    """, unsafe_allow_html=True)


def render_agent_overview(serialized: Dict[str, Dict[str, Any]]) -> None:
    """Render agent status cards."""
    st.markdown("""
    <div class="section-header">
        <h2>⚡ Agent Status</h2>
    </div>
    """, unsafe_allow_html=True)
    
    agent_items = sorted(serialized.items())
    cols = st.columns(min(4, len(agent_items)))

    for idx, (name, payload) in enumerate(agent_items):
        with cols[idx % len(cols)]:
            success = payload.get("success", False)
            # Check for latency_ms at top level first (AgentOutput), then in data
            latency = payload.get("latency_ms") or payload.get("data", {}).get("latency_ms")
            latency_text = f"{latency:.0f}ms" if isinstance(latency, (int, float)) else "—"
            
            status_class = "status-success" if success else "status-error"
            status_icon = "✓" if success else "✗"
            
            # Format agent name
            display_name = name.replace("_", " ").title()
            
            st.markdown(f"""
            <div class="agent-card">
                <div class="agent-card-header">
                    <span class="agent-name">{display_name}</span>
                    <span class="status-badge {status_class}">{status_icon}</span>
                </div>
                <div class="agent-latency">⏱ {latency_text}</div>
            </div>
            """, unsafe_allow_html=True)
            
            if payload.get("error"):
                st.error(payload["error"])


def render_metrics(serialized: Dict[str, Dict[str, Any]]) -> None:
    """Render key metrics."""
    # Calculate metrics
    total_agents = len(serialized)
    successful = sum(1 for p in serialized.values() if p.get("success"))
    
    synthesis_data = serialized.get("synthesis", {}).get("data", {})
    risk_score = synthesis_data.get("risk_score", "—")
    
    critic_data = serialized.get("critic", {}).get("data", {})
    quality_score = critic_data.get("score", critic_data.get("quality_score", 0))
    quality_pct = f"{quality_score * 100:.0f}%" if isinstance(quality_score, (int, float)) else "—"
    
    retrieval_rows = extract_retrieval_rows(serialized)
    chunks_count = len(retrieval_rows)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{successful}/{total_agents}</div>
            <div class="metric-label">Agents Succeeded</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{chunks_count}</div>
            <div class="metric-label">Patents Retrieved</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{quality_pct}</div>
            <div class="metric-label">Quality Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        risk_color = "#22c55e" if isinstance(risk_score, int) and risk_score < 40 else "#fbbf24" if isinstance(risk_score, int) and risk_score < 70 else "#ef4444"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: {risk_color}">{risk_score}</div>
            <div class="metric-label">Risk Score</div>
        </div>
        """, unsafe_allow_html=True)


def render_rlaif_panel(serialized: Dict[str, Dict[str, Any]]) -> None:
    """Render RLAIF (Reinforcement Learning from AI Feedback) panel."""
    critic_data = serialized.get("critic", {}).get("data", {})
    if not critic_data:
        return

    components = critic_data.get("components", {})
    total_score = critic_data.get("score", 0)
    feedback = critic_data.get("feedback", "")
    
    # AI Feedback components (70% of reward)
    ai_components = {
        "LLM Fluency": components.get("llm_fluency", 0),
        "LLM Relevance": components.get("llm_relevance", 0),
        "LLM Completeness": components.get("llm_completeness", 0),
    }
    
    # Heuristic components (30% of reward)
    heuristic_components = {
        "Citation Overlap": components.get("citation_overlap", 0),
        "CPC Relevance": components.get("cpc_relevance", 0),
        "Temporal Diversity": components.get("temporal_diversity", 0),
    }
    
    def get_score_class(score: float) -> str:
        if score >= 0.7:
            return "score-high"
        elif score >= 0.4:
            return "score-medium"
        return "score-low"
    
    st.markdown("""
    <div class="section-header">
        <h2>🤖 RLAIF - AI Feedback</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="rlaif-panel">
        <div class="rlaif-header">
            <span>🧠</span>
            <span>Reinforcement Learning from AI Feedback</span>
        </div>
        
        <div class="rlaif-breakdown">
            <div class="rlaif-section ai-feedback-section">
                <div class="rlaif-section-title">🤖 AI Feedback (70% weight)</div>
    """, unsafe_allow_html=True)
    
    for name, score in ai_components.items():
        score_class = get_score_class(score)
        st.markdown(f"""
                <div class="reward-item">
                    <span class="reward-name">{name}</span>
                    <span class="reward-score {score_class}">{score:.2f}</span>
                </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"""
            </div>
            
            <div class="rlaif-section heuristic-section">
                <div class="rlaif-section-title">📊 Heuristic Signals (30% weight)</div>
    """, unsafe_allow_html=True)
    
    for name, score in heuristic_components.items():
        score_class = get_score_class(score)
        st.markdown(f"""
                <div class="reward-item">
                    <span class="reward-name">{name}</span>
                    <span class="reward-score {score_class}">{score:.2f}</span>
                </div>
        """, unsafe_allow_html=True)
    
    score_class = get_score_class(total_score)
    st.markdown(f"""
            </div>
        </div>
        
        <div class="rlaif-total">
            <span class="rlaif-total-label">Total Reward Score</span>
            <span class="rlaif-total-score">{total_score:.3f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if feedback:
        st.caption(f"💡 {feedback}")


def render_langgraph_panel(serialized: Dict[str, Dict[str, Any]]) -> None:
    """Render LangGraph pipeline visualization."""
    
    # Get adaptive retrieval data for iterations
    adaptive_data = serialized.get("adaptive_retrieval", {}).get("data", {})
    rl_metadata = adaptive_data.get("rl_metadata", {})
    iterations = rl_metadata.get("iterations", adaptive_data.get("retrieval_depth", 1))
    actions = rl_metadata.get("actions", [])
    
    # Define the pipeline nodes
    nodes = [
        ("Claims Analyzer", True),
        ("Retrieval", True),
        ("Litigation", serialized.get("litigation_scout", {}).get("success", False)),
        ("Synthesis", serialized.get("synthesis", {}).get("success", False)),
        ("Critic", serialized.get("critic", {}).get("success", False)),
        ("Finalize", serialized.get("final", {}).get("success", False)),
    ]
    
    st.markdown("""
    <div class="section-header">
        <h2>🔄 LangGraph Pipeline</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="langgraph-panel">
        <div class="langgraph-header">
            <span>🔀</span>
            <span>Adaptive Agent Orchestration</span>
        </div>
        
        <div class="langgraph-flow">
    """, unsafe_allow_html=True)
    
    for i, (node_name, completed) in enumerate(nodes):
        node_class = "flow-node-complete" if completed else "flow-node-active"
        st.markdown(f"""
            <div class="flow-node {node_class}">{node_name}</div>
        """, unsafe_allow_html=True)
        
        if i < len(nodes) - 1:
            st.markdown('<span class="flow-arrow">→</span>', unsafe_allow_html=True)
    
    st.markdown(f"""
        </div>
        
        <div class="langgraph-stats">
            <div class="langgraph-stat">
                <div class="langgraph-stat-value">{iterations}</div>
                <div class="langgraph-stat-label">Iterations</div>
            </div>
            <div class="langgraph-stat">
                <div class="langgraph-stat-value">{len(actions)}</div>
                <div class="langgraph-stat-label">RL Actions</div>
            </div>
            <div class="langgraph-stat">
                <div class="langgraph-stat-value">{', '.join(actions) if actions else 'N/A'}</div>
                <div class="langgraph-stat-label">Action Sequence</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_policy_panel() -> None:
    """Render Q-Learning policy statistics."""
    from pathlib import Path
    import pickle
    
    policy_path = Path("models/policy.pkl")
    
    if not policy_path.exists():
        return
    
    try:
        with open(policy_path, "rb") as f:
            policy_data = pickle.load(f)
        
        q_table_size = len(policy_data.get("q_table", {}))
        exploration_rate = policy_data.get("exploration_rate", 0)
        
        st.markdown("""
        <div class="section-header">
            <h2>🎯 Q-Learning Policy</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="policy-panel">
            <div class="policy-header">
                <span>📈</span>
                <span>Trained Adaptive Retrieval Policy</span>
            </div>
            
            <div class="policy-stats">
                <div class="policy-stat">
                    <div class="policy-stat-value">{q_table_size}</div>
                    <div class="policy-stat-label">States Learned</div>
                </div>
                <div class="policy-stat">
                    <div class="policy-stat-value">{exploration_rate:.1%}</div>
                    <div class="policy-stat-label">Exploration Rate</div>
                </div>
                <div class="policy-stat">
                    <div class="policy-stat-value">Q-Learning</div>
                    <div class="policy-stat-label">Algorithm</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception:
        pass


def build_citation_index(serialized: Dict[str, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """Build a mapping of citation numbers to patent info including chunk text.
    
    First tries to use the new API response format (sources with ui_state),
    then falls back to retrieval rows.
    """
    citation_map = {}
    
    # Try to get sources from final/api_response (new format)
    final_data = serialized.get("final", {}).get("data", {})
    api_response = final_data.get("api_response", {})
    sources = api_response.get("sources", [])
    
    if sources:
        # Use new API response format with ui_state
        for source in sources:
            idx = source.get("index", 0)
            if idx > 0:
                ui_state = source.get("ui_state", {})
                chunk_text = ui_state.get("chunk_text", source.get("snippet", ""))
                
                citation_map[idx] = {
                    "patent_id": source.get("title", source.get("assignee", "Unknown")),
                    "chunk_text": chunk_text,
                    "preview": chunk_text[:300] + "..." if len(chunk_text) > 300 else chunk_text,
                    "chunk_type": source.get("type", "patent"),
                    "score": source.get("score"),
                    "url": source.get("url"),
                    "expandable": ui_state.get("expandable", True),
                    "ui_state": ui_state,
                }
    
    # Fallback to retrieval rows if no sources found
    if not citation_map:
        retrieval_rows = extract_retrieval_rows(serialized)
        
        for idx, row in enumerate(retrieval_rows, start=1):
            patent_id = row.get("patent_id") or row.get("chunk_id") or f"chunk-{idx}"
            chunk_text = row.get("chunk_text") or row.get("text") or "(no text available)"
            chunk_type = row.get("chunk_type", "")
            score = row.get("score", 0)
            
            # Truncate chunk text for tooltip (first 300 chars)
            preview = chunk_text[:300] + "..." if len(chunk_text) > 300 else chunk_text
            # Escape HTML characters for tooltip
            preview_escaped = preview.replace('"', '&quot;').replace("'", "&#39;").replace("\n", " ").replace("<", "&lt;").replace(">", "&gt;")
            
            citation_map[idx] = {
                "patent_id": patent_id,
                "chunk_text": chunk_text,
                "preview": preview_escaped,
                "chunk_type": chunk_type,
                "score": score,
                "expandable": True,
            }
    
    return citation_map


def make_citations_styled(text: str, citation_map: Dict[int, Dict[str, Any]]) -> str:
    """Convert citation references like [1], [2] into styled clickable badges."""
    import re
    
    def replace_citation(match):
        cite_num = int(match.group(1))
        if cite_num in citation_map:
            info = citation_map[cite_num]
            patent_id = info.get("patent_id", f"Source {cite_num}")
            # Create a styled badge with tooltip showing patent ID
            # The badge is styled to look clickable and will correspond to expandable sections below
            return f'''<span class="citation-badge" title="📄 {patent_id} - Click to view source details below">[{cite_num}]</span>'''
        return match.group(0)
    
    # Match patterns like [1], [2], [3], etc.
    return re.sub(r'\[(\d+)\]', replace_citation, text)


def render_citation_sources(citation_map: Dict[int, Dict[str, Any]]) -> None:
    """Render clickable source cards using Streamlit-native expandable components with UI state."""
    if not citation_map:
        return
    
    st.markdown("""
    <div class="citation-legend">
        <div class="legend-title">📑 Click citations above or expand sources below to view details</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create expandable sections for each source
    num_sources = min(len(citation_map), 20)
    cols = st.columns(min(5, num_sources))
    
    for idx, (num, info) in enumerate(list(citation_map.items())[:num_sources]):
        patent_id = info.get("patent_id", f"Source {num}")
        chunk_text = info.get("chunk_text", "")
        url = info.get("url")
        expandable = info.get("expandable", True)
        ui_state = info.get("ui_state", {})
        score = info.get("score")
        
        # Use full chunk text from ui_state if available
        if ui_state and ui_state.get("chunk_text"):
            chunk_text = ui_state.get("chunk_text")
        
        # Truncate for label
        label_text = patent_id[:30] + "..." if len(patent_id) > 30 else patent_id
        expander_label = f"[{num}] {label_text}"
        
        with cols[idx % len(cols)]:
            with st.expander(expander_label, expanded=False):
                st.markdown(f"""
                <div class="source-card">
                    <div class="source-card-header">
                        <span class="source-card-id">📄 {patent_id}</span>
                        <span class="source-card-num">[{num}]</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Show URL if available
                if url:
                    st.markdown(f"🔗 [View Patent]({url})")
                
                # Show score if available
                if score is not None:
                    st.caption(f"Relevance Score: {score:.2f}")
                
                # Show full chunk text in expandable section
                st.markdown("**Excerpt used by AI:**")
                if chunk_text:
                    st.text_area(
                        "Full text",
                        value=chunk_text,
                        height=200,
                        key=f"citation_{num}_text_{idx}",
                        label_visibility="collapsed",
                        disabled=True,
                    )
                else:
                    st.caption("(No text available)")


def render_synthesis(serialized: Dict[str, Dict[str, Any]]) -> None:
    """Render synthesis results using new API response format."""
    # Try to get API response from final agent (new format)
    final_data = serialized.get("final", {}).get("data", {})
    api_response = final_data.get("api_response", {})
    
    # Build citation index for clickable links
    citation_map = build_citation_index(serialized)
    
    # Use new API response format if available
    if api_response and api_response.get("answer_section"):
        answer_section = api_response.get("answer_section", {})
        technical_summary = answer_section.get("technical_summary", "")
        legal_summary = answer_section.get("legal_summary")
        novelty_assessment = answer_section.get("novelty_assessment")
        risk_score = api_response.get("risk_score", 50)
        quality_score = api_response.get("quality_score", 0.0)
        sources_header = api_response.get("sources_header", "")
        
        st.markdown("""
        <div class="section-header">
            <h2>📊 Analysis</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Render sources header
        if sources_header:
            st.markdown(f"""
            <div class="sources-header">
                <strong>Sources:</strong> {sources_header}
            </div>
            """, unsafe_allow_html=True)
        
        # Render clickable citation sources
        render_citation_sources(citation_map)
        
        # Technical Summary
        if technical_summary:
            styled_summary = make_citations_styled(technical_summary, citation_map)
            st.markdown(f"""
            <div class="synthesis-panel">
                <div class="synthesis-header">
                    <h3>🔬 Technical Landscape</h3>
                </div>
                <div class="executive-summary">{styled_summary}</div>
            """, unsafe_allow_html=True)
            
            # Risk and Quality meters
            risk_class = "risk-low" if risk_score < 40 else "risk-medium" if risk_score < 70 else "risk-high"
            quality_class = "risk-low" if quality_score < 0.4 else "risk-medium" if quality_score < 0.7 else "risk-high"
            
            st.markdown(f"""
                <div class="risk-meter">
                    <div class="risk-label">Risk Assessment: {risk_score}/100</div>
                    <div class="risk-bar">
                        <div class="risk-fill {risk_class}" style="width: {risk_score}%"></div>
                    </div>
                </div>
                <div class="risk-meter" style="margin-top: 1rem;">
                    <div class="risk-label">Quality Score: {quality_score:.2f}/1.0</div>
                    <div class="risk-bar">
                        <div class="risk-fill {quality_class}" style="width: {quality_score * 100}%"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Legal Summary
        if legal_summary:
            styled_legal = make_citations_styled(legal_summary, citation_map)
            st.markdown(f"""
            <div class="synthesis-panel" style="margin-top: 1.5rem;">
                <div class="synthesis-header">
                    <h3>⚖️ Legal Risk</h3>
                </div>
                <div class="executive-summary">{styled_legal}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Novelty Assessment
        if novelty_assessment:
            styled_novelty = make_citations_styled(novelty_assessment, citation_map)
            st.markdown(f"""
            <div class="synthesis-panel" style="margin-top: 1.5rem;">
                <div class="synthesis-header">
                    <h3>💡 Novelty Assessment</h3>
                </div>
                <div class="executive-summary">{styled_novelty}</div>
            </div>
            """, unsafe_allow_html=True)
        
        return
    
    # Fallback to legacy format (synthesis agent output)
    synthesis = serialized.get("synthesis")
    if not synthesis:
        return

    data = synthesis.get("data") or {}
    summary = data.get("executive_summary")
    sections = data.get("insight_sections") or []
    next_steps = data.get("next_steps") or data.get("action_items") or []
    citations = data.get("citations") or []
    risk_score = data.get("risk_score", 50)

    st.markdown("""
    <div class="section-header">
        <h2>📊 Analysis Summary</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Render clickable source cards using Streamlit-native components
    render_citation_sources(citation_map)

    # Patent Analysis Response
    if summary:
        # Make citations styled
        styled_summary = make_citations_styled(summary, citation_map)
        
        st.markdown(f"""
        <div class="synthesis-panel">
            <div class="synthesis-header">
                <h3>📊 Patent Landscape Analysis</h3>
            </div>
            <div class="executive-summary">{styled_summary}</div>
        """, unsafe_allow_html=True)
        
        # Risk meter
        risk_class = "risk-low" if risk_score < 40 else "risk-medium" if risk_score < 70 else "risk-high"
        st.markdown(f"""
            <div class="risk-meter">
                <div class="risk-label">Risk Assessment: {risk_score}/100</div>
                <div class="risk-bar">
                    <div class="risk-fill {risk_class}" style="width: {risk_score}%"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Synthesis agent did not return a summary.")

    # Insight sections
    if sections:
        for section in sections:
            title = section.get("title", "Insights")
            st.markdown(f"""
            <div class="insight-section">
                <div class="insight-title">💡 {title}</div>
            """, unsafe_allow_html=True)
            
            for bullet in section.get("bullets", []):
                headline = bullet.get("headline", "Insight")
                details = bullet.get("details") or []
                cites = bullet.get("citations") or []
                
                # Make details with citation badges
                clickable_details = [make_citations_styled(d, citation_map) for d in details]
                details_html = "".join([f'<div class="bullet-detail">• {d}</div>' for d in clickable_details])
                
                # Show citation patent IDs as badges (no external links)
                if cites:
                    cite_badges = []
                    for cite in cites:
                        cite_badges.append(f'<span class="citation-link">{cite}</span>')
                    cites_html = f'<div class="bullet-citations">Sources: {" ".join(cite_badges)}</div>'
                else:
                    cites_html = ""
                
                st.markdown(f"""
                <div class="insight-bullet">
                    <div class="bullet-headline">{make_citations_styled(headline, citation_map)}</div>
                    {details_html}
                    {cites_html}
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

    # Next steps
    if next_steps:
        st.markdown("""
        <div class="section-header">
            <h2>🎯 Recommended Actions</h2>
        </div>
        """, unsafe_allow_html=True)
        
        for item in next_steps:
            priority = (item.get("priority") or "medium").lower()
            recommendation = item.get("recommendation", "")
            rationale = item.get("rationale", "")
            
            st.markdown(f"""
            <div class="next-step">
                <span class="priority-badge priority-{priority}">{priority}</span>
                <div class="step-content">
                    <div class="step-recommendation">{recommendation}</div>
                    <div class="step-rationale">{rationale}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_retrieval_section(rows: List[Dict[str, Any]]) -> None:
    """Render retrieved patent chunks."""
    st.markdown(f"""
    <div class="section-header">
        <h2>📚 Retrieved Patents</h2>
        <span class="section-count">{len(rows)} results</span>
    </div>
    """, unsafe_allow_html=True)
    
    if not rows:
        st.info("No retrieval results returned for this query.")
        return

    top_rows = rows[:15]
    for idx, row in enumerate(top_rows, start=1):
        patent_id = row.get("patent_id") or row.get("chunk_id") or f"chunk-{idx}"
        score = row.get("score", 0)
        chunk_text = row.get("chunk_text") or row.get("text") or "(no preview available)"
        chunk_type = row.get("chunk_type", "")
        
        # Format patent ID for Google Patents link
        clean_id = patent_id.replace("-", "").replace(" ", "")
        google_url = f"https://patents.google.com/patent/{clean_id}"
        
        with st.expander(f"#{idx} {patent_id} • Score: {score:.3f}", expanded=(idx == 1)):
            st.markdown(f"""
            <div class="patent-card">
                <div class="patent-header">
                    <span class="patent-id">{patent_id}</span>
                    <span class="patent-score">{score:.3f}</span>
                </div>
                <p class="patent-text">{chunk_text[:500]}{'...' if len(chunk_text) > 500 else ''}</p>
                <a href="{google_url}" target="_blank" class="patent-link">
                    View on Google Patents →
                </a>
            </div>
            """, unsafe_allow_html=True)
            
            if chunk_type:
                st.caption(f"Section: {chunk_type}")


def render_agent_logs(serialized: Dict[str, Dict[str, Any]]) -> None:
    """Render detailed agent logs and raw data."""
    st.markdown("""
    <div class="section-header">
        <h2>🔍 Debug Details</h2>
    </div>
    """, unsafe_allow_html=True)
    
    tabs = st.tabs([name.replace("_", " ").title() for name in sorted(serialized.keys())])
    
    for tab, (name, payload) in zip(tabs, sorted(serialized.items())):
        with tab:
            data = payload.get("data") or {}
            
            # Show notes/logs if present
            logs = data.get("logs") or data.get("notes")
            if logs:
                st.markdown("**Agent Notes:**")
                if isinstance(logs, list):
                    for entry in logs:
                        st.markdown(f"- {entry}")
                else:
                    st.markdown(logs)
            
            # Show raw JSON
            with st.expander("Raw JSON Output", expanded=False):
                st.json(data)
            
            if payload.get("error"):
                st.error(f"Error: {payload['error']}")


def render_sidebar(history: List[Dict[str, Any]]) -> None:
    """Render the sidebar with history and info."""
    import pickle
    from pathlib import Path
    
    with st.sidebar:
        st.markdown('<div class="sidebar-header">📜 Query History</div>', unsafe_allow_html=True)
        
        if not history:
            st.caption("No queries yet. Run your first analysis!")
        else:
            for idx, entry in enumerate(history[:10]):
                timestamp = entry.get("timestamp", "")
                query_text = entry.get("query", "")
                short_query = query_text[:50] + "..." if len(query_text) > 50 else query_text
                
                # Make history items clickable
                if st.button(f"📌 {short_query}", key=f"history_{idx}", use_container_width=True):
                    st.session_state["selected_history"] = idx
                    st.rerun()
                
                st.caption(f"🕐 {timestamp}")
        
        st.markdown("---")
        
        # RLAIF Status
        st.markdown('<div class="sidebar-header">🤖 RLAIF Status</div>', unsafe_allow_html=True)
        
        policy_path = Path("models/policy.pkl")
        if policy_path.exists():
            try:
                with open(policy_path, "rb") as f:
                    policy_data = pickle.load(f)
                q_size = len(policy_data.get("q_table", {}))
                exp_rate = policy_data.get("exploration_rate", 0)
                st.caption(f"🎯 Q-Table: {q_size} states")
                st.caption(f"🎲 Exploration: {exp_rate:.1%}")
                st.caption("✅ Policy: Trained")
            except:
                st.caption("⚠️ Policy: Error loading")
        else:
            st.caption("⏳ Policy: Not trained")
        
        st.caption("🧠 AI Feedback: 70%")
        st.caption("📊 Heuristics: 30%")
        
        st.markdown("---")
        
        # LangGraph Status
        st.markdown('<div class="sidebar-header">🔄 LangGraph</div>', unsafe_allow_html=True)
        st.caption("🔀 Mode: Adaptive")
        st.caption("🔁 Max Iterations: 3")
        st.caption("📈 Quality Threshold: 0.6")
        
        st.markdown("---")
        
        st.markdown('<div class="sidebar-header">ℹ️ Services</div>', unsafe_allow_html=True)
        
        try:
            import httpx
            response = httpx.get("http://localhost:8000/status", timeout=2)
            if response.status_code == 200:
                status = response.json()
                services = status.get("services", {})
                
                for svc_name, svc_info in services.items():
                    svc_status = svc_info.get("status", "unknown")
                    icon = "🟢" if svc_status == "ok" else "🔴"
                    st.caption(f"{icon} {svc_name.title()}: {svc_status}")
        except:
            st.caption("API status unavailable")


def render_empty_state():
    """Render empty state when no results."""
    st.markdown("""
    <div class="empty-state">
        <div class="empty-state-icon">🔍</div>
        <div class="empty-state-title">Ready to Analyze</div>
        <div class="empty-state-desc">Enter a patent query above to start the multi-agent analysis pipeline.</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 💡 Example Queries")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Technical Search:**
        - "Graph neural network accelerators for edge devices"
        - "Blockchain-based supply chain tracking"
        - "Quantum computing error correction methods"
        """)
    
    with col2:
        st.markdown("""
        **Competitive Analysis:**
        - "IBM patents in machine learning optimization"
        - "Solar panel efficiency improvements since 2020"
        - "Patent litigation in autonomous vehicles"
        """)


# =============================================================================
# Main Application
# =============================================================================

def main() -> None:
    """Main application entry point."""

    # Initialize session state
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "selected_history" not in st.session_state:
        st.session_state["selected_history"] = 0
    
    # Render header
    render_header()
    
    # Query input section
    st.markdown('<div class="query-container">', unsafe_allow_html=True)

    query = st.text_area(
        "🔎 Enter your patent research query",
        height=100,
        placeholder="e.g., What are the key patents in transformer architecture for NLP?",
        label_visibility="collapsed"
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        run_clicked = st.button(
            "🚀 Run Analysis",
            type="primary",
            use_container_width=True
        )
    
    with col2:
        clear_clicked = st.button(
            "🗑️ Clear History",
            use_container_width=True
        )
    
    with col3:
        if st.button("📊 View Status", use_container_width=True):
            try:
                import httpx
                response = httpx.get("http://localhost:8000/status", timeout=5)
                st.json(response.json())
            except Exception as e:
                st.error(f"Failed to fetch status: {e}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Handle actions
    if clear_clicked:
        st.session_state["history"] = []
        st.rerun()

    if run_clicked:
        clean_query = query.strip()
        
        # Input validation
        if not clean_query:
            st.warning("⚠️ Please enter a query before running the analysis.")
        elif len(clean_query) < 3:
            st.warning("⚠️ Query too short. Please enter at least 3 characters.")
        elif len(clean_query) > 2000:
            st.warning("⚠️ Query too long. Maximum 2000 characters allowed.")
        else:
            with st.spinner("🔄 Running multi-agent analysis pipeline..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.markdown("*Initializing agents...*")
                    progress_bar.progress(10)
                    
                    results = run_orchestrator(clean_query)
                    progress_bar.progress(90)
                    
                    # Check for errors in results
                    if "error" in results and results["error"].error:
                        st.error(f"❌ {results['error'].error}")
                    else:
                        status_text.markdown("*Finalizing results...*")
                    serialized = serialize_results(results)
                        
                    st.session_state["history"].insert(
                        0,
                        {
                                "query": clean_query,
                            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                            "results": serialized,
                        },
                    )
                    
                    progress_bar.progress(100)
                    status_text.empty()
                    progress_bar.empty()
                    
                except Exception as exc:
                    progress_bar.empty()
                    status_text.empty()
                    logger.exception("Analysis failed: %s", exc)
                    st.error(f"❌ Analysis failed: {exc}")

    # Render sidebar
    history = st.session_state.get("history", [])
    render_sidebar(history)

    # Render results or empty state
    if not history:
        render_empty_state()
    else:
        # Get selected history item (default to most recent)
        selected_idx = st.session_state.get("selected_history", 0)
        if selected_idx >= len(history):
            selected_idx = 0
        selected = history[selected_idx]
        
        label = "Latest Analysis" if selected_idx == 0 else f"History #{selected_idx + 1}"
        
        st.markdown(f"""
        <div style="margin: 1.5rem 0; padding: 1rem; background: rgba(0,0,0,0.2); border-radius: 10px;">
            <span style="color: #94a3b8;">{label}</span> • 
            <span style="color: #64748b;">{selected.get('timestamp')}</span>
            <div style="color: #e2e8f0; font-size: 1.1rem; margin-top: 0.5rem;">
                "{selected.get('query', '')}"
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        serialized = selected.get("results", {})
        
        # Render metrics
        render_metrics(serialized)
        
        # Render agent overview
        render_agent_overview(serialized)
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Analysis", 
            "🤖 RLAIF", 
            "🔄 LangGraph", 
            "📚 Patents"
        ])
        
        with tab1:
            # Render synthesis
            render_synthesis(serialized)
        
        with tab2:
            # Render RLAIF panel
            render_rlaif_panel(serialized)
            # Render Q-Learning policy stats
            render_policy_panel()
        
        with tab3:
            # Render LangGraph visualization
            render_langgraph_panel(serialized)
        
        with tab4:
            # Render retrieved patents
            retrieval_rows = extract_retrieval_rows(serialized)
            render_retrieval_section(retrieval_rows)
        
        # Render debug details
        with st.expander("🔧 Debug Details", expanded=False):
            render_agent_logs(serialized)


if __name__ == "__main__":
    main()
