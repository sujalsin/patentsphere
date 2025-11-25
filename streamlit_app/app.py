from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional

import streamlit as st

from app.agents.base import AgentResult
from app.orchestrator import Orchestrator


st.set_page_config(page_title="PatentSphere RAG Console", layout="wide")


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


def extract_retrieval_rows(
    serialized: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
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


def render_agent_overview(serialized: Dict[str, Dict[str, Any]]) -> None:
    st.subheader("Agent Status")
    agent_items = sorted(serialized.items())
    columns = st.columns(2) if len(agent_items) > 1 else [st.container()]

    for idx, (name, payload) in enumerate(agent_items):
        container = columns[idx % len(columns)]
        with container:
            icon = "✅" if payload.get("success") else "⚠️"
            latency = payload.get("data", {}).get("latency_ms")
            latency_text = f"{latency:.0f} ms" if isinstance(latency, (int, float)) else "n/a"
            st.markdown(f"{icon} **{name}** · {latency_text}")
            if payload.get("error"):
                st.error(payload["error"])


def render_retrieval_section(rows: List[Dict[str, Any]]) -> None:
    st.subheader("Retrieved Chunks")
    if not rows:
        st.info("No retrieval results returned for this query.")
        return

    top_rows = rows[:20]
    for idx, row in enumerate(top_rows, start=1):
        patent_id = row.get("patent_id") or row.get("chunk_id") or f"chunk-{idx}"
        score = row.get("score")
        subtitle = f"{idx}. {patent_id}"
        if isinstance(score, (int, float)):
            subtitle += f" · score {score:.3f}"
        with st.expander(subtitle, expanded=idx == 1):
            preview = row.get("chunk_text") or row.get("text") or "(no preview available)"
            st.write(preview)
            meta = {k: v for k, v in row.items() if k not in {"chunk_text", "text"}}
            if meta:
                st.caption(meta)


def render_synthesis(serialized: Dict[str, Dict[str, Any]]) -> None:
    synthesis = serialized.get("synthesis")
    if not synthesis:
        return

    data = synthesis.get("data") or {}
    summary = data.get("executive_summary")
    action_items = data.get("action_items") or []
    citations = data.get("citations") or []

    st.subheader("Synthesis Summary")
    if summary:
        st.markdown(f"> {summary}")
    else:
        st.info("Synthesis agent did not return a summary.")

    if action_items:
        st.markdown("**Action Items**")
        for item in action_items:
            priority = item.get("priority", "n/a").upper()
            recommendation = item.get("recommendation", "")
            rationale = item.get("rationale", "")
            st.write(f"- `{priority}` {recommendation}\n  - {rationale}")

    if citations:
        st.markdown("**Citations**")
        for citation in citations:
            st.write(f"- {citation.get('patent_id', 'unknown')}: {citation.get('reason', '')}")


def render_agent_logs(serialized: Dict[str, Dict[str, Any]]) -> None:
    st.subheader("Agent Logs & Payloads")
    has_logs = False
    for name, payload in serialized.items():
        data = payload.get("data") or {}
        logs = data.get("logs") or data.get("notes")
        expanded = name == "synthesis"
        if logs:
            has_logs = True
            with st.expander(f"{name} notes", expanded=expanded):
                if isinstance(logs, list):
                    for entry in logs:
                        st.write(f"- {entry}")
                else:
                    st.write(logs)
        with st.expander(f"{name} raw data", expanded=False):
            st.json(data or {})
            if payload.get("error"):
                st.error(payload["error"])
    if not has_logs:
        st.info("Agents did not emit explicit log entries for this run.")


def render_history(history: List[Dict[str, Any]]) -> None:
    if not history:
        return
    st.sidebar.subheader("Recent Queries")
    for entry in history[:5]:
        timestamp = entry.get("timestamp")
        query = entry.get("query", "")[:80]
        st.sidebar.caption(f"{timestamp}: {query}")


def main() -> None:
    st.title("PatentSphere Control Room")
    st.caption("Run multi-agent RAG pipelines locally and inspect every hop.")

    if "history" not in st.session_state:
        st.session_state["history"] = []

    query = st.text_area(
        "Patent query or research prompt",
        height=140,
        placeholder="e.g., How do recent G06F filings handle graph neural network accelerators?",
    )

    col_run, col_clear = st.columns([3, 1])
    run_clicked = col_run.button("Run analysis", type="primary", use_container_width=True)
    clear_clicked = col_clear.button("Clear results", use_container_width=True)

    if clear_clicked:
        st.session_state["history"] = []
        st.experimental_rerun()

    if run_clicked:
        if not query.strip():
            st.warning("Enter a query before running the orchestrator.")
        else:
            with st.spinner("Contacting agents..."):
                try:
                    results = run_orchestrator(query.strip())
                    serialized = serialize_results(results)
                    st.session_state["history"].insert(
                        0,
                        {
                            "query": query.strip(),
                            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                            "results": serialized,
                        },
                    )
                except Exception as exc:
                    st.error(f"Failed to execute orchestrator: {exc}")

    history = st.session_state.get("history", [])
    render_history(history)

    if not history:
        st.info("No runs yet. Submit your first query to see agent telemetry.")
        return

    latest = history[0]
    st.markdown(
        f"### Latest run · {latest.get('timestamp')} · \"{latest.get('query', '')}\""
    )

    serialized = latest.get("results", {})
    render_agent_overview(serialized)
    render_synthesis(serialized)
    retrieval_rows = extract_retrieval_rows(serialized)
    render_retrieval_section(retrieval_rows)
    render_agent_logs(serialized)


if __name__ == "__main__":
    main()


