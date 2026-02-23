"""Chainlit frontend for PatentSphere with agent visualization and side panel citations.

Modes
-----
• Analysis Mode (default) – original multi-agent Q&A over the patent database.
  Triggered by any normal chat message.

• Full Draft Mode – 5-step Disclosure-to-Strategic-Patent-Draft wizard.
  Triggered by typing /draft  OR uploading a file (PDF/TXT/DOCX).
  Steps:
    1. Upload & Parse raw disclosure
    2. Review Decomposed Sections
    3. Review Strategic Claims
    4. View Full Specification
    5. Human-Wall Checklist & Export (.docx / .pdf)
"""
import chainlit as cl
import uuid
import re
import os
import tempfile
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from graph.graph import workflow, AgentState
from config import settings, get_pipeline_llm

# Lazy import of pipeline to avoid startup errors when deps are missing
try:
    from pipeline.orchestrator import run_pipeline_step_by_step
    from pipeline.exporter import extract_text_from_file, export_to_docx, export_to_pdf
    from pipeline.models import PipelineState
    PIPELINE_AVAILABLE = True
except ImportError as _pipeline_import_err:
    PIPELINE_AVAILABLE = False
    print(f"[app] Pipeline not available: {_pipeline_import_err}")


def generate_patent_url(patent_id: str) -> str:
    """Generate Google Patents URL from patent ID.
    
    Google Patents URLs require the patent ID without dashes, underscores, or slashes.
    Format: https://patents.google.com/patent/{CLEANED_ID}
    
    For US patents, the format is typically: US + Year + Number + Kind Code
    Examples:
    - US-2022091384-A1 -> https://patents.google.com/patent/US2022091384A1
    - US_2022091384_A1 -> https://patents.google.com/patent/US2022091384A1
    - US2015042245A1 -> https://patents.google.com/patent/US2015042245A1
    - EP-2891581-A1 -> https://patents.google.com/patent/EP2891581A1
    
    Note: Some patents may not exist in Google Patents, which will result in 404 errors.
    This is expected for certain patent types or older patents not indexed by Google.
    """
    if not patent_id:
        return ""
    
    # Remove all dashes, underscores, slashes, and spaces
    clean_id = patent_id.replace("-", "").replace("_", "").replace("/", "").replace(" ", "").strip()
    
    # Ensure uppercase for consistency
    clean_id = clean_id.upper()
    
    # Validate basic format (should start with country code)
    if len(clean_id) < 3:
        return ""
    
    return f"https://patents.google.com/patent/{clean_id}"


# ──────────────────────────────────────────────────────────────────────────────
# Chat lifecycle
# ──────────────────────────────────────────────────────────────────────────────

@cl.on_chat_start
async def start():
    """Welcome message when chat starts."""
    demo_indicator = ""
    if settings.groq_api_key:
        demo_indicator = " *(demo mode: Groq)*"
    elif settings.demo_mode:
        demo_indicator = " *(demo mode: local CPU)*"

    await cl.Message(
        content=(
            f"⚖️ **PatentSphere Ready.**{demo_indicator}\n\n"
            "**Analysis Mode:** Ask about prior art, litigation risk, or specific patents.\n\n"
            "**Full Draft Mode:** Type `/draft` or upload a PDF/TXT/DOCX file to turn "
            "a raw inventor disclosure into a complete, attorney-ready patent draft."
        )
    ).send()

    # Store pipeline state in user session
    cl.user_session.set("pipeline_state", None)
    cl.user_session.set("draft_mode", False)



# ──────────────────────────────────────────────────────────────────────────────
# Message router: /draft  vs. normal analysis
# ──────────────────────────────────────────────────────────────────────────────

@cl.on_message
async def main(message: cl.Message):
    """Route messages: file upload / /draft → Full Draft Mode; everything else → original analysis."""
    content = message.content.strip()

    # ── File upload (Chainlit 2.x: files arrive as message.elements) ──────
    if PIPELINE_AVAILABLE and message.elements:
        uploaded_files = [
            el for el in message.elements
            if hasattr(el, "path") and el.path  # AskFileMessage / uploaded files have a path
        ]
        if not uploaded_files:
            # Also handle cl.File elements by name
            uploaded_files = [
                el for el in message.elements
                if hasattr(el, "name") and el.name and hasattr(el, "content")
            ]

        if uploaded_files:
            el = uploaded_files[0]
            name = getattr(el, "name", "file")
            # Get file path or write content to temp file
            file_path = getattr(el, "path", None)
            if not file_path:
                # Write content to temp file
                suffix = Path(name).suffix or ".txt"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    raw_bytes = getattr(el, "content", b"")
                    if isinstance(raw_bytes, bytes):
                        tmp.write(raw_bytes)
                    else:
                        tmp.write(raw_bytes.encode("utf-8", errors="ignore"))
                    file_path = tmp.name
            cleanup = not bool(getattr(el, "path", None))  # only cleanup temp files we created

            await cl.Message(content=f"📄 Received **{name}** — starting **Full Draft Mode**...").send()
            try:
                raw_text = extract_text_from_file(file_path)
            finally:
                if cleanup:
                    try:
                        os.unlink(file_path)
                    except Exception:
                        pass

            if not raw_text.strip() or raw_text.startswith("[PDF text"):
                await cl.Message(
                    content="⚠️ Could not extract text. Please paste your disclosure text and type `/draft` first."
                ).send()
                return

            await _run_draft_pipeline(raw_text)
            return

    # ── Human-Wall approval action ─────────────────────────────────────────
    if content.lower() in ("approve", "approve export", "✅ approve"):
        state: Optional[PipelineState] = cl.user_session.get("pipeline_state")
        if state and state.checklist and not state.checklist.export_allowed:
            await _handle_human_wall_approval(state)
            return

    # ── Full Draft Mode trigger ───────────────────────────────────────────
    if PIPELINE_AVAILABLE and (content.lower().startswith("/draft") or cl.user_session.get("draft_mode")):
        remaining = content[6:].strip() if content.lower().startswith("/draft") else content
        if remaining:
            cl.user_session.set("draft_mode", False)
            await _run_draft_pipeline(remaining)
        else:
            # No text after /draft – ask user to paste or upload
            cl.user_session.set("draft_mode", True)
            await cl.Message(
                content=(
                    "📝 **Full Draft Mode activated.**\n\n"
                    "Please paste your raw inventor disclosure below, or upload a **.pdf / .txt / .docx** file.\n\n"
                    "*Your disclosure can be messy notes, email text, slide bullet points, or a transcript — "
                    "the pipeline will clean it up.*"
                )
            ).send()
        return

    # ── Original analysis mode (untouched) ───────────────────────────────
    await _original_analysis(message)


# ──────────────────────────────────────────────────────────────────────────────
# Full Draft Mode – 5-step wizard
# ──────────────────────────────────────────────────────────────────────────────

async def _run_draft_pipeline(raw_text: str) -> None:
    """Execute the 5-step pipeline and stream each step to the Chainlit UI."""
    if not PIPELINE_AVAILABLE:
        await cl.Message(content="⚠️ Pipeline unavailable. Install requirements: `pip install -r requirements.txt`").send()
        return

    # Determine LLM (demo-mode toggle)
    llm = get_pipeline_llm(role="synthesizer")
    score_llm = get_pipeline_llm(role="critic")
    tenant_id = settings.default_tenant_id  # ENTERPRISE BRIDGE

    # Step progress tracker
    step_msgs: Dict[str, cl.Message] = {}

    async def _step_header(step_num: int, label: str, key: str) -> cl.Message:
        m = cl.Message(content=f"**Step {step_num}/5 — {label}** ⏳")
        await m.send()
        step_msgs[key] = m
        return m

    async def _step_done(key: str, summary: str) -> None:
        if key in step_msgs:
            step_msgs[key].content = f"✅ {summary}"
            await step_msgs[key].update()

    # ── Step 1: Decompose ────────────────────────────────────────────────
    await _step_header(1, "Decomposing Disclosure into Sections", "decompose")

    last_state: Optional[PipelineState] = None
    step_num = 0
    step_names = ["decompose", "analyze", "draft_claims", "generate_spec", "review"]
    step_labels = [
        "Decomposing Disclosure",
        "Parallel Prior-Art Analysis",
        "Drafting Strategic Claims",
        "Generating Full Specification",
        "Human-Wall Review",
    ]

    gen = run_pipeline_step_by_step(
        raw_input=raw_text,
        llm=llm,
        workflow=workflow,
        score_llm=score_llm,
        tenant_id=tenant_id,
    )

    async for state in gen:
        last_state = state

        # Check for failure
        if state.failed_at_step:
            failed_key = step_names[step_num] if step_num < len(step_names) else "unknown"
            if failed_key in step_msgs:
                step_msgs[failed_key].content = f"❌ Step {step_num + 1} failed: {state.failed_at_step}"
                await step_msgs[failed_key].update()
            await cl.Message(
                content=(
                    f"⚠️ **Pipeline stopped at step: `{state.failed_at_step}`.**\n\n"
                    f"Check the audit log below for details.\n\n"
                    f"```\n"
                    + "\n".join(
                        f"[{e.step}] {e.status}: {e.detail or ''}"
                        for e in state.audit_log[-5:]
                    )
                    + "\n```"
                )
            ).send()
            cl.user_session.set("pipeline_state", state)
            return

        # Step complete – update UI
        sn = step_names[step_num] if step_num < len(step_names) else ""
        sl = step_labels[step_num] if step_num < len(step_labels) else ""

        if step_num == 0 and state.decomposition:  # Decompose done
            await _step_done("decompose", f"**{sl}** — {len(state.decomposition.sections)} sections extracted")
            await _show_decomposition(state)
            if step_num + 1 < len(step_names):
                await _step_header(step_num + 2, step_labels[step_num + 1], step_names[step_num + 1])

        elif step_num == 1 and state.analyses:  # Analyze done
            await _step_done(sn, f"**{sl}** — {len(state.analyses)} sections analyzed")
            await _show_analysis_summary(state)
            if step_num + 1 < len(step_names):
                await _step_header(step_num + 2, step_labels[step_num + 1], step_names[step_num + 1])

        elif step_num == 2 and state.claim_set:  # Claims done
            await _step_done(sn, f"**{sl}** — {len(state.claim_set.claims)} claims drafted")
            await _show_claims(state)
            if step_num + 1 < len(step_names):
                await _step_header(step_num + 2, step_labels[step_num + 1], step_names[step_num + 1])

        elif step_num == 3 and state.specification:  # Spec done
            await _step_done(sn, f"**{sl}** — ~{state.specification.word_count()} words")
            await _show_specification(state)
            if step_num + 1 < len(step_names):
                await _step_header(step_num + 2, step_labels[step_num + 1], step_names[step_num + 1])

        elif step_num == 4 and state.checklist:  # Review done
            await _step_done(sn, f"**{sl}** — confidence: {state.checklist.overall_confidence:.0%}, flags: {len(state.checklist.flags)}")
            await _show_checklist(state)

        step_num += 1

    if last_state:
        cl.user_session.set("pipeline_state", last_state)
        cl.user_session.set("draft_mode", False)


# ── Step display helpers ───────────────────────────────────────────────────────

async def _show_decomposition(state: PipelineState) -> None:
    """Render decomposed sections as formatted markdown."""
    disclosure = state.decomposition
    lines = ["### 📋 Decomposed Disclosure Sections\n"]
    lines.append("| Section | Confidence | Words | Excerpt |")
    lines.append("|---------|-----------|-------|---------|")
    for sec in disclosure.sections:
        excerpt = sec.provenance.excerpt[:50].replace("|", "I").replace("\n", " ")
        lines.append(
            f"| **{sec.section_name.value}** "
            f"| {sec.confidence:.0%} "
            f"| {sec.word_count} "
            f"| *{excerpt}...* |"
        )
    await cl.Message(content="\n".join(lines)).send()

    # Show full section text as expandable elements
    elements = []
    for sec in disclosure.sections:
        elements.append(cl.Text(
            name=sec.section_name.value,
            content=(
                f"**Section:** {sec.section_name.value}\n"
                f"**Confidence:** {sec.confidence:.0%}\n"
                f"**Provenance:** chars {sec.provenance.char_start}–{sec.provenance.char_end}\n"
                f"---\n{sec.text}"
            ),
            display="side",
        ))

    sections_msg = cl.Message(
        content="*Click any section name to read the cleaned text →* "
                + " | ".join(f"[[{s.section_name.value}]]" for s in disclosure.sections)
    )
    sections_msg.elements = elements
    await sections_msg.send()


async def _show_analysis_summary(state: PipelineState) -> None:
    """Render analysis scores as a risk table."""
    lines = ["### 🔬 Prior-Art & Risk Analysis\n"]
    lines.append("| Section | Novelty ↑ | Obviousness Risk ↓ | Litigation Risk ↓ | Prior Art Hits |")
    lines.append("|---------|-----------|-------------------|-----------------|---------------|")
    for a in state.analyses:
        n_bar = "🟢" if a.novelty_score > 0.6 else ("🟡" if a.novelty_score > 0.4 else "🔴")
        o_bar = "🔴" if a.obviousness_risk > 0.6 else ("🟡" if a.obviousness_risk > 0.4 else "🟢")
        l_bar = "🔴" if a.litigation_risk > 0.6 else ("🟡" if a.litigation_risk > 0.4 else "🟢")
        lines.append(
            f"| {a.section_name.value} "
            f"| {n_bar} {a.novelty_score:.2f} "
            f"| {o_bar} {a.obviousness_risk:.2f} "
            f"| {l_bar} {a.litigation_risk:.2f} "
            f"| {len(a.prior_art_hits)} |"
        )
    await cl.Message(content="\n".join(lines)).send()


async def _show_claims(state: PipelineState) -> None:
    """Render the strategic claims with reasoning."""
    lines = ["### ⚖️ Strategic Patent Claims\n"]
    for claim in state.claim_set.claims:
        dep_str = f" *(depends on claim {claim.depends_on})*" if claim.depends_on else " *(independent)*"
        breadth_bar = "█" * int(claim.breadth_score * 10) + "░" * (10 - int(claim.breadth_score * 10))
        lines.append(f"**Claim {claim.claim_number}**{dep_str} — Breadth: `{breadth_bar}` {claim.breadth_score:.2f}")
        lines.append(f"> {claim.text}\n")
        lines.append(f"💡 *Strategy:* {claim.strategy_reasoning}\n")
        lines.append("---")
    await cl.Message(content="\n".join(lines)).send()


async def _show_specification(state: PipelineState) -> None:
    """Render the full specification in structured markdown."""
    spec = state.specification
    content = f"""### 📄 Full Patent Specification Draft

**Title:** {spec.title}

**Word Count:** ~{spec.word_count()} words · **Generated:** {spec.generation_timestamp.strftime('%Y-%m-%d %H:%M UTC')}

---

#### Abstract
{spec.abstract}

#### Background
{spec.background}

#### Summary of the Invention
{spec.summary_of_invention}

#### Detailed Description
{spec.detailed_description}
"""

    if getattr(spec, "prior_art_navigation", None):
        content += f"\n#### Prior Art Navigation Strategy\n{spec.prior_art_navigation}\n"
    
    if getattr(spec, "competitive_positioning_map", None):
        content += f"\n#### Competitive Positioning Map\n{spec.competitive_positioning_map}\n"

    content += "\n#### Claims\n"
    for claim in spec.claims:
        dep = f" (depends on claim {claim.depends_on})" if claim.depends_on else ""
        content += f"\n**{claim.claim_number}.{dep}** {claim.text}\n"

    spec_element = cl.Text(
        name="full_specification",
        content=content,
        display="side",
    )
    msg = cl.Message(
        content="📄 Full specification generated! Click to read → [[full_specification]]\n\n"
                f"*Title: **{spec.title}***"
    )
    msg.elements = [spec_element]
    await msg.send()


async def _show_checklist(state: PipelineState) -> None:
    """Render Human-Wall Checklist and export controls."""
    checklist = state.checklist
    conf_pct = f"{checklist.overall_confidence:.0%}"
    conf_icon = "🟢" if checklist.overall_confidence > 0.7 else ("🟡" if checklist.overall_confidence > 0.5 else "🔴")

    lines = [
        f"### 🧱 Human-Wall Checklist\n",
        f"{conf_icon} **Overall Confidence:** {conf_pct}  |  "
        f"🚩 **Flags:** HIGH={checklist.high_severity_count} "
        f"MEDIUM={checklist.medium_severity_count} "
        f"LOW={checklist.low_severity_count}",
        f"\n*{checklist.reviewer_notes}*\n",
    ]

    # Per-section scores table
    lines.append("#### Section Confidence Scores")
    lines.append("| Section | Confidence |")
    lines.append("|---------|-----------|")
    for s in checklist.section_scores:
        icon = "🟢" if s.confidence > 0.7 else ("🟡" if s.confidence > 0.5 else "🔴")
        lines.append(f"| {s.section} | {icon} {s.confidence:.0%} |")

    # Flags
    if checklist.flags:
        lines.append("\n#### ⚠️ Flags for Attorney Attention")
        for flag in checklist.flags:
            sev_icon = {"high": "🔴", "medium": "🟡", "low": "🔵"}.get(flag.severity, "⚪")
            lines.append(
                f"\n{sev_icon} **[{flag.severity.upper()}] {flag.section}** — {flag.issue}\n"
                f"   *Suggestion:* {flag.suggestion}"
            )

    lines.append("\n---")
    lines.append(
        "\n✅ **Ready to approve?** Type `approve` to unlock export.\n"
        "🔒 *Export is locked until you approve this draft.*"
    )

    await cl.Message(content="\n".join(lines)).send()


async def _handle_human_wall_approval(state: PipelineState) -> None:
    """Handle user approval at the Human-Wall checkpoint and offer exports."""
    import tempfile, os

    approved_checklist = state.checklist.approve()
    state = state.model_copy(update={"checklist": approved_checklist})
    cl.user_session.set("pipeline_state", state)

    await cl.Message(
        content="✅ **Human-Wall Approved!** Generating export files..."
    ).send()

    spec = state.specification
    session_id = state.session_id
    export_dir = tempfile.mkdtemp(prefix="patentsphere_export_")

    elements = []
    messages = []

    # DOCX export
    try:
        docx_path = os.path.join(export_dir, f"patent_draft_{session_id[:8]}.docx")
        export_to_docx(spec, approved_checklist, docx_path, session_id=session_id)
        elements.append(cl.File(path=docx_path, name=f"patent_draft_{session_id[:8]}.docx", display="inline"))
        messages.append(f"📄 **DOCX:** `patent_draft_{session_id[:8]}.docx`")
    except Exception as e:
        messages.append(f"⚠️ DOCX export failed: {e}")

    # PDF export
    try:
        pdf_path = os.path.join(export_dir, f"patent_draft_{session_id[:8]}.pdf")
        export_to_pdf(spec, approved_checklist, pdf_path, session_id=session_id)
        elements.append(cl.File(path=pdf_path, name=f"patent_draft_{session_id[:8]}.pdf", display="inline"))
        messages.append(f"📋 **PDF:** `patent_draft_{session_id[:8]}.pdf` (with ✅ Human-Wall Approved stamp)")
    except Exception as e:
        messages.append(f"⚠️ PDF export failed: {e}")

    export_msg = cl.Message(
        content=(
            "### 📦 Export Complete\n\n"
            + "\n".join(messages)
            + "\n\n*Every exported file includes a Provenance Appendix tracing each "
              "section back to the original inventor disclosure.*"
        )
    )
    export_msg.elements = elements
    await export_msg.send()


# ──────────────────────────────────────────────────────────────────────────────
# Original analysis mode (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

async def _original_analysis(message: cl.Message) -> None:
    """Original PatentSphere analysis flow — untouched from baseline."""
    # (Renamed from main() to _original_analysis() for routing; logic is identical)
    # Initialize state with user query
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    initial_state: AgentState = {
        "query": message.content,
        "intent": None,
        "keywords": None,
        "date_range": None,
        "documents": [],
        "litigation_context": [],
        "draft": None,
        "critique": None,
        "retry_count": 0,
        "final_response": None,
    }
    
    # Create message for streaming response
    msg = cl.Message(content="")
    await msg.send()
    
    # Track agent status
    status_messages: Dict[str, cl.Message] = {}
    documents: List[Dict[str, Any]] = []
    litigation_context: List[Dict[str, Any]] = []
    accumulated_text = ""
    final_draft = ""
    
    # Stream events from workflow
    async for event in workflow.astream_events(
        initial_state,
        config=config,
        version="v1",
    ):
        event_type = event.get("event")
        event_name = event.get("name", "")
        event_data = event.get("data") or {}
        
        # Handle node start events - show status
        if event_type == "on_chain_start":
            if event_name == "router":
                status_msg = cl.Message(content="🚦 **Router:** Analyzing intent...")
                await status_msg.send()
                status_messages["router"] = status_msg
            
            elif event_name == "extractor":
                status_msg = cl.Message(content="🕷️ **Extractor:** Expanding query...")
                await status_msg.send()
                status_messages["extractor"] = status_msg
            
            elif event_name == "critic":
                status_msg = cl.Message(content="🔍 **Critic:** Verifying citations...")
                await status_msg.send()
                status_messages["critic"] = status_msg
        
        # Handle node end events - update status with results
        elif event_type == "on_chain_end":
            output = event_data.get("output") or {}
            
            if event_name == "router" and "router" in status_messages:
                intent = output.get("intent", "UNKNOWN")
                status_messages["router"].content = f"🚦 **Router:** Intent detected → **{intent}**"
                await status_messages["router"].update()
            
            elif event_name == "extractor" and "extractor" in status_messages:
                tech = output.get("technical_keywords") or output.get("keywords") or []
                legal = output.get("legal_entities") or []
                tech_str = ", ".join(tech[:3]) if tech else "N/A"
                legal_str = ", ".join(legal[:2]) if legal else "None"
                status_messages["extractor"].content = f"🕷️ **Extractor:** Tech → `{tech_str}` | Legal → `{legal_str}`"
                await status_messages["extractor"].update()
            
            elif event_name == "retrieval":
                # Capture documents from retrieval node
                documents = output.get("documents", [])
                litigation_context = output.get("litigation_context", [])
                doc_count = len(documents)
                await cl.Message(
                    content=f"📚 **Retriever:** Found **{doc_count}** patent(s) from database"
                ).send()
            
            elif event_name == "critic" and "critic" in status_messages:
                critique = output.get("critique", {})
                status = critique.get("status", "UNKNOWN")
                feedback = critique.get("feedback", "")
                
                if status == "FAIL":
                    retry_count = output.get("retry_count", 0)
                    status_messages["critic"].content = f"⚠️ **Critic:** REJECTED (Attempt {retry_count + 1}/3)\n\n{feedback[:200]}..."
                else:
                    status_messages["critic"].content = "✅ **Critic:** APPROVED - No hallucinations detected"
                await status_messages["critic"].update()
            
            elif event_name == "synthesizer":
                # Capture final draft from synthesizer
                draft = output.get("draft", "")
                if draft:
                    final_draft = draft
                    if not accumulated_text:
                        # If we didn't stream, set the full content
                        msg.content = final_draft
                        await msg.update()
            
            elif event_name == "finalize":
                # Get final response
                final_response = output.get("final_response", "")
                if final_response:
                    final_draft = final_response
        
        # Handle chat model stream events - stream tokens from synthesizer
        elif event_type == "on_chat_model_stream":
            chunk = event_data.get("chunk", {})
            if hasattr(chunk, "content") and chunk.content:
                accumulated_text += chunk.content
                await msg.stream_token(chunk.content)
        
        # Handle chat model end - finalize streaming
        elif event_type == "on_chat_model_end":
            # Ensure all text is in the message
            if accumulated_text:
                await msg.update()
    
    # Use final draft if available, otherwise use accumulated text
    if final_draft:
        msg.content = final_draft
        await msg.update()
    elif accumulated_text:
        msg.content = accumulated_text
        await msg.update()
    
    # If we still don't have documents or final draft, get them from final state
    # This is a fallback in case events didn't capture everything
    if not documents or (not final_draft and not accumulated_text):
        try:
            # Get the final state from the checkpointer (won't re-execute)
            final_state = None
            async for state in workflow.astream(initial_state, config=config, stream_mode="values"):
                final_state = state
            
            if final_state:
                if not documents:
                    documents = final_state.get("documents", [])
                if not litigation_context:
                    litigation_context = final_state.get("litigation_context", [])
                if not final_draft and not accumulated_text:
                    final_draft = final_state.get("final_response") or final_state.get("draft", "")
                    if final_draft:
                        msg.content = final_draft
                        await msg.update()
        except Exception as e:
            # If getting final state fails, continue with what we have
            print(f"Warning: Could not get final state: {e}")
    
    # Create side panel elements for each patent
    # Extract all cited patent IDs from the final draft for matching
    # Support both formats: [[PatentID]](URL) and (Patent: PatentID, ...)
    cited_patent_ids = set()
    if final_draft or accumulated_text:
        text_to_parse = final_draft or accumulated_text
        # Extract citations in old format: [[PatentID]](URL)
        citation_pattern_old = r'\[\[([^\]]+)\]\]'
        old_citations = re.findall(citation_pattern_old, text_to_parse)
        cited_patent_ids.update(old_citations)
        
        # Extract citations in new format: (Patent: PatentID, ...)
        citation_pattern_new = r'\(Patent:\s*([^,)]+)(?:,\s*[^)]+)*\)'
        new_citations = re.findall(citation_pattern_new, text_to_parse)
        for match in new_citations:
            patent_id = match[0].strip() if isinstance(match, tuple) else match.strip()
            # Keep both underscore and dash formats for matching
            cited_patent_ids.add(patent_id)  # Original format (with underscores)
            cited_patent_ids.add(patent_id.replace("_", "-"))  # Dash format
    
    elements: List[cl.Text] = []
    for doc in documents:
        patent_id = doc.get("patent_id", "")
        if not patent_id:
            continue
        
        # Format patent content for side panel
        title = doc.get("title", "No title available")
        abstract = doc.get("abstract", "No abstract available")
        # Use stored URL if available and valid, otherwise generate one
        stored_url = doc.get("url", "")
        if stored_url and stored_url.startswith("https://patents.google.com"):
            url = stored_url
        else:
            url = generate_patent_url(patent_id)
        publication_date = doc.get("publication_date", "N/A")
        cpc_codes = doc.get("cpc_codes", [])
        
        # Format CPC codes
        cpc_str = ""
        if cpc_codes:
            if isinstance(cpc_codes, list):
                cpc_str = ", ".join([str(c.get("code", c) if isinstance(c, dict) else c) for c in cpc_codes[:5]])
            else:
                cpc_str = str(cpc_codes)
        
        # Build content text
        content_parts = [
            f"# {patent_id}",
            f"",
            f"**Title:** {title}",
            f"",
            f"**Abstract:**",
            abstract[:1000] if len(abstract) > 1000 else abstract,
            f"",
            f"**Publication Date:** {publication_date}",
        ]
        
        if cpc_str:
            content_parts.append(f"**CPC Codes:** {cpc_str}")
        
        content = "\n".join(content_parts)
        
        # Create Text element with patent_id as name (in dash format, which is standard)
        # We'll convert citations in the message to match this format for auto-linking
        element = cl.Text(
            name=patent_id,  # Use original format (usually has dashes like US-2022091384-A1)
            content=content,
            display="side",
        )
        elements.append(element)
    
    # Create clickable elements for litigation cases
    for case in litigation_context:
        case_number = case.get("case_number", "")
        case_name = case.get("case_name", "Unknown")
        case_status = case.get("case_status", "Unknown")
        court_name = case.get("court_name", "N/A")
        plaintiff_name = case.get("plaintiff_name", "N/A")
        defendant_name = case.get("defendant_name", "N/A")
        filing_date = case.get("filing_date", "N/A")
        outcome = case.get("outcome", "N/A")
        related_patent_id = case.get("patent_id", "N/A")
        
        # Use case_number as identifier, fallback to case_name if no case_number
        case_identifier = case_number if case_number else case_name
        
        if not case_identifier or case_identifier == "Unknown":
            continue
        
        # Build litigation case content
        case_content_parts = [
            f"# {case_identifier}",
            f"",
            f"**Case Name:** {case_name}",
            f"",
            f"**Court:** {court_name}",
            f"**Status:** {case_status}",
            f"**Filing Date:** {filing_date}",
        ]
        
        if plaintiff_name and plaintiff_name != "N/A":
            case_content_parts.append(f"**Plaintiff:** {plaintiff_name}")
        if defendant_name and defendant_name != "N/A":
            case_content_parts.append(f"**Defendant:** {defendant_name}")
        if related_patent_id and related_patent_id != "N/A":
            case_content_parts.append(f"**Related Patent:** {related_patent_id}")
        if outcome and outcome != "N/A":
            case_content_parts.append(f"**Outcome:** {outcome}")
        
        case_content = "\n".join(case_content_parts)
        
        # Create Text element for litigation case
        case_element = cl.Text(
            name=case_identifier,  # Use case_number or case_name as identifier
            content=case_content,
            display="side",
        )
        elements.append(case_element)
    
    # Post-process message content to make citations clickable
    # Convert (Patent: ID, ...) format to include clickable patent ID that matches element names
    final_content = final_draft or accumulated_text or msg.content or ""
    if final_content and elements:
        # Create mappings: element names and their normalized versions for matching
        element_names = {elem.name for elem in elements}
        element_normalized = {}  # normalized_id -> element_name
        for elem_name in element_names:
            # Normalize: remove dashes/underscores, uppercase
            normalized = elem_name.upper().replace("-", "").replace("_", "")
            element_normalized[normalized] = elem_name
        
        # Replace (Patent: US_123, ...) with (Patent: [[US-123]], ...) for clickability
        def make_citation_clickable(match):
            full_match = match.group(0)
            patent_id_in_citation = match.group(1).strip()
            
            # Normalize the citation ID for matching
            citation_normalized = patent_id_in_citation.upper().replace("-", "").replace("_", "")
            
            # Find matching element
            matching_element_name = None
            if patent_id_in_citation in element_names:
                matching_element_name = patent_id_in_citation
            elif patent_id_in_citation.replace("_", "-") in element_names:
                matching_element_name = patent_id_in_citation.replace("_", "-")
            elif patent_id_in_citation.replace("-", "_") in element_names:
                matching_element_name = patent_id_in_citation.replace("-", "_")
            elif citation_normalized in element_normalized:
                matching_element_name = element_normalized[citation_normalized]
            else:
                # Try to find by normalized comparison
                for norm_id, elem_name in element_normalized.items():
                    if norm_id == citation_normalized:
                        matching_element_name = elem_name
                        break
            
            if matching_element_name:
                # Replace the patent ID part with clickable format [[ID]]
                clickable_id = f"[[{matching_element_name}]]"
                return full_match.replace(patent_id_in_citation, clickable_id, 1)
            else:
                # No matching element found, return original
                return full_match
        
        # Pattern to match (Patent: ID, ...) and make ID clickable
        citation_pattern = r'\(Patent:\s*([^,)]+)(?:,\s*[^)]+)*\)'
        final_content = re.sub(citation_pattern, make_citation_clickable, final_content)
        
        # Also convert standalone patent IDs to clickable format
        # Use a more comprehensive approach: find all patent ID patterns and match them to elements
        # Pattern matches patent IDs like: US_2018122245_A1, US-2018122245-A1, KR_101552074_B1, etc.
        # Format: [Country Code][_/-][Numbers/Year][_/-][Letter/Number][_/-][Letter/Number]?
        standalone_patent_pattern = r'\b([A-Z]{2,3}[_-][A-Z0-9]+(?:[_-][A-Z0-9]+)*)\b'
        
        def convert_any_patent_id(match):
            patent_id = match.group(1)
            start = match.start()
            end = match.end()
            
            # Check if already inside [[...]]
            text_before = final_content[:start]
            text_after = final_content[end:]
            last_open = text_before.rfind('[[')
            first_close = text_after.find(']]')
            
            if last_open >= 0 and first_close >= 0:
                between = final_content[last_open:end + first_close + 2]
                if patent_id in between:
                    return match.group(0)  # Already clickable
            
            # Normalize the patent ID for matching (remove dashes/underscores, uppercase)
            patent_normalized = patent_id.upper().replace("-", "").replace("_", "")
            
            # Find matching element by normalized comparison
            matching_element_name = None
            for elem_name in element_names:
                elem_normalized = elem_name.upper().replace("-", "").replace("_", "")
                if elem_normalized == patent_normalized:
                    matching_element_name = elem_name
                    break
            
            # Also try direct format matching (with dash/underscore conversion)
            if not matching_element_name:
                if patent_id in element_names:
                    matching_element_name = patent_id
                elif patent_id.replace("_", "-") in element_names:
                    matching_element_name = patent_id.replace("_", "-")
                elif patent_id.replace("-", "_") in element_names:
                    matching_element_name = patent_id.replace("-", "_")
            
            if matching_element_name:
                return f"[[{matching_element_name}]]"
            else:
                # No matching element found, return original
                return match.group(0)
        
        final_content = re.sub(standalone_patent_pattern, convert_any_patent_id, final_content)
        
        # Also convert litigation citations to clickable format
        # Pattern matches: (Case: CASE_NAME, p.XX) or (PTAB: CASE_ID, §XX)
        def make_litigation_citation_clickable(match):
            full_match = match.group(0)
            case_identifier = match.group(1).strip()
            
            # Find matching litigation element by checking all elements
            # Look for elements that are litigation cases (not patents)
            matching_case_name = None
            for elem in elements:
                elem_name = elem.name
                # Check if this element name matches the case identifier
                # Also check if the case identifier appears in the element content
                if (case_identifier == elem_name or 
                    case_identifier.upper() == elem_name.upper() or
                    case_identifier in elem.content or
                    case_identifier.upper() in elem.content.upper()):
                    # Verify this is a litigation case by checking if it's not a patent ID format
                    # Patent IDs typically match pattern: [A-Z]{2,3}[_-][A-Z0-9]+
                    is_patent_id = re.match(r'^[A-Z]{2,3}[_-][A-Z0-9]+', elem_name)
                    if not is_patent_id:
                        matching_case_name = elem_name
                        break
            
            if matching_case_name:
                # Replace the case identifier with clickable format [[CASE_ID]]
                clickable_id = f"[[{matching_case_name}]]"
                return full_match.replace(case_identifier, clickable_id, 1)
            else:
                # No matching element found, return original
                return full_match
        
        # Pattern to match (Case: CASE_NAME, ...) or (PTAB: CASE_ID, ...)
        litigation_citation_pattern = r'\((?:Case|PTAB):\s*([^,)]+)(?:,\s*[^)]+)*\)'
        final_content = re.sub(litigation_citation_pattern, make_litigation_citation_clickable, final_content)
    
    # Attach elements to the final message
    if elements:
        msg.elements = elements
        msg.content = final_content
        await msg.update()
    
    # Send final message if we haven't already
    if not msg.content:
        msg.content = final_content or "No response generated."
        await msg.update()

