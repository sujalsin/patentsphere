"""Chainlit frontend for PatentSphere with agent visualization and side panel citations."""
import chainlit as cl
import uuid
import re
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from graph.graph import workflow, AgentState


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


@cl.on_chat_start
async def start():
    """Welcome message when chat starts."""
    await cl.Message(
        content="⚖️ **PatentSphere Ready.** Ask about legal risks or prior art."
    ).send()


@cl.on_message
async def main(message: cl.Message):
    """Handle user messages and run the workflow with visualization."""
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
        event_data = event.get("data", {})
        
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
            output = event_data.get("output", {})
            
            if event_name == "router" and "router" in status_messages:
                intent = output.get("intent", "UNKNOWN")
                status_messages["router"].content = f"🚦 **Router:** Intent detected → **{intent}**"
                await status_messages["router"].update()
            
            elif event_name == "extractor" and "extractor" in status_messages:
                keywords = output.get("keywords", [])
                keywords_str = ", ".join(keywords[:3]) if keywords else "N/A"
                status_messages["extractor"].content = f"🕷️ **Extractor:** Keywords → `{keywords_str}`"
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

