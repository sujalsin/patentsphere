"""Agent nodes for the PatentSphere workflow."""
import json
import re
from typing import Dict, Any, AsyncIterator
try:
    from langchain_ollama import ChatOllama
except ImportError:
    # Fallback to deprecated import
    from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from config import settings


# Initialize LLM clients
router_llm = ChatOllama(
    model=settings.ollama_router_model,
    base_url=settings.ollama_base_url,
    temperature=0.1,
)

extractor_llm = ChatOllama(
    model=settings.ollama_extractor_model,
    base_url=settings.ollama_base_url,
    temperature=0.1,
)

synthesizer_llm = ChatOllama(
    model=settings.ollama_synthesizer_model,
    base_url=settings.ollama_base_url,
    temperature=0.3,
)

critic_llm = ChatOllama(
    model=settings.ollama_critic_model,
    base_url=settings.ollama_base_url,
    temperature=0.1,
)


async def router_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Route query to determine intent (LEGAL/TECHNICAL/BOTH)."""
    query = state.get("query", "")
    
    system_prompt = """You are a query router for a patent analysis system.
Analyze the input query and classify the intent.

Return ONLY a JSON object with this exact format:
{"intent": "LEGAL" | "TECHNICAL" | "BOTH"}

Classification rules:
- "LEGAL": Contains keywords like: court, lawsuit, infringement, risk, judge, litigation, plaintiff, defendant, case
- "TECHNICAL": Contains keywords like: prior art, method, device, system, algorithm, implementation, design, process
- "BOTH": Contains elements of both legal and technical aspects

Be precise and consistent."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Query: {query}"),
    ]
    
    response = await router_llm.ainvoke(messages)
    content = response.content.strip()
    
    # Extract JSON from response
    try:
        # Try to parse as-is
        intent_data = json.loads(content)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'\{[^}]+\}', content)
        if json_match:
            intent_data = json.loads(json_match.group())
        else:
            # Default fallback
            intent_data = {"intent": "BOTH"}
    
    return {"intent": intent_data.get("intent", "BOTH")}


async def extractor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract technical vs. legal signals for retrieval."""
    query = state.get("query", "")
    
    system_prompt = """You are a Patent Search Expert. Your goal is to convert user queries into a precise JSON search object.

### RULES:

1. **TRANSFORM, DON'T JUST EXTRACT:**
   - Input: "Patents on robotic arms"
   - BAD Output: ["Patents on robotic", "arms"]
   - GOOD Output: ["robotic arm", "manipulator", "end effector", "CPC: B25J"]
   - **Reasoning:** You must generate *synonyms* and *CPC codes*. Remove words like "Patents on", "regarding", "Show me".

2. **SEPARATE LEGAL vs TECHNICAL:**
   - `technical_keywords`: Concepts, inventions, machines, chemicals. (For Vector DB)
   - `legal_entities`: Company names (Apple, Tesla), Courts (E.D. Texas), or specific Case Names. (For SQL DB)
   - If the user asks "Has Apple sued?", "Apple" is a LEGAL entity, NOT a technical keyword.

3. **EXPANSION STRATEGY:**
   - For every technical term found, add 1-2 synonyms or related terms.
   - Example: "Drone" -> ["drone", "UAV", "quadcopter", "aerial vehicle"]

### RESPONSE FORMAT (JSON ONLY):
{
    "technical_keywords": ["string", "string"],
    "legal_entities": ["string", "string"],
    "patent_ids": ["string"],
    "date_range": null
}

### EXAMPLES (FOLLOW THESE PATTERNS):

Input: "Patents on robotic manipulators or grasp planning—key innovations."
Output:
{
    "technical_keywords": ["robotic manipulator", "grasp planning", "robotic arm", "end effector", "motion planning", "CPC: B25J"],
    "legal_entities": [],
    "patent_ids": [],
    "date_range": null
}

Input: "Has Apple sued Samsung over touchscreen patents?"
Output:
{
    "technical_keywords": ["touchscreen", "capacitive touch", "user interface", "display panel", "CPC: G06F"],
    "legal_entities": ["Apple", "Samsung"],
    "patent_ids": [],
    "date_range": null
}

Input: "Prior art for US-9876543 regarding transformer architecture"
Output:
{
    "technical_keywords": ["transformer architecture", "self-attention", "neural network", "natural language processing"],
    "legal_entities": [],
    "patent_ids": ["US9876543"],
    "date_range": null
}

Input: "Litigation involving Pfizer in 2023"
Output:
{
    "technical_keywords": ["pharmaceutical", "drug compound"],
    "legal_entities": ["Pfizer"],
    "patent_ids": [],
    "date_range": {"start": "2023-01-01", "end": "2023-12-31"}
}
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Query: {query}"),
    ]
    
    response = await extractor_llm.ainvoke(messages)
    content = response.content.strip()
    
    parsed = None
    if content:
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            pass
    if parsed is None:
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
            except json.JSONDecodeError:
                parsed = None
    if parsed is None:
        parsed = {
            "technical_keywords": [query],
            "legal_entities": [],
            "date_range": None,
        }
    
    def _clean_terms(items: list, allow_max: int = 7) -> list:
        seen = set()
        cleaned = []
        for item in items:
            if not isinstance(item, str):
                continue
            term = item.strip()
            if not term:
                continue
            low = term.lower()
            # Drop echoes of the system prompt or boilerplate
            if any(bad in low for bad in ["you are the queryextractor", "return only a json", "patent + litigation rag system", "system prompt"]):
                continue
            if "system" == low:
                continue
            # Skip very long phrases (reduce query echoing)
            if len(term.split()) > 6:
                continue
            if term in seen:
                continue
            seen.add(term)
            cleaned.append(term)
            if len(cleaned) >= allow_max:
                break
        return cleaned
    
    technical_keywords = parsed.get("technical_keywords") or []
    if not isinstance(technical_keywords, list):
        technical_keywords = []
    technical_keywords = _clean_terms(technical_keywords, allow_max=7)
    if len(technical_keywords) < 3:
        # Derive short fallbacks from the query without echoing full sentence
        fallback_terms = []
        words = [w.strip() for w in query.split() if w.strip()]
        if words:
            fallback_terms.append(" ".join(words[:3]))
        fallback_terms.append(f"{query.split()[0]} method" if words else "")
        technical_keywords.extend(_clean_terms(fallback_terms, allow_max=3))
    technical_keywords = technical_keywords[:7]
    
    legal_entities = parsed.get("legal_entities") or []
    if not isinstance(legal_entities, list):
        legal_entities = []
    legal_entities = _clean_terms(legal_entities, allow_max=7)
    
    patent_ids = parsed.get("patent_ids") or []
    if not isinstance(patent_ids, list):
        patent_ids = []
    patent_ids = _clean_terms(patent_ids, allow_max=7)
    
    return {
        "technical_keywords": technical_keywords,
        "legal_entities": legal_entities,
        "patent_ids": patent_ids,
        "date_range": parsed.get("date_range"),
        # Backward compatibility for downstream consumers that still expect "keywords"
        "keywords": technical_keywords,
    }


async def synthesizer_node(state: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
    """Synthesize retrieved data into a patent report with streaming support."""
    query = state.get("query", "")
    # Debug: Verify query is correct
    if not query or len(query) < 5:
        print(f"WARNING: Synthesizer received empty or very short query: '{query}'")
    print(f"DEBUG: Synthesizer processing query: '{query[:100]}...'")
    intent = state.get("intent", "UNKNOWN")
    keywords = state.get("technical_keywords") or state.get("keywords") or []
    documents = state.get("documents", [])
    litigation_context = state.get("litigation_context", [])
    feedback = state.get("critique", {}).get("feedback", "") if isinstance(state.get("critique"), dict) else ""
    retry_count = state.get("retry_count", 0)
    
    # CRITICAL: Check if we have any documents
    if not documents or len(documents) == 0:
        # Return a clear "no results" message instead of hallucinating
        no_results = f"""# Patent Analysis Report

## Executive Summary
No patents were found matching your query: "{query}"

## Query Analysis
- **Original Query:** {query}
- **Intent Classification:** {intent}
- **Search Variations Used:** {', '.join(keywords) if keywords else 'N/A'}
- **Documents Retrieved:** 0 patent(s)

## Key Findings
No relevant patents were retrieved from the database. This could mean:
- The search terms don't match any patents in the database
- The patents may use different terminology
- The database may need to be updated with more recent patents
- The search system may need adjustment

## Conclusion
Unable to provide analysis as no matching patents were found. Please try:
- Different search terms
- Broader query scope
- Alternative technical terminology
- Checking if the database has been populated with patent data

**Note**: This system requires patent data to be ingested into the database before it can provide analysis.
"""
        yield {"draft": no_results}
        return
    
    # Build structured patent context with sections
    # Format: [Patent: ID, Section: Type, Ref: ref_id]
    patent_chunks = []
    for doc in documents:
        patent_id = doc.get("patent_id", "")
        if not patent_id:
            continue
        
        # Get chunk information
        chunk_type = doc.get("chunk_type", "")
        chunk_text = doc.get("chunk_text", "")
        chunk_id = doc.get("chunk_id", "")
        title = doc.get("title", "")
        abstract = doc.get("abstract", "")
        
        # Determine section type and reference
        section_type = "Abstract"
        ref_id = ""
        
        if chunk_type == "title":
            section_type = "Title"
            ref_id = "title"
            if not chunk_text and title:
                chunk_text = title
        elif chunk_type == "abstract":
            section_type = "Abstract"
            ref_id = "abstract"
            if not chunk_text and abstract:
                chunk_text = abstract
        elif chunk_type and chunk_type.startswith("claim"):
            # Extract claim number from chunk_type (e.g., "claim_1" -> "Claim 1")
            claim_num = chunk_type.replace("claim_", "").replace("_", " ").strip()
            # Handle claim_1_part_2 format
            if "_part_" in claim_num:
                parts = claim_num.split("_part_")
                claim_num = parts[0]
            section_type = f"Claim {claim_num}"
            ref_id = chunk_type
        elif chunk_type and "description" in chunk_type:
            # Extract paragraph reference (e.g., "description_part_1" -> "Description ¶[0001]")
            part_num = chunk_type.replace("description_part_", "").replace("description_", "").strip()
            try:
                para_num = int(part_num) if part_num.isdigit() else part_num
                section_type = f"Description ¶[{para_num:04d}]" if isinstance(para_num, int) else f"Description {part_num}"
            except:
                section_type = f"Description {part_num}"
            ref_id = chunk_type
        else:
            # Fallback: if no chunk_type, treat as abstract or use available content
            if not chunk_type:
                # No chunk structure - use abstract if available
                if abstract:
                    section_type = "Abstract"
                    ref_id = "abstract"
                    chunk_text = abstract
                elif title:
                    section_type = "Title"
                    ref_id = "title"
                    chunk_text = title
                else:
                    # No usable content
                    continue
            else:
                # Has chunk_type but unknown format
                section_type = chunk_type.replace("_", " ").title()
                ref_id = chunk_id or chunk_type
        
        # Use chunk_text if available, otherwise fall back to abstract/title
        if not chunk_text:
            if section_type == "Abstract" and abstract:
                chunk_text = abstract
            elif section_type == "Title" and title:
                chunk_text = title
            else:
                continue  # Skip if no content
        
        # Format patent ID (replace dashes with underscores for citation format)
        patent_id_formatted = patent_id.replace("-", "_")
        
        patent_chunks.append({
            "patent_id": patent_id,
            "patent_id_formatted": patent_id_formatted,
            "section_type": section_type,
            "ref_id": ref_id,
            "text": chunk_text
        })
    
    # Build patent materials context
    patent_materials = []
    for chunk in patent_chunks:
        patent_materials.append(
            f"[Patent: {chunk['patent_id_formatted']}, Section: {chunk['section_type']}, Ref: {chunk['ref_id']}]\n"
            f"{chunk['text']}\n"
        )
    patent_context = "\n".join(patent_materials)
    
    # Debug: Print context info
    # Context built successfully with {len(context_parts)} patents
    
    # If no patent context was built, return early
    if not patent_context or len(patent_context.strip()) < 50:
        no_results = f"""# Patent Analysis Report

## Executive Summary
No usable patent data was found matching your query: "{query}"

## Query Analysis
- **Original Query:** {query}
- **Intent Classification:** {intent}
- **Search Variations Used:** {', '.join(keywords) if keywords else 'N/A'}
- **Documents Retrieved:** {len(documents)} patent(s) (but no usable content)

## Key Findings
The search retrieved {len(documents)} patent(s), but they did not contain sufficient information to generate an analysis.

## Conclusion
Unable to provide analysis as no usable patent content was found.
"""
        yield {"draft": no_results}
        return
    
    # Build litigation materials context (from PostgreSQL, not chunks)
    litigation_chunks = []
    if litigation_context:
        print(f"DEBUG: Synthesizer received {len(litigation_context)} litigation case(s)")
        for case in litigation_context:
            case_number = case.get("case_number", "")
            case_name = case.get("case_name", "")
            court = case.get("court_name", "")
            status = case.get("case_status", "")
            plaintiff = case.get("plaintiff_name", "")
            defendant = case.get("defendant_name", "")
            outcome = case.get("outcome", "")
            patent_id = case.get("patent_id", "")
            filing_date = case.get("filing_date", "")
            
            # Determine case type based on case number and court
            case_type = "Court Case"
            if case_number and ("PTAB" in case_number.upper() or "IPR" in case_number.upper() or "CBM" in case_number.upper()):
                case_type = "PTAB Decision"
            elif case_number and ("APPEAL" in case_number.upper() or "FED" in court.upper() or "CIRCUIT" in court.upper()):
                case_type = "Appellate Decision"
            
            # Build case identifier
            case_identifier = case_name or case_number or "Unknown Case"
            if court:
                case_identifier += f", {court}"
            if case_number and case_number not in case_identifier:
                case_identifier += f" ({case_number})"
            
            # Build case text/excerpt (litigation data from PostgreSQL)
            case_text_parts = []
            if case_name:
                case_text_parts.append(f"Case Name: {case_name}")
            if case_number:
                case_text_parts.append(f"Case Number: {case_number}")
            if court:
                case_text_parts.append(f"Court: {court}")
            if filing_date:
                case_text_parts.append(f"Filing Date: {filing_date}")
            if status:
                case_text_parts.append(f"Status: {status}")
            if plaintiff:
                case_text_parts.append(f"Plaintiff: {plaintiff}")
            if defendant:
                case_text_parts.append(f"Defendant: {defendant}")
            if patent_id:
                case_text_parts.append(f"Related Patent(s): {patent_id}")
            if outcome:
                case_text_parts.append(f"Outcome: {outcome}")
            
            case_text = "\n".join(case_text_parts)
            
            # Use case_number as ref_id, or generate one
            ref_id = case_number or f"case_{len(litigation_chunks) + 1}"
            
            litigation_chunks.append({
                "case_type": case_type,
                "ref_id": ref_id,
                "case_identifier": case_identifier,
                "related_patent_ids": patent_id or "N/A",
                "text": case_text
            })
    else:
        print(f"DEBUG: Synthesizer received NO litigation context (intent: {intent})")
    
    # Format litigation materials according to prompt template
    litigation_materials = []
    for chunk in litigation_chunks:
        litigation_materials.append(
            f"[Case Type: {chunk['case_type']}; Ref: {chunk['ref_id']}]\n"
            f"- Identifier: {chunk['case_identifier']}\n"
            f"- Related Patent(s): {chunk['related_patent_ids']}\n"
            f"- Excerpt:\n{chunk['text']}\n"
        )
    litigation_context_text = "\n".join(litigation_materials) if litigation_materials else ""
    
    # Normalize intent to match prompt format (TECHNICAL/LEGAL -> technical/legal)
    intent_normalized = intent.lower() if intent else "technical"
    if intent_normalized == "both":
        intent_normalized = "legal"  # Default to legal when both, to include litigation
    
    # Build comprehensive system prompt
    # Add stronger retry instructions when retrying
    retry_warning = ""
    if retry_count > 0:
        retry_warning = f"""
⚠️⚠️⚠️ CRITICAL: THIS IS RETRY ATTEMPT #{retry_count + 1} ⚠️⚠️⚠️

Your previous response was REJECTED by the quality control system. The previous draft had critical errors that you MUST fix.

MANDATORY RETRY INSTRUCTIONS:
1. **DO NOT** simply append corrections to the previous draft
2. **DO NOT** repeat the same mistakes from the previous attempt
3. **COMPLETELY REWRITE** the response from scratch, addressing ALL feedback points
4. **ACKNOWLEDGE** the previous failure and demonstrate you understand what went wrong
5. **BE EXTRA CAREFUL** about citations, grounding, and completeness
6. **VERIFY** every citation exists in the provided context before including it
7. **ENSURE** every factual claim is directly supported by the context materials

The critique feedback below lists SPECIFIC issues that caused the rejection. Address EACH point systematically.
If you fail again, the system will retry up to 3 times total. Make this attempt count.

"""
    
    system_prompt = f"""SYSTEM:

{retry_warning}You receive:
- An INTENT label from an upstream router node: either "technical" or "legal".
- PATENT MATERIALS: patent claims, specifications, abstracts, cited prior art, related patents.
- LITIGATION MATERIALS: court decisions, PTAB decisions, oppositions, office actions, litigation summaries.

Your behavior depends on the INTENT:

1) If INTENT = "technical":
   - Focus ONLY on technical understanding, novelty-style analysis, feature extraction, comparisons of technical content, and high-level explanations.
   - You MUST IGNORE the LITIGATION MATERIALS for your reasoning and conclusions.
   - You may briefly acknowledge that litigation materials exist in the context, but you MUST NOT rely on them for any claim about scope, validity, infringement, or risk.
   - Base your answer exclusively on the PATENT MATERIALS and clearly cite patent IDs, claims, and paragraphs.

2) If INTENT = "legal":
   - You SHOULD use both PATENT MATERIALS and LITIGATION MATERIALS.
   - Use PATENT MATERIALS for:
     - technical content, claim language, and disclosed embodiments.
   - Use LITIGATION MATERIALS for:
     - claim construction,
     - validity/invalidity determinations,
     - infringement/non-infringement holdings,
     - procedural posture and outcomes,
     - any other legal characterization of the patent.
   - If there is tension between broad claim wording and a narrower construction in a decision, explain both, and emphasize the construction in the LITIGATION MATERIALS.
   - If no relevant litigation materials are present, clearly say so and base your legal analysis only on the patent text, noting the limitation.

In all cases:
- Answer ONLY using the provided context.
- Be explicit, cautious, and conservative. Do NOT speculate beyond the text.
- When you make factual statements, you MUST cite the source as:
  - (Patent: <PATENT_ID>, Claim <N>) or (Patent: <PATENT_ID>, ¶[XXXX])
  - (Case: <CASE_NAME>, p.<page> / §<section>) or (PTAB: <CASE_ID>, §<section>) [LEGAL INTENT ONLY]
- You do NOT provide legal advice. You provide analytical explanations based on the documents.
{retry_warning}"""

    # Check if query asks for a specific patent ID
    query_lower = query.lower()
    patent_id_in_query = None
    if "us" in query_lower and any(c.isdigit() for c in query):
        # Try to extract patent ID from query
        patent_match = re.search(r'US[\d\-]+[A-Z]?\d*', query, re.IGNORECASE)
        if patent_match:
            patent_id_in_query = patent_match.group().upper()
    
    # Check if the requested patent is in the context
    context_has_requested_patent = False
    if patent_id_in_query:
        context_has_requested_patent = any(
            patent_id_in_query.upper() in doc.get("patent_id", "").upper() 
            for doc in documents
        )
    
    # Build user message according to new prompt template
    # Include feedback from previous critique if retrying
    feedback_section = ""
    if feedback and retry_count > 0:
        feedback_section = f"""

═══════════════════════════════════════════════════════════════
🚨 RETRY ATTEMPT #{retry_count + 1} - PREVIOUS DRAFT REJECTED 🚨
═══════════════════════════════════════════════════════════════

CRITIQUE FEEDBACK FROM PREVIOUS ATTEMPT (MUST FIX ALL ISSUES):

{feedback}

CRITICAL INSTRUCTIONS FOR THIS RETRY:
1. Read the feedback above CAREFULLY - it lists SPECIFIC errors that caused rejection
2. COMPLETELY REWRITE your response - do NOT copy/paste from the previous attempt
3. Address EACH point in the feedback systematically
4. Double-check ALL citations exist in the provided context before including them
5. Ensure EVERY factual claim is directly supported by the context materials
6. Be EXTRA careful about citation format and completeness
7. If the feedback mentions missing citations, verify those patent IDs are in the context list
8. If the feedback mentions hallucinations, remove ALL unsupported claims
9. If the feedback mentions incomplete response, ensure you cover all key findings

REMEMBER: This is attempt #{retry_count + 1} of 3. The system will reject again if you don't fix ALL issues.
═══════════════════════════════════════════════════════════════

"""
    
    user_message_content = f"""INTENT (router output):

{intent_normalized}  # "technical" or "legal"

USER QUESTION:

{query}

PATENT MATERIALS (retrieved by the patent RAG):

{patent_context if patent_context else "No patent materials retrieved."}

LITIGATION MATERIALS (retrieved by the litigation agent):

{litigation_context_text if litigation_context_text else "No litigation materials retrieved."}
{feedback_section}
ANSWER INSTRUCTIONS:

1. Respect INTENT strictly:

   - If INTENT = "technical":
     - Ignore LITIGATION MATERIALS in your reasoning.
     - Do not discuss claim construction, validity, infringement, risk assessment, or litigation outcomes.
     - Focus on: technical explanations, feature extraction, comparisons, summaries of inventions, mapping of claims to description, etc.

   - If INTENT = "legal":
     - Use both PATENT MATERIALS and LITIGATION MATERIALS as described above.
     - You may discuss claim construction, validity challenges, infringement issues, and case outcomes, always citing the relevant documents.

2. For technical analysis (both intents, but especially when INTENT = "technical"):

   - When discussing technical features, scope of claims as written, or prior art disclosures, use PATENT MATERIALS.
   - Cite specific claims or paragraphs, e.g., (Patent: US_10123456, Claim 1) or (Patent: US_10123456, ¶[0023]).

3. For legal analysis (ONLY when INTENT = "legal"):

   - When discussing:
     - infringement/non-infringement,
     - validity/invalidity,
     - claim construction (meaning of terms),
     - procedural posture or litigation risk,
     you MUST reference LITIGATION MATERIALS where available and clearly identify:
     - which case/decision you rely on,
     - what that decision says about the patent or its claims.
   - Example: (Case: Apple v. Samsung, p. 12) or (PTAB: IPR2023-01234, §III.B).
   - If no litigation materials are provided, state this explicitly and note that conclusions are based solely on the patent text.

4. Structure your final answer (when applicable) as:

   - For INTENT = "technical":
     - Background / summary of relevant patent(s)
     - Key technical features / claim breakdown
     - Technical comparisons or mappings requested by the user
     - Clear, concise conclusion focused on technical understanding

   - For INTENT = "legal":
     - Background / summary of relevant patent(s)
     - Relevant litigation history and findings
     - Analysis in view of both patent and litigation materials (e.g., overlap, differences, validity/infringement risk, remaining uncertainty)
     - Conclusion, clearly stating any limits due to missing or incomplete context, framed as analysis (not legal advice).

5. If something is unsupported or unclear in the provided materials:

   - Say explicitly that it is not supported by the context.
   - Do not invent facts or legal conclusions.

6. Use clear, structured writing (headings, bullet points, or tables) when it helps the user.

   - For complex comparisons, consider using tables.
   - For structured outputs (e.g., feature lists or JSON) follow any explicit schema requested by the user."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message_content),
    ]
    
    # Stream tokens
    draft_text = ""
    async for chunk in synthesizer_llm.astream(messages):
        if hasattr(chunk, 'content') and chunk.content:
            draft_text += chunk.content
            yield {"draft": draft_text}
    
    # Final yield with complete draft
    yield {"draft": draft_text}


async def critic_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Critique the draft response and verify citations."""
    draft = state.get("draft", "")
    documents = state.get("documents", [])
    
    # Check 0: Context Existence - CRITICAL
    # If no documents retrieved, fail immediately unless draft explicitly states "no results"
    if not documents or len(documents) == 0:
        # Check if draft acknowledges no results
        draft_lower = draft.lower()
        if any(phrase in draft_lower for phrase in ["no patents", "no documents", "no results", "unable to provide", "no matching"]):
            # This is acceptable - draft correctly states no results
            return {
                "critique": {
                    "status": "PASS",
                    "feedback": "Draft correctly acknowledges no documents retrieved."
                }
            }
        else:
            # Draft claims facts without context - FAIL
            return {
                "critique": {
                    "status": "FAIL",
                    "feedback": "CRITICAL: No documents retrieved. Cannot generate report without context. The draft must explicitly state that no patents were found."
                }
            }
    
    # CRITICAL: Check if documents exist - fail immediately if no context
    if not documents or len(documents) == 0:
        # Check if draft acknowledges no documents (acceptable)
        if "no patents" in draft.lower() or "no documents" in draft.lower() or "0 patent" in draft.lower():
            # Draft correctly states no results - this is acceptable
            return {
                "critique": {
                    "status": "PASS",
                    "feedback": "Draft correctly acknowledges no documents retrieved."
                }
            }
        else:
            # Draft makes claims without context - FAIL
            return {
                "critique": {
                    "status": "FAIL",
                    "feedback": "CRITICAL: No documents retrieved but draft contains claims. System must not generate factual claims without retrieved context. Draft should clearly state 'No patents found' instead of making unsupported claims."
                }
            }
    
    # Normalize patent ID for comparison (handles all format variations)
    def normalize_patent_id(patent_id: str) -> str:
        """Normalize patent ID to a canonical form for comparison."""
        if not patent_id:
            return ""
        # Remove dashes, slashes, spaces, and convert to uppercase
        normalized = patent_id.upper().replace("-", "").replace("/", "").replace(" ", "").strip()
        # Remove common prefixes but keep the core ID
        # Handle formats like: US-2022091384-A1, US2022091384A1, 2022091384A1
        if normalized.startswith("US"):
            normalized = normalized[2:]
        return normalized
    
    # Extract all patent IDs from context and create normalized set
    context_patent_ids = set()
    context_normalized = set()
    for doc in documents:
        patent_id = doc.get("patent_id", "")
        if patent_id:
            # Store original and normalized versions
            context_patent_ids.add(patent_id)
            # Also add underscore variant (for citation format matching)
            context_patent_ids.add(patent_id.replace("-", "_"))
            # Also add dash variant if it has underscores
            if "_" in patent_id:
                context_patent_ids.add(patent_id.replace("_", "-"))
            normalized = normalize_patent_id(patent_id)
            if normalized:
                context_normalized.add(normalized)
    
    # Extract cited patent IDs from draft
    # Support multiple formats:
    # - Old format: [[ID]](URL)
    # - New format: (Patent: ID, Claim N) or (Patent: ID, ¶[XXXX])
    # - New format with Section: (Patent: ID, Section: Type, Ref: ref_id)
    citation_pattern_old = r'\[\[([^\]]+)\]\]'  # Old format: [[US123]](URL)
    # New format: (Patent: ID, ...) - more flexible to handle Section: and Ref: formats
    citation_pattern_new = r'\(Patent:\s*([^,)]+)(?:,\s*[^)]+)*\)'
    
    cited_ids = []
    # Extract from old format
    cited_ids.extend(re.findall(citation_pattern_old, draft))
    # Extract from new format - handle (Patent: ID, ...) with any additional info
    new_citations = re.findall(citation_pattern_new, draft)
    for match in new_citations:
        patent_id = match[0].strip() if isinstance(match, tuple) else match.strip()
        # Clean up the patent ID
        patent_id = patent_id.strip()
        if not patent_id:
            continue
        # Convert underscore format to dash format for matching (context uses dashes)
        patent_id_normalized = patent_id.replace("_", "-")
        cited_ids.append(patent_id_normalized)
        # Also add the original with underscores in case context uses that format
        if "_" in patent_id and patent_id != patent_id_normalized:
            cited_ids.append(patent_id)
    
    # Remove duplicates
    cited_ids = list(set(cited_ids))
    
    # Debug: Log extracted citations
    if cited_ids:
        print(f"DEBUG: Critic extracted {len(cited_ids)} citation(s): {cited_ids}")
    else:
        print(f"DEBUG: Critic found no citations in draft")
    
    # Verify citations using normalized comparison
    missing_citations = []
    for cited_id in cited_ids:
        # Normalize cited ID (handle both underscore and dash formats)
        cited_id_clean = cited_id.replace("_", "-").replace("/", "")
        normalized_cited = normalize_patent_id(cited_id_clean)
        found = False
        
        # Check if normalized version matches any context patent
        if normalized_cited in context_normalized:
            found = True
        else:
            # Also check if any context patent's normalized version matches
            for context_id in context_patent_ids:
                context_id_clean = context_id.replace("_", "-").replace("/", "")
                if normalize_patent_id(context_id_clean) == normalized_cited:
                    found = True
                    break
                # Also do direct comparison with normalized versions
                if normalize_patent_id(context_id) == normalized_cited:
                    found = True
                    break
        
        if not found:
            missing_citations.append(cited_id)
    
    # Debug: Log verification results
    if missing_citations:
        print(f"DEBUG: Critic found {len(missing_citations)} potentially missing citation(s): {missing_citations}")
    else:
        print(f"DEBUG: Critic verified all {len(cited_ids)} citation(s) are valid")
    
    # Build list of ALL valid patent IDs from context (for the critic to reference)
    valid_patent_ids = sorted(list(context_patent_ids))
    valid_patent_ids_str = ", ".join(valid_patent_ids)
    
    # Extract ONLY the citations that actually appear in the draft
    actual_citations_in_draft = cited_ids.copy()
    
    system_prompt = """You are a STRICT quality control grader for patent analysis reports.
Review the draft response against the source context with ZERO tolerance for errors.

CRITICAL: You must ONLY check citations that actually appear in the draft text. Do NOT invent or hallucinate patent IDs.

IMPORTANT: Citations have been programmatically verified. If the verification shows all citations are valid, you should PASS unless there are other serious issues (hallucinations, unsupported claims, etc.).

Perform these STRICT checks:
1. **Hallucination Check (MANDATORY):** 
   - Is there ANY claim, fact, or statement in the draft NOT directly supported by the provided context?
   - Are there any technical details, dates, names, or numbers mentioned that are NOT in the source materials?
   - FAIL if you find ANY unsupported claims, even minor ones.

2. **Citation Format Check:** Are citations formatted correctly? Acceptable formats:
   - (Patent: <PATENT_ID>, Claim <N>) or (Patent: <PATENT_ID>, ¶[XXXX])
   - (Patent: <PATENT_ID>, Section: <TYPE>, Ref: <REF>) [also valid]
   - [[PatentID]](URL) [legacy format, also acceptable]
   - (Case: <CASE_NAME>, p.<page>) or (PTAB: <CASE_ID>, §<section>) [for legal citations]
   - FAIL if citations are missing or incorrectly formatted.

3. **Citation Verification (MANDATORY):** Do ALL cited patent IDs exist in the provided context?
   - NOTE: Patent IDs may appear with underscores (US_123) or dashes (US-123) - both are valid
   - The programmatic verification has already checked this - trust it unless you see obvious errors
   - FAIL if ANY cited patent ID is not in the context list.

4. **Completeness Check:**
   - Does the draft adequately address the user's query?
   - Are key findings from the context properly summarized?
   - FAIL if the response is incomplete or misses important information.

5. **Grounding Check (STRICT):**
   - Every factual statement MUST be traceable to the provided context
   - No speculation, no assumptions, no general knowledge beyond context
   - FAIL if the draft contains ungrounded statements.

STRICT RULES:
- ONLY verify citations that are actually in the draft text
- Do NOT mention patent IDs that are not in the draft
- The valid patent IDs from context are: {valid_patent_ids_str}
- When checking citations, compare them to this exact list
- If programmatic verification says citations are valid, do NOT reject based on citation format alone
- Be STRICT about hallucinations - even minor unsupported claims should result in FAIL
- Be STRICT about completeness - missing key information should result in FAIL

Return ONLY a JSON object with this exact format:
{{
  "status": "PASS" | "FAIL",
  "feedback": "Specific, actionable instructions on what to fix. Be detailed and precise."
}}

If status is "FAIL", provide detailed, actionable feedback with specific examples. Do NOT fail for citation format if the patent IDs are correct, but DO fail for hallucinations, missing citations, or incomplete responses."""

    # Build context summary
    context_summary = "\n".join([
        f"- {doc.get('patent_id', '')}: {doc.get('title', '')[:100]}"
        for doc in documents[:10]
    ])
    
    messages = [
        SystemMessage(content=system_prompt.format(valid_patent_ids_str=valid_patent_ids_str)),
        HumanMessage(
            content=f"Draft Response:\n{draft}\n\n"
                   f"Source Context (Patent IDs):\n{context_summary}\n\n"
                   f"VALID PATENT IDs FROM CONTEXT (use these for verification):\n{valid_patent_ids_str}\n\n"
                   f"ACTUAL CITATIONS IN DRAFT (only check these):\n{', '.join(actual_citations_in_draft) if actual_citations_in_draft else 'None'}\n\n"
                   f"{'⚠️ WARNING: These citations may be missing from context: ' + ', '.join(missing_citations) if missing_citations else 'All citations appear to be valid.'}"
        ),
    ]
    
    response = await critic_llm.ainvoke(messages)
    content = response.content.strip()
    
    # Extract JSON from response - handle control characters
    try:
        # Clean control characters except newlines/tabs
        cleaned_content = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', content)
        critique_data = json.loads(cleaned_content)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
        if json_match:
            try:
                # Clean the matched JSON string
                json_str = json_match.group()
                cleaned_json = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', json_str)
                # Try to fix common JSON issues
                cleaned_json = cleaned_json.replace('\n', ' ').replace('\r', ' ')
                critique_data = json.loads(cleaned_json)
            except json.JSONDecodeError:
                # Last resort: try to extract status and feedback manually
                status_match = re.search(r'"status"\s*:\s*"([^"]+)"', content, re.IGNORECASE)
                feedback_match = re.search(r'"feedback"\s*:\s*"([^"]+)"', content, re.IGNORECASE)
                status = status_match.group(1) if status_match else ("FAIL" if missing_citations else "PASS")
                feedback = feedback_match.group(1) if feedback_match else (f"Missing citations: {', '.join(missing_citations)}" if missing_citations else "No issues found")
                critique_data = {"status": status, "feedback": feedback}
        else:
            # Default: fail if we found missing citations
            critique_data = {
                "status": "FAIL" if missing_citations else "PASS",
                "feedback": f"Missing citations: {', '.join(missing_citations)}" if missing_citations else "No issues found",
            }
    
    # Force FAIL if citations are missing (but only for citations that actually appear in draft)
    # Filter out any citations that don't actually appear in the draft text
    actual_missing = []
    for missing_id in missing_citations:
        # Check if this citation actually appears in the draft in any format
        missing_id_variations = [
            missing_id,
            missing_id.replace("-", "_"),
            missing_id.replace("_", "-"),
            f"[[{missing_id}]]",
            f"(Patent: {missing_id}",
            f"(Patent: {missing_id.replace('-', '_')}",
        ]
        found_in_draft = any(var in draft for var in missing_id_variations)
        if found_in_draft:
            actual_missing.append(missing_id)
    
    # If we programmatically verified all citations are valid, trust that over LLM
    if not actual_missing and not missing_citations:
        # All citations verified programmatically - override LLM if it says FAIL
        if critique_data.get("status") == "FAIL" and "citation" in critique_data.get("feedback", "").lower():
            # LLM might be hallucinating - trust our verification
            print(f"WARNING: Critic LLM rejected but all citations verified programmatically. Overriding to PASS.")
            critique_data["status"] = "PASS"
            critique_data["feedback"] = "All citations verified and valid."
    elif actual_missing:
        critique_data["status"] = "FAIL"
        critique_data["feedback"] = (
            f"VERIFICATION FAILED: The following patent IDs are cited but not in context: {', '.join(actual_missing)}. "
            f"Remove these citations or only cite patents that exist in the provided context. "
            + critique_data.get("feedback", "")
        )
    elif missing_citations:
        # Citations were flagged but don't actually appear in draft - might be LLM hallucination
        # Log but don't fail
        print(f"WARNING: Critic flagged citations that don't appear in draft: {missing_citations}")
        # Only fail if the LLM explicitly says to fail for other reasons
        if critique_data.get("status") != "FAIL":
            critique_data["status"] = "PASS"
            critique_data["feedback"] = "All actual citations in the draft are valid."
    
    return {"critique": critique_data}

