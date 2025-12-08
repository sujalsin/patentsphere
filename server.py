"""FastAPI server for PatentSphere with REST API and Chainlit UI."""
import uuid
from typing import List, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chainlit.utils import mount_chainlit

from graph.graph import workflow, AgentState


# Pydantic Models
class QueryRequest(BaseModel):
    """Request model for patent analysis queries."""
    query: str
    thread_id: Optional[str] = None


class DocumentRef(BaseModel):
    """Reference model for citations."""
    id: str
    title: str
    url: str
    source: str  # "patent" or "litigation"


class QueryResponse(BaseModel):
    """Response model for patent analysis queries."""
    answer: str
    intent: Optional[str] = None
    citations: List[DocumentRef] = []
    thread_id: str


def generate_patent_url(patent_id: str) -> str:
    """Generate Google Patents URL from patent ID."""
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


# Initialize FastAPI app
app = FastAPI(title="PatentSphere")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/analyze", response_model=QueryResponse)
async def analyze_patent(request: QueryRequest):
    """
    Analyze a patent query and return structured results.
    
    This endpoint processes queries through the multi-agent workflow
    and returns the analysis with citations.
    """
    # Generate thread_id if not provided
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    # Initialize state
    initial_state: AgentState = {
        "query": request.query,
        "intent": None,
        "technical_keywords": None,
        "legal_entities": None,
        "patent_ids": None,
        "keywords": None,
        "date_range": None,
        "documents": [],
        "litigation_context": [],
        "draft": None,
        "critique": None,
        "retry_count": 0,
        "final_response": None,
    }
    
    # Invoke workflow
    final_state = await workflow.ainvoke(initial_state, config=config)
    
    # Extract final response text
    answer = final_state.get("final_response") or final_state.get("draft", "")
    if not answer:
        answer = "No response generated."
    
    # Extract intent
    intent = final_state.get("intent")
    
    # Extract citations from documents and litigation_context
    citations: List[DocumentRef] = []
    
    # Process patent documents
    documents = final_state.get("documents", [])
    for doc in documents:
        patent_id = doc.get("patent_id", "")
        if not patent_id:
            continue
        
        title = doc.get("title", "No title available")
        stored_url = doc.get("url", "")
        if stored_url and stored_url.startswith("https://patents.google.com"):
            url = stored_url
        else:
            url = generate_patent_url(patent_id)
        
        citations.append(
            DocumentRef(
                id=patent_id,
                title=title,
                url=url,
                source="patent"
            )
        )
    
    # Process litigation context
    litigation_context = final_state.get("litigation_context", [])
    for case in litigation_context:
        case_number = case.get("case_number", "")
        case_name = case.get("case_name", "Unknown Case")
        
        # Use case_number as ID, fallback to case_name
        case_id = case_number if case_number else case_name
        if not case_id or case_id == "Unknown":
            continue
        
        # Generate a URL for litigation cases (if available)
        # For now, use an empty string or a placeholder
        url = case.get("url", "")
        
        citations.append(
            DocumentRef(
                id=case_id,
                title=case_name,
                url=url,
                source="litigation"
            )
        )
    
    return QueryResponse(
        answer=answer,
        intent=intent,
        citations=citations,
        thread_id=thread_id
    )


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "PatentSphere"}


# Mount Chainlit UI at root path
# Important: This must be done after all FastAPI routes are defined
mount_chainlit(app=app, target="app.py", path="/")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
