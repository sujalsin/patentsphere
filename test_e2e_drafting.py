import asyncio
from config import get_pipeline_llm
from graph.graph import workflow
from pipeline.orchestrator import run_pipeline

async def test_drafting_pipeline():
    print("Initializing Groq LLM...")
    llm = get_pipeline_llm("synthesizer")
    
    disclosure = (
        "This invention relates to a system that uses large language models "
        "to automatically summarize patent litigation cases. It uses a vector database "
        "to store the cases and an LLM to generate a plain-english summary. The system "
        "can alert users when a new relevant case is filed."
    )
    
    print("\nStarting orchestrator.run_pipeline...")
    print("This will run: Decompose -> Analyze -> Draft Claims -> Generate Spec -> Review")
    
    state = await run_pipeline(
        raw_input=disclosure,
        llm=llm,
        workflow=workflow,
        score_llm=llm,
        tenant_id="test_tenant"
    )
    
    print("\n" + "="*50)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*50)
    
    if state.failed_at_step:
        print(f"PIPELINE FAILED AT STEP: {state.failed_at_step}")
        if state.checklist and state.checklist.reviewer_notes:
            print(f"Error notes: {state.checklist.reviewer_notes}")
        return False
        
    print(f"\nAudit Log contains {len(state.audit_log)} entries.")
    for entry in state.audit_log:
        print(f" - [{entry.step}] {entry.status}: {entry.detail}")
        
    print("\nDecomposition:")
    print(f" - Extracted {len(state.decomposition.sections)} sections")
    
    print("\nAnalysis:")
    print(f" - Analyzed {len(state.analyses)} sections")
    
    print("\nClaims Drafted:")
    print(f" - Independent claims: {state.claim_set.independent_claim_count}")
    print(f" - Dependent claims: {state.claim_set.dependent_claim_count}")
    
    print("\nSpecification Generated:")
    print(f" - Title: {state.specification.title}")
    print(f" - Abstract length: {len(state.specification.abstract)} chars")
    print(f" - Word count: ~{state.specification.word_count()} words")
    
    print("\nHuman-Wall Review Checklist:")
    print(f" - Overall Confidence: {state.checklist.overall_confidence:.2f}")
    print(f" - Flags: {len(state.checklist.flags)}")
    print(f" - Export Allowed: {state.checklist.export_allowed}")
    
    print("\nEnd-to-End test passed successfully!")
    return True

if __name__ == "__main__":
    success = asyncio.run(test_drafting_pipeline())
    if not success:
        exit(1)
