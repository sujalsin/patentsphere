from fastapi import Depends, FastAPI

from app.dependencies import get_app_settings, require_local_profile
from app.orchestrator import Orchestrator


def create_app() -> FastAPI:
    settings = get_app_settings()
    app = FastAPI(title=settings.app_name, version=settings.app_version)

    orchestrator = Orchestrator()

    @app.get("/health")
    def health():
        """Health check endpoint - no dependencies, fast response."""
        return {"status": "ok", "profile": settings.app_profile}
    
    @app.get("/status")
    async def status():
        """Detailed status endpoint - checks all services."""
        status_info = {
            "api": "ok",
            "profile": settings.app_profile,
            "services": {}
        }
        
        # Check PostgreSQL
        try:
            import psycopg
            conn_str = f"postgresql://{settings.database.user}:{settings.database.password}@{settings.database.host}:{settings.database.port}/{settings.database.database}"
            with psycopg.connect(conn_str, connect_timeout=2) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM patent_chunks")
                    count = cur.fetchone()[0]
                    status_info["services"]["postgresql"] = {"status": "ok", "patent_chunks": count}
        except Exception as e:
            status_info["services"]["postgresql"] = {"status": "error", "error": str(e)}
        
        # Check Qdrant
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(host=settings.qdrant.host, port=settings.qdrant.port, timeout=2)
            collections = client.get_collections()
            status_info["services"]["qdrant"] = {"status": "ok", "collections": len(collections.collections)}
        except Exception as e:
            status_info["services"]["qdrant"] = {"status": "error", "error": str(e)}
        
        # Check Ollama
        try:
            import httpx
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"{settings.llm.ollama_host}/api/tags")
                if response.status_code == 200:
                    models = [m.get("name") for m in response.json().get("models", [])]
                    status_info["services"]["ollama"] = {"status": "ok", "models": models}
                else:
                    status_info["services"]["ollama"] = {"status": "error", "code": response.status_code}
        except Exception as e:
            status_info["services"]["ollama"] = {"status": "error", "error": str(e)}
        
        return status_info

    @app.get("/whoami", dependencies=[Depends(require_local_profile)])
    def whoami(profile=Depends(require_local_profile)):
        return {
            "profile": profile.profile,
            "dataset": profile.dataset_path(),
            "max_parallel_agents": profile.max_parallel_agents(),
        }

    @app.get("/retrieve", dependencies=[Depends(require_local_profile)])
    async def retrieve(query: str):
        # Use citation_mapper agent (the actual key in orchestrator)
        citation_agent = orchestrator.agents.get("citation") or orchestrator.agents.get("citation_mapper")
        if not citation_agent:
            return {"error": "Citation agent not found"}
        results = await citation_agent.run(query)
        return results.data

    @app.get("/query", dependencies=[Depends(require_local_profile)])
    async def query_endpoint(query: str):
        results = await orchestrator.run_all(query)
        return {k: v.data for k, v in results.items()}

    return app


app = create_app()

