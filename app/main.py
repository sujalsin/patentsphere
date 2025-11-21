from fastapi import Depends, FastAPI

from app.dependencies import get_app_settings, require_local_profile
from app.orchestrator import Orchestrator


def create_app() -> FastAPI:
    settings = get_app_settings()
    app = FastAPI(title=settings.app_name, version=settings.app_version)

    orchestrator = Orchestrator()

    @app.get("/health")
    def health():
        return {"status": "ok", "profile": settings.app_profile}

    @app.get("/whoami", dependencies=[Depends(require_local_profile)])
    def whoami(profile=Depends(require_local_profile)):
        return {
            "profile": profile.profile,
            "dataset": profile.dataset_path(),
            "max_parallel_agents": profile.max_parallel_agents(),
        }

    @app.get("/retrieve", dependencies=[Depends(require_local_profile)])
    async def retrieve(query: str):
        results = await orchestrator.agents["citation"].run(query)
        return results.data

    @app.get("/query", dependencies=[Depends(require_local_profile)])
    async def query_endpoint(query: str):
        results = await orchestrator.run_all(query)
        return {k: v.data for k, v in results.items()}

    return app


app = create_app()

