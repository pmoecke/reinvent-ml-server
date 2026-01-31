import argparse
import os
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ml_service.routers import docling_router, scenescript_router

from .config import Settings
from .startup import (
    SceneScriptState,
    init_db,
    init_embedder,
    init_llm,
    init_pgvector,
    init_scenescript,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = Settings()

    # Set HF cache path early so downstream libs use it
    os.environ.setdefault("HF_HOME", app.state.settings.HF_HOME)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if app.state.settings.HF_HUB_OFFLINE:
        os.environ["HF_HUB_OFFLINE"] = "1"

    # Simple debug to verify env loading (avoid printing full secrets)
    print(f"DEBUG Settings.APP_ENV={app.state.settings.APP_ENV}")
    print(f"DEBUG Settings.ENABLE_RAG={app.state.settings.ENABLE_RAG}")
    print(f"DEBUG Settings.ML_API_TOKEN set={app.state.settings.ML_API_TOKEN}")

    # Init resources
    if app.state.settings.ENABLE_RAG:
        app.state.db = init_db(app.state.settings)
        app.state.vector_store = init_pgvector(app.state.settings)
        app.state.embedder = init_embedder(app.state.settings)  # may download/warmup
        app.state.llm = init_llm(app.state.settings)
    else:
        app.state.db = None
        app.state.vector_store = None
        app.state.embedder = None
        app.state.llm = None

    app.state.scenescript = init_scenescript(app.state.settings)

    app.state.model_ready = True  # flip after warmup completes
    yield

    # Teardown
    # try:
    #     app.state.db.close()
    # except Exception:
    #     pass


def create_app():
    app = FastAPI(
        title="ML-Service",
        description=("Internal ML Service for the ReInvent app"),
        version="0.0.1",
        lifespan=lifespan,
    )
    public_router = APIRouter()

    @public_router.get("/", tags=["health"])
    def health_check():
        return {"status": "healthy", "service": "ml-service"}

    @app.get("/health/ready", tags=["health"])
    def ready():
        scenescript_state: SceneScriptState | None = getattr(app.state, "scenescript", None)
        return {
            "ready": bool(getattr(app.state, "model_ready", False)),
            "embed_model": app.state.settings.EMBED_MODEL_NAME,
            "hf_cache": os.environ.get("HF_HOME"),
            "scenescript": {
                "loaded": bool(scenescript_state and scenescript_state.model),
                "device": getattr(scenescript_state, "device", None) if scenescript_state else None,
            },
        }

    app.include_router(public_router)
    # app.include_router(health_router)
    app.include_router(docling_router.router)
    app.include_router(scenescript_router.router)
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


    return app


def start_server():
    parser = argparse.ArgumentParser()

    # API flag
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="The host to run the server",
    )
    parser.add_argument(
        "--port",
        default=8000,
        help="The port to run the server",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Run in development mode",
    )

    args = parser.parse_args()

    # Import uvicorn here to avoid circular imports
    import uvicorn

    # Run the server using uvicorn
    uvicorn.run(
        "ml_service.app:create_app",
        host=args.host,
        port=int(args.port),
        reload=args.dev,
        factory=True,
    )


if __name__ == "__main__":
    start_server()
