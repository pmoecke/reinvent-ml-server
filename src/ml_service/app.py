import argparse
import os
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI

from ml_service.dependencies import require_bearer
from ml_service.routers import docling_router

from .config import Settings
from .startup import init_db, init_embedder, init_llm, init_pgvector

from dotenv import load_dotenv

health_router = APIRouter(prefix="/health", tags=["health"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = Settings()

    # Set HF cache path early so downstream libs use it
    os.environ.setdefault("HF_HOME", app.state.settings.HF_HOME)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if app.state.settings.HF_HUB_OFFLINE:
        os.environ["HF_HUB_OFFLINE"] = "1"

    # Init resources
    app.state.db = init_db(app.state.settings)
    app.state.embedder = init_embedder(app.state.settings)  # may download/warmup
    app.state.llm = init_llm(app.state.settings)
    app.state.vector_store = init_pgvector(app.state.settings)

    app.state.model_ready = True  # flip after warmup completes
    yield

    # Teardown
    # try:
    #     app.state.db.close()
    # except Exception:
    #     pass


@health_router.get("/health/live")
def live():
    return {"ok": True}


def create_app():
    app = FastAPI(
        title="ML-Service",
        description=("Internal ML Service for the ReInvent app"),
        version="0.0.1",
        lifespan=lifespan,
        dependencies=[Depends(require_bearer)],
    )

    app.include_router(health_router)
    app.include_router(docling_router.router)

    @app.get("/health/ready", tags=["health"])
    def ready():
        return {
            "ready": bool(getattr(app.state, "model_ready", False)),
            "embed_model": app.state.settings.EMBED_MODEL_NAME,
            "hf_cache": os.environ.get("HF_HOME"),
        }

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
        default=8001,
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
