from pathlib import Path

from pydantic_settings import BaseSettings

local_dotenv_file = ".env.local"

# Project root (repo root): .../reinvent-ml-server
# __file__ = .../reinvent-ml-server/src/ml_service/config.py
# parents[0] = ml_service, [1] = src, [2] = reinvent-ml-server
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    # App
    APP_ENV: str = "dev"
    LOG_LEVEL: str = "INFO"
    WORKERS: int = 1

    # DB
    # DATABASE_URL: str = ""
    DB_NAME: str = ""
    DB_HOST_EXTERNAL: str = ""
    DB_HOST_INTERNAL: str = ""
    DB_USER: str = ""
    DB_PASSWORD: str = ""
    DB_PORT: str = "5432"
    PG_SEARCH_PATH: str = "ai,public"

    # MinIO for connecting to MinIO later
    MINIO_ENDPOINT: str | None = None
    MINIO_ACCESS_KEY: str | None = None
    MINIO_SECRET_KEY: str | None = None
    MINIO_SECURE: bool = False

    # Embeddings
    EMBED_PROVIDER: str = "huggingface"  # "huggingface" | "api"
    EMBED_MODEL_NAME: str = "intfloat/multilingual-e5-small"
    EMBED_DIM: int = 384  # e5-small is 384
    EMBED_DEVICE: str = "auto"  # "auto" | "cpu" | "cuda" | "mps"
    EMBED_BATCH_SIZE: int = 128
    HF_HOME: str = ".cache/huggingface"  # mount a volume here
    HF_HUB_OFFLINE: bool = False

    # LLM (Apertus via OpenAI-like interface) from SwissAI initiative
    LLM_API_BASE: str = "https://api.swissai.cscs.ch/v1"
    LLM_API_KEY: str = ""  # Extract from environment variable: .env + docker compose
    LLM_MODEL: str = "swissai/Apertus-70B-Instruct"
    # LLM_TIMEOUT: int = 120

    # Table naming
    PG_SCHEMA: str = "ai"
    PG_TABLE: str = "llamaindex"

    # Feature toggles
    ENABLE_RAG: bool = True  # Disable to skip DB + pgvector init

    # SceneScript configuration
    SCENESCRIPT_BASE_DIR: str = str(_PROJECT_ROOT / "scenescript")
    SCENESCRIPT_WEIGHTS_REL: str = "weights/scenescript_model_ase.ckpt"
    SCENESCRIPT_POINTCLOUD_DIR: str = "pointclouds"
    SCENESCRIPT_UPLOAD_DIR: str = "uploaded_data"

    # API Bearer Tokens:
    ML_API_TOKEN: str = ""
    DB_API_TOKEN: str = ""

    # Read dotenv file from project root directory
    # _env_file = Path(__file__).resolve().parent.parent.parent.parent / local_dotenv_file
    _env_file = _PROJECT_ROOT / local_dotenv_file

    model_config = {
        "env_file": str(_env_file),
        "env_file_encoding": "utf-8",
    }
