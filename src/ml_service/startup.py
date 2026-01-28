# app/startup.py
import psycopg
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import create_engine, text

stmt = text("""
ALTER TABLE ai.data_llamaindex
  ADD COLUMN IF NOT EXISTS project_id INTEGER
  GENERATED ALWAYS AS ((metadata_ ->> 'project_id')::INTEGER) STORED;

ALTER TABLE ai.data_llamaindex
  ADD COLUMN IF NOT EXISTS inserted_at TIMESTAMPTZ
  DEFAULT now();

ALTER TABLE ai.data_llamaindex
  DROP CONSTRAINT IF EXISTS fk_llx_project,
  ADD CONSTRAINT fk_llx_project
  FOREIGN KEY (project_id) REFERENCES public.projects(id);

CREATE INDEX IF NOT EXISTS idx_llx_project
  ON ai.data_llamaindex(project_id);
""")


def init_db(cfg):
    """
    Set up vector table in Postgres
    """
    connection_string = (
        f"postgresql://{cfg.DB_USER}:{cfg.DB_PASSWORD}@{cfg.DB_HOST_EXTERNAL}:{cfg.DB_PORT}/{cfg.DB_NAME}"
    )
    print("DEBUG: ", connection_string)

    conn = psycopg.connect(connection_string, autocommit=True)
    with conn.cursor() as cur:
        cur.execute("SET search_path TO " + cfg.PG_SEARCH_PATH)
        cur.execute("SELECT 1 FROM pg_extension WHERE extname='vector'")
        if cur.fetchone() is None:
            raise RuntimeError("pgvector extension not installed")

    return conn


def init_pgvector(cfg):
    """
    todo
    """
    connection_string = f"postgresql://{cfg.DB_USER}:{cfg.DB_PASSWORD}@{cfg.DB_HOST_EXTERNAL}:{cfg.DB_PORT}/{cfg.DB_NAME}"
    print("DEBUG: ", connection_string)
    engine = create_engine(connection_string)

    vector_store = PGVectorStore.from_params(
        database=cfg.DB_NAME,
        host=cfg.DB_HOST_EXTERNAL,
        user=cfg.DB_USER,
        password=cfg.DB_PASSWORD,
        port=str(cfg.DB_PORT),
        schema_name="ai",
        table_name="llamaindex",
        embed_dim=cfg.EMBED_DIM,
        hnsw_kwargs={
            "hnsw_m": 16,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search": 40,
            "hnsw_dist_method": "vector_cosine_ops",
        },
    )
    try:
        vector_store._initialize()  # private, but works
        # Add project_id foreign key to allow to filter for correct knowledge base
        with engine.begin() as conn:
            conn.execute(stmt)

    except Exception:
        print("Failed to initialize vector store")
        pass

    return vector_store


def _pick_device(cfg):
    if cfg.EMBED_DEVICE != "auto":
        return cfg.EMBED_DEVICE
    device = ""
    try:
        import torch

        if torch.backends.mps.is_available():
            device = "mps"
        if torch.cuda.is_available():
            device = "cuda"
    except Exception:
        device = "cpu"
        pass
    print(f"Using device: {device}")
    return device


def init_embedder(cfg):
    device = _pick_device(cfg)
    # This line downloads on first use into HF_HOME, then caches.
    # model = SentenceTransformer(cfg.EMBED_MODEL_NAME, device=device)
    model = HuggingFaceEmbedding(model_name=cfg.EMBED_MODEL_NAME, device=device)
    # Optional warmup to compile kernels / build index, improves first query latency
    # _ = model.(
    #     ["hello"], batch_size=1, normalize_embeddings=True, show_progress_bar=False
    # )
    # Wrap for llama-index if you prefer; or keep SentenceTransformer and create a small
    # adapter
    return model


def init_llm(cfg):
    return OpenAILike(
        api_key=cfg.LLM_API_KEY,
        api_base=cfg.LLM_API_BASE,
        model=cfg.LLM_MODEL,
        is_chat_model=True,
        # timeout=cfg.LLM_TIMEOUT,
    )
