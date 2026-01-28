import time

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.readers.docling import DoclingReader
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import create_engine, text

# from llama_index.core.node_parser import MarkdownNodeParser


# Schema PGVectorStore creates is as follows:
# id	bigint Auto Increment [nextval('data_llamaindex_id_seq')]
# text	character varying
# metadata_	json NULL
# node_id	character varying NULL
# embedding	public.vector(768) NULL
# Add column
#     ALTER TABLE ai.data_llamaindex
#   ADD COLUMN project_id TEXT GENERATED ALWAYS AS ((metadata_ ->> 'project_id')) STORED;

# -- if your projects table is public.projects(id TEXT/UUID), add FK (adjust type/cast)
# -- ALTER TABLE ai.data_llamaindex
# --   ADD CONSTRAINT fk_llx_project FOREIGN KEY (project_id) REFERENCES public.projects(id);

# -- for speed when filtering by project
# CREATE INDEX IF NOT EXISTS idx_llx_project ON ai.data_llamaindex(project_id);

db_name = "reinvent_postgres"
host = "localhost"
password = "password"
user = "user"
engine = create_engine(
    "postgresql+psycopg://user:password@localhost:5432/reinvent_postgres"
)
stmt = text("""
ALTER TABLE ai.data_llamaindex
  ADD COLUMN IF NOT EXISTS project_id INTEGER
  GENERATED ALWAYS AS ((metadata_ ->> 'project_id')::INTEGER) STORED;

ALTER TABLE ai.data_llamaindex
  DROP CONSTRAINT IF EXISTS fk_llx_project,
  ADD CONSTRAINT fk_llx_project
  FOREIGN KEY (project_id) REFERENCES public.projects(id);

CREATE INDEX IF NOT EXISTS idx_llx_project
  ON ai.data_llamaindex(project_id);
""")

SOURCE = "https://arxiv.org/pdf/2408.09869"  # Docling Technical Report for testing RAG
QUERY = "Which are the main AI models in Docling?"
API_KEY = "sk-rc-TUMbnYMGNIHY1NgEgezBcg"
TEST_PROJECT_ID = 1

EMBED_MODEL = HuggingFaceEmbedding(
    model_name="intfloat/multilingual-e5-small"
)  # Automatiaclly normalized embeddings
GEN_MODEL = OpenAILike(
    api_key=API_KEY,
    api_base="https://api.swissai.cscs.ch/v1",
    model="swissai/Apertus-8B-Instruct",  # <-- put the real model id here
    is_chat_model=True,
    # temperature=0.1,
    # timeout=120,
)
EMBED_DIM = len(EMBED_MODEL.get_text_embedding("hi"))


def init_pgvector():
    reader = DoclingReader(export_type=DoclingReader.ExportType.JSON)
    node_parser = DoclingNodeParser()

    # Setup pgvector storage
    vector_store = PGVectorStore.from_params(
        database=db_name,
        host=host,
        password=password,
        port="5432",
        user=user,
        schema_name="ai",
        table_name="llamaindex",
        embed_dim=EMBED_DIM,
        hnsw_kwargs={
            "hnsw_m": 16,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search": 40,
            "hnsw_dist_method": "vector_cosine_ops",
        },
    )
    # print(vector_store.to_dict())
    try:
        vector_store._initialize()  # private, but works
    except Exception:
        print("Failed to initialize vector store")
        pass

    # Add project_id foreign key to allow to filter for correct knowledge base
    with engine.begin() as conn:  # opens a tx and commits on success
        conn.execute(stmt)

    filters = MetadataFilters(
        filters=[ExactMatchFilter(key="project_id", value=TEST_PROJECT_ID)]
    )

    # Optionally: reset the DB
    # vector_store.clear()

    # Convert PDFs etc. into Docling-type documents
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    documents = reader.load_data(SOURCE)
    # Add project_id to each document's metadata

    for doc in documents:
        # make sure metadata exists
        doc.metadata = doc.metadata or {}
        doc.metadata["project_id"] = TEST_PROJECT_ID

    # Adds to the knowledge base
    index = VectorStoreIndex.from_documents(
        documents=documents,
        node_parser=[node_parser],
        storage_context=storage_context,
        embed_model=EMBED_MODEL,
        show_progress=True,
    )
    query_engine = index.as_query_engine(llm=GEN_MODEL, filters=filters)

    # Run query by first embedding query string and then running top-k similarity search in DB
    # Then, making API request to LLM by including found context
    start_time = time.time()
    result = query_engine.query(QUERY)
    print(f"Q: {QUERY}\nA: {result.response.strip()}\n\nSources:")
    print([(n.text, n.metadata) for n in result.source_nodes])
    end_time = time.time()
    query_duration = end_time - start_time
    print("Execution time for query: ", query_duration, " seconds")


if __name__ == "__main__":
    init_pgvector()
