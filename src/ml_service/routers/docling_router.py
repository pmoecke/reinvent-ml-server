import time
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Path,
    Query,
    Request,
    Response,
    status,
)
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.readers.docling import DoclingReader
from llama_index.vector_stores.postgres import PGVectorStore

from ml_service.dependencies import get_embedder, get_llm_model, get_vector_store, require_bearer
from ml_service.models.docling_models import QueryRequest

router = APIRouter(prefix="/docling", tags=["rag"], dependencies=[Depends(require_bearer)])
reader = DoclingReader(export_type=DoclingReader.ExportType.JSON)
node_parser = DoclingNodeParser()
SOURCE = "https://arxiv.org/pdf/2408.09869"


@router.get("/knowledge-base/{project_id}/status")
async def get_knowledge_base_status(project_id: Annotated[int, Path(ge=1)]):
    """
    todo
    """
    pass


@router.post("/knowledge-base/{project_id}")
async def create_knowledge_base(
    request: Request,
    project_id: Annotated[int, Path(ge=1)],
    vector_store: PGVectorStore = Depends(get_vector_store),
    embed_model: HuggingFaceEmbedding = Depends(get_embedder),
):
    """
    Build/refresh the KB for a project by:
      1) loading documents (Docling),
      2) chunking (DoclingNodeParser),
      3) embedding + upserting into pgvector (via PGVectorStore).
    Blocking for MVP; single-worker ensures one-at-a-time.
    """
    # Convert PDFs etc. into Docling-type documents
    t0 = time.perf_counter()

    # Replace the below by extracting documents belonging to `project_id` from MinIO
    documents = reader.load_data(SOURCE)

    # Add project_id to each document's metadata
    for doc in documents:
        # make sure metadata exists
        doc.metadata = doc.metadata or {}
        doc.metadata["project_id"] = project_id

    # Adds to the knowledge base
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    _ = VectorStoreIndex.from_documents(
        documents=documents,
        node_parser=[node_parser],
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=False,
    )
    dt = time.perf_counter() - t0
    # Update index ??
    # request.app.state.index = index
    return {"project_id": project_id, "status": "READY", "ingest_seconds": round(dt, 2)}


@router.post("/knowledge-base/{project_id}/query")
async def query_knowledge_base(
    project_id: Annotated[int, Path(ge=1)],
    body: QueryRequest,
    top_k: Annotated[int, Query(ge=1)] = 5,
    gen_model=Depends(get_llm_model),
    vector_store=Depends(get_vector_store),
    embed_model=Depends(get_embedder),
):
    """
    todo
    """
    # Create a query-only view over the existing store (cheap; no re-embedding)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
        top_k=top_k,
    )
    filters = MetadataFilters(
        filters=[ExactMatchFilter(key="project_id", value=project_id)]
    )

    t0 = time.perf_counter()
    query_engine = index.as_query_engine(llm=gen_model, filters=filters)
    try:
        result = query_engine.query(body.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query failed: {e}")
    dt = time.perf_counter() - t0

    sources = [
        {
            "score": getattr(n, "score", None),
            "snippet": n.text[:500],
            "metadata": n.metadata,
        }
        for n in (result.source_nodes or [])[:5]
    ]

    response = result.response  # type: ignore
    return {
        "project_id": project_id,
        "answer": response,
        "latency_seconds": round(dt, 2),
        "sources": sources,
    }


@router.delete("")
async def delete_all_embeddings(
    vector_store: PGVectorStore = Depends(get_vector_store),
):
    """
    todo
    """
    vector_store.clear()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
