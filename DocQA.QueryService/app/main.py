from fastapi import Depends, FastAPI, HTTPException

from app.core.config import Settings, get_settings
from app.models.schemas import AskResponse, QueryRequest, QueryResponse
from app.services.generation import generate_answer
from app.services.retrieval import retrieve_chunks

app = FastAPI(
    title="Semantic Kernel Docs Query Service",
    description="Query-only RAG API over an existing Azure AI Search index of Semantic Kernel docs.",
    version="0.1.0",
)


@app.get("/health")
def health(settings: Settings = Depends(get_settings)) -> dict:
    """
    Basic liveness/config check. Confirms settings load successfully
    without exposing secret values.
    """
    return {
        "status": "ok",
        "search_index": settings.azure_search_index_name,
        "chat_deployment": settings.azure_openai_chat_deployment,
        "embedding_deployment": settings.azure_openai_embedding_deployment,
        "retrieval_top_k": settings.retrieval_top_k,
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """
    Embed the question and retrieve the top-k most relevant chunks from
    the existing Azure AI Search index.
    """
    try:
        chunks = retrieve_chunks(question=request.question, top_k=request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Retrieval failed: {exc}") from exc

    return QueryResponse(
        question=request.question,
        results=chunks,
        count=len(chunks),
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: QueryRequest) -> AskResponse:
    """
    Full RAG flow: retrieve relevant chunks, then generate a grounded
    answer using Azure OpenAI chat completion.
    """
    try:
        chunks = retrieve_chunks(question=request.question, top_k=request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Retrieval failed: {exc}") from exc

    try:
        answer = generate_answer(question=request.question, chunks=chunks)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Generation failed: {exc}") from exc

    return AskResponse(
        question=request.question,
        answer=answer,
        sources=chunks,
    )