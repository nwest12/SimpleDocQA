from azure.search.documents.models import VectorizedQuery

from app.core.azure_clients import get_openai_client, get_search_client
from app.core.config import Settings, get_settings
from app.models.schemas import RetrievedChunk

# --- Index field names ---
# These must match the field names used by DocQA.Ingestion when the index
# was created. Adjust these constants if your index schema differs.
FIELD_ID = "id"
FIELD_CONTENT = "content"
FIELD_SOURCE = "sourceFile"
FIELD_VECTOR = "contentVector"


def embed_query(question: str) -> list[float]:
    """Generate an embedding vector for the user's question via Azure OpenAI."""
    settings: Settings = get_settings()
    client = get_openai_client()

    response = client.embeddings.create(
        model=settings.azure_openai_embedding_deployment,
        input=question,
    )
    return response.data[0].embedding


def retrieve_chunks(question: str, top_k: int | None = None) -> list[RetrievedChunk]:
    """
    Embed the question and run a vector similarity search against the
    existing Azure AI Search index, returning the top-k matching chunks.
    """
    settings: Settings = get_settings()
    k = top_k or settings.retrieval_top_k

    query_vector = embed_query(question)

    search_client = get_search_client()
    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=k,
        fields=FIELD_VECTOR,
    )

    results = search_client.search(
        search_text=None,
        vector_queries=[vector_query],
        select=[FIELD_ID, FIELD_CONTENT, FIELD_SOURCE],
        top=k,
    )

    chunks: list[RetrievedChunk] = []
    for result in results:
        chunks.append(
            RetrievedChunk(
                id=result[FIELD_ID],
                content=result[FIELD_CONTENT],
                source=result.get(FIELD_SOURCE),
                score=result["@search.score"],
            )
        )

    return chunks