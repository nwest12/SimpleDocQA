import httpx
from langchain_core.tools import tool

from app.core.config import settings


@tool
def query_sk_docs(question: str) -> str:
    """Query the Semantic Kernel documentation. Use this to look up current API patterns,
    class names, method signatures, and migration guidance. Returns a grounded answer
    with citations from the official SK docs."""
    response = httpx.post(
        f"{settings.query_service_url}/ask",
        json={"question": question},
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()["answer"]
