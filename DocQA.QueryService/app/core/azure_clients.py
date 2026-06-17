from functools import lru_cache

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from openai import AzureOpenAI

from app.core.config import Settings, get_settings


@lru_cache
def get_search_client() -> SearchClient:
    """
    Returns a cached SearchClient pointed at the existing index created by
    the .NET ingestion pipeline (read-only usage from this service).
    """
    settings: Settings = get_settings()
    return SearchClient(
        endpoint=settings.azure_search_endpoint,
        index_name=settings.azure_search_index_name,
        credential=AzureKeyCredential(settings.azure_search_key),
    )


@lru_cache
def get_openai_client() -> AzureOpenAI:
    """
    Returns a cached AzureOpenAI client used for generating query embeddings
    (and, later, grounded chat completions).
    """
    settings: Settings = get_settings()
    return AzureOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_key,
        api_version=settings.azure_openai_api_version,
    )