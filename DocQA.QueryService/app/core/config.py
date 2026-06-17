from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Configuration for the query service, loaded from environment variables
    and/or a .env file. Mirrors the values used by the .NET ingestion
    pipeline so this service can query the same Azure AI Search index.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        extra="ignore",
    )

    # --- Azure AI Search ---
    azure_search_endpoint: str = Field(
        ...,
        description="e.g. https://<your-search-service>.search.windows.net",
    )
    azure_search_key: str = Field(..., description="Admin or query API key")
    azure_search_index_name: str = Field(
        ..., description="Name of the existing index created by the .NET ingestion pipeline"
    )

    # --- Azure OpenAI ---
    azure_openai_endpoint: str = Field(
        ..., description="e.g. https://<your-resource>.openai.azure.com"
    )
    azure_openai_key: str = Field(...)
    azure_openai_api_version: str = Field(default="2024-10-21")
    azure_openai_chat_deployment: str = Field(
        ..., description="Deployment name for the chat/completion model"
    )
    azure_openai_embedding_deployment: str = Field(
        ..., description="Deployment name for the embedding model"
    )

    # --- Service behavior ---
    retrieval_top_k: int = Field(default=5, description="Number of chunks to retrieve per query")
    log_level: str = Field(default="INFO")


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor; use as a FastAPI dependency."""
    return Settings()