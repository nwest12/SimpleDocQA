import os

# Set required env vars before any app import so pydantic-settings validation
# passes in environments without a .env file (e.g., CI). Env vars take
# precedence over .env, so local runs with a real .env are unaffected.
os.environ.update(
    {
        "AZURE_SEARCH_ENDPOINT": "https://fake.search.windows.net",
        "AZURE_SEARCH_KEY": "fake-key",
        "AZURE_SEARCH_INDEX_NAME": "fake-index",
        "AZURE_OPENAI_ENDPOINT": "https://fake.openai.azure.com",
        "AZURE_OPENAI_KEY": "fake-key",
        "AZURE_OPENAI_API_VERSION": "2024-10-21",
        "AZURE_OPENAI_CHAT_DEPLOYMENT": "gpt-4o-mini",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "text-embedding-3-small",
    }
)
