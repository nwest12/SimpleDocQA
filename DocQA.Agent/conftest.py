import os

# Set required env vars before any app import so pydantic-settings validation
# passes in environments without a .env file (e.g., CI).
os.environ.update(
    {
        "AZURE_OPENAI_ENDPOINT": "https://fake.openai.azure.com",
        "AZURE_OPENAI_KEY": "fake-key",
        "AZURE_OPENAI_CHAT_DEPLOYMENT": "gpt-4o-mini",
    }
)
