from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    azure_openai_endpoint: str
    azure_openai_key: str
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_chat_deployment: str

    query_service_url: str = "http://localhost:8000"

    @property
    def azure_chat_kwargs(self) -> dict:
        return {
            "azure_endpoint": self.azure_openai_endpoint,
            "api_key": self.azure_openai_key,
            "api_version": self.azure_openai_api_version,
            "azure_deployment": self.azure_openai_chat_deployment,
        }


settings = Settings()
