from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # API Keys for external services
    GROQ_API_KEY: str
    COHERE_API_KEY: str
    LLM_MODEL_NAME: str
    EMBEDDING_MODEL_NAME: str
    APP_TITLE: str
    APP_HOST: str
    APP_PORT: int

    class Config:
        env_file = ".env"


def get_settings() -> Settings:
    return Settings()
