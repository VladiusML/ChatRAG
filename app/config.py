import os
from typing import Any, Dict, Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "RAG Service API"
    API_V1_PREFIX: str = "/api/v1"

    DATABASE_URL: str = os.getenv("DATABASE_URL")
    CONFIDENCE_THRESHOLD: float = 0.5
    CURRENT_VECTORSTORE_ID: Optional[int] = None
    K_RESULTS: int = 5

    EMBEDDING_MODEL_TYPE: str = os.getenv(
        "EMBEDDING_MODEL_TYPE", "sentence_transformers"
    )
    EMBEDDING_MODEL_NAME: str = os.getenv(
        "EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-large"
    )

    @property
    def DATABASE_CONNECTION_CONFIG(self) -> Dict[str, Any]:
        """Получить конфигурацию подключения к базе данных."""
        return {
            "user": self.DATABASE_URL.split("://")[1].split(":")[0],
            "password": self.DATABASE_URL.split("://")[1].split(":")[1].split("@")[0],
            "host": self.DATABASE_URL.split("@")[1].split("/")[0],
            "port": "5432",
            "database": self.DATABASE_URL.split("/")[-1],
        }

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"


settings = Settings()
