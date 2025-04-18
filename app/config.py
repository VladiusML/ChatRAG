import os
from typing import Any, Dict

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "RAG Service API"
    API_V1_PREFIX: str = "/api/v1"

    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "mysecretpassword")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "rag_vectorstore")
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "postgres")
    POSTGRES_PORT: str = os.getenv("POSTGRES_PORT", "5432")

    EMBEDDING_MODEL_TYPE: str = os.getenv(
        "EMBEDDING_MODEL_TYPE", "sentence_transformers"
    )
    EMBEDDING_MODEL_NAME: str = os.getenv(
        "EMBEDDING_MODEL_NAME", "sentence-transformers/all-mpnet-base-v2"
    )

    @property
    def DATABASE_URL(self) -> str:
        """Получить URL подключения к базе данных."""
        postgres_url = "postgresql://{user}:{password}@{host}:{port}/{db}".format(
            user=self.POSTGRES_USER,
            password=self.POSTGRES_PASSWORD,
            host=self.POSTGRES_HOST,
            port=self.POSTGRES_PORT,
            db=self.POSTGRES_DB,
        )
        return postgres_url

    @property
    def DATABASE_CONNECTION_CONFIG(self) -> Dict[str, Any]:
        """Получить конфигурацию подключения к базе данных."""
        return {
            "user": self.POSTGRES_USER,
            "password": self.POSTGRES_PASSWORD,
            "host": self.POSTGRES_HOST,
            "port": self.POSTGRES_PORT,
            "database": self.POSTGRES_DB,
        }

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"


settings = Settings()
