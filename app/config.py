import os
from typing import Any, Dict, Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "RAG Service API"
    API_V1_PREFIX: str = "/api/v1"

    IS_DOCKER: bool = os.getenv("IS_DOCKER", "false").lower() == "true"

    DB_USER: str = os.getenv("DB_USER", "postgres")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "mysecretpassword")
    DB_NAME: str = os.getenv("DB_NAME", "rag_vectorstore")
    DB_HOST: str = os.getenv("DB_HOST", "postgres" if IS_DOCKER else "localhost")
    DB_PORT: str = os.getenv("DB_PORT", "5432" if IS_DOCKER else "5434")

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
    def DATABASE_URL(self) -> str:
        """Получить URL подключения к базе данных."""
        postgres_url = "postgresql://{user}:{password}@{host}:{port}/{db}".format(
            user=self.DB_USER,
            password=self.DB_PASSWORD,
            host=self.DB_HOST,
            port=self.DB_PORT,
            db=self.DB_NAME,
        )
        return postgres_url

    @property
    def DATABASE_CONNECTION_CONFIG(self) -> Dict[str, Any]:
        """Получить конфигурацию подключения к базе данных."""
        return {
            "user": self.DB_USER,
            "password": self.DB_PASSWORD,
            "host": self.DB_HOST,
            "port": self.DB_PORT,
            "database": self.DB_NAME,
        }

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"


settings = Settings()
