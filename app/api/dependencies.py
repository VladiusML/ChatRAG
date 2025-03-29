from typing import Generator, Any
from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from langchain.embeddings import HuggingFaceEmbeddings

from config import settings
from services.vectorstore import PostgresVectorStoreService

# Настройка подключения к базе данных
engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Создаем экземпляр embedding-модели
def get_embedding_model():
    if settings.EMBEDDING_MODEL_TYPE == "sentence_transformers":
        return HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME)
    else:
        # Можно добавить другие типы моделей по необходимости
        raise ValueError(f"Unsupported embedding model type: {settings.EMBEDDING_MODEL_TYPE}")

embedding_model = get_embedding_model()

# Зависимость для получения соединения с БД
def get_db() -> Generator[Session, Any, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Зависимость для получения сервиса векторного хранилища
def get_vectorstore_service() -> PostgresVectorStoreService:
    return PostgresVectorStoreService(
        embedding_model=embedding_model,
        connection_config=settings.DATABASE_CONNECTION_CONFIG
    )

# Зависимость для проверки существования пользователя
def get_user(
    user_id: int,
    db: Session = Depends(get_db),
    vectorstore_service: PostgresVectorStoreService = Depends(get_vectorstore_service)
):
    user = vectorstore_service.get_user(db, user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found"
        )
    return user

# Зависимость для проверки существования хранилища
def get_vectorstore(
    vectorstore_id: int,
    db: Session = Depends(get_db),
    vectorstore_service: PostgresVectorStoreService = Depends(get_vectorstore_service)
):
    vectorstore = vectorstore_service.get_vectorstore(db, vectorstore_id)
    if vectorstore is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Vector store with ID {vectorstore_id} not found"
        )
    return vectorstore