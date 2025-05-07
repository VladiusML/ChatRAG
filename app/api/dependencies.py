from typing import Any, Generator

import torch
from fastapi import Depends, HTTPException, status
from langchain_huggingface import HuggingFaceEmbeddings
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.services.vectorstore import PostgresVectorStoreService

engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_embedding_model():
    if settings.EMBEDDING_MODEL_TYPE == "sentence_transformers":
        model_kwargs = {"device": get_device()}
        encode_kwargs = {"normalize_embeddings": False}
        embedding_model = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_NAME,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        return embedding_model
    else:
        raise ValueError(
            f"Unsupported embedding model type: {settings.EMBEDDING_MODEL_TYPE}"
        )


embedding_model = get_embedding_model()


def get_db() -> Generator[Session, Any, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_vectorstore_service() -> PostgresVectorStoreService:
    return PostgresVectorStoreService(
        embedding_model=embedding_model,
        connection_config=settings.DATABASE_CONNECTION_CONFIG,
    )


def get_user(
    user_id: int,
    db: Session = Depends(get_db),
    vectorstore_service: PostgresVectorStoreService = Depends(get_vectorstore_service),
):
    user = vectorstore_service.get_user(db, user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found",
        )
    return user


def get_vectorstore(
    vectorstore_id: int,
    db: Session = Depends(get_db),
    vectorstore_service: PostgresVectorStoreService = Depends(get_vectorstore_service),
):
    vectorstore = vectorstore_service.get_vectorstore(db, vectorstore_id)
    if vectorstore is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Vector store with ID {vectorstore_id} not found",
        )
    return vectorstore
