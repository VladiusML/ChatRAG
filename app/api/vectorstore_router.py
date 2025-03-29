from typing import List

from api.dependencies import get_db, get_vectorstore, get_vectorstore_service
from fastapi import APIRouter, Depends
from models import models
from schemas import schemas
from services.vectorstore import PostgresVectorStoreService
from sqlalchemy.orm import Session

router = APIRouter(prefix="/vectorstores", tags=["Vectorstores"])


@router.get("/{vectorstore_id}", response_model=schemas.VectorStore)
def read_vectorstore(
    vectorstore: models.VectorStore = Depends(get_vectorstore),
    db: Session = Depends(get_db),
):
    """Получить информацию о векторном хранилище"""
    document_count = (
        db.query(models.Document)
        .filter(models.Document.vectorstore_id == vectorstore.vectorstore_id)
        .count()
    )
    return {**vectorstore.__dict__, "document_count": document_count}


@router.post("/{vectorstore_id}/documents/", response_model=schemas.AddTextsResponse)
def add_texts(
    request: schemas.AddTextsRequest,
    vectorstore_id: int,
    vectorstore_service: PostgresVectorStoreService = Depends(get_vectorstore_service),
    vectorstore: models.VectorStore = Depends(get_vectorstore),
):
    """Добавить тексты в векторное хранилище"""
    doc_ids = vectorstore_service.add_texts(
        vectorstore_id, request.texts, request.metadatas
    )
    return {"doc_ids": doc_ids}


@router.post(
    "/{vectorstore_id}/search/", response_model=List[schemas.SimilaritySearchResult]
)
def similarity_search(
    request: schemas.SimilaritySearchRequest,
    vectorstore_id: int,
    vectorstore_service: PostgresVectorStoreService = Depends(get_vectorstore_service),
    vectorstore: models.VectorStore = Depends(get_vectorstore),
):
    """Выполнить поиск по сходству в векторном хранилище"""
    results = vectorstore_service.similarity_search(
        vectorstore_id, request.query, request.k
    )
    return results
