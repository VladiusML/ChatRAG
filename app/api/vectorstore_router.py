from typing import List

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy.orm import Session

from app.api.dependencies import get_db, get_vectorstore, get_vectorstore_service
from app.core.logging import get_logger
from app.models import models
from app.schemas import schemas
from app.services.vectorstore import PostgresVectorStoreService

logger = get_logger(__name__)

router = APIRouter(prefix="/vectorstores", tags=["Vectorstores"])


async def send_to_llm_service(payload, url):
    logger.info(f"Отправка запроса в LLM-сервис: {url}")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            logger.info("Успешный ответ от LLM-сервиса")
            return response.json()
    except Exception as e:
        logger.error(f"Ошибка при отправке запроса в LLM-сервис: {str(e)}")
        raise


@router.get("/{vectorstore_id}", response_model=schemas.VectorStore)
def read_vectorstore(
    vectorstore: models.VectorStore = Depends(get_vectorstore),
    db: Session = Depends(get_db),
):
    """Получить информацию о векторном хранилище"""
    logger.info(
        f"Получение информации о векторном хранилище {vectorstore.vectorstore_id}"
    )
    document_count = (
        db.query(models.Document)
        .filter(models.Document.vectorstore_id == vectorstore.vectorstore_id)
        .count()
    )
    logger.info(f"Найдено {document_count} документов в хранилище")
    return {**vectorstore.__dict__, "document_count": document_count}


@router.post("/{vectorstore_id}/documents/", response_model=schemas.AddTextsResponse)
def add_texts(
    request: schemas.AddTextsRequest,
    vectorstore_id: int,
    vectorstore_service: PostgresVectorStoreService = Depends(get_vectorstore_service),
    vectorstore: models.VectorStore = Depends(get_vectorstore),
):
    """Добавить тексты в векторное хранилище"""
    logger.info(f"Добавление {len(request.texts)} текстов в хранилище {vectorstore_id}")
    doc_ids = vectorstore_service.add_texts(
        vectorstore_id, request.texts, request.metadatas
    )
    logger.info(f"Успешно добавлено {len(doc_ids)} документов")
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
    logger.info(f"Поиск по запросу '{request.query}' в хранилище {vectorstore_id}")
    results = vectorstore_service.similarity_search(
        vectorstore_id, request.query, request.k
    )
    logger.info(f"Найдено {len(results)} результатов")
    return results


@router.post(
    "/{vectorstore_id}/rag_query/",
    summary="RAG: Поиск релевантных документов и отправка их в LLM-сервис",
)
async def rag_query(
    vectorstore_id: int,
    request: schemas.SimilaritySearchRequest,
    background_tasks: BackgroundTasks,
    vectorstore_service: PostgresVectorStoreService = Depends(get_vectorstore_service),
    vectorstore: models.VectorStore = Depends(get_vectorstore),
):
    logger.info(
        f"Обработка RAG-запроса '{request.query}' для хранилища {vectorstore_id}"
    )
    results = vectorstore_service.similarity_search(
        vectorstore_id, request.query, request.k
    )
    logger.info(f"Найдено {len(results)} релевантных документов")

    payload = {
        "user_query": request.query,
        "candidates": results,
        "vectorstore_id": vectorstore_id,
    }

    external_url = "http://llm-service/api/generate_answer"
    background_tasks.add_task(send_to_llm_service, payload, external_url)
    logger.info("Задача отправки в LLM-сервис добавлена в фоновые задачи")

    return {"status": "processing", "message": "Документы отправлены в LLM-сервис"}
