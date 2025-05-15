from typing import List

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_db, get_vectorstore, get_vectorstore_service
from app.config import settings
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
            logger.info("Запрос успешно отправлен в LLM-сервис")
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
    request: schemas.RagQueryRequest,
    background_tasks: BackgroundTasks,
    vectorstore_service: PostgresVectorStoreService = Depends(get_vectorstore_service),
    vectorstore: models.VectorStore = Depends(get_vectorstore),
):
    """
    RAG (Retrieval-Augmented Generation) эндпоинт для обработки запросов пользователя.

    Процесс работы:
    1. Поиск релевантных документов в векторном хранилище
    2. Отправка найденных документов и запроса пользователя в LLM-сервис
    3. Асинхронная обработка ответа от LLM-сервиса

    Параметры:
    - vectorstore_id: ID векторного хранилища
    - request: Запрос пользователя, параметры поиска и ID пользователя
    - background_tasks: Фоновые задачи FastAPI

    Возвращает:
    - Статус обработки запроса
    - Сообщение о текущем состоянии
    """
    logger.info(
        f"Обработка RAG-запроса '{request.query}' для хранилища {vectorstore_id}"
    )

    if vectorstore.user_id != request.user_id:
        logger.warning(
            f"Попытка доступа к чужому vectorstore: user_id={request.user_id}, vectorstore_id={vectorstore_id}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="У вас нет доступа к этому векторному хранилищу",
        )

    results = vectorstore_service.similarity_search(
        vectorstore_id, request.query, request.k
    )
    logger.info(f"Найдено {len(results)} релевантных документов")

    filtered_results = [
        r for r in results if r["similarity"] >= settings.CONFIDENCE_THRESHOLD
    ]

    if not filtered_results:
        logger.info(
            f"Не найдено релевантных документов с порогом схожести >= {settings.CONFIDENCE_THRESHOLD}"
        )
        payload = {
            "user_query": request.query,
            "candidates": [],
            "vectorstore_id": vectorstore_id,
            "no_relevant_docs": True,
            "message": "Не найдено подходящей информации по вашему запросу.",
        }
    else:
        logger.info(
            f"После фильтрации найдено {len(filtered_results)} релевантных документов"
        )
        payload = {
            "user_query": request.query,
            "candidates": filtered_results,
            "vectorstore_id": vectorstore_id,
            "no_relevant_docs": False,
        }

    external_url = "http://llm-service/api/"
    background_tasks.add_task(send_to_llm_service, payload, external_url)
    logger.info("Задача отправки в LLM-сервис добавлена в фоновые задачи")

    return {"status": "accepted", "message": "Запрос принят в обработку"}
