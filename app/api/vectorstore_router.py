import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_db, get_vectorstore_service
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


@router.post("/select", response_model=schemas.VectorStore)
def select_current_vectorstore(
    request: schemas.SelectCurrentVectorStore, db: Session = Depends(get_db)
):
    """Выбрать векторное хранилище по file_name"""
    logger.info(f"Выбор векторного хранилища с file_name: {request.file_name}")
    vectorstore = (
        db.query(models.VectorStore)
        .filter(models.VectorStore.file_name == request.file_name)
        .first()
    )

    if not vectorstore:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Векторное хранилище с file_name '{request.file_name}' не найдено",
        )

    settings.CURRENT_VECTORSTORE_ID = vectorstore.vectorstore_id
    logger.info(f"Текущее векторное хранилище: {settings.CURRENT_VECTORSTORE_ID}")
    return vectorstore


@router.post(
    "/rag_query/",
    summary="RAG: Поиск релевантных документов и отправка их в LLM-сервис",
)
async def rag_query(
    request: schemas.RagQueryRequest,
    background_tasks: BackgroundTasks,
    vectorstore_service: PostgresVectorStoreService = Depends(get_vectorstore_service),
    db: Session = Depends(get_db),
):
    """
    RAG (Retrieval-Augmented Generation) эндпоинт для обработки запросов пользователя.

    Процесс работы:
    1. Поиск релевантных документов в векторном хранилище
    2. Отправка найденных документов и запроса пользователя в LLM-сервис
    3. Асинхронная обработка ответа от LLM-сервиса

    Параметры:
    - request: Запрос пользователя, параметры поиска
    - background_tasks: Фоновые задачи FastAPI

    Возвращает:
    - Статус обработки запроса
    - Сообщение о текущем состоянии
    """
    if settings.CURRENT_VECTORSTORE_ID is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Не выбрано текущее векторное хранилище. Используйте /vectorstores/select для выбора хранилища.",
        )

    vectorstore_id = settings.CURRENT_VECTORSTORE_ID
    vectorstore = (
        db.query(models.VectorStore)
        .filter(models.VectorStore.vectorstore_id == vectorstore_id)
        .first()
    )

    if not vectorstore:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Выбранное векторное хранилище не найдено",
        )

    user_id = vectorstore.user_id

    results = vectorstore_service.similarity_search(
        vectorstore_id, request.query, settings.K_RESULTS
    )
    payload = {
        "query": request.query,
        "candidates": [{"content": result["content"]} for result in results],
        "user_id": user_id,
        "similarity": sum(result["similarity"] for result in results) / len(results)
        if results
        else 0,
    }

    external_url = "http://llm-service/api/"
    background_tasks.add_task(send_to_llm_service, payload, external_url)
    logger.info("Задача отправки в LLM-сервис добавлена в фоновые задачи")

    return {"status": "accepted", "message": "Запрос принят в обработку"}
