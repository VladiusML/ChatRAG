import logging

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_db, get_vectorstore_service
from app.config import settings
from app.models import models
from app.schemas import schemas
from app.services.vectorstore import PostgresVectorStoreService

router = APIRouter(prefix="/vectorstores", tags=["Vectorstores"])


async def send_to_llm_service(payload, url):
    logging.info(f"Отправка запроса в LLM-сервис: {url}")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            logging.info("Запрос успешно отправлен в LLM-сервис")
    except Exception as e:
        logging.error(f"Ошибка при отправке запроса в LLM-сервис: {str(e)}")
        raise


@router.post("/{telegram_id}/select", response_model=schemas.VectorStore)
def select_current_vectorstore(
    telegram_id: str,
    request: schemas.SelectCurrentVectorStore,
    db: Session = Depends(get_db),
):
    """Выбрать векторное хранилище по file_name и telegram_id"""
    logging.info(
        f"Выбор векторного хранилища с file_name: {request.file_name} для пользователя {telegram_id}"
    )

    # Сначала находим пользователя по telegram_id
    user = db.query(models.User).filter(models.User.telegram_id == telegram_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Пользователь с telegram_id '{telegram_id}' не найден",
        )

    # Затем находим векторное хранилище, принадлежащее этому пользователю
    vectorstore = (
        db.query(models.VectorStore)
        .filter(
            models.VectorStore.file_name == request.file_name,
            models.VectorStore.user_id == user.user_id,
        )
        .first()
    )

    if not vectorstore:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Векторное хранилище с file_name '{request.file_name}' для пользователя '{telegram_id}' не найдено",
        )

    return vectorstore


@router.post(
    "/{telegram_id}/rag_query/",
    summary="RAG: Поиск релевантных документов и отправка их в LLM-сервис",
)
async def rag_query(
    telegram_id: str,
    request: schemas.RagQueryRequest,
    background_tasks: BackgroundTasks,
    vectorstore_service: PostgresVectorStoreService = Depends(get_vectorstore_service),
    db: Session = Depends(get_db),
):
    """
    RAG (Retrieval-Augmented Generation) эндпоинт для обработки запросов пользователя.

    Процесс работы:
    1. Поиск релевантных документов в векторном хранилище пользователя
    2. Отправка найденных документов и запроса пользователя в LLM-сервис
    3. Асинхронная обработка ответа от LLM-сервиса

    Параметры:
    - telegram_id: Идентификатор пользователя Telegram
    - request: Запрос пользователя с параметрами поиска и file_name
    - background_tasks: Фоновые задачи FastAPI

    Возвращает:
    - Статус обработки запроса
    - Сообщение о текущем состоянии
    """
    # Проверяем наличие пользователя
    user = db.query(models.User).filter(models.User.telegram_id == telegram_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Пользователь с telegram_id '{telegram_id}' не найден",
        )

    if not request.file_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Не указано имя файла (file_name)",
        )

    # Проверяем, что хранилище принадлежит запрашивающему пользователю
    vectorstore = (
        db.query(models.VectorStore)
        .filter(
            models.VectorStore.file_name == request.file_name,
            models.VectorStore.user_id == user.user_id,
        )
        .first()
    )

    results = vectorstore_service.similarity_search(
        vectorstore.vectorstore_id, request.query, settings.K_RESULTS
    )
    payload = {
        "query": request.query,
        "candidates": [{"content": result["content"]} for result in results],
        "user_id": user.user_id,
        "similarity": sum(result["similarity"] for result in results) / len(results)
        if results
        else 0,
    }

    external_url = "http://llm-service/api/"
    background_tasks.add_task(send_to_llm_service, payload, external_url)
    logging.info("Задача отправки в LLM-сервис добавлена в фоновые задачи")

    return {"status": "accepted", "message": "Запрос принят в обработку"}
