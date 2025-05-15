from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.dependencies import get_db, get_user, get_vectorstore_service
from app.core.logging import get_logger
from app.models import models
from app.schemas import schemas
from app.services.vectorstore import PostgresVectorStoreService

logger = get_logger(__name__)

router = APIRouter(
    prefix="/users",
    tags=["Users"],
)


@router.post(
    "/create_user/", response_model=schemas.User, status_code=status.HTTP_201_CREATED
)
def create_user(
    request: schemas.UserCreate,
    db: Session = Depends(get_db),
    vectorstore_service: PostgresVectorStoreService = Depends(get_vectorstore_service),
):
    """Создать нового пользователя"""
    logger.info(f"Попытка создания пользователя с telegram_id: {request.telegram_id}")
    try:
        new_user = vectorstore_service.create_user(db, request.telegram_id)
        logger.info(f"Пользователь успешно создан с ID: {new_user.user_id}")
        return new_user
    except IntegrityError:
        logger.warning(
            f"Попытка создания пользователя с существующим telegram_id: {request.telegram_id}"
        )
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Пользователь с таким telegram_id уже существует",
        )


@router.get("/{user_id}", response_model=schemas.User)
def read_user(request: schemas.User = Depends(get_user)):
    """Получить информацию о пользователе"""
    logger.info(f"Получение информации о пользователе с ID: {request.user_id}")
    return request


@router.post(
    "/{telegram_id}/create_vectorstore/",
    response_model=schemas.VectorStore,
    status_code=status.HTTP_201_CREATED,
)
def create_vectorstore(
    telegram_id: str,
    request: schemas.VectorStoreCreate,
    db: Session = Depends(get_db),
    vectorstore_service: PostgresVectorStoreService = Depends(get_vectorstore_service),
):
    """Создать новое векторное хранилище для пользователя и добвить в него документы"""
    logger.info(
        f"Создание векторного хранилища с именем: {request.file_name} для пользователя с telegram_id: {telegram_id}"
    )
    user = db.query(models.User).filter(models.User.telegram_id == telegram_id).first()
    if not user:
        logger.warning(f"Пользователь с telegram_id {telegram_id} не найден")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Пользователь с telegram_id {telegram_id} не найден",
        )
    new_vectorstore = vectorstore_service.create_vectorstore(
        db, user.user_id, request.file_name
    )
    metadata = {
        "file_name": request.file_name,
        "id": new_vectorstore.vectorstore_id,
    }
    vectorstore_service.add_texts(
        new_vectorstore.vectorstore_id, [request.text], [metadata]
    )
    logger.info(
        f"Векторное хранилище успешно создано с ID: {new_vectorstore.vectorstore_id}"
    )
    return new_vectorstore
