from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.dependencies import get_db, get_user, get_vectorstore_service
from app.core.logging import get_logger
from app.models import models
from app.models.models import User
from app.schemas import schemas
from app.services.vectorstore import PostgresVectorStoreService

logger = get_logger(__name__)

router = APIRouter(
    prefix="/users",
    tags=["Users"],
)


@router.post(
    "/сreate_user/", response_model=schemas.User, status_code=status.HTTP_201_CREATED
)
def create_user(
    user: schemas.UserCreate,
    db: Session = Depends(get_db),
    vectorstore_service: PostgresVectorStoreService = Depends(get_vectorstore_service),
):
    """Создать нового пользователя"""
    logger.info(f"Попытка создания пользователя с telegram_id: {user.telegram_id}")
    try:
        new_user = vectorstore_service.create_user(db, user.telegram_id)
        logger.info(f"Пользователь успешно создан с ID: {new_user.user_id}")
        return new_user
    except IntegrityError:
        logger.warning(
            f"Попытка создания пользователя с существующим telegram_id: {user.telegram_id}"
        )
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Пользователь с таким telegram_id уже существует",
        )


@router.get("/{user_id}", response_model=schemas.User)
def read_user(user: User = Depends(get_user)):
    """Получить информацию о пользователе"""
    logger.info(f"Получение информации о пользователе с ID: {user.user_id}")
    return user


@router.post(
    "/create_vectorstore/",
    response_model=schemas.VectorStore,
    status_code=status.HTTP_201_CREATED,
)
def create_vectorstore(
    vectorstore: schemas.VectorStoreCreate,
    db: Session = Depends(get_db),
    vectorstore_service: PostgresVectorStoreService = Depends(get_vectorstore_service),
    db_session: Session = Depends(get_db),
):
    """Создать новое векторное хранилище для пользователя и добвить в него документы"""
    logger.info(f"Создание векторного хранилища с именем: {vectorstore.file_name}")
    user = (
        db.query(models.User)
        .filter(models.User.telegram_id == vectorstore.telegram_id)
        .first()
    )
    new_vectorstore = vectorstore_service.create_vectorstore(
        db, user.user_id, vectorstore.file_name
    )
    metadata = {
        "file_name": vectorstore.file_name,
        "id": new_vectorstore.vectorstore_id,
    }
    vectorstore_service.add_texts(
        new_vectorstore.vectorstore_id, [vectorstore.text], [metadata]
    )
    logger.info(
        f"Векторное хранилище успешно создано с ID: {new_vectorstore.vectorstore_id}"
    )
    return new_vectorstore
