from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.dependencies import get_db, get_user, get_vectorstore_service
from app.core.logging import get_logger
from app.models.models import User
from app.schemas import schemas
from app.services.vectorstore import PostgresVectorStoreService

logger = get_logger(__name__)

router = APIRouter(
    prefix="/users",
    tags=["Users"],
)


@router.post("/", response_model=schemas.User, status_code=status.HTTP_201_CREATED)
def create_user(
    user: schemas.UserCreate,
    db: Session = Depends(get_db),
    vectorstore_service: PostgresVectorStoreService = Depends(get_vectorstore_service),
):
    """Создать нового пользователя"""
    logger.info(f"Попытка создания пользователя с username: {user.username}")
    try:
        new_user = vectorstore_service.create_user(db, user.username)
        logger.info(f"Пользователь успешно создан с ID: {new_user.user_id}")
        return new_user
    except IntegrityError:
        logger.warning(
            f"Попытка создания пользователя с существующим username: {user.username}"
        )
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Пользователь с таким username уже существует",
        )


@router.get("/{user_id}", response_model=schemas.User)
def read_user(user: User = Depends(get_user)):
    """Получить информацию о пользователе"""
    logger.info(f"Получение информации о пользователе с ID: {user.user_id}")
    return user


@router.post(
    "/{user_id}/create_vectorstore/",
    response_model=schemas.VectorStore,
    status_code=status.HTTP_201_CREATED,
)
def create_vectorstore(
    vectorstore: schemas.VectorStoreCreate,
    user_id: int,
    db: Session = Depends(get_db),
    vectorstore_service: PostgresVectorStoreService = Depends(get_vectorstore_service),
    user: User = Depends(get_user),
):
    """Создать новое векторное хранилище для пользователя и добвить в него документы"""
    logger.info(
        f"Создание векторного хранилища для пользователя {user_id} с именем: {vectorstore.file_name}"
    )
    new_vectorstore = vectorstore_service.create_vectorstore(
        db, user_id, vectorstore.file_name
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
