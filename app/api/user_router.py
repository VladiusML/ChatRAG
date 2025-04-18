from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.dependencies import get_db, get_user, get_vectorstore_service
from app.models.models import User
from app.schemas import schemas
from app.services.vectorstore import PostgresVectorStoreService

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
    try:
        return vectorstore_service.create_user(db, user.username)
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Пользователь с таким username уже существует",
        )


@router.get("/{user_id}", response_model=schemas.User)
def read_user(user: User = Depends(get_user)):
    """Получить информацию о пользователе"""
    return user


@router.post(
    "/{user_id}/vectorstores/",
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
    """Создать новое векторное хранилище для пользователя"""
    return vectorstore_service.create_vectorstore(
        db, user_id, vectorstore.name, vectorstore.description
    )


@router.get("/{user_id}/vectorstores/", response_model=List[schemas.VectorStore])
def read_vectorstores(
    user_id: int,
    db: Session = Depends(get_db),
    vectorstore_service: PostgresVectorStoreService = Depends(get_vectorstore_service),
    user: User = Depends(get_user),
):
    """Получить все векторные хранилища пользователя"""
    return vectorstore_service.get_vectorstores_for_user(db, user_id)
