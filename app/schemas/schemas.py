from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class UserBase(BaseModel):
    telegram_id: str


class UserCreate(UserBase):
    pass


class User(UserBase):
    user_id: int

    class Config:
        from_attributes = True


class VectorStoreBase(BaseModel):
    file_name: str
    text: str
    telegram_id: str


class VectorStoreCreate(VectorStoreBase):
    pass


class VectorStore(VectorStoreBase):
    vectorstore_id: int
    user_id: int
    created_at: datetime
    document_count: Optional[int] = 0
    text: Optional[str] = None

    class Config:
        from_attributes = True


class DocumentBase(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class DocumentCreate(DocumentBase):
    pass


class Document(DocumentBase):
    doc_id: int
    vectorstore_id: int
    created_at: datetime

    class Config:
        from_attributes = True


class SimilaritySearchRequest(BaseModel):
    query: str
    k: int = 4


class SimilaritySearchResult(BaseModel):
    doc_id: int
    content: str
    metadata: Dict[str, Any]
    similarity: float


class AddTextsRequest(BaseModel):
    texts: List[str]
    metadatas: Optional[List[Dict[str, Any]]] = None


class AddTextsResponse(BaseModel):
    doc_ids: List[str]


class ErrorResponse(BaseModel):
    detail: str


class RagQueryResponse(BaseModel):
    """Схема ответа RAG эндпоинта"""

    status: str = Field(..., description="Статус обработки запроса")
    message: str = Field(..., description="Сообщение о текущем состоянии")


class RagQueryRequest(BaseModel):
    """Схема запроса RAG эндпоинта"""

    query: str = Field(..., description="Запрос пользователя")
    file_name: str = Field(..., description="Имя файла векторного хранилища для поиска")


class SelectCurrentVectorStore(BaseModel):
    file_name: str = Field(..., description="Имя файла векторного хранилища")
