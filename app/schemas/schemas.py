from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class UserBase(BaseModel):
    username: str


class UserCreate(UserBase):
    pass


class User(UserBase):
    user_id: int

    class Config:
        from_attributes = True


class VectorStoreBase(BaseModel):
    name: str
    description: Optional[str] = None


class VectorStoreCreate(VectorStoreBase):
    pass


class VectorStore(VectorStoreBase):
    vectorstore_id: int
    user_id: int
    created_at: datetime
    document_count: Optional[int] = 0

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
