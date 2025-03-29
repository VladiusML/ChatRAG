import json
from typing import Any, Dict, List, Optional

import psycopg2
from config import settings
from langchain.embeddings.base import Embeddings
from models.models import Document as DBDocument
from models.models import User, VectorStore
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values
from sqlalchemy.orm import Session


class PostgresVectorStoreService:
    def __init__(
        self,
        embedding_model: Embeddings,
        connection_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Инициализация сервиса векторного хранилища в PostgreSQL.

        Args:
            embedding_model: Модель для создания эмбеддингов
            connection_config: Конфигурация подключения к PostgreSQL
        """
        self.connection_config = (
            connection_config or settings.DATABASE_CONNECTION_CONFIG
        )
        self.embedding_model = embedding_model

    def create_user(self, db: Session, username: str) -> User:
        """
        Создает нового пользователя.

        Args:
            db: Сессия SQLAlchemy
            username: Имя пользователя

        Returns:
            Объект User
        """
        user = User(username=username)
        db.add(user)
        db.commit()
        db.refresh(user)
        return user

    def get_user(self, db: Session, user_id: int) -> Optional[User]:
        """
        Получает пользователя по ID.

        Args:
            db: Сессия SQLAlchemy
            user_id: ID пользователя

        Returns:
            Объект User или None
        """
        return db.query(User).filter(User.user_id == user_id).first()

    def create_vectorstore(
        self, db: Session, user_id: int, name: str, description: Optional[str] = None
    ) -> VectorStore:
        """
        Создает новое векторное хранилище для пользователя.

        Args:
            db: Сессия SQLAlchemy
            user_id: ID пользователя
            name: Имя хранилища
            description: Описание хранилища

        Returns:
            Объект VectorStore
        """
        vectorstore = VectorStore(
            user_id=user_id,
            name=name,
            description=description or f"Vectorstore for {name}",
        )
        db.add(vectorstore)
        db.commit()
        db.refresh(vectorstore)
        return vectorstore

    def get_vectorstore(
        self, db: Session, vectorstore_id: int
    ) -> Optional[VectorStore]:
        """
        Получает хранилище по ID.

        Args:
            db: Сессия SQLAlchemy
            vectorstore_id: ID хранилища

        Returns:
            Объект VectorStore или None
        """
        return (
            db.query(VectorStore)
            .filter(VectorStore.vectorstore_id == vectorstore_id)
            .first()
        )

    def get_vectorstores_for_user(
        self, db: Session, user_id: int
    ) -> List[Dict[str, Any]]:
        """
        Получает все векторные хранилища для указанного пользователя.

        Args:
            db: Сессия SQLAlchemy
            user_id: ID пользователя

        Returns:
            Список словарей с информацией о хранилищах
        """
        stores = []
        vectorstores = (
            db.query(VectorStore).filter(VectorStore.user_id == user_id).all()
        )

        for vs in vectorstores:
            document_count = (
                db.query(DBDocument)
                .filter(DBDocument.vectorstore_id == vs.vectorstore_id)
                .count()
            )
            stores.append(
                {
                    "vectorstore_id": vs.vectorstore_id,
                    "name": vs.name,
                    "description": vs.description,
                    "user_id": vs.user_id,
                    "created_at": vs.created_at,
                    "document_count": document_count,
                }
            )

        return stores

    def add_texts(
        self,
        vectorstore_id: int,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """
        Добавляет тексты в векторное хранилище.

        Args:
            vectorstore_id: ID хранилища
            texts: Список текстов для добавления
            metadatas: Метаданные для каждого текста

        Returns:
            Список идентификаторов добавленных документов
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]

        embeddings = self.embedding_model.embed_documents(texts)
        conn = psycopg2.connect(**self.connection_config)
        register_vector(conn)
        try:
            with conn.cursor() as cur:
                data = [
                    (vectorstore_id, text, json.dumps(doc_metadata), embedding)
                    for text, doc_metadata, embedding in zip(
                        texts, metadatas, embeddings
                    )
                ]

                query = """
                INSERT INTO documents (vectorstore_id, content, doc_metadata, embedding)
                VALUES %s
                RETURNING doc_id
                """
                template = "(%s, %s, %s, %s)"

                doc_ids = execute_values(cur, query, data, template, fetch=True)
                conn.commit()

                return [str(id[0]) for id in doc_ids]
        finally:
            conn.close()

    def similarity_search(
        self,
        vectorstore_id: int,
        query: str,
        k: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Выполняет поиск по сходству в векторном хранилище.

        Args:
            vectorstore_id: ID хранилища
            query: Текст запроса
            k: Количество результатов для возврата

        Returns:
            Список результатов поиска
        """
        query_embedding = self.embedding_model.embed_query(query)
        conn = psycopg2.connect(**self.connection_config)
        register_vector(conn)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT doc_id, content, doc_metadata,
                        1 - (embedding <=> %s::vector) as similarity
                    FROM documents
                    WHERE vectorstore_id = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (query_embedding, vectorstore_id, query_embedding, k),
                )

                results = []
                for doc_id, content, metadata_str, similarity in cur.fetchall():
                    metadata = (
                        metadata_str
                        if isinstance(metadata_str, dict)
                        else json.loads(metadata_str)
                    )

                    results.append(
                        {
                            "doc_id": doc_id,
                            "content": content,
                            "metadata": metadata,
                            "similarity": similarity,
                        }
                    )

                return results
        finally:
            conn.close()
