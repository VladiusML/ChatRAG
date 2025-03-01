import os
import psycopg2
import json
from psycopg2.extras import execute_values
import numpy as np
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores.pgvector import PGVector
from langchain.embeddings.base import Embeddings
from langchain.schema.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv
load_dotenv()
DB_HOST = os.getenv('DB_HOST')
DB_USER = os.getenv('DB_USER')
DB_NAME = os.getenv('DB_NAME')
DB_PORT = os.getenv('DB_PORT')
DB_SYSTEM = os.getenv('DB_SYSTEM')

class PostgresVectorStore:
    def __init__(
        self,
        connection_string: str,
        embedding_model: Embeddings,
        collection_name: str = "documents",
        user_id: Optional[int] = None
    ):
        """
        Инициализация векторного хранилища в PostgreSQL.
        
        Args:
            connection_string: Строка подключения к PostgreSQL
            embedding_model: Модель для создания эмбеддингов
            collection_name: Имя коллекции/хранилища
            user_id: ID пользователя, которому принадлежит хранилище
        """
        self.connection_string = connection_string
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.user_id = user_id
        self.vectorstore_id = None
        
        self._init_vectorstore()
    
    def _init_vectorstore(self):
        """Инициализирует векторное хранилище для пользователя"""
        conn = psycopg2.connect(self.connection_string)
        register_vector(conn)
        try:
            with conn.cursor() as cur:
                if self.user_id is None:
                    cur.execute(
                        "INSERT INTO users (username) VALUES (%s) RETURNING user_id",
                        (f"user_{os.urandom(4).hex()}",)
                    )
                    self.user_id = cur.fetchone()[0]
                
                cur.execute(
                    """
                    SELECT vectorstore_id FROM vectorstores 
                    WHERE user_id = %s AND name = %s
                    """,
                    (self.user_id, self.collection_name)
                )
                result = cur.fetchone()
                
                if result:
                    self.vectorstore_id = result[0]
                else:
                    cur.execute(
                        """
                        INSERT INTO vectorstores (user_id, name, description)
                        VALUES (%s, %s, %s)
                        RETURNING vectorstore_id
                        """,
                        (self.user_id, self.collection_name, f"Vectorstore for {self.collection_name}")
                    )
                    self.vectorstore_id = cur.fetchone()[0]
                
                conn.commit()
        finally:
            conn.close()
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """
        Добавляет тексты в векторное хранилище.
        
        Args:
            texts: Список текстов для добавления
            metadatas: Метаданные для каждого текста
            
        Returns:
            Список идентификаторов добавленных документов
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        embeddings = self.embedding_model.embed_documents(texts)
        conn = psycopg2.connect(self.connection_string)
        register_vector(conn)
        try:
            with conn.cursor() as cur:
                data = [
                    (self.vectorstore_id, text, json.dumps(metadata), embedding)
                    for text, metadata, embedding in zip(texts, metadatas, embeddings)
                ]
                
                query = """
                INSERT INTO documents (vectorstore_id, content, metadata, embedding)
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
        query: str, 
        k: int = 4,
    ) -> List[Document]:
        """
        Выполняет поиск по сходству в векторном хранилище.
        
        Args:
            query: Текст запроса
            k: Количество результатов для возврата
            
        Returns:
            Список Document с результатами поиска
        """
        query_embedding = self.embedding_model.embed_query(query)
        conn = psycopg2.connect(self.connection_string)
        register_vector(conn)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT doc_id, content, metadata, 
                        1 - (embedding <=> %s::vector) as similarity
                    FROM documents
                    WHERE vectorstore_id = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (query_embedding, self.vectorstore_id, 
                    query_embedding, k)
                )
                
                results = []
                for doc_id, content, metadata_str, similarity in cur.fetchall():
                    metadata = metadata_str if isinstance(metadata_str, dict) else json.loads(metadata_str)
                    metadata["doc_id"] = doc_id
                    metadata["similarity"] = similarity
                    
                    results.append(
                        Document(
                            page_content=content,
                            metadata=metadata
                        )
                    )
                
                return results
        finally:
            conn.close()
    
    def get_all_vectorstores_for_user(self, user_id: int) -> List[Dict[str, Any]]:
        """
        Получает все векторные хранилища для указанного пользователя.
        
        Args:
            user_id: ID пользователя
            
        Returns:
            Список словарей с информацией о хранилищах
        """
        conn = psycopg2.connect(self.connection_string)
        register_vector(conn)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT vectorstore_id, name, description, created_at
                    FROM vectorstores
                    WHERE user_id = %s
                    """,
                    (user_id,)
                )
                
                stores = []
                for vectorstore_id, name, description, created_at in cur.fetchall():
                    cur.execute(
                        "SELECT COUNT(*) FROM documents WHERE vectorstore_id = %s",
                        (vectorstore_id,)
                    )
                    doc_count = cur.fetchone()[0]
                    
                    stores.append({
                        "vectorstore_id": vectorstore_id,
                        "name": name,
                        "description": description,
                        "created_at": created_at,
                        "document_count": doc_count
                    })
                
                return stores
        finally:
            conn.close()

def example_usage():
    connection_string = f"{DB_SYSTEM}://{DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}" 

    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'mps'}
    encode_kwargs = {'normalize_embeddings': False}
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        cache_folder = "cache_hf"
    )    

    vectorstore = PostgresVectorStore(
        connection_string=connection_string,
        embedding_model=embedding_model,
        collection_name="my_collection",
        user_id=None
    )
    
    texts = [
        "Архитектура RAG (Retrieval-Augmented Generation) объединяет поиск и генерацию.",
        "PostgreSQL с расширением pgvector позволяет эффективно хранить и искать векторные представления.",
        "Микросервисная архитектура повышает масштабируемость и отказоустойчивость системы."
    ]
    
    metadatas = [
        {"source": "article_1", "category": "architecture"},
        {"source": "article_2", "category": "database"},
        {"source": "article_3", "category": "design_patterns"}
    ]
    
    doc_ids = vectorstore.add_texts(texts, metadatas)
    print(f"Добавлено {len(doc_ids)} документов")
    
    query = "Как хранить векторные представления в базе данных?"
    results = vectorstore.similarity_search(query, k=2)
    
    for doc in results:
        print(f"Сходство: {doc.metadata['similarity']:.4f}")
        print(f"Содержание: {doc.page_content}")
        print(f"Метаданные: {doc.metadata}")
        print("---")
    
    all_stores = vectorstore.get_all_vectorstores_for_user(vectorstore.user_id)
    print(f"Пользователь имеет {len(all_stores)} векторных хранилищ")
    for store in all_stores:
        print(f"- {store['name']}: {store['document_count']} документов")

if __name__ == "__main__":
    example_usage()