import os
import pytest
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from vectorstore import PostgresVectorStore

load_dotenv(override=True)

DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')

@pytest.fixture
def vectorstore():
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'mps'}
    encode_kwargs = {'normalize_embeddings': False}
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        cache_folder="cache_hf"
    )  

    db_config = {
        "dbname": DB_NAME,
        "user": DB_USER,
        "password": DB_PASSWORD,
        "host": DB_HOST,
        "port": DB_PORT
    }

    return PostgresVectorStore(
        connection_config=db_config,
        embedding_model=embedding_model,
        collection_name="test_collection",
        user_id=None
    )

def test_add_and_search_texts(vectorstore):
    top_k = 2
    texts = [
        "Искусственный интеллект используется в медицине и финансах.",
        "Квантовые компьютеры могут решать сложные задачи быстрее обычных.",
        "Частные компании разрабатывают технологии для космических путешествий."
    ]
    
    metadatas = [
        {"source": "article_1", "category": "AI"},
        {"source": "article_2", "category": "Quantum Computing"},
        {"source": "article_3", "category": "Space Exploration"}
    ]

    doc_ids = vectorstore.add_texts(texts, metadatas)
    assert len(doc_ids) == len(texts), "Количество добавленных документов не совпадает"

    query = "Как работают квантовые компьютеры?"
    results = vectorstore.similarity_search(query, k=top_k)
    
    print(f"\nТоп {top_k} документов:")
    for i, doc in enumerate(results):
        print("\n")
        print(f"Документ {i+1}")
        print(doc.page_content)
        print(f"Схожесть: {doc.metadata['similarity']}")

    assert len(results) > 0, "Поиск не вернул результатов"
    assert any("квантовые компьютеры" in doc.page_content.lower() for doc in results), "В результатах отсутствует ожидаемый текст"

def test_get_all_vectorstores(vectorstore):
    all_stores = vectorstore.get_all_vectorstores_for_user(vectorstore.user_id)

    print("Хранилища:")
    for store in all_stores:
        print(f"ID: {store['vectorstore_id']}")
        print(f"Name: {store['name']}")
        print(f"Description: {store['description']}")

    assert isinstance(all_stores, list), "Возвращаемый результат должен быть списком"
