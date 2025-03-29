import sys
import os
import pytest
import random
import string
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main import app

client = TestClient(app)

def random_username():
    """Генерирует случайный username для тестов"""
    return "testuser_" + "".join(random.choices(string.ascii_lowercase, k=5))

def test_create_user():
    """Тест создания нового пользователя"""
    username = random_username()
    payload = {"username": username}
    response = client.post("/api/v1/users/", json=payload)
    assert response.status_code == 201, response.text
    data = response.json()
    assert "user_id" in data or "user_id" in data
    assert data["username"] == username

def test_create_duplicate_user():
    """Тест создания пользователя с уже существующим username"""
    username = random_username()
    payload = {"username": username}
    
    response1 = client.post("/api/v1/users/", json=payload)
    assert response1.status_code == 201, response1.text

    response2 = client.post("/api/v1/users/", json=payload)
    assert response2.status_code == 400, response2.text
    data = response2.json()
    assert "Пользователь с таким username уже существует" in data["detail"]

def test_read_user():
    """Тест получения информации о пользователе"""
    username = random_username()
    payload = {"username": username}
    response = client.post("/api/v1/users/", json=payload)
    assert response.status_code == 201, response.text

    user = response.json()
    user_id = user.get("user_id") or user.get("id")
    assert user_id, "Ошибка: отсутствует user_id"

    response = client.get(f"/api/v1/users/{user_id}")
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["username"] == username

def test_create_vectorstore():
    """Тест создания векторного хранилища"""
    username = random_username()
    user_response = client.post("/api/v1/users/", json={"username": username})
    assert user_response.status_code == 201, user_response.text

    user = user_response.json()
    user_id = user.get("user_id") or user.get("id")
    assert user_id, "Ошибка: отсутствует user_id"

    vs_payload = {"name": "test_vectorstore", "description": "Test description"}
    response = client.post(f"/api/v1/users/{user_id}/vectorstores/", json=vs_payload)
    assert response.status_code == 201, response.text
    data = response.json()
    assert data["name"] == "test_vectorstore"
    assert "vectorstore_id" in data, "Ошибка: отсутствует vectorstore_id"

def test_read_vectorstores():
    """Тест получения списка векторных хранилищ пользователя"""
    username = random_username()
    user_response = client.post("/api/v1/users/", json={"username": username})
    assert user_response.status_code == 201, user_response.text

    user = user_response.json()
    user_id = user.get("user_id") or user.get("id")
    assert user_id, "Ошибка: отсутствует user_id"

    vs_payload = {"name": "test_vectorstore2", "description": "Test desc"}
    client.post(f"/api/v1/users/{user_id}/vectorstores/", json=vs_payload)

    response = client.get(f"/api/v1/users/{user_id}/vectorstores/")
    assert response.status_code == 200, response.text
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1

def test_vectorstore_documents_flow():
    """Тест: получение информации о векторном хранилище, добавление текстов и поиск по сходству"""
    username = random_username()
    user_response = client.post("/api/v1/users/", json={"username": username})
    assert user_response.status_code == 201, user_response.text

    user = user_response.json()
    user_id = user.get("user_id") or user.get("id")
    assert user_id, "Ошибка: отсутствует user_id"

    vs_payload = {"name": "test_vectorstore3", "description": "Test desc 3"}
    vectorstore_response = client.post(f"/api/v1/users/{user_id}/vectorstores/", json=vs_payload)
    assert vectorstore_response.status_code == 201, vectorstore_response.text

    vs_data = vectorstore_response.json()
    vectorstore_id = vs_data.get("vectorstore_id")
    assert vectorstore_id, "Ошибка: отсутствует vectorstore_id"

    response = client.get(f"/api/v1/vectorstores/{vectorstore_id}")
    assert response.status_code == 200, response.text
    data = response.json()
    assert "document_count" in data

    add_texts_payload = {
        "texts": ["Hello world", "Test text"],
        "metadatas": [{"lang": "en"}, {"lang": "en"}]
    }
    response = client.post(f"/api/v1/vectorstores/{vectorstore_id}/documents/", json=add_texts_payload)
    assert response.status_code == 200, response.text
    add_texts_data = response.json()
    assert "doc_ids" in add_texts_data
    assert isinstance(add_texts_data["doc_ids"], list)

    search_payload = {"query": "Hello", "k": 1}
    response = client.post(f"/api/v1/vectorstores/{vectorstore_id}/search/", json=search_payload)
    assert response.status_code == 200, response.text
    search_results = response.json()
    assert isinstance(search_results, list)
