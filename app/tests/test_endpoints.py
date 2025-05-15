import os
import random
import string
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.main import app  # noqa: E402

client = TestClient(app)


def random_telegram_id():
    """Генерирует случайный telegram_id для тестов"""
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=10))


def test_create_user(client):
    """Тест создания нового пользователя"""
    telegram_id = random_telegram_id()
    payload = {"telegram_id": telegram_id}
    response = client.post("/api/v1/users/create_user/", json=payload)
    assert response.status_code == 201, response.text
    data = response.json()
    assert "user_id" in data
    assert data["telegram_id"] == telegram_id


def test_create_duplicate_user(client):
    """Тест создания пользователя с уже существующим telegram_id"""
    telegram_id = random_telegram_id()
    payload = {"telegram_id": telegram_id}

    response1 = client.post("/api/v1/users/create_user/", json=payload)
    assert response1.status_code == 201, response1.text

    response2 = client.post("/api/v1/users/create_user/", json=payload)
    assert response2.status_code == 400, response2.text
    data = response2.json()
    assert "Пользователь с таким telegram_id уже существует" in data["detail"]


def test_read_user(client):
    """Тест получения информации о пользователе"""
    telegram_id = random_telegram_id()
    payload = {"telegram_id": telegram_id}
    response = client.post("/api/v1/users/create_user/", json=payload)
    assert response.status_code == 201, response.text

    user = response.json()
    user_id = user["user_id"]
    assert user_id, "Ошибка: отсутствует user_id"

    response = client.get(f"/api/v1/users/{user_id}")
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["telegram_id"] == telegram_id


def test_create_vectorstore(client):
    """Тест создания векторного хранилища"""
    telegram_id = random_telegram_id()
    user_response = client.post(
        "/api/v1/users/create_user/", json={"telegram_id": telegram_id}
    )
    assert user_response.status_code == 201, user_response.text

    user = user_response.json()
    user_id = user["user_id"]
    assert user_id, "Ошибка: отсутствует user_id"

    vs_payload = {
        "file_name": "test_file.txt",
        "text": "Test content",
        "telegram_id": telegram_id,
    }
    response = client.post(
        f"/api/v1/users/{telegram_id}/create_vectorstore/", json=vs_payload
    )
    assert response.status_code == 201, response.text
    data = response.json()
    assert data["file_name"] == "test_file.txt"
    assert "vectorstore_id" in data
    assert data["user_id"] == user_id


def test_select_vectorstore(client):
    """Тест выбора векторного хранилища"""
    telegram_id = random_telegram_id()
    user_response = client.post(
        "/api/v1/users/create_user/", json={"telegram_id": telegram_id}
    )
    assert user_response.status_code == 201, user_response.text

    vs_payload = {
        "file_name": "test_file.txt",
        "text": "Test content",
        "telegram_id": telegram_id,
    }
    create_response = client.post(
        f"/api/v1/users/{telegram_id}/create_vectorstore/", json=vs_payload
    )
    assert create_response.status_code == 201, create_response.text

    select_payload = {"file_name": "test_file.txt"}
    response = client.post(
        f"/api/v1/vectorstores/{telegram_id}/select", json=select_payload
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["file_name"] == "test_file.txt"
    assert "vectorstore_id" in data
