---
title: ChatRAG
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
sdk_version: 3.0.0
app_file: app/main.py
pinned: false
---

# ChatRAG

Микросервис vectorstore для проекта по ПИУС

## Описание

Сервис для работы с векторным хранилищем на основе PostgreSQL и pgvector, интегрированный с HuggingFace моделями для эмбеддингов.

## Технологии

- FastAPI
- PostgreSQL + pgvector
- HuggingFace Transformers
- Docker

## Деплой

Проект развернут на [HuggingFace Spaces](https://huggingface.co/spaces/vladiusV/ChatRAG)

### Доступные сервисы

| Сервис | Ссылка | Описание |
|--------|--------|----------|
| API | [vladiusv-chatrag.hf.space](https://vladiusv-chatrag.hf.space) | Основной API сервис |
| Документация | [vladiusv-chatrag.hf.space/docs](https://vladiusv-chatrag.hf.space/docs) | Swagger UI документация |
| База данных | [Neon Console](https://console.neon.tech/app/projects/restless-dew-76629275/branches/br-purple-sound-a4cjk1l6/tables?database=neondb) | Управление базой данных |
