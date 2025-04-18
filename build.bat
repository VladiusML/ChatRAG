@echo off
docker-compose down
docker volume rm rag_system_postgres_data
docker-compose up --build