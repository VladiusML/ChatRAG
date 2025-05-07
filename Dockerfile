FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y libpq-dev gcc && \
    python -m pip install --upgrade pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install -r requirements.txt --verbose

COPY . /app/

EXPOSE 7860

# Добавляем проверку подключения к БД
CMD ["sh", "-c", "python -c 'import time; import psycopg2; import os; \
    while True: \
        try: \
            conn = psycopg2.connect(os.getenv(\"DATABASE_URL\")); \
            conn.close(); \
            break; \
        except Exception as e: \
            print(f\"Waiting for database connection... {e}\"); \
            time.sleep(5);' && \
    alembic upgrade head && \
    uvicorn app.main:app --host 0.0.0.0 --port 7860"]
