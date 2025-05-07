FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y libpq-dev gcc && \
    python -m pip install --upgrade pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/cache_hf /app/logs && \
    chown -R nobody:nogroup /app/cache_hf /app/logs && \
    chmod -R 777 /app/cache_hf /app/logs


COPY requirements.txt /app/
RUN pip install -r requirements.txt --verbose

COPY . /app/

EXPOSE 7860

CMD ["sh", "-c", "python wait_for_db.py && alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 7860"]
