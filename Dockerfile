FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y \
    libpq-dev \
    gcc \
    python3-dev

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

EXPOSE 8000

CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
