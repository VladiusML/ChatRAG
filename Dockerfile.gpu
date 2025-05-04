FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && \
    apt-get install -y python3.10 python3.10-dev python3-pip libpq-dev gcc && \
    ln -s /usr/bin/python3.10 /usr/bin/python && \
    python -m pip install --upgrade pip

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt --verbose

COPY . /app/

EXPOSE 8000

CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]