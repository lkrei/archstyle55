FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app:/repo

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgl1 libglib2.0-0 git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY service/backend/requirements.txt /app/requirements.txt
RUN pip install torch==2.6.0 torchvision==0.21.0 \
    && pip install -r /app/requirements.txt

COPY service/backend/app /app/app
COPY service/backend/alembic.ini /app/alembic.ini
COPY service/scripts /app/scripts

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -fsS http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
