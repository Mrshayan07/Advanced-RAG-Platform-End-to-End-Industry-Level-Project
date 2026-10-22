FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/huggingface \
    TRANSFORMERS_CACHE=/opt/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/opt/huggingface/sentence_transformers

WORKDIR /app

RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.14.0

COPY requirements.txt .

RUN pip install -r requirements.txt

# ---------------------------------------------------------
# Download production reranker during IMAGE BUILD.
# The model becomes part of the Docker image.
# ---------------------------------------------------------

RUN mkdir -p /opt/huggingface \
    && python -c "from sentence_transformers import CrossEncoder; print('Downloading BAAI/bge-reranker-v2-m3...'); CrossEncoder('BAAI/bge-reranker-v2-m3'); print('Reranker baked into image successfully.')"

COPY app ./app
COPY migrations ./migrations
COPY alembic.ini .

RUN mkdir -p /app/storage

RUN useradd \
        --create-home \
        --shell /bin/bash \
        appuser \
    && chown -R appuser:appuser /app /opt/huggingface

USER appuser

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]