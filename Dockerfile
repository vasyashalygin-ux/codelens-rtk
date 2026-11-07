# CodeLens — образ с предустановленными зависимостями и кодом.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    CODELENS_CHROMA_DIR=/app/chroma_db

WORKDIR /app

# git нужен части моделей sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# entrypoint: распаковать кодовую базу (если есть), проиндексировать, поднять UI
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 8501
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
