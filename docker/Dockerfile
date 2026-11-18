# ---------------------------------------------------------------------------
# CS Handbook RAG chatbot - application image
# Build context = project root (the folder that contains source_code/ and docker/)
#   docker build -f docker/Dockerfile -t cs-handbook-rag .
# ---------------------------------------------------------------------------
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Defaults - all can be overridden at "docker run -e" / compose time
    OLLAMA_URL=http://ollama:11434 \
    LLM_MODEL=qwen2.5:1.5b \
    EMBED_MODEL=nomic-embed-text \
    DATA_DIR=/app/data/handbook \
    VECTOR_ROOT=/app/vector_store

WORKDIR /app

# curl is only used by the HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first (cached layer)
COPY source_code/requirements.txt .
RUN pip install -r requirements.txt

# Application code + default knowledge base
COPY source_code/ /app/
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && useradd -m raguser && mkdir -p /app/vector_store \
    && chown -R raguser /app
USER raguser

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fs http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
# Default: web UI. Override, e.g.:  docker run ... cs-handbook-rag cli -q "..."
CMD ["web"]
