"""
Central configuration for the local RAG system.

Every setting can be overridden with an environment variable of the same
name (this is how the Docker container is configured), e.g.

    LLM_MODEL=qwen2.5:3b TOP_K=6 streamlit run app.py
"""
import os
from pathlib import Path


def _env(name: str, default, cast=str):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    if cast is bool:
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return cast(value)


BASE_DIR = Path(__file__).resolve().parent

# ----------------------------------------------------------------- documents
# Folder with the knowledge source (.pdf, .txt and .md files are supported).
DATA_DIR = _env("DATA_DIR", str(BASE_DIR / "data" / "handbook"))
# Folder where Chroma persists vectors. One sub-folder per data folder is used,
# so switching DATA_DIR never mixes two knowledge bases.
VECTOR_ROOT = _env("VECTOR_ROOT", str(BASE_DIR / "vector_store"))
COLLECTION = _env("COLLECTION", "rag_documents")

# What the assistant is an expert on (inserted into the prompt).
DOMAIN = _env(
    "DOMAIN",
    "the B.S. Computer Science Major Handbook of Fulbright University Vietnam",
)

# ----------------------------------------------------------------- chunking
CHUNK_SIZE = _env("CHUNK_SIZE", 800, int)         # characters per chunk
CHUNK_OVERLAP = _env("CHUNK_OVERLAP", 150, int)  # characters shared by neighbours

# ----------------------------------------------------------------- models
OLLAMA_URL = _env("OLLAMA_URL", "http://127.0.0.1:11434")
EMBED_MODEL = _env("EMBED_MODEL", "nomic-embed-text")
# nomic-embed-text was trained with task prefixes; using them improves retrieval.
EMBED_PREFIXES = _env("EMBED_PREFIXES", True, bool)
LLM_MODEL = _env("LLM_MODEL", "qwen2.5:1.5b")
TEMPERATURE = _env("TEMPERATURE", 0.1, float)
NUM_CTX = _env("NUM_CTX", 4096, int)             # LLM context window (tokens)
# Hard cap on generated tokens: small models sometimes fall into endless
# repetition loops (observed: 5,800+ tokens from qwen2.5:0.5b).
MAX_TOKENS = _env("MAX_TOKENS", 512, int)

# ----------------------------------------------------------------- retrieval
TOP_K = _env("TOP_K", 5, int)
# "hybrid" (vector + BM25 keyword search fused with RRF), "vector" or "bm25"
RETRIEVAL_MODE = _env("RETRIEVAL_MODE", "hybrid")
# If the best cosine similarity is below this value the system answers
# "not found" instead of letting the model guess.
SCORE_THRESHOLD = _env("SCORE_THRESHOLD", 0.60, float)
# Number of previous Q/A turns passed to the model as conversation history.
HISTORY_TURNS = _env("HISTORY_TURNS", 2, int)

SAMPLE_QUESTION = "What are the prerequisites for CS311 Advanced Deep Learning?"
