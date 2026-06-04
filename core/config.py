import torch

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Model Paths ───────────────────────────────────────────
    DPO_CHECKPOINT: str   = "E:\vs codes\gemma_model\Qwen2.5-0.5B-Instruct-DPO\checkpoint-1185"
    EMOTION_MODEL: str    = ""
    LANGUAGE_MODEL: str   = ""
    DENSE_ENCODER: str    = ""
    SPARSE_ENCODER: str   = ""
    CROSS_ENCODER: str    = ""
    TRANSLATOR_MODEL: str = ""

    # ── LLM Keys ──────────────────────────────────────────────
    GROQ_API_KEY: str = ""

    # ── Emotion Model ──────────────────────────────────────────────
    BERT_EMOTION_MODEL_NAME: str = "albert/albert-base-v2"

    # ── Pinecone ──────────────────────────────────────────────
    PINECONE_API_KEY: str    = ""
    PINECONE_INDEX_NAME: str = "mental-health"
    PINECONE_DIMENSION: int  = 384
    PINECONE_METRIC: str     = "dotproduct"

    # ── RAG Tuning ────────────────────────────────────────────
    RAG_TOP_K: int            = 20
    RERANKER_TOP_K: int       = 5
    RELEVANCE_THRESHOLD: float = 0.7
    MAX_REWRITE_ATTEMPTS: int  = 2

    # ── External API Keys (read from .env, optional) ──────────
    GEMINI_API_KEY: str  = ""
    HF_API_KEY: str      = ""
    OPENAI_API_KEY: str  = ""

    # ── LangSmith (read from .env, optional) ──────────────────
    LANGSMITH_TRACING: str  = ""
    LANGSMITH_ENDPOINT: str = ""
    LANGSMITH_API_KEY: str  = ""
    LANGSMITH_PROJECT: str  = ""

    # ── FastAPI ───────────────────────────────────────────────
    APP_TITLE: str = "PsyNet Assistant"
    APP_HOST: str  = "127.0.0.1"
    APP_PORT: int  = 8000

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    class Config:
        env_file     = ".env"
        extra        = "ignore"   # ← silently ignores any .env key not declared above
        case_sensitive = False    # ← gemini_api_key and GEMINI_API_KEY both match


settings = Settings()