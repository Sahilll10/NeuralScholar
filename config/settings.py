from pydantic_settings import BaseSettings
from typing import Optional, Literal


class Settings(BaseSettings):
    # OpenAI
    OPENAI_API_KEY: str
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_CHAT_MODEL: str = "gpt-4o"
    OPENAI_EMBEDDING_DIM: int = 1536

    # Pinecone
    PINECONE_API_KEY: str
    PINECONE_ENVIRONMENT: str = "gcp-starter"
    PINECONE_INDEX_NAME: str = "neuralscholar"
    PINECONE_DIMENSION: int = 768  # Must match LOCAL_EMBEDDING_DIM

    # Local embedding model
    LOCAL_EMBEDDING_MODEL: str = "all-mpnet-base-v2"
    LOCAL_EMBEDDING_DIM: int = 768

    # Text chunking
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64

    # Retrieval config
    TOP_K_RETRIEVAL: int = 20
    TOP_K_RERANK: int = 5
    RETRIEVAL_MODE: Literal["hybrid", "dense", "sparse"] = "hybrid"
    BM25_WEIGHT: float = 0.3
    DENSE_WEIGHT: float = 0.7
    RRF_K: int = 60

    # HyDE
    HYDE_ENABLED: bool = True
    HYDE_NUM_HYPOTHETICAL: int = 3

    # Generation
    MAX_CONTEXT_TOKENS: int = 8000
    TEMPERATURE: float = 0.1
    MAX_TOKENS: int = 1024
    STREAMING: bool = True

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    CACHE_TTL: int = 3600

    # FastAPI
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = False
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_PERIOD: int = 60

    # Storage paths
    FAISS_INDEX_PATH: str = "./data/faiss_index"
    BM25_INDEX_PATH: str = "./data/bm25_index"

    # Evaluation
    EVAL_SAMPLE_SIZE: int = 50

    # Logging
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()