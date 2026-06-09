import numpy as np
import logging
from typing import List, Literal, Optional
from embeddings.local_embedder import LocalEmbedder
from embeddings.openai_embedder import OpenAIEmbedder
from config.settings import settings

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Unified interface for local and OpenAI embedders.
    
    Supports three operational modes:
    1. primary="local": Use sentence-transformers only (no API cost, offline-capable)
    2. primary="openai": Use OpenAI API, fallback to local on failure
    3. primary="local", fallback=True: Use local, fallback to OpenAI (not typical)
    
    For production with the Pinecone index, you MUST use the same embedder
    for both ingestion and retrieval. The PINECONE_DIMENSION setting must
    match the chosen embedder's output dimension.
    """

    def __init__(
        self,
        primary: Literal["local", "openai"] = "local",
        fallback: bool = True
    ):
        self.primary = primary
        self.fallback = fallback
        self._local: Optional[LocalEmbedder] = None
        self._openai: Optional[OpenAIEmbedder] = None
        self._initialize()

    def _initialize(self):
        """Load only the embedders that are needed."""
        if self.primary == "local" or self.fallback:
            self._local = LocalEmbedder(
                model_name=settings.LOCAL_EMBEDDING_MODEL,
                normalize_embeddings=True,
                batch_size=32
            )

        if self.primary == "openai" or (self.fallback and self.primary == "local"):
            try:
                self._openai = OpenAIEmbedder(
                    api_key=settings.OPENAI_API_KEY,
                    model=settings.OPENAI_EMBEDDING_MODEL,
                    dimensions=settings.PINECONE_DIMENSION if settings.PINECONE_DIMENSION != settings.OPENAI_EMBEDDING_DIM else None
                )
            except Exception as e:
                logger.warning(f"OpenAI embedder init failed (will use local only): {e}")
                self._openai = None

    @property
    def dimension(self) -> int:
        if self.primary == "local":
            return self._local.dimension
        return self._openai.dimension if self._openai else (self._local.dimension if self._local else 768)

    def embed_query(self, text: str) -> List[float]:
        try:
            if self.primary == "local":
                return self._local.embed_query(text)
            return self._openai.embed_query(text)
        except Exception as e:
            if self.fallback and self.primary == "openai" and self._local:
                logger.warning(f"OpenAI embed failed, using local fallback: {e}")
                return self._local.embed_query(text)
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            if self.primary == "local":
                return self._local.embed_documents(texts)
            return self._openai.embed_documents(texts)
        except Exception as e:
            if self.fallback and self.primary == "openai" and self._local:
                logger.warning(f"OpenAI batch embed failed, using local fallback: {e}")
                return self._local.embed_documents(texts)
            raise