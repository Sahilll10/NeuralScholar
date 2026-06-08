import numpy as np
import time
import logging
from openai import OpenAI
from typing import List, Optional

logger = logging.getLogger(__name__)


class OpenAIEmbedder:
    """
    OpenAI text-embedding-3-small based embedder.
    
    Model details:
        text-embedding-3-small: 1536 dimensions, $0.02/1M tokens
        text-embedding-3-large: 3072 dimensions, $0.13/1M tokens
    
    The 'dimensions' parameter allows native dimension reduction via
    Matryoshka representation learning — embeddings remain meaningful
    even when truncated. Use dimensions=768 for Pinecone cost savings.
    
    Implements exponential backoff retry for rate limit errors (429)
    and server errors (500, 503).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimensions: Optional[int] = None,
        max_retries: int = 3,
        base_retry_delay: float = 1.0
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.dimensions = dimensions
        self.max_retries = max_retries
        self.base_retry_delay = base_retry_delay

        if "small" in model:
            self.dimension = dimensions or 1536
        elif "large" in model:
            self.dimension = dimensions or 3072
        else:
            self.dimension = dimensions or 1536

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single string. Returns shape (self.dimension,)."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """
        Embed texts in batches with retry logic.
        
        OpenAI supports up to 2048 inputs per request, but 100 per batch
        is safer for token limit compliance (max 8191 tokens per request).
        
        Each text is preprocessed: newlines replaced with spaces,
        leading/trailing whitespace stripped, empty strings become "empty".
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            batch = [t.replace("\n", " ").strip() or "empty" for t in batch]

            for attempt in range(self.max_retries):
                try:
                    kwargs = {"model": self.model, "input": batch}
                    if self.dimensions:
                        kwargs["dimensions"] = self.dimensions

                    response = self.client.embeddings.create(**kwargs)

                    # Sort by index to maintain order (API may return out of order)
                    sorted_data = sorted(response.data, key=lambda x: x.index)
                    batch_embeddings = [item.embedding for item in sorted_data]
                    all_embeddings.extend(batch_embeddings)
                    break

                except Exception as e:
                    if attempt < self.max_retries - 1:
                        delay = self.base_retry_delay * (2 ** attempt)
                        logger.warning(f"OpenAI embed attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        logger.error(f"OpenAI embed failed after {self.max_retries} attempts")
                        raise

        return np.array(all_embeddings, dtype=np.float32)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """LangChain-compatible interface."""
        return self.embed_batch(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        """LangChain-compatible query embedding."""
        return self.embed_text(text).tolist()