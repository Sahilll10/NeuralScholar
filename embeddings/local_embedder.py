import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class LocalEmbedder:
    """
    Dense embedder using sentence-transformers all-MiniLM-L6-v2.
    
    Model details:
        Architecture: MiniLM (Lightweight Transformer)
        Output dimension: 384
        Max input tokens: 256
        Trained on: 1B+ sentence pairs
        Advantage: 5x faster and 4x smaller than MPNet, perfect for Free Tier cloud hosting.
        
    normalize_embeddings=True: L2-normalization makes cosine similarity
    equivalent to dot product, which FAISS IndexFlatIP supports efficiently.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading {self.model_name} on {self.device}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedder ready. Dimension: {self.dimension}")

    def embed_text(self, text: str) -> np.ndarray:
        """Embed single text. Returns shape (384,)."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts in batches.

        Args:
            texts: List of raw strings to embed

        Returns:
            numpy array of shape (len(texts), 384), dtype=float32
            All vectors are L2-normalized when normalize_embeddings=True.
        """
        cleaned = [t.strip() if t and t.strip() else "empty document" for t in texts]

        embeddings = self.model.encode(
            cleaned,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=len(texts) > 200,
            convert_to_numpy=True,
            device=self.device
        )

        return embeddings.astype(np.float32)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """LangChain-compatible interface."""
        return self.embed_batch(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        """LangChain-compatible query embedding."""
        return self.embed_text(text).tolist()