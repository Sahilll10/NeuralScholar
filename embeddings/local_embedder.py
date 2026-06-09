import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class LocalEmbedder:
    """
    Dense embedder using sentence-transformers all-mpnet-base-v2.
    
    Model details:
        Architecture: MPNet (Masked and Permuted Pre-training for Language Understanding)
        Output dimension: 768
        Max input tokens: 384 (approximately 500-600 characters)
        Trained on: 1B+ sentence pairs with contrastive loss
        SBERT benchmark rank: Top-5 on semantic similarity tasks as of 2024
    
    Why all-mpnet-base-v2 over alternatives:
        - vs all-MiniLM-L6-v2: MPNet is slower but 5-8% more accurate on BEIR
        - vs all-MiniLM-L12-v2: Same accuracy range, MPNet architecture handles
          long-range dependencies better for technical text
        - vs E5-large: E5 is larger/more accurate but requires 2x compute
        
    normalize_embeddings=True: L2-normalization makes cosine similarity
    equivalent to dot product, which FAISS IndexFlatIP supports efficiently.
    """

    def __init__(
        self,
        model_name = "all-MiniLM-L6-v2",
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

        logger.info(f"Loading {model_name} on {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedder ready. Dimension: {self.dimension}")

    def embed_text(self, text: str) -> np.ndarray:
        """Embed single text. Returns shape (768,)."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts in batches.

        Args:
            texts: List of raw strings to embed

        Returns:
            numpy array of shape (len(texts), 768), dtype=float32
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