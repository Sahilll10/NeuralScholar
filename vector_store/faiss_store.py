import faiss
import numpy as np
import pickle
import os
import logging
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

class FAISSVectorStore:
    """
    Local FAISS-based vector store for development and offline use.
    
    Index type: IndexFlatIP (Inner Product)
    Since all-MiniLM-L6-v2 outputs L2-normalized vectors,
    inner product = cosine similarity. IndexFlatIP is an exact search
    (no approximation), so results are always optimal. For >1M vectors,
    consider IndexIVFFlat (approximate but much faster).
    
    Storage: the FAISS binary index is saved as {path}.faiss
    A companion pickle at {path}.meta stores the string→int→metadata mapping,
    because FAISS itself only handles integer IDs.
    """

    def __init__(self, dimension: int, index_path: Optional[str] = None):
        self.dimension = dimension
        self.index_path = index_path
        self.index = faiss.IndexFlatIP(dimension)
        self.id_to_metadata: Dict[str, Dict[str, Any]] = {}
        self.int_to_str_id: Dict[int, str] = {}
        self.str_to_int_id: Dict[str, int] = {}
        self._counter = 0

        if index_path and os.path.exists(f"{index_path}.faiss"):
            self.load(index_path)
            logger.info(f"FAISS index loaded: {self._counter} vectors")

    def add(self, vectors: List[Tuple[str, List[float], Dict[str, Any]]]):
        """
        Add vectors to the FAISS index.

        Args:
            vectors: List of (string_id, embedding, metadata) tuples.
                     Embeddings should already be L2-normalized.
        """
        if not vectors:
            return

        embeddings = np.array([v[1] for v in vectors], dtype=np.float32)

        # Ensure L2 normalization for cosine similarity via inner product
        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)

        for i, (vid, _, meta) in enumerate(vectors):
            int_id = self._counter + i
            self.int_to_str_id[int_id] = vid
            self.str_to_int_id[vid] = int_id
            self.id_to_metadata[vid] = meta

        self._counter += len(vectors)

    def query(self, embedding: List[float], top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Search the FAISS index for nearest neighbors.

        Args:
            embedding: Query vector of length self.dimension
            top_k: Number of results to return

        Returns:
            List of result dicts: [{"id": str, "score": float, "metadata": dict}]
            Scores are inner product values in range [-1, 1] for normalized vectors.
        """
        query_vec = np.array([embedding], dtype=np.float32)
        faiss.normalize_L2(query_vec)

        actual_top_k = min(top_k, self._counter)
        if actual_top_k == 0:
            return []

        scores, indices = self.index.search(query_vec, actual_top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            vid = self.int_to_str_id.get(int(idx))
            if vid:
                results.append({
                    "id": vid,
                    "score": float(score),
                    "metadata": self.id_to_metadata.get(vid, {})
                })

        return results

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}.meta", "wb") as f:
            pickle.dump({
                "id_to_metadata": self.id_to_metadata,
                "int_to_str_id": self.int_to_str_id,
                "str_to_int_id": self.str_to_int_id,
                "counter": self._counter
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str):
        self.index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}.meta", "rb") as f:
            data = pickle.load(f)
        self.id_to_metadata = data["id_to_metadata"]
        self.int_to_str_id = data["int_to_str_id"]
        self.str_to_int_id = data["str_to_int_id"]
        self._counter = data["counter"]