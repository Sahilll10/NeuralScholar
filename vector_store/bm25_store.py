from rank_bm25 import BM25Okapi
import pickle
import os
import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class BM25Store:
    """
    BM25 sparse retrieval index using BM25Okapi.
    
    BM25 (Best Match 25) is a classic probabilistic retrieval function
    based on term frequency (TF) and inverse document frequency (IDF)
    with length normalization. It outperforms TF-IDF for IR tasks.
    
    BM25Okapi parameters:
        k1=1.5: Term frequency saturation. Higher = more TF influence.
                Standard range: 1.2-2.0. 1.5 works well for longer docs.
        b=0.75: Document length normalization. b=1 = full normalization.
                b=0 = no normalization. 0.75 is the standard for IR.
    
    This is used in the hybrid retrieval pipeline alongside Pinecone dense search.
    BM25 excels at exact keyword matching (paper titles, author names, method names)
    while dense search excels at semantic similarity. Their combination
    is stronger than either alone on most IR benchmarks (BEIR, MTEB).
    """

    def __init__(self, index_path: Optional[str] = None):
        self.bm25: Optional[BM25Okapi] = None
        self.documents: List[str] = []
        self.ids: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self.index_path = index_path

        if index_path and os.path.exists(f"{index_path}.bm25"):
            self.load(index_path)
            logger.info(f"BM25 index loaded: {len(self.documents)} documents")

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Simple whitespace tokenizer with lowercase normalization and punctuation removal.
        For ML text, we keep numbers and preserve compound terms (hyphenated words
        get split at hyphens to improve recall).
        """
        text = text.lower()
        text = re.sub(r"-", " ", text)
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        tokens = [t for t in text.split() if len(t) > 1]
        return tokens

    def add(self, documents: List[Dict[str, Any]]):
        """
        Add documents to the BM25 index.

        Note: BM25Okapi must be rebuilt entirely when new documents are added
        (no incremental update support). This is acceptable because ingestion
        happens offline, not during query serving.

        Args:
            documents: List of dicts, each with keys:
                id: str — chunk_id
                text: str — raw chunk text
                metadata: dict — arbitrary metadata
        """
        for doc in documents:
            self.documents.append(doc["text"])
            self.ids.append(doc["id"])
            self.metadata.append(doc.get("metadata", {}))

        tokenized = [self.tokenize(t) for t in self.documents]
        self.bm25 = BM25Okapi(tokenized, k1=1.5, b=0.75)
        logger.debug(f"BM25 rebuilt with {len(self.documents)} documents")

    def query(self, query_text: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Retrieve top-k documents by BM25 score.

        Args:
            query_text: Raw query string (tokenized internally)
            top_k: Maximum results to return

        Returns:
            List sorted by BM25 score descending. Excludes zero-score results.
        """
        if not self.bm25:
            return []

        tokenized_query = self.tokenize(query_text)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        top_indices = scores.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "id": self.ids[idx],
                    "score": float(scores[idx]),
                    "metadata": self.metadata[idx]
                })

        return results

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(f"{path}.bm25", "wb") as f:
            pickle.dump({
                "bm25": self.bm25,
                "documents": self.documents,
                "ids": self.ids,
                "metadata": self.metadata
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str):
        with open(f"{path}.bm25", "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.documents = data["documents"]
        self.ids = data["ids"]
        self.metadata = data["metadata"]