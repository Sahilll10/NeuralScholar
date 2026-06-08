import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class HybridSearcher:
    """
    Hybrid dense + sparse retrieval with Reciprocal Rank Fusion (RRF).
    
    Retrieval strategies:
    1. Dense: vector similarity via Pinecone/FAISS — captures semantic meaning
    2. Sparse: keyword matching via BM25 — captures exact terminology
    3. Hybrid: fuse rankings from both using RRF
    
    Reciprocal Rank Fusion formula (Cormack et al., 2009):
        RRF_score(d) = Σ_i [ weight_i / (k + rank_i(d)) ]
    
    where:
        k = 60 (smoothing constant — reduces impact of rank 1 vs rank 2)
        rank_i(d) = rank of document d in retrieval list i (1-indexed)
        weight_i = importance weight for retrieval list i (sum = 1.0)
    
    Default weights: dense=0.7, sparse=0.3
    These reflect that dense retrieval is generally more accurate for
    semantic ML questions, while BM25 catches keyword-exact queries.
    
    The optimal weights depend on your dataset and query distribution.
    For factual/keyword-heavy domains, increase sparse_weight to 0.4-0.5.
    """

    def __init__(
        self,
        dense_store,
        sparse_store,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        rrf_k: int = 60
    ):
        assert abs(dense_weight + sparse_weight - 1.0) < 1e-6, "Weights must sum to 1.0"
        self.dense_store = dense_store
        self.sparse_store = sparse_store
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k

    def search(
        self,
        query_text: str,
        query_embedding: List[float],
        top_k: int = 20,
        filter: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid retrieval and fuse results.

        Args:
            query_text: Raw query string for BM25 tokenization
            query_embedding: Dense query vector for Pinecone/FAISS
            top_k: Number of final results after fusion
            filter: Optional Pinecone metadata filter (not applied to BM25)

        Returns:
            List of result dicts sorted by RRF score descending.
            Each dict: {"id": str, "score": float, "metadata": dict}
        """
        candidate_multiplier = 3  # Retrieve 3x candidates before fusion for diversity
        num_candidates = top_k * candidate_multiplier

        dense_results = self.dense_store.query(
            embedding=query_embedding,
            top_k=num_candidates,
            filter=filter
        )

        sparse_results = self.sparse_store.query(
            query_text=query_text,
            top_k=num_candidates
        )

        logger.debug(f"Dense candidates: {len(dense_results)}, Sparse candidates: {len(sparse_results)}")

        fused = self._rrf(
            result_lists=[dense_results, sparse_results],
            weights=[self.dense_weight, self.sparse_weight]
        )

        return fused[:top_k]

    def _rrf(
        self,
        result_lists: List[List[Dict[str, Any]]],
        weights: List[float]
    ) -> List[Dict[str, Any]]:
        """
        Apply weighted Reciprocal Rank Fusion.

        For each document d:
            rrf_score(d) = Σ_i weight_i / (k + rank_i(d))
        
        Documents that appear in multiple lists receive additive contributions.
        Documents not found in a list contribute 0 from that list.
        """
        rrf_scores: Dict[str, float] = {}
        metadata_cache: Dict[str, Dict] = {}

        for result_list, weight in zip(result_lists, weights):
            for rank, result in enumerate(result_list):
                doc_id = result["id"]
                contribution = weight / (self.rrf_k + rank + 1)
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + contribution

                if doc_id not in metadata_cache:
                    metadata_cache[doc_id] = result.get("metadata", {})

        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        return [
            {
                "id": doc_id,
                "score": score,
                "metadata": metadata_cache[doc_id]
            }
            for doc_id, score in sorted_docs
        ]