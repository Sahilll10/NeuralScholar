import numpy as np
from typing import List, Set, Dict, Any


class RetrievalMetrics:
    """
    Standard information retrieval evaluation metrics.
    
    These are computed during the automated evaluation pipeline to assess
    the retrieval component independently of generation quality.
    
    All methods are static and purely functional — no state, fully testable.
    """

    @staticmethod
    def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
        """
        Precision@k = |relevant ∩ retrieved[:k]| / k
        
        Measures what fraction of top-k retrieved items are relevant.
        Range: [0, 1]. Higher is better.
        """
        if k <= 0:
            return 0.0
        top_k = retrieved_ids[:k]
        hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
        return hits / k

    @staticmethod
    def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
        """
        Recall@k = |relevant ∩ retrieved[:k]| / |relevant|
        
        Measures what fraction of all relevant items appear in top-k.
        Range: [0, 1]. Higher is better. Can't exceed 1.
        """
        if not relevant_ids:
            return 0.0
        top_k = retrieved_ids[:k]
        hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
        return hits / len(relevant_ids)

    @staticmethod
    def average_precision(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
        """
        Average Precision (AP) = mean of Precision@k for each k where item k is relevant.
        
        Area under the Precision-Recall curve. Used to compute MAP across queries.
        """
        if not relevant_ids:
            return 0.0
        hits = 0
        sum_precision = 0.0
        for k, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_ids:
                hits += 1
                sum_precision += hits / k
        return sum_precision / len(relevant_ids)

    @staticmethod
    def mrr(
        retrieved_ids_list: List[List[str]],
        relevant_ids_list: List[Set[str]]
    ) -> float:
        """
        Mean Reciprocal Rank (MRR) = (1/|Q|) * Σ_q (1 / rank_first_relevant(q))
        
        Evaluates whether the first relevant document appears at rank 1, 2, 3, ...
        If no relevant doc is found, contribution is 0.
        Range: [0, 1]. Higher is better.
        
        MRR is appropriate when there is exactly one relevant document per query
        or when only the first relevant result matters (e.g., question answering).
        """
        reciprocal_ranks = []
        for retrieved, relevant in zip(retrieved_ids_list, relevant_ids_list):
            rr = 0.0
            for rank, doc_id in enumerate(retrieved, 1):
                if doc_id in relevant:
                    rr = 1.0 / rank
                    break
            reciprocal_ranks.append(rr)
        return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0

    @staticmethod
    def ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
        """
        Normalized Discounted Cumulative Gain at k.
        
        DCG@k = Σ_{i=1}^{k} rel_i / log2(i + 1)
        NDCG@k = DCG@k / IDCG@k   where IDCG = ideal DCG
        
        Uses binary relevance (1 if relevant, 0 if not).
        Penalizes relevant documents found at lower ranks more than higher ranks.
        """
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k]):
            rel = 1.0 if doc_id in relevant_ids else 0.0
            dcg += rel / np.log2(i + 2)

        ideal_rels = [1.0] * min(len(relevant_ids), k)
        idcg = sum(1.0 / np.log2(i + 2) for i, _ in enumerate(ideal_rels))

        return dcg / idcg if idcg > 0 else 0.0

    @classmethod
    def compute_all(
        cls,
        retrieved_ids_list: List[List[str]],
        relevant_ids_list: List[Set[str]],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, float]:
        """
        Compute all metrics for a set of queries. Returns averaged scores.
        """
        results = {}

        for k in k_values:
            p_scores = [cls.precision_at_k(r, rel, k) for r, rel in zip(retrieved_ids_list, relevant_ids_list)]
            r_scores = [cls.recall_at_k(r, rel, k) for r, rel in zip(retrieved_ids_list, relevant_ids_list)]
            ndcg_scores = [cls.ndcg_at_k(r, rel, k) for r, rel in zip(retrieved_ids_list, relevant_ids_list)]
            results[f"precision@{k}"] = float(np.mean(p_scores))
            results[f"recall@{k}"] = float(np.mean(r_scores))
            results[f"ndcg@{k}"] = float(np.mean(ndcg_scores))

        results["mrr"] = cls.mrr(retrieved_ids_list, relevant_ids_list)

        return results