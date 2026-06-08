import pytest
from evaluation.metrics import RetrievalMetrics


def test_precision_at_k():
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = {"doc1", "doc3", "doc5"}
    assert RetrievalMetrics.precision_at_k(retrieved, relevant, 5) == 3 / 5
    assert RetrievalMetrics.precision_at_k(retrieved, relevant, 1) == 1.0
    assert RetrievalMetrics.precision_at_k(retrieved, relevant, 2) == 0.5


def test_mrr_first_relevant():
    retrieved_list = [["doc1", "doc2", "doc3"]]
    relevant_list = [{"doc1"}]
    assert RetrievalMetrics.mrr(retrieved_list, relevant_list) == 1.0


def test_mrr_second_relevant():
    retrieved_list = [["doc0", "doc1", "doc2"]]
    relevant_list = [{"doc1"}]
    assert RetrievalMetrics.mrr(retrieved_list, relevant_list) == 0.5


def test_mrr_not_found():
    retrieved_list = [["doc1", "doc2"]]
    relevant_list = [{"doc9"}]
    assert RetrievalMetrics.mrr(retrieved_list, relevant_list) == 0.0


def test_ndcg_perfect():
    retrieved = ["doc1", "doc2", "doc3"]
    relevant = {"doc1", "doc2", "doc3"}
    assert abs(RetrievalMetrics.ndcg_at_k(retrieved, relevant, 3) - 1.0) < 1e-6


def test_compute_all():
    retrieved_list = [["doc1", "doc2", "doc3", "doc4", "doc5"]] * 10
    relevant_list = [{"doc1", "doc3"}] * 10
    results = RetrievalMetrics.compute_all(retrieved_list, relevant_list)
    assert "precision@5" in results
    assert "mrr" in results
    assert "ndcg@5" in results
    assert 0.0 <= results["precision@5"] <= 1.0