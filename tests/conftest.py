import pytest
import os
os.environ.setdefault("OPENAI_API_KEY", "test-key-placeholder")
os.environ.setdefault("PINECONE_API_KEY", "test-key-placeholder")
os.environ.setdefault("PINECONE_INDEX_NAME", "test-index")
os.environ.setdefault("PINECONE_DIMENSION", "768")


@pytest.fixture
def sample_texts():
    return [
        "The attention mechanism allows transformers to focus on relevant tokens.",
        "BERT uses bidirectional attention for masked language modeling pre-training.",
        "GPT models use causal (unidirectional) attention for autoregressive generation.",
        "Retrieval augmented generation combines parametric and non-parametric memory.",
        "Dense passage retrieval uses bi-encoders to embed queries and passages independently."
    ]


@pytest.fixture
def sample_chunks():
    return [
        {
            "chunk_id": f"doc1_chunk_{i}",
            "doc_id": "doc1",
            "text": f"Sample chunk {i} about machine learning and transformers.",
            "metadata": {"title": "Test Paper", "authors": "Author A", "published_date": "2024-01-01"}
        }
        for i in range(5)
    ]