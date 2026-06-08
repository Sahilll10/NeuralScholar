import pytest
from data.processing.chunker import DocumentChunker


def test_basic_chunking():
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    text = "This is a sentence. " * 20
    chunks = chunker.chunk_document(text, {"title": "Test"}, "test_doc")
    assert len(chunks) > 1
    for c in chunks:
        assert len(c["text"]) <= 100 + 20
        assert c["doc_id"] == "test_doc"
        assert c["chunk_id"].startswith("test_doc_chunk_")


def test_empty_text():
    chunker = DocumentChunker()
    chunks = chunker.chunk_document("", {}, "empty_doc")
    assert chunks == []


def test_overlap():
    chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
    text = "A" * 200
    chunks = chunker.chunk_document(text, {}, "test")
    assert len(chunks) > 1
    for c in chunks:
        assert c["total_chunks"] == len(chunks)


def test_metadata_preserved():
    chunker = DocumentChunker(chunk_size=200, chunk_overlap=20)
    meta = {"title": "My Paper", "authors": "Author A", "arxiv_id": "2401.12345"}
    chunks = chunker.chunk_document("Test content " * 50, meta, "paper1")
    for c in chunks:
        assert c["metadata"]["title"] == "My Paper"
        assert c["metadata"]["arxiv_id"] == "2401.12345"