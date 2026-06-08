#!/usr/bin/env python3
"""
CLI script to ingest ML papers from ArXiv into the vector stores.

Usage:
    python scripts/ingest_arxiv.py --query "retrieval augmented generation" --max-results 200
    python scripts/ingest_arxiv.py --default-corpus --papers-per-query 100
    python scripts/ingest_arxiv.py --ids 2005.11401 2302.12192 2309.15217
"""

import argparse
import logging
import sys
sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Ingest ArXiv papers into NeuralScholar")
    parser.add_argument("--query", type=str, help="ArXiv search query")
    parser.add_argument("--max-results", type=int, default=100)
    parser.add_argument("--categories", nargs="+", default=["cs.CL", "cs.AI", "cs.LG", "cs.IR"])
    parser.add_argument("--use-full-text", action="store_true", help="Download and parse PDFs")
    parser.add_argument("--default-corpus", action="store_true", help="Use built-in query set")
    parser.add_argument("--papers-per-query", type=int, default=100)
    parser.add_argument("--ids", nargs="+", help="Specific ArXiv IDs to ingest")
    args = parser.parse_args()

    from config.settings import settings
    from embeddings.embedding_manager import EmbeddingManager
    from vector_store.pinecone_store import PineconeVectorStore
    from vector_store.faiss_store import FAISSVectorStore
    from vector_store.bm25_store import BM25Store
    from data.pipeline import IngestionPipeline

    logger.info("Initializing components...")
    embedder = EmbeddingManager(primary="local")
    pinecone = PineconeVectorStore(
        api_key=settings.PINECONE_API_KEY,
        index_name=settings.PINECONE_INDEX_NAME,
        dimension=settings.PINECONE_DIMENSION
    )
    faiss = FAISSVectorStore(dimension=settings.LOCAL_EMBEDDING_DIM, index_path=settings.FAISS_INDEX_PATH)
    bm25 = BM25Store(index_path=settings.BM25_INDEX_PATH)
    pipeline = IngestionPipeline(embedder, pinecone, faiss, bm25)

    if args.ids:
        logger.info(f"Ingesting specific papers: {args.ids}")
        papers = pipeline.arxiv_loader.fetch_by_ids(args.ids)
        from data.processing.metadata_extractor import MetadataExtractor
        docs = []
        for p in papers:
            meta = MetadataExtractor.from_arxiv_paper(p)
            text = f"Title: {p.title}\n\nAbstract: {p.abstract}"
            docs.append({"doc_id": meta["doc_id"], "text": text, "metadata": meta})
        result = pipeline._