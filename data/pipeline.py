__author__ = "Sahil Kumar - 3252"

import logging
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from data.processing.metadata_extractor import MetadataExtractor

# Note: These loaders need to exist in your data/ingestion folder
try:
    from data.ingestion.arxiv_loader import ArxivLoader
    from data.ingestion.pdf_loader import PDFLoader
except ImportError:
    # Fallbacks if the specific loader files aren't created yet
    class ArxivLoader:
        def fetch_by_ids(self, ids): return []
        def fetch(self, query, max_results): return []
    class PDFLoader:
        def load_from_url(self, url): return {}

logger = logging.getLogger(__name__)

class IngestionPipeline:
    """
    Orchestrates the downloading, parsing, chunking, embedding, 
    and database upsertion of ML research papers.
    """
    def __init__(self, embedder, pinecone_store, faiss_store, bm25_store):
        self.embedder = embedder
        self.pinecone_store = pinecone_store
        self.faiss_store = faiss_store
        self.bm25_store = bm25_store
        self.arxiv_loader = ArxivLoader()
        self.pdf_loader = PDFLoader()
        
        # Configure chunking for dense scientific text
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def _process_and_index(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Takes raw text documents, splits them, embeds them, and pushes to vector databases.
        """
        chunks = []
        logger.info(f"Processing {len(documents)} documents for ingestion...")
        
        for doc in documents:
            doc_id = doc["doc_id"]
            text = doc["text"]
            metadata = MetadataExtractor.sanitize_for_pinecone(doc.get("metadata", {}))

            split_texts = self.text_splitter.split_text(text)
            for i, chunk_text in enumerate(split_texts):
                chunk_id = f"{doc_id}_chunk_{i}"
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = i
                chunk_metadata["text"] = chunk_text  # Store text for retrieval
                
                chunks.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "metadata": chunk_metadata
                })

        if not chunks:
            logger.warning("No chunks created from documents.")
            return {"documents_processed": 0, "chunks_created": 0, "vectors_upserted": 0}

        # Embed chunks using the selected model
        texts = [c["text"] for c in chunks]
        logger.info(f"Embedding {len(texts)} chunks...")
        embeddings = self.embedder.embed_documents(texts)

        # Prepare vectors for Pinecone
        pinecone_vectors = []
        for chunk, emb in zip(chunks, embeddings):
            pinecone_vectors.append({
                "id": chunk["id"],
                "values": emb,
                "metadata": chunk["metadata"]
            })

        # Upsert to all three vector stores
        logger.info("Upserting vectors to Pinecone, FAISS, and BM25...")
        self.pinecone_store.upsert(pinecone_vectors)
        self.faiss_store.add(texts, embeddings, [c["metadata"] for c in chunks])
        self.bm25_store.add_documents(texts, [c["metadata"] for c in chunks])

        logger.info("Ingestion complete.")
        return {
            "documents_processed": len(documents),
            "chunks_created": len(chunks),
            "vectors_upserted": len(chunks)
        }

    def ingest_arxiv(self, query: str, max_results: int = 5, categories: Optional[List[str]] = None, use_full_text: bool = False) -> Dict[str, Any]:
        """
        Fetch papers from ArXiv via query and push them through the pipeline.
        """
        logger.info(f"Fetching ArXiv papers for query: {query}")
        papers = self.arxiv_loader.fetch(query, max_results=max_results)
        docs = []
        
        for p in papers:
            meta = MetadataExtractor.from_arxiv_paper(p)
            text = f"Title: {p.title}\n\nAbstract: {p.abstract}"
            docs.append({"doc_id": meta["doc_id"], "text": text, "metadata": meta})
            
        return self._process_and_index(docs)