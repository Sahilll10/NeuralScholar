__author__ = "Sahil Kumar - 3252"

import logging
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

from data.ingestion.arxiv_loader import ArXivLoader
from data.ingestion.pdf_loader import PDFLoader
from data.processing.metadata_extractor import MetadataExtractor

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
        
        self.arxiv_loader = ArXivLoader()
        self.pdf_loader = PDFLoader()
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def _process_and_index(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
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
                chunk_metadata["text"] = chunk_text  
                
                chunks.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "metadata": chunk_metadata
                })

        if not chunks:
            logger.warning("No chunks created from documents.")
            return {"documents_processed": 0, "chunks_created": 0, "vectors_upserted": 0}

        texts = [c["text"] for c in chunks]
        logger.info(f"Embedding {len(texts)} chunks...")
        embeddings = self.embedder.embed_documents(texts)

        pinecone_vectors = []
        faiss_vectors = []
        
        for chunk, emb in zip(chunks, embeddings):
            clean_emb = emb.tolist() if hasattr(emb, 'tolist') else list(emb)
            clean_emb = [float(x) for x in clean_emb]
            
            # Formatted for Pinecone
            pinecone_vectors.append({
                "id": chunk["id"],
                "values": clean_emb,
                "metadata": chunk["metadata"]
            })
            
            # Formatted as a single tuple (id, embedding, metadata) for your FAISS class definition
            faiss_vectors.append((chunk["id"], clean_emb, chunk["metadata"]))

        logger.info("Upserting vectors to Pinecone, FAISS, and BM25...")
        
        if hasattr(self.pinecone_store, 'index'):
            batch_size = 100
            for i in range(0, len(pinecone_vectors), batch_size):
                self.pinecone_store.index.upsert(vectors=pinecone_vectors[i:i+batch_size])
        else:
            self.pinecone_store.upsert(pinecone_vectors)

        # Fixes the signature mismatch by passing a single list of tuples
        self.faiss_store.add(faiss_vectors)
        
        self.bm25_store.add_documents(texts, [c["metadata"] for c in chunks])

        logger.info("Ingestion complete.")
        return {
            "documents_processed": len(documents),
            "chunks_created": len(chunks),
            "vectors_upserted": len(chunks)
        }

    def ingest_arxiv(self, query: Optional[str] = None, arxiv_ids: Optional[List[str]] = None, max_results: int = 5) -> Dict[str, Any]:
        if arxiv_ids:
            logger.info(f"Fetching {len(arxiv_ids)} specific ArXiv papers by ID...")
            papers = self.arxiv_loader.fetch_by_ids(arxiv_ids)
        elif query:
            logger.info(f"Fetching ArXiv papers for query: {query}")
            papers = self.arxiv_loader.fetch_papers(query, max_results=max_results)
        else:
            logger.warning("No query or arxiv_ids provided.")
            return {"documents_processed": 0, "chunks_created": 0, "vectors_upserted": 0}

        docs = []
        for p in papers:
            meta = MetadataExtractor.from_arxiv_paper(p)
            text = f"Title: {p.title}\n\nAbstract: {p.abstract}"
            docs.append({"doc_id": meta["doc_id"], "text": text, "metadata": meta})
            
        return self._process_and_index(docs)