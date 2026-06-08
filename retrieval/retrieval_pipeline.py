import numpy as np
import logging
import time
from typing import List, Dict, Any, Optional
from retrieval.hyde import HyDEGenerator
from retrieval.hybrid_search import HybridSearcher
from retrieval.reranker import CrossEncoderReranker
from embeddings.embedding_manager import EmbeddingManager
from config.settings import settings

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """
    Full retrieval pipeline: Query → HyDE → Hybrid Search → Cross-Encoder Rerank → Top-K
    
    This class is instantiated once at API startup and shared across requests.
    All components are thread-safe and stateless per query.
    
    Complete flow for a single query:
    
    1. HyDE (if enabled):
       query → LLM → [hyp_1, hyp_2, hyp_3]
       [query, hyp_1, hyp_2, hyp_3] → embedder → [e_q, e_1, e_2, e_3]
       query_embedding = mean([e_q, e_1, e_2, e_3])  ← richer embedding
    
    2. Dense retrieval (Pinecone or FAISS):
       query_embedding → cosine search → top-60 chunks
    
    3. Sparse retrieval (BM25):
       query text → BM25 scoring → top-60 chunks
    
    4. RRF fusion:
       dense_top60 + sparse_top60 → RRF → 20 unique candidates
    
    5. Cross-encoder reranking:
       [(query, chunk_i) for i in range(20)] → cross-encoder → 5 final results
    
    6. Return 5 chunks with metadata to generation layer.
    """

    def __init__(
        self,
        embedder: EmbeddingManager,
        dense_store,
        sparse_store,
        use_hyde: bool = True,
        use_hybrid: bool = True,
        use_reranker: bool = True
    ):
        self.embedder = embedder
        self.dense_store = dense_store
        self.sparse_store = sparse_store
        self.use_hyde = use_hyde
        self.use_hybrid = use_hybrid
        self.use_reranker = use_reranker

        if use_hyde:
            self.hyde = HyDEGenerator(
                api_key=settings.OPENAI_API_KEY,
                num_hypothetical=settings.HYDE_NUM_HYPOTHETICAL
            )

        if use_hybrid:
            self.searcher = HybridSearcher(
                dense_store=dense_store,
                sparse_store=sparse_store,
                dense_weight=settings.DENSE_WEIGHT,
                sparse_weight=settings.BM25_WEIGHT,
                rrf_k=settings.RRF_K
            )

        if use_reranker:
            self.reranker = CrossEncoderReranker()

    def retrieve(
        self,
        query: str,
        top_k_retrieval: int = None,
        top_k_final: int = None,
        filter: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute the full retrieval pipeline for a query.

        Args:
            query: Raw user query string
            top_k_retrieval: Candidates to retrieve before reranking (default from settings)
            top_k_final: Final results after reranking (default from settings)
            filter: Optional Pinecone metadata filter dict

        Returns:
            List of top-k dicts. Each dict:
            {
                "id": "arxiv_2305.14283_chunk_7",
                "score": 0.0142,          ← RRF score (or cosine if no hybrid)
                "rerank_score": 3.45,     ← Cross-encoder score (if reranker enabled)
                "metadata": {
                    "doc_id": "arxiv_2305.14283",
                    "title": "...",
                    "authors": "...",
                    "text": "...",         ← The actual chunk text for generation
                    ...
                }
            }
        """
        top_k_retrieval = top_k_retrieval or settings.TOP_K_RETRIEVAL
        top_k_final = top_k_final or settings.TOP_K_RERANK

        t0 = time.time()

        # Step 1: Embed query (with or without HyDE)
        if self.use_hyde:
            hypotheticals = self.hyde.generate_hypothetical_documents(query)
            all_texts = [query] + hypotheticals
            all_embeddings = [self.embedder.embed_query(t) for t in all_texts]
            query_embedding = np.mean(all_embeddings, axis=0).tolist()
            logger.debug(f"HyDE: averaged {len(all_embeddings)} embeddings in {time.time() - t0:.2f}s")
        else:
            query_embedding = self.embedder.embed_query(query)

        # Step 2: Retrieve candidates
        if self.use_hybrid:
            candidates = self.searcher.search(
                query_text=query,
                query_embedding=query_embedding,
                top_k=top_k_retrieval,
                filter=filter
            )
        else:
            candidates = self.dense_store.query(
                embedding=query_embedding,
                top_k=top_k_retrieval,
                filter=filter
            )

        logger.info(f"Retrieved {len(candidates)} candidates in {(time.time() - t0) * 1000:.0f}ms")

        if not candidates:
            logger.warning("No candidates retrieved — returning empty list")
            return []

        # Step 3: Cross-encoder reranking
        if self.use_reranker:
            results = self.reranker.rerank(
                query=query,
                candidates=candidates,
                top_k=top_k_final
            )
        else:
            results = candidates[:top_k_final]

        logger.info(
            f"Final retrieval: {len(results)} docs | "
            f"Total: {(time.time() - t0) * 1000:.0f}ms"
        )

        return results