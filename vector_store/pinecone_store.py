import time
import logging
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Any, Optional, Tuple
from config.settings import settings

logger = logging.getLogger(__name__)


class PineconeVectorStore:
    """
    Pinecone serverless vector store for production dense retrieval.
    
    Index configuration:
        metric: cosine — measures angular similarity between L2-normalized vectors.
                With normalized vectors, cosine = dot product = Euclidean distance (up to monotone transform).
        pod_type: serverless — auto-scales, no idle cost
        cloud/region: aws/us-east-1 — lowest latency for most regions
    
    Namespace usage: use distinct namespaces to partition different document
    collections (e.g., "arxiv-2024", "custom-corpus") within the same index.
    This avoids the cost of creating multiple indexes.
    
    Upsert semantics: identical chunk_ids are updated (not duplicated).
    This makes re-ingestion idempotent.
    """

    def __init__(
        self,
        api_key: str,
        index_name: str,
        dimension: int,
        metric: str = "cosine",
        namespace: str = "default"
    ):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        self.namespace = namespace
        self.index = self._get_or_create_index()
        logger.info(f"Pinecone store ready: index={index_name}, dim={dimension}")

    def _get_or_create_index(self):
        existing = {idx.name for idx in self.pc.list_indexes()}
        if self.index_name not in existing:
            logger.info(f"Creating Pinecone index '{self.index_name}'...")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric,
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            time.sleep(3)  # Allow index to initialize on Pinecone's backend
        return self.pc.Index(self.index_name)

    def upsert(
        self,
        vectors: List[Tuple[str, List[float], Dict[str, Any]]],
        batch_size: int = 100
    ):
        """
        Upsert (insert or update) vectors into Pinecone.

        Args:
            vectors: List of (chunk_id, embedding_list, metadata_dict) tuples.
                     chunk_id must be unique per chunk.
                     embedding_list must have exactly self.dimension elements.
                     metadata_dict values must be str, int, float, bool, or List[str].
            batch_size: Maximum 100 vectors per Pinecone upsert call.
        """
        records = [
            {"id": vid, "values": emb, "metadata": meta}
            for vid, emb, meta in vectors
        ]

        for i in range(0, len(records), batch_size):
            batch = records[i: i + batch_size]
            self.index.upsert(vectors=batch, namespace=self.namespace)
            logger.debug(f"Upserted batch {i // batch_size + 1}/{(len(records) - 1) // batch_size + 1}")

        logger.info(f"Pinecone upsert complete: {len(records)} vectors")

    def query(
        self,
        embedding: List[float],
        top_k: int = 20,
        filter: Optional[Dict] = None,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Query Pinecone with a dense vector. Returns cosine similarity scores.

        Args:
            embedding: Query vector of length self.dimension
            top_k: Number of nearest neighbors to return (max 10000 per Pinecone)
            filter: Pinecone metadata filter using MongoDB-style operators.
                    Example: {"published_date": {"$gte": "2023-01-01"}}
                    Example: {"categories": {"$in": ["cs.CL"]}}
            include_metadata: Whether to return metadata with results

        Returns:
            List of dicts sorted by score descending:
            [{"id": str, "score": float, "metadata": dict}, ...]
        """
        response = self.index.query(
            vector=embedding,
            top_k=top_k,
            namespace=self.namespace,
            filter=filter,
            include_metadata=include_metadata,
            include_values=False
        )

        return [
            {
                "id": match.id,
                "score": float(match.score),
                "metadata": dict(match.metadata) if match.metadata else {}
            }
            for match in response.matches
        ]

    def delete_all(self):
        """Delete all vectors in the namespace."""
        self.index.delete(delete_all=True, namespace=self.namespace)
        logger.warning(f"Deleted all vectors in namespace '{self.namespace}'")

    def get_stats(self) -> Dict[str, Any]:
        """Return index statistics including vector count."""
        return self.index.describe_index_stats()