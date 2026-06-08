from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DocumentChunker:
    """
    Splits raw document text into overlapping fixed-size chunks.
    
    Uses LangChain's RecursiveCharacterTextSplitter which tries to split
    at natural boundaries in order: paragraph breaks, line breaks, sentences,
    spaces, then characters. This preserves semantic coherence better than
    naive character-level splitting.
    
    Parameters:
        chunk_size=512: Each chunk contains at most 512 characters.
            At ~5 chars/word, this is ~100 words per chunk — fits comfortably
            within sentence transformer context windows (max 384 tokens for
            all-mpnet-base-v2 which is ~500-600 characters).
        chunk_overlap=64: Adjacent chunks share 64 characters to prevent
            context loss at chunk boundaries. This is ~12.5% overlap,
            which is standard practice.
    
    Important: chunk_size refers to characters, not tokens. Tokens are
    approximately chunk_size / 4 on average for English text.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]
        )

    def chunk_document(
        self,
        text: str,
        metadata: Dict[str, Any],
        doc_id: str
    ) -> List[Dict[str, Any]]:
        """
        Split one document into chunks and attach metadata to each chunk.

        Args:
            text: Full raw document text
            metadata: Document-level metadata dict to copy into each chunk
            doc_id: Unique string identifier for this document

        Returns:
            List of chunk dicts. Each dict has:
                chunk_id: "{doc_id}_chunk_{i}" — unique across the entire corpus
                doc_id: Parent document ID
                text: The chunk text content
                chunk_index: Position of this chunk (0-indexed)
                total_chunks: How many chunks the document was split into
                start_index: Character offset in original text where this chunk starts
                metadata: Copy of document metadata plus chunk-level fields
        """
        if not text or not text.strip():
            logger.warning(f"Empty text received for doc_id={doc_id}")
            return []

        lc_chunks = self.splitter.create_documents(
            texts=[text],
            metadatas=[{**metadata, "doc_id": doc_id}]
        )

        result = []
        for i, chunk in enumerate(lc_chunks):
            chunk_meta = {
                **chunk.metadata,
                "chunk_id": f"{doc_id}_chunk_{i}",
                "doc_id": doc_id,
                "chunk_index": i,
                "total_chunks": len(lc_chunks),
                "start_index": chunk.metadata.get("start_index", 0),
                "text": chunk.page_content  # store text in metadata for retrieval
            }
            result.append({
                "chunk_id": f"{doc_id}_chunk_{i}",
                "doc_id": doc_id,
                "text": chunk.page_content,
                "chunk_index": i,
                "total_chunks": len(lc_chunks),
                "metadata": chunk_meta
            })

        return result

    def chunk_batch(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a list of documents. Each document dict must have:
            doc_id: str
            text: str
            metadata: dict

        Returns flat list of all chunks from all documents.
        """
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(
                text=doc["text"],
                metadata=doc.get("metadata", {}),
                doc_id=doc["doc_id"]
            )
            all_chunks.extend(chunks)
        logger.info(f"Chunked {len(documents)} docs → {len(all_chunks)} chunks")
        return all_chunks