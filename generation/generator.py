import logging
import json
import tiktoken
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any, AsyncGenerator, Optional
from generation.prompts import RAG_PROMPT
from config.settings import settings

logger = logging.getLogger(__name__)


class RAGGenerator:
    """
    GPT-4o based RAG answer generator with streaming support.
    
    Key responsibilities:
    1. Format retrieved chunks into a structured context string with source labels
    2. Apply token budget management using tiktoken to avoid exceeding context window
    3. Generate answers with inline citations matching the source labels
    4. Support both synchronous (for testing) and async streaming (for FastAPI SSE)
    
    Token budget management:
        GPT-4o context window: 128K tokens
        We reserve 8K tokens for context (from settings.MAX_CONTEXT_TOKENS)
        Remaining: system prompt (~300 tokens) + query (~50 tokens) + generation (~1K)
        This ensures we never exceed the model's limit even with many retrieved chunks.
    
    Streaming:
        Uses LangChain's astream() which yields BaseMessageChunk objects.
        Each chunk.content contains 1-5 tokens from the GPT-4o stream.
        The streaming endpoint yields these via SSE to minimize time-to-first-token.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        max_context_tokens: int = 8000,
        streaming: bool = True
    ):
        self.model = model
        self.max_context_tokens = max_context_tokens
        self.streaming = streaming

        self.llm = ChatOpenAI(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming
        )

        # tiktoken for accurate token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def _format_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Build the context string from retrieved documents.
        
        Each document is labeled with an index [1], [2], ... so the LLM can
        reference them in inline citations. We include title, authors, year, and
        the chunk text. Documents are added until the token budget is exhausted.
        
        Token budget: settings.MAX_CONTEXT_TOKENS characters worth of context.
        """
        context_parts = []
        total_tokens = 0

        for i, doc in enumerate(retrieved_docs, 1):
            meta = doc.get("metadata", {})
            title = meta.get("title", "Unknown Paper")
            year = (meta.get("published_date", "") or "")[:4] or "Unknown Year"
            authors = meta.get("authors", "Unknown Authors")
            arxiv_id = meta.get("arxiv_id", "")
            text = meta.get("text", meta.get("abstract", "No content available"))

            source_label = f"[{i}] {title} ({year})"
            if arxiv_id:
                source_label += f" [arXiv:{arxiv_id}]"

            chunk_text = (
                f"{source_label}\n"
                f"Authors: {authors}\n"
                f"Content: {text}\n"
                f"{'─' * 40}"
            )

            chunk_tokens = self._count_tokens(chunk_text)
            if total_tokens + chunk_tokens > self.max_context_tokens:
                logger.debug(f"Context token budget reached at document {i}/{len(retrieved_docs)}")
                break

            context_parts.append(chunk_text)
            total_tokens += chunk_tokens

        return "\n\n".join(context_parts)

    def _extract_citations(self, retrieved_docs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Extract citation metadata for the API response."""
        citations = []
        for doc in retrieved_docs:
            meta = doc.get("metadata", {})
            citations.append({
                "title": meta.get("title", "Unknown"),
                "authors": meta.get("authors", "Unknown"),
                "year": (meta.get("published_date", "") or "")[:4],
                "arxiv_id": meta.get("arxiv_id", ""),
                "paper_url": meta.get("paper_url", ""),
                "pdf_url": meta.get("pdf_url", ""),
                "retrieval_score": str(round(doc.get("score", 0), 6)),
                "rerank_score": str(round(doc.get("rerank_score", 0), 4))
            })
        return citations

    def generate(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Synchronous non-streaming generation. Returns complete response.

        Returns:
            Dict with:
                answer: str — The LLM-generated answer with inline citations
                citations: List[dict] — Metadata for all source documents
                context_used: int — Number of documents included in context
                model: str — Model used for generation
        """
        context = self._format_context(retrieved_docs)
        citations = self._extract_citations(retrieved_docs)
        messages = RAG_PROMPT.format_messages(context=context, query=query)

        response = self.llm.invoke(messages)

        return {
            "answer": response.content,
            "citations": citations,
            "context_used": len([c for c in context.split("─" * 40) if c.strip()]),
            "model": self.model
        }

    async def astream_generate(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]]
    ) -> AsyncGenerator[str, None]:
        """
        Async streaming generation for FastAPI SSE endpoints.
        
        Yields:
            Token strings as they arrive from GPT-4o.
            At the end, yields a special JSON block:
            "[CITATIONS]{...json...}[/CITATIONS]"
            
        The frontend parses this final block to extract citation metadata
        for display in the sidebar without re-fetching.
        """
        context = self._format_context(retrieved_docs)
        citations = self._extract_citations(retrieved_docs)
        messages = RAG_PROMPT.format_messages(context=context, query=query)

        async for chunk in self.llm.astream(messages):
            if chunk.content:
                yield chunk.content

        yield f"\n\n[CITATIONS]{json.dumps(citations, ensure_ascii=False)}[/CITATIONS]"