import hashlib
from typing import Dict, Any, Optional
from datetime import datetime


class MetadataExtractor:
    """
    Normalizes metadata from different sources (ArXiv papers, PDFs, custom docs)
    into a consistent schema used throughout the vector stores and retrieval layer.
    
    Pinecone metadata constraints:
    - Values must be str, int, float, bool, or List[str]
    - Each metadata entry has a 40KB limit
    - Keys must be strings without leading underscores
    """

    @staticmethod
    def from_arxiv_paper(paper) -> Dict[str, Any]:
        """
        Create unified metadata dict from ArXivPaper dataclass.
        The doc_id follows the pattern 'arxiv_{arxiv_id}'.
        """
        return {
            "doc_id": f"arxiv_{paper.arxiv_id}",
            "source_type": "arxiv",
            "arxiv_id": paper.arxiv_id,
            "title": paper.title[:500],
            "authors": ", ".join(paper.authors[:5]),
            "abstract": paper.abstract[:1000],
            "categories": ", ".join(paper.categories),
            "published_date": paper.published_date,
            "updated_date": paper.updated_date,
            "paper_url": paper.paper_url,
            "pdf_url": paper.pdf_url,
            "ingested_at": datetime.utcnow().isoformat()
        }

    @staticmethod
    def from_pdf(pdf_data: Dict[str, Any], custom_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Create metadata dict from parsed PDF data dict.
        doc_id follows the pattern 'pdf_{hash}'.
        """
        file_name = pdf_data.get("file_name", "unknown.pdf")
        doc_hash = custom_id or hashlib.md5(file_name.encode()).hexdigest()[:12]

        return {
            "doc_id": f"pdf_{doc_hash}",
            "source_type": "pdf",
            "title": pdf_data.get("title", "Unknown Document")[:500],
            "author": pdf_data.get("author", "Unknown")[:200],
            "source": pdf_data.get("source", ""),
            "file_name": file_name,
            "num_pages": pdf_data.get("num_pages", 0),
            "creation_date": pdf_data.get("creation_date", ""),
            "ingested_at": datetime.utcnow().isoformat()
        }

    @staticmethod
    def sanitize_for_pinecone(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure all metadata values conform to Pinecone's type requirements.
        Converts any non-conforming values to strings.
        None values become empty strings.
        """
        sanitized = {}
        for key, value in metadata.items():
            if isinstance(value, bool):
                sanitized[key] = value
            elif isinstance(value, (int, float)):
                sanitized[key] = value
            elif isinstance(value, str):
                sanitized[key] = value[:1000]  # Pinecone string limit safety
            elif isinstance(value, list):
                sanitized[key] = [str(v)[:200] for v in value[:20]]
            elif value is None:
                sanitized[key] = ""
            else:
                sanitized[key] = str(value)[:500]
        return sanitized