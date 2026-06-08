import arxiv
import time
import logging
from typing import List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ArXivPaper:
    arxiv_id: str
    title: str
    abstract: str
    authors: List[str]
    categories: List[str]
    published_date: str
    updated_date: str
    pdf_url: str
    paper_url: str
    full_text: Optional[str] = None


class ArXivLoader:
    """
    Fetches papers from the ArXiv public API.
    
    Uses the 'arxiv' Python library which wraps the ArXiv API v2.
    Rate limit: be respectful — use delay_between_requests >= 0.5s.
    The ArXiv API has a 3-requests-per-second soft limit.
    """

    DEFAULT_QUERIES = [
        "retrieval augmented generation",
        "large language models fine-tuning",
        "transformer architecture attention",
        "contrastive learning sentence embeddings",
        "information retrieval dense retrieval",
        "question answering extractive abstractive",
        "vector database approximate nearest neighbor",
        "chain of thought prompting reasoning",
        "instruction tuning RLHF alignment",
        "knowledge graph language model"
    ]

    def __init__(self, delay_between_requests: float = 0.5):
        self.delay = delay_between_requests
        self.client = arxiv.Client(
            page_size=100,
            delay_seconds=delay_between_requests,
            num_retries=3
        )

    def fetch_papers(
        self,
        query: str,
        max_results: int = 500,
        sort_by: arxiv.SortCriterion = arxiv.SortCriterion.SubmittedDate,
        categories: Optional[List[str]] = None
    ) -> List[ArXivPaper]:
        """
        Fetch papers from ArXiv matching the query.

        Args:
            query: Natural language search string
            max_results: Maximum number of papers to return
            sort_by: arxiv.SortCriterion.SubmittedDate | Relevance | LastUpdatedDate
            categories: Optional ArXiv category filters e.g. ["cs.CL", "cs.AI", "cs.LG"]

        Returns:
            List of ArXivPaper dataclass instances
        """
        if categories:
            cat_filter = " OR ".join([f"cat:{c}" for c in categories])
            full_query = f"({query}) AND ({cat_filter})"
        else:
            full_query = query

        search = arxiv.Search(
            query=full_query,
            max_results=max_results,
            sort_by=sort_by
        )

        papers = []
        try:
            for result in self.client.results(search):
                paper = ArXivPaper(
                    arxiv_id=result.entry_id.split("/abs/")[-1],
                    title=result.title.replace("\n", " ").strip(),
                    abstract=result.summary.replace("\n", " ").strip(),
                    authors=[str(a) for a in result.authors],
                    categories=result.categories,
                    published_date=result.published.strftime("%Y-%m-%d"),
                    updated_date=result.updated.strftime("%Y-%m-%d"),
                    pdf_url=result.pdf_url,
                    paper_url=result.entry_id
                )
                papers.append(paper)

                if len(papers) % 50 == 0:
                    logger.info(f"Fetched {len(papers)} papers...")

        except Exception as e:
            logger.error(f"ArXiv fetch error: {e}")
            raise

        logger.info(f"Fetched {len(papers)} papers for query: '{query}'")
        return papers

    def fetch_by_ids(self, arxiv_ids: List[str]) -> List[ArXivPaper]:
        """Fetch specific papers by ArXiv ID strings (e.g., '2005.11401')."""
        search = arxiv.Search(id_list=arxiv_ids)
        papers = []
        for result in self.client.results(search):
            papers.append(ArXivPaper(
                arxiv_id=result.entry_id.split("/abs/")[-1],
                title=result.title.replace("\n", " ").strip(),
                abstract=result.summary.replace("\n", " ").strip(),
                authors=[str(a) for a in result.authors],
                categories=result.categories,
                published_date=result.published.strftime("%Y-%m-%d"),
                updated_date=result.updated.strftime("%Y-%m-%d"),
                pdf_url=result.pdf_url,
                paper_url=result.entry_id
            ))
        return papers

    def fetch_default_corpus(self, papers_per_query: int = 100) -> List[ArXivPaper]:
        """
        Fetch a comprehensive corpus using the default ML/NLP queries.
        Deduplicates by arxiv_id.
        """
        all_papers = {}
        for query in self.DEFAULT_QUERIES:
            logger.info(f"Fetching: '{query}'")
            papers = self.fetch_papers(
                query=query,
                max_results=papers_per_query,
                categories=["cs.CL", "cs.AI", "cs.LG", "cs.IR"]
            )
            for p in papers:
                all_papers[p.arxiv_id] = p  # deduplicate by id

        logger.info(f"Default corpus: {len(all_papers)} unique papers")
        return list(all_papers.values())