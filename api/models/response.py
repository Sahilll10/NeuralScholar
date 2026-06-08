from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class Citation(BaseModel):
    title: str
    authors: str
    year: str
    arxiv_id: Optional[str] = ""
    paper_url: Optional[str] = ""
    pdf_url: Optional[str] = ""
    retrieval_score: Optional[str] = ""
    rerank_score: Optional[str] = ""


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    context_used: int
    model: str
    cached: bool = False
    retrieval_latency_ms: Optional[float] = None
    generation_latency_ms: Optional[float] = None
    e2e_latency_ms: Optional[float] = None


class IngestResponse(BaseModel):
    success: bool
    documents_processed: int
    chunks_created: int
    vectors_upserted: int
    message: str


class EvalResponse(BaseModel):
    ragas_scores: Dict[str, float]
    retrieval_metrics: Dict[str, float]
    latency_report: Dict[str, Any]
    sample_size: int