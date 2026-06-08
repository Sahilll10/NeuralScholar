from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List, Literal


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    retrieval_mode: Literal["hybrid", "dense", "sparse"] = "hybrid"
    use_hyde: bool = True
    use_cache: bool = True
    filter: Optional[Dict[str, Any]] = None

    @validator("query")
    def clean_query(cls, v):
        return v.strip()


class IngestRequest(BaseModel):
    source: Literal["arxiv", "pdf_url"]
    query: Optional[str] = None
    arxiv_ids: Optional[List[str]] = None
    pdf_url: Optional[str] = None
    max_results: int = Field(default=100, ge=1, le=500)
    categories: Optional[List[str]] = None
    use_full_text: bool = False


class EvalRequest(BaseModel):
    sample_size: int = Field(default=50, ge=5, le=200)