import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks
from api.models.request import IngestRequest
from api.models.response import IngestResponse

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest, background_tasks: BackgroundTasks):
    import api.main as app_state
    pipeline = app_state.ingestion_pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Ingestion pipeline not initialized")

    try:
        if request.source == "arxiv":
            if not request.query and not request.arxiv_ids:
                raise HTTPException(status_code=422, detail="Provide 'query' or 'arxiv_ids' for ArXiv ingestion")

            if request.arxiv_ids:
                papers = pipeline.arxiv_loader.fetch_by_ids(request.arxiv_ids)
                docs = []
                for p in papers:
                    from data.processing.metadata_extractor import MetadataExtractor
                    meta = MetadataExtractor.from_arxiv_paper(p)
                    text = f"Title: {p.title}\n\nAbstract: {p.abstract}"
                    docs.append({"doc_id": meta["doc_id"], "text": text, "metadata": meta})
                result = pipeline._process_and_index(docs)
            else:
                result = pipeline.ingest_arxiv(
                    query=request.query,
                    max_results=request.max_results,
                    categories=request.categories,
                    use_full_text=request.use_full_text
                )

        elif request.source == "pdf_url":
            if not request.pdf_url:
                raise HTTPException(status_code=422, detail="Provide 'pdf_url' for PDF ingestion")
            pdf_data = pipeline.pdf_loader.load_from_url(request.pdf_url)
            if not pdf_data:
                raise HTTPException(status_code=422, detail="Could not load PDF from URL")
            from data.processing.metadata_extractor import MetadataExtractor
            meta = MetadataExtractor.from_pdf(pdf_data)
            docs = [{"doc_id": meta["doc_id"], "text": pdf_data["text"], "metadata": meta}]
            result = pipeline._process_and_index(docs)

        else:
            raise HTTPException(status_code=422, detail=f"Unknown source: {request.source}")

        if app_state.cache:
            app_state.cache.invalidate()

        return IngestResponse(
            success=True,
            documents_processed=result["documents_processed"],
            chunks_created=result["chunks_created"],
            vectors_upserted=result["vectors_upserted"],
            message=f"Successfully ingested {result['documents_processed']} documents"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingest error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))