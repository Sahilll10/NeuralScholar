import logging
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config.settings import settings

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("neuralscholar.api")

# Global component references (initialized in lifespan)
retrieval_pipeline = None
generator = None
cache = None
benchmarker = None
ingestion_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    All ML models and connections are initialized once at startup
    and shared across requests. This avoids the massive overhead
    of loading sentence-transformers and cross-encoder on every request.
    """
    global retrieval_pipeline, generator, cache, benchmarker, ingestion_pipeline

    logger.info("=== NeuralScholar API starting up ===")

    # 1. Initialize embedding manager (loads sentence-transformers model)
    from embeddings.embedding_manager import EmbeddingManager
    embedder = EmbeddingManager(primary="local", fallback=True)
    logger.info(f"Embedder ready. Dimension: {embedder.dimension}")

    # 2. Initialize vector stores
    from vector_store.pinecone_store import PineconeVectorStore
    from vector_store.faiss_store import FAISSVectorStore
    from vector_store.bm25_store import BM25Store

    pinecone_store = PineconeVectorStore(
        api_key=settings.PINECONE_API_KEY,
        index_name=settings.PINECONE_INDEX_NAME,
        dimension=settings.PINECONE_DIMENSION,
        metric="cosine",
        namespace="default"
    )

    faiss_store = FAISSVectorStore(
        dimension=settings.LOCAL_EMBEDDING_DIM,
        index_path=settings.FAISS_INDEX_PATH
    )

    bm25_store = BM25Store(index_path=settings.BM25_INDEX_PATH)

    # 3. Initialize retrieval pipeline (loads cross-encoder model)
    from retrieval.retrieval_pipeline import RetrievalPipeline
    retrieval_pipeline = RetrievalPipeline(
        embedder=embedder,
        dense_store=pinecone_store,
        sparse_store=bm25_store,
        use_hyde=settings.HYDE_ENABLED,
        use_hybrid=True,
        use_reranker=True
    )

    # 4. Initialize generator
    from generation.generator import RAGGenerator
    generator = RAGGenerator(
        api_key=settings.OPENAI_API_KEY,
        model=settings.OPENAI_CHAT_MODEL,
        temperature=settings.TEMPERATURE,
        max_tokens=settings.MAX_TOKENS,
        max_context_tokens=settings.MAX_CONTEXT_TOKENS
    )

    # 5. Initialize Redis cache
    from cache.redis_client import RedisCache
    cache = RedisCache(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB,
        password=settings.REDIS_PASSWORD or None,
        ttl=settings.CACHE_TTL
    )

    # 6. Initialize benchmarker
    from evaluation.benchmarker import LatencyBenchmarker
    benchmarker = LatencyBenchmarker()

    # 7. Initialize ingestion pipeline
    from data.pipeline import IngestionPipeline
    ingestion_pipeline = IngestionPipeline(
        embedder=embedder,
        pinecone_store=pinecone_store,
        faiss_store=faiss_store,
        bm25_store=bm25_store
    )

    logger.info("=== All components initialized. API ready. ===")
    yield

    logger.info("=== NeuralScholar API shutting down ===")


app = FastAPI(
    title="NeuralScholar API",
    description="Semantic RAG Pipeline for ML Research Literature Intelligence",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Import and register routers
from api.routes.query import router as query_router
from api.routes.ingest import router as ingest_router
from api.routes.eval import router as eval_router

app.include_router(query_router, prefix="/api/v1", tags=["Query"])
app.include_router(ingest_router, prefix="/api/v1", tags=["Ingestion"])
app.include_router(eval_router, prefix="/api/v1", tags=["Evaluation"])


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "cache_available": cache.available if cache else False
    }


@app.get("/benchmark", tags=["Health"])
async def get_benchmark():
    return benchmarker.report() if benchmarker else {}


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_level=settings.LOG_LEVEL.lower()
    )