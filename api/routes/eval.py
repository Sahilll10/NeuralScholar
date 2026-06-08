import logging
from fastapi import APIRouter, HTTPException
from api.models.request import EvalRequest
from api.models.response import EvalResponse
import api.main as app_state

logger = logging.getLogger(__name__)
router = APIRouter()

# Default evaluation questions for the ML/NLP domain
DEFAULT_EVAL_QUESTIONS = [
    "What is the attention mechanism in transformers and how does it work?",
    "How does retrieval-augmented generation improve factual accuracy?",
    "What are the key differences between GPT and BERT architectures?",
    "How does contrastive learning work for sentence embeddings?",
    "What is the RLHF training procedure for aligning language models?",
    "How does BM25 differ from TF-IDF for document retrieval?",
    "What is LoRA and how does it enable parameter-efficient fine-tuning?",
    "How does knowledge distillation work for model compression?",
    "What are the main challenges in multi-hop question answering?",
    "How does chain-of-thought prompting improve reasoning capabilities?"
]


@router.post("/evaluate", response_model=EvalResponse)
async def evaluate(request: EvalRequest):
    """
    Run full evaluation: RAGAS + retrieval metrics + latency benchmarks.
    """
    pipeline = app_state.retrieval_pipeline
    gen = app_state.generator
    bench = app_state.benchmarker

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    questions = (request.questions or DEFAULT_EVAL_QUESTIONS)[:request.sample_size]

    try:
        all_retrieved = []
        all_answers = []
        all_contexts = []

        for question in questions:
            docs = pipeline.retrieve(query=question, top_k_retrieval=20, top_k_final=5)
            result = gen.generate(query=question, retrieved_docs=docs)
            all_retrieved.append([d["id"] for d in docs])
            all_answers.append(result["answer"])
            all_contexts.append([d["metadata"].get("text", "")[:500] for d in docs])

        # RAGAS evaluation
        from evaluation.ragas_evaluator import RAGASEvaluator
        ragas_eval = RAGASEvaluator(openai_api_key=app_state.settings.OPENAI_API_KEY if hasattr(app_state, 'settings') else "")
        
        ground_truths = [f"Answer about: {q}" for q in questions]
        
        from config.settings import settings
        ragas_eval = RAGASEvaluator(openai_api_key=settings.OPENAI_API_KEY)
        ragas_scores = ragas_eval.evaluate_batch(
            questions=questions,
            answers=all_answers,
            contexts=all_contexts,
            ground_truths=ground_truths
        )

        # Retrieval metrics (using RAGAS context precision as proxy for relevance)
        from evaluation.metrics import RetrievalMetrics
        retrieval_metrics = {"precision@5": ragas_scores.get("context_precision", 0)}

        return EvalResponse(
            ragas_scores=ragas_scores,
            retrieval_metrics=retrieval_metrics,
            latency_report=bench.report(),
            sample_size=len(questions)
        )

    except Exception as e:
        logger.error(f"Eval error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/benchmark")
async def get_latency_benchmark():
    """Return accumulated latency benchmarks from the running API."""
    bench = app_state.benchmarker
    if bench is None:
        return {"message": "No benchmarks recorded yet"}
    return bench.report()