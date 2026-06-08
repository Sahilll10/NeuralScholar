import logging
import os
from typing import List, Dict, Any
from datasets import Dataset

logger = logging.getLogger(__name__)


class RAGASEvaluator:
    """
    RAGAS (Retrieval Augmented Generation Assessment) evaluation.
    
    Reference: "RAGAS: Automated Evaluation of Retrieval Augmented Generation" 
               Shahul Es et al., 2023 (arxiv:2309.15217)
    
    RAGAS metrics and what they measure:
    
    Faithfulness (0-1):
        Does the generated answer contain ONLY claims that can be inferred from
        the retrieved context? A faithfulness of 1.0 means no hallucination.
        Method: NLI — each sentence of the answer is checked against the context.
    
    Answer Relevancy (0-1):
        How relevant is the generated answer to the original question?
        Method: Reverse generation — generate N questions from the answer,
        measure cosine similarity between original question and generated questions.
    
    Context Precision (0-1):
        Are the retrieved context chunks actually useful for answering the question?
        Method: For each chunk, ask LLM if it is useful for the given question+answer.
    
    Context Recall (0-1):
        Are all facts from the ground truth answer present in the retrieved context?
        Method: Decompose ground truth answer into sentences, check each against context.
    
    RAGAS Score = arithmetic mean of all four metrics.
    
    Requirements: OPENAI_API_KEY must be set (RAGAS uses GPT for NLI evaluation).
    """

    def __init__(self, openai_api_key: str):
        os.environ["OPENAI_API_KEY"] = openai_api_key

        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            )
            self.evaluate = evaluate
            self.metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
            logger.info("RAGAS evaluator initialized")
        except ImportError as e:
            logger.error(f"RAGAS import failed: {e}. Install: pip install ragas")
            raise

    def evaluate_batch(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str]
    ) -> Dict[str, float]:
        """
        Run RAGAS evaluation on a batch of QA triplets.

        Args:
            questions: List of user queries
            answers: List of model-generated answers (one per query)
            contexts: List of lists of retrieved chunk texts (one list per query)
                      Example: [["chunk text 1", "chunk text 2"], ["chunk text A"]]
            ground_truths: List of reference answers (human or LLM-generated)
                           Used for context_recall computation.

        Returns:
            Dict of metric name → averaged float score across the batch.
        """
        if not (len(questions) == len(answers) == len(contexts) == len(ground_truths)):
            raise ValueError("All input lists must have the same length")

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        })

        logger.info(f"Running RAGAS on {len(questions)} samples (this uses OpenAI API)...")
        result = self.evaluate(dataset, metrics=self.metrics)

        scores = {
            "faithfulness": float(result["faithfulness"]),
            "answer_relevancy": float(result["answer_relevancy"]),
            "context_precision": float(result["context_precision"]),
            "context_recall": float(result["context_recall"]),
        }
        scores["ragas_score"] = float(sum(scores.values()) / len(scores))

        logger.info(f"RAGAS scores: {scores}")
        return scores