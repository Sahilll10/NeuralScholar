import numpy as np
import logging
from openai import OpenAI
from typing import List

logger = logging.getLogger(__name__)


class HyDEGenerator:
    """
    Hypothetical Document Embeddings (Gao et al., 2022 — arxiv:2212.10496).
    
    Problem: Short user queries (5-15 words) have sparse overlap in embedding space
    with long, technical document chunks (100+ words). This creates a domain gap
    between query embeddings and document embeddings.
    
    HyDE Solution:
    1. Feed the query to an LLM, ask it to generate a hypothetical answer.
    2. Embed that hypothetical answer — it lives in the same region as
       real document embeddings.
    3. Average multiple hypothetical embeddings to reduce hallucination noise.
    4. Use the averaged embedding as the query vector for retrieval.
    
    Why it works: The LLM-generated text uses the same vocabulary, style, and
    technical language as the actual documents, creating a richer embedding
    that is semantically closer to the relevant document regions.
    
    Improvement over baseline: ~3-8% on BEIR benchmark for dense retrieval tasks.
    
    Implementation note: We use gpt-4o-mini for HyDE generation (not gpt-4o)
    because HyDE needs speed, not quality — a plausible answer is sufficient.
    """

    SYSTEM_PROMPT = """You are an expert in machine learning and AI research.
Given a research question, write a concise technical passage (4-6 sentences) 
that would appear in an academic ML paper as a direct answer to this question.

Rules:
- Be technical and use domain-specific terminology
- Reference specific models, metrics, architectures, or techniques where appropriate
- Write as factual prose, not as a question-answer format
- Do not say "According to research" or hedge — write authoritatively
- Do not introduce information unrelated to the question"""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        num_hypothetical: int = 3,
        temperature: float = 0.7,
        max_tokens: int = 250
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.num_hypothetical = num_hypothetical
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate_hypothetical_documents(self, query: str) -> List[str]:
        """
        Generate N hypothetical document passages for a query.

        Uses temperature=0.7 to introduce variation across hypotheticals,
        which improves averaged embedding diversity and retrieval robustness.

        Returns the original query as a fallback if all API calls fail.
        """
        hypotheticals = []

        for i in range(self.num_hypothetical):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": f"Research question: {query}"}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    n=1
                )
                hypothesis = response.choices[0].message.content.strip()
                hypotheticals.append(hypothesis)
                logger.debug(f"HyDE hypothesis {i + 1}: {hypothesis[:100]}...")

            except Exception as e:
                logger.warning(f"HyDE generation {i + 1} failed: {e}")
                hypotheticals.append(query)

        return hypotheticals