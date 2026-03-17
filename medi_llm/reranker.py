"""
Cross-encoder reranker for the medi_llm RAG pipeline.

Strategy (mirroring week5-RAG reranking intent but without an LLM API call):
  1. The vector-DB stage fetches a large candidate pool (top_k × multiplier).
  2. A cross-encoder model scores every (query, chunk) pair together, which is
     significantly more accurate than cosine distance alone because the model
     sees the query and passage in the same attention context.
  3. Candidates are sorted descending by score; only the best top_k are returned.

The cross-encoder model is loaded once and cached so repeated calls within a
session do not reload weights.
"""

from functools import lru_cache
from typing import List

try:
    from .config import DEFAULT_SETTINGS
    from .schemas import RetrievalResult
except ImportError:
    from config import DEFAULT_SETTINGS
    from schemas import RetrievalResult


class CrossEncoderReranker:
    """Score query-document pairs with a cross-encoder and return ranked results."""

    def __init__(self, model_name: str) -> None:
        """Load the cross-encoder model identified by model_name."""
        from sentence_transformers import CrossEncoder  # local import keeps top-level fast

        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def score_pairs(self, query: str, texts: List[str]) -> List[float]:
        """Return a relevance score for each (query, text) pair.

        Higher scores indicate greater relevance.
        """
        pairs = [(query, text) for text in texts]
        scores: List[float] = self.model.predict(pairs).tolist()
        return scores

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        """Score candidates against query, sort by score, return the top_k results."""
        if not candidates:
            return []

        texts = [result.text for result in candidates]
        scores = self.score_pairs(query=query, texts=texts)

        scored_candidates = sorted(
            zip(scores, candidates),
            key=lambda pair: pair[0],
            reverse=True,
        )

        reranked: List[RetrievalResult] = []
        for score, result in scored_candidates[:top_k]:
            reranked.append(
                RetrievalResult(
                    chunk_id=result.chunk_id,
                    document_id=result.document_id,
                    text=result.text,
                    metadata=result.metadata,
                    distance=result.distance,
                    rerank_score=round(float(score), 6),
                )
            )

        return reranked


@lru_cache(maxsize=2)
def get_reranker(
    model_name: str = DEFAULT_SETTINGS.reranker_model_name,
) -> CrossEncoderReranker:
    """Return a cached cross-encoder reranker for the given model name."""
    return CrossEncoderReranker(model_name=model_name)
